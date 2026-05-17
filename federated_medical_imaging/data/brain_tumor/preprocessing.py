import numpy as np
import cv2
import SimpleITK as sitk
import nibabel as nib
from scipy.ndimage import gaussian_filter
from typing import Tuple, Optional
import os
import glob
from utils.config_loader import ConfigLoader
from utils.logger import get_logger
from utils.common import ensure_dir

logger = get_logger("brain_tumor_preprocessing")

class BrainTumorPreprocessor:
    """
    Preprocessing pipeline for brain tumor MRI images.
    Implements: Gaussian filtering, AGSW, Guided filtering, CLAHE, normalization.
    """
    
    def __init__(self, config_path: str = "config/brain_tumor_config.yaml"):
        self.config = ConfigLoader.load(config_path)
        self.gaussian_kernel = self.config.preprocessing.gaussian_kernel_size
        self.gaussian_sigma = self.config.preprocessing.gaussian_sigma
        self.agsw_window = self.config.preprocessing.agsw_window_size
        self.guided_radius = self.config.preprocessing.guided_filter_radius
        self.guided_eps = self.config.preprocessing.guided_filter_eps
        self.clahe_clip = self.config.preprocessing.clahe_clip_limit
        self.clahe_grid = tuple(self.config.preprocessing.clahe_grid_size)
    
    def gaussian_filter_3d(self, volume: np.ndarray) -> np.ndarray:
        """
        Apply 3D Gaussian smoothing to reduce noise.
        """
        return gaussian_filter(volume, sigma=self.gaussian_sigma)
    
    def anisotropic_gaussian_side_window(self, volume: np.ndarray) -> np.ndarray:
        """
        Apply Anisotropic Gaussian Side Window (AGSW) filtering.
        """
        out_volume = np.zeros_like(volume)
        r = self.agsw_window // 2
        ksize = self.agsw_window
        sigma = self.gaussian_sigma

        # Create basic Gaussian kernel
        kernel_1d = cv2.getGaussianKernel(ksize, sigma)
        kernel_2d = kernel_1d @ kernel_1d.T

        # Generate 8 directional kernels by masking
        kernels = []
        
        # L, R, U, D
        mask_L = np.zeros((ksize, ksize)); mask_L[:, :r+1] = 1; kernels.append(kernel_2d * mask_L)
        mask_R = np.zeros((ksize, ksize)); mask_R[:, r:] = 1; kernels.append(kernel_2d * mask_R)
        mask_U = np.zeros((ksize, ksize)); mask_U[:r+1, :] = 1; kernels.append(kernel_2d * mask_U)
        mask_D = np.zeros((ksize, ksize)); mask_D[r:, :] = 1; kernels.append(kernel_2d * mask_D)
        
        # Diagonals
        mask_NW = np.zeros((ksize, ksize))
        mask_NE = np.zeros((ksize, ksize))
        mask_SW = np.zeros((ksize, ksize))
        mask_SE = np.zeros((ksize, ksize))
        for i in range(ksize):
            for j in range(ksize):
                if i <= j and i + j <= ksize - 1: mask_U[i, j] = 1
                if i >= j and i + j >= ksize - 1: mask_D[i, j] = 1
                if i >= j and i + j <= ksize - 1: mask_L[i, j] = 1
                if i <= j and i + j >= ksize - 1: mask_R[i, j] = 1

                if i <= r and j <= r: mask_NW[i, j] = 1
                if i <= r and j >= r: mask_NE[i, j] = 1
                if i >= r and j <= r: mask_SW[i, j] = 1
                if i >= r and j >= r: mask_SE[i, j] = 1

        kernels.extend([kernel_2d * mask_NW, kernel_2d * mask_NE, kernel_2d * mask_SW, kernel_2d * mask_SE])
        
        # Normalize kernels so sum is 1
        kernels = [k / np.sum(k) for k in kernels]

        for z in range(volume.shape[2]):
            slice_2d = volume[:, :, z].astype(np.float32)
            filtered_slices = [cv2.filter2D(slice_2d, -1, k) for k in kernels]
            
            # Find the one that minimizes difference with original
            errors = [np.square(f - slice_2d) for f in filtered_slices]
            min_idx = np.argmin(errors, axis=0)
            
            # Select the output
            out_slice = np.zeros_like(slice_2d)
            for i, f in enumerate(filtered_slices):
                out_slice[min_idx == i] = f[min_idx == i]
                
            out_volume[:, :, z] = out_slice
            
        return out_volume
    
    def guided_filter(self, image: np.ndarray, guide: np.ndarray) -> np.ndarray:
        """
        Apply guided image filtering.
        """
        out_image = np.zeros_like(image)
        r = self.guided_radius
        eps = self.guided_eps
        
        def filter_2d(I, p):
            mean_I = cv2.boxFilter(I, cv2.CV_64F, (r, r))
            mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
            mean_Ip = cv2.boxFilter(I * p, cv2.CV_64F, (r, r))
            mean_II = cv2.boxFilter(I * I, cv2.CV_64F, (r, r))
            
            var_I = mean_II - mean_I * mean_I
            cov_Ip = mean_Ip - mean_I * mean_p
            
            a = cov_Ip / (var_I + eps)
            b = mean_p - a * mean_I
            
            mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
            mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))
            
            return mean_a * I + mean_b

        if len(image.shape) == 3: # 3D volume
            for z in range(image.shape[2]):
                out_image[:, :, z] = filter_2d(guide[:, :, z].astype(np.float32), 
                                               image[:, :, z].astype(np.float32))
        else:
            out_image = filter_2d(guide.astype(np.float32), image.astype(np.float32))
            
        return out_image
    
    def apply_clahe(self, image_2d: np.ndarray) -> np.ndarray:
        """
        Apply Contrast Limited Adaptive Histogram Equalization.
        """
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=self.clahe_grid)
        # CLAHE expects uint8 or uint16.
        if image_2d.dtype != np.uint8 and image_2d.dtype != np.uint16:
            min_v, max_v = np.min(image_2d), np.max(image_2d)
            if max_v > min_v:
                image_2d = ((image_2d - min_v) / (max_v - min_v) * 255).astype(np.uint8)
            else:
                image_2d = image_2d.astype(np.uint8)
                
        return clahe.apply(image_2d)
    
    def normalize_volume(self, volume: np.ndarray) -> np.ndarray:
        """
        Normalize volume intensity to [0, 1] range.
        Uses min-max normalization with clipping at 1st and 99th percentiles.
        """
        p1, p99 = np.percentile(volume, [1, 99])
        if p99 > p1:
            clipped = np.clip(volume, p1, p99)
            normalized = (clipped - p1) / (p99 - p1)
        else:
            normalized = np.zeros_like(volume)
        return normalized.astype(np.float32)
    
    def _crop_volume(self, volume: np.ndarray, target_shape=(128, 128, 128)) -> np.ndarray:
        """Center crop a volume to target shape."""
        # volume shape (H, W, D) or (H, W, D, C)
        h, w, d = volume.shape[:3]
        th, tw, td = target_shape
        
        dh = max(0, (h - th) // 2)
        dw = max(0, (w - tw) // 2)
        dd = max(0, (d - td) // 2)
        
        if len(volume.shape) == 4:
            return volume[dh:dh+th, dw:dw+tw, dd:dd+td, :]
        return volume[dh:dh+th, dw:dw+tw, dd:dd+td]
    
    def preprocess_brats_volume(self, volume_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Full preprocessing pipeline for a single BraTS2020 volume.
        """
        modalities = ["t1", "t1ce", "t2", "flair"]
        vols = []
        
        # Path usually contains BraTS20_Training_XXX
        base_name = os.path.basename(volume_path)
        
        for mod in modalities:
            file_path = os.path.join(volume_path, f"{base_name}_{mod}.nii.gz")
            if not os.path.exists(file_path):
                # Fallback search if exact naming is a bit off
                matches = glob.glob(os.path.join(volume_path, f"*_{mod}.nii.gz"))
                if matches:
                    file_path = matches[0]
                else:
                    logger.warning(f"Missing {mod} modality in {volume_path}")
                    # Return empty to handle gracefully
                    return np.zeros((128, 128, 128, 4)), np.zeros((128, 128, 128))
            
            img = nib.load(file_path).get_fdata()
            # 1. Gaussian filtering
            img_g = self.gaussian_filter_3d(img)
            # 2. AGSW
            img_a = self.anisotropic_gaussian_side_window(img_g)
            # 3. Guided filter (self-guided)
            img_gf = self.guided_filter(img_a, img_a)
            # 4. Normalize
            img_norm = self.normalize_volume(img_gf)
            vols.append(img_norm)
            
        # Segmentations
        seg_path = os.path.join(volume_path, f"{base_name}_seg.nii.gz")
        if not os.path.exists(seg_path):
            matches = glob.glob(os.path.join(volume_path, f"*_seg.nii.gz"))
            if matches:
                seg_path = matches[0]
            else:
                seg_mask = np.zeros_like(img)
        
        if os.path.exists(seg_path):
            seg_mask = nib.load(seg_path).get_fdata()
            
        stacked_vols = np.stack(vols, axis=-1)
        
        cropped_vols = self._crop_volume(stacked_vols)
        cropped_seg = self._crop_volume(seg_mask)
        
        return cropped_vols, cropped_seg
    
    def preprocess_figshare_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess a single Figshare dataset image.
        """
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Maybe it's a mat file or unavailable
            logger.warning(f"Could not load image: {image_path}")
            return np.zeros((224, 224, 1))
            
        # 1. Resize to 224x224
        img_resized = cv2.resize(img, (224, 224))
        
        # 2. CLAHE enhancement
        img_clahe = self.apply_clahe(img_resized)
        
        # 3. Gaussian filtering (2D)
        img_blur = cv2.GaussianBlur(img_clahe, (self.gaussian_kernel, self.gaussian_kernel), self.gaussian_sigma)
        
        # 4. Normalize to [0, 1]
        img_norm = self.normalize_volume(img_blur.astype(np.float32))
        
        return np.expand_dims(img_norm, axis=-1)
    
    def preprocess_all_brats(self, raw_dir: str, output_dir: str) -> None:
        """
        Preprocess ALL BraTS2020 volumes in a directory.
        """
        ensure_dir(output_dir)
        patient_dirs = [os.path.join(raw_dir, d) for d in os.listdir(raw_dir) 
                        if os.path.isdir(os.path.join(raw_dir, d))]
        
        logger.info(f"Found {len(patient_dirs)} BraTS volumes to preprocess.")
        
        for i, pdir in enumerate(patient_dirs):
            bname = os.path.basename(pdir)
            vol, seg = self.preprocess_brats_volume(pdir)
            
            out_path = os.path.join(output_dir, f"{bname}.npy")
            np.save(out_path, {"volume": vol, "mask": seg})
            if i % 10 == 0:
                logger.info(f"Processed {i+1}/{len(patient_dirs)} volumes.")
                
    def preprocess_all_figshare(self, raw_dir: str, output_dir: str) -> None:
        """
        Preprocess ALL Figshare images in a directory.
        """
        ensure_dir(output_dir)
        classes = ["glioma", "meningioma", "pituitary"]
        
        image_paths = []
        for c in classes:
            c_dir = os.path.join(raw_dir, c)
            if os.path.exists(c_dir):
                paths = glob.glob(os.path.join(c_dir, "*.png")) + glob.glob(os.path.join(c_dir, "*.jpg"))
                for p in paths:
                    image_paths.append((p, c))
                    
        logger.info(f"Found {len(image_paths)} Figshare images to preprocess.")
        
        for i, (path, cls) in enumerate(image_paths):
            bname = os.path.basename(path).split(".")[0]
            img = self.preprocess_figshare_image(path)
            
            # Map class to int: Glioma (0), Meningioma (1), Pituitary (2)
            label = classes.index(cls)
            out_path = os.path.join(output_dir, f"{cls}_{bname}.npy")
            np.save(out_path, {"image": img, "label": label})
            if i % 100 == 0:
                logger.info(f"Processed {i+1}/{len(image_paths)} images.")
