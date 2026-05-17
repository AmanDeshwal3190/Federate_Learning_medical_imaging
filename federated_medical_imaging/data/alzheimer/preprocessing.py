"""
Preprocessing pipeline for Alzheimer's Disease MRI data.
Implements: co-registration, skull stripping, brain masking, resampling, normalization.
"""
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import scipy.ndimage as ndi
import os
from typing import Tuple, Optional, List
from utils.config_loader import ConfigLoader
from utils.logger import get_logger
from utils.common import ensure_dir

logger = get_logger("alzheimer_preprocessing")

class AlzheimerPreprocessor:
    """
    Full preprocessing pipeline for Alzheimer's MRI data.
    Handles both ADNI (3T) and OASIS (1.5T) datasets.
    """
    
    def __init__(self, config_path: str = "config/alzheimer_config.yaml"):
        self.config = ConfigLoader.load(config_path)
        self.adni_dims = tuple(self.config.dataset.adni.final_dimensions)   # (182, 218, 182)
        self.oasis_dims = tuple(self.config.dataset.oasis.final_dimensions) # (176, 208, 176)
        self.target_voxel = tuple(self.config.preprocessing.target_voxel_size) # (1.0, 1.0, 1.0)
    
    def load_nifti(self, filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load a NIfTI (.nii or .nii.gz) file.
        
        Args:
            filepath: Path to NIfTI file
        Returns:
            Tuple of (volume_data, affine_matrix)
            volume_data: 3D numpy array
            affine_matrix: 4x4 affine transformation matrix
        """
        img = nib.load(filepath)
        volume_data = img.get_fdata().astype(np.float32)
        affine_matrix = img.affine
        return volume_data, affine_matrix
    
    def coregister_to_template(self, moving_image: sitk.Image, 
                                 template_path: str) -> sitk.Image:
        """
        Co-register a moving image to a template using SyN (Symmetric Normalization) or Affine registration.
        """
        if not template_path or not os.path.exists(template_path):
            logger.warning(f"Template not found at {template_path}, skipping registration or using identity.")
            return moving_image
            
        fixed_image = sitk.ReadImage(template_path, sitk.sitkFloat32)
        moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)

        initial_transform = sitk.CenteredTransformInitializer(
            fixed_image, 
            moving_image, 
            sitk.Euler3DTransform(), 
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )

        registration_method = sitk.ImageRegistrationMethod()
        
        # Metric
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(0.01)

        # Optimizer
        registration_method.SetOptimizerAsGradientDescentLineSearch(
            learningRate=1.0, 
            numberOfIterations=200, 
            convergenceMinimumValue=1e-6, 
            convergenceWindowSize=10
        )
        
        # Interpolator
        registration_method.SetInterpolator(sitk.sitkLinear)
        
        # Initial Transform
        registration_method.SetInitialTransform(initial_transform, inPlace=False)
        
        # Pyramids
        registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
        registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
        registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        
        try:
            final_transform = registration_method.Execute(fixed_image, moving_image)
            
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(fixed_image)
            resampler.SetInterpolator(sitk.sitkLinear)
            resampler.SetDefaultPixelValue(0)
            resampler.SetTransform(final_transform)
            
            registered_image = resampler.Execute(moving_image)
            return registered_image
        except Exception as e:
            logger.error(f"Registration failed: {e}")
            return moving_image
    
    def skull_strip(self, volume: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove non-brain tissue from MRI volume.
        """
        # 1. Apply Gaussian smoothing
        smoothed = ndi.gaussian_filter(volume, sigma=1.0)
        
        # 2. Compute Otsu threshold
        brain_voxels = smoothed[smoothed > 0]
        if len(brain_voxels) > 0:
            hist, bin_edges = np.histogram(brain_voxels, bins=256)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
            
            weight1 = np.cumsum(hist)
            weight2 = np.cumsum(hist[::-1])[::-1]
            
            mean1 = np.cumsum(hist * bin_centers) / (weight1 + 1e-8)
            mean2 = (np.cumsum((hist * bin_centers)[::-1]) / (weight2[::-1] + 1e-8))[::-1]
            
            variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
            if len(variance12) > 0:
                idx = np.argmax(variance12)
                thresh = bin_centers[idx]
            else:
                thresh = 0.5 * np.max(smoothed)
        else:
            thresh = 0
            
        binary_mask = smoothed > thresh
        
        # 4. Apply morphological operations
        # Create a spherical structuring element roughly corresponding to radius 3 and 2
        ball_3 = ndi.generate_binary_structure(3, 2)  # Approximation of ball
        ball_3 = ndi.iterate_structure(ball_3, 3)
        ball_2 = ndi.generate_binary_structure(3, 2)
        ball_2 = ndi.iterate_structure(ball_2, 2)
        
        # a. Binary closing
        closed_mask = ndi.binary_closing(binary_mask, structure=ball_3)
        
        # b. Binary opening
        opened_mask = ndi.binary_opening(closed_mask, structure=ball_2)
        
        # 5. Keep only the largest connected component
        labeled_array, num_features = ndi.label(opened_mask)
        if num_features == 0:
            return volume, np.zeros_like(volume, dtype=bool)
            
        component_sizes = np.bincount(labeled_array.ravel())
        component_sizes[0] = 0 # Ignore background
        largest_component_label = component_sizes.argmax()
        
        brain_mask = labeled_array == largest_component_label
        
        # 6. Apply mask
        skull_stripped_volume = volume * brain_mask
        
        return skull_stripped_volume, brain_mask
    
    def resample_volume(self, volume: np.ndarray, current_spacing: Tuple[float, ...],
                         target_spacing: Tuple[float, ...],
                         target_shape: Tuple[int, ...]) -> np.ndarray:
        """
        Resample volume to target spacing and shape.
        """
        # 1. Compute zoom factors
        zoom_factors = [c / t for c, t in zip(current_spacing, target_spacing)]
        
        # 2. Apply resampling
        resampled_volume = ndi.zoom(volume, zoom=zoom_factors, order=3, mode='nearest')
        
        # 3. Center-crop or pad to target_shape
        current_shape = resampled_volume.shape
        cropped_padded_volume = np.zeros(target_shape, dtype=resampled_volume.dtype)
        
        start_in = [max(0, (c - t) // 2) for c, t in zip(current_shape, target_shape)]
        end_in = [s + min(c, t) for s, c, t in zip(start_in, current_shape, target_shape)]
        
        start_out = [max(0, (t - c) // 2) for c, t in zip(current_shape, target_shape)]
        end_out = [s + min(c, t) for s, c, t in zip(start_out, current_shape, target_shape)]
        
        cropped_padded_volume[start_out[0]:end_out[0], 
                              start_out[1]:end_out[1], 
                              start_out[2]:end_out[2]] = \
            resampled_volume[start_in[0]:end_in[0], 
                             start_in[1]:end_in[1], 
                             start_in[2]:end_in[2]]
        
        return cropped_padded_volume
    
    def normalize_zscore(self, volume: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """
        Z-score normalization of the brain volume.
        """
        if mask is None:
            mask = volume > 0
            
        brain_voxels = volume[mask]
        if len(brain_voxels) == 0:
            return volume
            
        mean_val = np.mean(brain_voxels)
        std_val = np.std(brain_voxels)
        
        if std_val == 0:
            std_val = 1.0
            
        normalized = np.zeros_like(volume)
        normalized[mask] = (volume[mask] - mean_val) / std_val
        
        return normalized
    
    def preprocess_single_scan(self, nifti_path: str, dataset_type: str,
                                 template_path: str = None) -> np.ndarray:
        """
        Full preprocessing pipeline for a single MRI scan.
        """
        # 1. Load NIfTI file
        volume, affine = self.load_nifti(nifti_path)
        
        # Extract spacing from affine
        # Norm of columns in affine matrix gives spacing
        spacing = tuple(np.linalg.norm(affine[:3, i]) for i in range(3))
        
        # 2. Co-register to MNI152 template (optional)
        if self.config.preprocessing.coregistration and template_path:
            sitk_image = sitk.GetImageFromArray(volume.T)
            sitk_image.SetSpacing(spacing)
            # Not setting origin/direction here as it's a simplification
            registered_image = self.coregister_to_template(sitk_image, template_path)
            volume = sitk.GetArrayFromImage(registered_image).T
            spacing = registered_image.GetSpacing()
            
        # 3, 4. Skull strip and Brain mask
        if self.config.preprocessing.skull_stripping:
            volume, mask = self.skull_strip(volume)
        else:
            mask = volume > 0
            
        # 5. Resample
        if self.config.preprocessing.resampling:
            target_shape = self.adni_dims if dataset_type.lower() == "adni" else self.oasis_dims
            volume = self.resample_volume(volume, spacing, self.target_voxel, target_shape)
            mask = self.resample_volume(mask.astype(float), spacing, self.target_voxel, target_shape) > 0.5
            
        # 6. Z-score normalize
        if self.config.preprocessing.normalization == "z_score":
            volume = self.normalize_zscore(volume, mask)
            
        # 7. Add channel dimension
        volume = np.expand_dims(volume, axis=-1)
        
        return volume

    def preprocess_dataset(self, raw_dir: str, output_dir: str, 
                            dataset_type: str, template_path: str = None) -> dict:
        """
        Preprocess an entire dataset (all subjects).
        """
        ensure_dir(os.path.join(output_dir, "AD"))
        ensure_dir(os.path.join(output_dir, "HC"))
        
        stats = {
            "total_processed": 0,
            "failed": 0,
            "processing_time": 0.0
        }
        
        import glob
        import time
        start_time = time.time()
        
        # Searching for .nii or .nii.gz files in AD and HC dirs
        search_pattern_ad = os.path.join(raw_dir, "AD", "**", "*.nii*")
        search_pattern_hc = os.path.join(raw_dir, "HC", "**", "*.nii*")
        
        ad_files = glob.glob(search_pattern_ad, recursive=True)
        hc_files = glob.glob(search_pattern_hc, recursive=True)
        
        all_files = [(f, "AD") for f in ad_files] + [(f, "HC") for f in hc_files]
        
        for i, (filepath, label) in enumerate(all_files):
            filename = os.path.basename(filepath)
            # Remove extensions for ID
            subject_id = filename.split('.')[0]
            out_filepath = os.path.join(output_dir, label, f"{subject_id}.npy")
            
            if os.path.exists(out_filepath):
                continue
                
            try:
                processed_volume = self.preprocess_single_scan(filepath, dataset_type, template_path)
                np.save(out_filepath, processed_volume)
                stats["total_processed"] += 1
            except Exception as e:
                logger.error(f"Failed to process {filepath}: {e}")
                stats["failed"] += 1
                
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(all_files)} scans...")
                
        stats["processing_time"] = time.time() - start_time
        return stats
