import numpy as np
import cv2
from scipy.ndimage import rotate, zoom
from typing import Tuple, Optional
import random

class BrainTumorAugmentor:
    """
    Data augmentation for brain tumor MRI images.
    Supports both 2D (Figshare) and 3D (BraTS) augmentation.
    """
    
    def __init__(self, rotation_range: int = 15, horizontal_flip: bool = True,
                 zoom_range: Tuple[float, float] = (0.9, 1.1),
                 brightness_range: Tuple[float, float] = (0.9, 1.1),
                 seed: int = 42):
        self.rotation_range = rotation_range
        self.horizontal_flip = horizontal_flip
        self.zoom_range = zoom_range
        self.brightness_range = brightness_range
        random.seed(seed)
        np.random.seed(seed)
    
    def augment_3d(self, volume: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply random augmentations to a 3D volume and its mask simultaneously.
        """
        aug_vol = volume.copy()
        aug_mask = mask.copy()
        
        # 1. Random rotation
        if random.random() > 0.5:
            angle = random.uniform(-self.rotation_range, self.rotation_range)
            # Rotate along z-axis (axis 0 and 1)
            aug_vol = rotate(aug_vol, angle, axes=(0, 1), reshape=False, order=1, mode='nearest')
            aug_mask = rotate(aug_mask, angle, axes=(0, 1), reshape=False, order=0, mode='nearest')
            
        # 2. Random horizontal flip
        if self.horizontal_flip and random.random() > 0.5:
            aug_vol = np.flip(aug_vol, axis=1)
            aug_mask = np.flip(aug_mask, axis=1)
            
        # 3. Random zoom
        if random.random() > 0.5:
            zoom_factor = random.uniform(self.zoom_range[0], self.zoom_range[1])
            # Zoom only H and W
            # volume shape: (H, W, D, C)
            # mask shape: (H, W, D)
            h, w, d = aug_mask.shape
            
            # Use ndimage.zoom
            if aug_vol.ndim == 4:
                z_vol = zoom(aug_vol, (zoom_factor, zoom_factor, 1, 1), order=1, mode='nearest')
            else:
                z_vol = zoom(aug_vol, (zoom_factor, zoom_factor, 1), order=1, mode='nearest')
            
            z_mask = zoom(aug_mask, (zoom_factor, zoom_factor, 1), order=0, mode='nearest')
            
            # Crop or pad to original size
            zh, zw = z_mask.shape[:2]
            
            if zoom_factor > 1.0:
                # Crop
                dh = (zh - h) // 2
                dw = (zw - w) // 2
                aug_vol = z_vol[dh:dh+h, dw:dw+w]
                aug_mask = z_mask[dh:dh+h, dw:dw+w]
            else:
                # Pad
                dh = (h - zh) // 2
                dw = (w - zw) // 2
                pad_h1 = dh
                pad_h2 = h - zh - dh
                pad_w1 = dw
                pad_w2 = w - zw - dw
                
                if aug_vol.ndim == 4:
                    aug_vol = np.pad(z_vol, ((pad_h1, pad_h2), (pad_w1, pad_w2), (0, 0), (0, 0)), mode='edge')
                else:
                    aug_vol = np.pad(z_vol, ((pad_h1, pad_h2), (pad_w1, pad_w2), (0, 0)), mode='edge')
                aug_mask = np.pad(z_mask, ((pad_h1, pad_h2), (pad_w1, pad_w2), (0, 0)), mode='edge')
        
        # 4. Random brightness (volume only)
        if random.random() > 0.5:
            factor = random.uniform(self.brightness_range[0], self.brightness_range[1])
            aug_vol = aug_vol * factor
            aug_vol = np.clip(aug_vol, 0, 1)
            
        return aug_vol, aug_mask
    
    def augment_2d(self, image: np.ndarray, label: int) -> Tuple[np.ndarray, int]:
        """
        Apply random augmentations to a 2D image.
        """
        aug_img = image.copy()
        
        has_channel = len(aug_img.shape) == 3
        
        # 1. Random rotation
        if random.random() > 0.5:
            angle = random.uniform(-self.rotation_range, self.rotation_range)
            aug_img = rotate(aug_img, angle, reshape=False, order=1, mode='nearest')
            
        # 2. Random horizontal flip
        if self.horizontal_flip and random.random() > 0.5:
            aug_img = np.flip(aug_img, axis=1)
            
        # 3. Random zoom
        if random.random() > 0.5:
            zoom_factor = random.uniform(self.zoom_range[0], self.zoom_range[1])
            h, w = aug_img.shape[:2]
            
            if has_channel:
                z_img = zoom(aug_img, (zoom_factor, zoom_factor, 1), order=1, mode='nearest')
            else:
                z_img = zoom(aug_img, (zoom_factor, zoom_factor), order=1, mode='nearest')
                
            zh, zw = z_img.shape[:2]
            
            if zoom_factor > 1.0:
                dh = (zh - h) // 2
                dw = (zw - w) // 2
                aug_img = z_img[dh:dh+h, dw:dw+w]
            else:
                dh = (h - zh) // 2
                dw = (w - zw) // 2
                pad_h1, pad_h2 = dh, h - zh - dh
                pad_w1, pad_w2 = dw, w - zw - dw
                
                if has_channel:
                    aug_img = np.pad(z_img, ((pad_h1, pad_h2), (pad_w1, pad_w2), (0, 0)), mode='edge')
                else:
                    aug_img = np.pad(z_img, ((pad_h1, pad_h2), (pad_w1, pad_w2)), mode='edge')
                    
        # 4. Random brightness
        if random.random() > 0.5:
            factor = random.uniform(self.brightness_range[0], self.brightness_range[1])
            aug_img = aug_img * factor
            aug_img = np.clip(aug_img, 0, 1)
            
        return aug_img, label
