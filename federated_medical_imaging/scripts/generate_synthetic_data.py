"""
Generate synthetic data for testing the pipeline without real medical datasets.
Creates fake 3D MRI volumes and labels that match expected formats.

Usage:
    python scripts/generate_synthetic_data.py --type brain_tumor --num_samples 50
    python scripts/generate_synthetic_data.py --type alzheimer --num_samples 100
    python scripts/generate_synthetic_data.py --type both --num_samples 50
"""
import numpy as np
import os
import argparse
import sys
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.common import ensure_dir, save_json
from utils.logger import get_logger

logger = get_logger("synthetic_data")

def generate_brain_tumor_volumes(output_dir: str, num_samples: int = 50, 
                                   volume_shape: tuple = (128, 128, 128)) -> Dict:
    """
    Generate synthetic brain tumor MRI volumes and segmentation masks.
    Shapes: (128, 128, 128, 4), Mask: (128, 128, 128)
    """
    logger.info(f"Generating {num_samples} synthetic brain tumor volumes of shape {volume_shape}...")
    volumes_dir = ensure_dir(os.path.join(output_dir, "volumes"))
    masks_dir = ensure_dir(os.path.join(output_dir, "masks"))
    
    for i in range(num_samples):
        # Generate random base noise mimicking an MRI
        volume = np.random.normal(loc=0.5, scale=0.1, size=(*volume_shape, 4)).astype(np.float32)
        volume = np.clip(volume, 0.0, 1.0)
        
        # Initialize empty mask
        mask = np.zeros(volume_shape, dtype=np.uint8)
        
        # Add a "tumor" region in the center
        center_x, center_y, center_z = volume_shape[0]//2, volume_shape[1]//2, volume_shape[2]//2
        radius = volume_shape[0]//8
        
        x, y, z = np.ogrid[:volume_shape[0], :volume_shape[1], :volume_shape[2]]
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2 + (z - center_z)**2)
        
        # Create tumor classes (1=NCR, 2=ED, 3=ET)
        tumor_core = distance <= radius * 0.5
        tumor_edema = (distance > radius * 0.5) & (distance <= radius)
        enhancing = (distance > radius) & (distance <= radius * 1.5)
        
        mask[enhancing] = 3
        mask[tumor_edema] = 2
        mask[tumor_core] = 1
        
        # Alter the volume intensity based on tumor regions to simulate tumor presence
        for c in range(4):
            volume[mask > 0, c] += 0.3
            
        volume = np.clip(volume, 0.0, 1.0)
        
        np.save(os.path.join(volumes_dir, f"sample_{i}_volume.npy"), volume)
        np.save(os.path.join(masks_dir, f"sample_{i}_mask.npy"), mask)
        
    stats = {
        "dataset_type": "brain_tumor_3d_synthetic",
        "num_samples": num_samples,
        "volume_shape": f"{volume_shape}x4",
        "mask_shape": str(volume_shape)
    }
    save_json(stats, os.path.join(output_dir, "metadata.json"))
    logger.info("Brain tumor synthetic data generation complete.")
    return stats

def generate_figshare_images(output_dir: str, num_samples: int = 100) -> Dict:
    """Generate synthetic 2D brain tumor classification images."""
    logger.info(f"Generating {num_samples} synthetic Figshare 2D images...")
    classes = ["Glioma", "Meningioma", "Pituitary"]
    for c in classes:
        ensure_dir(os.path.join(output_dir, c))
    
    samples_per_class = num_samples // len(classes)
    
    for c in classes:
        for i in range(samples_per_class):
            image = np.random.normal(0.5, 0.1, (224, 224, 1)).astype(np.float32)
            # Add patterns specific to class
            if c == "Glioma":
                image[100:150, 100:150] += 0.4
            elif c == "Meningioma":
                image[50:100, 50:100] += 0.4
            else:
                image[150:200, 150:200] += 0.4
                
            image = np.clip(image, 0.0, 1.0)
            np.save(os.path.join(output_dir, c, f"image_{i}.npy"), image)
            
    logger.info("Figshare synthetic data generation complete.")
    return {"dataset": "figshare_synthetic", "samples": num_samples}

def generate_alzheimer_volumes(output_dir: str, num_samples: int = 100,
                                 dataset_type: str = "adni") -> Dict:
    """Generate synthetic Alzheimer's MRI volumes for testing."""
    logger.info(f"Generating {num_samples} synthetic Alzheimer ({dataset_type}) volumes...")
    
    ad_dir = ensure_dir(os.path.join(output_dir, "AD"))
    hc_dir = ensure_dir(os.path.join(output_dir, "HC"))
    
    if dataset_type.lower() == "adni":
        shape = (182, 218, 182, 1)
    else:
        shape = (176, 208, 176, 1)
        
    samples_per_class = num_samples // 2
    
    for i in range(samples_per_class):
        # Generate AD
        ad_vol = np.random.normal(0.4, 0.1, shape).astype(np.float32)
        # Simulate atrophy (lower intensity in center)
        center_region = (slice(shape[0]//3, 2*shape[0]//3),
                         slice(shape[1]//3, 2*shape[1]//3),
                         slice(shape[2]//3, 2*shape[2]//3))
        ad_vol[center_region] -= 0.2
        ad_vol = np.clip(ad_vol, 0.0, 1.0)
        np.save(os.path.join(ad_dir, f"subject_{i}.npy"), ad_vol)
        
        # Generate HC
        hc_vol = np.random.normal(0.5, 0.1, shape).astype(np.float32)
        hc_vol = np.clip(hc_vol, 0.0, 1.0)
        np.save(os.path.join(hc_dir, f"subject_{i}.npy"), hc_vol)
        
    stats = {
        "dataset_type": f"alzheimer_{dataset_type}_synthetic",
        "num_samples": num_samples,
        "volume_shape": str(shape),
        "classes": ["AD", "HC"]
    }
    save_json(stats, os.path.join(output_dir, "metadata.json"))
    logger.info("Alzheimer's synthetic data generation complete.")
    return stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=["brain_tumor", "alzheimer", "both"], default="both")
    parser.add_argument("--num_samples", type=int, default=10)
    args = parser.parse_args()
    
    if args.type in ["brain_tumor", "both"]:
        bt_out = ensure_dir("data/raw/brain_tumor")
        # generate small volumes to save disk space and time
        generate_brain_tumor_volumes(os.path.join(bt_out, "brats"), 
                                     num_samples=args.num_samples, 
                                     volume_shape=(32, 32, 32))
        generate_figshare_images(os.path.join(bt_out, "figshare"), 
                                 num_samples=args.num_samples)
                                 
    if args.type in ["alzheimer", "both"]:
        ad_out = ensure_dir("data/raw/alzheimer")
        generate_alzheimer_volumes(os.path.join(ad_out, "adni"), 
                                   num_samples=args.num_samples, dataset_type="adni")
        generate_alzheimer_volumes(os.path.join(ad_out, "oasis"), 
                                   num_samples=args.num_samples, dataset_type="oasis")

if __name__ == "__main__":
    main()
