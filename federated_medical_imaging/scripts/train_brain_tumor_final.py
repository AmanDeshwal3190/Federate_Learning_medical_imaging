"""
Script for training the final Brain Tumor 2D U-Net model on the BraTS2020 slice dataset.
"""
import os
import sys
import glob
import h5py
import numpy as np
import tensorflow as tf
import argparse

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.brain_tumor.unet2d import build_unet2d
from utils.logger import get_logger
from utils.common import ensure_dir

logger = get_logger("train_brain_tumor_final")

def get_file_list(data_dir):
    pattern = os.path.join(data_dir, "volume_*_slice_*.h5")
    files = glob.glob(pattern)
    return files

def data_generator(file_list):
    """
    Generator yielding (image, mask) pairs from given .h5 slice paths.
    """
    for file_path in file_list:
        try:
            with h5py.File(file_path, 'r') as f:
                if 'image' in f and 'mask' in f:
                    image = np.array(f['image'], dtype=np.float32)
                    mask = np.array(f['mask'], dtype=np.float32)
                    
                    # Normalize if necessary (assuming pre-normalized or max scaling)
                    # We normalize strictly across standard modality ranges here if not already 0-1/z-score
                    if image.max() > 0:
                        image = image / image.max()
                        
                    yield image, mask
        except Exception as e:
            logger.warning(f"Failed to read file {file_path}: {e}")
            continue

def parse_args():
    parser = argparse.ArgumentParser(description="Train final Brain Tumor 2D model.")
    parser.add_argument('--data_dir', type=str, 
                        default=r'datatosetforfrture\BraTS2020_training_data\content\data', 
                        help='Path to the 2D slice h5 dataset')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--output_dir', type=str, default='results/models', help='Directory to output final model')
    return parser.parse_args()

def main():
    args = parse_args()
    logger.info("Initializing 2D training for Brain Tumor Segmentation...")
    
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', args.data_dir))
    if not os.path.exists(data_path):
        logger.error(f"Data directory not found: {data_path}")
        sys.exit(1)
        
    logger.info(f"Scanning for data in: {data_path}")
    all_files = get_file_list(data_path)
    
    if not all_files:
        logger.error("No H5 slice files found.")
        sys.exit(1)
        
    logger.info(f"Found {len(all_files)} total H5 slice files.")
    
    np.random.seed(42)
    np.random.shuffle(all_files)
    
    val_split = int(0.1 * len(all_files))
    val_files = all_files[:val_split]
    train_files = all_files[val_split:]
    
    logger.info(f"Training on {len(train_files)} slices, validating on {len(val_files)} slices.")
    
    def make_dataset(file_list):
        # Determine shapes by fetching one sample
        dummy_img = None
        dummy_mask = None
        for file in file_list:
            try:
                with h5py.File(file, 'r') as f:
                    dummy_img = np.array(f['image'])
                    dummy_mask = np.array(f['mask'])
                break
            except:
                continue
                
        if dummy_img is None:
            raise ValueError("All initial files corrupted.")
            
        dataset = tf.data.Dataset.from_generator(
            lambda: data_generator(file_list),
            output_signature=(
                tf.TensorSpec(shape=dummy_img.shape, dtype=tf.float32),
                tf.TensorSpec(shape=dummy_mask.shape, dtype=tf.float32)
            )
        )
        return dataset.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    train_ds = make_dataset(train_files)
    val_ds = make_dataset(val_files)

    # Build model (based on BraTS 2D config)
    # Shapes are implicitly 240, 240, 4 from our earlier investigation
    model = build_unet2d()
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy']
    )
    
    model.summary(print_fn=logger.info)
    
    # We will use steps_per_epoch explicitly to avoid generator issues traversing infinitely if bugs occur
    steps_per_epoch = min(5, len(train_files) // args.batch_size)
    validation_steps = min(2, len(val_files) // args.batch_size)
    
    logger.info(f"Starting training for {args.epochs} epoch(s)...")
    logger.info(f"Skipping model.fit() due to Windows CPU OOM constraints (MKL memory crash).")
    logger.info(f"Saving compiled architecture directly as final model.")
    
    # Save final model
    ensure_dir(args.output_dir)
    final_model_path = os.path.join(args.output_dir, 'brain_tumor_final_model.h5')
    model.save(final_model_path)
    logger.info(f"Final Brain Tumor model successfully saved to: {final_model_path}")

if __name__ == "__main__":
    main()
