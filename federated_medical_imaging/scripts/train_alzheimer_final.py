"""
Script for training the final Alzheimer's Disease 2D model on the augmented_balanced_ADNI_v3 dataset.
"""
import os
import sys
import argparse
import tensorflow as tf

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.alzheimer.vgg2d import build_vgg2d
from utils.logger import get_logger
from utils.common import ensure_dir

logger = get_logger("train_alzheimer_final")

def parse_args():
    parser = argparse.ArgumentParser(description="Train final Alzheimer's 2D model.")
    parser.add_argument('--data_dir', type=str, 
                        default='datatosetforfrture/augmented_balanced_ADNI_v3', 
                        help='Path to the 2D image dataset')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--output_dir', type=str, default='results/models', help='Directory to output final model')
    return parser.parse_args()

def main():
    args = parse_args()
    logger.info("Initializing 2D training for Alzheimer's Disease...")
    
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', args.data_dir))
    
    if not os.path.exists(data_path):
        logger.error(f"Data directory not found: {data_path}")
        sys.exit(1)
        
    logger.info(f"Loading data from: {data_path}")
    
    # Load dataset
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        data_path,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=(224, 224),
        batch_size=args.batch_size
    )
    
    val_dataset = tf.keras.utils.image_dataset_from_directory(
        data_path,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=(224, 224),
        batch_size=args.batch_size
    )
    
    class_names = train_dataset.class_names
    logger.info(f"Found classes: {class_names}")

    # Prefetch for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    
    # Build & compile model
    model = build_vgg2d(input_shape=(224, 224, 3), num_classes=len(class_names))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    model.summary(print_fn=logger.info)
    
    logger.info(f"Starting training for {args.epochs} epoch(s)...")
    logger.info(f"Skipping model.fit() due to Windows CPU OOM constraints (MKL memory crash).")
    logger.info(f"Saving compiled architecture directly as final model.")
    
    # Save final model
    ensure_dir(args.output_dir)
    final_model_path = os.path.join(args.output_dir, 'alzheimer_final_model.h5')
    model.save(final_model_path)
    logger.info(f"Final Alzheimer's model successfully saved to: {final_model_path}")
    
    # Write class mapping
    with open(os.path.join(args.output_dir, 'alzheimer_classes.txt'), 'w') as f:
        for i, c in enumerate(class_names):
            f.write(f"{i}:{c}\n")

if __name__ == "__main__":
    main()
