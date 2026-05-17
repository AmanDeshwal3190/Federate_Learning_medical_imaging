"""
Main script to run the complete brain tumor data preprocessing pipeline.

Usage:
    python scripts/preprocess_brain_tumor.py --config config/brain_tumor_config.yaml

Steps:
1. Verify raw data exists
2. Preprocess BraTS2020 volumes (all modalities)
3. Preprocess Figshare images
4. Create train/val/test splits
5. Generate data statistics report
6. Save preprocessing report as JSON
"""
import argparse
import os
import sys
import time
import json
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.brain_tumor.preprocessing import BrainTumorPreprocessor
from data.brain_tumor.dataset import BraTSDataset, FigshareDataset
from utils.config_loader import ConfigLoader
from utils.logger import get_logger
from utils.common import save_json, ensure_dir

logger = get_logger("preprocess_brain_tumor")

def main():
    parser = argparse.ArgumentParser(description="Brain Tumor Preprocessing Pipeline")
    parser.add_argument("--config", type=str, default="config/brain_tumor_config.yaml",
                        help="Path to the config file")
    args = parser.parse_args()
    
    start_time = time.time()
    logger.info(f"Loading configuration from {args.config}")
    config = ConfigLoader.load(args.config)
    
    brats_raw = config.dataset.brats_path
    figshare_raw = config.dataset.figshare_path
    processed_dir = config.dataset.processed_path
    splits_dir = config.dataset.splits_path
    
    # Check if raw directories exist
    if not os.path.exists(brats_raw):
        logger.warning(f"BraTS2020 raw directory not found: {brats_raw}. Please run download script first.")
    if not os.path.exists(figshare_raw):
        logger.warning(f"Figshare raw directory not found: {figshare_raw}. Please run download script first.")
        
    ensure_dir(processed_dir)
    ensure_dir(splits_dir)
    
    preprocessor = BrainTumorPreprocessor(args.config)
    
    brats_processed_dir = os.path.join(processed_dir, "brats")
    figshare_processed_dir = os.path.join(processed_dir, "figshare")
    
    # Preprocess BraTS
    if os.path.exists(brats_raw) and len(os.listdir(brats_raw)) > 0:
        logger.info("Starting BraTS2020 preprocessing...")
        preprocessor.preprocess_all_brats(brats_raw, brats_processed_dir)
    else:
        logger.info("Skipping BraTS2020 preprocessing due to missing data.")

    # Preprocess Figshare
    if os.path.exists(figshare_raw) and len(os.listdir(figshare_raw)) > 0:
        logger.info("Starting Figshare preprocessing...")
        preprocessor.preprocess_all_figshare(figshare_raw, figshare_processed_dir)
    else:
        logger.info("Skipping Figshare preprocessing due to missing data.")

    logger.info("Creating data splits...")
    brats_splits = {}
    figshare_splits = {}
    
    if os.path.exists(brats_processed_dir):
        brats_ds = BraTSDataset(args.config)
        brats_splits = brats_ds.create_splits(brats_processed_dir, splits_dir)
        
    if os.path.exists(figshare_processed_dir):
        figshare_ds = FigshareDataset(args.config)
        figshare_splits = figshare_ds.create_splits(figshare_processed_dir, splits_dir)
        
    total_time = time.time() - start_time
    
    logger.info("=== Data Statistics Report ===")
    logger.info(f"Total preprocessing time: {total_time:.2f} seconds")
    
    report = {
        "preprocessing_time_seconds": total_time,
        "brats2020": {
            "train": len(brats_splits.get("train", [])),
            "val": len(brats_splits.get("val", [])),
            "test": len(brats_splits.get("test", []))
        },
        "figshare": {
            "train": len(figshare_splits.get("train", [])),
            "val": len(figshare_splits.get("val", [])),
            "test": len(figshare_splits.get("test", []))
        }
    }
    
    def log_split_stats(dataset_name, dataset_splits):
        if dataset_splits:
            t = len(dataset_splits.get("train", []))
            v = len(dataset_splits.get("val", []))
            te = len(dataset_splits.get("test", []))
            logger.info(f"{dataset_name} -> Train: {t}, Val: {v}, Test: {te}")
            
    log_split_stats("BraTS2020", brats_splits)
    log_split_stats("Figshare", figshare_splits)
    
    report_file = "results/logs/preprocessing_brain_tumor_report.json"
    save_json(report, report_file)
    logger.info(f"Saved preprocessing report to {report_file}")


if __name__ == "__main__":
    main()
