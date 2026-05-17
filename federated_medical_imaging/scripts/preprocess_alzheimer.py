"""
Main script for Alzheimer's data preprocessing pipeline.
"""
import argparse
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.alzheimer.preprocessing import AlzheimerPreprocessor
from data.alzheimer.dataset import AlzheimerDataset
from data.alzheimer.federated_splitter import AlzheimerFederatedSplitter
from utils.config_loader import ConfigLoader
from utils.logger import get_logger
from utils.common import save_json, ensure_dir

logger = get_logger("preprocess_alzheimer")

def main():
    parser = argparse.ArgumentParser(description="Alzheimer Data Preprocessing")
    parser.add_argument("--config", type=str, default="config/alzheimer_config.yaml")
    parser.add_argument("--dataset", type=str, choices=["adni", "oasis", "both"], default="both")
    parser.add_argument("--template", type=str, help="Path to MNI152 template")
    args = parser.parse_args()
    
    config = ConfigLoader.load(args.config)
    adni_raw = config.dataset.adni_path
    oasis_raw = config.dataset.oasis_path
    processed_dir = config.dataset.processed_path
    splits_dir = config.dataset.splits_path
    
    ensure_dir(processed_dir)
    ensure_dir(splits_dir)
    
    preprocessor = AlzheimerPreprocessor(args.config)
    
    total_stats = {}
    
    # Process ADNI
    if args.dataset in ["adni", "both"]:
        logger.info("Starting ADNI Preprocessing...")
        adni_out = os.path.join(processed_dir, "ADNI")
        stats = preprocessor.preprocess_dataset(adni_raw, adni_out, "adni", args.template)
        total_stats['ADNI'] = stats
        
    # Process OASIS
    if args.dataset in ["oasis", "both"]:
        logger.info("Starting OASIS Preprocessing...")
        oasis_out = os.path.join(processed_dir, "OASIS")
        stats = preprocessor.preprocess_dataset(oasis_raw, oasis_out, "oasis", args.template)
        total_stats['OASIS'] = stats
        
    # Datasets
    logger.info("Creating CV splits and Federated splits...")
    dataset_mgr = AlzheimerDataset(args.config)
    
    # To handle both, we pool filepaths and track origin
    # Here we simplify: if user selected "both", we can do split_by_institution
    all_files = []
    all_labels = []
    all_subjects = []
    
    adni_files, adni_labels, adni_subs = [], [], []
    oasis_files, oasis_labels, oasis_subs = [], [], []
    
    if args.dataset in ["adni", "both"] and os.path.exists(os.path.join(processed_dir, "ADNI")):
        a_files, a_labels = dataset_mgr.scan_processed_directory(os.path.join(processed_dir, "ADNI"))
        a_subs = [dataset_mgr.extract_subject_id(f) for f in a_files]
        all_files.extend(a_files)
        all_labels.extend(a_labels)
        all_subjects.extend(a_subs)
        adni_files, adni_labels, adni_subs = a_files, a_labels, a_subs
        
        cv_dir = os.path.join(splits_dir, "cv_adni")
        dataset_mgr.create_subject_level_cv_splits(a_files, a_labels, cv_dir)
        
    if args.dataset in ["oasis", "both"] and os.path.exists(os.path.join(processed_dir, "OASIS")):
        o_files, o_labels = dataset_mgr.scan_processed_directory(os.path.join(processed_dir, "OASIS"))
        o_subs = [dataset_mgr.extract_subject_id(f) for f in o_files]
        all_files.extend(o_files)
        all_labels.extend(o_labels)
        all_subjects.extend(o_subs)
        oasis_files, oasis_labels, oasis_subs = o_files, o_labels, o_subs
        
        cv_dir = os.path.join(splits_dir, "cv_oasis")
        dataset_mgr.create_subject_level_cv_splits(o_files, o_labels, cv_dir)
        
    # Federated splits
    splitter = AlzheimerFederatedSplitter(num_clients=3)
    if args.dataset == "both":
        logger.info("Creating Institution-based Federated Splits...")
        fed_splits = splitter.split_by_institution(adni_files, adni_labels, adni_subs,
                                                   oasis_files, oasis_labels, oasis_subs)
    else:
        logger.info("Creating Subject-based Federated Splits...")
        fed_splits = splitter.split_by_subject(all_files, all_labels, all_subjects)
        
    fed_dir = os.path.join(splits_dir, "federated")
    splitter.save_splits(fed_splits, fed_dir)
    splitter.print_statistics(fed_splits)
    
    report_path = "results/logs/preprocessing_alzheimer_report.json"
    ensure_dir("results/logs")
    save_json(total_stats, report_path)
    logger.info(f"Preprocessing fully complete. Report saved to {report_path}")

if __name__ == "__main__":
    main()
