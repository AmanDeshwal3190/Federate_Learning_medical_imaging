"""
TensorFlow dataset for Alzheimer's Disease classification.
Implements subject-level 5-fold cross-validation (NO data leakage).
"""
import tensorflow as tf
import numpy as np
import os
import glob
import re
import json
from typing import Tuple, List, Dict, Generator
from sklearn.model_selection import StratifiedKFold
from utils.config_loader import ConfigLoader
from utils.logger import get_logger
from utils.common import set_seed, ensure_dir, save_json, load_json

logger = get_logger("alzheimer_dataset")

class AlzheimerDataset:
    """
    Dataset manager for Alzheimer's classification with subject-level CV.
    """
    
    def __init__(self, config_path: str = "config/alzheimer_config.yaml", seed: int = 42):
        self.config = ConfigLoader.load(config_path)
        self.seed = seed
        self.num_folds = self.config.training.cv_folds
        self.batch_size = self.config.training.batch_size
        set_seed(seed)
    
    def scan_processed_directory(self, processed_dir: str) -> Tuple[List[str], List[int]]:
        """
        Scan processed directory and return file paths with labels.
        Labels: AD=1, HC=0
        """
        ad_dir = os.path.join(processed_dir, "AD")
        hc_dir = os.path.join(processed_dir, "HC")
        
        file_paths = []
        labels = []
        
        if os.path.exists(ad_dir):
            for f in glob.glob(os.path.join(ad_dir, "*.npy")):
                file_paths.append(f)
                labels.append(1)
                
        if os.path.exists(hc_dir):
            for f in glob.glob(os.path.join(hc_dir, "*.npy")):
                file_paths.append(f)
                labels.append(0)
                
        return file_paths, labels
    
    def extract_subject_id(self, filepath: str) -> str:
        """
        Extract subject ID from filepath for subject-level splitting.
        """
        filename = os.path.basename(filepath)
        name_without_ext = os.path.splitext(filename)[0]
        
        # subject_001 -> subject_001
        # sub-001_ses-01 -> sub-001
        # OAS1_0001_MR1 -> OAS1_0001
        
        # Try finding sub-XXX
        sub_match = re.search(r'(sub-[a-zA-Z0-9]+)', name_without_ext)
        if sub_match:
            return sub_match.group(1)
            
        # Try OASIS mapping (OAS1_0001)
        oasis_match = re.search(r'(OAS\d_\d+)', name_without_ext)
        if oasis_match:
            return oasis_match.group(1)
            
        # Default split by underscore
        parts = name_without_ext.split('_')
        if len(parts) >= 2:
            return f"{parts[0]}_{parts[1]}"
            
        return name_without_ext
    
    def create_subject_level_cv_splits(self, file_paths: List[str], labels: List[int],
                                        output_dir: str) -> List[Dict[str, List[str]]]:
        """
        Create 5-fold cross-validation splits at the SUBJECT level.
        """
        # 1. Group files by subject_id
        subject_to_files = {}
        subject_to_labels = {}
        
        for fp, label in zip(file_paths, labels):
            subj_id = self.extract_subject_id(fp)
            if subj_id not in subject_to_files:
                subject_to_files[subj_id] = []
                subject_to_labels[subj_id] = []
            
            subject_to_files[subj_id].append(fp)
            subject_to_labels[subj_id].append(label)
            
        # 2. Assign one label per subject
        subjects = list(subject_to_files.keys())
        subj_majority_labels = []
        for s in subjects:
            lbls = subject_to_labels[s]
            majority = max(set(lbls), key=lbls.count)
            subj_majority_labels.append(majority)
            
        subjects = np.array(subjects)
        subj_majority_labels = np.array(subj_majority_labels)
        
        # 3. StratifiedKFold
        skf = StratifiedKFold(n_splits=self.num_folds, shuffle=True, random_state=self.seed)
        
        ensure_dir(output_dir)
        folds = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(subjects, subj_majority_labels)):
            train_subjects = subjects[train_idx]
            val_subjects = subjects[val_idx]
            
            # 4. Expand subject assignments back to file-level
            train_files, train_labels = [], []
            val_files, val_labels = [], []
            
            for s in train_subjects:
                train_files.extend(subject_to_files[s])
                train_labels.extend(subject_to_labels[s])
                
            for s in val_subjects:
                val_files.extend(subject_to_files[s])
                val_labels.extend(subject_to_labels[s])
                
            fold_data = {
                'train_files': train_files,
                'train_labels': train_labels,
                'val_files': val_files,
                'val_labels': val_labels
            }
            folds.append(fold_data)
            
            # 6. Save fold
            save_json(fold_data, os.path.join(output_dir, f"fold_{fold_idx}_split.json"))
            
        return folds
    
    def create_tf_dataset(self, file_paths: List[str], labels: List[int],
                           dataset_type: str = "adni", 
                           batch_size: int = None,
                           shuffle: bool = True) -> tf.data.Dataset:
        """
        Create a tf.data.Dataset from file paths and labels.
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        expected_shape = self.config.dataset.adni.final_dimensions if dataset_type.lower() == "adni" else self.config.dataset.oasis.final_dimensions
        expected_shape = tuple(expected_shape) + (1,)
        
        def load_npy(filepath_tensor, label_tensor):
            filepath = filepath_tensor.numpy().decode('utf-8')
            data = np.load(filepath).astype(np.float32)
            # Ensure shape matches expected
            if data.shape != expected_shape:
                 # Truncation or padding might be needed if something went wrong in prep
                 pass
            return data, label_tensor
            
        def tf_load(filepath, label):
            [data, label_mapped] = tf.py_function(load_npy, [filepath, label], [tf.float32, tf.int32])
            data.set_shape(expected_shape)
            label_mapped.set_shape([])
            return data, label_mapped
            
        dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(file_paths) if len(file_paths) > 0 else 1)
            
        dataset = dataset.map(tf_load, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def get_fold_datasets(self, fold_index: int, splits_dir: str,
                           dataset_type: str = "adni"
                           ) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Get train and validation datasets for a specific fold.
        """
        fold_file = os.path.join(splits_dir, f"fold_{fold_index}_split.json")
        fold_data = load_json(fold_file)
        
        train_ds = self.create_tf_dataset(fold_data['train_files'], fold_data['train_labels'], dataset_type, shuffle=True)
        val_ds = self.create_tf_dataset(fold_data['val_files'], fold_data['val_labels'], dataset_type, shuffle=False)
        
        return train_ds, val_ds
    
    def get_all_folds(self, splits_dir: str, dataset_type: str = "adni"
                       ) -> List[Tuple[tf.data.Dataset, tf.data.Dataset]]:
        """
        Get all 5 fold datasets for cross-validation.
        """
        all_folds = []
        for i in range(self.num_folds):
            all_folds.append(self.get_fold_datasets(i, splits_dir, dataset_type))
            
        return all_folds
