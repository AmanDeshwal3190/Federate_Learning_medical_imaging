"""
Split Alzheimer's data among federated clients.
Supports subject-level splitting to prevent data leakage in FL setting.
"""
import numpy as np
import os
import json
from typing import Dict, List, Tuple
from collections import defaultdict
from utils.config_loader import ConfigLoader
from utils.logger import get_logger
from utils.common import save_json, ensure_dir, load_json

logger = get_logger("alzheimer_federated_splitter")

class AlzheimerFederatedSplitter:
    """
    Split Alzheimer's data among FL clients with subject-level integrity.
    Ensures no subject's data is split across multiple clients.
    """
    
    def __init__(self, num_clients: int = 3, seed: int = 42):
        self.num_clients = num_clients
        self.seed = seed
        np.random.seed(seed)
    
    def split_by_subject(self, file_paths: List[str], labels: List[int],
                          subject_ids: List[str]) -> Dict[int, Dict[str, list]]:
        """
        Split data by subject across clients.
        Each client gets complete subjects (all scans from a subject stay together).
        """
        subject_to_files = defaultdict(list)
        subject_to_labels = defaultdict(list)
        
        for fp, lbl, sid in zip(file_paths, labels, subject_ids):
            subject_to_files[sid].append(fp)
            subject_to_labels[sid].append(lbl)
            
        unique_subjects = list(subject_to_files.keys())
        np.random.shuffle(unique_subjects)
        
        # Determine labels for balancing (majority vote per subject)
        subject_major_labels = {}
        for s in unique_subjects:
            lbls = subject_to_labels[s]
            subject_major_labels[s] = max(set(lbls), key=lbls.count)
            
        # Group subjects by their majority label
        ad_subjects = [s for s in unique_subjects if subject_major_labels[s] == 1]
        hc_subjects = [s for s in unique_subjects if subject_major_labels[s] == 0]
        
        splits = {i: {'files': [], 'labels': [], 'subjects': []} for i in range(self.num_clients)}
        
        def distribute_subjects(sub_list, splits_dict):
            # Round robin distribution
            for idx, s in enumerate(sub_list):
                client_id = idx % self.num_clients
                splits_dict[client_id]['subjects'].append(s)
                splits_dict[client_id]['files'].extend(subject_to_files[s])
                splits_dict[client_id]['labels'].extend(subject_to_labels[s])
                
        distribute_subjects(ad_subjects, splits)
        distribute_subjects(hc_subjects, splits)
        
        return splits
    
    def split_by_institution(self, adni_files: List[str], adni_labels: List[int], adni_subjects: List[str],
                               oasis_files: List[str], oasis_labels: List[int], oasis_subjects: List[str]
                               ) -> Dict[int, Dict[str, list]]:
        """
        Simulate institution-based splitting: 
        - Client 0: ADNI data (Hospital A with 3T scanner)
        - Client 1: OASIS data (Hospital B with 1.5T scanner)
        - Client 2: Mixed subset from both
        """
        # Create full ADNI dict
        adni_dict = defaultdict(lambda: {'files': [], 'labels': []})
        for f, l, s in zip(adni_files, adni_labels, adni_subjects):
            adni_dict[s]['files'].append(f)
            adni_dict[s]['labels'].append(l)
            
        # Create full OASIS dict
        oasis_dict = defaultdict(lambda: {'files': [], 'labels': []})
        for f, l, s in zip(oasis_files, oasis_labels, oasis_subjects):
            oasis_dict[s]['files'].append(f)
            oasis_dict[s]['labels'].append(l)
            
        adni_unique_subjects = list(adni_dict.keys())
        oasis_unique_subjects = list(oasis_dict.keys())
        np.random.shuffle(adni_unique_subjects)
        np.random.shuffle(oasis_unique_subjects)
        
        # Take 20% of ADNI and 20% of OASIS for Client 2
        adni_c2_split = int(0.2 * len(adni_unique_subjects))
        oasis_c2_split = int(0.2 * len(oasis_unique_subjects))
        
        c0_subjects = adni_unique_subjects[adni_c2_split:]
        c1_subjects = oasis_unique_subjects[oasis_c2_split:]
        c2_adni_subjects = adni_unique_subjects[:adni_c2_split]
        c2_oasis_subjects = oasis_unique_subjects[:oasis_c2_split]
        
        splits = {0: {'files': [], 'labels': [], 'subjects': []},
                  1: {'files': [], 'labels': [], 'subjects': []},
                  2: {'files': [], 'labels': [], 'subjects': []}}
                  
        def assign_subjects(client_id, subjects, src_dict):
            for s in subjects:
                splits[client_id]['subjects'].append(s)
                splits[client_id]['files'].extend(src_dict[s]['files'])
                splits[client_id]['labels'].extend(src_dict[s]['labels'])
                
        assign_subjects(0, c0_subjects, adni_dict)
        assign_subjects(1, c1_subjects, oasis_dict)
        assign_subjects(2, c2_adni_subjects, adni_dict)
        assign_subjects(2, c2_oasis_subjects, oasis_dict)
        
        return splits
    
    def save_splits(self, splits: Dict[int, Dict[str, list]], output_dir: str) -> None:
        """Save splits as JSON files per client."""
        ensure_dir(output_dir)
        for client_id, data in splits.items():
            filepath = os.path.join(output_dir, f"client_{client_id}_split.json")
            save_json(data, filepath)
            logger.info(f"Saved Client {client_id} split to {filepath}")
    
    def load_splits(self, splits_dir: str) -> Dict[int, Dict[str, list]]:
        """Load previously saved splits."""
        splits = {}
        client_id = 0
        while True:
            filepath = os.path.join(splits_dir, f"client_{client_id}_split.json")
            if not os.path.exists(filepath):
                break
            splits[client_id] = load_json(filepath)
            client_id += 1
        return splits
    
    def print_statistics(self, splits: Dict[int, Dict[str, list]]) -> None:
        """Print detailed statistics about federated data distribution."""
        print("-" * 50)
        print("FEDERATED DATA SPLIT STATISTICS")
        print("-" * 50)
        for client_id, data in splits.items():
            num_samples = len(data['files'])
            num_subjects = len(data['subjects'])
            ad_count = sum(1 for label in data['labels'] if label == 1)
            hc_count = sum(1 for label in data['labels'] if label == 0)
            
            print(f"Client {client_id}:")
            print(f"  Total Samples : {num_samples}")
            print(f"  Total Subjects: {num_subjects}")
            print(f"  Class Balance : {ad_count} AD / {hc_count} HC")
            print("-" * 30)
