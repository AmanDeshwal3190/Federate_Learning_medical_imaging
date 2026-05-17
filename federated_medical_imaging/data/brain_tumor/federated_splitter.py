"""
Split brain tumor data among federated clients (simulated hospitals).
Supports both IID and non-IID distributions.
"""
import numpy as np
import os
import json
import random
from typing import Dict, List, Tuple
from utils.config_loader import ConfigLoader
from utils.logger import get_logger
from utils.common import save_json, load_json, ensure_dir

logger = get_logger("federated_splitter")


class FederatedDataSplitter:
    """Split dataset among multiple clients for federated learning."""
    
    def __init__(self, num_clients: int = 3, distribution: str = "iid", seed: int = 42):
        self.num_clients = num_clients
        self.distribution = distribution
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
    
    def split_iid(self, file_list: List[str]) -> Dict[int, List[str]]:
        """
        Split data equally among clients in IID fashion.
        Each client gets a random, equally-sized subset.
        """
        shuffled = list(file_list)
        random.shuffle(shuffled)
        
        splits = {}
        chunk_size = len(shuffled) // self.num_clients
        for i in range(self.num_clients):
            start = i * chunk_size
            # The last client takes whatever is remaining
            end = len(shuffled) if i == self.num_clients - 1 else (i + 1) * chunk_size
            splits[i] = shuffled[start:end]
            
        return splits
    
    def split_non_iid(self, file_list: List[str], labels: List[int],
                       alpha: float = 0.5) -> Dict[int, List[str]]:
        """
        Split data among clients in non-IID fashion using Dirichlet distribution.
        """
        if len(file_list) != len(labels):
            raise ValueError("Length of file_list and labels must be equal.")
            
        unique_labels = list(set(labels))
        num_classes = len(unique_labels)
        
        # Group file indices by label
        indices_by_label = {label: [] for label in unique_labels}
        for idx, label in enumerate(labels):
            indices_by_label[label].append(idx)
            
        client_indices = {i: [] for i in range(self.num_clients)}
        
        for label in unique_labels:
            label_indices = indices_by_label[label]
            random.shuffle(label_indices)
            
            # Sample proportions for this label across clients
            proportions = np.random.dirichlet([alpha] * self.num_clients)
            # Scale proportions to number of items and convert to integers
            splits = (proportions * len(label_indices)).astype(int)
            
            start = 0
            for i in range(self.num_clients):
                if i == self.num_clients - 1:
                    chunk = label_indices[start:]
                else:
                    end = start + splits[i]
                    chunk = label_indices[start:end]
                    start = end
                client_indices[i].extend(chunk)
                
        # Map indices back to file paths and shuffle within each client
        splits_files = {}
        for i in range(self.num_clients):
            files = [file_list[idx] for idx in client_indices[i]]
            random.shuffle(files)
            splits_files[i] = files
            
        return splits_files

    def split_by_hospital(self, file_list: List[str], 
                           fractions: List[float]) -> Dict[int, List[str]]:
        """
        Split data according to specified fractions per hospital.
        """
        if abs(sum(fractions) - 1.0) > 1e-5:
            raise ValueError("Fractions must sum to 1.0")
            
        shuffled = list(file_list)
        random.shuffle(shuffled)
        
        splits = {}
        num_files = len(shuffled)
        start = 0
        for i, frac in enumerate(fractions):
            if i == len(fractions) - 1:
                splits[i] = shuffled[start:]
            else:
                end = start + int(frac * num_files)
                splits[i] = shuffled[start:end]
                start = end
                
        return splits
    
    def save_splits(self, splits: Dict[int, List[str]], output_dir: str) -> None:
        """Save client data splits as JSON files."""
        ensure_dir(output_dir)
        for client_id, files in splits.items():
            file_path = os.path.join(output_dir, f"client_{client_id}_files.json")
            save_json(files, file_path)
            logger.info(f"Saved {len(files)} files for client {client_id} to {file_path}")
    
    def load_splits(self, splits_dir: str) -> Dict[int, List[str]]:
        """Load previously saved client data splits."""
        splits = {}
        for filename in os.listdir(splits_dir):
            if filename.startswith("client_") and filename.endswith("_files.json"):
                # Extract client ID length
                parts = filename.replace("_files.json", "").split("_")
                if len(parts) == 2 and parts[1].isdigit():
                    client_id = int(parts[1])
                    file_path = os.path.join(splits_dir, filename)
                    splits[client_id] = load_json(file_path)
        return splits
    
    def print_split_statistics(self, splits: Dict[int, List[str]], 
                                labels: Dict[int, List[int]] = None) -> None:
        """Print statistics about the data distribution across clients."""
        logger.info("=== Federated Split Statistics ===")
        total_samples = sum(len(files) for files in splits.values())
        logger.info(f"Total samples distributed: {total_samples}")
        
        for client_id, files in sorted(splits.items()):
            msg = f"Client {client_id}: {len(files)} samples ({(len(files)/total_samples)*100:.1f}%)"
            if labels and client_id in labels:
                client_labels = labels[client_id]
                unique, counts = np.unique(client_labels, return_counts=True)
                dist = {u: c for u, c in zip(unique, counts)}
                msg += f" | Label distribution: {dist}"
            logger.info(msg)
