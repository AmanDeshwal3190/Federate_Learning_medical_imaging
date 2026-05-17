import tensorflow as tf
import numpy as np
import os
import json
import glob
from typing import Tuple, List, Dict, Optional, Any
from sklearn.model_selection import train_test_split
from utils.config_loader import ConfigLoader
from utils.logger import get_logger
from utils.common import set_seed, ensure_dir, save_json, load_json
from data.brain_tumor.augmentation import BrainTumorAugmentor

logger = get_logger("brain_tumor_dataset")


def _coerce_str_list(raw_data: Any, split_file: str) -> List[str]:
    """Validate split JSON payload and return a list of file paths."""
    if not isinstance(raw_data, list):
        raise ValueError(f"Expected list in split file: {split_file}")
    if not all(isinstance(item, str) for item in raw_data):
        raise ValueError(f"Split file contains non-string items: {split_file}")
    return raw_data


class BraTSDataset:
    """
    TensorFlow dataset for BraTS2020 brain tumor segmentation.
    Loads preprocessed .npy volumes and creates tf.data.Dataset pipelines.
    """
    
    def __init__(self, config_path: str = "config/brain_tumor_config.yaml", seed: int = 42):
        self.config = ConfigLoader.load(config_path)
        self.seed = seed
        self.augmentor = BrainTumorAugmentor(seed=seed)
        set_seed(seed)
    
    def create_splits(self, processed_dir: str, splits_dir: str) -> Dict[str, List[str]]:
        """
        Split preprocessed data into train/val/test sets.
        """
        ensure_dir(splits_dir)
        # Gather all .npy files
        files = glob.glob(os.path.join(processed_dir, "*.npy"))
        
        # Determine labels if tumor types are provided in names (e.g. HGG/LGG)
        # If not, just split randomly
        labels = []
        for f in files:
            fname = os.path.basename(f)
            if "HGG" in fname: labels.append("HGG")
            elif "LGG" in fname: labels.append("LGG")
            else: labels.append("Unknown")
            
        train_ratio = self.config.dataset.train_split
        val_ratio = self.config.dataset.val_split
        test_ratio = self.config.dataset.test_split
        
        if len(set(labels)) > 1:
            try:
                train_f, temp_f, train_l, temp_l = train_test_split(
                    files, labels, train_size=train_ratio, stratify=labels, random_state=self.seed
                )
                val_f, test_f = train_test_split(
                    temp_f, test_size=(test_ratio / (test_ratio + val_ratio)), stratify=temp_l, random_state=self.seed
                )
            except ValueError:
                # Fallback if too few samples for stratify
                train_f, temp_f = train_test_split(files, train_size=train_ratio, random_state=self.seed)
                val_f, test_f = train_test_split(temp_f, test_size=(test_ratio / (test_ratio + val_ratio)), random_state=self.seed)
        else:
            train_f, temp_f = train_test_split(files, train_size=train_ratio, random_state=self.seed)
            val_f, test_f = train_test_split(temp_f, test_size=(test_ratio / (test_ratio + val_ratio)), random_state=self.seed)
            
        splits = {
            "train": train_f,
            "val": val_f,
            "test": test_f
        }
        
        for k, v in splits.items():
            save_json(v, os.path.join(splits_dir, f"brats_{k}.json"))
            logger.info(f"Created BraTS {k} split with {len(v)} files.")
            
        return splits
    
    def _load_volume_pair(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load a preprocessed volume and its corresponding mask."""
        data = np.load(file_path, allow_pickle=True).item()
        vol = data["volume"]
        mask = data["mask"]
        
        # One-hot encode mask if needed or keep as integer map
        # Background(0), NCR/NET(1), Edema(2), Enhancing Tumor(4)
        processed_mask = np.zeros_like(mask)
        processed_mask[mask == 1] = 1
        processed_mask[mask == 2] = 2
        processed_mask[mask == 4] = 3
        # Ensure exact types
        return vol.astype(np.float32), processed_mask.astype(np.int32)
    
    def _tf_load_and_augment(self, file_path: tf.Tensor, augment: bool) -> Tuple[tf.Tensor, tf.Tensor]:
        def _func(path_tensor):
            path = path_tensor.numpy().decode('utf-8')
            vol, mask = self._load_volume_pair(path)
            if augment:
                vol, mask = self.augmentor.augment_3d(vol, mask)
            # Expand dims or one-hot encode mask to [H,W,D,C]
            mask_oh = tf.keras.utils.to_categorical(mask, num_classes=self.config.dataset.num_classes)
            return vol.astype(np.float32), mask_oh.astype(np.float32)
            
        vol_shape = self.config.model.segmentation.input_shape # [128, 128, 128, 4]
        mask_shape = vol_shape[:3] + [self.config.dataset.num_classes] # [128, 128, 128, 4]
        
        vol, mask = tf.py_function(
            func=_func,
            inp=[file_path],
            Tout=[tf.float32, tf.float32]
        )
        vol.set_shape(vol_shape)
        mask.set_shape(mask_shape)
        return vol, mask

    def create_tf_dataset(self, file_list: List[str], batch_size: int = 4, 
                           shuffle: bool = True, augment: bool = False) -> tf.data.Dataset:
        """
        Create a tf.data.Dataset from a list of preprocessed .npy file paths.
        """
        ds = tf.data.Dataset.from_tensor_slices(file_list)
        if shuffle:
            ds = ds.shuffle(buffer_size=len(file_list), seed=self.seed)
            
        ds = ds.map(lambda x: self._tf_load_and_augment(x, augment),
                    num_parallel_calls=tf.data.AUTOTUNE)
        
        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds
        
    def get_train_val_test_datasets(self, splits_dir: str, batch_size: int = 4
                                     ) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Get ready-to-use train, validation, and test datasets.
        """
        train_split_path = os.path.join(splits_dir, "brats_train.json")
        val_split_path = os.path.join(splits_dir, "brats_val.json")
        test_split_path = os.path.join(splits_dir, "brats_test.json")

        train_files = _coerce_str_list(load_json(train_split_path), train_split_path)
        val_files = _coerce_str_list(load_json(val_split_path), val_split_path)
        test_files = _coerce_str_list(load_json(test_split_path), test_split_path)
        
        train_ds = self.create_tf_dataset(train_files, batch_size, shuffle=True, augment=True)
        val_ds = self.create_tf_dataset(val_files, batch_size, shuffle=False, augment=False)
        test_ds = self.create_tf_dataset(test_files, batch_size, shuffle=False, augment=False)
        
        return train_ds, val_ds, test_ds


class FigshareDataset:
    """
    TensorFlow dataset for Figshare brain tumor classification.
    """
    
    def __init__(self, config_path: str = "config/brain_tumor_config.yaml", seed: int = 42):
        self.config = ConfigLoader.load(config_path)
        self.seed = seed
        self.class_names = ["Glioma", "Meningioma", "Pituitary"]
        self.augmentor = BrainTumorAugmentor(seed=seed)
        set_seed(seed)
    
    def create_splits(self, processed_dir: str, splits_dir: str) -> Dict[str, List[str]]:
        """
        Split Figshare data into train/val/test with stratification by class.
        """
        ensure_dir(splits_dir)
        files = glob.glob(os.path.join(processed_dir, "*.npy"))
        
        labels = []
        for f in files:
            fname = os.path.basename(f)
            # Extracted during preprocessing "glioma_...", etc.
            if fname.startswith("glioma"): labels.append(0)
            elif fname.startswith("meningioma"): labels.append(1)
            elif fname.startswith("pituitary"): labels.append(2)
            else: labels.append(-1)
            
        train_ratio = self.config.dataset.train_split
        val_ratio = self.config.dataset.val_split
        test_ratio = self.config.dataset.test_split
        
        train_f, temp_f, train_l, temp_l = train_test_split(
            files, labels, train_size=train_ratio, stratify=labels, random_state=self.seed
        )
        val_f, test_f = train_test_split(
            temp_f, test_size=(test_ratio / (test_ratio + val_ratio)), stratify=temp_l, random_state=self.seed
        )
        
        splits = {"train": train_f, "val": val_f, "test": test_f}
        
        for k, v in splits.items():
            save_json(v, os.path.join(splits_dir, f"figshare_{k}.json"))
            logger.info(f"Created Figshare {k} split with {len(v)} files.")
            
        return splits
        
    def _load_image_pair(self, file_path: str) -> Tuple[np.ndarray, int]:
        data = np.load(file_path, allow_pickle=True).item()
        return data["image"].astype(np.float32), int(data["label"])

    def _tf_load_and_augment(self, file_path: tf.Tensor, augment: bool) -> Tuple[tf.Tensor, tf.Tensor]:
        def _func(path_tensor):
            path = path_tensor.numpy().decode('utf-8')
            img, label = self._load_image_pair(path)
            if augment:
                img, label = self.augmentor.augment_2d(img, label)
            label_oh = tf.keras.utils.to_categorical(label, num_classes=3)
            return img.astype(np.float32), label_oh.astype(np.float32)
            
        img, label = tf.py_function(
            func=_func,
            inp=[file_path],
            Tout=[tf.float32, tf.float32]
        )
        img.set_shape([224, 224, 1])
        label.set_shape([3])
        return img, label

    def create_tf_dataset(self, file_list: List[str], batch_size: int = 32,
                           shuffle: bool = True, augment: bool = False) -> tf.data.Dataset:
        """
        Create tf.data.Dataset for classification.
        """
        ds = tf.data.Dataset.from_tensor_slices(file_list)
        if shuffle:
            ds = ds.shuffle(buffer_size=len(file_list), seed=self.seed)
            
        ds = ds.map(lambda x: self._tf_load_and_augment(x, augment),
                    num_parallel_calls=tf.data.AUTOTUNE)
        
        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds
        
    def get_train_val_test_datasets(self, splits_dir: str, batch_size: int = 32
                                     ) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Get ready-to-use train, validation, and test datasets."""
        train_split_path = os.path.join(splits_dir, "figshare_train.json")
        val_split_path = os.path.join(splits_dir, "figshare_val.json")
        test_split_path = os.path.join(splits_dir, "figshare_test.json")

        train_files = _coerce_str_list(load_json(train_split_path), train_split_path)
        val_files = _coerce_str_list(load_json(val_split_path), val_split_path)
        test_files = _coerce_str_list(load_json(test_split_path), test_split_path)
        
        train_ds = self.create_tf_dataset(train_files, batch_size, shuffle=True, augment=True)
        val_ds = self.create_tf_dataset(val_files, batch_size, shuffle=False, augment=False)
        test_ds = self.create_tf_dataset(test_files, batch_size, shuffle=False, augment=False)
        
        return train_ds, val_ds, test_ds
