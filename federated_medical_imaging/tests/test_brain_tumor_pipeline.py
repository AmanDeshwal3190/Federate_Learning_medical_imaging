"""Tests for the brain tumor data pipeline."""
import pytest
import numpy as np
import os
import shutil
import tempfile
import json
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.brain_tumor.preprocessing import BrainTumorPreprocessor
from data.brain_tumor.augmentation import BrainTumorAugmentor
from data.brain_tumor.dataset import BraTSDataset, FigshareDataset
from data.brain_tumor.federated_splitter import FederatedDataSplitter

@pytest.fixture
def preprocessor():
    # Setup minimal config for test if necessary or mock
    # Using defaults from the class which loads config/brain_tumor_config.yaml
    prep = BrainTumorPreprocessor.__new__(BrainTumorPreprocessor)
    prep.gaussian_kernel = 5
    prep.gaussian_sigma = 1.0
    prep.agsw_window = 7
    prep.guided_radius = 8
    prep.guided_eps = 0.01
    prep.clahe_clip = 2.0
    prep.clahe_grid = (8, 8)
    return prep

def test_gaussian_filter(preprocessor):
    vol = np.random.rand(16, 16, 16).astype(np.float32)
    filtered = preprocessor.gaussian_filter_3d(vol)
    assert filtered.shape == vol.shape
    # Check smoothing
    assert np.std(filtered) < np.std(vol)

def test_normalize_volume(preprocessor):
    vol = np.random.rand(16, 16, 16) * 100 + 50 # random values between 50 and 150
    norm = preprocessor.normalize_volume(vol)
    assert norm.shape == vol.shape
    assert np.min(norm) >= 0.0
    assert np.max(norm) <= 1.0

def test_clahe(preprocessor):
    img = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
    enhanced = preprocessor.apply_clahe(img)
    assert enhanced.shape == img.shape
    assert enhanced.dtype == np.uint8

def test_guided_filter(preprocessor):
    img = np.random.rand(16, 16).astype(np.float32)
    guide = img.copy()
    filtered = preprocessor.guided_filter(img, guide)
    assert filtered.shape == img.shape

def test_agsw(preprocessor):
    vol = np.random.rand(16, 16, 8).astype(np.float32)
    agsw = preprocessor.anisotropic_gaussian_side_window(vol)
    assert agsw.shape == vol.shape

@pytest.fixture
def augmentor():
    return BrainTumorAugmentor(rotation_range=15, horizontal_flip=True, 
                               zoom_range=(0.9, 1.1), brightness_range=(0.9, 1.1))

def test_augment_3d_shape(augmentor):
    vol = np.random.rand(32, 32, 32, 4)
    mask = np.random.randint(0, 4, (32, 32, 32))
    aug_vol, aug_mask = augmentor.augment_3d(vol, mask)
    assert aug_vol.shape == vol.shape
    assert aug_mask.shape == mask.shape

def test_augment_2d_shape(augmentor):
    img = np.random.rand(64, 64, 1)
    label = 1
    aug_img, out_label = augmentor.augment_2d(img, label)
    assert aug_img.shape == img.shape
    assert out_label == label

def test_augment_preserves_label(augmentor):
    img = np.random.rand(32, 32)
    for label in [0, 1, 2]:
        _, out_label = augmentor.augment_2d(img, label)
        assert out_label == label

# Use tempdir for dataset tests
@pytest.fixture
def temp_dataset_dir():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

def test_create_splits(temp_dataset_dir):
    # Mock preprocessed directory
    proc_dir = os.path.join(temp_dataset_dir, "processed")
    splits_dir = os.path.join(temp_dataset_dir, "splits")
    os.makedirs(proc_dir)
    os.makedirs(splits_dir)
    
    # Create 100 dummy files
    for i in range(100):
        with open(os.path.join(proc_dir, f"file_{i}.npy"), 'w') as f:
            f.write("dummy")
            
    # Mock dataset
    ds = BraTSDataset.__new__(BraTSDataset)
    ds.seed = 42
    # mock config values inline
    class MockConfig:
        class dataset:
            train_split = 0.7
            val_split = 0.15
            test_split = 0.15
    ds.config = MockConfig()
    
    splits = ds.create_splits(proc_dir, splits_dir)
    
    assert len(splits["train"]) == 70
    assert len(splits["val"]) == 15
    assert len(splits["test"]) == 15

def test_no_data_leakage(temp_dataset_dir):
    proc_dir = os.path.join(temp_dataset_dir, "processed")
    splits_dir = os.path.join(temp_dataset_dir, "splits")
    os.makedirs(proc_dir)
    os.makedirs(splits_dir)
    
    for i in range(100):
        with open(os.path.join(proc_dir, f"file_{i}.npy"), 'w') as f: f.write("dummy")
            
    ds = BraTSDataset.__new__(BraTSDataset)
    ds.seed = 42
    class MockConfig:
        class dataset:
            train_split = 0.7
            val_split = 0.15
            test_split = 0.15
    ds.config = MockConfig()
    
    splits = ds.create_splits(proc_dir, splits_dir)
    
    # Verify no intersection
    train_set = set(splits["train"])
    val_set = set(splits["val"])
    test_set = set(splits["test"])
    
    assert len(train_set.intersection(val_set)) == 0
    assert len(train_set.intersection(test_set)) == 0
    assert len(val_set.intersection(test_set)) == 0

@pytest.fixture
def splitter():
    return FederatedDataSplitter(num_clients=3, distribution="iid", seed=42)

def test_iid_split(splitter):
    file_list = [f"file_{i}.npy" for i in range(90)]
    splits = splitter.split_iid(file_list)
    
    assert len(splits) == 3
    for k, v in splits.items():
        assert len(v) == 30
    
    # Check no overlap
    s0 = set(splits[0])
    s1 = set(splits[1])
    s2 = set(splits[2])
    assert len(s0.intersection(s1)) == 0

def test_non_iid_split(splitter):
    file_list = [f"file_{i}.npy" for i in range(100)]
    # Create 2 classes
    labels = [0]*50 + [1]*50
    splits = splitter.split_non_iid(file_list, labels, alpha=0.5)
    
    assert len(splits) == 3
    # Verify total distributed is 100
    total = sum([len(v) for v in splits.values()])
    assert total == 100

def test_split_by_hospital(splitter):
    file_list = [f"file_{i}.npy" for i in range(100)]
    fractions = [0.4, 0.3, 0.3]
    splits = splitter.split_by_hospital(file_list, fractions)
    
    assert len(splits[0]) == 40
    assert len(splits[1]) == 30
    assert len(splits[2]) == 30

def test_save_load_splits(splitter, temp_dataset_dir):
    file_list = [f"file_{i}.npy" for i in range(30)]
    splits = splitter.split_iid(file_list)
    
    splitter.save_splits(splits, temp_dataset_dir)
    loaded = splitter.load_splits(temp_dataset_dir)
    
    assert len(loaded) == 3
    assert set(splits[0]) == set(loaded[0])
