"""Tests for the Alzheimer's data pipeline."""
import pytest
import numpy as np
import os
import sys
import tempfile
import json
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.alzheimer.preprocessing import AlzheimerPreprocessor
from data.alzheimer.dataset import AlzheimerDataset
from data.alzheimer.federated_splitter import AlzheimerFederatedSplitter

class ConfigMock:
    class Dataset:
        class ADNI:
             final_dimensions = [182, 218, 182]
        class OASIS:
             final_dimensions = [176, 208, 176]
        adni = ADNI()
        oasis = OASIS()
    class Preprocessing:
        target_voxel_size = [1.0, 1.0, 1.0]
        skull_stripping = True
        coregistration = False
        resampling = True
        normalization = "z_score"
    class Training:
        batch_size = 2
        cv_folds = 5
    dataset = Dataset()
    preprocessing = Preprocessing()
    training = Training()

@pytest.fixture
def preprocessor(monkeypatch):
    from utils.config_loader import ConfigLoader
    monkeypatch.setattr(ConfigLoader, "load", lambda x: ConfigMock())
    return AlzheimerPreprocessor()

@pytest.fixture
def dataset_mgr(monkeypatch):
    from utils.config_loader import ConfigLoader
    monkeypatch.setattr(ConfigLoader, "load", lambda x: ConfigMock())
    return AlzheimerDataset()

def test_load_nifti(preprocessor):
    import nibabel as nib
    import tempfile
    import os
    tmp = tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False)
    tmp.close()
    try:
        data = np.zeros((10, 10, 10))
        img = nib.Nifti1Image(data, np.eye(4))
        nib.save(img, tmp.name)
        vol, affine = preprocessor.load_nifti(tmp.name)
        assert vol.shape == (10, 10, 10)
        assert affine.shape == (4, 4)
    finally:
        os.remove(tmp.name)

def test_skull_strip(preprocessor):
    vol = np.zeros((20, 20, 20))
    vol[5:15, 5:15, 5:15] = 100  # "Brain"
    vol[0:3, 0:3, 0:3] = 10      # "Noise/Skull"
    
    stripped, mask = preprocessor.skull_strip(vol)
    
    assert np.all(stripped[0:3, 0:3, 0:3] == 0) # Skull should be removed
    assert mask[10, 10, 10] == True             # Brain should be kept

def test_normalize_zscore(preprocessor):
    vol = np.random.randn(20, 20, 20) * 5 + 10
    mask = np.ones((20, 20, 20), dtype=bool)
    
    normed = preprocessor.normalize_zscore(vol, mask)
    
    assert abs(np.mean(normed[mask])) < 1e-5
    assert abs(np.std(normed[mask]) - 1.0) < 1e-5

def test_resample_volume(preprocessor):
    vol = np.zeros((30, 30, 30))
    resampled = preprocessor.resample_volume(vol, (2.0, 2.0, 2.0), (1.0, 1.0, 1.0), (40, 40, 40))
    assert resampled.shape == (40, 40, 40)

def test_full_pipeline_shape(preprocessor, monkeypatch):
    import nibabel as nib
    def mock_load(path):
        return np.ones((50, 50, 50)), np.diag([2.0, 2.0, 2.0, 1.0])
    monkeypatch.setattr(preprocessor, "load_nifti", mock_load)
    
    res = preprocessor.preprocess_single_scan("fake.nii", "adni", None)
    assert res.shape == (182, 218, 182, 1)

def test_extract_subject_id(dataset_mgr):
    assert dataset_mgr.extract_subject_id("subject_001.npy") == "subject_001"
    assert dataset_mgr.extract_subject_id("sub-001_ses-01.npy") == "sub-001"
    assert dataset_mgr.extract_subject_id("OAS1_0001_MR1.npy") == "OAS1_0001"

def test_subject_level_cv(dataset_mgr):
    with tempfile.TemporaryDirectory() as tmpdir:
        files = [f"sub-{i}_ses-1.npy" for i in range(10)] + [f"sub-{i}_ses-2.npy" for i in range(10)]
        labels = [0]*10 + [1]*10 # Just dummy, wait: subject labels should be consistent
        # Make consistent labels for subs 0-9
        labels = [(0 if i < 5 else 1) for i in range(10)] * 2
        
        folds = dataset_mgr.create_subject_level_cv_splits(files, labels, tmpdir)
        
        assert len(folds) == 5
        for fold in folds:
            train_subs = set([dataset_mgr.extract_subject_id(f) for f in fold['train_files']])
            val_subs = set([dataset_mgr.extract_subject_id(f) for f in fold['val_files']])
            assert len(train_subs.intersection(val_subs)) == 0

def test_split_by_subject():
    splitter = AlzheimerFederatedSplitter(num_clients=2)
    files = [f"f{i}.npy" for i in range(4)]
    labels = [0, 0, 1, 1]
    subs = ["s1", "s1", "s2", "s3"]
    
    splits = splitter.split_by_subject(files, labels, subs)
    
    assert len(splits) == 2
    for client, data in splits.items():
        assert len(data['subjects']) == len(set(data['subjects']))

def test_split_by_institution():
    splitter = AlzheimerFederatedSplitter(num_clients=3)
    a_f = ["a1", "a2"]
    a_l = [0, 1]
    a_s = ["A1", "A2"]
    
    o_f = ["o1", "o2"]
    o_l = [1, 0]
    o_s = ["O1", "O2"]
    
    splits = splitter.split_by_institution(a_f, a_l, a_s, o_f, o_l, o_s)
    
    assert len(splits) == 3
    # Client 0 should have only A elements or mixed? Wait, C0 is A, C1 is O, C2 mixed.
    assert len(splits[0]['subjects']) > 0 or len(splits[2]['subjects']) > 0
