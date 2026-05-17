"""
Pytest configuration and shared fixtures.
"""
import pytest
import numpy as np
import tensorflow as tf
import os
import sys
import tempfile

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@pytest.fixture(autouse=True)
def set_seeds():
    """Set random seeds for reproducibility in all tests."""
    np.random.seed(42)
    tf.random.set_seed(42)

@pytest.fixture
def tmp_dir():
    """Provide a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

@pytest.fixture
def sample_brain_tumor_volume():
    """Generate a sample brain tumor volume for testing."""
    volume = np.random.randn(32, 32, 32, 4).astype(np.float32)
    mask = np.random.randint(0, 4, (32, 32, 32)).astype(np.int32)
    return volume, mask

@pytest.fixture
def sample_alzheimer_volume():
    """Generate a sample Alzheimer's volume for testing."""
    volume = np.random.randn(32, 40, 32, 1).astype(np.float32)
    label = np.random.randint(0, 2)
    return volume, label

@pytest.fixture
def small_unet():
    """Build a small U-Net for fast testing."""
    from models.brain_tumor.unet3d import build_unet3d
    return build_unet3d(input_shape=(32, 32, 32, 4), num_classes=4,
                         num_filters=[8, 16, 32, 64, 128])

@pytest.fixture
def small_vgg3d():
    """Build a small VGG3D for fast testing."""
    from models.alzheimer.vgg3d import build_vgg3d_small
    return build_vgg3d_small(input_shape=(32, 40, 32, 1), num_classes=2)
