import pytest
import os
import shutil
import numpy as np
from utils.common import set_seed, ensure_dir, save_json, load_json, format_metrics

def test_set_seed():
    set_seed(42)
    val1 = np.random.rand()
    set_seed(42)
    val2 = np.random.rand()
    assert val1 == val2

def test_ensure_dir(tmp_path):
    d = os.path.join(tmp_path, "test_dir")
    assert not os.path.exists(d)
    ensure_dir(d)
    assert os.path.exists(d)

def test_json_round_trip(tmp_path):
    filepath = os.path.join(tmp_path, "data.json")
    data = {"key1": "value1", "key2": 42}
    
    save_json(data, filepath)
    loaded = load_json(filepath)
    
    assert loaded == data

def test_format_metrics():
    metrics = {"loss": 0.123456, "accuracy": 0.987654}
    formatted = format_metrics(metrics, precision=3)
    assert "loss: 0.123" in formatted
    assert "accuracy: 0.988" in formatted
