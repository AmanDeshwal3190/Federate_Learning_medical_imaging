import pytest
import os
from utils.config_loader import ConfigLoader

def test_load_config():
    # Load one of the configs
    config = ConfigLoader.load("config/brain_tumor_config.yaml")
    assert config is not None
    assert config.dataset.name == "BraTS2020_Figshare"
    
def test_dot_notation_access():
    config = ConfigLoader.load("config/brain_tumor_config.yaml")
    assert config.training.batch_size == 4
    assert config.evaluation.metrics[0] == "dice_score"

def test_merge_functionality():
    base = {"a": 1, "b": {"c": 2}}
    override = {"b": {"c": 3, "d": 4}, "e": 5}
    merged = ConfigLoader.merge(base, override)
    assert merged.a == 1
    assert merged.b.c == 3
    assert merged.b.d == 4
    assert merged.e == 5

def test_validation_required_keys():
    config = {"dataset": {"name": "Test", "path": "path/to/data"}}
    # Should not raise exception
    ConfigLoader.validate_keys(config, ["dataset.name", "dataset.path"])

def test_missing_key_errors():
    config = {"dataset": {"name": "Test"}}
    with pytest.raises(ValueError):
        ConfigLoader.validate_keys(config, ["dataset.name", "dataset.path"])
