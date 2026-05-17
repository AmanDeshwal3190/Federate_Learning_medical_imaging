import os
import json
import random
import numpy as np
from typing import Any, Dict

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across Python, NumPy, and TensorFlow."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass

def get_device() -> str:
    """Return the best available device string."""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            return f"GPU ({len(gpus)} device(s))"
        return "CPU"
    except ImportError:
        return "CPU"

def count_parameters(model) -> int:
    """Count the total number of trainable parameters in a Keras model."""
    return int(np.sum([np.prod(w.shape) for w in model.trainable_weights]))

def ensure_dir(path: str) -> str:
    """Create directory if it does not exist. Returns the path."""
    os.makedirs(path, exist_ok=True)
    return path

def save_json(data: Dict[str, Any], filepath: str) -> None:
    """Save dictionary to JSON file."""
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=str)

def load_json(filepath: str) -> Dict[str, Any]:
    """Load dictionary from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def format_metrics(metrics: Dict[str, float], precision: int = 4) -> str:
    """Format metrics dict into a readable string."""
    parts = [f"{k}: {v:.{precision}f}" for k, v in metrics.items()]
    return " | ".join(parts)
