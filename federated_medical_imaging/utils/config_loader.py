import yaml
import os
from typing import Any, Dict, Optional

class DotDict(dict):
    """Dictionary with dot notation access."""
    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
            if isinstance(value, dict) and not isinstance(value, DotDict):
                value = DotDict(value)
                self[key] = value
            return value
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")
    
    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value
    
    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")


class ConfigLoader:
    """Loads and manages YAML configuration files with caching."""
    
    _cache: Dict[str, 'DotDict'] = {}
    
    @classmethod
    def load(cls, config_path: str) -> DotDict:
        """Load a YAML config file. Returns cached version if already loaded."""
        abs_path = os.path.abspath(config_path)
        if abs_path not in cls._cache:
            if not os.path.exists(abs_path):
                raise FileNotFoundError(f"Config file not found: {abs_path}")
            with open(abs_path, 'r') as f:
                raw = yaml.safe_load(f)
            cls._cache[abs_path] = DotDict(raw)
        return cls._cache[abs_path]
    
    @classmethod
    def clear_cache(cls) -> None:
        """Clear all cached configs."""
        cls._cache.clear()

    @staticmethod
    def merge(base: dict, override: dict) -> DotDict:
        """Deep merge two config dicts. Override values take precedence."""
        result = DotDict(base.copy())
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigLoader.merge(result[key], value)
            else:
                result[key] = value
        return result

    @staticmethod
    def validate_keys(config: dict, required_keys: list) -> None:
        """Validate that all required keys exist in config."""
        missing = []
        for key in required_keys:
            parts = key.split('.')
            current = config
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    missing.append(key)
                    break
        if missing:
            raise ValueError(f"Missing required config keys: {missing}")
