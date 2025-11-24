"""Configuration utilities for loading and merging YAML configs with command-line args."""

from pathlib import Path
from typing import Any, Dict, Optional
import yaml


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file. If None, uses default.yaml.

    Returns:
        Dictionary containing configuration parameters.
    """
    if config_path is None:
        # Use default config in same directory as this file
        config_path = Path(__file__).parent / "default.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def override_config(config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """
    Override config values with command-line arguments.

    Args:
        config: Base configuration dictionary
        overrides: Dictionary of override values (nested keys use dot notation)

    Returns:
        Updated configuration dictionary

    Example:
        config = {"training": {"num_envs": 1024}}
        overrides = {"training.num_envs": 2048}
        result = override_config(config, overrides)
        # result["training"]["num_envs"] == 2048
    """
    import copy
    config_copy = copy.deepcopy(config)

    for key, value in overrides.items():
        if value is None:
            continue

        # Handle nested keys (e.g., "training.num_envs")
        if '.' in key:
            keys = key.split('.')
            current = config_copy
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = value
        else:
            config_copy[key] = value

    return config_copy
