"""Configuration module for WildRobot training.

This module provides centralized configuration management.
"""

from training.configs.training_config import (
    clear_config_cache,
    get_obs_indices,
    get_robot_config,
    get_training_config,
    load_robot_config,
    load_training_config,
    RobotConfig,
    TrainingConfig,
    WandbConfig,
)

__all__ = [
    "RobotConfig",
    "TrainingConfig",
    "WandbConfig",
    "load_robot_config",
    "load_training_config",
    "get_robot_config",
    "get_training_config",
    "get_obs_indices",
    "clear_config_cache",
]
