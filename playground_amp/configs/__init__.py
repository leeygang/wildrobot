"""Configuration module for WildRobot training.

Exports:
    - RobotConfig: Robot configuration dataclass
    - TrainingConfig: Training configuration dataclass
    - load_robot_config: Load robot config from YAML
    - load_training_config: Load training config from YAML
    - get_robot_config: Get cached robot config
    - get_training_config: Get cached training config
"""

from playground_amp.configs.config import (
    RobotConfig,
    TrainingConfig,
    load_robot_config,
    load_training_config,
    get_robot_config,
    get_training_config,
    clear_config_cache,
    get_obs_indices,
)

__all__ = [
    "RobotConfig",
    "TrainingConfig",
    "load_robot_config",
    "load_training_config",
    "get_robot_config",
    "get_training_config",
    "clear_config_cache",
    "get_obs_indices",
]
