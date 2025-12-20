"""Centralized configuration management for WildRobot training.

This module provides utilities to load and manage:
1. Robot configuration (from robot_config.yaml) - robot-specific info
2. Training configuration (from training YAML files) - training parameters

All other modules should import configuration through this module instead of
hardcoding robot-specific values.

Usage:
    from playground_amp.configs.config import load_robot_config, load_training_config, RobotConfig

    # Load robot config (path passed from train_amp.py)
    robot_config = load_robot_config("assets/robot_config.yaml")

    # Access robot properties
    print(robot_config.action_dim)  # 9
    print(robot_config.actuator_names)  # ['waist_yaw', 'left_hip_pitch', ...]

    # Load training config
    training_config = load_training_config("configs/wildrobot_phase3_training.yaml")
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


@dataclass
class RobotConfig:
    """Robot configuration loaded from robot_config.yaml.

    This dataclass provides easy access to all robot-specific parameters
    extracted from the MuJoCo XML file.

    Attributes:
        robot_name: Name of the robot model
        actuator_names: List of actuator names in order
        actuator_joints: List of joint names corresponding to actuators
        action_dim: Number of actuators (action space dimension)
        observation_dim: Total observation dimension
        amp_feature_dim: AMP feature dimension
        floating_base_name: Name of the floating base joint
        floating_base_body: Name of the root body
        joint_names: List of all joint names
        joint_limits: Dict mapping joint name to (min, max) limits
        observation_indices: Dict mapping observation component to (start, end) indices
        sensors: Dict mapping sensor type to list of sensor info
    """

    robot_name: str
    actuator_names: List[str]
    actuator_joints: List[str]
    action_dim: int
    observation_dim: int
    amp_feature_dim: int
    floating_base_name: Optional[str]
    floating_base_body: Optional[str]
    floating_base_qpos_dim: int
    floating_base_qvel_dim: int
    joint_names: List[str]
    joint_limits: Dict[str, Tuple[float, float]]
    observation_indices: Dict[str, Dict[str, int]]
    observation_breakdown: Dict[str, int]
    amp_feature_breakdown: Dict[str, int]
    sensors: Dict[str, List[Dict[str, Any]]]
    raw_config: Dict[str, Any] = field(repr=False)

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "RobotConfig":
        """Load robot configuration from YAML file.

        Args:
            config_path: Path to robot_config.yaml

        Returns:
            RobotConfig instance
        """
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Extract actuator info
        actuators = config.get("actuators", {})
        actuator_names = actuators.get("names", [])
        actuator_joints = actuators.get("joints", [])
        action_dim = actuators.get("count", len(actuator_names))

        # Extract dimensions
        dims = config.get("dimensions", {})
        observation_dim = dims.get("observation_dim", 38)
        amp_feature_dim = dims.get("amp_feature_dim", 29)

        # Extract floating base info
        fb = config.get("floating_base", {}) or {}
        floating_base_name = fb.get("name")
        floating_base_body = fb.get("root_body")
        floating_base_qpos_dim = fb.get("qpos_dim", 7)
        floating_base_qvel_dim = fb.get("qvel_dim", 6)

        # Extract joint info
        joints = config.get("joints", {})
        joint_names = [j for j in joints.get("names", []) if j is not None]

        # Build joint limits dict
        joint_limits = {}
        for joint_detail in joints.get("details", []):
            name = joint_detail.get("name")
            if name and "range" in joint_detail:
                joint_limits[name] = tuple(joint_detail["range"])

        # Extract observation indices
        observation_indices = config.get("observation_indices", {})
        observation_breakdown = dims.get("observation_breakdown", {})
        amp_feature_breakdown = dims.get("amp_feature_breakdown", {})

        # Extract sensors
        sensors = config.get("sensors", {})

        return cls(
            robot_name=config.get("robot_name", "unknown"),
            actuator_names=actuator_names,
            actuator_joints=actuator_joints,
            action_dim=action_dim,
            observation_dim=observation_dim,
            amp_feature_dim=amp_feature_dim,
            floating_base_name=floating_base_name,
            floating_base_body=floating_base_body,
            floating_base_qpos_dim=floating_base_qpos_dim,
            floating_base_qvel_dim=floating_base_qvel_dim,
            joint_names=joint_names,
            joint_limits=joint_limits,
            observation_indices=observation_indices,
            observation_breakdown=observation_breakdown,
            amp_feature_breakdown=amp_feature_breakdown,
            sensors=sensors,
            raw_config=config,
        )

    def get_obs_slice(self, component: str) -> slice:
        """Get slice for extracting observation component.

        Args:
            component: Name of observation component (e.g., 'joint_positions')

        Returns:
            slice object for indexing into observation array
        """
        indices = self.observation_indices.get(component, {})
        return slice(indices.get("start", 0), indices.get("end", 0))

    def get_sensor_names(self, sensor_type: str) -> List[str]:
        """Get list of sensor names for a given type.

        Args:
            sensor_type: Sensor type (e.g., 'gyro', 'accelerometer')

        Returns:
            List of sensor names
        """
        return [s["name"] for s in self.sensors.get(sensor_type, [])]


@dataclass
class TrainingConfig:
    """Training configuration loaded from YAML file.

    This dataclass provides easy access to all training parameters.
    """

    # Environment
    ctrl_dt: float
    sim_dt: float
    target_height: float
    min_height: float
    max_height: float
    min_velocity: float
    max_velocity: float
    max_episode_steps: int
    use_action_filter: bool
    action_filter_alpha: float

    # Trainer
    num_envs: int
    rollout_steps: int
    iterations: int
    learning_rate: float
    gamma: float
    gae_lambda: float
    clip_epsilon: float
    entropy_coef: float
    value_loss_coef: float
    max_grad_norm: float
    num_minibatches: int
    update_epochs: int
    seed: int
    log_interval: int
    save_interval: int
    checkpoint_dir: str

    # Networks
    policy_hidden_dims: Tuple[int, ...]
    value_hidden_dims: Tuple[int, ...]
    log_std_min: float
    log_std_max: float

    # AMP
    amp_enabled: bool
    amp_weight: float
    disc_learning_rate: float
    disc_updates_per_iter: int
    disc_batch_size: int
    gradient_penalty_weight: float
    disc_hidden_dims: Tuple[int, ...]
    ref_buffer_size: int
    ref_seq_len: int
    ref_motion_mode: str
    ref_motion_path: Optional[str]
    normalize_amp_features: bool

    # Reward weights
    reward_weights: Dict[str, float]

    # Raw config for additional access
    raw_config: Dict[str, Any] = field(repr=False)

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "TrainingConfig":
        """Load training configuration from YAML file.

        Args:
            config_path: Path to training config YAML

        Returns:
            TrainingConfig instance
        """
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        env = config.get("env", {})
        trainer = config.get("trainer", {})
        networks = config.get("networks", {})
        amp = config.get("amp", {})
        rewards = config.get("reward_weights", {})

        return cls(
            # Environment
            ctrl_dt=env.get("ctrl_dt", 0.02),
            sim_dt=env.get("sim_dt", 0.002),
            target_height=env.get("target_height", 0.45),
            min_height=env.get("min_height", 0.2),
            max_height=env.get("max_height", 0.7),
            min_velocity=env.get("min_velocity", 0.5),
            max_velocity=env.get("max_velocity", 1.0),
            max_episode_steps=env.get("max_episode_steps", 500),
            use_action_filter=env.get("use_action_filter", True),
            action_filter_alpha=env.get("action_filter_alpha", 0.7),
            # Trainer
            num_envs=trainer.get("num_envs", 512),
            rollout_steps=trainer.get("rollout_steps", 10),
            iterations=trainer.get("iterations", 3000),
            learning_rate=trainer.get("lr", 3e-4),
            gamma=trainer.get("gamma", 0.99),
            gae_lambda=trainer.get("gae_lambda", 0.95),
            clip_epsilon=trainer.get("clip_epsilon", 0.2),
            entropy_coef=trainer.get("entropy_coef", 0.01),
            value_loss_coef=trainer.get("value_loss_coef", 0.5),
            max_grad_norm=trainer.get("max_grad_norm", 0.5),
            num_minibatches=trainer.get("num_minibatches", 4),
            update_epochs=trainer.get("epochs", 4),
            seed=trainer.get("seed", 42),
            log_interval=trainer.get("log_interval", 10),
            save_interval=trainer.get("save_interval", 100),
            checkpoint_dir=trainer.get("checkpoint_dir", "playground_amp/checkpoints"),
            # Networks
            policy_hidden_dims=tuple(networks.get("policy_hidden_dims", [512, 256, 128])),
            value_hidden_dims=tuple(networks.get("value_hidden_dims", [512, 256, 128])),
            log_std_min=networks.get("log_std_min", -20.0),
            log_std_max=networks.get("log_std_max", 2.0),
            # AMP
            amp_enabled=amp.get("enabled", True),
            amp_weight=amp.get("weight", 1.0),
            disc_learning_rate=amp.get("disc_lr", 1e-4),
            disc_updates_per_iter=amp.get("update_steps", 2),
            disc_batch_size=amp.get("batch_size", 512),
            gradient_penalty_weight=amp.get("gradient_penalty_weight", 5.0),
            disc_hidden_dims=tuple(amp.get("discriminator_hidden", [1024, 512, 256])),
            ref_buffer_size=amp.get("ref_buffer_size", 2000),
            ref_seq_len=amp.get("ref_seq_len", 32),
            ref_motion_mode=amp.get("ref_motion_mode", "file"),
            ref_motion_path=amp.get("dataset_path"),
            normalize_amp_features=amp.get("normalize_features", True),
            # Reward weights
            reward_weights=rewards,
            # Raw config
            raw_config=config,
        )


# Cached configs
_robot_config: Optional[RobotConfig] = None
_training_config: Optional[TrainingConfig] = None


def load_robot_config(config_path: str | Path) -> RobotConfig:
    """Load robot configuration.

    Caches the config after first load for efficiency.

    Args:
        config_path: Path to robot_config.yaml (required).

    Returns:
        RobotConfig instance
    """
    global _robot_config

    # Return cached if already loaded
    if _robot_config is not None:
        return _robot_config

    _robot_config = RobotConfig.from_yaml(config_path)
    return _robot_config


def load_training_config(config_path: str | Path) -> TrainingConfig:
    """Load training configuration.

    Args:
        config_path: Path to training config YAML (required).

    Returns:
        TrainingConfig instance
    """
    global _training_config

    _training_config = TrainingConfig.from_yaml(config_path)
    return _training_config


def get_robot_config() -> RobotConfig:
    """Get cached robot configuration.

    Raises:
        RuntimeError: If config hasn't been loaded yet
    """
    if _robot_config is None:
        raise RuntimeError("Robot config not loaded. Call load_robot_config(path) first.")
    return _robot_config


def get_training_config() -> TrainingConfig:
    """Get cached training configuration.

    Raises:
        RuntimeError: If config hasn't been loaded yet
    """
    if _training_config is None:
        raise RuntimeError("Training config not loaded. Call load_training_config(path) first.")
    return _training_config


def clear_config_cache() -> None:
    """Clear cached configurations."""
    global _robot_config, _training_config
    _robot_config = None
    _training_config = None


# Convenience function to get observation indices
def get_obs_indices(component: str) -> Tuple[int, int]:
    """Get (start, end) indices for an observation component.

    Args:
        component: Name of observation component

    Returns:
        Tuple of (start, end) indices
    """
    robot_cfg = get_robot_config()
    indices = robot_cfg.observation_indices.get(component, {})
    return indices.get("start", 0), indices.get("end", 0)
