"""Centralized configuration management for WildRobot training.

This module provides utilities to load and manage training configuration
from training YAML files.

Design Principle: Shared config classes (PPOConfig, AMPConfig, etc.) are defined
in training_runtime_config.py and composed into both TrainingConfig (mutable)
and TrainingRuntimeConfig (frozen for JIT).

Usage:
    from playground_amp.configs.training_config import load_training_config, TrainingConfig
    from playground_amp.configs.robot_config import load_robot_config, get_robot_config, RobotConfig

    # Load robot config (path passed from train.py)
    robot_config = load_robot_config("assets/robot_config.yaml")

    # Access robot properties
    print(robot_config.action_dim)  # 9
    print(robot_config.actuator_names)  # ['waist_yaw', 'left_hip_pitch', ...]

    # Load training config
    training_config = load_training_config("configs/wildrobot_phase3_training.yaml")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Re-export for backward compatibility

from playground_amp.configs.robot_config import (  # noqa: F401
    clear_robot_config_cache,
    get_obs_indices,
    get_robot_config,
    load_robot_config,
    RobotConfig,
)
from playground_amp.configs.training_runtime_config import (
    freeze_amp,
    freeze_network,
    freeze_ppo,
    freeze_training_loop,
    MutableAMPConfig,
    MutableNetworkConfig,
    MutablePPOConfig,
    MutableTrainingLoopConfig,
    TrainingRuntimeConfig,
)


@dataclass
class WandbConfig:
    """W&B experiment tracking configuration."""

    enabled: bool = True
    project: str = "wildrobot-locomotion"
    entity: Optional[str] = None
    name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    mode: str = "online"
    log_frequency: int = 10
    log_dir: str = "playground_amp/wandb"


@dataclass
class VideoConfig:
    """Video generation configuration for evaluation rollouts."""

    enabled: bool = True
    num_videos: int = 1
    episode_length: int = 500
    render_every: int = 2
    width: int = 640
    height: int = 480
    fps: int = 25
    upload_to_wandb: bool = True
    output_subdir: str = "videos"


@dataclass
class TrainingConfig:
    """Training configuration loaded from YAML file.

    Uses composed mutable config objects for easy CLI overrides.
    """

    # Version tracking (see CHANGELOG.md)
    version: str
    version_name: str

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

    # Composed mutable configs (supports direct modification)
    ppo: MutablePPOConfig
    amp: MutableAMPConfig
    network: MutableNetworkConfig
    training: MutableTrainingLoopConfig

    # Paths and other settings
    checkpoint_dir: str
    ref_motion_path: Optional[str]

    # v0.6.3: Feature Cleaning (removes discriminator cheats)
    velocity_filter_alpha: float  # EMA filter for velocity smoothing (0=off)
    ankle_offset: float  # Ankle pitch calibration offset (rad)

    # Reward weights
    reward_weights: Dict[str, float]

    # W&B configuration
    wandb: WandbConfig

    # Raw config for additional access
    raw_config: Dict[str, Any] = field(repr=False)

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "TrainingConfig":
        """Load training configuration from YAML file.

        Config format uses 'ppo:' section for PPO hyperparameters.

        Args:
            config_path: Path to training config YAML

        Returns:
            TrainingConfig instance
        """
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        env = config.get("env", {})
        networks = config.get("networks", {})
        amp_cfg = config.get("amp", {})
        ppo_cfg = config.get("ppo", {})

        # Support both 'rewards:' (Stage 1) and 'reward_weights:' (AMP) format
        rewards = config.get("rewards", config.get("reward_weights", {}))

        # Create composed mutable configs
        ppo = MutablePPOConfig(
            learning_rate=ppo_cfg.get("learning_rate", 3e-4),
            gamma=ppo_cfg.get("gamma", 0.99),
            gae_lambda=ppo_cfg.get("gae_lambda", 0.95),
            clip_epsilon=ppo_cfg.get("clip_epsilon", 0.2),
            value_loss_coef=ppo_cfg.get("value_loss_coef", 0.5),
            entropy_coef=ppo_cfg.get("entropy_coef", 0.01),
            max_grad_norm=ppo_cfg.get("max_grad_norm", 0.5),
            num_minibatches=ppo_cfg.get("num_minibatches", 4),
            update_epochs=ppo_cfg.get("epochs", 4),
        )

        amp = MutableAMPConfig(
            reward_weight=amp_cfg.get("weight", 1.0),
            disc_learning_rate=amp_cfg.get("disc_lr", 1e-4),
            disc_updates_per_iter=amp_cfg.get("update_steps", 2),
            disc_batch_size=amp_cfg.get("batch_size", 512),
            r1_gamma=amp_cfg.get("r1_gamma", 5.0),
            disc_hidden_dims=tuple(
                amp_cfg.get("discriminator_hidden", [1024, 512, 256])
            ),
            disc_input_noise_std=amp_cfg.get("disc_input_noise_std", 0.0),
            use_estimated_contacts=amp_cfg.get("use_estimated_contacts", True),
            use_finite_diff_vel=amp_cfg.get("use_finite_diff_vel", True),
            contact_threshold_angle=amp_cfg.get("contact_threshold_angle", 0.1),
            contact_knee_scale=amp_cfg.get("contact_knee_scale", 0.5),
            contact_min_confidence=amp_cfg.get("contact_min_confidence", 0.3),
            drop_contacts=amp_cfg.get("feature", {}).get("drop_contacts", False),
            drop_height=amp_cfg.get("feature", {}).get("drop_height", False),
            normalize_velocity=amp_cfg.get("feature", {}).get(
                "normalize_velocity", False
            ),
        )

        network = MutableNetworkConfig(
            policy_hidden_dims=tuple(
                networks.get("policy_hidden_dims", [512, 256, 128])
            ),
            value_hidden_dims=tuple(networks.get("value_hidden_dims", [512, 256, 128])),
        )

        training_loop = MutableTrainingLoopConfig(
            num_envs=ppo_cfg.get("num_envs", 512),
            num_steps=ppo_cfg.get("rollout_steps", 10),
            total_iterations=ppo_cfg.get("iterations", 3000),
            seed=ppo_cfg.get("seed", 42),
            log_interval=ppo_cfg.get("log_interval", 10),
        )

        return cls(
            # Version tracking
            version=config.get("version", "0.0.0"),
            version_name=config.get("version_name", "Unknown"),
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
            # Composed configs
            ppo=ppo,
            amp=amp,
            network=network,
            training=training_loop,
            # Paths
            checkpoint_dir=config.get("checkpoints", {}).get(
                "dir", "playground_amp/checkpoints"
            ),
            ref_motion_path=amp_cfg.get("dataset_path"),
            # v0.6.3: Feature Cleaning
            velocity_filter_alpha=amp_cfg.get("velocity_filter_alpha", 0.0),
            ankle_offset=amp_cfg.get("ankle_offset", 0.18),
            # Reward weights
            reward_weights=rewards,
            # W&B configuration
            wandb=cls._parse_wandb_config(config.get("wandb", {})),
            # Raw config
            raw_config=config,
        )

    @staticmethod
    def _parse_wandb_config(wandb_cfg: Dict[str, Any]) -> WandbConfig:
        """Parse W&B configuration from YAML dict."""
        return WandbConfig(
            enabled=wandb_cfg.get("enabled", True),
            project=wandb_cfg.get("project", "wildrobot-locomotion"),
            entity=wandb_cfg.get("entity"),
            name=wandb_cfg.get("name"),
            tags=wandb_cfg.get("tags", []),
            mode=wandb_cfg.get("mode", "online"),
            log_frequency=wandb_cfg.get("log_frequency", 10),
            log_dir=wandb_cfg.get("log_dir", "playground_amp/wandb"),
        )

    def to_runtime_config(self) -> TrainingRuntimeConfig:
        """Create TrainingRuntimeConfig for JIT-compiled training.

        Converts mutable composed configs to frozen versions for JIT.

        Returns:
            TrainingRuntimeConfig with all values needed for JIT compilation
        """
        # Get obs_dim and action_dim from RobotConfig
        robot_config = get_robot_config()

        return TrainingRuntimeConfig(
            obs_dim=robot_config.observation_dim,
            action_dim=robot_config.action_dim,
            ppo=freeze_ppo(self.ppo),
            amp=freeze_amp(self.amp),
            network=freeze_network(self.network),
            training=freeze_training_loop(self.training),
        )


# Cached training config
_training_config: Optional[TrainingConfig] = None


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


def get_training_config() -> TrainingConfig:
    """Get cached training configuration.

    Raises:
        RuntimeError: If config hasn't been loaded yet
    """
    if _training_config is None:
        raise RuntimeError(
            "Training config not loaded. Call load_training_config(path) first."
        )
    return _training_config


def clear_config_cache() -> None:
    """Clear cached configurations (both robot and training)."""
    global _training_config
    _training_config = None
    clear_robot_config_cache()
