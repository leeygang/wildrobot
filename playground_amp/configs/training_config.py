"""Centralized configuration management for WildRobot training.

This module provides utilities to load and manage training configuration
from training YAML files.

Design Principle: Single freezable config classes defined in training_runtime_config.py.
Call config.freeze() after CLI overrides to make configs JIT-compatible.

Usage:
    from playground_amp.configs.training_config import load_training_config
    from assets.robot_config import load_robot_config

    # Load robot config (path passed from train.py)
    robot_config = load_robot_config("assets/robot_config.yaml")

    # Load training config
    config = load_training_config("configs/ppo_walking.yaml")

    # Access config properties (new unified access pattern)
    print(config.networks.actor.hidden_sizes)  # (256, 256, 128)
    print(config.ppo.learning_rate)            # 3e-4
    print(config.amp.discriminator.r1_gamma)   # 10.0

    # Modify config before training
    config.ppo.learning_rate = 1e-4

    # Freeze for JIT (call once before training)
    config.freeze()
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# Re-export robot config utilities
from assets.robot_config import (  # noqa: F401
    clear_robot_config_cache,
    get_obs_indices,
    get_robot_config,
    load_robot_config,
    RobotConfig,
)

# Import all config classes from runtime config
from playground_amp.configs.training_runtime_config import (
    ActorNetworkConfig,
    AMPConfig,
    AMPFeatureConfig,
    AMPTargetsConfig,
    CheckpointConfig,
    CriticNetworkConfig,
    DiscriminatorNetworkConfig,
    DiscriminatorTrainingConfig,
    EnvConfig,
    Freezable,
    FrozenInstanceError,
    NetworksConfig,
    PPOConfig,
    RewardCompositionConfig,
    RewardWeightsConfig,
    TrainingConfig,
    VideoConfig,
    WandbConfig,
)


__all__ = [
    # Main config
    "TrainingConfig",
    "load_training_config",
    "get_training_config",
    "clear_config_cache",
    # Sub-configs
    "EnvConfig",
    "PPOConfig",
    "AMPConfig",
    "NetworksConfig",
    "ActorNetworkConfig",
    "CriticNetworkConfig",
    "DiscriminatorNetworkConfig",
    "RewardWeightsConfig",
    "RewardCompositionConfig",
    "CheckpointConfig",
    "WandbConfig",
    "VideoConfig",
    # Utils
    "Freezable",
    "FrozenInstanceError",
    # Robot config re-exports
    "RobotConfig",
    "load_robot_config",
    "get_robot_config",
]


def _parse_list_to_tuple(value: Any, default: Tuple) -> Tuple[int, ...]:
    """Convert list from YAML to tuple for hidden_sizes."""
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return default


def _parse_env_config(config: Dict[str, Any]) -> EnvConfig:
    """Parse environment configuration from YAML dict."""
    env = config.get("env", {})

    return EnvConfig(
        model_path=env.get("model_path", "assets/scene_flat_terrain.xml"),
        sim_dt=env.get("sim_dt", 0.002),
        ctrl_dt=env.get("ctrl_dt", 0.02),
        max_episode_steps=env.get("max_episode_steps", 500),
        target_height=env.get("target_height", 0.45),
        min_height=env.get("min_height", 0.20),
        max_height=env.get("max_height", 0.70),
        max_pitch=env.get("max_pitch", 0.8),
        max_roll=env.get("max_roll", 0.8),
        min_velocity=env.get("min_velocity", 0.0),
        max_velocity=env.get("max_velocity", 1.0),
        contact_threshold_force=env.get("contact_threshold_force", 5.0),
        contact_scale=env.get("contact_scale", 10.0),
        foot_switch_threshold=env.get("foot_switch_threshold", 2.0),
        # action_filter_alpha=0 disables filtering
        action_filter_alpha=env.get("action_filter_alpha", 0.7),
        push_enabled=env.get("push_enabled", False),
        push_start_step_min=env.get("push_start_step_min", 20),
        push_start_step_max=env.get("push_start_step_max", 200),
        push_duration_steps=env.get("push_duration_steps", 10),
        push_force_min=env.get("push_force_min", 0.0),
        push_force_max=env.get("push_force_max", 0.0),
        push_body=env.get("push_body", "waist"),
    )


def _parse_ppo_config(config: Dict[str, Any]) -> PPOConfig:
    """Parse PPO configuration from YAML dict."""
    ppo = config.get("ppo", {})
    return PPOConfig(
        num_envs=ppo.get("num_envs", 1024),
        rollout_steps=ppo.get("rollout_steps", 128),
        iterations=ppo.get("iterations", 1000),
        learning_rate=ppo.get("learning_rate", 3e-4),
        gamma=ppo.get("gamma", 0.99),
        gae_lambda=ppo.get("gae_lambda", 0.95),
        clip_epsilon=ppo.get("clip_epsilon", 0.2),
        entropy_coef=ppo.get("entropy_coef", 0.01),
        value_loss_coef=ppo.get("value_loss_coef", 0.5),
        epochs=ppo.get("epochs", 4),
        num_minibatches=ppo.get("num_minibatches", 32),
        max_grad_norm=ppo.get("max_grad_norm", 0.5),
        log_interval=ppo.get("log_interval", 10),
    )


def _parse_amp_config(config: Dict[str, Any]) -> AMPConfig:
    """Parse AMP configuration from YAML dict."""
    amp = config.get("amp", {})
    disc = amp.get("discriminator", {})
    feature = amp.get("feature_config", {})
    targets = amp.get("targets", {})

    return AMPConfig(
        enabled=amp.get("enabled", False),
        dataset_path=amp.get("dataset_path"),
        weight=amp.get("weight", 0.0),
        discriminator=DiscriminatorTrainingConfig(
            learning_rate=disc.get("learning_rate", 8e-5),
            batch_size=disc.get("batch_size", 256),
            updates_per_ppo_update=disc.get("updates_per_ppo_update", 2),
            r1_gamma=disc.get("r1_gamma", 10.0),
            input_noise_std=disc.get("input_noise_std", 0.03),
        ),
        feature_config=AMPFeatureConfig(
            use_finite_diff_vel=feature.get("use_finite_diff_vel", True),
            use_estimated_contacts=feature.get("use_estimated_contacts", True),
            mask_waist=feature.get("mask_waist", False),
            enable_mirror_augmentation=feature.get("enable_mirror_augmentation", False),
        ),
        targets=AMPTargetsConfig(
            disc_acc_min=targets.get("disc_acc_min", 0.55),
            disc_acc_max=targets.get("disc_acc_max", 0.80),
        ),
    )


def _parse_networks_config(config: Dict[str, Any]) -> NetworksConfig:
    """Parse networks configuration from YAML dict."""
    networks = config.get("networks", {})
    actor = networks.get("actor", {})
    critic = networks.get("critic", {})
    disc = networks.get("discriminator", {})

    return NetworksConfig(
        actor=ActorNetworkConfig(
            hidden_sizes=_parse_list_to_tuple(
                actor.get("hidden_sizes"), (256, 256, 128)
            ),
            activation=actor.get("activation", "elu"),
            log_std_init=actor.get("log_std_init", -1.0),
            min_log_std=actor.get("min_log_std", -5.0),
            max_log_std=actor.get("max_log_std", 2.0),
        ),
        critic=CriticNetworkConfig(
            hidden_sizes=_parse_list_to_tuple(
                critic.get("hidden_sizes"), (256, 256, 128)
            ),
            activation=critic.get("activation", "elu"),
        ),
        discriminator=DiscriminatorNetworkConfig(
            hidden_sizes=_parse_list_to_tuple(disc.get("hidden_sizes"), (512, 256)),
            activation=disc.get("activation", "relu"),
        ),
    )


def _parse_reward_weights_config(config: Dict[str, Any]) -> RewardWeightsConfig:
    """Parse reward weights from YAML dict."""
    # Support both 'rewards:' (old) and 'reward_weights:' (new) format
    rewards = config.get("reward_weights", config.get("rewards", {}))
    return RewardWeightsConfig(
        tracking_lin_vel=rewards.get("tracking_lin_vel", 2.0),
        lateral_velocity=rewards.get("lateral_velocity", -0.5),
        base_height=rewards.get("base_height", 0.5),
        orientation=rewards.get("orientation", -0.5),
        angular_velocity=rewards.get("angular_velocity", -0.05),
        height_target=rewards.get("height_target", 0.0),
        height_target_sigma=rewards.get("height_target_sigma", 0.05),
        torque=rewards.get("torque", -0.001),
        saturation=rewards.get("saturation", -0.1),
        action_rate=rewards.get("action_rate", -0.01),
        joint_velocity=rewards.get("joint_velocity", -0.001),
        slip=rewards.get("slip", -0.5),
        clearance=rewards.get("clearance", 0.1),
        gait_periodicity=rewards.get("gait_periodicity", 0.0),
        hip_swing=rewards.get("hip_swing", 0.0),
        knee_swing=rewards.get("knee_swing", 0.0),
        hip_swing_min=rewards.get("hip_swing_min", 0.0),
        knee_swing_min=rewards.get("knee_swing_min", 0.0),
        flight_phase_penalty=rewards.get("flight_phase_penalty", 0.0),
        stance_width_penalty=rewards.get("stance_width_penalty", 0.0),
        stance_width_target=rewards.get("stance_width_target", 0.10),
        stance_width_sigma=rewards.get("stance_width_sigma", 0.05),
        forward_velocity_scale=rewards.get("forward_velocity_scale", 4.0),
        velocity_standing_penalty=rewards.get("velocity_standing_penalty", 0.0),
        velocity_standing_threshold=rewards.get("velocity_standing_threshold", 0.2),
        velocity_cmd_min=rewards.get("velocity_cmd_min", 0.2),
    )


def _parse_reward_composition_config(config: Dict[str, Any]) -> RewardCompositionConfig:
    """Parse reward composition from YAML dict."""
    reward = config.get("reward", {})

    # Parse clip values (can be null or [min, max])
    task_clip = reward.get("task_reward_clip")
    amp_clip = reward.get("amp_reward_clip", [0.0, 1.0])

    return RewardCompositionConfig(
        task_weight=reward.get("task_weight", 1.0),
        amp_weight=reward.get("amp_weight", 0.0),
        task_reward_clip=tuple(task_clip) if task_clip else None,
        amp_reward_clip=tuple(amp_clip) if amp_clip else None,
    )


def _parse_checkpoint_config(config: Dict[str, Any]) -> CheckpointConfig:
    """Parse checkpoint configuration from YAML dict."""
    ckpt = config.get("checkpoints", {})
    return CheckpointConfig(
        dir=ckpt.get("dir", "playground_amp/checkpoints"),
        interval=ckpt.get("interval", 50),
    )


def _parse_wandb_config(config: Dict[str, Any]) -> WandbConfig:
    """Parse W&B configuration from YAML dict."""
    wandb = config.get("wandb", {})
    return WandbConfig(
        enabled=wandb.get("enabled", True),
        project=wandb.get("project", "wildrobot"),
        mode=wandb.get("mode", "online"),
        tags=wandb.get("tags", []),
        entity=wandb.get("entity"),
        name=wandb.get("name"),
        log_frequency=wandb.get("log_frequency", 10),
        log_dir=wandb.get("log_dir", "playground_amp/wandb"),
    )


def _parse_video_config(config: Dict[str, Any]) -> VideoConfig:
    """Parse video configuration from YAML dict."""
    video = config.get("video", {})
    return VideoConfig(
        enabled=video.get("enabled", False),
        num_videos=video.get("num_videos", 1),
        episode_length=video.get("episode_length", 500),
        render_every=video.get("render_every", 2),
        width=video.get("width", 640),
        height=video.get("height", 480),
        fps=video.get("fps", 25),
        upload_to_wandb=video.get("upload_to_wandb", True),
        output_subdir=video.get("output_subdir", "videos"),
    )


def load_training_config_from_dict(config: Dict[str, Any]) -> TrainingConfig:
    """Create TrainingConfig from a parsed YAML dict.

    Args:
        config: Parsed YAML configuration dictionary

    Returns:
        TrainingConfig instance (mutable, call freeze() when ready)
    """
    return TrainingConfig(
        # Version
        version=config.get("version", "0.11.1"),
        version_name=config.get("version_name", "Unified PPO + AMP"),
        # Global seed
        seed=config.get("seed", 42),
        # Composed configs
        env=_parse_env_config(config),
        ppo=_parse_ppo_config(config),
        amp=_parse_amp_config(config),
        networks=_parse_networks_config(config),
        reward_weights=_parse_reward_weights_config(config),
        reward=_parse_reward_composition_config(config),
        checkpoints=_parse_checkpoint_config(config),
        wandb=_parse_wandb_config(config),
        video=_parse_video_config(config),
        # Raw config for additional access
        raw_config=config,
    )


# Cached training config
_training_config: Optional[TrainingConfig] = None


def load_training_config(config_path: str | Path) -> TrainingConfig:
    """Load training configuration from YAML file.

    The returned config is mutable. After making any modifications,
    call config.freeze() to make it JIT-compatible.

    Args:
        config_path: Path to training config YAML (required)

    Returns:
        TrainingConfig instance (mutable)

    Example:
        config = load_training_config("configs/ppo_walking.yaml")
        config.ppo.learning_rate = 1e-4  # Override via CLI
        config.freeze()  # Freeze before training
    """
    global _training_config

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    _training_config = load_training_config_from_dict(config)
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
