"""Runtime configuration for JIT-compiled training.

This module provides a unified config system where:
1. All configs start as mutable dataclasses for easy CLI overrides
2. Call `freeze()` to create JIT-compatible frozen versions
3. Single class definition - no duplicate mutable/frozen classes

Usage:
    from training.configs.training_config import load_training_config

    config = load_training_config("configs/ppo_walking.yaml")

    # Modify via mutable configs
    config.ppo.learning_rate = 1e-4
    config.networks.actor.hidden_sizes = [512, 256]

    # Freeze for JIT (call once before training)
    config.freeze()

    # After freeze, modifications raise FrozenInstanceError
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple


class FrozenInstanceError(Exception):
    """Raised when attempting to modify a frozen config instance."""

    pass


class Freezable:
    """Mixin that allows dataclasses to be frozen after initialization.

    After calling freeze(), all attribute assignments raise FrozenInstanceError.
    This makes the config JIT-compatible while allowing initial mutations.
    """

    _frozen: bool = False

    def freeze(self) -> None:
        """Freeze this config and all nested Freezable configs.

        After freezing:
        - Attribute assignment raises FrozenInstanceError
        - The config becomes JIT-compatible
        """
        object.__setattr__(self, "_frozen", True)

        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, Freezable):
                value.freeze()

    def __setattr__(self, name: str, value) -> None:
        if getattr(self, "_frozen", False):
            raise FrozenInstanceError(
                f"Cannot modify '{name}' on frozen config. "
                "Call freeze() only after all modifications are complete."
            )
        object.__setattr__(self, name, value)

    @property
    def is_frozen(self) -> bool:
        """Check if this config is frozen."""
        return getattr(self, "_frozen", False)


# =============================================================================
# Environment Config
# =============================================================================
@dataclass
class EnvConfig(Freezable):
    """Environment configuration."""

    model_path: str = "assets/scene_flat_terrain.xml"

    # Timing
    sim_dt: float = 0.002
    ctrl_dt: float = 0.02

    # Episode
    max_episode_steps: int = 500

    # Health / termination
    target_height: float = 0.45
    min_height: float = 0.20
    max_height: float = 0.70
    max_pitch: float = 0.8
    max_roll: float = 0.8

    # Commands
    min_velocity: float = 0.0
    max_velocity: float = 1.0

    # Contacts
    contact_threshold_force: float = 5.0
    contact_scale: float = 10.0
    foot_switch_threshold: float = 2.0

    # Action filtering (alpha=0 disables filtering)
    action_filter_alpha: float = 0.7

    # Disturbance pushes (disabled by default)
    push_enabled: bool = False
    push_start_step_min: int = 20
    push_start_step_max: int = 200
    push_duration_steps: int = 10
    push_force_min: float = 0.0
    push_force_max: float = 0.0
    push_body: str = "waist"


# =============================================================================
# PPO Config
# =============================================================================
@dataclass
class PPOConfig(Freezable):
    """PPO algorithm configuration."""

    num_envs: int = 1024
    rollout_steps: int = 128
    iterations: int = 1000

    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95

    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5

    epochs: int = 4
    num_minibatches: int = 32
    max_grad_norm: float = 0.5

    log_interval: int = 10


# =============================================================================
# AMP Discriminator Training Config (NOT architecture)
# =============================================================================
@dataclass
class DiscriminatorTrainingConfig(Freezable):
    """Discriminator training hyperparameters (NOT architecture)."""

    learning_rate: float = 8e-5
    batch_size: int = 256
    updates_per_ppo_update: int = 2
    r1_gamma: float = 10.0
    input_noise_std: float = 0.03


@dataclass
class AMPFeatureConfig(Freezable):
    """AMP feature parity controls."""

    use_finite_diff_vel: bool = True
    use_estimated_contacts: bool = True
    mask_waist: bool = False
    enable_mirror_augmentation: bool = False


@dataclass
class AMPTargetsConfig(Freezable):
    """AMP diagnostic targets for alerts/logging."""

    disc_acc_min: float = 0.55
    disc_acc_max: float = 0.80


@dataclass
class AMPConfig(Freezable):
    """AMP (Adversarial Motion Prior) configuration."""

    enabled: bool = False
    dataset_path: Optional[str] = None
    weight: float = 0.0

    # Discriminator training config
    discriminator: DiscriminatorTrainingConfig = field(
        default_factory=DiscriminatorTrainingConfig
    )

    # Feature parity
    feature_config: AMPFeatureConfig = field(default_factory=AMPFeatureConfig)

    # Diagnostic targets
    targets: AMPTargetsConfig = field(default_factory=AMPTargetsConfig)


# =============================================================================
# Network Configs (Option A: algorithm-agnostic)
# =============================================================================
@dataclass
class ActorNetworkConfig(Freezable):
    """Actor (policy) network configuration."""

    hidden_sizes: Tuple[int, ...] = (256, 256, 128)
    activation: str = "elu"
    log_std_init: float = -1.0
    min_log_std: float = -5.0
    max_log_std: float = 2.0


@dataclass
class CriticNetworkConfig(Freezable):
    """Critic (value) network configuration."""

    hidden_sizes: Tuple[int, ...] = (256, 256, 128)
    activation: str = "elu"


@dataclass
class DiscriminatorNetworkConfig(Freezable):
    """Discriminator network configuration (architecture only)."""

    hidden_sizes: Tuple[int, ...] = (512, 256)
    activation: str = "relu"


@dataclass
class NetworksConfig(Freezable):
    """All network configurations."""

    actor: ActorNetworkConfig = field(default_factory=ActorNetworkConfig)
    critic: CriticNetworkConfig = field(default_factory=CriticNetworkConfig)
    discriminator: DiscriminatorNetworkConfig = field(
        default_factory=DiscriminatorNetworkConfig
    )


# =============================================================================
# Reward Configs
# =============================================================================
@dataclass
class RewardWeightsConfig(Freezable):
    """Task reward weights (environment-side)."""

    # Primary objectives
    tracking_lin_vel: float = 2.0
    lateral_velocity: float = -0.5
    base_height: float = 0.5
    orientation: float = -0.5
    angular_velocity: float = -0.05
    height_target: float = 0.0
    height_target_sigma: float = 0.05

    # Effort and safety
    torque: float = -0.001
    saturation: float = -0.1

    # Smoothness
    action_rate: float = -0.01
    joint_velocity: float = -0.001

    # Foot stability
    slip: float = -0.5
    clearance: float = 0.1
    gait_periodicity: float = 0.0

    # Gait shaping (v0.10.6)
    hip_swing: float = 0.0
    knee_swing: float = 0.0
    hip_swing_min: float = 0.0
    knee_swing_min: float = 0.0
    flight_phase_penalty: float = 0.0
    stance_width_penalty: float = 0.0
    stance_width_target: float = 0.10
    stance_width_sigma: float = 0.05

    # Shaping
    forward_velocity_scale: float = 4.0

    # v0.10.4: Standing penalty to discourage velocity=0
    velocity_standing_penalty: float = 0.0  # Penalty for standing still
    velocity_standing_threshold: float = 0.2  # Below this = standing still
    velocity_cmd_min: float = 0.2  # Only apply standing penalty if cmd > this


@dataclass
class RewardCompositionConfig(Freezable):
    """Reward composition (trainer-side)."""

    task_weight: float = 1.0
    amp_weight: float = 0.0
    task_reward_clip: Optional[Tuple[float, float]] = None
    amp_reward_clip: Optional[Tuple[float, float]] = (0.0, 1.0)


# =============================================================================
# Checkpoint Config
# =============================================================================
@dataclass
class CheckpointConfig(Freezable):
    """Checkpoint configuration."""

    dir: str = "training/checkpoints"
    interval: int = 50


# =============================================================================
# Logging Configs
# =============================================================================
@dataclass
class WandbConfig(Freezable):
    """W&B experiment tracking configuration."""

    enabled: bool = True
    project: str = "wildrobot"
    mode: str = "online"
    tags: List[str] = field(default_factory=list)
    entity: Optional[str] = None
    name: Optional[str] = None
    log_frequency: int = 10
    log_dir: str = "training/wandb"


@dataclass
class VideoConfig(Freezable):
    """Video generation configuration."""

    enabled: bool = False
    num_videos: int = 1
    episode_length: int = 500
    render_every: int = 2
    width: int = 640
    height: int = 480
    fps: int = 25
    upload_to_wandb: bool = True
    output_subdir: str = "videos"


# =============================================================================
# Main Training Config
# =============================================================================
@dataclass
class TrainingConfig(Freezable):
    """Main training configuration.

    Access pattern examples:
        config.networks.actor.hidden_sizes  # [256, 256, 128]
        config.ppo.learning_rate            # 3e-4
        config.amp.discriminator.r1_gamma   # 10.0
        config.reward_weights.tracking_lin_vel  # 2.0
    """

    # Version (from YAML config)
    version: str = ""
    version_name: str = ""

    # Global seed
    seed: int = 42

    # Composed configs
    env: EnvConfig = field(default_factory=EnvConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    amp: AMPConfig = field(default_factory=AMPConfig)
    networks: NetworksConfig = field(default_factory=NetworksConfig)
    reward_weights: RewardWeightsConfig = field(default_factory=RewardWeightsConfig)
    reward: RewardCompositionConfig = field(default_factory=RewardCompositionConfig)
    checkpoints: CheckpointConfig = field(default_factory=CheckpointConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    video: VideoConfig = field(default_factory=VideoConfig)

    # Raw config for additional access (not frozen)
    raw_config: Dict[str, Any] = field(default_factory=dict, repr=False)

    def apply_overrides(self, overrides: Dict[str, Any]) -> None:
        """Apply nested overrides from a dict (e.g., quick_verify section).

        The overrides dict follows the same schema as the main config.
        Example: {"ppo": {"num_envs": 4}, "wandb": {"enabled": false}}

        Call this before freeze() to apply overrides.
        """
        for key, value in overrides.items():
            if hasattr(self, key):
                target = getattr(self, key)
                if isinstance(value, dict) and isinstance(target, Freezable):
                    # Recursively apply nested overrides
                    self._apply_nested_overrides(target, value)
                else:
                    # Direct value assignment
                    setattr(self, key, value)

    def _apply_nested_overrides(
        self, target: Freezable, overrides: Dict[str, Any]
    ) -> None:
        """Recursively apply overrides to a nested Freezable config."""
        for key, value in overrides.items():
            if hasattr(target, key):
                current = getattr(target, key)
                if isinstance(value, dict) and isinstance(current, Freezable):
                    self._apply_nested_overrides(current, value)
                else:
                    setattr(target, key, value)
