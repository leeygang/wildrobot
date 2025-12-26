"""Runtime configuration for JIT-compiled training.

This module contains shared config dataclasses that are used by both
TrainingConfig (mutable, from YAML) and TrainingRuntimeConfig (frozen, for JIT).

Design Principle: Define fields in both Mutable*Config and *Config classes.
- Mutable versions are used in TrainingConfig for CLI overrides
- Frozen versions are used in TrainingRuntimeConfig for JIT compilation

Usage:
    from playground_amp.configs.training_config import load_training_config

    training_config = load_training_config("configs/ppo_amass_training.yaml")

    # Modify via mutable configs
    training_config.ppo.learning_rate = 1e-4

    # Convert to frozen for JIT
    runtime_config = training_config.to_runtime_config()
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Tuple


# =============================================================================
# Mutable Config Classes (for TrainingConfig - supports CLI overrides)
# =============================================================================


@dataclass
class MutablePPOConfig:
    """Mutable PPO hyperparameters for TrainingConfig."""

    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    num_minibatches: int = 4
    update_epochs: int = 4


@dataclass
class MutableAMPConfig:
    """Mutable AMP configuration for TrainingConfig."""

    # Reward weighting
    reward_weight: float = 0.5

    # Discriminator training
    disc_learning_rate: float = 1e-4
    disc_updates_per_iter: int = 2
    disc_batch_size: int = 256
    r1_gamma: float = 10.0
    disc_hidden_dims: Tuple[int, ...] = (512, 256)
    disc_input_noise_std: float = 0.0

    # v0.6.0: Policy Replay Buffer
    replay_buffer_size: int = 0
    replay_buffer_ratio: float = 0.5

    # v0.6.2: Golden Rule Configuration (Mathematical Parity)
    use_estimated_contacts: bool = True
    use_finite_diff_vel: bool = True
    contact_threshold_angle: float = 0.1
    contact_knee_scale: float = 0.5
    contact_min_confidence: float = 0.3

    # v0.8.0: Feature Dropping
    drop_contacts: bool = False
    drop_height: bool = False
    normalize_velocity: bool = False


@dataclass
class MutableNetworkConfig:
    """Mutable network architecture configuration for TrainingConfig."""

    policy_hidden_dims: Tuple[int, ...] = (512, 256, 128)
    value_hidden_dims: Tuple[int, ...] = (512, 256, 128)


@dataclass
class MutableTrainingLoopConfig:
    """Mutable training loop parameters for TrainingConfig."""

    num_envs: int = 512
    num_steps: int = 64  # rollout_steps
    total_iterations: int = 1000
    seed: int = 42
    log_interval: int = 10


# =============================================================================
# Frozen Config Classes (for TrainingRuntimeConfig - JIT compatible)
# =============================================================================


@dataclass(frozen=True)
class PPOConfig:
    """Frozen PPO hyperparameters for JIT compilation."""

    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    num_minibatches: int = 4
    update_epochs: int = 4


@dataclass(frozen=True)
class AMPConfig:
    """Frozen AMP configuration for JIT compilation."""

    # Reward weighting
    reward_weight: float = 0.5

    # Discriminator training
    disc_learning_rate: float = 1e-4
    disc_updates_per_iter: int = 2
    disc_batch_size: int = 256
    r1_gamma: float = 10.0
    disc_hidden_dims: Tuple[int, ...] = (512, 256)
    disc_input_noise_std: float = 0.0

    # v0.6.0: Policy Replay Buffer
    replay_buffer_size: int = 0
    replay_buffer_ratio: float = 0.5

    # v0.6.2: Golden Rule Configuration (Mathematical Parity)
    use_estimated_contacts: bool = True
    use_finite_diff_vel: bool = True
    contact_threshold_angle: float = 0.1
    contact_knee_scale: float = 0.5
    contact_min_confidence: float = 0.3

    # v0.8.0: Feature Dropping
    drop_contacts: bool = False
    drop_height: bool = False
    normalize_velocity: bool = False


@dataclass(frozen=True)
class NetworkConfig:
    """Frozen network architecture configuration for JIT compilation."""

    policy_hidden_dims: Tuple[int, ...] = (512, 256, 128)
    value_hidden_dims: Tuple[int, ...] = (512, 256, 128)


@dataclass(frozen=True)
class TrainingLoopConfig:
    """Frozen training loop parameters for JIT compilation."""

    num_envs: int = 512
    num_steps: int = 64  # rollout_steps
    total_iterations: int = 1000
    seed: int = 42
    log_interval: int = 10


# =============================================================================
# Freeze Helpers (convert mutable → frozen)
# =============================================================================


def freeze_ppo(mutable: MutablePPOConfig) -> PPOConfig:
    """Convert MutablePPOConfig to frozen PPOConfig."""
    return PPOConfig(**asdict(mutable))


def freeze_amp(mutable: MutableAMPConfig) -> AMPConfig:
    """Convert MutableAMPConfig to frozen AMPConfig."""
    return AMPConfig(**asdict(mutable))


def freeze_network(mutable: MutableNetworkConfig) -> NetworkConfig:
    """Convert MutableNetworkConfig to frozen NetworkConfig."""
    return NetworkConfig(**asdict(mutable))


def freeze_training_loop(mutable: MutableTrainingLoopConfig) -> TrainingLoopConfig:
    """Convert MutableTrainingLoopConfig to frozen TrainingLoopConfig."""
    return TrainingLoopConfig(**asdict(mutable))


# =============================================================================
# TrainingRuntimeConfig (composed from frozen configs)
# =============================================================================


@dataclass(frozen=True)
class TrainingRuntimeConfig:
    """Runtime configuration for JIT-compiled AMP+PPO training.

    This class composes frozen config objects for JIT compatibility.
    All values must be static (known at compile time) for JIT.

    NOTE: This class should only be created through TrainingConfig.to_runtime_config().

    Usage:
        config = training_cfg.to_runtime_config()

        # Access via composed configs
        config.ppo.learning_rate
        config.amp.reward_weight
        config.network.policy_hidden_dims
        config.training.num_envs
    """

    # Environment dimensions (from RobotConfig)
    obs_dim: int
    action_dim: int

    # Composed frozen configs
    ppo: PPOConfig
    amp: AMPConfig
    network: NetworkConfig
    training: TrainingLoopConfig
