"""Runtime configuration for JIT-compiled training.

This module contains the TrainingRuntimeConfig class which holds all
compile-time constants needed for JAX JIT compilation.

TrainingRuntimeConfig is created ONLY through TrainingConfig.to_runtime_config()
to ensure consistency between training config and runtime config.

Usage:
    from playground_amp.configs.training_config import load_training_config

    training_config = load_training_config("configs/ppo_amass_training.yaml")
    runtime_config = training_config.to_runtime_config(
        obs_dim=env.observation_size,
        action_dim=env.action_size,
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class TrainingRuntimeConfig:
    """Runtime configuration for JIT-compiled AMP+PPO training.

    All values must be static (known at compile time) for JIT.

    NOTE: This class should only be created through TrainingConfig.to_runtime_config().
    Do not instantiate directly.
    """

    # Environment
    obs_dim: int
    action_dim: int
    num_envs: int
    num_steps: int  # Unroll length per iteration

    # PPO hyperparameters
    learning_rate: float
    gamma: float
    gae_lambda: float
    clip_epsilon: float
    value_loss_coef: float
    entropy_coef: float
    max_grad_norm: float

    # PPO training
    num_minibatches: int
    update_epochs: int

    # Network architecture
    policy_hidden_dims: Tuple[int, ...]
    value_hidden_dims: Tuple[int, ...]

    # AMP configuration
    amp_reward_weight: float
    disc_learning_rate: float
    disc_updates_per_iter: int
    disc_batch_size: int
    r1_gamma: float  # R1 regularizer weight
    disc_hidden_dims: Tuple[int, ...]
    disc_input_noise_std: float  # Gaussian noise std for discriminator inputs

    # Training
    total_iterations: int
    seed: int
    log_interval: int

    # v0.6.0: Policy Replay Buffer (defaults at end to satisfy dataclass rules)
    replay_buffer_size: int = 0  # 0 = disabled, >0 = enabled with this capacity
    replay_buffer_ratio: float = 0.5  # Ratio of historical samples (0.5 = 50% each)

    # v0.6.2: Golden Rule Configuration (Mathematical Parity)
    use_estimated_contacts: bool = True  # Use joint-based contact estimation
    use_finite_diff_vel: bool = True  # Use finite difference velocities
    contact_threshold_angle: float = (
        0.1  # Hip pitch threshold for contact detection (rad)
    )
    contact_knee_scale: float = 0.5  # Knee angle for confidence scaling (rad)
    contact_min_confidence: float = 0.3  # Min confidence when hip indicates contact

    # v0.8.0: Feature Dropping (prevent discriminator shortcuts)
    drop_contacts: bool = False  # Drop foot contacts (4 dims) from features
    drop_height: bool = False  # Drop root height (1 dim) from features
    normalize_velocity: bool = False  # Normalize root_linvel to unit direction
