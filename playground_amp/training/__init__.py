"""Training modules for WildRobot.

This package provides training utilities for the WildRobot environment:
- trainer_jit: Fully JIT-compiled AMP+PPO trainer (recommended, 1000x faster)
- ppo_core: Core PPO components (networks, GAE, losses)
- experiment_tracking: W&B integration for logging
"""

from playground_amp.training.trainer_jit import (
    IterationMetrics,
    TrainingState,
    train_amp_ppo_jit,
)
from playground_amp.training.ppo_core import (
    compute_gae,
    compute_ppo_loss,
    compute_values,
    create_networks,
    init_network_params,
    sample_actions,
)
from playground_amp.configs.training_runtime_config import TrainingRuntimeConfig

__all__ = [
    # Training
    "train_amp_ppo_jit",
    "TrainingState",
    "IterationMetrics",
    "TrainingRuntimeConfig",
    # PPO core
    "compute_gae",
    "compute_ppo_loss",
    "compute_values",
    "create_networks",
    "init_network_params",
    "sample_actions",
]
