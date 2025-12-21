"""Training modules for WildRobot.

This package provides training utilities for the WildRobot environment:
- trainer_jit: Fully JIT-compiled AMP+PPO trainer (recommended, 1000x faster)
- ppo_core: Core PPO components (networks, GAE, losses)
- experiment_tracking: W&B integration for logging
"""

from playground_amp.training.ppo_core import (
    create_networks,
    compute_gae,
    compute_ppo_loss,
    sample_actions,
    compute_values,
)
from playground_amp.training.trainer_jit import (
    AMPPPOConfigJit,
    train_amp_ppo_jit,
    TrainingState,
    IterationMetrics,
)

__all__ = [
    # JIT trainer (recommended)
    "AMPPPOConfigJit",
    "train_amp_ppo_jit",
    "TrainingState",
    "IterationMetrics",
    # PPO core
    "create_networks",
    "compute_gae",
    "compute_ppo_loss",
    "sample_actions",
    "compute_values",
]
