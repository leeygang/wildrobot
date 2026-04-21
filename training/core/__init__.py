"""Training core utilities for WildRobot.

This package provides training utilities for the WildRobot environment:
- training_loop: JIT-compiled PPO trainer
- ppo_core: Core PPO components (networks, GAE, losses)
- rollout: Rollout collector
- experiment_tracking: W&B integration for logging

Usage:
    from training.core import train
    train(env_step_fn, env_reset_fn, config)
"""

from training.configs.training_config import TrainingConfig
from training.algos.ppo.ppo_core import (
    compute_gae,
    compute_ppo_loss,
    compute_values,
    create_networks,
    init_network_params,
    sample_actions,
)
from training.core.rollout import (
    collect_rollout,
    TrajectoryBatch,
)
from training.core.training_loop import IterationMetrics, train, TrainingState

__all__ = [
    "train",
    "TrainingState",
    "IterationMetrics",
    "TrainingConfig",
    # Rollout
    "TrajectoryBatch",
    "collect_rollout",
    # PPO core
    "compute_gae",
    "compute_ppo_loss",
    "compute_values",
    "create_networks",
    "init_network_params",
    "sample_actions",
]
