"""Training modules for WildRobot.

This package provides training utilities for the WildRobot environment:
- training_loop: Unified JIT-compiled trainer for PPO and AMP+PPO (v0.10+)
- ppo_core: Core PPO components (networks, GAE, losses)
- rollout: Unified rollout collector
- experiment_tracking: W&B integration for logging

Usage (v0.10+):
    from playground_amp.training import train
    train(env_step_fn, env_reset_fn, config, ref_motion_data)  # AMP mode
    train(env_step_fn, env_reset_fn, config, ref_motion_data=None)  # PPO-only
"""

from playground_amp.configs.training_config import TrainingConfig
from playground_amp.training.ppo_core import (
    compute_gae,
    compute_ppo_loss,
    compute_values,
    create_networks,
    init_network_params,
    sample_actions,
)
from playground_amp.training.rollout import (
    collect_rollout,
    compute_reward_total,
    TrajectoryBatch,
)
from playground_amp.training.training_loop import IterationMetrics, train, TrainingState

__all__ = [
    # Unified training (v0.10+)
    "train",
    "TrainingState",
    "IterationMetrics",
    "TrainingConfig",
    # Rollout
    "TrajectoryBatch",
    "collect_rollout",
    "compute_reward_total",
    # PPO core
    "compute_gae",
    "compute_ppo_loss",
    "compute_values",
    "create_networks",
    "init_network_params",
    "sample_actions",
]
