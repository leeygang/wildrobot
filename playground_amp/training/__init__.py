"""Custom PPO+AMP training loop for WildRobot.

This module provides a complete training pipeline that integrates:
- PPO policy optimization (using Brax's networks and our custom loop)
- AMP discriminator for natural motion learning
- Custom rollout collection with AMP features
- Separate discriminator training

Architecture:
- Uses Brax's battle-tested PPO networks (via make_ppo_networks)
- Custom GAE and PPO loss for AMP reward shaping
- Owns the outer training loop (not using ppo.train())
- Collects AMPTransition during rollouts
- Trains discriminator between rollout and PPO update
- Shapes rewards using discriminator output
"""

from playground_amp.training.ppo_core import (
    compute_gae,
    compute_ppo_loss,
    compute_values,
    create_networks,
    create_optimizer,
    init_network_params,
    make_inference_fn,
    make_ppo_networks,
    PPOLossMetrics,
    sample_actions,
    TrainingState,
)
from playground_amp.training.trainer import AMPPPOConfig, train_amp_ppo
from playground_amp.training.transitions import AMPTransition

__all__ = [
    # Trainer
    "AMPPPOConfig",
    "train_amp_ppo",
    # PPO core (using Brax)
    "compute_gae",
    "compute_ppo_loss",
    "create_networks",
    "create_optimizer",
    "init_network_params",
    "make_inference_fn",
    "make_ppo_networks",
    "PPOLossMetrics",
    "sample_actions",
    "compute_values",
    "TrainingState",
    # Transitions
    "AMPTransition",
]
