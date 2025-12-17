"""Custom PPO+AMP training loop for WildRobot.

This module provides a complete training pipeline that integrates:
- PPO policy optimization (using Brax building blocks)
- AMP discriminator for natural motion learning
- Custom rollout collection with AMP features
- Separate discriminator training

Architecture:
- Uses Brax's battle-tested PPO loss functions (compute_gae, ppo_loss)
- Owns the outer training loop (not using ppo.train())
- Collects AMPTransition during rollouts
- Trains discriminator between rollout and PPO update
- Shapes rewards using discriminator output
"""

from playground_amp.training.trainer import AMPPPOConfig, train_amp_ppo
from playground_amp.training.ppo_core import (
    compute_gae,
    create_policy_network,
    create_value_network,
    ppo_loss,
)
from playground_amp.training.transitions import AMPTransition

__all__ = [
    "AMPTransition",
    "compute_gae",
    "ppo_loss",
    "create_policy_network",
    "create_value_network",
    "train_amp_ppo",
    "AMPPPOConfig",
]
