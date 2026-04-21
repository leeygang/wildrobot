"""Determinism tests for the PPO trainer building blocks.

Usage:
    pytest training/tests/test_trainer_invariance.py -v
"""

import jax
import jax.numpy as jnp
import pytest

from training.core.rollout import TrajectoryBatch
from training.core.metrics_registry import NUM_METRICS
from training.algos.ppo.ppo_core import (
    compute_gae,
    compute_ppo_loss,
    create_networks,
    init_network_params,
)


class TestGAEEquivalence:
    """Test that GAE computation is deterministic."""

    def test_gae_deterministic(self):
        """GAE should produce identical results for identical inputs."""
        rewards = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        values = jnp.array([[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]])
        dones = jnp.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
        bootstrap_value = jnp.array([1.0, 2.0, 3.0])

        adv1, ret1 = compute_gae(rewards, values, dones, bootstrap_value)
        adv2, ret2 = compute_gae(rewards, values, dones, bootstrap_value)

        assert jnp.allclose(adv1, adv2), "GAE advantages should be deterministic"
        assert jnp.allclose(ret1, ret2), "GAE returns should be deterministic"


class TestPPOLossEquivalence:
    """Test that PPO loss computation is deterministic."""

    @pytest.fixture
    def ppo_setup(self):
        """Create minimal PPO setup for testing."""
        obs_dim = 10
        action_dim = 4
        batch_size = 8

        # Create networks
        ppo_network = create_networks(
            obs_dim=obs_dim,
            action_dim=action_dim,
            policy_hidden_dims=(32,),
            value_hidden_dims=(32,),
        )

        # Initialize params
        processor_params, policy_params, value_params = init_network_params(
            ppo_network, obs_dim, action_dim, seed=42
        )

        # Create dummy data
        rng = jax.random.PRNGKey(0)
        obs = jax.random.normal(rng, (batch_size, obs_dim))
        actions = jax.random.normal(rng, (batch_size, action_dim))
        old_log_probs = jax.random.normal(rng, (batch_size,))
        advantages = jax.random.normal(rng, (batch_size,))
        returns = jax.random.normal(rng, (batch_size,))

        return {
            "ppo_network": ppo_network,
            "processor_params": processor_params,
            "policy_params": policy_params,
            "value_params": value_params,
            "obs": obs,
            "actions": actions,
            "old_log_probs": old_log_probs,
            "advantages": advantages,
            "returns": returns,
        }

    def test_ppo_loss_deterministic(self, ppo_setup):
        """PPO loss should produce identical results for identical inputs."""
        rng = jax.random.PRNGKey(123)

        loss1, metrics1 = compute_ppo_loss(
            processor_params=ppo_setup["processor_params"],
            policy_params=ppo_setup["policy_params"],
            value_params=ppo_setup["value_params"],
            ppo_network=ppo_setup["ppo_network"],
            obs=ppo_setup["obs"],
            value_obs=None,
            actions=ppo_setup["actions"],
            old_log_probs=ppo_setup["old_log_probs"],
            advantages=ppo_setup["advantages"],
            returns=ppo_setup["returns"],
            rng=rng,
        )

        loss2, metrics2 = compute_ppo_loss(
            processor_params=ppo_setup["processor_params"],
            policy_params=ppo_setup["policy_params"],
            value_params=ppo_setup["value_params"],
            ppo_network=ppo_setup["ppo_network"],
            obs=ppo_setup["obs"],
            value_obs=None,
            actions=ppo_setup["actions"],
            old_log_probs=ppo_setup["old_log_probs"],
            advantages=ppo_setup["advantages"],
            returns=ppo_setup["returns"],
            rng=rng,  # Same RNG
        )

        assert jnp.allclose(loss1, loss2), "PPO loss should be deterministic"
        assert jnp.allclose(metrics1.policy_loss, metrics2.policy_loss)
        assert jnp.allclose(metrics1.value_loss, metrics2.value_loss)


class TestRolloutFieldConsistency:
    """Test that rollout fields are consistent."""

    def test_trajectory_batch_has_all_ppo_fields(self):
        """TrajectoryBatch should always have PPO fields."""
        # Create a minimal trajectory
        shape = (10, 4)  # (num_steps, num_envs)
        obs_dim = 39
        action_dim = 8

        trajectory = TrajectoryBatch(
            obs=jnp.zeros((*shape, obs_dim)),
            critic_obs=jnp.zeros((*shape, obs_dim)),
            actions=jnp.zeros((*shape, action_dim)),
            log_probs=jnp.zeros(shape),
            values=jnp.zeros(shape),
            task_rewards=jnp.zeros(shape),
            dones=jnp.zeros(shape),
            truncations=jnp.zeros(shape),
            next_obs=jnp.zeros((*shape, obs_dim)),
            bootstrap_value=jnp.zeros((shape[1],)),
            metrics_vec=jnp.zeros((*shape, NUM_METRICS)),
            step_counts=jnp.zeros(shape),
        )

        # Core PPO fields must be present
        assert trajectory.obs is not None
        assert trajectory.actions is not None
        assert trajectory.log_probs is not None
        assert trajectory.values is not None
        assert trajectory.task_rewards is not None
        assert trajectory.dones is not None
        assert trajectory.bootstrap_value is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
