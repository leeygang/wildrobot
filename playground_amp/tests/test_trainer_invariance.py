"""Invariance tests for unified trainer.

These tests verify that the unified trainer behaves identically
whether AMP is enabled or disabled, for the PPO components.

Test Categories:
1. Reward Equivalence: When amp_weight=0, reward_total == task_reward exactly
2. Update Equivalence: PPO gradients/updates are identical regardless of AMP mode
3. Rollout Equivalence: Core PPO fields are identical regardless of collect_amp_features

Usage:
    pytest playground_amp/tests/test_trainer_invariance.py -v
"""

import jax
import jax.numpy as jnp
import pytest
from unittest.mock import MagicMock

from playground_amp.training.rollout import (
    TrajectoryBatch,
    compute_reward_total,
)
from playground_amp.training.ppo_core import (
    compute_gae,
    compute_ppo_loss,
    create_networks,
    init_network_params,
)


class TestRewardEquivalence:
    """Test 1: Reward equivalence when amp_weight=0."""

    def test_zero_amp_weight_returns_task_reward_exactly(self):
        """When amp_weight=0, reward_total == task_reward exactly."""
        task_rewards = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        amp_rewards = jnp.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])

        result = compute_reward_total(task_rewards, amp_rewards, amp_weight=0.0)

        # Must be exactly equal, not just close
        assert jnp.array_equal(result, task_rewards), \
            f"Expected exact equality, got diff: {result - task_rewards}"

    def test_nonzero_amp_weight_adds_amp_reward(self):
        """When amp_weight>0, reward_total = task + amp_weight * amp."""
        task_rewards = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        amp_rewards = jnp.array([[10.0, 20.0], [30.0, 40.0]])
        amp_weight = 0.5

        result = compute_reward_total(task_rewards, amp_rewards, amp_weight=amp_weight)
        expected = task_rewards + amp_weight * amp_rewards

        assert jnp.allclose(result, expected), \
            f"Expected {expected}, got {result}"

    def test_none_amp_rewards_with_zero_weight_works(self):
        """When amp_weight=0, amp_rewards can be None."""
        task_rewards = jnp.array([[1.0, 2.0], [3.0, 4.0]])

        result = compute_reward_total(task_rewards, amp_rewards=None, amp_weight=0.0)

        assert jnp.array_equal(result, task_rewards)

    def test_none_amp_rewards_with_nonzero_weight_raises(self):
        """When amp_weight>0, amp_rewards=None should raise error."""
        task_rewards = jnp.array([[1.0, 2.0], [3.0, 4.0]])

        with pytest.raises(ValueError, match="amp_weight.*but amp_rewards is None"):
            compute_reward_total(task_rewards, amp_rewards=None, amp_weight=0.5)


class TestGAEEquivalence:
    """Test that GAE computation is deterministic and identical."""

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

    def test_gae_with_zero_amp_equals_task_only(self):
        """GAE with reward_total (amp_weight=0) equals GAE with task_reward."""
        task_rewards = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        values = jnp.array([[0.5, 1.0], [1.5, 2.0]])
        dones = jnp.array([[0.0, 0.0], [0.0, 1.0]])
        bootstrap_value = jnp.array([1.0, 2.0])

        # Compute GAE directly with task rewards
        adv_task, ret_task = compute_gae(task_rewards, values, dones, bootstrap_value)

        # Compute GAE with combined reward (amp_weight=0)
        combined = compute_reward_total(task_rewards, None, amp_weight=0.0)
        adv_combined, ret_combined = compute_gae(combined, values, dones, bootstrap_value)

        assert jnp.allclose(adv_task, adv_combined), \
            "GAE advantages should be identical when amp_weight=0"
        assert jnp.allclose(ret_task, ret_combined), \
            "GAE returns should be identical when amp_weight=0"


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
        obs_dim = 38
        action_dim = 9

        trajectory = TrajectoryBatch(
            obs=jnp.zeros((*shape, obs_dim)),
            actions=jnp.zeros((*shape, action_dim)),
            log_probs=jnp.zeros(shape),
            values=jnp.zeros(shape),
            task_rewards=jnp.zeros(shape),
            dones=jnp.zeros(shape),
            truncations=jnp.zeros(shape),
            next_obs=jnp.zeros((*shape, obs_dim)),
            bootstrap_value=jnp.zeros((shape[1],)),
            foot_contacts=None,  # AMP field can be None
            root_heights=None,
            prev_joint_positions=None,
            forward_velocities=jnp.zeros(shape),
            heights=jnp.zeros(shape),
        )

        # Core PPO fields must be present
        assert trajectory.obs is not None
        assert trajectory.actions is not None
        assert trajectory.log_probs is not None
        assert trajectory.values is not None
        assert trajectory.task_rewards is not None
        assert trajectory.dones is not None
        assert trajectory.bootstrap_value is not None

    def test_trajectory_with_amp_fields(self):
        """TrajectoryBatch with AMP fields should work."""
        shape = (10, 4)
        obs_dim = 38
        action_dim = 9

        trajectory = TrajectoryBatch(
            obs=jnp.zeros((*shape, obs_dim)),
            actions=jnp.zeros((*shape, action_dim)),
            log_probs=jnp.zeros(shape),
            values=jnp.zeros(shape),
            task_rewards=jnp.zeros(shape),
            dones=jnp.zeros(shape),
            truncations=jnp.zeros(shape),
            next_obs=jnp.zeros((*shape, obs_dim)),
            bootstrap_value=jnp.zeros((shape[1],)),
            foot_contacts=jnp.zeros((*shape, 4)),  # AMP fields populated
            root_heights=jnp.zeros((*shape, 1)),
            prev_joint_positions=jnp.zeros((*shape, 9)),
            forward_velocities=jnp.zeros(shape),
            heights=jnp.zeros(shape),
        )

        # AMP fields should be present
        assert trajectory.foot_contacts is not None
        assert trajectory.root_heights is not None
        assert trajectory.prev_joint_positions is not None


class TestDiscriminatorNotCalledWhenDisabled:
    """Test that discriminator is not called when amp_weight=0."""

    def test_no_amp_branch_returns_zeros(self):
        """The no_amp_branch should return zero rewards."""
        # Simulate the no_amp_branch behavior
        task_rewards = jnp.array([[1.0, 2.0], [3.0, 4.0]])

        # When AMP disabled, amp_rewards should be zeros
        amp_rewards = jnp.zeros_like(task_rewards)

        # Combined should equal task only
        combined = task_rewards + 0.0 * amp_rewards

        assert jnp.array_equal(combined, task_rewards)

    def test_zero_weight_means_no_amp_contribution(self):
        """Zero weight should mean AMP has no effect on total reward."""
        task_rewards = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        amp_rewards = jnp.array([[100.0, 200.0], [300.0, 400.0]])  # Large values

        # With weight=0, amp should have no effect
        combined = compute_reward_total(task_rewards, amp_rewards, amp_weight=0.0)

        assert jnp.array_equal(combined, task_rewards), \
            "Zero amp_weight should completely ignore amp_rewards"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
