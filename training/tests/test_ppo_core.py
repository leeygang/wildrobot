"""Tests for PPO building blocks using Brax networks."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from training.algos.ppo.ppo_core import (
    compute_gae,
    compute_ppo_loss,
    compute_values,
    create_networks,
    create_optimizer,
    init_network_params,
    make_inference_fn,
    sample_actions,
)


# Test dimensions
OBS_DIM = 44
ACTION_DIM = 11
BATCH_SIZE = 64
NUM_STEPS = 10
NUM_ENVS = 16
POLICY_HIDDEN_DIMS = (512, 256, 128)
VALUE_HIDDEN_DIMS = (512, 256, 128)


@pytest.fixture
def ppo_network():
    """Create PPO networks using Brax."""
    return create_networks(
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        policy_hidden_dims=POLICY_HIDDEN_DIMS,
        value_hidden_dims=VALUE_HIDDEN_DIMS,
    )


@pytest.fixture
def network_params(ppo_network):
    """Initialize network parameters."""
    processor_params, policy_params, value_params = init_network_params(
        ppo_network, OBS_DIM, ACTION_DIM, seed=0
    )
    return processor_params, policy_params, value_params


class TestNetworkCreation:
    """Tests for network creation and initialization."""

    def test_create_networks(self, ppo_network):
        """Test PPO networks are created successfully."""
        assert ppo_network is not None
        assert hasattr(ppo_network, "policy_network")
        assert hasattr(ppo_network, "value_network")
        assert hasattr(ppo_network, "parametric_action_distribution")

    def test_init_network_params(self, ppo_network):
        """Test network parameter initialization."""
        processor_params, policy_params, value_params = init_network_params(
            ppo_network, OBS_DIM, ACTION_DIM, seed=0
        )
        assert processor_params == ()
        assert policy_params is not None
        assert value_params is not None


class TestPolicySampling:
    """Tests for policy action sampling."""

    def test_sample_actions_shape(self, ppo_network, network_params):
        """Test policy sample returns correct shapes."""
        processor_params, policy_params, _ = network_params
        rng = jax.random.PRNGKey(0)
        obs = jax.random.normal(rng, (BATCH_SIZE, OBS_DIM))

        postprocessed_actions, raw_actions, log_probs = sample_actions(
            processor_params, policy_params, ppo_network, obs, rng
        )

        assert postprocessed_actions.shape == (BATCH_SIZE, ACTION_DIM)
        assert raw_actions.shape == (BATCH_SIZE, ACTION_DIM)
        assert log_probs.shape == (BATCH_SIZE,)

    def test_sample_actions_deterministic(self, ppo_network, network_params):
        """Test deterministic action sampling."""
        processor_params, policy_params, _ = network_params
        rng = jax.random.PRNGKey(0)
        obs = jax.random.normal(rng, (BATCH_SIZE, OBS_DIM))

        postprocessed_actions1, _, _ = sample_actions(
            processor_params, policy_params, ppo_network, obs, rng, deterministic=True
        )
        postprocessed_actions2, _, _ = sample_actions(
            processor_params, policy_params, ppo_network, obs, rng, deterministic=True
        )

        assert jnp.allclose(postprocessed_actions1, postprocessed_actions2)


class TestValueComputation:
    """Tests for value function computation."""

    def test_compute_values_shape(self, ppo_network, network_params):
        """Test value computation returns correct shape."""
        processor_params, _, value_params = network_params
        rng = jax.random.PRNGKey(0)
        obs = jax.random.normal(rng, (BATCH_SIZE, OBS_DIM))

        values = compute_values(processor_params, value_params, ppo_network, obs)

        assert values.shape == (BATCH_SIZE,)


class TestGAE:
    """Tests for Generalized Advantage Estimation."""

    def test_compute_gae_shape(self):
        """Test GAE computation returns correct shapes."""
        rng = jax.random.PRNGKey(0)
        rewards = jax.random.normal(rng, (NUM_STEPS, NUM_ENVS))
        step_values = jax.random.normal(rng, (NUM_STEPS, NUM_ENVS))
        dones = jnp.zeros((NUM_STEPS, NUM_ENVS))
        bootstrap = jax.random.normal(rng, (NUM_ENVS,))

        advantages, returns = compute_gae(rewards, step_values, dones, bootstrap)

        assert advantages.shape == (NUM_STEPS, NUM_ENVS)
        assert returns.shape == (NUM_STEPS, NUM_ENVS)

    def test_compute_gae_returns_equals_advantages_plus_values(self):
        """Test that returns = advantages + values."""
        rng = jax.random.PRNGKey(42)
        rewards = jax.random.normal(rng, (NUM_STEPS, NUM_ENVS))
        step_values = jax.random.normal(rng, (NUM_STEPS, NUM_ENVS))
        dones = jnp.zeros((NUM_STEPS, NUM_ENVS))
        bootstrap = jax.random.normal(rng, (NUM_ENVS,))

        advantages, returns = compute_gae(rewards, step_values, dones, bootstrap)

        assert jnp.allclose(returns, advantages + step_values)


class TestPPOLoss:
    """Tests for PPO loss computation."""

    def test_compute_ppo_loss(self, ppo_network, network_params):
        """Test PPO loss computation."""
        processor_params, policy_params, value_params = network_params
        rng = jax.random.PRNGKey(0)
        rng, loss_rng = jax.random.split(rng)

        obs_batch = jax.random.normal(rng, (BATCH_SIZE, OBS_DIM))
        action_batch = jax.random.normal(rng, (BATCH_SIZE, ACTION_DIM))
        log_prob_batch = jax.random.normal(rng, (BATCH_SIZE,))
        adv_batch = jax.random.normal(rng, (BATCH_SIZE,))
        ret_batch = jax.random.normal(rng, (BATCH_SIZE,))

        loss, metrics = compute_ppo_loss(
            processor_params,
            policy_params,
            value_params,
            ppo_network,
            obs_batch,
            action_batch,
            log_prob_batch,
            adv_batch,
            ret_batch,
            loss_rng,
        )

        assert loss.shape == ()
        assert jnp.isfinite(loss)
        assert metrics.policy_loss.shape == ()
        assert metrics.value_loss.shape == ()
        assert metrics.entropy_loss.shape == ()


class TestOptimizer:
    """Tests for optimizer creation."""

    def test_create_optimizer(self, network_params):
        """Test optimizer creation and initialization."""
        _, policy_params, _ = network_params
        optimizer = create_optimizer()
        opt_state = optimizer.init(policy_params)

        assert opt_state is not None


class TestInference:
    """Tests for inference function."""

    def test_make_inference_fn(self, ppo_network, network_params):
        """Test inference function creation."""
        processor_params, policy_params, value_params = network_params
        rng = jax.random.PRNGKey(0)

        inference_fn = make_inference_fn(ppo_network)
        params_tuple = (processor_params, policy_params, value_params)
        policy_fn = inference_fn(params_tuple, deterministic=True)

        test_obs = jax.random.normal(rng, (1, OBS_DIM))
        test_action, _ = policy_fn(test_obs, rng)

        assert test_action.shape == (1, ACTION_DIM)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
