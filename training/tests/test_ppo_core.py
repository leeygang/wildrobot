"""Tests for PPO building blocks using Brax networks."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
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
from training.core.training_loop import (
    _rsl_timeout_bootstrap_rewards,
    ppo_update_scan_toddlerbot,
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

    def test_toddlerbot_normal_actor_uses_global_log_std(self):
        network = create_networks(
            obs_dim=OBS_DIM,
            action_dim=ACTION_DIM,
            policy_hidden_dims=POLICY_HIDDEN_DIMS,
            value_hidden_dims=VALUE_HIDDEN_DIMS,
            activation="elu",
            distribution_type="normal",
            noise_std_type="log",
            init_noise_std=0.5,
            state_dependent_std=False,
        )
        processor, policy_params, _ = init_network_params(
            network,
            OBS_DIM,
            ACTION_DIM,
            seed=0,
            policy_init_action=jnp.zeros(ACTION_DIM),
            policy_init_std=0.5,
        )
        mean, std = network.policy_network.apply(
            processor, policy_params, jnp.zeros((3, OBS_DIM))
        )

        assert mean.shape == (3, ACTION_DIM)
        assert std.shape == (ACTION_DIM,)
        assert jnp.allclose(std, 0.5)


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

    def test_compute_values_shape_with_privileged_critic(self):
        """Test value computation with a separate privileged critic input."""
        critic_obs_dim = 12
        network = create_networks(
            obs_dim=OBS_DIM,
            action_dim=ACTION_DIM,
            policy_hidden_dims=POLICY_HIDDEN_DIMS,
            value_hidden_dims=VALUE_HIDDEN_DIMS,
            critic_obs_dim=critic_obs_dim,
        )
        processor_params, _, value_params = init_network_params(
            network, OBS_DIM, ACTION_DIM, seed=0
        )
        rng = jax.random.PRNGKey(0)
        obs = jax.random.normal(rng, (BATCH_SIZE, OBS_DIM))
        critic_obs = jax.random.normal(rng, (BATCH_SIZE, critic_obs_dim))

        values = compute_values(
            processor_params, value_params, network, obs, critic_obs
        )

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

    def test_rsl_timeout_bootstrap_matches_toddlerbot_runner(self):
        rewards = jnp.asarray([[1.0, 2.0]], dtype=jnp.float32)
        values = jnp.asarray([[3.0, 4.0]], dtype=jnp.float32)
        truncations = jnp.asarray([[1.0, 0.0]], dtype=jnp.float32)

        adjusted = _rsl_timeout_bootstrap_rewards(
            rewards, values, truncations, gamma=0.97
        )

        assert adjusted[0, 0] == pytest.approx(1.0 + 0.97 * 3.0)
        assert adjusted[0, 1] == pytest.approx(2.0)


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
            None,
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
        assert metrics.mirror_loss == pytest.approx(0.0)
        assert metrics.source_policy_kl == pytest.approx(0.0)

    def test_clipped_value_loss_matches_rsl_rl_formula(
        self, ppo_network, network_params
    ):
        processor_params, policy_params, value_params = network_params
        obs = jnp.zeros((BATCH_SIZE, OBS_DIM), dtype=jnp.float32)
        values = compute_values(processor_params, value_params, ppo_network, obs)
        returns = values + 1.0
        old_values = values - 1.0
        actions, raw_actions, old_log_probs = sample_actions(
            processor_params,
            policy_params,
            ppo_network,
            obs,
            jax.random.PRNGKey(17),
        )
        del actions

        _, metrics = compute_ppo_loss(
            processor_params=processor_params,
            policy_params=policy_params,
            value_params=value_params,
            ppo_network=ppo_network,
            obs=obs,
            value_obs=None,
            actions=raw_actions,
            old_log_probs=old_log_probs,
            old_values=old_values,
            advantages=jnp.zeros(BATCH_SIZE),
            returns=returns,
            rng=jax.random.PRNGKey(18),
            entropy_coef=0.0,
            use_clipped_value_loss=True,
            legacy_value_loss_half_factor=False,
        )

        # value_clipped = old + 0.2 = current - 0.8, so the clipped error is
        # 1.8 and dominates the unclipped current-vs-return error of 1.0.
        assert metrics.value_loss == pytest.approx(1.8**2, rel=1e-5)

    def test_compute_ppo_loss_adds_fixed_source_policy_kl(
        self, ppo_network, network_params
    ):
        processor_params, source_policy_params, value_params = network_params
        rng = jax.random.PRNGKey(11)
        obs_batch = jax.random.normal(rng, (BATCH_SIZE, OBS_DIM))
        action_batch = jax.random.normal(rng, (BATCH_SIZE, ACTION_DIM))
        log_prob_batch = jax.random.normal(rng, (BATCH_SIZE,))
        adv_batch = jax.random.normal(rng, (BATCH_SIZE,))
        ret_batch = jax.random.normal(rng, (BATCH_SIZE,))
        shifted_policy_params = jax.tree_util.tree_map(
            lambda value: value + jnp.asarray(1.0e-3, dtype=value.dtype),
            source_policy_params,
        )
        common = dict(
            processor_params=processor_params,
            policy_params=shifted_policy_params,
            value_params=value_params,
            ppo_network=ppo_network,
            obs=obs_batch,
            value_obs=None,
            actions=action_batch,
            old_log_probs=log_prob_batch,
            advantages=adv_batch,
            returns=ret_batch,
            rng=rng,
        )
        base_loss, _ = compute_ppo_loss(**common)
        anchored_loss, metrics = compute_ppo_loss(
            **common,
            source_policy_params=source_policy_params,
            source_policy_kl_coef=0.5,
        )

        assert metrics.source_policy_kl > 0.0
        assert anchored_loss == pytest.approx(
            float(base_loss + 0.5 * metrics.source_policy_kl), rel=1e-5
        )

    def test_compute_ppo_loss_adds_actor_mirror_regularizer(
        self, ppo_network, network_params
    ):
        processor_params, policy_params, value_params = network_params
        rng = jax.random.PRNGKey(7)
        obs_batch = jax.random.normal(rng, (BATCH_SIZE, OBS_DIM))
        action_batch = jax.random.normal(rng, (BATCH_SIZE, ACTION_DIM))
        log_prob_batch = jax.random.normal(rng, (BATCH_SIZE,))
        adv_batch = jax.random.normal(rng, (BATCH_SIZE,))
        ret_batch = jax.random.normal(rng, (BATCH_SIZE,))

        common = dict(
            processor_params=processor_params,
            policy_params=policy_params,
            value_params=value_params,
            ppo_network=ppo_network,
            obs=obs_batch,
            value_obs=None,
            actions=action_batch,
            old_log_probs=log_prob_batch,
            advantages=adv_batch,
            returns=ret_batch,
            rng=rng,
        )
        base_loss, _ = compute_ppo_loss(**common)
        weighted_loss, metrics = compute_ppo_loss(
            **common,
            mirror_loss_coef=0.1,
            mirror_observation_fn=lambda obs: -obs,
            mirror_action_fn=lambda action: -action,
        )

        assert metrics.mirror_loss > 0.0
        assert weighted_loss == pytest.approx(
            float(base_loss + 0.1 * metrics.mirror_loss), rel=1e-5
        )

    def test_compute_ppo_loss_requires_both_mirror_transforms(
        self, ppo_network, network_params
    ):
        processor_params, policy_params, value_params = network_params
        rng = jax.random.PRNGKey(9)
        obs_batch = jnp.zeros((BATCH_SIZE, OBS_DIM))
        with pytest.raises(ValueError, match="requires mirror observation/action"):
            compute_ppo_loss(
                processor_params=processor_params,
                policy_params=policy_params,
                value_params=value_params,
                ppo_network=ppo_network,
                obs=obs_batch,
                value_obs=None,
                actions=jnp.zeros((BATCH_SIZE, ACTION_DIM)),
                old_log_probs=jnp.zeros((BATCH_SIZE,)),
                advantages=jnp.zeros((BATCH_SIZE,)),
                returns=jnp.zeros((BATCH_SIZE,)),
                rng=rng,
                mirror_loss_coef=0.1,
            )


class TestToddlerBotUpdate:
    def test_executes_every_epoch_and_minibatch(self):
        obs_dim = 4
        action_dim = 2
        network = create_networks(
            obs_dim=obs_dim,
            action_dim=action_dim,
            policy_hidden_dims=(8,),
            value_hidden_dims=(8,),
            activation="elu",
            distribution_type="normal",
            noise_std_type="log",
            init_noise_std=0.5,
            state_dependent_std=False,
        )
        processor, policy_params, value_params = init_network_params(
            network, obs_dim, action_dim, seed=23
        )
        obs = jax.random.normal(jax.random.PRNGKey(24), (8, obs_dim))
        _, actions, old_log_probs = sample_actions(
            processor,
            policy_params,
            network,
            obs,
            jax.random.PRNGKey(25),
        )
        old_values = compute_values(processor, value_params, network, obs)
        advantages = jnp.linspace(-1.0, 1.0, 8, dtype=jnp.float32)
        returns = old_values + advantages
        policy_optimizer = optax.adam(1.0)
        value_optimizer = optax.adam(1.0)

        result = ppo_update_scan_toddlerbot(
            policy_params=policy_params,
            value_params=value_params,
            processor_params=processor,
            policy_opt_state=policy_optimizer.init(policy_params),
            value_opt_state=value_optimizer.init(value_params),
            ppo_network=network,
            policy_optimizer=policy_optimizer,
            value_optimizer=value_optimizer,
            obs=obs,
            critic_obs=None,
            actions=actions,
            old_log_probs=old_log_probs,
            old_values=old_values,
            advantages=advantages,
            returns=returns,
            rng=jax.random.PRNGKey(26),
            num_epochs=2,
            num_minibatches=2,
            clip_epsilon=0.2,
            value_loss_coef=0.25,
            entropy_coef=5.0e-4,
            max_grad_norm=1.0,
            learning_rate=3.0e-5,
            target_kl=0.01,
            adaptive_kl_factor=1.5,
            min_learning_rate=1.0e-5,
            max_learning_rate=1.0e-2,
        )

        assert float(result[12]) == pytest.approx(2.0)
        assert float(result[13]) == pytest.approx(4.0)
        assert 1.0e-5 <= float(result[15]) <= 1.0e-2


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
