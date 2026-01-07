from __future__ import annotations

import pytest


def test_policy_init_action_bias_sets_mean_near_target() -> None:
    jnp = pytest.importorskip("jax.numpy")

    from training.algos.ppo.ppo_core import create_networks, init_network_params

    obs_dim = 36
    action_dim = 8
    ppo = create_networks(
        obs_dim=obs_dim,
        action_dim=action_dim,
        policy_hidden_dims=(16,),
        value_hidden_dims=(16,),
    )

    target_action = jnp.linspace(-0.5, 0.5, action_dim, dtype=jnp.float32)
    processor_params, policy_params, _ = init_network_params(
        ppo,
        obs_dim,
        action_dim,
        seed=0,
        policy_init_action=target_action,
        policy_init_std=0.05,
    )

    logits = ppo.policy_network.apply(processor_params, policy_params, jnp.zeros((obs_dim,)))
    loc, _scale_param = jnp.split(logits, 2, axis=-1)
    mean_action = jnp.tanh(loc)

    # Mean should match closely at init (weights are small; bias dominates).
    assert jnp.max(jnp.abs(mean_action - target_action)) < 1e-2

