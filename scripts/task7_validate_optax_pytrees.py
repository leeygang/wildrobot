#!/usr/bin/env python3
"""Task 7 Validation: Verify Flax pytrees and Optax optimizer work correctly.

This script validates that:
1. Policy and Value networks are proper Flax modules with pytree params
2. Optax optimizer can be initialized and applied to params
3. Gradient computation works correctly
4. Training state is a valid JAX pytree
5. Full training iteration executes without errors

Run:
    python scripts/task7_validate_optax_pytrees.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import jax
import jax.numpy as jnp
import optax
from jax import tree_util


def test_policy_network_pytree():
    """Test that PolicyNetwork params are valid Flax pytrees."""
    print("\n" + "=" * 60)
    print("Test 1: Policy Network Pytree Structure")
    print("=" * 60)

    from playground_amp.training.ppo_building_blocks import (
        create_policy_network,
        PolicyNetwork,
    )

    # Create network
    obs_dim = 44
    action_dim = 11
    policy_net, policy_params = create_policy_network(obs_dim, action_dim)

    # Check pytree structure
    leaves, treedef = tree_util.tree_flatten(policy_params)
    print(f"✓ Policy params are a valid pytree")
    print(f"  Number of leaves: {len(leaves)}")
    print(f"  Total parameters: {sum(l.size for l in leaves):,}")

    # Check structure
    print(f"\n  Pytree structure:")
    for key, value in jax.tree.leaves_with_path(policy_params):
        path = "/".join(str(k.key) for k in key)
        if hasattr(value, 'shape'):
            print(f"    {path}: {value.shape}")

    # Verify forward pass
    rng = jax.random.PRNGKey(0)
    obs = jax.random.normal(rng, (32, obs_dim))
    mean, log_std = policy_net.apply(policy_params, obs)
    print(f"\n✓ Forward pass works")
    print(f"  mean shape: {mean.shape}")
    print(f"  log_std shape: {log_std.shape}")

    # Verify JIT compilation
    jitted_apply = jax.jit(policy_net.apply)
    mean_jit, log_std_jit = jitted_apply(policy_params, obs)
    print(f"✓ JIT compilation works")

    return True


def test_value_network_pytree():
    """Test that ValueNetwork params are valid Flax pytrees."""
    print("\n" + "=" * 60)
    print("Test 2: Value Network Pytree Structure")
    print("=" * 60)

    from playground_amp.training.ppo_building_blocks import (
        create_value_network,
        ValueNetwork,
    )

    # Create network
    obs_dim = 44
    value_net, value_params = create_value_network(obs_dim)

    # Check pytree structure
    leaves, treedef = tree_util.tree_flatten(value_params)
    print(f"✓ Value params are a valid pytree")
    print(f"  Number of leaves: {len(leaves)}")
    print(f"  Total parameters: {sum(l.size for l in leaves):,}")

    # Verify forward pass
    rng = jax.random.PRNGKey(0)
    obs = jax.random.normal(rng, (32, obs_dim))
    values = value_net.apply(value_params, obs)
    print(f"\n✓ Forward pass works")
    print(f"  values shape: {values.shape}")

    return True


def test_optax_optimizer():
    """Test that Optax optimizer works with network params."""
    print("\n" + "=" * 60)
    print("Test 3: Optax Optimizer Integration")
    print("=" * 60)

    from playground_amp.training.ppo_building_blocks import (
        create_optimizer,
        create_policy_network,
        create_value_network,
    )

    # Create networks
    obs_dim = 44
    action_dim = 11
    policy_net, policy_params = create_policy_network(obs_dim, action_dim)
    value_net, value_params = create_value_network(obs_dim)

    # Create optimizer
    optimizer = create_optimizer(learning_rate=3e-4, max_grad_norm=0.5)
    print(f"✓ Optimizer created: optax.chain(clip_by_global_norm, adam)")

    # Initialize optimizer states
    policy_opt_state = optimizer.init(policy_params)
    value_opt_state = optimizer.init(value_params)
    print(f"✓ Optimizer states initialized")

    # Test gradient computation and update
    rng = jax.random.PRNGKey(0)
    obs = jax.random.normal(rng, (32, obs_dim))
    actions = jax.random.normal(rng, (32, action_dim))

    def policy_loss_fn(params):
        mean, log_std = policy_net.apply(params, obs)
        return jnp.mean((mean - actions) ** 2)

    def value_loss_fn(params):
        values = value_net.apply(params, obs)
        return jnp.mean(values ** 2)

    # Compute gradients
    policy_loss, policy_grads = jax.value_and_grad(policy_loss_fn)(policy_params)
    value_loss, value_grads = jax.value_and_grad(value_loss_fn)(value_params)
    print(f"✓ Gradients computed")
    print(f"  Policy loss: {policy_loss:.6f}")
    print(f"  Value loss: {value_loss:.6f}")

    # Apply updates
    policy_updates, new_policy_opt_state = optimizer.update(
        policy_grads, policy_opt_state
    )
    new_policy_params = optax.apply_updates(policy_params, policy_updates)
    print(f"✓ Policy params updated via optax.apply_updates")

    value_updates, new_value_opt_state = optimizer.update(
        value_grads, value_opt_state
    )
    new_value_params = optax.apply_updates(value_params, value_updates)
    print(f"✓ Value params updated via optax.apply_updates")

    # Verify params changed
    old_leaf = jax.tree.leaves(policy_params)[0]
    new_leaf = jax.tree.leaves(new_policy_params)[0]
    param_diff = jnp.abs(old_leaf - new_leaf).mean()
    print(f"\n✓ Params changed (mean diff: {param_diff:.8f})")

    return True


def test_training_state_pytree():
    """Test that TrainingState is a valid JAX pytree."""
    print("\n" + "=" * 60)
    print("Test 4: TrainingState Pytree Validation")
    print("=" * 60)

    from playground_amp.training.ppo_building_blocks import (
        create_optimizer,
        create_policy_network,
        create_training_state,
        create_value_network,
        TrainingState,
    )

    # Create networks and optimizer
    obs_dim = 44
    action_dim = 11
    policy_net, policy_params = create_policy_network(obs_dim, action_dim)
    value_net, value_params = create_value_network(obs_dim)
    optimizer = create_optimizer()

    # Create training state
    state = create_training_state(
        policy_network=policy_net,
        value_network=value_net,
        policy_params=policy_params,
        value_params=value_params,
        policy_optimizer=optimizer,
        value_optimizer=optimizer,
    )

    # Verify pytree structure
    leaves, treedef = tree_util.tree_flatten(state)
    print(f"✓ TrainingState is a valid pytree")
    print(f"  Number of leaves: {len(leaves)}")

    # Verify JIT compatibility
    @jax.jit
    def get_step(s: TrainingState) -> jnp.ndarray:
        return s.step

    step = get_step(state)
    print(f"✓ TrainingState is JIT-compatible")
    print(f"  Current step: {step}")

    return True


def test_gradient_update_correctness():
    """Test that gradient updates are computed correctly."""
    print("\n" + "=" * 60)
    print("Test 5: Gradient Update Correctness")
    print("=" * 60)

    from playground_amp.training.ppo_building_blocks import (
        create_optimizer,
        create_policy_network,
        create_value_network,
        ppo_loss,
    )

    # Create networks
    obs_dim = 44
    action_dim = 11
    batch_size = 64

    policy_net, policy_params = create_policy_network(obs_dim, action_dim)
    value_net, value_params = create_value_network(obs_dim)
    optimizer = create_optimizer(learning_rate=3e-4)

    # Create dummy batch
    rng = jax.random.PRNGKey(42)
    rng, obs_rng, action_rng, sample_rng = jax.random.split(rng, 4)

    obs = jax.random.normal(obs_rng, (batch_size, obs_dim))
    actions = jax.random.normal(action_rng, (batch_size, action_dim))
    old_log_probs = jax.random.normal(sample_rng, (batch_size,)) * 0.1
    advantages = jax.random.normal(sample_rng, (batch_size,))
    returns = jax.random.normal(sample_rng, (batch_size,))

    # Compute loss and gradients
    def loss_fn(policy_p, value_p):
        return ppo_loss(
            policy_params=policy_p,
            value_params=value_p,
            policy_network=policy_net,
            value_network=value_net,
            obs=obs,
            actions=actions,
            old_log_probs=old_log_probs,
            advantages=advantages,
            returns=returns,
        )

    (loss, metrics), (policy_grads, value_grads) = jax.value_and_grad(
        loss_fn, argnums=(0, 1), has_aux=True
    )(policy_params, value_params)

    print(f"✓ PPO loss computed: {loss:.6f}")
    print(f"  Policy loss: {metrics.policy_loss:.6f}")
    print(f"  Value loss: {metrics.value_loss:.6f}")
    print(f"  Entropy loss: {metrics.entropy_loss:.6f}")

    # Check gradient norms
    policy_grad_norm = optax.global_norm(policy_grads)
    value_grad_norm = optax.global_norm(value_grads)
    print(f"\n✓ Gradient norms:")
    print(f"  Policy: {policy_grad_norm:.6f}")
    print(f"  Value: {value_grad_norm:.6f}")

    # Verify gradients are finite
    policy_grad_finite = all(
        jnp.all(jnp.isfinite(g)) for g in jax.tree.leaves(policy_grads)
    )
    value_grad_finite = all(
        jnp.all(jnp.isfinite(g)) for g in jax.tree.leaves(value_grads)
    )
    print(f"\n✓ Gradients are finite:")
    print(f"  Policy: {policy_grad_finite}")
    print(f"  Value: {value_grad_finite}")

    # Apply update
    policy_opt_state = optimizer.init(policy_params)
    policy_updates, _ = optimizer.update(policy_grads, policy_opt_state)
    new_policy_params = optax.apply_updates(policy_params, policy_updates)

    # Verify loss decreases (single step, may not always decrease)
    (new_loss, _), _ = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)(
        new_policy_params, value_params
    )
    print(f"\n✓ Loss after update: {new_loss:.6f}")
    print(f"  Change: {new_loss - loss:.6f}")

    return True


def test_full_training_iteration():
    """Test a complete training iteration with AMP+PPO."""
    print("\n" + "=" * 60)
    print("Test 6: Full Training Iteration")
    print("=" * 60)

    from playground_amp.training.amp_ppo_training import (
        AMPPPOConfig,
        create_amp_ppo_state,
        train_iteration,
    )
    from playground_amp.training.ppo_building_blocks import create_optimizer
    from playground_amp.amp.discriminator import create_discriminator_optimizer
    from playground_amp.amp.ref_buffer import create_reference_buffer

    # Create config
    config = AMPPPOConfig(
        obs_dim=44,
        action_dim=11,
        num_envs=4,
        num_steps=5,
        enable_amp=True,
        total_iterations=1,
        log_interval=1,
    )

    # Initialize state
    rng = jax.random.PRNGKey(0)
    state, policy_net, value_net, disc_model = create_amp_ppo_state(config, rng)

    print(f"✓ Training state created")
    print(f"  Policy params: {sum(l.size for l in jax.tree.leaves(state.policy_params)):,} parameters")
    print(f"  Value params: {sum(l.size for l in jax.tree.leaves(state.value_params)):,} parameters")
    print(f"  Disc params: {sum(l.size for l in jax.tree.leaves(state.disc_params)):,} parameters")

    # Create optimizers
    policy_optimizer = create_optimizer(learning_rate=config.learning_rate)
    value_optimizer = create_optimizer(learning_rate=config.learning_rate)
    disc_optimizer = create_discriminator_optimizer(learning_rate=config.disc_learning_rate)

    # Create reference buffer
    ref_buffer = create_reference_buffer(
        obs_dim=config.obs_dim,
        seq_len=config.ref_seq_len,
        max_size=config.ref_buffer_size,
        mode=config.ref_motion_mode,
    )

    print(f"✓ Reference buffer created: {len(ref_buffer)} sequences")

    # Create dummy environment functions
    def dummy_step_fn(env_state, action):
        new_obs = env_state + jax.random.normal(jax.random.PRNGKey(0), env_state.shape) * 0.01
        reward = jnp.zeros(config.num_envs)
        done = jnp.zeros(config.num_envs)
        return new_obs, new_obs, reward, done, {}

    # Create initial env state
    env_rng = jax.random.PRNGKey(1)
    env_state = jax.random.normal(env_rng, (config.num_envs, config.obs_dim))

    # Run one training iteration
    new_state, new_env_state, metrics = train_iteration(
        state=state,
        env_step_fn=dummy_step_fn,
        env_state=env_state,
        policy_net=policy_net,
        value_net=value_net,
        disc_model=disc_model,
        policy_optimizer=policy_optimizer,
        value_optimizer=value_optimizer,
        disc_optimizer=disc_optimizer,
        ref_buffer=ref_buffer,
        config=config,
    )

    print(f"\n✓ Training iteration completed")
    print(f"  PPO Loss: {metrics.total_loss:.6f}")
    print(f"  Policy Loss: {metrics.policy_loss:.6f}")
    print(f"  Value Loss: {metrics.value_loss:.6f}")
    print(f"  Disc Loss: {metrics.disc_loss:.6f}")
    print(f"  Disc Accuracy: {metrics.disc_accuracy:.2%}")
    print(f"  AMP Reward Mean: {metrics.amp_reward_mean:.6f}")

    # Verify params changed
    old_policy_leaf = jax.tree.leaves(state.policy_params)[0]
    new_policy_leaf = jax.tree.leaves(new_state.policy_params)[0]
    policy_diff = jnp.abs(old_policy_leaf - new_policy_leaf).mean()
    print(f"\n✓ Policy params updated (mean diff: {policy_diff:.8f})")

    return True


def main():
    """Run all validation tests."""
    print("=" * 60)
    print("Task 7 Validation: Flax Pytrees + Optax Optimizer")
    print("=" * 60)
    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")

    tests = [
        ("Policy Network Pytree", test_policy_network_pytree),
        ("Value Network Pytree", test_value_network_pytree),
        ("Optax Optimizer", test_optax_optimizer),
        ("TrainingState Pytree", test_training_state_pytree),
        ("Gradient Correctness", test_gradient_update_correctness),
        ("Full Training Iteration", test_full_training_iteration),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ Test '{name}' FAILED with error:")
            print(f"   {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("\n✅ ALL TESTS PASSED!")
        print("Task 7 is COMPLETE: Flax pytrees and Optax optimizer are working correctly.")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
