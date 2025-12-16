"""Test script for AMP components (discriminator + reference buffer).

This verifies that the AMP components work correctly before integration
into the full training pipeline.
"""

import jax
import jax.numpy as jnp
import numpy as np

print("=" * 70)
print("AMP Components Test")
print("=" * 70)

# Test 1: Reference Motion Buffer
print("\n[Test 1] Reference Motion Buffer")
print("-" * 70)

from playground_amp.amp.ref_buffer import create_reference_buffer

# Create buffer with synthetic walking motion
buffer = create_reference_buffer(
    obs_dim=44,
    seq_len=32,
    max_size=100,
    mode="walking",
    num_sequences=50
)

print(f"✓ Buffer created with {len(buffer)} sequences")

# Test sampling
single_frames = buffer.sample_single_frames(batch_size=10)
print(f"✓ Sampled {single_frames.shape[0]} single frames: shape {single_frames.shape}")

sequences = buffer.sample(batch_size=5)
print(f"✓ Sampled {sequences.shape[0]} sequences: shape {sequences.shape}")

# Test 2: Discriminator Creation
print("\n[Test 2] Discriminator Creation")
print("-" * 70)

from playground_amp.amp.discriminator import create_discriminator

model, params = create_discriminator(obs_dim=44, seed=0)
print(f"✓ Discriminator created")
print(f"  Model: {model}")
print(f"  Params keys: {list(params.keys())}")

# Test 3: Discriminator Forward Pass
print("\n[Test 3] Discriminator Forward Pass")
print("-" * 70)

# Create dummy observations
dummy_obs = jnp.zeros((8, 44), dtype=jnp.float32)
logits = model.apply(params, dummy_obs, training=False)
print(f"✓ Forward pass successful")
print(f"  Input shape: {dummy_obs.shape}")
print(f"  Output shape: {logits.shape}")
print(f"  Logits range: [{float(jnp.min(logits)):.4f}, {float(jnp.max(logits)):.4f}]")

# Test 4: Discriminator Loss
print("\n[Test 4] Discriminator Loss Computation")
print("-" * 70)

from playground_amp.amp.discriminator import discriminator_loss

# Sample real and fake observations
real_obs = buffer.sample_single_frames(batch_size=16)
fake_obs = np.random.randn(16, 44).astype(np.float32)  # Random fake data

real_obs_jax = jnp.asarray(real_obs)
fake_obs_jax = jnp.asarray(fake_obs)

rng = jax.random.PRNGKey(42)
loss, metrics = discriminator_loss(
    params, model, real_obs_jax, fake_obs_jax, rng,
    gradient_penalty_weight=5.0
)

print(f"✓ Loss computation successful")
print(f"  Total loss: {float(loss):.4f}")
print(f"  BCE loss: {float(metrics['discriminator_bce']):.4f}")
print(f"  Gradient penalty: {float(metrics['discriminator_gp']):.4f}")
print(f"  Accuracy: {float(metrics['discriminator_accuracy']):.4f}")
print(f"  Real mean prob: {float(metrics['discriminator_real_mean']):.4f}")
print(f"  Fake mean prob: {float(metrics['discriminator_fake_mean']):.4f}")

# Test 5: AMP Reward Computation
print("\n[Test 5] AMP Reward Computation")
print("-" * 70)

from playground_amp.amp.discriminator import compute_amp_reward

# Compute AMP rewards for fake observations
amp_rewards = compute_amp_reward(params, model, fake_obs_jax)
print(f"✓ AMP reward computation successful")
print(f"  Reward shape: {amp_rewards.shape}")
print(f"  Reward range: [{float(jnp.min(amp_rewards)):.4f}, {float(jnp.max(amp_rewards)):.4f}]")
print(f"  Reward mean: {float(jnp.mean(amp_rewards)):.4f}")

# Test 6: Discriminator Training Step
print("\n[Test 6] Discriminator Training Step")
print("-" * 70)

from playground_amp.amp.discriminator import create_discriminator_optimizer
import optax

optimizer = create_discriminator_optimizer(learning_rate=1e-4)
opt_state = optimizer.init(params)

# Training step
def train_step(params, opt_state, real_obs, fake_obs, rng_key):
    loss_fn = lambda p: discriminator_loss(p, model, real_obs, fake_obs, rng_key)
    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)

    return new_params, new_opt_state, loss, metrics

# Run a few training steps
print("Running 5 training iterations...")
for i in range(5):
    real_batch = buffer.sample_single_frames(batch_size=32)
    fake_batch = np.random.randn(32, 44).astype(np.float32)

    real_batch_jax = jnp.asarray(real_batch)
    fake_batch_jax = jnp.asarray(fake_batch)

    rng, step_rng = jax.random.split(rng)
    params, opt_state, loss, metrics = train_step(
        params, opt_state, real_batch_jax, fake_batch_jax, step_rng
    )

    print(f"  Step {i+1}: loss={float(loss):.4f}, acc={float(metrics['discriminator_accuracy']):.4f}")

print("✓ Training steps successful")

# Test 7: Wrapper Interface (Backward Compatibility)
print("\n[Test 7] Wrapper Interface (Backward Compatibility)")
print("-" * 70)

from playground_amp.amp.discriminator import AMPDiscriminatorWrapper

wrapper = AMPDiscriminatorWrapper(input_dim=44, seed=123)
print(f"✓ Wrapper created")

# Test score method
test_obs = np.random.randn(5, 44).astype(np.float32)
scores = wrapper.score(test_obs)
print(f"✓ Score method works: shape {scores.shape}, range [{scores.min():.4f}, {scores.max():.4f}]")

# Test update method
real_batch_np = buffer.sample_single_frames(batch_size=16)
fake_batch_np = np.random.randn(16, 44).astype(np.float32)
loss, metrics = wrapper.update(real_batch_np, fake_batch_np)
print(f"✓ Update method works: loss={float(loss):.4f}, acc={float(metrics['discriminator_accuracy']):.4f}")

# Summary
print("\n" + "=" * 70)
print("✅ All AMP Components Tests Passed!")
print("=" * 70)
print("\nComponents are ready for integration into train.py")
print("Next step: Update train.py to actually train the discriminator during PPO")
