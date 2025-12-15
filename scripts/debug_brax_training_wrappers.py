"""Debug script to understand Brax PPO training wrapper dtype requirements.

This script simulates what happens during Brax PPO training by wrapping
a standard environment with the same wrappers used by ppo.train().
"""

import jax
import jax.numpy as jnp
from brax import envs
from brax.training.agents.ppo import networks as ppo_networks
from brax.envs import wrappers

# Load a standard Brax environment
base_env = envs.get_environment('ant')
print(f"Base environment: {type(base_env)}")

# Apply the same wrappers that PPO training uses
# From brax/training/agents/ppo/train.py around line 580-600
env = wrappers.training.wrap(
    base_env,
    episode_length=1000,
    action_repeat=1,
)

print(f"Wrapped environment: {type(env)}")

# Reset and get initial state
rng = jax.random.PRNGKey(0)
# Use single RNG for non-batched environment
state = base_env.reset(rng)

# Now wrap the state as the training wrapper would
from brax.envs.wrappers.training import EpisodeWrapper, AutoResetWrapper, VmapWrapper
# Manually create what the wrappers create
print(f"\nChecking what training wrappers add to state.info...")

print("\nAttempting to wrap with single wrapper layers to see what each adds...")

print("\n" + "=" * 70)
print("Initial State (after reset with training wrappers)")
print("=" * 70)
print(f"obs shape: {state.obs.shape}, dtype: {state.obs.dtype}")
print(f"reward shape: {state.reward.shape}, dtype: {state.reward.dtype}")
print(f"done shape: {state.done.shape}, dtype: {state.done.dtype}")
print(f"\ninfo dict keys: {list(state.info.keys())}")
for key, value in state.info.items():
    if hasattr(value, 'shape') and hasattr(value, 'dtype'):
        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
    else:
        print(f"  {key}: {type(value)} = {value}")

# Take a step
action = jnp.zeros(env.action_size)
new_state = env.step(state, action)

print("\n" + "=" * 70)
print("State After Step (with training wrappers)")
print("=" * 70)
print(f"obs shape: {new_state.obs.shape}, dtype: {new_state.obs.dtype}")
print(f"reward shape: {new_state.reward.shape}, dtype: {new_state.reward.dtype}")
print(f"done shape: {new_state.done.shape}, dtype: {new_state.done.dtype}")
print(f"\ninfo dict keys: {list(new_state.info.keys())}")
for key, value in new_state.info.items():
    if hasattr(value, 'shape') and hasattr(value, 'dtype'):
        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
    else:
        print(f"  {key}: {type(value)} = {value}")

# Check if dtypes are consistent
print("\n" + "=" * 70)
print("Dtype Consistency Check")
print("=" * 70)
for key in state.info.keys():
    if key in new_state.info:
        old_val = state.info[key]
        new_val = new_state.info[key]
        if hasattr(old_val, 'dtype') and hasattr(new_val, 'dtype'):
            match = old_val.dtype == new_val.dtype
            status = "✓" if match else "✗"
            print(f"{status} {key}: {old_val.dtype} → {new_val.dtype}")
        else:
            print(f"  {key}: {type(old_val)} → {type(new_val)}")
