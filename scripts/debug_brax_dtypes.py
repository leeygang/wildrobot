"""Debug script to understand Brax training wrapper dtype requirements.

This script creates a simple Brax environment and inspects how the training
wrappers modify State.info dtypes.
"""

import jax
import jax.numpy as jnp
from brax import envs
from brax.envs.base import State

# Load a standard Brax environment
env = envs.get_environment('ant')

# Reset and get initial state
rng = jax.random.PRNGKey(0)
state = env.reset(rng)

print("=" * 70)
print("Initial State (after reset)")
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
print("State After Step")
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
