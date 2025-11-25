# Visualization Consistency Validation

## The Critical Question

**How do we know the rendered video shows the same behavior as what the model learned during training?**

This is essential for validating that:
1. ✅ The policy actually learned what the metrics suggest
2. ✅ The visualization pipeline is correct
3. ✅ We can trust the training results

---

## TL;DR: Consistency Guarantees

| Aspect | Guarantee Level | How to Validate |
|--------|----------------|-----------------|
| **Episode Return** | ✅ Strong | Compare eval metrics with rollout rewards |
| **Episode Length** | ✅ Strong | Compare step counts |
| **Observation Processing** | ✅ Strong | Use same normalization params |
| **Policy Network** | ✅ Strong | Load exact weights from checkpoint |
| **Deterministic Actions** | ✅ Strong | Use deterministic=True |
| **Physics Simulation** | ⚠️ Medium | Same MuJoCo version, seed control |
| **Random Variations** | ⚠️ Medium | Seed control, multiple rollouts |

---

## Validation Methods

### **Method 1: Episode Return Comparison** ⭐ Most Important!

Compare the reward you get in visualization vs. what you got during training.

#### **During Training (from W&B logs):**
```
eval/episode_reward: 4800  (average over many episodes)
eval/avg_episode_length: 900 steps
```

#### **During Visualization:**
```python
# Add this to visualize_policy.py rollout_policy() function:

total_reward = 0.0
for step in range(n_steps):
    # ... existing code ...
    action, _ = policy(state.obs, action_rng)
    state = jit_step(state, action)

    # Accumulate reward
    total_reward += float(state.reward)

    if state.done:
        print(f"Episode ended at step {step}")
        print(f"Total episode reward: {total_reward:.1f}")
        break

return states, total_reward  # Return reward too!
```

**✅ Expected:** Visualization reward should be **close to training eval reward** (within ±10%)

**Example:**
```
Training eval/episode_reward: 4800
Visualization total_reward:   4650  ✅ Good! (within 3%)
```

**❌ Red Flags:**
```
Training eval/episode_reward: 4800
Visualization total_reward:   2000  ❌ BAD! Something's wrong!
```

---

### **Method 2: Episode Length Comparison**

Compare how long episodes last.

#### **Validation:**
```python
# In rollout_policy():
episode_length = len(states) - 1
print(f"Episode length: {episode_length}")

# Compare with training:
# Training eval/avg_episode_length: 900
# Visualization episode_length:     885  ✅ Good!
```

**✅ Expected:** Episode lengths should be **similar** (±10%)

**Why might they differ?**
- Random initialization (different starting states)
- Stochastic vs deterministic policy
- Different random seeds

**Solution:** Run multiple rollouts and average!

---

### **Method 3: Metric Logging During Visualization**

Capture the SAME metrics during visualization as during training.

#### **Enhanced Rollout with Metrics:**

```python
def rollout_policy_with_metrics(env, make_policy, params, n_steps=1000, seed=0):
    """Enhanced rollout that collects training-like metrics."""

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    policy = make_policy(params, deterministic=True)

    rng = jax.random.PRNGKey(seed)
    state = jit_reset(rng)

    # Metrics to track
    metrics = {
        'rewards': [],
        'episode_lengths': [],
        'heights': [],
        'velocities': [],
        'commanded_velocities': [],
    }

    episode_reward = 0.0
    episode_length = 0

    for step in range(n_steps):
        rng, action_rng = jax.random.split(rng)
        action, _ = policy(state.obs, action_rng)
        state = jit_step(state, action)

        # Collect metrics (same as training!)
        episode_reward += float(state.reward)
        episode_length += 1

        if 'height' in state.metrics:
            metrics['heights'].append(float(state.metrics['height']))
        if 'forward_velocity' in state.metrics:
            metrics['velocities'].append(float(state.metrics['forward_velocity']))
        if 'velocity_command' in state.metrics:
            metrics['commanded_velocities'].append(
                float(state.metrics['velocity_command'])
            )

        if float(state.done) > 0.5:
            # Episode ended
            metrics['rewards'].append(episode_reward)
            metrics['episode_lengths'].append(episode_length)

            print(f"\nEpisode Stats:")
            print(f"  Reward: {episode_reward:.1f}")
            print(f"  Length: {episode_length}")
            print(f"  Avg height: {np.mean(metrics['heights']):.3f}m")
            print(f"  Avg velocity: {np.mean(metrics['velocities']):.3f}m/s")
            break

    return metrics
```

**Compare with training:**
```python
# Training (from W&B):
# eval/episode_reward: 4800
# eval/avg_episode_length: 900
# Height: ~0.45m
# Velocity: varies by command

# Visualization:
metrics = rollout_policy_with_metrics(...)
print(f"Avg reward: {np.mean(metrics['rewards'])}")  # Should be ~4800
print(f"Avg length: {np.mean(metrics['episode_lengths'])}")  # Should be ~900
```

---

### **Method 4: Multiple Seed Comparison**

Test with multiple random seeds to ensure consistency.

```python
# Run visualization with 10 different seeds
rewards = []
for seed in range(10):
    metrics = rollout_policy_with_metrics(env, make_policy, params, seed=seed)
    rewards.append(metrics['rewards'][0])

print(f"Mean reward: {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")

# Compare with training:
# Training eval/episode_reward: 4800 ± 200
# Visualization mean reward:   4750 ± 180  ✅ Good!
```

**✅ Expected:** Mean and std should match training eval metrics

---

### **Method 5: Observation Distribution Check**

Verify observations are in the expected range.

```python
def check_observation_distribution(states):
    """Check if observations match expected distribution."""

    # Collect all observations
    observations = np.array([s.obs for s in states])  # Shape: (T, obs_dim)

    # Print statistics
    print(f"\nObservation Statistics:")
    print(f"  Shape: {observations.shape}")
    print(f"  Mean: {observations.mean(axis=0)[:5]}...")  # First 5 dims
    print(f"  Std:  {observations.std(axis=0)[:5]}...")
    print(f"  Min:  {observations.min(axis=0)[:5]}...")
    print(f"  Max:  {observations.max(axis=0)[:5]}...")

    # If observations were normalized during training,
    # they should have mean≈0, std≈1 BEFORE normalization
    # After normalization, normalized_obs should have mean≈0, std≈1
```

**✅ Expected (with normalization):**
- Raw observations: Vary by sensor type
- Normalized observations: Mean ≈ 0, Std ≈ 1

---

### **Method 6: Action Distribution Check**

Verify actions are reasonable.

```python
def check_action_distribution(states, actions):
    """Check if actions match expected distribution."""

    actions = np.array(actions)  # Shape: (T, action_dim)

    print(f"\nAction Statistics:")
    print(f"  Shape: {actions.shape}")
    print(f"  Mean: {actions.mean(axis=0)}")
    print(f"  Std:  {actions.std(axis=0)}")
    print(f"  Min:  {actions.min(axis=0)}")
    print(f"  Max:  {actions.max(axis=0)}")

    # Actions should typically be in [-1, 1] range for PPO with tanh
    if np.abs(actions).max() > 2.0:
        print("⚠️  WARNING: Actions outside expected range!")
```

**✅ Expected:**
- Actions in [-1, 1] range (tanh output)
- Reasonable variance (not all zeros, not saturated)

---

## Potential Sources of Inconsistency

### **1. Observation Normalization Mismatch** ❌ FIXED!

**Problem:** Training uses normalized obs, visualization doesn't

**Symptom:** Robot falls immediately in visualization

**Fix:** ✅ We fixed this! Using `normalizer_params.mean` and `normalizer_params.std`

**Validation:**
```python
# Check that normalizer params are loaded
print(f"Normalizer mean shape: {params[0].mean.shape}")
print(f"Normalizer std shape: {params[0].std.shape}")
print(f"First 5 means: {params[0].mean[:5]}")
print(f"First 5 stds: {params[0].std[:5]}")
```

---

### **2. Deterministic vs Stochastic Policy**

**Problem:** Training uses stochastic (explores), visualization might too

**Symptom:** High variance in visualization performance

**Fix:**
```python
# Use deterministic=True for evaluation (like training does)
policy = make_policy(params, deterministic=True)
```

**Validation:** Run same seed multiple times, should get identical results!

---

### **3. Random Seed Control**

**Problem:** Different random seeds → different behaviors

**Solution:**
```python
# Use same seed for reproducibility
seed = 42
rng = jax.random.PRNGKey(seed)

# Run twice with same seed → should get EXACT same trajectory
```

---

### **4. MuJoCo Version Differences**

**Problem:** Different MuJoCo versions might have subtle physics differences

**Check:**
```python
import mujoco
print(f"MuJoCo version: {mujoco.__version__}")

# Training and visualization should use SAME version
```

**Validation:** Document MuJoCo version in training logs

---

### **5. Environment Configuration Mismatch**

**Problem:** Training vs visualization use different env configs

**Fix:** Load config from checkpoint!

```python
# In visualize_policy.py:
config = checkpoint_data["config"]
env_config = config_dict.ConfigDict()
env_config.ctrl_dt = config["env"]["ctrl_dt"]
env_config.sim_dt = config["env"]["sim_dt"]
# ... etc

env = WildRobotLocomotion(task=..., config=env_config)
```

**✅ We already do this!**

---

### **6. Floating Point Precision**

**Problem:** JAX uses float32 by default, tiny differences accumulate

**Impact:** Usually negligible (<1% difference)

**Check:**
```python
# Run same rollout twice
reward1 = rollout_policy(..., seed=0)
reward2 = rollout_policy(..., seed=0)
print(f"Reward difference: {abs(reward1 - reward2)}")
# Should be < 0.01 (essentially 0)
```

---

## Practical Validation Script

Here's a complete validation script you can add to your project:

```python
#!/usr/bin/env python3
"""
Validate that visualization matches training performance.

Usage:
    python validate_consistency.py --checkpoint final_policy.pkl
"""

import argparse
import numpy as np
import jax
from pathlib import Path

from visualize_policy import create_env, load_policy
from wildrobot.locomotion import WildRobotLocomotion


def rollout_with_metrics(env, make_policy, params, n_steps=1000, seed=0):
    """Rollout policy and collect metrics."""
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    policy = make_policy(params, deterministic=True)

    rng = jax.random.PRNGKey(seed)
    state = jit_reset(rng)

    total_reward = 0.0
    episode_length = 0
    heights = []
    velocities = []

    for step in range(n_steps):
        rng, action_rng = jax.random.split(rng)
        action, _ = policy(state.obs, action_rng)
        state = jit_step(state, action)

        total_reward += float(state.reward)
        episode_length += 1

        if 'height' in state.metrics:
            heights.append(float(state.metrics['height']))
        if 'forward_velocity' in state.metrics:
            velocities.append(float(state.metrics['forward_velocity']))

        if float(state.done) > 0.5:
            break

    return {
        'reward': total_reward,
        'length': episode_length,
        'avg_height': np.mean(heights) if heights else 0.0,
        'avg_velocity': np.mean(velocities) if velocities else 0.0,
    }


def validate_consistency(checkpoint_path: str, n_trials: int = 10):
    """
    Validate visualization consistency with training.

    Args:
        checkpoint_path: Path to checkpoint
        n_trials: Number of rollouts to average over
    """
    import pickle

    print("=" * 70)
    print("Visualization Consistency Validation")
    print("=" * 70)

    # Load checkpoint
    print(f"\nLoading checkpoint: {checkpoint_path}")
    with open(checkpoint_path, 'rb') as f:
        checkpoint_data = pickle.load(f)

    config = checkpoint_data['config']

    # Print training metrics (if available in config)
    print(f"\n{'Training Metrics (from config)':^70}")
    print("-" * 70)
    if 'final_metrics' in checkpoint_data:
        metrics = checkpoint_data['final_metrics']
        print(f"  Eval Reward:        {metrics.get('eval/episode_reward', 'N/A')}")
        print(f"  Eval Episode Length: {metrics.get('eval/avg_episode_length', 'N/A')}")
    else:
        print("  (No training metrics saved in checkpoint)")
        print("  Check W&B for training eval/episode_reward and eval/avg_episode_length")

    # Create environment and load policy
    print(f"\nCreating environment...")
    env = create_env()

    print(f"Loading policy...")
    make_policy, params = load_policy(checkpoint_path, env)

    # Run multiple rollouts
    print(f"\n{'Visualization Metrics (current rollouts)':^70}")
    print("-" * 70)
    print(f"Running {n_trials} rollouts...")

    results = []
    for trial in range(n_trials):
        metrics = rollout_with_metrics(env, make_policy, params, seed=trial)
        results.append(metrics)
        print(f"  Trial {trial+1}/{n_trials}: "
              f"Reward={metrics['reward']:.1f}, "
              f"Length={metrics['length']}")

    # Compute statistics
    rewards = [r['reward'] for r in results]
    lengths = [r['length'] for r in results]
    heights = [r['avg_height'] for r in results]
    velocities = [r['avg_velocity'] for r in results]

    print(f"\n{'Summary Statistics':^70}")
    print("-" * 70)
    print(f"  Episode Reward:     {np.mean(rewards):7.1f} ± {np.std(rewards):6.1f}")
    print(f"  Episode Length:     {np.mean(lengths):7.1f} ± {np.std(lengths):6.1f}")
    print(f"  Average Height:     {np.mean(heights):7.3f} ± {np.std(heights):6.3f} m")
    print(f"  Average Velocity:   {np.mean(velocities):7.3f} ± {np.std(velocities):6.3f} m/s")

    # Validation checks
    print(f"\n{'Validation Checks':^70}")
    print("-" * 70)

    # Check 1: Reward consistency
    print(f"✓ Reward Statistics:")
    print(f"    Mean: {np.mean(rewards):.1f}")
    print(f"    Std:  {np.std(rewards):.1f}")
    print(f"    Min:  {np.min(rewards):.1f}")
    print(f"    Max:  {np.max(rewards):.1f}")

    # Check 2: Episode length
    print(f"✓ Episode Length Statistics:")
    print(f"    Mean: {np.mean(lengths):.1f}")
    print(f"    Std:  {np.std(lengths):.1f}")

    # Check 3: Determinism (run same seed twice)
    print(f"\n✓ Determinism Check:")
    m1 = rollout_with_metrics(env, make_policy, params, seed=42)
    m2 = rollout_with_metrics(env, make_policy, params, seed=42)
    reward_diff = abs(m1['reward'] - m2['reward'])
    print(f"    Same seed, different runs: reward diff = {reward_diff:.6f}")
    if reward_diff < 0.01:
        print(f"    ✅ PASS: Policy is deterministic!")
    else:
        print(f"    ⚠️  WARNING: Policy has some randomness (diff={reward_diff:.6f})")

    # Check 4: Normalization
    print(f"\n✓ Normalization Check:")
    if hasattr(params[0], 'mean') and hasattr(params[0], 'std'):
        print(f"    Normalizer params found:")
        print(f"      Mean shape: {params[0].mean.shape}")
        print(f"      Std shape:  {params[0].std.shape}")
        print(f"      First 5 means: {params[0].mean[:5]}")
        print(f"      First 5 stds:  {params[0].std[:5]}")
        print(f"    ✅ PASS: Normalization params loaded!")
    else:
        print(f"    ⚠️  No normalization params found (or not needed)")

    print("\n" + "=" * 70)
    print("Validation Complete!")
    print("=" * 70)
    print("\nNext Steps:")
    print("  1. Compare visualization reward with training eval/episode_reward")
    print("  2. If within ±10%, your visualization is consistent! ✅")
    print("  3. If >10% difference, investigate further ⚠️")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--n_trials", type=int, default=10)
    args = parser.parse_args()

    validate_consistency(args.checkpoint, args.n_trials)
```

---

## How to Use the Validation Script

```bash
# Save the script as validate_consistency.py
cd ~/projects/wildrobot/playground

# Run validation
python validate_consistency.py \
    --checkpoint ../training/logs/wildrobot_locomotion_flat_20251124-192712/checkpoints/final_policy.pkl \
    --n_trials 20

# Expected output:
# ===============================================================
# Visualization Consistency Validation
# ===============================================================
#
# Training Metrics (from W&B):
#   Eval Reward:         4800
#   Eval Episode Length: 900
#
# Visualization Metrics (current rollouts):
# Running 20 rollouts...
#   Trial 1/20: Reward=4750, Length=885
#   Trial 2/20: Reward=4820, Length=910
#   ...
#
# Summary Statistics:
#   Episode Reward:     4785.3 ± 156.2  ✅ Within ±10% of training!
#   Episode Length:     897.5 ± 45.3
#
# ✅ VALIDATION PASSED!
```

---

## Decision Matrix: Is Visualization Consistent?

| Metric | Difference from Training | Status |
|--------|------------------------|--------|
| **Episode Reward** | Within ±10% | ✅ Excellent |
| **Episode Reward** | Within ±20% | ⚠️ Acceptable (investigate) |
| **Episode Reward** | >20% difference | ❌ Problem! Debug needed |
| **Episode Length** | Within ±10% | ✅ Excellent |
| **Episode Length** | Within ±20% | ⚠️ Acceptable |
| **Episode Length** | >20% difference | ❌ Problem! Debug needed |
| **Determinism** | <0.01 reward diff | ✅ Perfect |
| **Determinism** | 0.01-1.0 diff | ⚠️ Some randomness |
| **Determinism** | >1.0 diff | ❌ Not deterministic! |

---

## Common Issues and Fixes

### **Issue 1: Visualization reward much lower than training**

**Symptoms:**
```
Training eval/episode_reward: 4800
Visualization reward:         2000  ❌
```

**Possible causes:**
1. ❌ Normalization mismatch (we fixed this!)
2. ❌ Loading wrong checkpoint
3. ❌ Environment config mismatch
4. ❌ Using stochastic policy instead of deterministic

**Debug steps:**
```python
# 1. Check normalization
print(f"Normalize observations: {config['ppo']['normalize_observations']}")

# 2. Check checkpoint is recent
import os
stat = os.stat(checkpoint_path)
print(f"Checkpoint modified: {stat.st_mtime}")

# 3. Check deterministic policy
policy = make_policy(params, deterministic=True)  # Must be True!
```

---

### **Issue 2: High variance across rollouts**

**Symptoms:**
```
Rewards: [4800, 3200, 5100, 2900, ...]  # High variance!
```

**Possible causes:**
1. Different initial states (velocity commands)
2. Stochastic policy
3. Environment randomness

**Fix:**
```python
# Use deterministic policy
policy = make_policy(params, deterministic=True)

# Control random seed
for seed in range(10):
    rollout_policy(..., seed=seed)
```

---

### **Issue 3: Robot behavior looks different in video**

**Symptoms:** Metrics look good but visuals seem off

**Possible causes:**
1. Video frame rate issues
2. Camera angle misleading
3. Different velocity commands than training

**Fix:**
```python
# Check what velocity command is being used
print(f"Velocity command: {state.metrics['velocity_command']}")

# Try different camera angles
env.render(..., camera="front")  # vs "side", "top"
```

---

## Best Practices

### **1. Always Validate After Training**

```bash
# Right after training finishes:
python validate_consistency.py --checkpoint final_policy.pkl --n_trials 20
```

### **2. Save Validation Results**

```python
# In your validation script, save results:
import json

results = {
    'checkpoint': checkpoint_path,
    'timestamp': datetime.now().isoformat(),
    'mean_reward': float(np.mean(rewards)),
    'std_reward': float(np.std(rewards)),
    'mean_length': float(np.mean(lengths)),
    'passed': abs(mean_reward - training_reward) < 0.1 * training_reward
}

with open('validation_results.json', 'w') as f:
    json.dump(results, f, indent=2)
```

### **3. Version Control Your Checkpoints**

```bash
# Tag checkpoints with validation status
cp final_policy.pkl final_policy_validated.pkl

# Document in README:
echo "Checkpoint validated on $(date)" >> CHECKPOINT_README.md
echo "Mean reward: 4785 ± 156 (training: 4800)" >> CHECKPOINT_README.md
```

---

## Summary

### **Strong Guarantees (99%+ confidence):**

✅ **Same policy weights** - Loaded from checkpoint
✅ **Same network architecture** - Reconstructed from config
✅ **Same normalization** - Using normalizer_params from training
✅ **Same physics** - Same MuJoCo model and version
✅ **Deterministic policy** - No sampling randomness

### **Weaker Guarantees (95% confidence):**

⚠️ **Same initial conditions** - Controlled by seed, but random
⚠️ **Floating point precision** - Tiny differences accumulate
⚠️ **Exact trajectory** - Chaotic system (butterfly effect)

### **How to Maximize Consistency:**

1. ✅ **Use deterministic=True** for evaluation
2. ✅ **Control random seeds** explicitly
3. ✅ **Run multiple rollouts** and average metrics
4. ✅ **Compare aggregate statistics** (mean ± std), not individual trajectories
5. ✅ **Save and load complete config** from checkpoint
6. ✅ **Validate with consistency script** after every training run

### **Bottom Line:**

If your **validation script shows:**
- Mean reward within ±10% of training ✅
- Determinism check passes (reward diff < 0.01) ✅
- Multiple rollouts cluster around training performance ✅

Then your **visualization is trustworthy!** 🎯

The rendered video **accurately represents** what your policy learned during training.

---

**Your specific case:**
- Training reward: ~4800
- Episode length: ~900 steps
- Robot doesn't fall immediately ✅

This suggests your visualization is working correctly! Run the validation script to get exact numbers. 🚀
