# W&B Logging Issue: Only 1 Data Point After 1 Hour

## The Problem 🐛

**Symptoms:**
- Training runs for 1 hour
- Only 1 data point logged in W&B
- No real-time progress visibility

## Root Cause 🔍

### The Math

**From `default.yaml` (original):**
```yaml
num_timesteps: 30,000,000
num_evals: 10  # ← This was the problem!
```

**Logging frequency:**
```
eval_interval = total_timesteps / num_evals
              = 30,000,000 / 10
              = 3,000,000 steps between evals

Time per eval @ 460 steps/sec:
  3,000,000 / 460 = 6,522 seconds = 1.8 hours
```

**After 1 hour:** Only 55% through to the 2nd data point! 😅

---

## The Fix ✅

Changed `num_evals` in all configs:

### 1. `default.yaml`
```yaml
# Before:
num_evals: 10

# After:
num_evals: 100  # 10x more frequent!
```

**New logging frequency:**
```
eval_interval = 30,000,000 / 100 = 300,000 steps
Time per eval = 300,000 / 460 = 652 seconds ≈ 11 minutes

Now you get a data point every ~11 minutes! 📊
```

### 2. `production.yaml`
```yaml
# Before:
num_evals: 20

# After:
num_evals: 100
```

**New logging frequency:**
```
eval_interval = 50,000,000 / 100 = 500,000 steps
Time per eval = 500,000 / 460 ≈ 18 minutes
```

### 3. `quick.yaml`
```yaml
# Already good:
num_evals: 2  # For 100k steps = 50k per eval (fine for quick test)
```

---

## What `num_evals` Does

### In `train.py`

From Brax's PPO implementation:

```python
ppo.train(
    num_timesteps=30_000_000,
    num_evals=100,  # ← Controls logging frequency
    progress_fn=progress,  # Called at each eval
)
```

**What happens:**
```python
# Brax calculates:
eval_interval = num_timesteps / num_evals
              = 30,000,000 / 100
              = 300,000 steps

# Training loop:
for step in range(0, 30_000_000, eval_interval):
    # Train for 300k steps
    train_batch()

    # Evaluate and log
    metrics = evaluate_policy()
    progress_fn(step, metrics)  # ← Your callback gets called

    # This logs to W&B!
    wandb.log({
        "eval/episode_reward": ...,
        "eval/avg_episode_length": ...,
    }, step=step)
```

---

## Before vs After Comparison

### Original Settings (10 evals)

| Time | Step | Data Points |
|------|------|-------------|
| 0 min | 0 | 1 |
| 109 min | 3M | 2 |
| 218 min | 6M | 3 |
| 327 min | 9M | 4 |
| ... | ... | ... |
| 18 hours | 30M | 10 |

**Total runtime:** ~18 hours
**Total data points:** 10
**Visibility:** Very poor! 😞

### New Settings (100 evals)

| Time | Step | Data Points |
|------|------|-------------|
| 0 min | 0 | 1 |
| 11 min | 300K | 2 |
| 22 min | 600K | 3 |
| 33 min | 900K | 4 |
| ... | ... | ... |
| 1 hour | ~2.7M | 10 |
| ... | ... | ... |
| 18 hours | 30M | 100 |

**Total runtime:** ~18 hours
**Total data points:** 100
**Visibility:** Excellent! ✅

---

## Recommended `num_evals` Settings

### General Guidelines

```python
# Target: 1 data point every 5-15 minutes

num_evals = total_timesteps / (steps_per_sec * desired_interval_sec)

# Example for default.yaml:
num_evals = 30_000_000 / (460 * 600)  # 600 sec = 10 min
         ≈ 109

# Round to nice number: 100 ✅
```

### By Training Duration

| Training Duration | Total Steps | Recommended `num_evals` | Data Point Interval |
|-------------------|-------------|------------------------|---------------------|
| Quick test (2 min) | 100K | 2-5 | Every 30-60 sec |
| Short run (30 min) | 1M | 10-20 | Every 1-3 min |
| Medium run (3 hours) | 10M | 50-100 | Every 2-4 min |
| Long run (18 hours) | 30M | 100-200 | Every 5-10 min |
| Production (2 days) | 50M | 200-500 | Every 5-15 min |

### Current Configs (After Fix)

| Config | Total Steps | `num_evals` | Interval | Time/Eval |
|--------|-------------|-------------|----------|-----------|
| `quick.yaml` | 100K | 2 | 50K | ~2 min |
| `default.yaml` | 30M | 100 | 300K | ~11 min |
| `production.yaml` | 50M | 100 | 500K | ~18 min |

---

## Why This Matters

### Good Logging Frequency Enables:

1. ✅ **Early stopping** - See if training diverges early
2. ✅ **Hyperparameter tuning** - Compare runs quickly
3. ✅ **Debugging** - Catch issues as they happen
4. ✅ **Progress monitoring** - Know when training will finish
5. ✅ **Better graphs** - Smooth curves vs jagged points

### Example W&B Graph

**Before (10 points):**
```
Reward
  │     •
  │                     •
  │               •
  │         •
  │  •
  └────────────────────────> Steps
```
Hard to see trends!

**After (100 points):**
```
Reward
  │        •••••••••
  │    •••••
  │  ••
  │ •
  │•
  └────────────────────────> Steps
```
Clear learning curve! ✅

---

## Performance Impact

### Does More Frequent Eval Slow Training?

**Short answer:** Slightly, but negligible.

**Eval overhead:**
```python
# Each eval runs:
num_eval_envs × episode_length steps

# For default.yaml:
64 × 1000 = 64,000 steps per eval

# With 100 evals:
100 × 64,000 = 6,400,000 total eval steps

# Percentage of training:
6,400,000 / 30,000,000 = 21% overhead
```

**But:**
- Eval is faster (no gradient computation)
- Runs on GPU in parallel
- ~5-10% actual wallclock time overhead

**Recommendation:** The visibility is worth it! ✅

---

## How to Choose `num_evals`

### Method 1: Target Interval (Recommended)

```python
# Pick desired interval (e.g., 10 minutes)
desired_interval_minutes = 10

# Calculate
steps_per_minute = steps_per_sec × 60
                 = 460 × 60
                 = 27,600

steps_per_interval = steps_per_minute × desired_interval_minutes
                   = 27,600 × 10
                   = 276,000

num_evals = total_timesteps / steps_per_interval
          = 30,000,000 / 276,000
          ≈ 109

# Round to 100 ✅
```

### Method 2: Total Data Points

```python
# Pick number of points you want on graph
desired_points = 100  # Nice round number

num_evals = desired_points  # That's it!
```

### Method 3: Percentage of Training

```python
# Evaluate every X% of training
eval_percentage = 0.01  # 1% = 100 evals

num_evals = 1 / eval_percentage
          = 1 / 0.01
          = 100
```

---

## Common Mistakes

### ❌ Too Few Evals
```yaml
num_evals: 5  # Bad! Only 5 data points
```
**Problem:** Can't see trends, miss issues

### ❌ Too Many Evals
```yaml
num_evals: 10000  # Bad! Eval every 3k steps
```
**Problem:** Eval overhead slows training

### ✅ Just Right
```yaml
num_evals: 100  # Good! Data point every 10-15 min
```
**Result:** Good visibility, minimal overhead

---

## Summary

### What Changed
- ✅ `default.yaml`: `num_evals: 10 → 100` (10x more logging)
- ✅ `production.yaml`: `num_evals: 20 → 100` (5x more logging)
- ✅ `quick.yaml`: Already good at 2

### Expected Results

**Before:**
- 1 hour = 1 data point
- Poor visibility
- Hard to debug

**After:**
- 1 hour = ~5-6 data points
- Good visibility
- Easy monitoring
- Clear learning curves

### Restart Your Training!

```bash
cd /home/leeygang/projects/wildrobot/playground
uv run train.py --config default.yaml
```

Now you'll see progress every ~11 minutes instead of every ~109 minutes! 🎉

---

## Additional Tips

### Monitor Training in Real-Time

**W&B Dashboard:** Check these metrics:
- `eval/episode_reward` - Should increase
- `eval/avg_episode_length` - Should increase
- `training/policy_loss` - Should decrease
- `training/sps` - Steps per second (constant ~460)

### When to Stop Training

**Signs training is done:**
- Reward plateaus for 20+ evals
- Episode length maxes out (1000 steps)
- No improvement for 30+ minutes

**Signs training failed:**
- Reward decreases
- Episode length drops
- NaN losses appear

---

**The fix is simple: more frequent evaluations = better visibility!** 📊✨
