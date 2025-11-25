# Robot Falls in Video Despite Good Training Metrics

## The Problem 🐛

**Symptoms:**
- Training metrics look excellent:
  - `eval/avg_episode_length`: 50 → 900 steps ✅
  - `eval/episode_reward`: 200 → 4800 ✅
  - Training stable, losses decreasing ✅
- **BUT** robot falls down immediately in the rendered video ❌

## Root Cause 🔍

### **Observation Normalization Mismatch!**

**During Training** (`train.py` line 296):
```python
normalize_observations=True  # Observations normalized to mean=0, std=1
```

The policy learns to work with **normalized observations**:
```python
# Raw observation: [0.45, 0.23, -9.81, ...]
# Normalized:     [1.2, -0.8, 0.0, ...]  ← Policy trained on THIS
```

**During Visualization** (`visualize_policy.py` line 68):
```python
preprocess_observations_fn=lambda obs, params: obs,  # NO normalization!
```

The policy receives **raw observations**:
```python
# Raw observation: [0.45, 0.23, -9.81, ...]  ← Policy gets THIS instead!
```

**Result:** The policy outputs garbage actions because observations are completely wrong! The robot falls immediately.

---

## The Solution ✅

The saved checkpoint needs to include the **normalization parameters** (running mean and standard deviation), and visualization needs to apply them.

### Two Approaches:

### **Option 1: Use Brax's Built-in Normalization (Recommended)**

Brax's trained parameters already include normalization state. We just need to use them correctly in visualization.

### **Option 2: Disable Normalization Entirely**

Train without normalization. Simpler but may hurt performance.

---

## Fix #1: Load Normalization Parameters in Visualization

The issue is that Brax saves normalization parameters in `params`, but `visualize_policy.py` doesn't use them.

### Update `visualize_policy.py`:

```python
def load_policy(checkpoint_path: str, env):
    """Load trained policy from checkpoint."""
    import pickle

    with open(checkpoint_path, "rb") as f:
        checkpoint_data = pickle.load(f)

    params = checkpoint_data["params"]
    config = checkpoint_data["config"]

    # Extract network architecture from saved config
    policy_hidden_layers = config["network"]["policy_hidden_layers"]
    value_hidden_layers = config["network"]["value_hidden_layers"]

    # Check if normalization was used during training
    normalize_observations = config["ppo"]["normalize_observations"]

    print(f"  Policy network: {policy_hidden_layers}")
    print(f"  Value network: {value_hidden_layers}")
    print(f"  Normalize observations: {normalize_observations}")

    # Create policy network with SAME preprocessing as training
    if normalize_observations:
        # Use Brax's normalization preprocessing
        # params['normalizer_params'] contains running mean/std from training
        from brax.training import normalization

        def preprocess_fn(obs, params):
            # Extract normalizer params (running mean/std from training)
            normalizer_params = params.get('normalizer_params', params)
            return normalization.normalize(obs, normalizer_params)

        print(f"  Using observation normalization from training")
    else:
        # No normalization
        preprocess_fn = lambda obs, params: obs
        print(f"  No observation normalization")

    # Create policy network
    network_factory = ppo_networks.make_ppo_networks
    ppo_network = network_factory(
        env.observation_size,
        env.action_size,
        preprocess_observations_fn=preprocess_fn,  # Use correct preprocessing!
        policy_hidden_layer_sizes=policy_hidden_layers,
        value_hidden_layer_sizes=value_hidden_layers,
    )

    make_policy = ppo_networks.make_inference_fn(ppo_network)

    return make_policy, params
```

---

## Fix #2: Ensure Normalization Params are Saved

The default Brax checkpoint should include normalization parameters. Let's verify this is happening in `train.py`:

```python
# In train.py, after training (line 332-335):
final_checkpoint = ckpt_dir / "final_policy.pkl"
with open(final_checkpoint, "wb") as f:
    # params already includes normalizer_params from Brax
    pickle.dump({"params": params, "config": cfg}, f)  # This is correct!
```

**This should already be correct** - Brax's `params` object includes `normalizer_params`.

---

## Why This Happens

### Observation Normalization in RL

**Purpose:** Makes learning more stable

**How it works:**
```python
# During training, track running statistics
mean = running_mean(observations)  # e.g., [0.45, 0.23, -9.81, ...]
std = running_std(observations)    # e.g., [0.1, 0.15, 0.5, ...]

# Normalize each observation
normalized_obs = (obs - mean) / (std + epsilon)
# Result: mean ≈ 0, std ≈ 1 for all dimensions
```

**Example:**
```python
# Raw joint position: 0.45 radians
# Mean during training: 0.40
# Std during training: 0.10

# Normalized: (0.45 - 0.40) / 0.10 = 0.5

# Policy learns: "When I see 0.5, stand upright"
```

**What happens without normalization in visualization:**
```python
# Policy receives raw value: 0.45
# Policy expects normalized value: 0.5
# Policy thinks: "0.45 is WAY too high! Fall over to correct!"
# Result: Robot falls
```

---

## Debugging: Check Normalization Parameters

### Inspect the checkpoint:

```python
import pickle
import jax

checkpoint_path = "final_policy.pkl"
with open(checkpoint_path, "rb") as f:
    data = pickle.load(f)

params = data["params"]

# Print parameter structure
print("Keys in params:", jax.tree_util.tree_map(lambda x: x.shape, params).keys())

# Look for normalizer_params
if "normalizer_params" in params:
    print("\n✅ Normalizer params found!")
    normalizer = params["normalizer_params"]
    print("  Mean shape:", normalizer.get("mean", "Not found"))
    print("  Std shape:", normalizer.get("std", "Not found"))
else:
    print("\n❌ Normalizer params NOT found!")
    print("Available keys:", params.keys())
```

---

## Quick Test

Try running visualization with **stochastic policy** instead of deterministic:

```bash
python visualize_policy.py \
    --checkpoint final_policy.pkl \
    --output test.mp4 \
    --stochastic  # ← Try this!
```

**Why?** Stochastic policy adds noise, which might mask the normalization issue slightly. If the robot still falls, it's definitely normalization.

---

## Alternative: Retrain Without Normalization

If fixing visualization is too complex, you can retrain without normalization:

### Update `default.yaml`:

```yaml
ppo:
  normalize_observations: false  # Disable normalization
```

**Pros:**
- Visualization works immediately
- Simpler to debug

**Cons:**
- Training might be less stable
- May need to tune learning rate
- Potentially slower learning

---

## Summary

### The Bug
Training uses **observation normalization**, visualization doesn't.

### The Fix
Update `visualize_policy.py` to:
1. Check if normalization was used (`config["ppo"]["normalize_observations"]`)
2. If yes, apply normalization using `params['normalizer_params']`
3. Use Brax's `normalization.normalize()` function

### Files to Update
1. `/Users/ygli/projects/wildrobot/playground/visualize_policy.py` - Line 68

### Expected Result
Robot should stand/walk in video, matching the good training metrics!

---

## Next Steps

1. ✅ Update `visualize_policy.py` with normalization fix
2. ✅ Re-render video
3. ✅ Verify robot stands/walks
4. 🎉 Celebrate working policy!

The training worked! The visualization just needed to use the same preprocessing as training. 🚀
