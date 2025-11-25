# Configurable Validation Metrics

## 🎯 Overview

The validation system is now **fully configurable via YAML**. You can:
- Define which metrics to track for validation
- Set custom tolerance thresholds
- Add new metrics without code changes
- All metrics are automatically saved in checkpoints

---

## ✨ What Changed

### **Before (Hardcoded):**

```python
# Old: Metrics hardcoded in train.py and visualize_policy.py
final_metrics = {
    "eval/episode_reward": 4800.0,
    "eval/avg_episode_length": 900.0,
}

# Old: Thresholds hardcoded
if reward_diff_pct <= 10:  # Hardcoded 10%
    print("PASS")
```

### **After (YAML-Configurable):** ✨

```yaml
# default.yaml
validation:
  # Define metrics to track (extensible!)
  metrics:
    reward: "eval/episode_reward"
    episode_length: "eval/avg_episode_length"
    # Add more metrics easily:
    # velocity: "eval/avg_velocity"
    # height: "eval/avg_height"

  # Configure thresholds
  thresholds:
    reward_tolerance: 0.10       # ±10%
    length_tolerance: 0.10       # ±10%
    determinism_threshold: 0.01  # < 0.01
```

---

## 📋 Configuration Structure

### **`validation.metrics`**

Map of `display_name: metric_key` pairs. Defines which metrics to save and validate.

```yaml
validation:
  metrics:
    reward: "eval/episode_reward"           # Required
    episode_length: "eval/avg_episode_length"  # Required
    velocity: "eval/avg_velocity"           # Optional - add if available
    height: "eval/avg_height"               # Optional - add if available
    action_rate: "eval/avg_action_rate"     # Optional - future metric
```

**How it works:**
1. During training, `train.py` reads this config
2. For each metric defined, it saves the final value in `final_metrics`
3. During validation, `visualize_policy.py` reads these metrics from checkpoint
4. Validation compares visualization metrics against saved training metrics

---

### **`validation.thresholds`**

Configurable pass/fail thresholds for validation checks.

```yaml
validation:
  thresholds:
    reward_tolerance: 0.10       # ±10% tolerance for reward consistency
    length_tolerance: 0.10       # ±10% tolerance for episode length
    determinism_threshold: 0.01  # Max reward diff for determinism check
```

**Threshold meanings:**

| Threshold | Default | Meaning |
|-----------|---------|---------|
| `reward_tolerance` | 0.10 | ±10% deviation allowed between visualization and training reward |
| `length_tolerance` | 0.10 | ±10% deviation allowed between visualization and training episode length |
| `determinism_threshold` | 0.01 | Max reward difference when running same seed twice (determinism check) |

---

## 🔄 How It Works

### **1. Training: Save Metrics**

`train.py` reads config and saves all specified metrics:

```python
# train.py (lines 264-276)
# Get validation metrics to track from config
validation_metrics = cfg.get("validation", {}).get("metrics", {})

def progress(num_steps, metrics):
    # Save final metrics based on validation config
    final_metrics = {"training_timesteps": num_steps}

    # Add all configured validation metrics
    for metric_name, metric_key in validation_metrics.items():
        if metric_key in metrics:
            final_metrics[metric_key] = metrics[metric_key]
```

**Checkpoint structure:**

```python
checkpoint_data = {
    "params": params,
    "config": cfg,  # Includes validation config!
    "final_metrics": {
        "eval/episode_reward": 4800.0,
        "eval/avg_episode_length": 900.0,
        "training_timesteps": 30000000,
        # Any additional metrics from config...
    }
}
```

---

### **2. Validation: Auto-Load Metrics**

`visualize_policy.py` automatically loads metrics from checkpoint:

```python
# visualize_policy.py (lines 448-464)
if "final_metrics" in checkpoint_data:
    final_metrics = checkpoint_data["final_metrics"]

    # Auto-load standard metrics
    if training_reward is None:
        training_reward = final_metrics.get("eval/episode_reward")
        print(f"Auto-loaded training_reward: {training_reward:.1f}")

    # Also load any additional metrics from config
    validation_metric_mapping = config.get("validation", {}).get("metrics", {})
    for metric_name, metric_key in validation_metric_mapping.items():
        if metric_key in final_metrics:
            auto_loaded_metrics[metric_name] = final_metrics[metric_key]
```

---

### **3. Validation: Use Config Thresholds**

Validation checks use thresholds from config:

```python
# visualize_policy.py
thresholds = validation_config.get("thresholds", {})
reward_tolerance = thresholds.get("reward_tolerance", 0.10)

# Check reward consistency
if reward_diff_pct <= reward_tolerance:
    print(f"✅ PASS: Within ±{reward_tolerance*100:.0f}% of training!")
```

---

## 🚀 Usage Examples

### **Example 1: Default Configuration**

```bash
# Use default thresholds (±10% for reward/length, 0.01 for determinism)
python train.py

# Validate with auto-loaded metrics
python visualize_policy.py --validate
```

**Output:**
```
✅ Found training metrics in checkpoint!
   Auto-loaded training_reward: 4800.0
   Auto-loaded training_length: 900.0

1️⃣  Reward Consistency:
    Difference from training: 0.3%
    Tolerance threshold: ±10% (from config)
    ✅ PASS: Within ±10% of training reward!
```

---

### **Example 2: Strict Validation (±5% tolerance)**

Edit `default.yaml`:

```yaml
validation:
  thresholds:
    reward_tolerance: 0.05  # ±5% (stricter!)
    length_tolerance: 0.05  # ±5% (stricter!)
```

Then:

```bash
python train.py
python visualize_policy.py --validate
```

**Output:**
```
1️⃣  Reward Consistency:
    Difference from training: 3.2%
    Tolerance threshold: ±5% (from config)
    ✅ PASS: Within ±5% of training reward!
```

---

### **Example 3: Track Additional Metrics**

Edit `default.yaml` to track velocity and height:

```yaml
validation:
  metrics:
    reward: "eval/episode_reward"
    episode_length: "eval/avg_episode_length"
    velocity: "eval/avg_velocity"  # NEW!
    height: "eval/avg_height"      # NEW!
```

**Requirements:**
1. Your environment must compute these metrics during eval
2. They must be returned in the `metrics` dict during training

**Then train and validate:**

```bash
python train.py
python visualize_policy.py --validate
```

**Output:**
```
✅ Found training metrics in checkpoint!
   Auto-loaded training_reward: 4800.0
   Auto-loaded training_length: 900.0
   Additional metrics from config:
     velocity: 0.52
     height: 0.45
```

---

## 📊 Adding New Metrics (Step-by-Step)

### **Step 1: Compute Metric in Environment**

Ensure your environment computes the metric during evaluation:

```python
# In your environment's step() or reset() method
def step(self, action):
    # ... existing code ...

    # Add custom metric
    avg_velocity = jp.mean(jp.abs(self.data.qvel[:3]))

    state.metrics.update({
        "avg_velocity": avg_velocity,  # Add this!
    })

    return state
```

---

### **Step 2: Add to YAML Config**

Edit `default.yaml`:

```yaml
validation:
  metrics:
    reward: "eval/episode_reward"
    episode_length: "eval/avg_episode_length"
    velocity: "eval/avg_velocity"  # Add this!
```

---

### **Step 3: Train Model**

```bash
python train.py
```

The metric will automatically be:
1. ✅ Tracked during training
2. ✅ Saved in checkpoint's `final_metrics`
3. ✅ Auto-loaded during validation

---

### **Step 4: Validate**

```bash
python visualize_policy.py --validate
```

**Output:**
```
✅ Found training metrics in checkpoint!
   Auto-loaded training_reward: 4800.0
   Auto-loaded training_length: 900.0
   Additional metrics from config:
     velocity: 0.52  ← Your new metric!
```

---

## 🎯 Benefits

### **1. No Code Changes Needed**
- Add metrics by editing YAML
- No need to modify Python code

### **2. Consistent Configuration**
- Same config used for training and validation
- Thresholds defined once, used everywhere

### **3. Extensible**
- Easy to add new metrics
- Easy to customize thresholds
- Future-proof design

### **4. Backward Compatible**
- Old checkpoints still work (fallback to defaults)
- Manual metrics still supported via CLI flags

### **5. Self-Documenting**
- Config file shows exactly what's being tracked
- Validation output shows threshold sources

---

## 🔧 Advanced Usage

### **Per-Experiment Custom Thresholds**

Create a custom config for strict validation:

```yaml
# strict_validation.yaml
validation:
  thresholds:
    reward_tolerance: 0.02      # ±2% (very strict!)
    length_tolerance: 0.02      # ±2%
    determinism_threshold: 0.001 # Almost perfect determinism
```

Then:

```bash
python train.py --config strict_validation.yaml
python visualize_policy.py --validate
```

---

### **Disable Specific Checks**

Set tolerance to very high value to effectively disable:

```yaml
validation:
  thresholds:
    reward_tolerance: 1.0  # ±100% (always passes)
    length_tolerance: 0.10 # ±10% (still checked)
```

---

### **Domain-Specific Metrics**

For manipulation tasks:

```yaml
validation:
  metrics:
    reward: "eval/episode_reward"
    episode_length: "eval/avg_episode_length"
    success_rate: "eval/success_rate"     # Task-specific!
    grasp_force: "eval/avg_grasp_force"   # Task-specific!
```

For locomotion tasks:

```yaml
validation:
  metrics:
    reward: "eval/episode_reward"
    episode_length: "eval/avg_episode_length"
    forward_velocity: "eval/avg_forward_velocity"  # Locomotion-specific!
    cost_of_transport: "eval/cost_of_transport"    # Locomotion-specific!
```

---

## 📝 Migration Guide

### **Old Checkpoints (No `final_metrics`)**

If you have old checkpoints without `final_metrics`:

**Option 1: Manual metrics (works immediately)**
```bash
python visualize_policy.py \
    --validate \
    --training-reward 4800 \
    --training-length 900
```

**Option 2: Retrain (future-proof)**
```bash
python train.py
# New checkpoint will have final_metrics!
```

---

### **Old Config Files**

If your config doesn't have `validation` section:

**Automatic fallback:**
- Code uses default metrics: reward + episode_length
- Code uses default thresholds: 10%, 10%, 0.01

**Recommended:**
Add `validation` section to your config:

```yaml
validation:
  metrics:
    reward: "eval/episode_reward"
    episode_length: "eval/avg_episode_length"
  thresholds:
    reward_tolerance: 0.10
    length_tolerance: 0.10
    determinism_threshold: 0.01
```

---

## 🎯 Summary

### **Key Points**

✅ **Metrics are now YAML-configurable** - no code changes needed to add new metrics
✅ **Thresholds are YAML-configurable** - easy to customize per-experiment
✅ **Auto-loading from checkpoint** - no manual metric specification needed
✅ **Backward compatible** - old checkpoints still work with manual metrics
✅ **Extensible** - easy to add domain-specific metrics

### **Quick Workflow**

```bash
# 1. Configure metrics (one-time)
vim default.yaml  # Add metrics to validation.metrics

# 2. Train
python train.py

# 3. Validate (auto-loads metrics!)
python visualize_policy.py --validate

# 4. Deploy with confidence! 🚀
```

---

## 🔗 Related Documentation

- **`VALIDATION_USAGE.md`** - User guide for validation
- **`VISUALIZATION_CONSISTENCY_VALIDATION.md`** - Theory and troubleshooting
- **`default.yaml`** - Default configuration with validation section

---

**Questions?** The validation system is now fully configuration-driven and ready for any custom metrics you want to track! 🎉
