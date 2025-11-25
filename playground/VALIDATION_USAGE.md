# Policy Consistency Validation - Quick Start Guide

## 🎯 Purpose

Validate that your trained policy's behavior in videos **accurately represents** what it will do on a real robot. This is critical before deploying to hardware!

---

## 🚀 Quick Usage

### **1. Basic Validation (No Training Metrics)**

```bash
cd ~/projects/wildrobot/playground

# Validate latest checkpoint
python visualize_policy.py --validate

# Validate specific checkpoint
python visualize_policy.py \
    --checkpoint ../training/logs/wildrobot_locomotion_flat_20251124-192712/checkpoints/final_policy.pkl \
    --validate
```

**Output:**
```
======================================================================
🔍 POLICY CONSISTENCY VALIDATION
======================================================================

📦 Loading checkpoint: checkpoints/wildrobot_locomotion_flat_20251124-192712/final_policy.pkl

📊 Training Metrics (from checkpoint)
----------------------------------------------------------------------
  Expected Reward:        (not provided - check W&B)
  Expected Episode Length: (not provided - check W&B)

🎯 Visualization Metrics (current rollouts)
----------------------------------------------------------------------
Running 10 rollouts to measure consistency...
  Trial  1/10: Reward= 4750.2, Length= 885
  Trial  2/10: Reward= 4820.5, Length= 910
  ...
  Trial 10/10: Reward= 4765.8, Length= 892

📈 Summary Statistics
----------------------------------------------------------------------
  Episode Reward:      4785.3 ±  156.2
  Episode Length:       897.5 ±   45.3
  Average Height:        0.448 ± 0.012 m
  Average Velocity:      0.523 ± 0.089 m/s

✅ Validation Checks
----------------------------------------------------------------------

1️⃣  Reward Consistency:
    Mean: 4785.3
    Std:  156.2
    Min:  4580.0
    Max:  4920.0
    ℹ️  No training reward provided - cannot validate

2️⃣  Episode Length Consistency:
    Mean: 897.5
    Std:  45.3
    ℹ️  No training length provided - cannot validate

3️⃣  Determinism Check:
    Same seed, different runs: reward diff = 0.000000
    ✅ PASS: Policy is deterministic!

4️⃣  Observation Normalization Check:
    ✅ Normalizer params found:
       Mean shape: (58,)
       Std shape:  (58,)
       First 5 means: [-0.002, 0.001, -0.003, 0.000, 0.002]
       First 5 stds:  [0.125, 0.118, 0.132, 0.095, 0.102]

5️⃣  Action Statistics:
    Shape: (8975, 11)
    Mean:  0.012
    Std:   0.234
    Min:   -0.987
    Max:   0.976
    ✅ Actions in expected range

======================================================================
🎉 VALIDATION PASSED!
======================================================================

✅ Your visualization accurately represents training performance!
✅ You can trust this policy for robot deployment.
```

---

### **2. Full Validation (With Training Metrics)**

Get training metrics from W&B first:

```bash
# Check your W&B dashboard for:
# - eval/episode_reward: 4800
# - eval/avg_episode_length: 900
```

Then run validation with expected values:

```bash
python visualize_policy.py \
    --validate \
    --training-reward 4800 \
    --training-length 900 \
    --n-trials 20
```

**Output:**
```
======================================================================
🔍 POLICY CONSISTENCY VALIDATION
======================================================================

📊 Training Metrics (from checkpoint)
----------------------------------------------------------------------
  Expected Reward:        4800.0
  Expected Episode Length: 900.0

...

1️⃣  Reward Consistency:
    Mean: 4785.3
    Std:  156.2
    Min:  4580.0
    Max:  4920.0
    Difference from training: 0.3%
    ✅ PASS: Within ±10% of training reward!

2️⃣  Episode Length Consistency:
    Mean: 897.5
    Std:  45.3
    Difference from training: 0.3%
    ✅ PASS: Within ±10% of training length!

...

======================================================================
🎉 VALIDATION PASSED!
======================================================================

✅ Your visualization accurately represents training performance!
✅ You can trust this policy for robot deployment.

📝 Next Steps:
  3. Proceed with confidence - visualization matches training! 🚀
======================================================================
```

---

### **3. Validation + Video Rendering**

You can run validation and then generate video:

```bash
# Step 1: Validate first
python visualize_policy.py --validate --training-reward 4800 --training-length 900

# Step 2: If validation passes, render video
python visualize_policy.py --output videos/validated_policy.mp4 --steps 2000
```

Or combine both in your workflow:

```bash
#!/bin/bash
# validate_and_render.sh

CHECKPOINT="checkpoints/wildrobot_locomotion_flat_20251124-192712/final_policy.pkl"
TRAINING_REWARD=4800
TRAINING_LENGTH=900

# Validate
echo "Step 1: Validating policy..."
python visualize_policy.py \
    --checkpoint "$CHECKPOINT" \
    --validate \
    --training-reward "$TRAINING_REWARD" \
    --training-length "$TRAINING_LENGTH" \
    --n-trials 20

# Check if validation passed (you'd parse the output in practice)
echo "Step 2: Rendering video..."
python visualize_policy.py \
    --checkpoint "$CHECKPOINT" \
    --output videos/validated_policy.mp4 \
    --steps 2000 \
    --height 720 \
    --width 1280
```

---

## 📋 Command-Line Options

### **Validation Flags**

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--validate` | flag | False | Run validation mode (no video) |
| `--n-trials` | int | 10 | Number of rollouts to average |
| `--training-reward` | float | None | Expected reward from W&B |
| `--training-length` | float | None | Expected episode length from W&B |

### **Checkpoint Selection**

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--checkpoint` | str | "latest" | Checkpoint path or "latest" |

### **Example Combinations**

```bash
# Quick validation (10 trials, no training metrics)
python visualize_policy.py --validate

# Thorough validation (20 trials, with training metrics)
python visualize_policy.py \
    --validate \
    --n-trials 20 \
    --training-reward 4800 \
    --training-length 900

# Validate specific checkpoint
python visualize_policy.py \
    --checkpoint path/to/checkpoint.pkl \
    --validate \
    --training-reward 4800
```

---

## ✅ Validation Checks Explained

### **1. Reward Consistency** ⭐ Most Important!

**What it checks:** Does visualization reward match training?

**Pass criteria:**
- ✅ Within ±10% of training: **Excellent**
- ⚠️ Within ±20% of training: **Acceptable** (investigate)
- ❌ >20% difference: **Problem!** (debug needed)

**Why it matters:** If rewards don't match, the policy isn't behaving as trained!

---

### **2. Episode Length Consistency**

**What it checks:** Do episodes last as long as during training?

**Pass criteria:**
- ✅ Within ±10%: **Excellent**
- ⚠️ Within ±20%: **Acceptable**
- ❌ >20% difference: **Problem!**

**Why it matters:** Short episodes = robot falls/fails early!

---

### **3. Determinism Check** ⭐ Critical for Robots!

**What it checks:** Same seed → same behavior?

**Pass criteria:**
- ✅ Reward diff < 0.01: **Perfect** (fully deterministic)
- ⚠️ Reward diff < 1.0: **Small randomness** (acceptable)
- ❌ Reward diff > 1.0: **Not deterministic!** (problem)

**Why it matters:** Real robots need predictable, repeatable behavior!

---

### **4. Observation Normalization Check**

**What it checks:** Are normalization parameters loaded correctly?

**Pass criteria:**
- ✅ Normalizer params found with correct shape
- ℹ️ No normalizer (if training didn't use normalization)

**Why it matters:** Wrong normalization = robot falls immediately!

---

### **5. Action Range Check**

**What it checks:** Are actions in expected range [-1, 1]?

**Pass criteria:**
- ✅ All actions in [-1, 1]: **Good**
- ⚠️ Actions > 2.0: **Warning** (may saturate)

**Why it matters:** Out-of-range actions indicate network issues!

---

## 🔍 Interpreting Results

### **✅ VALIDATION PASSED**

```
🎉 VALIDATION PASSED!
✅ Your visualization accurately represents training performance!
✅ You can trust this policy for robot deployment.
```

**What this means:**
- Video shows true policy behavior ✅
- Safe to deploy to robot ✅
- Training was successful ✅

**Next steps:**
1. Generate final video for documentation
2. Deploy to robot with confidence
3. Celebrate! 🎉

---

### **⚠️ VALIDATION FAILED**

```
⚠️  VALIDATION FAILED!
❌ Visualization does not match training performance.
❌ Investigate issues before deploying to robot.
```

**What this means:**
- Something's wrong with visualization OR training
- Not safe to deploy yet
- Debug needed

**Next steps:**
1. Check which specific checks failed
2. See troubleshooting guide below
3. Fix issues and re-validate

---

## 🔧 Troubleshooting

### **Issue 1: Reward Much Lower Than Training**

```
1️⃣  Reward Consistency:
    Difference from training: 45.2%
    ❌ FAIL: >20% difference from training!
```

**Possible causes:**
1. ❌ Normalization mismatch (check normalization params loaded)
2. ❌ Environment config mismatch (check ctrl_dt, sim_dt)
3. ❌ Wrong checkpoint loaded (verify checkpoint path)
4. ❌ Training metrics wrong (double-check W&B)

**Debug steps:**
```bash
# 1. Check normalization is loaded
python visualize_policy.py --validate | grep "Normalizer params"

# 2. Compare with training reward from W&B
# Visit: https://wandb.ai/your-project/runs/...

# 3. Try different checkpoint
python visualize_policy.py --checkpoint path/to/other_checkpoint.pkl --validate
```

---

### **Issue 2: Not Deterministic**

```
3️⃣  Determinism Check:
    Same seed, different runs: reward diff = 5.234000
    ❌ FAIL: Policy is not deterministic!
```

**Possible causes:**
1. ❌ Using stochastic policy (should use deterministic=True)
2. ❌ Environment has randomness (check reset())
3. ❌ JAX random state not controlled

**Fix:**
```python
# Deterministic is forced in validation - shouldn't happen!
# If you see this, report as a bug
```

---

### **Issue 3: High Variance Across Trials**

```
  Episode Reward:      4785.3 ±  856.2  # High std!
```

**Possible causes:**
1. Different velocity commands (expected)
2. Random initial states (expected)
3. Actual policy issues (investigate)

**Debug:**
```bash
# Run more trials to see if variance is real
python visualize_policy.py --validate --n-trials 50
```

---

## 🎯 Real Robot Deployment Checklist

Before deploying to hardware, ensure:

- [ ] ✅ Validation passed (reward within ±10%)
- [ ] ✅ Determinism check passed (diff < 0.01)
- [ ] ✅ Episode lengths consistent (within ±10%)
- [ ] ✅ Actions in valid range ([-1, 1])
- [ ] ✅ Video shows desired behavior
- [ ] ✅ Training metrics documented (W&B link)
- [ ] ✅ Checkpoint backed up and versioned
- [ ] ✅ Validation results saved

**Save validation results:**
```bash
# Run validation and save output
python visualize_policy.py \
    --validate \
    --training-reward 4800 \
    --training-length 900 \
    --n-trials 20 \
    | tee validation_results.txt

# Add to version control
git add validation_results.txt
git commit -m "Add validation results for checkpoint xyz"
```

---

## 📊 Advanced Usage

### **Programmatic Validation**

```python
from visualize_policy import validate_policy_consistency

# Run validation programmatically
results = validate_policy_consistency(
    checkpoint_path="checkpoints/final_policy.pkl",
    n_trials=20,
    training_reward=4800,
    training_length=900,
)

# Check results
if results["passed"]:
    print("✅ Safe to deploy!")
    print(f"Mean reward: {results['mean_reward']:.1f}")
else:
    print("❌ Not ready for deployment")
    print(f"Failed checks: {[k for k, v in results['checks'].items() if v == 'fail']}")
```

### **Save Validation Results**

```python
import json
from datetime import datetime

# Run validation
results = validate_policy_consistency(...)

# Add metadata
results["timestamp"] = datetime.now().isoformat()
results["checkpoint"] = checkpoint_path
results["training_reward_expected"] = 4800
results["training_length_expected"] = 900

# Save to JSON
with open("validation_results.json", "w") as f:
    json.dump(results, f, indent=2)
```

---

## 🔗 Related Documentation

- **`VISUALIZATION_CONSISTENCY_VALIDATION.md`** - Detailed explanation of consistency validation
- **`REWARD_COMPARISON.md`** - Reward function analysis
- **`OBSERVATION_NORMALIZATION_EXPLAINED.md`** - Why normalization matters

---

## 📝 Summary

### **Quick Workflow**

1. **Train model** → Get to high reward (4800+)
2. **Note training metrics** → From W&B: reward & length
3. **Validate policy** → `python visualize_policy.py --validate --training-reward 4800`
4. **Check results** → Must pass all checks
5. **Render video** → `python visualize_policy.py --output video.mp4`
6. **Deploy to robot** → With confidence! 🚀

### **Golden Rule**

> **"If validation passes, video = reality!"** ✅
>
> You can trust that what you see in the video is what the robot will do in real life.

---

**Questions? Issues?**

See `VISUALIZATION_CONSISTENCY_VALIDATION.md` for comprehensive troubleshooting! 🔍
