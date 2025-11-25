# Observation Normalization Explained

## What is Observation Normalization?

**Simple answer:** Converting sensor readings to have **mean=0** and **standard deviation=1**.

**Why?** Makes neural networks learn faster and more reliably.

---

## Real Example from Your WildRobot

### Your Robot's Raw Observations (58 dimensions)

```python
# What the robot actually sees (raw sensor data):
observation = [
    # Joint positions (11 joints, radians)
    0.45,    # hip joint at 0.45 radians
    -0.23,   # knee joint at -0.23 radians
    0.67,    # ankle joint at 0.67 radians
    ...      # 8 more joints

    # Joint velocities (11 joints, rad/s)
    0.5,     # hip velocity
    -2.3,    # knee velocity
    1.2,     # ankle velocity
    ...

    # Gravity vector (3D)
    0.0,     # x component
    0.0,     # y component
    -9.81,   # z component (gravity!)

    # Angular velocity (3D, rad/s)
    0.1,     # roll rate
    -0.05,   # pitch rate
    0.02,    # yaw rate

    # Gyroscope (3D, rad/s)
    0.08,
    -0.03,
    0.01,

    # Accelerometer (3D, m/s²)
    0.2,
    0.1,
    -9.8,    # Gravity again!

    # Previous action (11 dimensions)
    -0.4,
    0.0,
    0.8,
    ...

    # Velocity command (3D: forward, lateral, angular)
    0.5,     # Want to move forward at 0.5 m/s
    0.0,     # No lateral movement
    0.0,     # No turning
]
```

### The Problem: Huge Value Ranges!

```python
Joint positions:   -1.0  to  1.0   (range: 2)
Joint velocities:  -10.0 to 10.0   (range: 20)
Gravity:           -9.81 to  0.0   (range: ~10)
Angular velocity:  -5.0  to  5.0   (range: 10)
Accelerometer:     -20.0 to 20.0   (range: 40)
Velocity command:   0.0  to  1.0   (range: 1)
```

**Notice:** Different sensors have **wildly different scales**!

---

## Why This Breaks Neural Networks

### Neural Network Learning

Neural networks use **gradients** to learn. Gradients are sensitive to input scale.

**Example: Policy Network**

```python
# Simplified neural network
Input layer (58 neurons):
  [0.45, -0.23, 0.67, ..., -9.81, ..., 0.5, ...]
       ↓
  Multiply by weights
       ↓
Hidden layer (256 neurons)
       ↓
Output layer (11 neurons - joint commands)
```

**Problem with raw observations:**

```python
# Input to first layer:
neuron_1 = w1 * 0.45 + w2 * (-0.23) + w3 * (-9.81) + ...
#          ^^^^^^^^^^^^   ^^^^^^^^^^^^   ^^^^^^^^^^^^^
#          tiny value     tiny value     HUGE value!

# The gravity term (-9.81) DOMINATES all other inputs!
# Network can't learn subtle patterns in joint positions
```

**Result:**
- Network focuses on large values (gravity, accelerometer)
- Ignores small but important values (joint positions)
- Learning is unstable and slow

---

## How Normalization Fixes This

### Step 1: Collect Statistics During Training

```python
# Over millions of steps, track:
mean = np.mean(all_observations_seen, axis=0)
std = np.std(all_observations_seen, axis=0)

# Example means after 1M steps:
mean = [
    0.42,    # Hip joint usually at 0.42 rad
    -0.25,   # Knee joint usually at -0.25 rad
    ...
    0.0,     # Gravity x always 0
    0.0,     # Gravity y always 0
    -9.81,   # Gravity z always -9.81
    ...
    0.5,     # Usually walking forward at 0.5 m/s
]

# Example standard deviations:
std = [
    0.15,    # Hip varies by ±0.15 rad
    0.20,    # Knee varies by ±0.20 rad
    ...
    0.0,     # Gravity x doesn't vary (always 0)
    0.0,     # Gravity y doesn't vary
    0.1,     # Gravity z varies slightly due to tilt
    ...
    0.3,     # Velocity command varies by ±0.3 m/s
]
```

### Step 2: Normalize Each Observation

```python
normalized_obs = (obs - mean) / (std + epsilon)
# epsilon = 1e-8 to avoid division by zero

# Example:
# Raw hip joint:        0.45 rad
# Mean:                 0.42 rad
# Std:                  0.15 rad
# Normalized: (0.45 - 0.42) / 0.15 = 0.2

# Raw gravity z:        -9.81
# Mean:                 -9.81
# Std:                  0.1
# Normalized: (-9.81 - (-9.81)) / 0.1 = 0.0

# Raw velocity cmd:     0.5
# Mean:                 0.5
# Std:                  0.3
# Normalized: (0.5 - 0.5) / 0.3 = 0.0
```

### Step 3: All Values Now Similar Scale!

```python
# Before normalization:
[0.45, -0.23, 0.67, ..., -9.81, ..., 0.5]
 ^^^^  ^^^^^  ^^^^      ^^^^^^      ^^^
 small small  small     HUGE        small

# After normalization:
[0.2, -0.1, 1.3, ..., 0.0, ..., 0.0]
 ^^^  ^^^^  ^^^       ^^^       ^^^
 ALL SIMILAR SCALE! (mean≈0, std≈1)
```

**Benefits:**
- ✅ All inputs have similar importance
- ✅ Network can learn from all features equally
- ✅ Faster learning (gradients are balanced)
- ✅ More stable training (no exploding/vanishing gradients)

---

## Concrete Example: Walking Forward

### Without Normalization (Broken)

```python
# Robot trying to walk forward at 0.8 m/s
raw_obs = [
    0.48,    # Hip joint (slightly different from training mean)
    -9.81,   # Gravity (HUGE compared to other values)
    0.8,     # Velocity command (want to go faster)
]

# Network input:
hidden = relu(W * raw_obs + b)
       = relu(W * [0.48, -9.81, 0.8] + b)

# The -9.81 term DOMINATES:
# hidden ≈ relu(w2 * -9.81 + tiny_contributions)

# Network learned: "When gravity is -9.81, output default standing actions"
# Network IGNORES the velocity command (0.8) because it's so small compared to gravity!
# Robot can't learn to vary speed ❌
```

### With Normalization (Works!)

```python
# Same observation, normalized:
normalized_obs = [
    (0.48 - 0.42) / 0.15,     # = 0.4   (hip 0.4 std devs above mean)
    (-9.81 - (-9.81)) / 0.1,  # = 0.0   (gravity at expected value)
    (0.8 - 0.5) / 0.3,        # = 1.0   (velocity 1 std dev above mean)
]
# = [0.4, 0.0, 1.0]

# Network input:
hidden = relu(W * [0.4, 0.0, 1.0] + b)

# All values are similar scale!
# Network can detect: "velocity command is HIGH (1.0)"
# Network learned: "When velocity command is 1.0, increase step frequency"
# Robot walks faster! ✅
```

---

## Why It Broke Your Visualization

### During Training (Working Correctly)

```python
# Step 1: Robot gets raw sensor data
raw_obs = [0.45, -0.23, ..., -9.81, ..., 0.5]

# Step 2: Brax AUTOMATICALLY normalizes it
normalized_obs = (raw_obs - mean) / std
               = [0.2, -0.1, ..., 0.0, ..., 0.0]

# Step 3: Policy receives normalized observation
action = policy(normalized_obs)  # ✅ Correct!

# Step 4: Robot takes action and stays balanced
```

### During Visualization (BROKEN - Before Fix)

```python
# Step 1: Robot gets raw sensor data
raw_obs = [0.45, -0.23, ..., -9.81, ..., 0.5]

# Step 2: NO NORMALIZATION! ❌
# (visualize_policy.py didn't apply it)

# Step 3: Policy receives RAW observation
action = policy(raw_obs)  # ❌ WRONG!

# What the policy thinks it sees:
# "Hip joint is at 0.45" (but it learned 0.45 means normalized value!)
# "Gravity is -9.81" (but it learned -9.81 is a normalized value!)
#
# The policy was trained to interpret:
#   0.45 = "hip is 3 std devs above mean - EMERGENCY!"
#   -9.81 = "gravity is 100 std devs below normal - FALLING!"

# Policy outputs PANIC actions!
# Robot falls immediately ❌
```

### After Our Fix (Working!)

```python
# Step 1: Robot gets raw sensor data
raw_obs = [0.45, -0.23, ..., -9.81, ..., 0.5]

# Step 2: Load normalization params from checkpoint
mean = params['normalizer_params']['mean']
std = params['normalizer_params']['std']

# Step 3: NORMALIZE (just like training!)
normalized_obs = (raw_obs - mean) / std
               = [0.2, -0.1, ..., 0.0, ..., 0.0]

# Step 4: Policy receives normalized observation
action = policy(normalized_obs)  # ✅ Correct!

# Step 5: Robot walks/stands as trained! ✅
```

---

## Visual Explanation

### Before Normalization

```
Input Space (what neural network sees):

Joint Position Axis:
  -1.0 -------- 0.0 -------- 1.0
   |     (small range)       |

Gravity Axis:
  -9.81 -------- 0.0 -------- ???
   |     (HUGE RANGE!)       |

Velocity Cmd Axis:
   0.0 -------- 0.5 -------- 1.0
   |     (small range)       |

Problem: Network can't "see" the small ranges
when gravity is so much bigger!
```

### After Normalization

```
All Axes Normalized to Similar Scale:

Joint Position:
  -3σ ---- 0 (mean) ---- +3σ
   |   (balanced)      |

Gravity:
  -3σ ---- 0 (mean) ---- +3σ
   |   (balanced)      |

Velocity Cmd:
  -3σ ---- 0 (mean) ---- +3σ
   |   (balanced)      |

Now network can see all features equally!
```

---

## The Math Behind It

### Running Statistics (Welford's Algorithm)

During training, Brax updates normalization statistics incrementally:

```python
# Initialize
count = 0
mean = zeros(58)
M2 = zeros(58)  # Sum of squared deviations

# For each new observation batch:
for obs in new_observations:
    count += 1
    delta = obs - mean
    mean += delta / count
    delta2 = obs - mean
    M2 += delta * delta2

# Final statistics:
variance = M2 / count
std = sqrt(variance)
```

### Normalization Formula

```python
normalized = (x - mean) / (std + epsilon)

# Properties:
# - E[normalized] ≈ 0     (mean is 0)
# - Var[normalized] ≈ 1   (variance is 1)
# - std[normalized] ≈ 1   (standard deviation is 1)
```

### Why Epsilon?

```python
# Some dimensions might have zero variance (e.g., gravity x = 0.0 always)
# Division by zero would give NaN or Inf!

# With epsilon:
normalized = (0.0 - 0.0) / (0.0 + 1e-8)
           = 0.0 / 1e-8
           = 0.0  # Safe!
```

---

## Real Training Data Example

From your training (hypothetical statistics after 1M steps):

```python
# Observation dimension: Hip joint position
Raw values seen: [0.25, 0.48, 0.35, 0.52, 0.38, ...]
Mean: 0.42
Std: 0.12

# Normalized values:
0.25 → (0.25 - 0.42) / 0.12 = -1.42
0.48 → (0.48 - 0.42) / 0.12 =  0.50
0.35 → (0.35 - 0.42) / 0.12 = -0.58
0.52 → (0.52 - 0.42) / 0.12 =  0.83
```

**Interpretation:**
- `-1.42 std devs` = "Hip is bent more than usual"
- `+0.50 std devs` = "Hip is slightly extended"
- `-0.58 std devs` = "Hip is slightly bent"

The policy learned to **interpret these normalized values**, not raw radians!

---

## Common Misconceptions

### ❌ "Normalization changes the robot's behavior"

**Wrong!** Normalization is just a change of units, like converting inches to centimeters. The underlying physics doesn't change.

### ❌ "I can skip normalization for simple tasks"

**Risky!** Even simple tasks benefit from normalization. Your robot has 58 observations with very different scales - normalization is essential.

### ❌ "I need to normalize actions too"

**Usually no!** Actions are typically already in a bounded range (e.g., -1 to 1). Observations are the main problem because sensor ranges vary wildly.

### ✅ "Training and inference must use SAME normalization"

**Exactly!** This is why visualization broke - it didn't use the same normalization as training.

---

## Analogy: Temperature Scales

Imagine you're learning to predict weather comfort:

### Without Normalization (Like different units)

```python
# Training data uses Fahrenheit:
comfort = learn([32°F, 50°F, 68°F, 86°F, 104°F])
# Model learns: "68°F is comfortable"

# Inference uses Celsius:
prediction = model(20°C)
# Model thinks: "20 is very cold!" (because it learned Fahrenheit)
# WRONG! 20°C = 68°F = comfortable
```

### With Normalization (Standard units)

```python
# Training: Convert to standard scale
normalized_F = (F - 68) / 18  # Mean 68°F, Std 18°F
comfort = learn([-2.0, -1.0, 0.0, 1.0, 2.0])
# Model learns: "0.0 is comfortable"

# Inference: Convert to SAME standard scale
normalized_C = (20 - 20) / 10  # Must use Celsius stats!
prediction = model(normalized_C)
# Wrong if we use Fahrenheit stats!

# CORRECT: Use same temperature scale's normalization
```

**Your robot issue is similar:** Training used "normalized sensors", visualization used "raw sensors" → mismatch!

---

## How to Check Normalization is Working

### Inspect Saved Parameters

```python
import pickle

with open("final_policy.pkl", "rb") as f:
    data = pickle.load(f)

params = data["params"]

# Check for normalizer params
if "normalizer_params" in params:
    print("✅ Normalization params found!")
    mean = params["normalizer_params"]["mean"]
    std = params["normalizer_params"]["std"]

    print(f"Observation mean shape: {mean.shape}")  # Should be (58,)
    print(f"Observation std shape: {std.shape}")    # Should be (58,)

    print(f"\nFirst 5 dimensions:")
    print(f"Mean: {mean[:5]}")
    print(f"Std:  {std[:5]}")
else:
    print("❌ No normalization params!")
```

### Verify Normalization During Rollout

```python
# Add debugging to visualize_policy.py
def rollout_policy(...):
    ...
    for step in range(n_steps):
        # Print raw vs normalized
        if step == 0:
            print(f"Raw obs (first 5): {state.obs[:5]}")
            normalized = preprocess_fn(state.obs, params)
            print(f"Normalized (first 5): {normalized[:5]}")

        action, _ = policy(state.obs, action_rng)
        ...
```

---

## Performance Impact

### Memory

```python
# Normalization params storage:
mean: (58,) float32 = 232 bytes
std:  (58,) float32 = 232 bytes
Total: ~500 bytes (negligible!)
```

### Compute

```python
# Normalization per step:
normalized = (obs - mean) / (std + epsilon)

# 3 operations per dimension:
# - Subtraction: 58 ops
# - Addition: 58 ops (epsilon)
# - Division: 58 ops
# Total: 174 float operations

# On GPU: ~0.001ms (negligible!)
```

**Conclusion:** Normalization overhead is **tiny** compared to neural network inference.

---

## Best Practices

### ✅ Always normalize observations in RL

```yaml
ppo:
  normalize_observations: true  # Default for good reason!
```

### ✅ Save normalization params with policy

```python
checkpoint = {
    "params": params,  # Includes normalizer_params automatically
    "config": config,
}
```

### ✅ Use same preprocessing in training and inference

```python
# Training uses Brax's built-in normalization
# Inference must use params['normalizer_params']

preprocess_fn = lambda obs, params: normalize(obs, params['normalizer_params'])
```

### ✅ Test inference immediately after training

Don't wait hours to discover visualization is broken!

```python
# Right after training:
python visualize_policy.py --checkpoint final_policy.pkl --output test.mp4 --steps 100
```

---

## Summary

### What is Observation Normalization?

Converting sensor data to **mean=0, std=1** for each dimension.

### Why We Need It?

- Different sensors have **wildly different ranges**
- Neural networks learn better with **balanced inputs**
- Prevents one feature from **dominating** others

### How It Works?

1. Track running mean/std during training
2. Normalize: `(obs - mean) / std`
3. Policy learns from normalized observations

### Why It Broke Visualization?

- Training normalized observations
- Visualization didn't (before fix)
- Policy received wrong input scale
- Robot fell immediately

### The Fix?

- Load `normalizer_params` from checkpoint
- Apply same normalization in visualization
- Now it works! ✅

---

**The key insight:** Your policy didn't learn to read **raw sensors**. It learned to read **normalized sensors**. Give it the wrong format, and it acts like it's never seen a robot before! 🤖
