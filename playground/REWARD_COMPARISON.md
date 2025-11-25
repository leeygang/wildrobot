# Reward Function Comparison: WildRobot vs Loco-MuJoCo vs Open_Duck_Playground

## TL;DR Summary

| Framework | Optimization Goal | Complexity | Best For |
|-----------|------------------|------------|----------|
| **Your WildRobot** | **Velocity tracking** (standing OR walking) | Medium (7 components) | ✅ General locomotion |
| **Loco-MuJoCo** | **Velocity tracking** (standing OR walking) | High (13+ components) | ✅ Research, motion imitation |
| **Open_Duck_Playground** | **Standing stability** | Low (3-4 components) | Standing, simple tasks |

---

## Your Current WildRobot Reward Function

### Location
`/Users/ygli/projects/wildrobot/playground/wildrobot/locomotion.py` (Lines 197-256)

### Components & Weights

```python
total_reward = (
    3.0 * velocity_reward         # Velocity tracking (most important!)
    + lateral_penalty             # Stay straight (-0.5 * lateral_vel²)
    + 1.5 * height_reward         # Maintain 0.45m height
    + 1.5 * upright_reward        # Stay vertical (gravity aligned)
    + angvel_penalty              # Reduce wobbling (-0.01 * angvel²)
    + action_penalty              # Smooth actions (-0.001 * action²)
    + energy_penalty              # Energy efficiency (-0.001 * power)
)
```

### Breakdown

#### 1. **Velocity Tracking Reward** (Weight: 3.0) ⭐ Most Important!

```python
velocity_error = linvel[0] - velocity_cmd  # Forward velocity error
velocity_reward = exp(-velocity_error² / 0.01)
```

**What it does:**
- Exponential reward for matching commanded velocity
- Works for **standing** (velocity_cmd=0) AND **walking** (velocity_cmd>0)
- Tolerance: 0.1 m/s (σ)

**Examples:**
```python
# Standing (velocity_cmd = 0.0 m/s)
actual_velocity = 0.0 → velocity_reward = 1.0  ✅ Perfect!
actual_velocity = 0.1 → velocity_reward = 0.37 ❌ Bad!

# Walking (velocity_cmd = 0.5 m/s)
actual_velocity = 0.5 → velocity_reward = 1.0  ✅ Perfect!
actual_velocity = 0.6 → velocity_reward = 0.37 ❌ Bad!
```

#### 2. **Lateral Penalty** (Weight: -0.5)

```python
lateral_penalty = -0.5 * linvel[1]²
```

**What it does:**
- Penalize sideways drift
- Robot should move **straight** (no lateral velocity)

#### 3. **Height Tracking Reward** (Weight: 1.5)

```python
height_error = height - 0.45  # Target: 0.45m
height_reward = exp(-height_error² / 0.01)
```

**What it does:**
- Maintain standing height at 0.45m
- Tolerance: 0.1m (σ)

#### 4. **Upright Posture Reward** (Weight: 1.5)

```python
upright_reward = exp(-(gravity[2] - 1.0)² / 0.1)
```

**What it does:**
- Gravity vector should point up (z=1.0)
- Prevents robot from leaning/tilting

#### 5. **Angular Velocity Penalty** (Weight: -0.01)

```python
angvel_penalty = -0.01 * sum(angvel²)
```

**What it does:**
- Penalize wobbling/spinning
- Encourage stable upright posture

#### 6. **Action Smoothness Penalty** (Weight: -0.001)

```python
action_penalty = -0.001 * sum(action²)
```

**What it does:**
- Penalize large joint commands
- Encourage gentle, smooth movements

#### 7. **Energy Efficiency Penalty** (Weight: -0.001)

```python
energy_penalty = -0.001 * sum((action * qvel)²) * (velocity_cmd + 0.1)
```

**What it does:**
- Penalize mechanical power (torque × velocity)
- Scales with velocity command (more important when walking)

### Your Reward Analysis

**✅ Strengths:**
- **Unified policy**: Single reward for standing AND walking
- **Velocity-conditioned**: Different behaviors via velocity command
- **Balanced weights**: Good mix of tracking vs regularization
- **Exponential rewards**: Smooth, differentiable objectives
- **Simple**: Only 7 components (easy to understand/tune)

**❌ Potential Issues:**
- **No gait regularization**: Doesn't explicitly encourage walking patterns
- **No foot contact tracking**: Can't reward proper stepping
- **No joint acceleration penalty**: May allow jerky movements
- **No torque limits**: Can use unlimited power
- **No symmetry rewards**: Left/right legs treated independently

**🎯 Optimized For:**
- General locomotion (0-1 m/s)
- Smooth, stable movement
- Energy efficiency
- **Both standing AND walking in one policy**

---

## Loco-MuJoCo Reward Function

### Location
`/Users/ygli/projects/loco-mujoco/loco_mujoco/core/reward/default.py`

### Components (LocomotionReward class)

```python
total_reward = (
    tracking_reward               # Velocity tracking (x, y, yaw)
    + z_vel_penalty              # Minimize vertical bobbing
    + roll_pitch_vel_penalty     # Minimize roll/pitch rates
    + roll_pitch_pos_penalty     # Upright posture
    + nominal_joint_pos_penalty  # Stay near default pose
    + joint_position_limit_penalty # Don't exceed joint limits
    + joint_vel_penalty          # Minimize joint velocities
    + joint_acc_penalty          # Smooth joint motions
    + joint_torque_penalty       # Minimize torques
    + action_rate_penalty        # Smooth action changes
    + air_time_reward            # Reward foot lift (gait)
    + symmetry_air_reward        # Symmetric gait
    + energy_penalty             # Minimize power
)
```

### Detailed Breakdown

#### 1. **Velocity Tracking** (Primary reward)

```python
# 2D velocity + yaw tracking
goal_vel = [vel_x, vel_y, vel_yaw]  # Target velocities
vel_local = [measured_x, measured_y, measured_yaw]

tracking_reward_xy = exp(-w_xy * mean((vel_local[:2] - goal_vel[:2])²))
tracking_reward_yaw = exp(-w_yaw * mean((vel_local[2] - goal_vel[2])²))
total_tracking = w_sum_xy * tracking_reward_xy + w_sum_yaw * tracking_reward_yaw
```

**More sophisticated than yours:**
- Tracks **2D velocity** (x, y) not just forward
- Tracks **yaw rate** (turning)
- Separate weights for xy vs yaw

#### 2. **Z-Velocity Penalty** (Coeff: 2.0)

```python
z_vel_penalty = -2.0 * (z_velocity)²
```

**What it does:**
- Penalize vertical bobbing
- Encourages smooth, level walking

#### 3. **Roll/Pitch Penalties**

```python
# Velocity
roll_pitch_vel_penalty = -0.05 * sum(roll_vel², pitch_vel²)

# Position
roll_pitch_pos_penalty = -0.2 * sum(roll², pitch²)
```

**What it does:**
- Keep torso upright
- Minimize roll/pitch motions

#### 4. **Nominal Joint Position Penalty** (Coeff: 0.0 default)

```python
joint_qpos_penalty = -coeff * sum((qpos - qpos_nominal)²)
```

**What it does:**
- Encourage robot to stay near default pose
- Prevents extreme joint configurations

#### 5. **Joint Position Limit Penalty** (Coeff: 10.0)

```python
lower_penalty = -min(joint_pos - lower_limit, 0).sum()
upper_penalty = max(joint_pos - upper_limit, 0).sum()
joint_limit_penalty = -10.0 * (lower_penalty + upper_penalty)
```

**What it does:**
- **Hard penalty** for exceeding joint limits
- Prevents unrealistic poses

#### 6. **Joint Velocity Penalty** (Coeff: 0.0 default)

```python
joint_vel_penalty = -coeff * sum(joint_vel²)
```

**What it does:**
- Penalize fast joint movements
- Encourages slow, controlled motion

#### 7. **Joint Acceleration Penalty** (Coeff: 2e-7)

```python
acceleration = (joint_vel - last_joint_vel) / dt
acceleration_penalty = -2e-7 * sum(acceleration²)
```

**What it does:**
- **Penalize jerky movements**
- Encourages smooth joint trajectories
- **Missing in your reward!**

#### 8. **Joint Torque Penalty** (Coeff: 2e-5)

```python
torque_penalty = -2e-5 * sum(actuator_torque²)
```

**What it does:**
- Penalize high torques
- Encourages energy-efficient gaits

#### 9. **Action Rate Penalty** (Coeff: 1e-2)

```python
action_rate_penalty = -0.01 * sum((action - last_action)²)
```

**What it does:**
- Penalize rapid action changes
- Smoother than your action penalty (looks at change, not magnitude)

#### 10. **Air Time Reward** (Optional)

```python
# Reward for lifting feet during walking
air_time_reward = coeff * air_time (if foot not touching ground)
```

**What it does:**
- **Explicitly rewards stepping!**
- Encourages dynamic walking (not shuffling)
- **Missing in your reward!**

#### 11. **Symmetry Reward** (Optional)

```python
symmetry_reward = coeff * symmetry_between_left_right_feet
```

**What it does:**
- Encourages **symmetric gait**
- Left and right legs should behave similarly
- **Missing in your reward!**

#### 12. **Energy Penalty** (Optional)

```python
energy_penalty = -coeff * sum(torque * velocity)
```

**What it does:**
- Similar to yours but may have different formulation

### Loco-MuJoCo Analysis

**✅ Strengths:**
- **Comprehensive**: 13+ reward components
- **Gait-aware**: Explicit foot contact rewards
- **Smooth**: Joint acceleration + action rate penalties
- **Symmetric**: Can enforce left/right symmetry
- **Configurable**: All coefficients are tunable
- **Research-grade**: Used in published papers

**❌ Potential Issues:**
- **Complex**: Many hyperparameters to tune
- **Slower training**: More penalties = harder optimization
- **May overfit**: Lots of shaping can bias the policy

**🎯 Optimized For:**
- **Research**: Motion imitation, gait analysis
- **High-quality gaits**: Natural, efficient walking
- **Domain randomization**: Robust policies
- **Multiple tasks**: Standing, walking, running

---

## Open_Duck_Playground Reward (Hypothetical)

I don't have direct access to Open_Duck_Playground, but based on typical standing tasks:

### Typical Standing Reward

```python
total_reward = (
    upright_reward      # Stay vertical
    + height_reward     # Maintain height
    + stability_reward  # Minimize movement
)
```

### Components

#### 1. **Upright Reward**

```python
upright_reward = exp(-(gravity[2] - 1.0)²)
```

#### 2. **Height Reward**

```python
height_reward = exp(-(height - target_height)²)
```

#### 3. **Stability Reward**

```python
stability_reward = -sum(velocities²) - sum(angular_velocities²)
```

### Analysis

**✅ Strengths:**
- **Simple**: 3-4 components only
- **Fast training**: Easy optimization
- **Clear objective**: Just stand still

**❌ Limitations:**
- **No locomotion**: Can't walk
- **No velocity tracking**: No forward movement
- **Limited behaviors**: Standing only

**🎯 Optimized For:**
- **Standing stability**
- **Balance tasks**
- **Simple benchmarks**

---

## Side-by-Side Comparison

### Reward Component Table

| Component | Your WildRobot | Loco-MuJoCo | Open_Duck | Importance for Walking |
|-----------|----------------|-------------|-----------|----------------------|
| **Velocity tracking (forward)** | ✅ (3.0) | ✅ | ❌ | ⭐⭐⭐ Essential |
| **Velocity tracking (2D+yaw)** | ❌ | ✅ | ❌ | ⭐⭐ Nice to have |
| **Lateral penalty** | ✅ (-0.5) | ✅ (implicit) | ❌ | ⭐⭐⭐ Essential |
| **Height tracking** | ✅ (1.5) | ❌ (implicit) | ✅ | ⭐⭐ Important |
| **Upright posture** | ✅ (1.5) | ✅ (roll/pitch) | ✅ | ⭐⭐⭐ Essential |
| **Angular velocity penalty** | ✅ (-0.01) | ✅ (roll/pitch) | ✅ | ⭐⭐ Important |
| **Action smoothness** | ✅ (-0.001) | ❌ | ❌ | ⭐ Nice |
| **Action rate penalty** | ❌ | ✅ (-0.01) | ❌ | ⭐⭐ Important |
| **Energy efficiency** | ✅ (-0.001) | ✅ | ❌ | ⭐ Nice |
| **Joint acceleration penalty** | ❌ | ✅ (-2e-7) | ❌ | ⭐⭐ Smoothness |
| **Joint torque penalty** | ❌ | ✅ (-2e-5) | ❌ | ⭐⭐ Realism |
| **Foot air time reward** | ❌ | ✅ (optional) | ❌ | ⭐⭐⭐ Walking quality! |
| **Symmetry reward** | ❌ | ✅ (optional) | ❌ | ⭐⭐ Gait quality |
| **Joint limit penalty** | ❌ | ✅ (-10.0) | ❌ | ⭐⭐ Safety |
| **Z-velocity penalty** | ❌ | ✅ (-2.0) | ❌ | ⭐⭐ Smoothness |

### Complexity Comparison

```
Your WildRobot:     7 components  ████░░░░░░ 40% complex
Loco-MuJoCo:       13 components  ██████████ 100% complex
Open_Duck:          3 components  ██░░░░░░░░ 20% complex
```

---

## What's Missing in Your Reward?

### 🔴 Critical for Walking

#### 1. **Foot Contact / Air Time Reward**

**Problem:** Robot might "shuffle" instead of lift feet

**Current behavior:**
```
Standing: ███████████████ (feet always on ground) ✅ OK
Walking:  ███████████████ (shuffling, no stepping) ❌ Bad gait!
```

**With air time reward:**
```
Standing: ███████████████ (feet on ground) ✅
Walking:  ██░░░██░░░██░░░ (proper steps!) ✅
```

**How to add:**
```python
# Track foot contacts
left_foot_contact = check_contact(data, "left_foot", "floor")
right_foot_contact = check_contact(data, "right_foot", "floor")

# Reward for lifting feet during walking (but not during standing!)
if velocity_cmd > 0.1:  # Only when walking
    air_time_reward = 0.1 * (
        (1 - left_foot_contact) + (1 - right_foot_contact)
    )
else:
    air_time_reward = 0.0
```

#### 2. **Joint Acceleration Penalty**

**Problem:** Jerky, unrealistic movements

**Solution:**
```python
# Add to _get_reward()
last_qvel = state.metrics.get("last_qvel", qvel)
joint_acc = (qvel - last_qvel) / self.dt
acceleration_penalty = -2e-7 * jp.sum(jp.square(joint_acc))

# Update state
state.metrics["last_qvel"] = qvel
```

#### 3. **Action Rate Penalty (vs Action Magnitude)**

**Current:** Penalize large actions
```python
action_penalty = -0.001 * sum(action²)
```

**Better:** Penalize rapid action changes
```python
last_action = state.metrics.get("last_action", action)
action_rate_penalty = -0.01 * sum((action - last_action)²)
state.metrics["last_action"] = action
```

**Why better?** Allows large actions if they're smooth!

### 🟡 Nice to Have

#### 4. **Symmetry Reward**

```python
left_joints = action[:5]   # Assuming first 5 are left leg
right_joints = action[5:10]  # Next 5 are right leg

# Mirror symmetry (right should mirror left with opposite sign for some joints)
symmetry_penalty = -0.01 * sum((left_joints + right_joints)²)
```

#### 5. **Joint Limit Penalty**

```python
# Hard penalty for exceeding limits
joint_pos = qpos[7:]  # Actuated joints
lower_limits = model.jnt_range[:, 0]
upper_limits = model.jnt_range[:, 1]

limit_violation = (
    jp.sum(jp.maximum(lower_limits - joint_pos, 0))
    + jp.sum(jp.maximum(joint_pos - upper_limits, 0))
)
limit_penalty = -10.0 * limit_violation
```

---

## Recommendations

### For Your Current Goal (General Locomotion)

**Your reward is actually pretty good!** Here's why:

✅ **Velocity tracking** works for both standing and walking
✅ **Balanced weights** (not over-penalizing)
✅ **Simple** (easy to understand and debug)
✅ **Training works** (you got 900-step episodes, 4800 reward!)

### Suggested Improvements (Priority Order)

#### 🔴 **High Priority** (Add these!)

1. **Joint Acceleration Penalty**
   - Prevents jerky movements
   - Makes gait look more natural
   - Easy to add (3 lines of code)

2. **Action Rate Penalty** (replace action magnitude penalty)
   - Better than penalizing action size
   - Allows large but smooth actions
   - Standard in modern RL

3. **Foot Contact Tracking** (for walking only)
   - Reward lifting feet when velocity_cmd > 0.1
   - Prevents shuffling
   - Creates proper walking gait

#### 🟡 **Medium Priority** (Consider adding)

4. **Symmetry Reward**
   - Encourages natural gait
   - Reduces left/right bias
   - Good for aesthetics

5. **Joint Limit Penalty**
   - Prevents unrealistic poses
   - Adds safety
   - Useful for sim-to-real

#### 🟢 **Low Priority** (Probably not needed)

6. **2D Velocity Tracking** (x, y, yaw)
   - Your robot only needs forward movement
   - Skip unless you want strafing/turning

7. **Torque Penalty**
   - Energy penalty already covers this
   - May be redundant

---

## Example: Enhanced Reward Function

Here's your reward with the top 3 improvements:

```python
def _get_reward(
    self, data: mjx.Data, action: jax.Array, velocity_cmd: jax.Array,
    last_qvel: jax.Array, last_action: jax.Array,
) -> jax.Array:
    """Enhanced reward with acceleration, action rate, and gait rewards."""

    linvel = self.get_local_linvel(data)
    qvel = self.get_actuator_joints_qvel(data.qvel)

    # 1. Velocity tracking (most important)
    velocity_error = linvel[0] - velocity_cmd
    velocity_reward = jp.exp(-jp.square(velocity_error) / 0.01)

    # 2. Lateral penalty
    lateral_penalty = -0.5 * jp.square(linvel[1])

    # 3. Height tracking
    height = self.get_floating_base_qpos(data.qpos)[2]
    height_error = height - self._target_height
    height_reward = jp.exp(-jp.square(height_error) / 0.01)

    # 4. Upright posture
    gravity = self.get_gravity(data)
    upright_reward = jp.exp(-jp.square(gravity[2] - 1.0) / 0.1)

    # 5. Angular velocity penalty
    angvel = self.get_global_angvel(data)
    angvel_penalty = -0.01 * jp.sum(jp.square(angvel))

    # 6. ✨ NEW: Joint acceleration penalty (smoothness!)
    joint_acc = (qvel - last_qvel) / self.dt
    acceleration_penalty = -2e-7 * jp.sum(jp.square(joint_acc))

    # 7. ✨ NEW: Action rate penalty (replace action magnitude)
    action_rate = action - last_action
    action_rate_penalty = -0.01 * jp.sum(jp.square(action_rate))

    # 8. ✨ NEW: Foot contact reward (only when walking!)
    if velocity_cmd > 0.1:  # Walking
        # Check foot contacts (you'll need to implement this)
        left_contact = self._check_foot_contact(data, "left_foot")
        right_contact = self._check_foot_contact(data, "right_foot")
        # Reward for lifting feet
        gait_reward = 0.1 * (2 - left_contact - right_contact)
    else:  # Standing
        gait_reward = 0.0

    # 9. Energy efficiency
    energy_penalty = (
        -0.001 * jp.sum(jp.square(action * qvel)) * (velocity_cmd + 0.1)
    )

    # Weighted sum
    total_reward = (
        3.0 * velocity_reward
        + lateral_penalty
        + 1.5 * height_reward
        + 1.5 * upright_reward
        + angvel_penalty
        + acceleration_penalty    # ✨ NEW
        + action_rate_penalty     # ✨ NEW (replaces action_penalty)
        + gait_reward             # ✨ NEW
        + energy_penalty
    )

    return total_reward
```

---

## Summary Table

| Aspect | Your WildRobot | Loco-MuJoCo | Recommendation |
|--------|----------------|-------------|----------------|
| **Goal** | Standing + Walking | Standing + Walking | ✅ Same, good! |
| **Velocity tracking** | ✅ Forward only | ✅ 2D + yaw | ✅ Your approach is fine |
| **Smoothness** | ⚠️ Action magnitude | ✅ Action rate + Joint acc | 🔧 Add these! |
| **Gait quality** | ❌ No foot rewards | ✅ Air time + symmetry | 🔧 Add foot contact |
| **Safety** | ⚠️ No joint limits | ✅ Hard limits | 🟡 Consider adding |
| **Complexity** | ✅ Simple (7 terms) | ❌ Complex (13+ terms) | ✅ Keep it simple! |
| **Training speed** | ✅ Fast | ⚠️ Slower | ✅ Advantage! |
| **Tunability** | ✅ Easy | ❌ Many hyperparams | ✅ Advantage! |

**Verdict:** Your reward is **80% there!** Add the 3 high-priority improvements and you'll have research-quality rewards. 🎯

---

## Next Steps

1. ✅ **Current reward is working** - You got 900 steps, 4800 reward!
2. 🔧 **Add joint acceleration penalty** - 5 minutes
3. 🔧 **Replace action penalty with action rate** - 5 minutes
4. 🔧 **Add foot contact rewards** (if gait looks bad in video) - 30 minutes
5. 🎬 **Re-render video** with fixed normalization
6. 📊 **Evaluate gait quality** - Does robot shuffle or step properly?
7. 🔁 **Iterate** if needed

**Don't over-complicate!** Your current approach is solid. 👍
