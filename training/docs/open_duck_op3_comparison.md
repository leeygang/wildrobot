# Training Recipe Comparison: WildRobot vs Open Duck Mini vs OP3

**Created:** 2026-03-22
**Purpose:** Identify concrete gaps in the WildRobot training recipe by comparing
against two proven pipelines on similar hardware: Open Duck Mini (Open Duck
Playground) and ROBOTIS OP3 (mujoco_playground / DeepMind).

---

## Hardware Context

| Property | WildRobot | Open Duck Mini | OP3 |
|---|---|---|---|
| Actuated DOFs | 9 (4+4 legs, 1 waist) | 10 (5+5 legs) | 20 (6+6 legs, 3+3 arms, 2 head) |
| Servo type | HTD-45H position servo | STS3215 position servo | Dynamixel position servo |
| Torque limit | 4.41 Nm | 3.35 Nm | 5 Nm (XML forcerange) |
| Control freq | 50 Hz | 50 Hz | 50 Hz |
| IMU | 1x chest | 1x body | 1x torso |
| Foot sensing | 4 binary switches | 2 binary contacts | None used in RL |
| Ankle roll | No | Yes (via hip_yaw) | Yes |

All three robots are small position-controlled humanoids in the same class.
Open Duck Mini is the closest match to WildRobot.

---

## Side-by-Side Training Recipe

### Timing

| Parameter | WildRobot | Open Duck Mini | OP3 |
|---|---|---|---|
| ctrl_dt | 0.02 s (50 Hz) | 0.02 s (50 Hz) | 0.02 s (50 Hz) |
| sim_dt | 0.002 s (500 Hz) | 0.002 s (500 Hz) | 0.004 s (250 Hz) |
| Substeps | 10 | 10 | 5 |
| Episode length | 500 steps (10 s) | 1000 steps (20 s) | 1000 steps (20 s) |

### Action Space

| Parameter | WildRobot | Open Duck Mini | OP3 |
|---|---|---|---|
| Representation | Full-range: `center + action * span` | **Residual: `default + action * 0.25`** | **Residual: `default + action * 0.3`** |
| Action meaning at 0 | Center of joint range | Standing pose | Standing pose |
| Action scale | 40-50 deg per unit | **14 deg per unit** | **17 deg per unit** |
| Smoothing | Low-pass alpha=0.6 | Motor speed limit 5.24 rad/s | None |
| Motor speed limit | None | 5.24 rad/s (0.1 rad/step max) | None |

### Observation Space

| Component | WildRobot | Open Duck Mini | OP3 |
|---|---|---|---|
| Gravity vector | 3 (local) | via accelerometer (3) | 3 (upvector sensor) |
| Angular velocity | 3 (heading-local) | 3 (gyro) | 3 (gyro) |
| Joint positions | N (normalized) | 10 (relative to default) | 20 (relative to default) |
| Joint velocities | N (normalized) | 10 (scaled x0.05) | **Not included** |
| Foot contacts | 4 binary | 2 binary | **Not included** |
| Previous action | 1 frame | **3 frames** (t-1, t-2, t-3) | 1 frame |
| Motor targets | No | Yes (10) | No |
| Observation history | **None** | No (but 3 prev actions) | **3-frame stack (147 dims)** |
| Velocity command | 1 (forward only) | 3 (vx, vy, yaw) | 3 (vx, vy, yaw) |
| Gait phase | No (standing task) | 2 (cos/sin imitation phase) | No |
| Sensor noise | **Disabled** | Per-joint type noise | Uniform +/-0.05 |

### Reward Structure

| Term | WildRobot (standing) | Open Duck Mini | OP3 |
|---|---|---|---|
| **Alive / survival** | `base_height` = 2.0-3.0 | **`alive` = 20.0** | None |
| Orientation | -2.0 to -4.0 | via imitation | -5.0 |
| Tracking lin vel | 0.1-1.0 | +2.5 | +1.5 |
| Tracking ang vel | N/A | +6.0 | +0.8 |
| Torque | -0.001 | -1e-3 | -0.0002 |
| Action rate | -0.01 | -0.5 | -0.01 (includes jerk) |
| Slip | -1.0 | N/A | -0.1 |
| Collapse height | -0.5 to -1.0 | N/A | N/A |
| Collapse vz | -0.3 to -0.5 | N/A | N/A |
| Zero cmd / stand still | N/A | -0.2 | -0.5 |
| Imitation | None | +1.0 (composite) | None |
| Posture return | 0.3 | N/A | N/A |
| Energy | N/A | N/A | -0.0001 |
| Termination penalty | N/A | N/A | -1.0 |
| lin_vel_z | N/A | N/A | -2.0 |

### Termination Conditions

| Condition | WildRobot | Open Duck Mini | OP3 |
|---|---|---|---|
| Height too low | < 0.40 m | **None** | < 0.21 m |
| Tilt | pitch or roll > 0.8 rad (~46 deg) | **gravity_z < 0 (upside down only)** | gravity_z < 0.85 (~32 deg) |
| Joint limits | No | No | Yes |
| Height too high | > 0.70 m | No | No |

### Domain Randomization

| Parameter | WildRobot | Open Duck Mini | OP3 |
|---|---|---|---|
| Floor friction | **None** | U(0.5, 1.0) | **None** |
| Link masses | **None** | nominal * U(0.9, 1.1) | **None** |
| Extra torso mass | **None** | +U(-0.1, 0.1) kg | **None** |
| Actuator Kp | **None** | nominal * U(0.9, 1.1) | **None** |
| Joint armature | **None** | nominal * U(1.0, 1.05) | **None** |
| Joint frictionloss | **None** | nominal * U(0.9, 1.1) | **None** |
| CoM position | **None** | +U(-0.05, 0.05) m each axis | **None** |
| Joint zero offsets | **None** | +U(-0.03, 0.03) rad | **None** |

### Delay Simulation

| Parameter | WildRobot | Open Duck Mini | OP3 |
|---|---|---|---|
| Action delay | **None** | 0-2 ctrl steps (0-40 ms) | **None** |
| IMU delay | Implemented, **disabled** | 0-2 ctrl steps (0-40 ms) | **None** |

### Push / Perturbation

| Parameter | WildRobot | Open Duck Mini | OP3 |
|---|---|---|---|
| Type | Constant force (N) | Velocity impulse (m/s) | Sinusoidal force (**disabled**) |
| Magnitude | 3-10 N | 0.1-1.0 m/s | 1.0-5.0 m/s (if enabled) |
| Duration | 10 ctrl steps | Instantaneous | 0.05-0.2 s |
| Interval | Once per episode | Every 5-10 s | Every 1-3 s |
| Application | `xfrc_applied` on waist/upperbody | Direct `qvel` kick | `xfrc_applied` on torso |

### Reset Randomization

| Parameter | WildRobot | Open Duck Mini | OP3 |
|---|---|---|---|
| Joint noise | +/-0.05 rad | nominal * U(0.5, 1.5) | +/-0.03 rad |
| Base position | None | +/-0.05 m XY | +/-0.05 m XY |
| Base yaw | None | U(-pi, pi) | U(-pi, pi) |
| Base velocity | None | +/-0.05 all 6 DOFs | Angular vel noise |

### PPO Hyperparameters

| Parameter | WildRobot (v0.17.1) | Open Duck Mini | OP3 |
|---|---|---|---|
| num_envs | 512 | (Brax default) | 8192 |
| Rollout steps | 256 | (Brax default) | 20 |
| Learning rate | 1e-4 | 3e-4 | 3e-4 |
| Gamma | 0.99 | (Brax default) | 0.97 |
| Entropy coef | 0.005 | (Brax default) | 0.01 |
| Epochs | 2 | (Brax default) | 4 |
| Minibatches | 32 | (Brax default) | 32 |
| Network (actor) | [256, 256, 128] ELU | (Brax default) | [128, 128, 128, 128] |
| Network (critic) | [256, 256, 128] ELU | (Brax default) | [256, 256, 256, 256, 256] |
| Obs normalization | No | Yes | Yes |

---

## The 5 Gaps Most Likely Blocking Push Recovery

### Gap 1: Action representation (full-range vs residual)

**WildRobot** maps action [-1, 1] to the full joint range. An action output of
zero means the center of the joint range, which is NOT a standing pose. The
policy must simultaneously discover what standing looks like AND how to recover.

**Open Duck and OP3** both use residual actions around a standing default pose.
Action zero IS standing. The policy only learns small corrections. This
massively constrains the search space and gives the policy a free standing
baseline.

**Impact**: This is the highest-leverage gap. With full-range mapping, the
policy wastes capacity learning what "upright" means. With residual actions,
upright is free and all capacity goes to balance and recovery.

### Gap 2: Termination is too strict for learning recovery

**WildRobot** terminates at height < 0.40m and pitch/roll > 0.8 rad (46 deg).
This kills episodes before the policy can experience the states it would need
to recover from.

**Open Duck** terminates only when the robot is upside down (gravity_z < 0).
The robot can crouch to 0.10m, lean 80 degrees, stumble, and recover — all
within one episode. This is essential for learning recovery.

**Impact**: The policy never sees recovery-relevant states. It learns to avoid
falling (stay stiff, resist) rather than to recover from falling (step, absorb,
return). This directly explains why `term_height_low` is the dominant failure
mode — the policy has no training signal for what to do below 0.40m.

### Gap 3: Alive bonus is too small

**WildRobot** uses `base_height` weight 2.0-3.0 as a binary healthy signal,
competing with orientation penalty -4.0, collapse terms, and many others. The
net reward gradient may favor "dying quickly and cleanly" over "fighting through
an ugly recovery."

**Open Duck** uses `alive` = 20.0 — the single largest reward term by far.
Every step the robot survives generates +20.0. This creates overwhelming
pressure to stay upright at any cost.

**Impact**: Without a dominant survival signal, the policy finds local optima
where clean termination scores better than messy recovery (which triggers
orientation, torque, and action_rate penalties during the recovery).

### Gap 4: No domain randomization

**WildRobot** has zero domain randomization. The policy overfits to the exact
friction, mass, and gain parameters of the nominal simulation.

**Open Duck** randomizes 8 different physics parameters every episode. This
forces the policy to learn strategies that work across a range of conditions
rather than exploiting specific physics.

**Impact**: A push-recovery strategy that works for one exact friction/mass
combination is brittle. Domain randomization forces the policy to learn
generalizable balance rather than simulator-specific tricks. This also matters
for sim-to-real transfer.

### Gap 5: No action/sensor delay simulation

**WildRobot** has zero latency — the policy sees the current state and its
action is applied instantly.

**Open Duck** simulates 0-40ms random delay on both actions and IMU readings.

**Impact**: Without delay, the policy learns to rely on instantaneous reaction,
which is not available on real hardware at 50Hz. More importantly for sim
performance, delay forces the policy to develop anticipatory and conservative
strategies rather than purely reactive ones. This produces more robust
behavior even in simulation.

---

## Recommendations (Priority Order)

### 1. Switch to residual action representation

```
motor_targets = default_standing_pose + action * action_scale
```

Suggested `action_scale = 0.25 rad` (~14 deg), matching Open Duck. This means
action=0 produces the standing pose. The policy only learns corrections.

This is the single highest-leverage change.

### 2. Relax termination conditions

Change to Open Duck style:
- Remove pitch/roll termination entirely
- Lower height termination to 0.10-0.15m (or use gravity_z < 0)
- Let the policy experience near-fall states and learn to recover

### 3. Add a large alive bonus

Set alive/survival reward to 15-20x the scale of other terms. Survival must
dominate the reward landscape. This creates the gradient pressure to fight
through ugly recoveries rather than accepting clean termination.

### 4. Add domain randomization

Minimum set (matching Open Duck):
- Floor friction: U(0.5, 1.0)
- Link masses: nominal * U(0.9, 1.1)
- Actuator Kp: nominal * U(0.9, 1.1)
- Joint zero offsets: +U(-0.03, 0.03) rad

### 5. Enable action and IMU delay

- Action delay: 0-2 ctrl steps random
- IMU delay: 0-2 ctrl steps random (infrastructure already exists)
- Enable sensor noise (infrastructure already exists)

---

## What About the Footstep Planner (v0.17.3)?

Open Duck Mini achieves walking and push recovery on similar hardware with pure
RL using the recipe above. No footstep planner, no capture point, no LIP model.

The key was not adding a planner but fixing the training recipe fundamentals:
residual actions, relaxed termination, alive bonus, domain randomization.

The planner approach in v0.17.3 is architecturally reasonable, but the 5 gaps
above would block it too — a planner providing step targets will not help if the
RL execution layer cannot learn recovery because of the action representation
and termination constraints.

Suggested sequence:
1. Fix the 5 recipe gaps on the current pure-RL standing stack
2. Re-evaluate against the hard-push gate (eval_hard > 60%)
3. If the gap closes, the planner may not be needed for current goals
4. If the gap remains, add the planner on top of the fixed recipe

---

## Key References

| Project | Robot Class | Source |
|---|---|---|
| Open Duck Playground | Open Duck Mini, hobby servos | github.com/apirrone/Open_Duck_Playground |
| Open Duck Mini Runtime | Deployment on RPi | github.com/apirrone/Open_Duck_Mini_Runtime |
| mujoco_playground OP3 | ROBOTIS OP3, Dynamixel servos | github.com/google-deepmind/mujoco_playground |
| DeepMind OP3 Soccer | OP3 sim-to-real | arXiv:2304.13653 |
| Rhoban BAM | Servo actuator modeling | github.com/Rhoban/bam |
| Rhoban IKWalk | Spline walk, RoboCup | github.com/Rhoban/IKWalk |
| ROBOTIS OP3 | Sinusoidal walk + gyro PD | github.com/ROBOTIS-GIT/ROBOTIS-OP3 |

## Local Reference Code

- Open Duck env: `/Users/ygli/projects/Open_Duck_Playground/playground/open_duck_mini_v2/joystick.py`
- Open Duck rewards: `/Users/ygli/projects/Open_Duck_Playground/playground/common/rewards.py`
- Open Duck domain rand: `/Users/ygli/projects/Open_Duck_Playground/playground/common/randomize.py`
- Open Duck reference motion: `/Users/ygli/projects/Open_Duck_Playground/playground/common/poly_reference_motion.py`
- OP3 env: `/Users/ygli/projects/mujoco_playground/mujoco_playground/_src/locomotion/op3/joystick.py`
- WildRobot env: `/Users/ygli/projects/wildrobot/training/envs/wildrobot_env.py`
- WildRobot standing config: `/Users/ygli/projects/wildrobot/training/configs/` (v0.17.x configs)
