# WildRobot Humanoid: Sim2Real Development Roadmap
**Version:** 2.0
**Target Platform:** MuJoCo Playground (MJX/JAX)
**Hardware Target:** 11-DOF Humanoid (4Nm Torque Limit)
**Methodology:** Zero-Shot Sim2Real with PPO + AMP

---

## 1. Executive Summary
This roadmap outlines the end-to-end development of a locomotion policy for the "WildRobot" humanoid. The core constraint is the **4Nm torque limit**, which necessitates a high-efficiency gait rather than high-dynamic hopping. We adopt **MuJoCo Playground (MJX)** to leverage massive GPU parallelism for rapid iteration and **Adversarial Motion Priors (AMP)** to enforce natural, energy-efficient walking styles from Day 1.

---

## 2. Technology Stack Selection

### Simulation Physics: **MuJoCo MJX**
* **Choice:** Google DeepMind's MJX (JAX-based MuJoCo) for physics simulation.
* **Justification:**
    * **Throughput:** Enables running 4,000+ parallel environments on a single GPU (vs. ~60 on CPU).
    * **Sim2Real Fidelity:** Exact same physics as standard MuJoCo used for hardware validation.
    * **JAX Integration:** Native support for JIT compilation and GPU acceleration.

### Training Framework: **Brax v2 with MJX Backend**
* **Choice:** Brax v2 training infrastructure using MJX physics.
* **Justification:**
    * **Battle-Tested:** Production-quality PPO implementation from Google DeepMind.
    * **Fast Prototyping:** Pre-built training loops, logging, checkpointing.
    * **Best Practice:** Leverage proven RL infrastructure instead of custom implementation.
    * **Maintains Physics:** Uses MJX backend so physics matches our MuJoCo model exactly.

### Algorithm: **PPO + AMP**
* **Choice:** Proximal Policy Optimization (PPO) with Adversarial Motion Priors (AMP).
* **Justification:**
    * **Energy Efficiency:** Pure RL tends to learn "jittery" or high-torque gaits. AMP forces the robot to mimic human biomechanics, which are naturally energy-efficient (critical for the 4Nm limit).
    * **Robustness:** Standard for modern legged robotics research.

---

## 3. Development Phases

### Phase I: Digital Twin & Asset Prep (Days 1-3)
**Goal:** Create a physics-accurate MJCF (`.xml`) simulation model from CAD.

1.  **Onshape Export:**
    * Use `onshape-to-robot` to export URDF/MJCF.
    * **Critical:** Verify mass and inertia tensors. Visual meshes must be separate from collision meshes (use primitive shapes for collisions to speed up MJX collision detection).
2.  **Model Configuration (The "4Nm" Constraint):**
    * Define actuators explicitly in XML:
        ```xml
        <motor name="knee_pitch" gear="1" ctrllimited="true" ctrlrange="-4.0 4.0"/>
        ```
    * Set joint damping and friction to realistic initial estimates (e.g., `damping="0.5"`, `frictionloss="0.1"`).
3.  **Sensor Site Placement:**
    * Add virtual sites for the IMU (torso) and foot contact sensors that match hardware mounting points exactly.

---

### Phase II: Physics Validation / "The Sanity Check" (Day 4)
**Goal:** Verify the robot is physically capable of walking *before* attempting RL. This prevents the "black box" problem where RL fails because the physics are impossible.

#### 1. Passive Drop Test
* **Script Logic:** Initialize robot 0.5m in air. Drop it.
* **Success Criteria:**
    * No explosion (model stability).
    * Feet rest flat on ground without jittering (contact solver stability).
    * Rest height matches expected kinematic height.

#### 2. Active Torque Test (CRITICAL)
**Why:** Your 4Nm limit is tight. If standing requires 3.8Nm, walking is impossible.
**Action:** Write a Python script (`validate_torque.py`) with this logic:
1.  **Load Model:** Load `wildrobot.xml` in standard MuJoCo.
2.  **Pose Robot:** Set joint positions to a "Deep Squat" (knees bent ~60 degrees) or "Single Leg Stance".
3.  **Inverse Dynamics:** Call `mj_inverse(model, data)` to calculate required torques to hold this pose against gravity.
4.  **Check:** Read `data.qfrc_inverse`.
    * **Pass:** Max torque < 2.5 Nm (leaving 1.5 Nm headroom for dynamic acceleration).
    * **Fail:** Max torque > 3.5 Nm.
    * **Resolution:** If Fail, you **MUST** return to Onshape to reduce torso weight or shorten the thigh/shank links. **Do not proceed to RL.**

---

### Phase III: RL Training Pipeline (Days 5-8)
**Goal:** Train the walking policy using Brax v2 + MJX for fast iteration.

**Training Approach:**
1. Wrap `WildRobotEnv` (MJX-based) in Brax's `Env` interface
2. Use Brax's PPO training infrastructure (no custom PPO implementation)
3. Leverage Brax's vectorization and JIT compilation for 2048+ parallel environments
4. Maintain exact MuJoCo physics fidelity through MJX backend

#### 1. Environment & Network Setup
* **Observation Space (44-dim):**
    * Projected Gravity Vector (3)
    * Joint Positions (11)
    * Joint Velocities (11)
    * Previous Action (11)
    * Clock/Phase signal (optional, 2)
    * **Noise:** Add Gaussian noise to all observations ($\sigma \approx 0.01-0.05$) during training to match real sensor noise.
* **Action Space:**
    * Target Joint Angles (PD setpoints).
    * **Control Frequency:** 50Hz (matches your real servos).

#### 2. Reward Function Configuration
The reward function guides the RL agent. For a 4Nm robot, **Energy Penalty** is the most important term.

| Component | Weight | Description |
| :--- | :--- | :--- |
| **Tracking Lin Vel** | `1.0` | $exp(-|v_{x} - v_{target}|^2 / \sigma)$. Drives forward motion. |
| **Tracking Ang Vel** | `0.5` | Keeps robot walking straight (penalizes yaw drift). |
| **AMP (Style)** | `0.5` | Discriminator reward. "Does this pose look like the AMASS dataset?" |
| **Torque Penalty** | **`-0.05`** | **Critical.** $sum(\tau^2)$. Penalizes high-torque moves. |
| **Action Rate** | `-0.01` | Penalizes jerky movements (derivative of action). Smooths motion. |
| **Dof Pos Limits** | `-0.1` | Penalizes hitting mechanical hard stops. |
| **Base Height** | `0.3` | Rewards keeping torso at target height (prevents crawling). |

#### 3. Domain Randomization (DR) Config
To bridge the Sim2Real gap, we randomize physics parameters every episode.

```python
domain_randomization = {
    "friction": [0.4, 1.25],       # Slippery to sticky floors
    "mass_scale": [0.85, 1.15],    # +/- 15% total robot mass
    "link_mass_scale": [0.8, 1.2], # Randomize individual link masses
    "push_force": [0, 10.0],       # Newtons (applied to torso every 2-4s)
    "motor_strength": [0.9, 1.1],  # Simulates weak/strong motors
    "joint_damping": [0.5, 1.5],   # +/- 50% damping variance
    "control_latency": [0, 0.02]   # Seconds (simulate comms delay)
}
```

---

## 4. AMP+PPO Training Architecture

### The Brax v2 Impedance Mismatch

**Problem:** Brax v2's `ppo.train()` is a monolithic, fully-JIT compiled training loop with no hooks for:
- Accessing intermediate `(s_t, s_{t+1})` transitions during rollout
- Separate discriminator updates
- Reward shaping using discriminator logits

**This is not a bug—it's a design choice in Brax v2.**

| Brax v2 Design | AMP Requirements |
|----------------|------------------|
| `ppo.train()` is monolithic JIT | Needs open training loop |
| Pure functional, no side effects | Needs discriminator state |
| Rollout + GAE + update fused | Needs access to `(s_t, s_{t+1})` |
| No callbacks or hooks | Needs reward shaping mid-rollout |

### Solution: Custom PPO+AMP Training Loop

We implement a custom training loop using Brax's building blocks, but owning the outer loop:

```
┌─────────────────────────────────────────────────────────────┐
│                    train.py (CLI)                       │
│                                                             │
│  python train.py --iterations 3000 --amp-weight 1.0     │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│              amp_ppo_training.py                            │
│          (Custom Training Loop - OWNER)                     │
│                                                             │
│  ┌────────────────────────────────────────────────────┐     │
│  │  for iteration in range(num_iterations):           │     │
│  │                                                    │     │
│  │    1. Collect rollout with AMP features            │     │
│  │    2. Train discriminator (separate optimizer)     │     │
│  │    3. Compute AMP-shaped rewards                   │     │
│  │    4. Compute GAE with shaped rewards              │     │
│  │    5. PPO update (4 epochs, 4 minibatches)         │     │
│  │                                                    │     │
│  └────────────────────────────────────────────────────┘     │
└──────┬─────────────────┬────────────────────┬───────────────┘
       │                 │                    │
┌──────▼──────┐  ┌───────▼────────┐  ┌────────▼────────┐
│ ppo_building│  │ discriminator  │  │ amp_features    │
│ _blocks.py  │  │ .py            │  │ .py             │
│             │  │                │  │                 │
│ • compute_  │  │ • AMPDiscrimi- │  │ • extract_amp_  │
│   gae()     │  │   nator        │  │   features()    │
│ • ppo_loss()│  │ • disc_loss()  │  │ • normalize_    │
│ • Policy/   │  │ • compute_amp_ │  │   features()    │
│   Value Net │  │   reward()     │  │ • RunningMean   │
└─────────────┘  └────────────────┘  │   Std           │
                                     └─────────────────┘
```

### File Structure

```
playground_amp/
├── training/
│   ├── __init__.py              # Module exports
│   ├── transitions.py           # AMPTransition dataclass
│   ├── ppo_building_blocks.py   # GAE, PPO loss, networks
│   └── amp_ppo_training.py      # Custom training loop
├── amp/
│   ├── discriminator.py         # AMP discriminator (existing)
│   ├── ref_buffer.py            # Reference motion buffer (existing)
│   └── amp_features.py          # Feature extraction
└── train.py                 # CLI entry point
```

### Training Loop Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                     One Training Iteration                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. ROLLOUT COLLECTION                                              │
│     ┌──────────────────────────────────────────────────────────┐    │
│     │  for step in range(num_steps):                           │    │
│     │      action, log_prob = policy.sample(obs)               │    │
│     │      value = value_net(obs)                              │    │
│     │      next_obs, reward, done = env.step(action)           │    │
│     │      amp_feat = extract_amp_features(obs, next_obs)      │    │
│     │      store(obs, next_obs, action, reward, done,          │    │
│     │            log_prob, value, amp_feat)                    │    │
│     └──────────────────────────────────────────────────────────┘    │
│                              │                                      │
│                              ▼                                      │
│  2. DISCRIMINATOR TRAINING (2 updates per iteration)                │
│     ┌──────────────────────────────────────────────────────────┐    │
│     │  agent_feat = rollout.amp_feat                           │    │
│     │  expert_feat = ref_buffer.sample()                       │    │
│     │                                                          │    │
│     │  loss = BCE(D(agent)→0, D(expert)→1) + GP               │    │
│     │  disc_params = disc_optimizer.update(grads)              │    │
│     └──────────────────────────────────────────────────────────┘    │
│                              │                                      │
│                              ▼                                      │
│  3. AMP REWARD SHAPING                                              │
│     ┌──────────────────────────────────────────────────────────┐    │
│     │  amp_reward = log(sigmoid(D(agent_feat)))                │    │
│     │  shaped_reward = env_reward + amp_weight * amp_reward    │    │
│     └──────────────────────────────────────────────────────────┘    │
│                              │                                      │
│                              ▼                                      │
│  4. GAE COMPUTATION                                                 │
│     ┌──────────────────────────────────────────────────────────┐    │
│     │  advantages, returns = compute_gae(                      │    │
│     │      rewards=shaped_reward,                              │    │
│     │      values=rollout.value,                               │    │
│     │      dones=rollout.done,                                 │    │
│     │      gamma=0.99, gae_lambda=0.95                         │    │
│     │  )                                                       │    │
│     └──────────────────────────────────────────────────────────┘    │
│                              │                                      │
│                              ▼                                      │
│  5. PPO UPDATE (4 epochs × 4 minibatches)                           │
│     ┌──────────────────────────────────────────────────────────┐    │
│     │  for epoch in range(4):                                  │    │
│     │      shuffle data                                        │    │
│     │      for minibatch in split(data, 4):                    │    │
│     │          loss = ppo_loss(policy, value, minibatch)       │    │
│     │          policy_params = optimizer.update(grads)         │    │
│     └──────────────────────────────────────────────────────────┘    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Data Structures

```python
class AMPTransition(NamedTuple):
    """Transition with AMP features for discriminator training."""
    obs: jnp.ndarray           # Current observation (num_envs, obs_dim)
    next_obs: jnp.ndarray      # Next observation (num_envs, obs_dim)
    action: jnp.ndarray        # Action taken (num_envs, act_dim)
    reward: jnp.ndarray        # Environment reward (num_envs,)
    done: jnp.ndarray          # Episode termination (num_envs,)
    log_prob: jnp.ndarray      # Log probability of action (num_envs,)
    value: jnp.ndarray         # Value function estimate (num_envs,)
    amp_feat: jnp.ndarray      # Motion features for AMP (num_envs, feat_dim)
```

### AMP Feature Extraction

Features extracted for discriminator (29-dim for WildRobot):
- Joint positions: 11 values (indices 7-18 of observation)
- Joint velocities: 11 values (indices 32-43 of observation)
- Base height: 1 value (index 0)
- Base orientation: 6 values (indices 1-7, 6D rotation representation)

```python
def extract_amp_features(obs, next_obs=None):
    joint_pos = obs[..., 7:18]       # 11 joints
    joint_vel = obs[..., 32:43]      # 11 velocities
    base_height = obs[..., 0:1]      # 1 value
    base_orient = obs[..., 1:7]      # 6D rotation

    return jnp.concatenate([
        joint_pos,      # 11
        joint_vel,      # 11
        base_height,    # 1
        base_orient,    # 6
    ], axis=-1)  # Total: 29 features
```

### Configuration

```python
@dataclass
class AMPPPOConfig:
    # Environment
    obs_dim: int = 44
    action_dim: int = 11
    num_envs: int = 16
    num_steps: int = 10

    # PPO hyperparameters
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    num_minibatches: int = 4
    update_epochs: int = 4

    # AMP configuration
    enable_amp: bool = True
    amp_reward_weight: float = 1.0  # Equal to task reward (2024 best practice)
    disc_learning_rate: float = 1e-4
    disc_updates_per_iter: int = 2
    r1_gamma: float = 5.0

    # Network architecture
    policy_hidden_dims: Tuple[int, ...] = (512, 256, 128)
    value_hidden_dims: Tuple[int, ...] = (512, 256, 128)
    disc_hidden_dims: Tuple[int, ...] = (1024, 512, 256)
```

### Usage

```bash
# Quick smoke test (10 iterations)
python playground_amp/train.py --verify

# Full training with AMP
python playground_amp/train.py \
    --iterations 3000 \
    --num-envs 32 \
    --amp-weight 1.0

# Training without AMP (pure PPO baseline)
python playground_amp/train_amp.py --no-amp

# Custom discriminator settings
python playground_amp/train_amp.py \
    --amp-weight 1.5 \
    --disc-updates 3 \
    --disc-lr 5e-5
```

### Why This Architecture Works

| Problem | Solution |
|---------|----------|
| JIT black box | We control the outer loop |
| Cannot inject AMP logic | AMP runs between rollout & update |
| No intermediate obs access | Collected in AMPTransition |
| Wrapper tracing errors | No Python-side wrappers, pure JAX tensors |
| Discriminator can't train | Separate update step with own optimizer |

### Expected Training Dynamics

```
Iteration 0-500:     Policy explores, AMP reward low (~-2.0)
Iteration 500-1500:  Gait stabilizes, AMP reward rises to ~0.0
Iteration 1500-3000: Natural walking emerges, AMP reward >0.5
Iteration 3000+:     Fine-tuning, AMP reward converges to ~0.7
```

### Metrics Logged

- **PPO Metrics:** policy_loss, value_loss, entropy_loss, clip_fraction, approx_kl
- **AMP Metrics:** disc_loss, disc_accuracy, amp_reward_mean, amp_reward_std
- **Environment:** episode_reward, episode_length, env_steps_per_sec

---

## 5. Phase IV: Sim2Real Transfer (Days 9-10)

**Goal:** Deploy trained policy to physical hardware.

### Hardware Interface
1. **Observation Mapping:** Map physical IMU/encoders to observation space
2. **Action Execution:** Convert policy output to servo commands (50Hz)
3. **Safety Limits:** Hard-coded torque and velocity limits in firmware

### Deployment Checklist
- [ ] ONNX export of policy network (<5ms inference)
- [ ] Safety wrapper with E-stop conditions
- [ ] Calibration procedure for joint offsets
- [ ] Initial slow-motion testing (reduced velocity commands)

---

## 6. Success Criteria

| Metric | Target | Rationale |
|--------|--------|-----------|
| **Forward Velocity** | 0.3-0.8 m/s | Natural walking pace |
| **Avg Joint Torque** | <2.5 Nm | Stay within 4Nm limit with headroom |
| **Episode Length** | >400 steps (8s) | Sustained walking without falling |
| **AMP Reward** | >0.7 | Natural, human-like gait |
| **Push Recovery** | Survives 10N | Robustness to perturbations |
| **Sim2Real Gap** | <20% performance drop | Successful transfer |

---

## 7. Risk Mitigation

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Policy doesn't learn | Medium | Start without AMP, add once walking emerges |
| Torque limit violation | High | Aggressive torque penalty in reward function |
| Sim2Real gap | High | Extensive domain randomization, sim2sim testing |
| Discriminator collapse | Low | Gradient penalty, limit disc updates per iter |

---

## 8. References

- **AMP Paper:** "AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control" (Peng et al., 2021)
- **Brax v2:** https://github.com/google/brax
- **MuJoCo MJX:** https://mujoco.readthedocs.io/en/stable/mjx.html
- **ETH Zurich Locomotion:** "Learning Quadrupedal Locomotion over Challenging Terrain" (2020)
