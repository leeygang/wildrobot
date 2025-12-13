# Phase 3: RL Training Pipeline - Detailed Execution Plan
**Status:** Ready to Execute (Phase 2 Validated ✓)  
**Target:** Robust walking policy for 11-DOF, 4Nm-limited humanoid  
**Timeline:** 7-10 days (GPU-accelerated)  
**Framework:** MuJoCo Playground (MJX/JAX) + PPO + AMP

---

## Overview & Success Criteria

**Phase Goal:** Train a zero-shot sim2real locomotion policy that achieves:
- Forward walking at 0.3-0.8 m/s on flat terrain
- Energy efficiency: <3Nm average torque per joint
- Robustness: Recovers from 10N external pushes
- Natural gait: AMP discriminator reward >0.7

**Exit Criteria for Phase 3:**
1. Policy walks 50m+ without falling in sim (100% success over 20 rollouts)
2. Average power consumption <30W during steady-state walking
3. Pass sim2sim transfer test (frozen policy works in IsaacLab/Sim with same MJCF)
4. ONNX export succeeds with <5ms inference latency on target hardware

---

## Step 3.1: Environment & Infrastructure Setup (Day 1)

### Goal
Establish reproducible training infrastructure with GPU-accelerated environments and monitoring.

### Tasks

#### 3.1.1: Install MJX/JAX Training Stack
```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Navigate to project root
cd /home/leeygang/projects/wildrobot

# Create isolated Python environment with uv
uv venv --python 3.10 .venv
source .venv/bin/activate

# Install JAX with CUDA support (adjust for your CUDA version)
uv pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install MuJoCo and MJX
uv pip install mujoco mujoco-mjx

# Install RL dependencies
uv pip install brax==0.10.3  # For env wrappers and PPO
uv pip install wandb tensorboard  # For logging
uv pip install onnx onnxruntime  # For deployment

# Sync all dependencies to pyproject.toml (recommended)
uv pip compile --all-extras pyproject.toml -o requirements-rl.txt
uv pip sync requirements-rl.txt
```

**Exit Criteria:**
- ✓ `python -c "import jax; print(jax.devices())"` shows GPU
- ✓ `python -c "import mujoco.mjx as mjx"` imports without error
- ✓ Test script runs 4000+ parallel envs at >1000 FPS
- ✓ `uv pip list` shows all packages installed

#### 3.1.2: Create WildRobot Brax/MJX Environment
**Location:** `/home/leeygang/projects/wildrobot/mujoco/playground/envs/wildrobot_env.py`

**Implementation Checklist:**
- [ ] Observation space: 44-dim (IMU orientation, joint pos/vel, previous action, command)
- [ ] Action space: 11-dim target joint positions (PD control)
- [ ] Control frequency: 50Hz (0.02s timestep)
- [ ] Domain randomization: 7 parameters (friction, mass, damping, latency, push, motor strength, terrain)
- [ ] Termination: base height <0.25m OR pitch/roll >45° OR timeout (10s)

**Validation:**
```python
# Test environment creation
env = WildRobotEnv(backend='mjx')
jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)

# Benchmark throughput
key = jax.random.PRNGKey(0)
state = jit_reset(key)
# Should achieve >2000 env-steps/sec on RTX 3090
```

**Exit Criteria:**
- ✓ Environment resets without NaN/Inf in state
- ✓ Random actions run 1000 steps without crashes
- ✓ Vectorized env (4096 parallel) achieves >1500 FPS
- ✓ Domain randomization varies parameters correctly (inspect 10 random resets)

#### 3.1.3: Setup Experiment Tracking
**Tools:** Weights & Biases (primary) + TensorBoard (backup)

```python
# Initialize tracking
import wandb
wandb.init(
    project="wildrobot-locomotion",
    config={
        "phase": "3.1_baseline",
        "algorithm": "PPO",
        "num_envs": 4096,
        "learning_rate": 3e-4,
        "gamma": 0.99,
        ...
    }
)
```

**Metrics to Log (per iteration):**
- Reward components (velocity tracking, AMP, torque penalty, etc.)
- Episode length, success rate
- Average torque per joint, max torque
- Policy entropy, value loss, policy loss
- FPS, wall-clock time

**Exit Criteria:**
- ✓ WandB dashboard shows live training curves
- ✓ TensorBoard backup logs to `logs/` directory

---

## Step 3.2: PPO+AMP Training Foundation (Days 2-5)

### Goal
Train walking policy with AMP from day 1 for natural, energy-efficient gait (industry SoTA approach).

### Modern Best Practice (2024-2025)
**Start with AMP immediately** rather than baseline PPO first. Recent industry deployments (Unitree Go2, Figure 01, ANYmal) and academic work (ETH Zurich, DeepMind) demonstrate that:
- Motion priors prevent learning inefficient gaits that need unlearning
- Faster convergence to natural locomotion
- Better sim2real transfer (human motion is already real-world validated)

**Optional Debug Path:** If training fails, you can disable AMP temporarily to isolate reward function issues, but this is not the primary workflow.

### Tasks

#### 3.2.1: Prepare Reference Motion Dataset (Day 2 Morning)
**Source Options (in priority order):**

1. **AMASS Dataset (Recommended):**
   - Download: https://amass.is.tue.mpg.de/
   - Use SMPL-H walking sequences (subjects: CMU, BioMotionLab)
   - Retarget to WildRobot skeleton using IK

2. **CMU MoCap (Fallback):**
   - Subject 35 (walking trials 01-10)
   - Already in BVH format, easier retargeting

**Retargeting Script:**
```python
# Location: mujoco/playground/utils/retarget_mocap.py
# Input: SMPL walking sequence (60 fps)
# Output: WildRobot joint angles (50 fps, 11 DOF)
# Method: Inverse kinematics with MuJoCo IK solver
```

**Processing Steps:**
1. Load SMPL mesh sequence
2. Extract keypoint trajectories (pelvis, knees, ankles)
3. Solve IK for WildRobot MJCF model to match end-effector positions
4. Smooth and resample to 50Hz
5. Validate: Play back in MuJoCo viewer - motion should look stable, feet on ground

**Exit Criteria:**
- ✓ 50+ seconds of retargeted walking motion (2500+ frames)
- ✓ Playback in MuJoCo shows robot feet grounded (no floating/penetration)
- ✓ Motion saved as `.pkl` with shape `(num_frames, 22)` [11 pos + 11 vel]
- ✓ Visual inspection: motion looks smooth and natural

#### 3.2.2: Implement AMP Discriminator
**Architecture (DeepMind AMP paper + 2024 improvements):**

```python
class AMPDiscriminator(nn.Module):
    """Distinguishes reference motion from policy rollouts."""
    def __init__(self, obs_dim=22):  # 11 joint pos + 11 joint vel
        self.layers = [
            nn.Linear(obs_dim, 1024),
            nn.ELU(),
            nn.LayerNorm(1024),  # 2024 addition: stabilizes training
            nn.Linear(1024, 512),
            nn.ELU(),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 1),  # Output: logit (real vs fake)
        ]
    
    def forward(self, x):
        # x: (batch, 22) joint pos/vel
        # Returns: (batch, 1) discriminator score
        for layer in self.layers:
            x = layer(x)
        return x
```

**Training Protocol:**
- For every policy update:
  1. Sample 512 frames from reference motion (real)
  2. Sample 512 frames from current policy rollouts (fake)
  3. Update discriminator with binary cross-entropy: `D(real)→1, D(fake)→0`
  4. Add AMP reward to policy: `r_amp = log(sigmoid(D(policy_frame)))`
  5. Use gradient penalty (WGAN-GP style) for stability

**Hyperparameters (2024 SoTA):**
```python
amp_config = {
    "discriminator_lr": 1e-4,        # Separate optimizer from policy
    "amp_reward_weight": 1.0,        # ⚠️ Equal to task reward (not 0.5)
    "replay_buffer_size": 200000,    # Larger buffer for diversity
    "discriminator_batch_size": 512, # Doubled from 256
    "gradient_penalty_weight": 5.0,  # WGAN-GP regularization
    "discriminator_updates": 2,      # Update D twice per policy update
}
```

**Exit Criteria:**
- ✓ Discriminator achieves 70-80% accuracy (not 100% - means policy is learning!)
- ✓ AMP reward visible in logs, starts negative but trends upward
- ✓ Implementation tested: forward pass, backward pass, no NaN gradients

#### 3.2.3: Configure PPO Hyperparameters (State-of-Art 2024/2025)
**Reference:** Based on DeepMind/Unitree/ETH Zurich best practices

```python
ppo_config = {
    # Environment
    "num_envs": 4096,              # Parallel environments
    "num_steps": 10,               # Steps per env before update (horizon = num_envs * num_steps)
    "episode_length": 500,         # Max steps (10s at 50Hz)
    
    # Network Architecture
    "policy_layers": [512, 256, 128],  # Actor network
    "value_layers": [512, 256, 128],   # Critic network
    "activation": "elu",               # ELU > ReLU for stability
    
    # PPO Parameters
    "learning_rate": 3e-4,         # AdamW optimizer
    "gamma": 0.99,                 # Discount factor
    "gae_lambda": 0.95,            # Generalized Advantage Estimation
    "clip_param": 0.2,             # PPO clipping epsilon
    "entropy_coef": 0.01,          # Encourage exploration (decay to 0.001)
    "value_loss_coef": 0.5,        # V-function loss weight
    "max_grad_norm": 1.0,          # Gradient clipping
    
    # Training Schedule
    "num_iterations": 3000,        # ~30M timesteps
    "num_minibatches": 4,          # Split batch for SGD
    "update_epochs": 8,            # PPO epochs per iteration
    
    # Early Stopping
    "target_reward": 300,          # Stop if average episode reward exceeds this
}
```

**Modern Enhancements Applied:**
1. **Reward Scaling:** Normalize rewards by running standard deviation (prevents gradient explosion)
2. **Observation Normalization:** Running mean/std for all observations
3. **Learning Rate Schedule:** Cosine annealing from 3e-4 to 1e-5
4. **Entropy Decay:** Start 0.01, decay to 0.001 over 1000 iterations
5. **Mixed Precision Training:** Use JAX FP16 for 2x speedup (2024 standard)

##Configuration (With AMP from Start):**

```python
reward_scales = {
    # Task objectives
    "tracking_lin_vel": 1.5,      # Main objective: forward velocity
    "tracking_ang_vel": 0.5,      # Yaw stability
    
    # AMP (motion style) - ⚠️ EQUAL WEIGHT to task (2024 SoTA)
    "amp_style": 1.0,             # Discriminator reward
    
    # Efficiency constraints
    "torque": -2e-4,              # CRITICAL: Energy (sum of squared torques)
    "action_rate": -0.01,         # Smoothness (derivative of action)
    
    # Safety & stability
    "dof_pos_limits": -10.0,      # Harsh penalty for joint limits
    "base_height": 0.5,           # Maintain 0.5m torso height
    "orientation": 1.0,           # Penalize pitch/roll deviation
    
    # Gait quality
    "feet_air_time": 0.5,         # Reward swing phase (>0.1s airborne)
    "feet_contact_forces": -0.01, # Discourage hard landings
    "dof_vel": -5e-5,             # Gentle joint velocity penalty
}
```

**Key Difference from Old Approach:**
- `amp_style` weight is **1.0** (equal to velocity tracking), not 0.5
- This is 2024 best practice: motion quality is as important as task completion
- Prevents learning fast but jerky/unnatural gaits

**Tuning Protocol:**
1. **Day 2 Afternoon:** Train for 1000 iterations (~10M steps)
2. **Analyze:** 
   - Is AMP reward increasing? (Should go from ~-2 to ~0.5)
   - Is robot walking or stuck? (If stuck, temporarily reduce AMP weight to 0.5)
   - Check torque values (should be <3.5Nm average)
3. **Adjust:** Fine-tune weights based on behavior
4. **Day 3:** Continue training to 3000 iterations

**Exit Criteria:**
- ✓ Robot achieves forward motion >0.3 m/s within 2000 iterations
- ✓ AMP discriminator reward >0.5 (motion is becoming natural)
- ✓ Average episode length >350 steps (7 seconds)
- ✓ No catastrophic failures (reward hacking)
- ✓ Average torque <3.2Nm per joint

#### 3.2.5: Initial Training Run (Days 2-3)
#### 3.3.3: Train PPO+AMP Policy
**Training Run:**
```bash
python mujoco/playground/train_ppo_amp.py \
    --config configs/wildrobot_amp.yaml \
    --num_envs 4096 \
    --num_iterations 5000 \
    --amp_data data/retargeted_walking.pkl \
    --wandb_project wildrobot-locomotion \
    --wandb_name "ppo-amp-v1"
```

**Expected Training Dynamics:**
- **Iterations 0-500:** Policy explores, AMP reward low (~-2.0)
- **Iterations 500-1500:** Gait stabilizes, AMP reward rises to ~0.0
- **Iterations 1500-3000:** Natural walking emerges, AMP reward >0.5
- **Iterations 3000+:** Fine-tuning, diminishing returns

**Monitoring Checklist (check every 500 iters):**
- [ ] Task reward increasing (velocity tracking improving)
- [ ] AMP discriminator reward rising (motion becoming more natural)
- [ ] Torque penalty decreasing (energy efficiency improving)
- [ ] Episode length increasing (fewer falls)

**Exit Criteria:**
- ✓ Training completes 5000 iterations (~50M timesteps)
- ✓ Final policy achieves:
  - Episode reward >350 (vs baseline ~250)
  - AMP reward >0.7
  - Success rate >85% on flat terrain
  - Average torque <2.8Nm per joint
- ✓ Model saved as `checkpoints/ppo_amp_final.pkl`

---

## Step 3.4: Curriculum Learning & Robustness (Day 6)

### Goal
Progressively increase task difficulty to improve robustness and generalization.

### Tasks

#### 3.4.1: Velocity Command Curriculum
**Progression:**
1. **Stage 1 (Iters 0-1000):** Fixed command: 0.4 m/s forward
2. **Stage 2 (Iters 1000-2000):** Random commands: 0.2-0.6 m/s forward
3. **Stage 3 (Iters 2000-3000):** + Lateral velocity: ±0.2 m/s
4. **Stage 4 (Iters 3000+):** + Yaw rate: ±0.5 rad/s

**Implementation:**
```python
# Auto-curriculum based on success rate
if avg_success_rate > 0.8:
    curriculum_stage += 1
    velocity_range = CURRICULUM_STAGES[curriculum_stage]
```

**Exit Criteria:**
- ✓ Policy tracks commanded velocities with <10% error
- ✓ Smooth transitions between different velocity commands

#### 3.4.2: Terrain Randomization
**Progressive Difficulty:**
- **Flat (baseline):** µ=1.0 friction, no perturbations
- **Slippery (stage 1):** µ ∈ [0.4, 1.2]
- **Uneven (stage 2):** Add small height field (±2cm noise)
- **Perturbed (stage 3):** Random 5-10N pushes every 2-4 seconds

**Implementation:**
```python
domain_randomization = {
    "friction": lambda: jax.random.uniform(key, minval=0.4, maxval=1.25),
    "push_interval": lambda: jax.random.uniform(key, minval=2.0, maxval=4.0),
    "push_force": lambda: jax.random.uniform(key, minval=0.0, maxval=10.0),
}
```

**Exit Criteria:**
- ✓ Policy maintains >70% success rate under:
  - Friction range [0.4, 1.25]
  - 10N lateral pushes
  - ±15% mass variations

#### 3.3.3: Adversarial Robustness Testing
**Test Suite:**
1. **Worst-case friction:** µ=0.3 (icy floor)
2. **Heavy payload:** +30% torso mass
3. **Motor weakness:** -20% torque limit
4. **Sensor noise:** 3x normal noise level
5. **Combined worst-case:** All above simultaneously

**Exit Criteria:**
- ✓ Policy walks >10m in at least 3/5 worst-case scenarios
- ✓ Identifies failure modes for future improvement
Command:**
```bash
python mujoco/playground/train.py \
    --config configs/wildrobot_ppo_amp.yaml \
    --num_envs 4096 \
    --num_iterations 5000 \
    --amp_data data/retargeted_walking.pkl \
    --wandb_project wildrobot-locomotion \
    --wandb_name "ppo-amp-v1-foundation"
```

**Expected Training Dynamics (AMP from Start):**
- **Iterations 0-300:** Chaotic exploration, both task and AMP rewards low
- **Iterations 300-800:** Policy discovers walking, AMP reward rises from -2 to -0.5
- **Iterations 800-2000:** Gait stabilizes and naturalizes, AMP reward >0.3
- **Iterations 2000-3500:** Fine-tuning, AMP reward reaches >0.7
- **Iterations 3500-5000:** Diminishing returns, focus shifts to robustness

**Monitoring Checklist (check every 500 iters):**
- [ ] Task reward (velocity tracking) steadily increasing
- [ ] **AMP reward is the leading indicator** - should rise faster than task reward initially
- [ ] Torque values staying within bounds (<4Nm)
- [ ] Episode length increasing (fewer early terminations)
- [ ] Video inspection: gait looks progressively more natural

**Debugging: If AMP Prevents Learning (Rare)**
If after 1000 iterations robot hasn't discovered forward motion:
1. Temporarily reduce `amp_reward_weight` to 0.3
2. Train for 500 more iterations until walking emerges
3. Gradually increase back to 1.0 over next 500 iterations
4. This is the "baseline first" fallback, but not standard practice

**Exit Criteria:**
- ✓ Training completes 5000 iterations (~50M timesteps, ~6-8 hours on RTX 3090)
- ✓ Final 3olicy achieves:
  - Task reward: forward velocity tracking error <0.1 m/s
  - **AMP reward >0.7** (primary quality metric)
  - Success rate >85% on flat terrain, no DR
  - Average torque <2.8Nm per joint
- ✓ Model checkpoint saved: `checkpoints/ppo_amp_foundation/iter_5000.pkl`
- ✓ WandB logs complete with videos of best episodestorque
    ret3rn final_success_rate - 0.1 * avg_torque
```

**Budget:** 20 trials × 2000 iterations each (~2-3 hours on RTX 3090)

**Exit Criteria:**
- ✓ Top 3 configurations achieve >5% improvement over baseline
- ✓ Best config saved to `configs/wildrobot_optimized.yaml`

#### 3.4.2: Ablation Study
**Components to Ablate (For Research/Understanding Only):**
1. **No AMP** (pure PPO with task rewards only)
2. **No torque penalty** (allow high-power gaits)
3. **No action smoothing** (allow jerky motions)
4. **No domain randomization** (sim-only tuning)
5. **Full model** (all components - baseline)

**Expected Results (based on 2024 literature):**
- No AMP: Works but gait is unnatural, higher sim2real gap
- No torque penalty: Exceeds 4Nm limits, won't work on hardware
- No action smoothing: Works but causes motor wear
- No DR: Works in sim, fails on hardware (large sim2real gap)
- Full model: Best overall performance

**Exit Criteria:**
- ✓ Ablation results logged in WandB comparative table
- ✓ Quantify: AMP improves sim2real transfer by ~20-30%
- ✓ Quantify: Torque penalty reduces energy by ~25%
5
---

## Step 3.6: Long-Horizon Stability Training (Day 8)

### Goal
Ensure policy can walk indefinitely without drift or cumulative errors.
5
### Tasks

#### 3.6.1: Extended Episode Training
**Configuration Changes:**
```python
# Increase episode length
episode_length = 2000  # 40 seconds at 50Hz (was 500)

# Add drift penalties
reward_scales["xy_drift"] = -0.1  # Penalize deviation from straight line
reward_scales["base_height_variance"] = -0.05  # Penalize bobbing
```

**Training:** Continue from best checkpoint for 2000 more iterations

**Exit Criteria:**
- ✓ Policy walks straight for 100m without falling (sim test)
- ✓ Hei5ht variance <5cm over 50-step window
- ✓ XY drift <0.5m over 100m trajectory

#### 3.6.2: Failure Recovery Training
**Inject Failures:**
- Random joint position resets (simulate slip)
- Large push forces (15-20N)
- Temporary motor dropout (set one motor torque to 0 for 0.5s)

**Recovery Objective:** Policy should return to stable walking within 2 seconds

**Exit Criteria:**
- ✓ Recovery success rate >60% for 15N pushes
- ✓ No catastrophic failures (robot getting stuck in bad attractor)

---

## Step 3.6: Sim2Sim Validation (Day 9)

### Goal
Verify policy transfers to different simulators (pre-flight check before real hardware).

### Tasks

#### 3.6.1: IsaacLab Transfer Test
**Procedure:**
1. Export MJCF to USD (if not done): `bash scripts/convert_mjcf_to_usd.sh`
2. Create IsaacLab environment using same USD
3. Export policy to ONNX: 
   ```python
   # Export JAX policy to ONNX
   dummy_obs = jnp.zeros((1, 44))
   policy_fn = jax.jit(policy.apply)
   onnx_model = jax2onnx.convert(policy_fn, dummy_obs)
   onnx.save(onnx_model, "policy.onnx")
   ```
4. Run policy in IsaacLab with ONNX runtime
5. Compare behavior to MJX

**Expected:** Some performance degradation (IsaacLab uses PhysX, different contact model)

**Exit Criteria:**
- ✓ Policy loads and runs in IsaacLab without errors
- ✓ Success rate >60% (vs 85% in MJX)
- ✓ Gait pattern qualitatively similar
- ✓ If success rate <50%: Re-tune domain randomization in MJX to better match PhysX

#### 3.6.2: Policy Inspection & Analysis
**Behavioral Tests:**
1. **Proprioception ablation:** Zero out joint velocity observations → does policy still walk?
2. **Frequency response:** Apply sinusoidal velocity commands (0.1-2 Hz) → track bandwidth
3. **Latency sensitivity:** Add artificial 50ms, 100ms delay → measure degradation

**Exit Criteria:**
- ✓ Policy robust to 50ms latency (success rate >70%)
- ✓ Can track velocity commands up to 0.5 Hz
- ✓ Degrades gracefully with missing observations (no crashes)

---

## Step 3.7: Deployment Preparation (Day 10)

### Goal
Package policy for real-hardware deployment with monitoring and safety checks.

### Tasks

#### 3.7.1: ONNX Export & Optimization
```python
# Export with optimizations
import onnx
from onnxruntime.quantization import quantize_dynamic

# 1. Export from JAX
onnx_model = export_jax_to_onnx(policy_network, dummy_input)

# 2. Optimize graph
from onnx import optimizer
passes = ['eliminate_unused_nodes', 'fuse_consecutive_transposes']
optimized = optimizer.optimize(onnx_model, passes)

# 3. (Optional) Quantize to INT8 for edge devices
quantized = quantize_dynamic(optimized, weight_type=QuantType.QInt8)

onnx.save(optimized, "wildrobot_policy_fp32.onnx")
onnx.save(quantized, "wildrobot_policy_int8.onnx")
```

**Validation:**
```python
# Test inference speed
import onnxruntime as ort
session = ort.InferenceSession("wildrobot_policy_fp32.onnx")

obs = np.random.randn(1, 44).astype(np.float32)
for _ in range(1000):
    start = time.time()
    action = session.run(None, {"obs": obs})[0]
    latency = time.time() - start
# Average latency should be <5ms on target hardware
```

**Exit Criteria:**
- ✓ FP32 ONNX model: <5ms inference on target CPU
- ✓ INT8 ONNX model: <2ms inference (if quantization doesn't degrade success rate)
- ✓ Model size <50MB

#### 3.7.2: Safety Wrapper Implementation
**Location:** `wildrobot/runtime/control/safety.py`

**Features:**
1. **Torque Limiting:** Clip commands to ±4Nm
2. **Velocity Limiting:** Max joint velocity = 15 rad/s
3. **Orientation Watchdog:** E-stop if pitch/roll >60° (robot flipped)
4. **Temperature Monitoring:** Reduce torque if motor temp >80°C
5. **Emergency Stop:** Kill switch interrupts policy, sets all joints to damping mode

```python
class PolicySafetyWrapper:
    def __init__(self, policy_path, torque_limit=4.0):
        self.policy = ort.InferenceSession(policy_path)
        self.torque_limit = torque_limit
        
    def get_action(self, obs):
        raw_action = self.policy.run(None, {"obs": obs})[0]
        
        # Safety checks
        if abs(obs['pitch']) > 1.05 or abs(obs['roll']) > 1.05:  # ~60 degrees
            return self.emergency_stop()
        
        # Clip to safe ranges
        safe_action = np.clip(raw_action, -self.torque_limit, self.torque_limit)
        return safe_action
```

**Exit Criteria:**
- ✓ Safety wrapper tested in sim (inject unsafe states)
- ✓ E-stop triggers correctly
- ✓ Minimal latency overhead (<0.5ms)

#### 3.7.3: Create Deployment Package
**Structure:**
```
deployment/
├── policy.onnx              # Trained policy
├── config.yaml              # Robot config (joint limits, control freq, etc.)
├── safety_config.yaml       # Safety thresholds
├── deploy.py                # Main control loop
├── README.md                # Hardware setup instructions
└── tests/
    ├── test_policy_load.py
    ├── test_latency.py
    └── test_safety.py
```

**Exit Criteria:**
- ✓ Package tested on deployment hardware (if available)
- ✓ Documentation complete (pin configurations, calibration steps)
- ✓ Unit tests pass (pytest)

---

## Step 3.8: Final Validation & Documentation (Day 10)

### Goal
Comprehensive testing and documentation for Phase 3 completion.

### Tasks

#### 3.8.1: Sim Evaluation Suite
**Test Scenarios:**
1. **Flat terrain, 0.5 m/s forward:** 100 episodes
2. **Slippery floor (µ=0.5):** 100 episodes
3. **Random pushes (10N):** 100 episodes
4. **Velocity tracking (0.2-0.8 m/s):** 100 episodes
5. **Turning (±0.5 rad/s yaw):** 100 episodes

**Success Metrics:**
| Scenario | Target Success Rate | Target Avg Torque |
|----------|---------------------|-------------------|
| Flat terrain | >90% | <2.5 Nm |
| Slippery | >70% | <3.0 Nm |
| Pushes | >75% | <3.2 Nm |
| Vel tracking | >85% | <2.8 Nm |
| Turning | >80% | <3.0 Nm |

**Exit Criteria:**
- ✓ All scenarios meet or exceed targets
- ✓ Results logged in WandB dashboard
- ✓ Video compilation of successful runs

#### 3.8.2: Performance Report
**Document Contents:**
1. Training curves (reward, success rate, torque over iterations)
2. Ablation study results
3. Hyperparameter search outcomes
4. Sim2sim transfer performance
5. Energy efficiency analysis (W·h per meter traveled)
6. Failure mode analysis (what causes falls?)

**Exit Criteria:**
- ✓ Report saved as `docs/phase3_results.md`
- ✓ Includes ablation study demonstrating AMP contribution

#### 3.8.3: Checkpoint Organization
```
checkpoints/
├── ppo_amp_foundation/       # Main training run
│   ├── iter_1000.pkl
│   ├── iter_3000.pkl
│   └── iter_5000.pkl
├── curriculum_training/       # After robustness training
│   └── iter_7000_final.pkl
├── optimized/                 # Hyperparameter tuning
│   └── best_config.pkl
├── ablation_studies/          # For analysis only
│   ├── no_amp.pkl
│   ├── no_torque_penalty.pkl
│   └── no_dr.pkl
└── deployment/
    ├── policy_fp32.onnx
    └── policy_int8.onnx
```

**Exit Criteria:**
- ✓ All checkpoints tagged with metadata (git hash, config, performance)
- ✓ Final policy promoted to `training/policies/wildrobot_v1.onnx`

---: PPO + AMP from Day 1
**Choice: PPO with Adversarial Motion Priors from initialization**
- **Industry Leaders:**
  - Unitree Go2/H1: Uses motion imitation for natural humanoid gaits
  - Figure 01: Reference motion tracking for manipulation + locomotion
  - Boston Dynamics: Motion libraries guide RL policies (Atlas backflips)
  - 1X Technologies NEO: Human motion retargeting to humanoid
  
- **Academic State-of-Art (2024-2025):**
  - **"Deep Whole-Body Control" (ETH Zurich, 2024):** AMP used from start, no baseline phase
  - **"Expressive Humanoid Robots" (CMU, 2024):** Motion priors critical for sim2real
  - **"Learning Agile Soccer Skills" (DeepMind, 2024):** Combines task RL + motion style from iteration 0
  
- **Why NOT baseline PPO first:**
  - Pure RL learns exploits (shuffling, knee-dragging) that must be unlearned
  - AMP provides strong inductive bias toward physically plausible motions
  - **Training time is actually faster** with AMP (fewer dead-end explorations)
  - Better sim2real transfer (human motion is real-world validated)
### 1. Algorithm Selection
**Choice: PPO + AMP**
- **Justification:** Industry standard for legged robots (Unitree Go2, ANYbotics ANYmal, Boston Dynamics Atlas RL research all use PPO-family algorithms)
- **State-of-art:** AMP introduced by DeepMind (2021) is now widely adopted. Recent work (ETH Zurich, 2024) shows AMP+PPO achieves zero-shot sim2real for quadrupeds.

### 2. Massive Parallelism (MJX/JAX)
**Choice: 4096 parallel environments on GPU**
- **Justification:** Google DeepMind's MJX (2023) enables 10-100x speedup vs CPU. Training time reduced from days to hours.
- **Comparison:** IsaacGym (2021) also uses GPU parallelism but is NVIDIA-specific. MJX is open-source and runs on any GPU (AMD, TPU).

### 3. Domain Randomization
**Parameters randomized:** friction, mass, damping, latency, external forces
- **Reference:** "Learning Quadrupedal Locomotion over Challenging Terrain" (ETH Zurich, 2020)
- **Modern best practice:** Use automatic domain randomization (ADR) to adaptively tune randomization ranges. Implementation: If sim2real gap is large, increase randomization variance.

### 4. Curriculum Learning
**Progressive difficulty:** Fixed velocity → Random velocity → Multi-directional → Terrain variation
- **Reference:** "Rapid Locomotion via Reinforcement Learning" (Berkley, 2023) - curriculum essential for complex tasks
- **Enhancement:** Auto-curriculum based on success rate (if >80%, increase difficulty)

### 5. Energy Efficiency Focus
**Torque penAMP Prevents Policy from Learning Task
**Likelihood:** Low (5-10% with proper reward balance)
**Symptoms:** AMP reward increases but robot doesn't move forward
**Mitigation:**
1. Ensure `amp_reward_weight` = 1.0, not higher (equal to task rewards)
2. If stuck after 1000 iters: temporarily reduce to 0.5, then ramp back up
3. Check reference motion is appropriate (walking, not standing/sitting)
4. Last resort: Train 500 iters without AMP, then enable (old baseline-first approach
- **Reference:** "Closing the Sim2Real Gap" (Google, 2024) - recommend testing in 2+ simulators
- **Implementation:** MJX → IsaacLab → Hardware

### 7. ONNX Deployment
**Framework-agnostic inference**
- **Justification:** ONNX Runtime enables deployment on embedded systems (Raspberry Pi, Jetson)
- **Optimization:** INT8 quantization for 2-4x speedup with minimal accuracy loss

---

## Risk Mitigation

### Risk 1: Policy Doesn't Learn to Walk
**Likelihood:** Medium (10-20% chance with complex humanoid)
**Mitigation:**
1. Start with Stage 3.2 (baseline PPO) to validate reward function
2. If no progress after 1000 iters → simplify task (fix velocity command, disable DR)
3. Check Phase 2 validation passed (torque budget sufficient)

### Risk 2: Sim2Real Gap Too Large
**Likelihood:** High (50%+ for first attempt)
**Mitigation:**
1. Extensive domain randomization (Step 3.4)
2. Sim2sim validation (Step 3.7) catches issues early
3. Plan for Phase 4: Sim2real fine-tuning with real-world data

### Risk 3: Training Too Slow
**Likelihood:** Low (MJX should achieve >1000 FPS)
**Mitigation:**
1. Profile code with `jax.profiler`
2. Reduce num_envs if GPU OOM
3. Use mixed precision (FP16) for 2x speedup

### Risk 4: Overfitting to Sim
**Likelihood:** Medium
**Mitigation:**
1. Aggressive domain randomization
2. AMP encourages natural motion (less likely to exploit sim artifacts)
3. Regularization: Entropy bonus, action smoothing penalties

---

## Success Criteria Summary

**Phase 3 Complete When:**
1. ✅ Policy walks 50m+ on flat terrain without falling (>90% success rate)
2. ✅ Average torque per joint <2.5Nm during steady-state walking
3.  | MoCap retargeting + discriminator | 50s of human walking data ready |
| 3-5 | PPO+AMP training | Natural walking with AMP reward >0.7 |
| 6 | Curriculum & robustnessward >0.7 (natural gait)
6. ✅ Sim2sim transfer: >60% success rate in IsaacLab
7. ✅ ONNX policy exported with <5ms inference latency
8. ✅ Comprehensive documentation and checkpoints organized

**Estimated Timeline:** 7-10 days with single RTX 3090 GPU

**Next Phase:** Phase 4 - Sim2Real Transfer (hardware deployment, online adaptation)

---

## Quick Reference: Daily Milestones

| Day | Milestone | Deliverable |
|-----|-----------|-------------|
| 1 | Environment setup | WildRobotEnv runs at >1500 FPS |
| 2-3 | Baseline PPO | Policy achieves 0.3 m/s walking |
| 4-5 | AMP integration | Natural gait with >0.7 AMP reward |
| 6 | Robustness training | Survives 10N pushes, varied terrain |
| 7 | Hyperparameter optimization | +5% performance improvement |
| 8 | Long-horizon stability | Walks 100m without drift |
| 9 | Sim2sim validation | Works in IsaacLab (>60% success) |
| 10 | Deployment prep | ONNX exported, safety wrapper tested |

**Total Compute:** ~100 GPU-hours (3-4 days wall-clock with background training)

---

## Appendix A: UV Package Management Best Practices

### Why UV?
- **Speed:** 10-100x faster than pip for dependency resolution
- **Reliability:** Deterministic, reproducible builds
- **Modern:** Built-in support for pyproject.toml, lockfiles

### Project Structure
```toml
# Add to pyproject.toml
[project.optional-dependencies]
rl = [
    "jax[cuda12_pip]",
    "mujoco>=3.0.0",
    "mujoco-mjx>=3.0.0",
    "brax==0.10.3",
    "wandb>=0.16.0",
    "tensorboard>=2.15.0",
    "onnx>=1.15.0",
    "onnxruntime>=1.16.0",
]

[tool.uv]
find-links = ["https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"]
```

### Common Commands
```bash
# Install all RL dependencies
uv pip install -e ".[rl]"

# Add new dependency
uv pip install numpy>=1.24.0

# Generate lockfile
uv pip compile pyproject.toml -o requirements-rl.lock

# Reproduce exact environment
uv pip sync requirements-rl.lock

# Update all packages
uv pip compile --upgrade pyproject.toml -o requirements-rl.lock
```

### CI/CD Integration
```yaml
# .github/workflows/train.yml
- name: Setup Python
  run: |
    curl -LsSf https://astral.sh/uv/install.sh | sh
    uv venv
    source .venv/bin/activate
    uv pip sync requirements-rl.lock
```

---

## Appendix B: Debugging Checklist

If training fails, check in order:

1. **Environment issues:**
   - [ ] Random actions run without NaN/Inf
   - [ ] Observations in reasonable range (-10, 10)
   - [ ] Termination conditions trigger correctly

2. **Reward function:**
   - [ ] Log all reward components separately
   - [ ] Check for reward hacking (e.g., spinning for velocity)
   - [ ] Torque values in expected range (0-4 Nm)

3. **Network architecture:**
   - [ ] Gradient norms not exploding (should be <10)
   - [ ] Value function converging (V-loss decreasing)
   - [ ] Policy entropy decaying gradually

4. **Hyperparameters:**
   - [ ] Learning rate not too high (try 1e-4 if 3e-4 fails)
   - [ ] Batch size sufficient (num_envs × num_steps >10k)
   - [ ] GAE lambda not too low (try 0.97)

5. **Domain randomization:**
   - [ ] Not too aggressive (if robot can't walk in any config, reduce variance)
   - [ ] Friction not too low (min should be >0.3)

**Emergency fallback:** Train on IsaacLab instead (proven stable, more examples available)
