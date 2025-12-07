# Isaac Lab Humanoid Training Strategy (Production-Ready)

## Overview
This strategy leverages Isaac Lab's mature ecosystem for training your MuJoCo-based humanoid robot with proven industry practices from Figure AI, Fourier Intelligence, Mentee Robotics, and academic leaders (Berkeley, ETH Zurich). The approach follows Isaac Lab's validated workflow with **joint task-style training from day one** using AMP.

**Key Advantages of Isaac Lab**:
- Mature, production-tested codebase (used by Figure 02, Unitree H1/G1)
- Built-in AMP support with extensive examples
- Excellent documentation and community support
- PhysX GPU acceleration (10,000+ parallel environments)
- Proven sim-to-real transfer pipeline

---

## Phase 0: Setup & MJCF → USD Conversion (Week 1)

### 0.1 Environment Setup

```bash
# Install Isaac Lab (requires Ubuntu 22.04+)
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab

# Run installer (includes Isaac Sim, dependencies)
./isaaclab.sh --install

# Verify installation
./isaaclab.sh -p source/standalone/tutorials/00_sim/create_empty.py
```

### 0.2 Convert Your MuJoCo Robot to USD

Isaac Lab provides a **built-in MJCF converter** that handles the conversion automatically.

**Step 1: Prepare Your MJCF File**
```bash
# Your MuJoCo robot structure should be:
/your_robot/
  ├── robot.xml          # Main MJCF file
  ├── meshes/            # STL/OBJ files
  └── textures/          # Optional
```

**Step 2: Convert Using Isaac Lab Script**
```bash
cd IsaacLab

# Convert your robot
./isaaclab.sh -p scripts/tools/convert_mjcf.py \
  /path/to/your_robot/robot.xml \
  source/extensions/omni.isaac.lab_assets/data/Robots/YourRobot/your_robot.usd \
  --import-sites \
  --make-instanceable
```

**Conversion Options**:
- `--import-sites`: Import MuJoCo site markers (useful for sensors)
- `--make-instanceable`: Required for parallel simulation (critical!)
- `--fix-base`: Set true for fixed-base robots
- `--import-inertia-tensor`: Default true (preserves MuJoCo dynamics)

**Step 3: Validate Conversion**
```bash
# Test the converted USD visually
./isaaclab.sh -p scripts/tools/visualize_robot.py \
  source/extensions/omni.isaac.lab_assets/data/Robots/YourRobot/your_robot.usd
```

**Important**: Press Play in Isaac Sim viewer. The robot should:
- ✅ Fall naturally under gravity (not explode)
- ✅ Have all joints visible and moving
- ✅ Collision meshes properly aligned

**If the robot explodes or behaves weirdly**:
```bash
# Common fixes:
1. Check self-collision in original MJCF (disable if needed)
2. Verify mass/inertia properties are realistic
3. Adjust collision margins in USD:
   collision_props=sim_utils.CollisionPropertiesCfg(
       contact_offset=0.02,  # Increase if penetrating
       rest_offset=0.0
   )
```

### 0.3 Create Robot Configuration

Create `source/extensions/omni.isaac.lab_assets/omni/isaac/lab_assets/your_robot.py`:

```python
"""Configuration for Your Custom Humanoid Robot."""

from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR
import omni.isaac.lab.sim as sim_utils

YOUR_ROBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/YourRobot/your_robot.usd",
        activate_contact_sensors=True,  # Critical for foot contacts
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,  # Usually false for humanoids
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.02,
            rest_offset=0.0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),  # Adjust to your robot's standing height
        joint_pos={
            # Set default standing pose from your MuJoCo model
            ".*hip_pitch": 0.0,
            ".*knee": 0.0,
            ".*ankle": 0.0,
            # Add all your joint patterns
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        # PD control for all joints (adjust gains from your MuJoCo tuning)
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*hip.*", ".*knee.*", ".*ankle.*"],
            stiffness=150.0,  # Tune from MuJoCo Kp
            damping=5.0,      # Tune from MuJoCo Kd
            effort_limit=300.0,
            velocity_limit=100.0,
        ),
        "torso": ImplicitActuatorCfg(
            joint_names_expr=["torso.*"],
            stiffness=100.0,
            damping=5.0,
            effort_limit=200.0,
            velocity_limit=100.0,
        ),
        # Add arms if your robot has them
    },
)
```

**Tuning PD Gains (MuJoCo → Isaac Lab)**:
```python
# If your MuJoCo XML has:
# <motor joint="knee" gear="100" ctrlrange="-1 1"/>
# with <joint damping="5"/>

# Isaac Lab equivalent:
stiffness = gear_ratio * some_factor  # Usually 100-300 for legs
damping = original_damping * 1.5      # PhysX needs slightly higher
```

**Validate Your Config**:
```bash
# Test robot spawning
./isaaclab.sh -p source/standalone/demos/articulation.py \
  --num_envs 10 \
  --robot YourRobot
```

---

## Phase 1: Foundation Training with AMP (0-8M steps, ~24-36 hours)

Isaac Lab's strength is **immediate AMP integration**. We start with joint task-style learning from step 1.

### 1.1 Prepare Motion Data

**Option A: Use CMU Mocap (Recommended for generic walking)**
```bash
cd IsaacLab

# Download CMU walking motions
wget https://github.com/isaac-sim/IsaacGymEnvs/raw/main/assets/amp_humanoid_walk.npy \
  -O source/extensions/omni.isaac.lab_tasks/data/amp/cmu_walk.npy

# Download additional motions
wget https://github.com/isaac-sim/IsaacGymEnvs/raw/main/assets/amp_humanoid_run.npy \
  -O source/extensions/omni.isaac.lab_tasks/data/amp/cmu_run.npy
```

**Option B: Convert Your Own Mocap (AMASS, SFU, etc.)**
```bash
# Isaac Lab includes retargeting tools
./isaaclab.sh -p scripts/tools/convert_fbx_to_npy.py \
  /path/to/your_motion.fbx \
  source/extensions/omni.isaac.lab_tasks/data/amp/custom_walk.npy \
  --robot-config YourRobotCfg
```

### 1.2 Create Training Environment

Create `source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/locomotion/your_robot_env.py`:

```python
"""Environment for Your Robot Locomotion with AMP."""

from __future__ import annotations

import torch
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.sensors import ContactSensorCfg

from omni.isaac.lab_assets.your_robot import YOUR_ROBOT_CFG

@configclass
class YourRobotEnvCfg(DirectRLEnvCfg):
    """Configuration for your robot locomotion environment."""
    
    # Simulation settings
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 50,  # 50 Hz control
        physics_material=...,  # Configure friction
    )
    
    # Environment settings
    episode_length_s: float = 20.0  # 20 seconds per episode
    decimation: int = 2  # Action every 2 sim steps
    num_observations: int = 48  # Adjust based on your obs space
    num_actions: int = 19  # Number of actuated joints
    
    # Robot
    robot: ArticulationCfg = YOUR_ROBOT_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )
    
    # Contact sensors (critical for AMP)
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*_foot",
        update_period=0.0,  # Update every step
        track_air_time=True,
    )
    
    # Terrain
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",  # Start with flat
        physics_material=...,
    )
    
    # Commands (velocity targets)
    commands: CommandsCfg = CommandsCfg(
        base_velocity=UniformVelocityCommandCfg(
            asset_name="robot",
            heading_command=True,
            resampling_time_range=(5.0, 10.0),
            ranges=VelocityCommandRanges(
                lin_vel_x=(0.5, 1.5),  # Forward velocity
                lin_vel_y=(-0.0, 0.0),  # No lateral initially
                ang_vel_z=(-0.0, 0.0),  # No yaw initially
            ),
        )
    )
    
    # Rewards (AMP + Task)
    rewards: RewardsCfg = RewardsCfg(
        # Task rewards (40%)
        velocity_tracking=RewTerm(
            func=track_lin_vel_xy_exp,
            weight=4.0,
            params={"std": 0.5},
        ),
        upright_posture=RewTerm(
            func=base_orientation_penalty,
            weight=3.0,
        ),
        
        # Style rewards via AMP (40%)
        amp_discriminator=RewTerm(
            func=amp_reward,
            weight=4.0,
            params={
                "motion_files": ["cmu_walk.npy"],
                "state_features": ["joint_pos", "joint_vel", "foot_contacts"],
            },
        ),
        
        # Energy & smoothness (20%)
        energy_cost=RewTerm(
            func=power_consumption,
            weight=1.0e-4,
        ),
        action_smoothness=RewTerm(
            func=action_rate_l2,
            weight=0.01,
        ),
        
        # Termination penalties
        termination_penalty=RewTerm(
            func=is_terminated,
            weight=-200.0,
        ),
    )
    
    # Termination conditions
    terminations: TerminationsCfg = TerminationsCfg(
        base_contact=DoneTerm(
            func=illegal_contact,
            params={"sensor_cfg": SceneEntityCfg("contact_sensor"),
                    "body_names": [".*torso", ".*pelvis"]},
        ),
        base_height=DoneTerm(
            func=base_height_below_threshold,
            params={"minimum_height": 0.3},
        ),
    )


class YourRobotEnv(DirectRLEnv):
    """Environment for your robot locomotion."""
    
    cfg: YourRobotEnvCfg
    
    def __init__(self, cfg: YourRobotEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)
        
        # Load AMP discriminator
        self._setup_amp_discriminator()
    
    def _setup_amp_discriminator(self):
        """Initialize AMP discriminator from motion data."""
        # Isaac Lab provides built-in AMP utilities
        from omni.isaac.lab.utils.amp import AMPDiscriminator
        
        self.amp_disc = AMPDiscriminator(
            motion_files=self.cfg.rewards.amp_discriminator.params["motion_files"],
            state_dim=...,  # Auto-compute from features
            hidden_dims=[256, 128],
            device=self.device,
        )
    
    def _get_observations(self) -> dict:
        """Compute observations."""
        obs = torch.cat([
            # Proprioception
            self.robot.data.root_lin_vel_b,  # Base velocity (3)
            self.robot.data.root_ang_vel_b,  # Base angular velocity (3)
            self.robot.data.projected_gravity_b,  # Gravity vector (3)
            self.robot.data.joint_pos - self.robot.data.default_joint_pos,  # Joint pos (19)
            self.robot.data.joint_vel,  # Joint vel (19)
            self.commands["base_velocity"],  # Velocity commands (3)
        ], dim=-1)
        
        return {"policy": obs}
    
    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards."""
        # Task rewards computed automatically from RewardsCfg
        task_rewards = super()._get_rewards()
        
        # AMP reward computed separately
        amp_reward = self.amp_disc.compute_reward(
            self._get_amp_observations()
        )
        
        return task_rewards + amp_reward
    
    def _get_amp_observations(self) -> torch.Tensor:
        """Get state features for AMP discriminator."""
        return torch.cat([
            self.robot.data.joint_pos,
            self.robot.data.joint_vel,
            self.contact_sensor.data.net_forces_w_history[:, :, 2],  # Vertical forces
        ], dim=-1)
```

### 1.3 Training Configuration

Create `source/extensions/omni.isaac.lab_tasks/config/your_robot/agents/rsl_rl_ppo_cfg.py`:

```python
"""PPO configuration for your robot."""

from omni.isaac.lab.utils import configclass
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlPpoActorCriticCfg

@configclass
class YourRobotPPORunnerCfg:
    """Configuration for PPO runner."""
    
    seed = 42
    runner_type = "OnPolicyRunner"
    
    # Parallelization
    num_steps_per_env = 24  # Horizon length
    max_iterations = 5000  # ~8M steps with 4096 envs
    
    # PPO hyperparameters
    policy = RslRlPpoActorCriticCfg(
        # Network architecture
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        
        # Learning rates
        learning_rate=3e-4,
        schedule="adaptive",  # Decay when KL diverges
        
        # PPO specific
        clip_param=0.2,
        desired_kl=0.01,
        entropy_coef=0.01,
        value_loss_coef=1.0,
        
        # Training
        num_mini_batches=4,
        num_learning_epochs=5,
        gamma=0.99,
        lam=0.95,
        
        # AMP discriminator
        amp_reward_coef=1.0,  # Weight of AMP vs task
        amp_discriminator_hidden_dims=[256, 128],
        amp_discriminator_learning_rate=3e-4,
        amp_gradient_penalty_coef=5.0,  # CRITICAL: tune [0.1, 10]
    )
    
    # Logging
    save_interval = 100  # Save every 100 iterations
    log_interval = 10
    
    # Device
    device = "cuda:0"
```

**Critical AMP Hyperparameter: `amp_gradient_penalty_coef`**

This is the **single most important** hyperparameter for AMP training:
- Too high (>10): Policy can't deviate from reference, gets stuck
- Too low (<0.1): Style degrades, unnatural motion
- Start: 5.0
- If motion quality poor: Decrease to 2.0 → 1.0 → 0.5
- If task performance poor: Increase to 7.0 → 10.0

### 1.4 Register and Launch Training

Edit `source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/__init__.py`:

```python
gym.register(
    id="Isaac-YourRobot-Flat-v0",
    entry_point="omni.isaac.lab_tasks.direct.locomotion:YourRobotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": YourRobotEnvCfg,
        "rsl_rl_cfg_entry_point": YourRobotPPORunnerCfg,
    },
)
```

**Launch Training:**
```bash
# With 4096 parallel environments (adjust based on GPU memory)
./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py \
  --task Isaac-YourRobot-Flat-v0 \
  --num_envs 4096 \
  --headless

# With visualization (slower, good for debugging)
./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py \
  --task Isaac-YourRobot-Flat-v0 \
  --num_envs 512 \
  --video \
  --video_interval 500
```

### 1.5 Monitor Training (Real-Time)

Isaac Lab integrates with **Weights & Biases** (W&B) and **TensorBoard**:

```bash
# TensorBoard (local)
tensorboard --logdir logs/rsl_rl/your_robot_flat

# Weights & Biases (better for remote)
# Add to your training script:
wandb login
# Automatic logging is built-in
```

**Key Metrics to Track:**
```python
metrics = {
    # Task performance
    "Episode/rew_velocity_tracking": target > 3.0,
    "Episode/rew_upright_posture": target > 2.5,
    "Episode/episode_length": target > 800,  # ~16 seconds
    
    # AMP quality
    "Episode/rew_amp_discriminator": target > 2.5,  # >0.6 after normalization
    "Train/amp_discriminator_loss": "should decrease",
    "Train/amp_grad_penalty": target ~ 1.0,
    
    # Learning health
    "Train/learning_rate": "should decay gradually",
    "Train/kl": target < 0.02,  # Stay below threshold
    "Train/policy_loss": "should decrease",
    
    # Robustness
    "Episode/termination_rate": target < 0.20,  # <20% falls
    "Episode/base_height": target ~ 1.0,  # Standing height
}
```

### 1.6 Success Criteria for Phase 1

Progress to Phase 2 when:
- ✅ **Velocity tracking RMSE** < 0.20 m/s
- ✅ **Episode length** > 800 steps (16s @ 50Hz)
- ✅ **AMP discriminator reward** > 2.5
- ✅ **Termination rate** < 20%
- ✅ **Visual quality**: Motion looks human-like (watch videos)

**Expected Timeline**: 5-8M steps, 24-36 GPU-hours on A100

---

## Phase 2: Robust Locomotion (8-20M steps, ~48-72 hours)

### 2.1 Terrain Curriculum

Isaac Lab provides **procedurally generated terrains**:

```python
# Update your environment config
terrain: TerrainImporterCfg = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="generator",
    terrain_generator=ROUGH_TERRAINS_CFG,  # Built-in
    curriculum=True,  # Gradually increase difficulty
)

# Terrain progression (automatic with curriculum=True)
ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": SubTerrainCfg(proportion=0.4),  # 40% flat initially
        "random_rough": SubTerrainCfg(proportion=0.2, random_rough_cfg=...),
        "slope": SubTerrainCfg(proportion=0.2, slope_cfg=...),
        "stairs": SubTerrainCfg(proportion=0.1, stairs_cfg=...),
        "boxes": SubTerrainCfg(proportion=0.1, box_cfg=...),
    },
)
```

### 2.2 Domain Randomization Schedule

```python
# Add to YourRobotEnvCfg
domain_rand: DomainRandomizationCfg = DomainRandomizationCfg(
    # Dynamics randomization
    randomize_rigids_after_start=True,
    rigid_props=RigidPropertiesRandomCfg(
        num_buckets=250,
        mass_range=(0.8, 1.2),  # ±20%
        com_range=(-0.05, 0.05),  # COM offset
    ),
    
    # Actuator randomization
    actuator_props=ActuatorRandomCfg(
        stiffness_range=(0.85, 1.15),
        damping_range=(0.85, 1.15),
        effort_range=(0.90, 1.10),
    ),
    
    # External perturbations
    push_robot=PushRobotCfg(
        interval_range_s=(2.0, 5.0),  # Push every 2-5 seconds
        velocity_range=(0.0, 1.0),  # Push strength increases
    ),
    
    # Observation noise
    add_obs_noise=True,
    obs_noise=ObservationNoiseCfg(
        joint_pos_std=0.01,  # 0.01 rad
        joint_vel_std=0.5,   # 0.5 rad/s
        ang_vel_std=0.05,    # 0.05 rad/s
        lin_vel_std=0.1,     # 0.1 m/s
    ),
)
```

### 2.3 Multi-Directional Commands

```python
# Expand command ranges
commands: CommandsCfg = CommandsCfg(
    base_velocity=UniformVelocityCommandCfg(
        resampling_time_range=(5.0, 10.0),
        ranges=VelocityCommandRanges(
            lin_vel_x=(0.0, 2.0),      # Include backwards (negative)
            lin_vel_y=(-0.4, 0.4),     # Lateral walking
            ang_vel_z=(-0.5, 0.5),     # Turning
            heading=(-3.14, 3.14),     # Full rotation
        ),
    )
)
```

### 2.4 Update Training Config for Phase 2

```bash
# Resume from Phase 1 checkpoint
./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py \
  --task Isaac-YourRobot-Rough-v0 \  # New terrain config
  --num_envs 4096 \
  --resume \
  --load_run logs/rsl_rl/your_robot_flat/YYYY-MM-DD_HH-MM-SS \
  --checkpoint model_5000.pt  # Load from Phase 1 peak
```

### 2.5 Success Criteria for Phase 2

- ✅ **Velocity tracking on rough terrain** < 0.25 m/s RMSE
- ✅ **Performance drop vs flat** < 15%
- ✅ **Multi-directional success** > 80% (forward/backward/lateral/turn)
- ✅ **Push recovery** > 85% (returns to stable gait within 5s)
- ✅ **AMP reward maintained** > 2.0 (some degradation acceptable)

**Expected Timeline**: 12M additional steps, 48-72 GPU-hours

---

## Phase 3: Sim-to-Real Preparation (Parallel to Phase 2)

### 3.1 Sim2Sim Validation (Critical!)

Before touching real hardware, validate in different physics engines:

```bash
# Export policy to ONNX
./isaaclab.sh -p source/standalone/workflows/rsl_rl/export_onnx.py \
  --load_run logs/rsl_rl/your_robot_rough/... \
  --checkpoint model_best.pt

# Test in MuJoCo (your original env)
# This tests if PhysX → MuJoCo transfer works
python scripts/validate/test_policy_mujoco.py \
  --policy policy.onnx \
  --robot your_robot.xml \
  --episodes 100

# Success threshold: >85% performance retained
```

**If sim2sim fails (<70% performance)**:
1. Check PD gains match (PhysX vs MuJoCo)
2. Verify contact dynamics (friction, restitution)
3. Increase domain randomization in Phase 2
4. Add more observation noise

### 3.2 System Identification (Real Hardware Required)

```python
# Measure actual robot parameters
system_id = {
    "control_latency": measure_control_loop(),  # Typically 5-20ms
    "observation_delay": measure_sensor_delay(),  # Typically 2-10ms
    "actual_masses": weigh_each_link(),
    "friction_coefficients": sliding_tests(),
    "motor_characteristics": torque_vs_current_test(),
}

# Update sim to match reality
sim_params.update({
    "action_delay_steps": int(system_id["control_latency"] * sim_freq),
    "obs_delay_steps": int(system_id["observation_delay"] * sim_freq),
    "mass_scale_factors": system_id["actual_masses"] / nominal_masses,
    # ... update all measured params
})

# Retrain final policy with corrected parameters
```

### 3.3 Safety Layers for Deployment

```python
class SafetyWrapper:
    """Real-world safety checks."""
    
    def __init__(self, policy):
        self.policy = policy
        self.emergency_stop = False
        
        # Safety limits (more conservative than training)
        self.joint_limits = {
            "position": (lower * 0.95, upper * 0.95),  # 5% margin
            "velocity": max_vel * 0.85,
            "torque": max_torque * 0.85,
        }
        
        self.base_limits = {
            "min_height": 0.4,  # Crouch detection
            "max_tilt": 25.0,  # degrees
        }
    
    def get_action(self, obs):
        """Get action with safety checks."""
        # Check for emergency conditions
        if self._check_emergency(obs):
            return self._safe_shutdown_action()
        
        # Get policy action
        action = self.policy(obs)
        
        # Clip to safe ranges
        action = self._clip_action(action)
        
        # Gradual action changes (no sudden jumps)
        action = self._smooth_action(action, self.last_action)
        
        self.last_action = action
        return action
    
    def _check_emergency(self, obs):
        """Detect dangerous states."""
        base_height = obs["base_height"]
        base_tilt = obs["base_orientation_euler"]
        
        if base_height < self.base_limits["min_height"]:
            return True  # Too low, about to fall
        
        if abs(base_tilt).max() > self.base_limits["max_tilt"]:
            return True  # Excessive tilt
        
        # Check for unexpected collisions
        if obs["contact_forces"]["torso"] > 0:
            return True  # Torso shouldn't touch ground
        
        return False
    
    def _safe_shutdown_action(self):
        """Controlled collapse to prevent damage."""
        # Damped fall: high damping, zero stiffness
        return torch.zeros_like(self.last_action)
```

### 3.4 Deployment Progression

**Week 1: Suspended Testing**
```python
# Robot suspended from ceiling/harness
# Validate policy runs on real hardware
# Check for timing issues, NaN values, crashes
test_suspended(episodes=100)
```

**Week 2: Foam Mat Testing**
```python
# Robot on thick foam mats (minimize damage from falls)
# Start with very conservative policies
# Gradually increase confidence
test_foam_mat(episodes=200, max_velocity=0.5)
```

**Week 3: Flat Ground Validation**
```python
# Hard floor, controlled environment
# Start slow, ramp up velocity
test_flat_ground(episodes=500, velocity_range=(0.3, 1.0))
```

**Week 4+: Outdoor Challenging Terrain**
```python
# Only after flat ground is robust
test_outdoor(terrain_types=["grass", "gravel", "mild_slopes"])
```

---

## Phase 4: Advanced Skills (Optional, 20-30M steps)

### 4.1 Multi-Skill Conditioning

If you need specialized behaviors beyond basic walking:

```python
# Update environment with skill encoding
class MultiSkillYourRobotEnv(YourRobotEnv):
    """Environment with multiple locomotion skills."""
    
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        
        # Define skills
        self.skills = {
            "walk_normal": {"motion_file": "cmu_walk.npy", "weight": 0.4},
            "walk_fast": {"motion_file": "cmu_run_slow.npy", "weight": 0.2},
            "walk_careful": {"motion_file": "cmu_walk_slow.npy", "weight": 0.2},
            "recovery": {"motion_file": "recovery_motions.npy", "weight": 0.2},
        }
        
        # Skill conditioning vector (one-hot or learned embedding)
        self.skill_encoding_dim = 8
    
    def _get_observations(self):
        """Add skill encoding to observations."""
        base_obs = super()._get_observations()
        skill_vec = self._get_current_skill_encoding()
        
        return torch.cat([base_obs, skill_vec], dim=-1)
```

### 4.2 Transformer Policy (For Long Context)

```python
# Update policy architecture for temporal reasoning
policy = RslRlPpoActorCriticCfg(
    actor_type="transformer",  # Instead of MLP
    context_length=64,  # Past 64 observations
    num_attention_heads=4,
    hidden_dims=[256],
    feedforward_dims=[1024],
    
    # Rest same as before
    learning_rate=3e-4,
    ...
)
```

---

## Common Issues & Solutions

### Issue 1: Robot Explodes on Spawn
**Symptoms**: Robot flies away or vibrates violently when simulation starts

**Solutions**:
```bash
# 1. Check collision meshes
./isaaclab.sh -p scripts/tools/inspect_usd.py your_robot.usd --check-collisions

# 2. Reduce physics timestep
sim.dt = 1/200  # Instead of 1/50

# 3. Increase solver iterations
articulation_props.solver_position_iteration_count = 8  # Instead of 4

# 4. Check for mass/inertia issues
# Look for very small or very large inertias in USD
```

### Issue 2: Training Reward Collapse
**Symptoms**: Reward suddenly drops to near zero and never recovers

**Solutions**:
```python
# 1. Reduce AMP gradient penalty
amp_gradient_penalty_coef = 2.0  # From 5.0

# 2. Increase entropy bonus
entropy_coef = 0.02  # From 0.01

# 3. Reduce learning rate
learning_rate = 1e-4  # From 3e-4

# 4. Check discriminator isn't overpowering
amp_reward_coef = 0.5  # From 1.0
```

### Issue 3: Policy "Freezes" (Learns to Stand Still)
**Symptoms**: High reward but zero forward velocity

**Solutions**:
```python
# 1. Increase velocity tracking weight
velocity_tracking.weight = 6.0  # From 4.0

# 2. Add forward progress bonus
forward_progress = RewTerm(
    func=forward_displacement,
    weight=2.0,
)

# 3. Penalize standing still
velocity_penalty = RewTerm(
    func=penalize_zero_velocity,
    weight=-1.0,
)
```

### Issue 4: Unnatural Gait Despite High AMP Score
**Symptoms**: AMP discriminator happy but motion looks weird

**Solutions**:
```python
# 1. Check your reference motions are correct
# Visualize them:
./isaaclab.sh -p scripts/tools/visualize_motion.py cmu_walk.npy

# 2. Increase AMP gradient penalty
amp_gradient_penalty_coef = 7.0  # From 5.0

# 3. Add explicit contact rewards
foot_contact_consistency = RewTerm(
    func=reward_foot_contact_pattern,
    weight=1.0,
)

# 4. Use multiple reference motions
amp_discriminator.params["motion_files"] = [
    "cmu_walk_01.npy",
    "cmu_walk_02.npy",
    "cmu_walk_03.npy",
]
```

### Issue 5: Sim2Sim Transfer Fails
**Symptoms**: Policy works great in PhysX but fails in MuJoCo/real robot

**Solutions**:
```python
# 1. Match PD gains exactly
# Export PhysX gains:
print(robot.actuators["legs"].stiffness)
print(robot.actuators["legs"].damping)

# Set in MuJoCo XML:
<motor joint="knee" gear="150" ctrlrange="-1 1"/>
<joint name="knee" damping="5.0"/>

# 2. Increase observation noise in training
obs_noise.joint_pos_std = 0.02  # From 0.01
obs_noise.joint_vel_std = 1.0   # From 0.5

# 3. Add action delays in training
action_delay_steps = 2  # Simulate real control latency

# 4. Retrain with wider domain randomization
mass_range = (0.7, 1.3)  # From (0.8, 1.2)
```

---

## Computational Requirements

### Minimum (Workstation)
```yaml
hardware:
  gpu: RTX 3090 (24GB)
  cpu: 16 cores
  ram: 32GB

performance:
  num_envs: 2048
  steps_per_hour: ~250k
  phase_1_time: 32 hours
  phase_2_time: 96 hours
  total_cost: $0 (your hardware)
```

### Recommended (Cloud)
```yaml
hardware:
  gpu: A100 (40GB) or H100
  cpu: 32 cores
  ram: 64GB

performance:
  num_envs: 4096
  steps_per_hour: ~500k
  phase_1_time: 16 hours
  phase_2_time: 48 hours
  total_cost: $400-800 (AWS/GCP)
```

### Optimal (Research Lab)
```yaml
hardware:
  gpu: 4x A100 (80GB)
  cpu: 64 cores
  ram: 256GB

performance:
  num_envs: 16384
  steps_per_hour: ~2M
  phase_1_time: 4 hours
  phase_2_time: 12 hours
  total_cost: Priceless
```

---

## Metrics Dashboard Configuration

### TensorBoard Logging
```python
# Auto-configured in Isaac Lab
# View at: http://localhost:6006
tensorboard --logdir logs/rsl_rl
```

### Weights & Biases Setup
```python
# Add to train.py
import wandb

wandb.init(
    project="humanoid-locomotion",
    name=f"your_robot_{cfg.task}",
    config={
        "num_envs": cfg.num_envs,
        "amp_weight": cfg.policy.amp_reward_coef,
        "terrain": cfg.terrain.terrain_type,
    },
    tags=["phase1", "flat", "amp"],
)

# Automatic logging of all metrics
# Custom metrics:
wandb.log({
    "custom/gait_frequency": compute_gait_freq(),
    "custom/step_length": compute_step_length(),
})
```

### Real-Time Alerts
```python
# Add alert thresholds
if episode_reward < 50:  # Should be >150
    wandb.alert(
        title="Training Collapse",
        text=f"Reward dropped to {episode_reward}",
        level="ERROR",
    )

if termination_rate > 0.5:  # Should be <0.2
    wandb.alert(
        title="High Fall Rate",
        text=f"Termination rate: {termination_rate:.2%}",
        level="WARNING",
    )
```

---

## Quick Start Checklist

### Day 1: Setup
- [ ] Install Isaac Lab + dependencies
- [ ] Convert MuJoCo MJCF to USD
- [ ] Validate USD loads correctly
- [ ] Test spawning 100 parallel robots

### Day 2: Configuration
- [ ] Create robot config file
- [ ] Implement environment class
- [ ] Download/prepare motion data
- [ ] Test single environment run

### Day 3: Initial Training
- [ ] Launch Phase 1 training (4096 envs)
- [ ] Verify TensorBoard/W&B logging
- [ ] Monitor for crashes/NaNs
- [ ] Save checkpoints every 500k steps

### Week 1: Phase 1 Completion
- [ ] Achieve velocity tracking <0.20 m/s
- [ ] Validate AMP reward >2.5
- [ ] Visual inspection: motion looks natural
- [ ] Export best checkpoint

### Week 2: Phase 2 Launch
- [ ] Add terrain generation
- [ ] Enable domain randomization
- [ ] Resume from Phase 1 checkpoint
- [ ] Monitor robustness metrics

### Week 3-4: Sim2Sim Validation
- [ ] Export policy to ONNX
- [ ] Test in MuJoCo (your original)
- [ ] Validate >85% performance retained
- [ ] If fail: tune gains, increase DR

### Week 5: Hardware Prep (If Available)
- [ ] Implement safety wrapper
- [ ] Test policy on suspended robot
- [ ] Measure system latencies
- [ ] Update sim parameters

### Week 6+: Real-World Deployment
- [ ] Foam mat tests (200 episodes)
- [ ] Flat ground validation (500 episodes)
- [ ] Progressive terrain difficulty
- [ ] Iterate based on failures

---

## Summary: Key Success Factors

### ✅ Do These Things
1. **Start with AMP from day 1** - Don't train pure RL first
2. **Use 4000+ environments** - Parallelization is critical
3. **Validate conversion early** - Test USD before long training
4. **Monitor discriminator** - AMP reward should be >2.5
5. **Sim2sim before real** - MuJoCo validation is mandatory
6. **Progressive curriculum** - Don't jump to hard terrain
7. **Save checkpoints often** - Every 500k steps minimum

### ❌ Avoid These Mistakes
1. ❌ Training on CPU with <1000 envs
2. ❌ Skipping USD validation step
3. ❌ Using only flat terrain (no generalization)
4. ❌ Testing on real robot without sim2sim
5. ❌ Ignoring observation noise/delays
6. ❌ Setting amp_gradient_penalty too high (>10)
7. ❌ Not watching training videos regularly

---

## Expected Outcomes Timeline

| Week | Milestone | Success Metric |
|------|-----------|----------------|
| 1 | Phase 1 training | Forward walking @ 1.0 m/s |
| 2 | Phase 1 complete | Natural gait, AMP >2.5 |
| 3 | Phase 2 launch | Robust to pushes |
| 4 | Phase 2 complete | Multi-terrain success |
| 5 | Sim2sim validation | >85% MuJoCo transfer |
| 6 | First real steps | Walking on flat ground |
| 8 | Deployment ready | Outdoor robust locomotion |

---

## Additional Resources

### Isaac Lab Documentation
- Official Docs: https://isaac-sim.github.io/IsaacLab/
- Tutorials: https://isaac-sim.github.io/IsaacLab/source/tutorials/
- API Reference: https://isaac-sim.github.io/IsaacLab/source/api/

### Example Implementations
- Unitree H1: `source/extensions/omni.isaac.lab_tasks/.../unitree_h1/`
- Unitree G1: `source/extensions/omni.isaac.lab_tasks/.../unitree_g1/`
- Anymal-D: `source/extensions/omni.isaac.lab_tasks/.../anymal_d/`

### Community
- Isaac Lab GitHub: https://github.com/isaac-sim/IsaacLab
- Discord: NVIDIA Omniverse Server
- Forums: https://forums.developer.nvidia.com/c/omniverse/isaac-sim/

### Papers (Implementation References)
- AMP (2021): "Adversarial Motion Priors for Stylized Physics-Based Character Control"
- ASE (2022): "Learning Robust Skills with Imitation Games"
- ProtoMotions (2023): "Prototyping Motion Planning with Predictive Models"

---

## Appendix A: Full Training Command Reference

```bash
# Phase 1: Flat terrain with AMP
./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py \
  --task Isaac-YourRobot-Flat-v0 \
  --num_envs 4096 \
  --headless \
  --enable_cameras false \
  --max_iterations 5000

# Phase 2: Rough terrain with DR
./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py \
  --task Isaac-YourRobot-Rough-v0 \
  --num_envs 4096 \
  --headless \
  --resume \
  --load_run logs/rsl_rl/your_robot_flat/2025-01-01_00-00-00 \
  --checkpoint model_5000.pt

# Evaluation (with video recording)
./isaaclab.sh -p source/standalone/workflows/rsl_rl/play.py \
  --task Isaac-YourRobot-Rough-v0 \
  --num_envs 64 \
  --load_run logs/rsl_rl/your_robot_rough/2025-01-15_00-00-00 \
  --checkpoint model_best.pt \
  --video \
  --video_length 500

# Export to ONNX
./isaaclab.sh -p source/standalone/workflows/rsl_rl/export.py \
  --task Isaac-YourRobot-Rough-v0 \
  --load_run logs/rsl_rl/your_robot_rough/2025-01-15_00-00-00 \
  --checkpoint model_best.pt \
  --export_path policy.onnx
```

---

## Appendix B: Reward Function Implementation Examples

```python
def track_lin_vel_xy_exp(env, std: float = 0.5):
    """Exponential velocity tracking reward."""
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command("base_velocity")[:, :2] 
                    - env.robot.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    return torch.exp(-lin_vel_error / std**2)


def base_orientation_penalty(env):
    """Penalize deviation from upright orientation."""
    # Projected gravity should point down (0, 0, -1)
    gravity_proj = env.robot.data.projected_gravity_b
    upright_penalty = torch.sum(torch.square(gravity_proj[:, :2]), dim=1)
    return torch.exp(-upright_penalty * 10.0)


def power_consumption(env):
    """Penalize mechanical power (torque × velocity)."""
    power = torch.sum(
        torch.abs(env.robot.data.applied_torque * env.robot.data.joint_vel),
        dim=1,
    )
    # Normalize by typical walking power (~100-200W for humanoid)
    return -power / 200.0


def action_rate_l2(env):
    """Penalize rapid action changes."""
    if env.last_actions is None:
        return torch.zeros(env.num_envs, device=env.device)
    return -torch.sum(
        torch.square(env.actions - env.last_actions),
        dim=1,
    )


def reward_foot_contact_pattern(env):
    """Reward alternating foot contacts."""
    left_contact = env.contact_sensor.data.net_forces_w[:, 0, 2] > 1.0
    right_contact = env.contact_sensor.data.net_forces_w[:, 1, 2] > 1.0
    
    # Reward when exactly one foot is in contact (not both, not neither)
    alternating = (left_contact ^ right_contact).float()
    return alternating
```

---

**End of Strategy Document**

*This strategy is based on Isaac Lab v1.2+ (January 2025). For the latest updates, always check the official documentation.*