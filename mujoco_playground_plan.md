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
