# WildRobot Test Strategy (v2)

## Executive Summary

This document defines a comprehensive, stage-gated test strategy for the WildRobot project, enabling confident progression from MuJoCo model design through PPO + AMP training and ultimately sim-to-real deployment.

This version explicitly introduces **robot_config.yaml as the single runtime source of truth**, generated deterministically from `wildrobot.xml`, and defines mandatory **XML ↔ robot_config consistency contracts**.

---

## Core Principles

1. **Single Source of Truth at Runtime**
   - All product code and test code MUST consume `robot_config.yaml`
   - Direct parsing of `wildrobot.xml` is only allowed inside the generator and its tests

2. **Authoritative Authoring Artifact**
   - `wildrobot.xml` is the authoritative model definition
   - Any semantic change to the robot must flow through regeneration of `robot_config.yaml`

3. **Fail Fast on Structural Drift**
   - XML/config inconsistencies are treated as hard errors
   - No training, reward, or AMP tests run if the contract fails


## Non-negotiables

### 1) No hard-coded indices
Never hard-code qpos or qvel slices, geom IDs, site IDs, or actuator indices in reward, observations, features, or tests.

Instead:
- derive indices from the MuJoCo model once,
- store them in a **schema contract artifact**,
- assert contract stability in CI,
- use the contract everywhere (env, reward, AMP features, MJX parity).

### 2) Determinism and reproducibility
Every training and validation run must record:
- git commit hash
- config hash
- model hash (xml and referenced assets)
- random seed
- MuJoCo version
- JAX and XLA versions
- device type

### 3) Tests are stage-gated
A test is not "nice to have." It is a gate for moving to the next stage:
- Stage 1: PPO task learning
- Stage 2: dataset generation
- Stage 3: AMP
- Stage 4+: style blending and sim-to-real

---

## Scope and artifacts under test

**Inputs**
- `assets/wildrobot.xml` (MuJoCo model), referenced meshes and materials
- environment scene XML (e.g. `assets/scene_flat_terrain.xml`)
- training config (`playground_amp/configs/ppo_walking.yaml`)
- training and rollout code (`playground_amp/*`)

**Outputs**
- schema contract JSON
- trained checkpoints
- evaluation rollouts and metrics
- reference dataset segments (Stage 2)
- sim-to-real calibration pack (mass, friction, actuator limits, sensor noise)

---

## Test tiers and CI schedule

| Tier | When | Runtime | Purpose |
|---|---:|---:|---|
| T0: Pre-commit | local | < 2 min | fast unit and schema checks |
| T1: PR gate | CI | 10-20 min | simulation correctness and small training smoke |
| T2: Nightly | CI | 1-3 hours | MJX parity, longer training, robustness sweeps |
| T3: Release | manual | half day | sim-to-real readiness pack and hardware checklist |

Recommended pytest markers:
- `@pytest.mark.unit` (T0)
- `@pytest.mark.sim` (T1)
- `@pytest.mark.train` (T1/T2)
- `@pytest.mark.mjx` (T2)
- `@pytest.mark.robust` (T2/T3)
- `@pytest.mark.s2r` (T3)

---

## Layer 0: Schema Contract (Foundation)

### Objective
Guarantee consistency between `wildrobot.xml` and `robot_config.yaml` and enforce config-only runtime access.

### 0.1 Generation and Snapshot Tests

**Test 0.1.1: Deterministic Generation**
```python
def test_deterministic_generation(self):
    """
    Purpose: Verify schema generation is deterministic.
    
    Procedure:
    1. Generate robot_config.yaml twice from the same XML
    2. Compare outputs
    
    Assertions:
    - Byte-identical or semantically identical output
    """
    config1 = generate_robot_config("assets/wildrobot.xml")
    config2 = generate_robot_config("assets/wildrobot.xml")
    assert config1 == config2, "Generation not deterministic"
```

**Test 0.1.2: Snapshot Consistency**
```python
def test_snapshot_consistency(self):
    """
    Purpose: Detect unintentional XML changes.
    
    Procedure:
    1. Regenerate config from current XML
    2. Diff against committed robot_config.yaml
    
    Assertions:
    - Any difference fails with actionable diff message
    """
    regenerated = generate_robot_config("assets/wildrobot.xml")
    committed = load_yaml("assets/robot_config.yaml")
    assert regenerated == committed, (
        "Schema mismatch! XML has changed. "
        "Run `python scripts/generate_config.py` if intentional."
    )
```

**Test 0.1.3: Provenance Validation**
```python
def test_provenance_fields(self, robot_config):
    """
    Purpose: Ensure config contains provenance metadata.
    
    Assertions:
    - source_xml_sha256 exists and is 64 hex chars
    - generator_version exists
    - schema_version exists
    """
    assert "source_xml_sha256" in robot_config
    assert len(robot_config["source_xml_sha256"]) == 64
    assert "generator_version" in robot_config
    assert "schema_version" in robot_config
```

### 0.2 Contract Content (Minimum Required)

Derived from MuJoCo `MjModel` created from `assets/wildrobot.xml`:

#### 0.2.1 Joints (Real Joints Only)

| Field | Description |
|-------|-------------|
| `name` | Joint name (excluding `_mimic`) |
| `type` | Joint type (hinge, ball, etc.) |
| `qpos_adr` | Address in qpos array |
| `dof_adr` | Address in qvel/dof array |
| `range` | Joint limits [min, max] |
| `limited` | Whether limits are enforced |
| `body_name` | Parent body name |

**Test 0.2.1: No Mimic Joints in Schema**
```python
def test_no_mimic_joints_in_schema(self, robot_schema):
    """
    Purpose: Verify _mimic joints are excluded.
    
    Assertions:
    - No joint name contains "_mimic"
    """
    for joint in robot_schema["joints"]:
        assert "_mimic" not in joint["name"], f"Mimic joint leaked: {joint['name']}"
```

#### 0.2.2 Actuators

| Field | Description |
|-------|-------------|
| `name` | Actuator name |
| `joint_name` | Target joint name |
| `dof_adr` | DOF address |
| `gear` | Gear ratio |
| `ctrlrange` | Control limits [min, max] |
| `forcerange` | Force limits (if applicable) |

**Test 0.2.2: Actuators Target Valid Joints**
```python
def test_actuator_targets_valid_joints(self, robot_schema):
    """
    Purpose: Verify no actuator targets a _mimic joint.
    
    Assertions:
    - All actuator target joints exist in schema joints
    - No actuator targets a _mimic joint
    """
    joint_names = {j["name"] for j in robot_schema["joints"]}
    for actuator in robot_schema["actuators"]:
        assert "_mimic" not in actuator["joint_name"]
        assert actuator["joint_name"] in joint_names
```

#### 0.2.3 Foot Contact Sets

Two disjoint sets of geom IDs:
- Left foot: `left_toe`, `left_heel`
- Right foot: `right_toe`, `right_heel`

**Test 0.2.3: Foot Geoms Are Valid**
```python
def test_foot_geoms_valid(self, robot_schema):
    """
    Purpose: Verify foot geoms are correctly identified.
    
    Assertions:
    - Left and right foot geom sets are disjoint
    - All foot geoms are collision-enabled
    - No foot geom belongs to a _mimic body
    """
    left_ids = {g["id"] for g in robot_schema["left_foot_geoms"]}
    right_ids = {g["id"] for g in robot_schema["right_foot_geoms"]}
    
    assert left_ids.isdisjoint(right_ids), "Foot geom sets overlap!"
    
    for geom in robot_schema["left_foot_geoms"] + robot_schema["right_foot_geoms"]:
        assert "_mimic" not in geom["body"], f"Mimic geom leaked: {geom}"
        assert geom["contype"] > 0 or geom["conaffinity"] > 0, f"Geom not collision-enabled: {geom}"
```

#### 0.2.4 Base Definition

| Field | Description |
|-------|-------------|
| `body_name` | Base body name |
| `freejoint_name` | Free joint name (if used) |
| `qpos_adr` | qpos slice for pos + quat (7 values) |
| `dof_adr` | qvel slice for linvel + angvel (6 values) |

---

## Layer 1: Config and Runtime Contracts

### 1.1 Training Config Schema Validation

**Test 1.1.1: Required Keys Exist**
```python
def test_training_config_required_keys(self, training_config):
    """
    Purpose: Verify all required config sections exist.
    
    Assertions:
    - Required keys: env, ppo, amp, networks, reward_weights, checkpoints, logging
    """
    required = ["env", "ppo", "amp", "networks", "reward_weights", "checkpoints", "logging"]
    for key in required:
        assert key in training_config, f"Missing required key: {key}"
```

**Test 1.1.2: Timestep Consistency**
```python
def test_timestep_consistency(self, training_config):
    """
    Purpose: Verify timestep configuration is valid.
    
    Assertions:
    - sim_dt < ctrl_dt
    - ctrl_dt / sim_dt is integer
    - ctrl_dt = 0.02, sim_dt = 0.002 → ratio = 10
    """
    sim_dt = training_config.env.sim_dt
    ctrl_dt = training_config.env.ctrl_dt
    
    assert sim_dt < ctrl_dt, f"sim_dt ({sim_dt}) must be < ctrl_dt ({ctrl_dt})"
    ratio = ctrl_dt / sim_dt
    assert ratio == int(ratio), f"ctrl_dt/sim_dt must be integer, got {ratio}"
```

**Test 1.1.3: Reward Weights Valid**
```python
def test_reward_weights_valid(self, training_config):
    """
    Purpose: Verify reward weights are valid numbers.
    
    Assertions:
    - All weights are finite (not NaN or Inf)
    - All weights are non-negative
    """
    for name, weight in training_config.reward_weights.items():
        assert np.isfinite(weight), f"Reward weight {name} is not finite: {weight}"
        assert weight >= 0, f"Reward weight {name} is negative: {weight}"
```

### 1.2 Static Consistency Checks

**Test 1.2.1: Action Dim Matches Actuators**
```python
def test_action_dim_matches_actuators(self, training_config, robot_schema):
    """
    Purpose: Verify action dimension matches robot actuator count.
    
    Assertions:
    - training_config.action_dim == len(robot_schema.actuators)
    """
    assert training_config.env.action_dim == len(robot_schema["actuators"]), (
        f"Action dim mismatch: config={training_config.env.action_dim}, "
        f"actuators={len(robot_schema['actuators'])}"
    )
```

---

## Layer 2: MuJoCo Model Validation (Authoring Quality)

These tests run on the raw MuJoCo model without RL.

### 2.1 Load and Step Smoke

**Test 2.1.1: Model Loads and Steps Without Explosion**
```python
def test_model_stability(self, mj_model, mj_data):
    """
    Purpose: Verify model is stable under simulation.
    
    Procedure:
    1. Load scene_flat_terrain.xml
    2. Run 1000 sim steps with zero controls
    
    Assertions:
    - No NaN in qpos or qvel
    - No constraint solver warnings above threshold
    - Energy does not explode (< 1e6 J)
    """
    import mujoco
    
    for _ in range(1000):
        mj_data.ctrl[:] = 0
        mujoco.mj_step(mj_model, mj_data)
        
        assert not np.any(np.isnan(mj_data.qpos)), "NaN in qpos"
        assert not np.any(np.isnan(mj_data.qvel)), "NaN in qvel"
    
    # Check energy didn't explode
    kinetic = 0.5 * np.sum(mj_data.qvel ** 2)
    assert kinetic < 1e6, f"Energy exploded: {kinetic}"
```

### 2.2 Kinematic Sanity

**Test 2.2.1: Initial Pose Valid**
```python
def test_initial_pose_valid(self, mj_model, mj_data, training_config):
    """
    Purpose: Verify reset pose is physically valid.
    
    Assertions:
    - Base height in [min_height, max_height]
    - Base pitch and roll < 0.1 rad
    - All joints within limits
    - No initial penetrations
    """
    import mujoco
    
    mujoco.mj_resetData(mj_model, mj_data)
    mujoco.mj_forward(mj_model, mj_data)
    
    # Base height
    base_height = mj_data.qpos[2]
    assert training_config.env.min_height <= base_height <= training_config.env.max_height
    
    # Base orientation (pitch, roll near zero)
    quat = mj_data.qpos[3:7]
    # Convert to euler and check
    # ... (implementation depends on your math utilities)
    
    # Joint limits
    for i in range(mj_model.njnt):
        if mj_model.jnt_limited[i]:
            qpos_adr = mj_model.jnt_qposadr[i]
            pos = mj_data.qpos[qpos_adr]
            low, high = mj_model.jnt_range[i]
            assert low <= pos <= high, f"Joint {i} out of limits: {pos} not in [{low}, {high}]"
```

### 2.3 Actuator Isolation Test

**Test 2.3.1: Each Actuator Controls Its Joint**
```python
def test_actuator_isolation(self, mj_model, mj_data, robot_schema):
    """
    Purpose: Verify actuator-to-joint mapping is correct.
    
    Procedure:
    For each actuator:
    1. Reset to standing pose
    2. Zero all controls
    3. Apply small control (+0.1) to THIS actuator only
    4. Step simulation
    
    Assertions:
    - Target actuator produces significant force (|τ| > 0.01 Nm)
    - Target joint velocity changes (|Δv| > 1e-6 rad/s)
    - Other actuators produce near-zero force (|τ| < 0.001 Nm)
    
    Thresholds:
    - MIN_ACTUATOR_FORCE = 0.01 Nm
    - MIN_VELOCITY_CHANGE = 1e-6 rad/s
    """
    import mujoco
    
    MIN_ACTUATOR_FORCE = 0.01
    MIN_VELOCITY_CHANGE = 1e-6
    
    for actuator in robot_schema["actuators"]:
        act_name = actuator["name"]
        act_idx = next(i for i in range(mj_model.nu) 
                       if mj_model.actuator(i).name == act_name)
        target_dof = actuator["dof_adr"]
        
        # Reset
        mujoco.mj_resetData(mj_model, mj_data)
        mujoco.mj_forward(mj_model, mj_data)
        
        # Zero all controls, apply to this actuator only
        mj_data.ctrl[:] = 0
        mj_data.ctrl[act_idx] = 0.1
        
        # Record initial state
        initial_vel = mj_data.qvel[target_dof]
        
        # Step
        mujoco.mj_step(mj_model, mj_data)
        
        # Check actuator produces force
        act_force = mj_data.actuator_force[act_idx]
        assert abs(act_force) > MIN_ACTUATOR_FORCE, \
            f"Actuator {act_name} produced insufficient force: {act_force}"
        
        # Check target joint moved
        final_vel = mj_data.qvel[target_dof]
        vel_change = abs(final_vel - initial_vel)
        assert vel_change > MIN_VELOCITY_CHANGE, \
            f"Actuator {act_name} didn't move target joint: Δv={vel_change}"
```

---

## Layer 3: State and Observation Correctness

This layer ensures reward and AMP features read the right state.

### 3.1 Finite Difference Velocity Cross-Check

**Test 3.1.1: Base Linear Velocity FD vs qvel**
```python
def test_base_linear_velocity_fd_vs_qvel(self, mj_model, mj_data, robot_schema):
    """
    Purpose: Confirm base qvel is correct by comparing to finite difference.
    
    Procedure:
    1. Reset to standing pose
    2. Apply random actions for N=100 steps
    3. At each step compute FD velocity: v_fd = (pos_after - pos_before) / dt
    4. Compare FD velocities to qvel velocities
    
    Assertions:
    - Correlation coefficient > 0.95 for each axis (x, y, z)
    - Mean Absolute Error < 0.1 m/s
    
    Why This Matters:
    If this fails, YOUR VELOCITY REWARD IS MEANINGLESS.
    
    Thresholds:
    - MIN_CORRELATION = 0.95
    - MAX_MAE = 0.1 m/s
    """
    import mujoco
    
    MIN_CORRELATION = 0.95
    MAX_MAE = 0.1
    
    mujoco.mj_resetData(mj_model, mj_data)
    mujoco.mj_forward(mj_model, mj_data)
    
    dt = mj_model.opt.timestep
    n_steps = 100
    
    fd_velocities = []
    qvel_velocities = []
    
    base_dof_adr = robot_schema["base"]["dof_adr"]
    
    for _ in range(n_steps):
        mj_data.ctrl[:] = np.random.uniform(-1, 1, mj_model.nu)
        
        pos_before = mj_data.qpos[:3].copy()
        mujoco.mj_step(mj_model, mj_data)
        pos_after = mj_data.qpos[:3].copy()
        
        v_fd = (pos_after - pos_before) / dt
        fd_velocities.append(v_fd)
        
        v_qvel = mj_data.qvel[base_dof_adr:base_dof_adr+3].copy()
        qvel_velocities.append(v_qvel)
    
    fd_velocities = np.array(fd_velocities)
    qvel_velocities = np.array(qvel_velocities)
    
    # Check correlation for each axis
    for axis, name in enumerate(['x', 'y', 'z']):
        corr = np.corrcoef(fd_velocities[:, axis], qvel_velocities[:, axis])[0, 1]
        assert corr > MIN_CORRELATION, f"Velocity {name} correlation too low: {corr}"
    
    # Check MAE
    mae = np.mean(np.abs(fd_velocities - qvel_velocities))
    assert mae < MAX_MAE, f"Velocity MAE too high: {mae} m/s"
```

**Test 3.1.2: Base Angular Velocity FD vs qvel**
```python
def test_base_angular_velocity_fd_vs_qvel(self, mj_model, mj_data, robot_schema):
    """
    Purpose: Verify angular velocity is read correctly.
    
    Procedure:
    1. Reset and step while base rotates
    2. Compute angular velocity from quaternion delta
    3. Compare to qvel angular slice
    
    Assertions:
    - Correlation > 0.90 for each axis
    - MAE < 0.5 rad/s
    
    Thresholds:
    - MIN_CORRELATION = 0.90
    - MAX_MAE = 0.5 rad/s
    """
    from scipy.spatial.transform import Rotation
    
    MIN_CORRELATION = 0.90
    MAX_MAE = 0.5
    
    # ... (implementation similar to linear velocity test)
```

### 3.2 Joint-to-Foot Causality Test

**Test 3.2.1: Joint Movement Affects Correct Foot**
```python
def test_joint_to_foot_causality(self, mj_model, mj_data, robot_schema):
    """
    Purpose: Verify qpos indexing and geom IDs are correct by testing
    that moving a joint on one leg affects only that leg's foot.
    
    Procedure:
    1. Reset to standing pose
    2. Call mj_forward()
    3. Record left and right foot positions
    4. Add +0.1 rad to left knee qpos
    5. Call mj_forward()
    6. Record new foot positions
    
    Assertions:
    - Left foot Z changed significantly (|Δz| > 0.01m)
    - Right foot Z changed minimally (|Δz| < 0.001m)
    - Base position unchanged (|Δpos| < 0.001m)
    
    Thresholds:
    - MIN_AFFECTED_DELTA = 0.01 m
    - MAX_UNAFFECTED_DELTA = 0.001 m
    """
    import mujoco
    
    MIN_AFFECTED_DELTA = 0.01
    MAX_UNAFFECTED_DELTA = 0.001
    
    # Find left knee joint
    left_knee = next(j for j in robot_schema["joints"] 
                     if "left" in j["name"].lower() and "knee" in j["name"].lower())
    qpos_adr = left_knee["qpos_adr"]
    
    # Get foot geom IDs
    left_foot_geom = robot_schema["left_foot_geoms"][0]
    right_foot_geom = robot_schema["right_foot_geoms"][0]
    
    # Reset and forward
    mujoco.mj_resetData(mj_model, mj_data)
    mujoco.mj_forward(mj_model, mj_data)
    
    # Record initial positions
    left_foot_z_initial = mj_data.geom_xpos[left_foot_geom["id"]][2]
    right_foot_z_initial = mj_data.geom_xpos[right_foot_geom["id"]][2]
    base_pos_initial = mj_data.qpos[:3].copy()
    
    # Perturb left knee
    mj_data.qpos[qpos_adr] += 0.1
    mujoco.mj_forward(mj_model, mj_data)
    
    # Record new positions
    left_foot_z_final = mj_data.geom_xpos[left_foot_geom["id"]][2]
    right_foot_z_final = mj_data.geom_xpos[right_foot_geom["id"]][2]
    base_pos_final = mj_data.qpos[:3].copy()
    
    # Assertions
    left_delta = abs(left_foot_z_final - left_foot_z_initial)
    right_delta = abs(right_foot_z_final - right_foot_z_initial)
    base_delta = np.linalg.norm(base_pos_final - base_pos_initial)
    
    assert left_delta > MIN_AFFECTED_DELTA, \
        f"Left foot should move significantly: Δz={left_delta}"
    assert right_delta < MAX_UNAFFECTED_DELTA, \
        f"Right foot should not move: Δz={right_delta}"
    assert base_delta < MAX_UNAFFECTED_DELTA, \
        f"Base should not move: Δpos={base_delta}"
```

### 3.3 Contact Force Validation

**Test 3.3.1: Static Load Support (ΣFn ≈ mg)**
```python
def test_static_equilibrium(self, mj_model, mj_data, robot_schema):
    """
    Purpose: Verify total contact force equals robot weight.
    
    Procedure:
    1. Reset to standing pose
    2. Step until settled (500 steps)
    3. Sum normal forces on all foot geoms
    4. Compare to robot weight: mg = total_mass × 9.81
    
    Assertions:
    - ΣFn / mg ∈ [0.8, 1.2]
    
    Why This Matters:
    If this fails, CONTACT READING IS WRONG.
    
    Thresholds:
    - FORCE_RATIO_MIN = 0.8
    - FORCE_RATIO_MAX = 1.2
    """
    import mujoco
    
    FORCE_RATIO_MIN = 0.8
    FORCE_RATIO_MAX = 1.2
    
    mujoco.mj_resetData(mj_model, mj_data)
    mujoco.mj_forward(mj_model, mj_data)
    
    # Let robot settle
    for _ in range(500):
        mj_data.ctrl[:] = 0
        mujoco.mj_step(mj_model, mj_data)
    
    # Get foot geom IDs
    left_geom_ids = [g["id"] for g in robot_schema["left_foot_geoms"]]
    right_geom_ids = [g["id"] for g in robot_schema["right_foot_geoms"]]
    all_foot_geom_ids = set(left_geom_ids + right_geom_ids)
    
    # Sum normal forces from contacts
    total_normal_force = 0
    for i in range(mj_data.ncon):
        contact = mj_data.contact[i]
        geom1, geom2 = contact.geom1, contact.geom2
        
        if geom1 in all_foot_geom_ids or geom2 in all_foot_geom_ids:
            force = np.zeros(6)
            mujoco.mj_contactForce(mj_model, mj_data, i, force)
            normal_force = force[0]
            total_normal_force += abs(normal_force)
    
    # Compute robot weight
    total_mass = sum(mj_model.body(i).mass for i in range(mj_model.nbody))
    expected_weight = total_mass * 9.81
    
    # Check ratio
    ratio = total_normal_force / expected_weight
    assert FORCE_RATIO_MIN <= ratio <= FORCE_RATIO_MAX, \
        f"Contact force ratio wrong: {ratio:.2f} (expected ~1.0), " \
        f"Fn={total_normal_force:.1f}N, mg={expected_weight:.1f}N"
```

**Test 3.3.2: Left-Right Load Shift**
```python
def test_left_right_load_asymmetry(self, mj_model, mj_data, robot_schema):
    """
    Purpose: Verify feet differentiation is correct.
    
    Procedure:
    1. Reset to standing pose
    2. Shift COM slightly left (5cm)
    3. Step until settled
    4. Measure normal forces on left vs right feet
    
    Assertions:
    - Left Fn > Right Fn (since COM shifted left)
    - ΣFn / mg ∈ [0.8, 1.2]
    
    Thresholds:
    - COM_SHIFT = 0.05 m
    - FORCE_RATIO_MIN = 0.8
    - FORCE_RATIO_MAX = 1.2
    """
    import mujoco
    
    COM_SHIFT = 0.05
    FORCE_RATIO_MIN = 0.8
    FORCE_RATIO_MAX = 1.2
    
    mujoco.mj_resetData(mj_model, mj_data)
    mj_data.qpos[1] -= COM_SHIFT  # Shift left
    mujoco.mj_forward(mj_model, mj_data)
    
    # Let settle
    for _ in range(500):
        mj_data.ctrl[:] = 0
        mujoco.mj_step(mj_model, mj_data)
    
    left_geom_ids = set(g["id"] for g in robot_schema["left_foot_geoms"])
    right_geom_ids = set(g["id"] for g in robot_schema["right_foot_geoms"])
    
    left_force = 0
    right_force = 0
    
    for i in range(mj_data.ncon):
        contact = mj_data.contact[i]
        force = np.zeros(6)
        mujoco.mj_contactForce(mj_model, mj_data, i, force)
        normal_force = abs(force[0])
        
        if contact.geom1 in left_geom_ids or contact.geom2 in left_geom_ids:
            left_force += normal_force
        if contact.geom1 in right_geom_ids or contact.geom2 in right_geom_ids:
            right_force += normal_force
    
    # Left should have more load
    assert left_force > right_force, \
        f"Expected left > right force. Left={left_force:.1f}N, Right={right_force:.1f}N"
    
    # Total should still be body weight
    total_mass = sum(mj_model.body(i).mass for i in range(mj_model.nbody))
    expected_weight = total_mass * 9.81
    total_force = left_force + right_force
    ratio = total_force / expected_weight
    
    assert FORCE_RATIO_MIN <= ratio <= FORCE_RATIO_MAX, \
        f"Total force ratio: {ratio:.2f}"
```

**Test 3.3.3: Drop Impact Detection**
```python
def test_drop_impact_force(self, mj_model, mj_data, robot_schema):
    """
    Purpose: Verify impact forces are detected correctly.
    
    Procedure:
    1. Start robot 5cm above ground
    2. Let it drop
    3. Monitor contact forces
    
    Assertions:
    - Peak force spike detected (Fn_peak > 2 × mg)
    - Force stabilizes to ~mg after impact
    
    Thresholds:
    - DROP_HEIGHT = 0.05 m
    - PEAK_FORCE_MULTIPLIER = 2.0
    - STABILIZATION_RATIO_MIN = 0.8
    - STABILIZATION_RATIO_MAX = 1.2
    """
    import mujoco
    
    DROP_HEIGHT = 0.05
    PEAK_FORCE_MULTIPLIER = 2.0
    
    mujoco.mj_resetData(mj_model, mj_data)
    mj_data.qpos[2] += DROP_HEIGHT
    mujoco.mj_forward(mj_model, mj_data)
    
    # Get foot geom IDs
    all_foot_geom_ids = set(
        g["id"] for g in robot_schema["left_foot_geoms"] + robot_schema["right_foot_geoms"]
    )
    
    peak_force = 0
    forces = []
    
    for _ in range(200):
        mj_data.ctrl[:] = 0
        mujoco.mj_step(mj_model, mj_data)
        
        total_force = 0
        for i in range(mj_data.ncon):
            contact = mj_data.contact[i]
            if contact.geom1 in all_foot_geom_ids or contact.geom2 in all_foot_geom_ids:
                force = np.zeros(6)
                mujoco.mj_contactForce(mj_model, mj_data, i, force)
                total_force += abs(force[0])
        
        forces.append(total_force)
        peak_force = max(peak_force, total_force)
    
    total_mass = sum(mj_model.body(i).mass for i in range(mj_model.nbody))
    expected_weight = total_mass * 9.81
    
    assert peak_force > PEAK_FORCE_MULTIPLIER * expected_weight, \
        f"No impact spike. Peak={peak_force:.1f}N, expected > {PEAK_FORCE_MULTIPLIER * expected_weight:.1f}N"
    
    # Check stabilization
    final_force = np.mean(forces[-50:])
    ratio = final_force / expected_weight
    assert 0.8 <= ratio <= 1.2, \
        f"Force didn't stabilize. Final={final_force:.1f}N, expected={expected_weight:.1f}N"
```

---

## Layer 4: Reward Component Correctness

Use deterministic micro-scenarios, not training, to validate each term.

### 4.1 Velocity Reward Monotonicity

**Test 4.1.1: Velocity Tracking Reward**
```python
def test_velocity_reward_monotonicity(self, reward_fn):
    """
    Purpose: Verify velocity reward decreases as error increases.
    
    Procedure:
    1. Create states with varying velocity errors
    2. Compute velocity reward for each
    
    Assertions:
    - Reward is maximum at v_actual = v_target
    - Reward strictly decreases as |v_actual - v_target| increases
    """
    target_velocity = 1.0  # m/s
    velocities = np.linspace(-1.0, 3.0, 20)
    
    rewards = [reward_fn.compute_velocity_reward(v, target_velocity) for v in velocities]
    rewards = np.array(rewards)
    
    # Find max reward
    max_idx = np.argmax(rewards)
    assert np.isclose(velocities[max_idx], target_velocity, atol=0.2), \
        f"Max reward not at target velocity"
    
    # Check monotonicity on both sides
    for i in range(max_idx):
        assert rewards[i] < rewards[i+1], "Not increasing toward target"
    for i in range(max_idx, len(rewards)-1):
        assert rewards[i] > rewards[i+1], "Not decreasing away from target"
```

### 4.2 Slip Gating Test (Critical)

**Test 4.2.1: Slip Penalty Gating**
```python
def test_slip_penalty_gating(self, reward_fn):
    """
    Purpose: Verify slip penalty only applies when foot is loaded.
    
    Test Cases:
    A. Fn > threshold, tangential velocity > 0 → penalty applies
    B. Fn < threshold, tangential velocity > 0 → penalty suppressed
    C. Fn > threshold, tangential velocity = 0 → no penalty
    
    Assertions:
    - Case A: penalty > 0
    - Case B: penalty ≈ 0 (gated by low contact force)
    - Case C: penalty ≈ 0 (no slip occurring)
    
    Why This Matters:
    Without proper gating, robot learns to avoid lifting feet.
    
    Thresholds:
    - CONTACT_THRESHOLD = 50.0 N
    - MIN_PENALTY = 0.01
    """
    CONTACT_THRESHOLD = 50.0
    MIN_PENALTY = 0.01
    
    # Case A: Loaded foot with slip
    penalty_a = reward_fn.compute_slip_penalty(
        normal_force=100.0,
        tangential_velocity=0.5,
        threshold=CONTACT_THRESHOLD
    )
    assert penalty_a > MIN_PENALTY, f"Case A: Expected penalty > {MIN_PENALTY}, got {penalty_a}"
    
    # Case B: Unloaded foot with motion (swing phase)
    penalty_b = reward_fn.compute_slip_penalty(
        normal_force=10.0,
        tangential_velocity=1.0,
        threshold=CONTACT_THRESHOLD
    )
    assert penalty_b < MIN_PENALTY, f"Case B: Penalty should be gated, got {penalty_b}"
    
    # Case C: Loaded foot, no slip
    penalty_c = reward_fn.compute_slip_penalty(
        normal_force=100.0,
        tangential_velocity=0.0,
        threshold=CONTACT_THRESHOLD
    )
    assert penalty_c < MIN_PENALTY, f"Case C: No slip, expected no penalty, got {penalty_c}"
```

### 4.3 Orientation Penalty

**Test 4.3.1: Orientation Penalty Increases with Tilt**
```python
def test_orientation_penalty(self, reward_fn):
    """
    Purpose: Verify orientation penalty increases with pitch/roll.
    
    Procedure:
    1. Test with pitch = 0 (upright)
    2. Test with pitch = 10°
    3. Test with pitch = 30°
    
    Assertions:
    - penalty(0°) < penalty(10°) < penalty(30°)
    - penalty(0°) ≈ 0
    
    Thresholds:
    - MAX_UPRIGHT_PENALTY = 0.01
    """
    MAX_UPRIGHT_PENALTY = 0.01
    
    penalty_0 = reward_fn.compute_orientation_penalty(pitch=0.0, roll=0.0)
    penalty_10 = reward_fn.compute_orientation_penalty(pitch=np.radians(10), roll=0.0)
    penalty_30 = reward_fn.compute_orientation_penalty(pitch=np.radians(30), roll=0.0)
    
    assert penalty_0 < MAX_UPRIGHT_PENALTY, f"Upright penalty should be ~0: {penalty_0}"
    assert penalty_0 < penalty_10 < penalty_30, "Penalty should increase with tilt"
```

---

## Layer 5: Environment Step Integrity

### 5.1 Reset Contract

**Test 5.1.1: Reset Returns Valid State**
```python
def test_reset_returns_valid_state(self, env, rng):
    """
    Purpose: Verify reset returns properly shaped, finite values.
    
    Assertions:
    - Observation shape matches env.observation_size
    - All observation values are finite
    - Reset with same seed produces identical state
    """
    state1 = env.reset(rng)
    state2 = env.reset(rng)
    
    assert state1.obs.shape == (env.observation_size,)
    assert np.all(np.isfinite(state1.obs))
    assert np.allclose(state1.obs, state2.obs), "Reset not deterministic with same seed"
```

### 5.2 Termination Correctness

**Test 5.2.1: Height Termination**
```python
def test_height_termination(self, env, training_config):
    """
    Purpose: Verify termination triggers at min_height.
    
    Procedure:
    1. Reset environment
    2. Force base height below min_height
    3. Step
    
    Assertions:
    - done = True
    
    Thresholds:
    - Uses training_config.env.min_height
    """
    state = env.reset(rng)
    
    # Force base below min_height
    state = state.replace(qpos=state.qpos.at[2].set(training_config.env.min_height - 0.1))
    
    new_state, info = env.step(state, action=np.zeros(env.action_dim))
    
    assert new_state.done, "Should terminate when base below min_height"
```

**Test 5.2.2: Orientation Termination**
```python
def test_orientation_termination(self, env, training_config):
    """
    Purpose: Verify termination triggers at extreme pitch/roll.
    
    Assertions:
    - done = True when pitch > max_pitch
    - done = True when roll > max_roll
    
    Thresholds:
    - Uses training_config.env.max_pitch
    - Uses training_config.env.max_roll
    """
    # Test extreme pitch
    state = env.reset(rng)
    state = force_pitch(state, angle_deg=70)  # Exceeds typical max_pitch
    new_state, _ = env.step(state, action=np.zeros(env.action_dim))
    assert new_state.done, "Should terminate at extreme pitch"
    
    # Test extreme roll
    state = env.reset(rng)
    state = force_roll(state, angle_deg=70)
    new_state, _ = env.step(state, action=np.zeros(env.action_dim))
    assert new_state.done, "Should terminate at extreme roll"
```

---

## Layer 6: Training Correctness (PPO)

This tier proves training is wired correctly before you spend GPU days.

### 6.1 PPO Smoke Test

**Test 6.1.1: Short Training Completes**
```python
def test_ppo_smoke(self, training_config):
    """
    Purpose: Verify minimal training runs without error.
    
    Procedure:
    1. Run with num_envs=64, rollout_steps=32, iterations=10
    2. Verify completion
    
    Assertions:
    - No exceptions
    - Loss values are finite (not NaN)
    - Policy parameters change (KL > 0)
    - Training completes all iterations
    
    Thresholds:
    - NUM_ENVS = 64
    - ROLLOUT_STEPS = 32
    - NUM_ITERATIONS = 10
    """
    config = training_config.replace(
        num_envs=64,
        rollout_steps=32,
        num_iterations=10,
    )
    
    result = train(config)
    
    assert not np.isnan(result["final_loss"]), "Training produced NaN loss"
    assert result["iterations_completed"] == 10
    assert result["kl_divergence"] > 0, "Policy didn't change"
```

### 6.2 Regression Training (Nightly)

**Test 6.2.1: Training Reaches Baseline**
```python
@pytest.mark.train
def test_training_reaches_baseline(self, training_config):
    """
    Purpose: Verify training produces learning signal.
    
    Procedure:
    1. Train for 100 iterations with fixed seed
    2. Record metrics
    
    Assertions:
    - Average episode length > 400 steps
    - Forward velocity tracking error < 0.2 m/s
    - Fall rate < 5%
    - Torque usage < 80% of limit
    
    Thresholds (Stage 1 Exit Criteria):
    - MIN_EPISODE_LENGTH = 400
    - MAX_VELOCITY_ERROR = 0.2 m/s
    - MAX_FALL_RATE = 0.05
    - MAX_TORQUE_RATIO = 0.80
    """
    MIN_EPISODE_LENGTH = 400
    MAX_VELOCITY_ERROR = 0.2
    MAX_FALL_RATE = 0.05
    MAX_TORQUE_RATIO = 0.80
    
    result = train(training_config, num_iterations=100, seed=42)
    
    assert result["avg_episode_length"] > MIN_EPISODE_LENGTH
    assert result["velocity_error"] < MAX_VELOCITY_ERROR
    assert result["fall_rate"] < MAX_FALL_RATE
    assert result["torque_ratio"] < MAX_TORQUE_RATIO
```

---

## Layer 7: AMP Integration Tests

Only relevant once `amp.enabled: true`.

### 7.1 Feature Parity

**Test 7.1.1: Policy and Reference Features Match**
```python
def test_feature_dimension_parity(self, policy_feature_builder, ref_feature_builder):
    """
    Purpose: Verify policy and reference feature builders produce same dimensions.
    
    Assertions:
    - Output dimensions match
    - Feature ordering is identical
    """
    policy_dim = policy_feature_builder.output_dim
    ref_dim = ref_feature_builder.output_dim
    
    assert policy_dim == ref_dim, f"Dimension mismatch: policy={policy_dim}, ref={ref_dim}"
```

### 7.2 AMP Mirror Symmetry

**Test 7.2.1: Mirror Transform Correctness**
```python
def test_amp_mirror_symmetry(self):
    """
    Purpose: Verify mirror transform correctly swaps left/right.
    
    Procedure:
    1. Create reference features with known left/right asymmetry
    2. Apply mirror transform
    3. Verify left/right indices are swapped
    
    Assertions:
    - Left joint values become right joint values
    - Right joint values become left joint values
    - Symmetric values (e.g., base height) unchanged
    - Mirrored + original gives 2x dataset size
    """
    from playground_amp.amp.amp_mirror import augment_with_mirror
    
    # Create asymmetric test data
    original = np.zeros((10, 29))  # 10 samples, 29 features
    original[:, 0] = 1.0  # Left hip
    original[:, 3] = 2.0  # Right hip
    
    augmented = augment_with_mirror(original)
    
    # Should have 2x samples
    assert augmented.shape[0] == 20, f"Expected 20 samples, got {augmented.shape[0]}"
    
    # Original samples preserved
    assert np.allclose(augmented[:10], original)
    
    # Mirrored samples have swapped left/right
    # (specific indices depend on your feature layout)
```

### 7.3 Discriminator Health

**Test 7.3.1: Discriminator Overfit Test**
```python
def test_discriminator_overfit(self, discriminator, tiny_dataset):
    """
    Purpose: Verify discriminator can learn on tiny dataset.
    
    Procedure:
    1. Train discriminator on 100 samples for 1000 steps
    2. Check accuracy
    
    Assertions:
    - Accuracy > 0.95 on training set
    - Loss converges
    
    Thresholds:
    - MIN_ACCURACY = 0.95
    """
    MIN_ACCURACY = 0.95
    
    for _ in range(1000):
        discriminator.train_step(tiny_dataset)
    
    accuracy = discriminator.evaluate(tiny_dataset)
    assert accuracy > MIN_ACCURACY, f"Discriminator failed to overfit: {accuracy}"
```

---

## Layer 8: MJX Parity

Purpose: ensure your JAX/MJX environment matches MuJoCo Python numerically.

### 8.1 Step Parity

**Test 8.1.1: MJX vs MuJoCo Step Comparison**
```python
def test_mjx_step_parity(self, mj_env, mjx_env, rng):
    """
    Purpose: Verify MJX produces similar results to MuJoCo Python.
    
    Procedure:
    1. Initialize both with same state
    2. Run 100 steps with identical actions
    3. Compare observations, rewards, done flags
    
    Assertions:
    - Observation MAE < 0.1
    - Reward MAE < 0.1
    - Done flag agreement > 95%
    
    Thresholds:
    - MAX_OBS_MAE = 0.1
    - MAX_REWARD_MAE = 0.1
    - MIN_DONE_AGREEMENT = 0.95
    """
    MAX_OBS_MAE = 0.1
    MAX_REWARD_MAE = 0.1
    MIN_DONE_AGREEMENT = 0.95
    
    state_mj = mj_env.reset(rng)
    state_mjx = mjx_env.reset(rng)
    
    obs_diffs = []
    reward_diffs = []
    done_matches = []
    
    for i in range(100):
        action = np.random.uniform(-1, 1, mj_env.action_dim)
        
        state_mj, info_mj = mj_env.step(state_mj, action)
        state_mjx, info_mjx = mjx_env.step(state_mjx, action)
        
        obs_diffs.append(np.abs(state_mj.obs - state_mjx.obs).mean())
        reward_diffs.append(abs(info_mj["reward"] - info_mjx["reward"]))
        done_matches.append(state_mj.done == state_mjx.done)
    
    assert np.mean(obs_diffs) < MAX_OBS_MAE
    assert np.mean(reward_diffs) < MAX_REWARD_MAE
    assert np.mean(done_matches) > MIN_DONE_AGREEMENT
```

---

## Layer 9: Robustness and Generalization (Sim)

### 9.1 Domain Randomization

**Test 9.1.1: Randomization Reproducibility**
```python
def test_domain_randomization_reproducible(self, env):
    """
    Purpose: Verify randomization is reproducible with same seed.
    
    Assertions:
    - Same seed produces identical randomized parameters
    """
    params1 = env.sample_domain_params(seed=42)
    params2 = env.sample_domain_params(seed=42)
    
    assert params1 == params2, "Randomization not reproducible"
```

### 9.2 Perturbation Recovery

**Test 9.2.1: Lateral Push Recovery**
```python
@pytest.mark.robust
def test_lateral_push_recovery(self, trained_policy, env):
    """
    Purpose: Verify policy recovers from lateral pushes.
    
    Procedure:
    1. Run policy until stable
    2. Apply lateral impulse
    3. Measure recovery time
    
    Assertions:
    - Recovery rate > 80%
    - Recovery time < 50 steps
    
    Thresholds:
    - PUSH_FORCE = 50 N
    - PUSH_DURATION = 5 steps
    - MIN_RECOVERY_RATE = 0.80
    - MAX_RECOVERY_STEPS = 50
    """
    # Implementation depends on your policy evaluation setup
    pass
```

---

## Layer 10: Sim-to-Real Readiness Pack

This is a **process plus validation** layer.

### 10.1 Hardware Safety Gate

**Test 10.1.1: Joint Limits Respected**
```python
def test_joint_limits_respected(self, trained_policy, env, robot_schema):
    """
    Purpose: Verify policy never commands positions outside joint limits.
    
    Procedure:
    1. Run policy for 1000 steps
    2. Record all commanded positions
    
    Assertions:
    - All positions within [joint_min - margin, joint_max + margin]
    
    Thresholds:
    - MARGIN = 0.05 rad
    """
    MARGIN = 0.05
    
    for joint in robot_schema["joints"]:
        low, high = joint["range"]
        # Check all recorded positions are within limits + margin
        # ...
```

### 10.2 Deliverables Checklist

Before real deployment, verify:
- [ ] Calibration parameter file exists
- [ ] Real-time telemetry schema defined
- [ ] Side-by-side sim vs real comparison completed
- [ ] E-stop logic validated
- [ ] First runs use reduced amplitude scaling

---

## Traceability Matrix

| Stage | Required Layers | Notes |
|-------|-----------------|-------|
| Stage 1 (PPO walking) | 0, 1, 2, 3, 4, 5, 6 | No AMP |
| Stage 2 (dataset) | 0-6 + 9 | Robustness for dataset diversity |
| Stage 3 (AMP) | 0-8 + 9 | Feature parity and discriminator |
| Stage 4+ (sim-to-real) | 0-10 | Full readiness pack |

---

## Shared Test Fixtures (conftest.py)

```python
# playground_amp/tests/conftest.py

import pytest
import mujoco
import numpy as np
import jax

@pytest.fixture(scope="session")
def robot_schema():
    """Load robot schema once per session."""
    import json
    with open("playground_amp/schema/wildrobot_schema.json") as f:
        return json.load(f)

@pytest.fixture(scope="session")
def mj_model():
    """Load MuJoCo model once per session."""
    return mujoco.MjModel.from_xml_path("assets/scene_flat_terrain.xml")

@pytest.fixture(scope="function")
def mj_data(mj_model):
    """Create fresh MuJoCo data for each test."""
    return mujoco.MjData(mj_model)

@pytest.fixture(scope="session")
def training_config():
    """Load training config once per session."""
    from playground_amp.configs.training_config import load_training_config
    return load_training_config("playground_amp/configs/ppo_walking.yaml")

@pytest.fixture(scope="module")
def env(training_config):
    """Create WildRobot environment."""
    from playground_amp.envs.wildrobot_env import WildRobotEnv
    return WildRobotEnv(config=training_config)

@pytest.fixture
def rng():
    """JAX random key."""
    return jax.random.PRNGKey(42)
```

---

## Running Tests

```bash
# All tests
uv run pytest playground_amp/tests -v

# By tier
uv run pytest playground_amp/tests -m "unit" -v          # T0: Pre-commit
uv run pytest playground_amp/tests -m "sim" -v           # T1: PR gate
uv run pytest playground_amp/tests -m "train" -v         # T1/T2: Training
uv run pytest playground_amp/tests -m "mjx" -v           # T2: MJX parity
uv run pytest playground_amp/tests -m "robust" -v        # T2/T3: Robustness

# By layer
uv run pytest playground_amp/tests/test_schema_contract.py -v    # Layer 0
uv run pytest playground_amp/tests/test_model_validation.py -v   # Layer 2
uv run pytest playground_amp/tests/test_state_mapping.py -v      # Layer 3
uv run pytest playground_amp/tests/test_reward_terms.py -v       # Layer 4
uv run pytest playground_amp/tests/test_env_step.py -v           # Layer 5

# With coverage
uv run pytest playground_amp/tests --cov=playground_amp --cov-report=html
```

---

## Summary of Critical Thresholds

| Test | Metric | Threshold |
|------|--------|-----------|
| Linear velocity FD vs qvel | Correlation | > 0.95 |
| Linear velocity FD vs qvel | MAE | < 0.1 m/s |
| Angular velocity FD vs qvel | Correlation | > 0.90 |
| Angular velocity FD vs qvel | MAE | < 0.5 rad/s |
| Static equilibrium | ΣFn / mg | [0.8, 1.2] |
| Drop impact | Peak force | > 2 × mg |
| Actuator isolation | Min force | > 0.01 Nm |
| Actuator isolation | Min Δv | > 1e-6 rad/s |
| Joint-to-foot causality | Affected Δz | > 0.01 m |
| Joint-to-foot causality | Unaffected Δz | < 0.001 m |
| Slip gating | Min penalty | > 0.01 |
| PPO smoke | Loss | finite |
| Stage 1 exit | Episode length | > 400 |
| Stage 1 exit | Velocity error | < 0.2 m/s |
| Stage 1 exit | Fall rate | < 5% |
| MJX parity | Obs MAE | < 0.1 |
| MJX parity | Reward MAE | < 0.1 |
| MJX parity | Done agreement | > 95% |

---

## Immediate Next Steps

1. **Create schema directory**: `playground_amp/schema/wildrobot_schema.json`
2. **Implement `test_schema_contract.py`**: Layer 0 tests
3. **Add velocity FD cross-check**: `test_state_mapping.py`
4. **Add contact force tests**: `test_contacts.py` with ΣFn ≈ mg
5. **Add actuator isolation test**: `test_model_validation.py`
6. **Add slip gating test**: `test_reward_terms.py`
7. **Add PPO smoke test as PR gate**: `test_ppo_smoke.py`
