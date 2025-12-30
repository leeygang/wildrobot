# Control Abstraction Layer (CAL) Proposal

## Executive Summary

This document proposes introducing a **Control Abstraction Layer (CAL)** to decouple high-level flows (training, testing, sim2real) from MuJoCo primitives. This addresses the current issues of ambiguous actuator semantics, incorrect symmetry assumptions, and training pipelines learning simulator quirks.

---

## 1. Problem Statement

### 1.1 Current Issues Identified

#### Direct MuJoCo Primitive Access
Multiple layers directly interact with MuJoCo primitives:

| Location | Primitive Access | Issue |
|----------|------------------|-------|
| `wildrobot_env.py:524` | `data.replace(ctrl=...)` | Assumes ctrl is joint position |
| `wildrobot_env.py:712` | `data.replace(ctrl=filtered_action)` | Ambiguous semantics |
| `test_physics_validation.py:559` | `mj_data.ctrl[act_idx] = 0.1` | Direct manipulation |
| `debug_gmr_physics.py:170-260` | `mj_data.qpos[...]`, `mj_data.qvel[...]` | Raw state access |

#### Ambiguous Actuator Semantics
From `robot_config.yaml`, all actuators are `type: position`:
```yaml
actuators:
  details:
  - name: left_hip_pitch
    joint: left_hip_pitch
    type: position  # MuJoCo position actuator with internal PD
    class: htd45hServo
```

**Problem**: The code passes actions directly to `ctrl`, but:
- Position actuators expect **target angles** (radians), not normalized actions
- MuJoCo's position actuators have an **internal PD controller**
- Policy outputs are typically in `[-1, 1]` range (Gaussian distribution)

#### Incorrect Symmetry Assumptions (Joint Range Mirroring)
From `robot_config.yaml`:
```yaml
joints:
  details:
  - name: left_hip_pitch
    range: [-0.087, 1.57]   # Asymmetric: more forward range
  - name: right_hip_pitch
    range: [-1.57, 0.087]   # MIRRORED: more backward range
```

**Problem**: Left/right hip pitch have **inverted ranges** due to physical mirror construction. If the policy learns `action[0] = 0.5` for left hip, the same `0.5` for right hip will produce a **physically opposite** motion, not a symmetric one.

#### Training Learning Simulator Quirks
Current flow:
```
Policy Output [-1, 1] → (no conversion) → ctrl (expects radians) → MuJoCo PD → Torque
```

The policy is forced to learn:
1. The MuJoCo-specific input range
2. The PD controller dynamics
3. Joint range asymmetries

This creates **non-transferable behaviors** that won't work on real hardware.

---

## 2. Best Practices Reference

### 2.1 Industry Standards

| Framework | Pattern | Reference |
|-----------|---------|-----------|
| **dm_control** | `action_spec` with bounds, automatic scaling | DeepMind |
| **Gymnasium** | `action_space` normalization | OpenAI/Farama |
| **MuJoCo Playground** | Separate `ctrl_dt` / `sim_dt`, action postprocessing | Brax/Google |
| **RLlib** | `normalize_actions` flag | CartPole benchmark issues |


### 2.2 Hardware Abstraction Layer (HAL) Concept

Applied to robot control:
- **Input**: Normalized, symmetric, physics-agnostic actions
- **Output**: Simulator-specific control signals
- **Benefit**: Same policy works in sim and real

---

## 3. Proposed Architecture

### 3.1 Control Abstraction Layer (CAL)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         HIGH-LEVEL LAYER                                │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────────┐  │
│  │ Training │    │  Test    │    │ Sim2Real │    │ Policy Inference │  │
│  │ Pipeline │    │  Suite   │    │ Transfer │    │                  │  │
│  └────┬─────┘    └────┬─────┘    └────┬─────┘    └────────┬─────────┘  │
│       │               │               │                   │            │
│       └───────────────┴───────────────┴───────────────────┘            │
│                              │                                          │
│                              ▼                                          │
│       ┌──────────────────────────────────────────────────┐             │
│       │         CONTROL ABSTRACTION LAYER (CAL)          │             │
│       │                                                  │             │
│       │  ┌────────────────┐  ┌─────────────────────────┐ │             │
│       │  │ Action Spec    │  │ Joint Spec              │ │             │
│       │  │ - bounds       │  │ - range [min, max]      │ │             │
│       │  │ - normalized   │  │ - default               │ │             │
│       │  │ - shape        │  │ - mirror_sign           │ │             │
│       │  └────────────────┘  │ - symmetry_group        │ │             │
│       │                      └─────────────────────────┘ │             │
│       │  ┌─────────────────────────────────────────────┐ │             │
│       │  │ ActionTransformer                           │ │             │
│       │  │ - normalize():   raw → [-1, 1]              │ │             │
│       │  │ - denormalize(): [-1, 1] → raw              │ │             │
│       │  │ - to_ctrl():     action → MuJoCo ctrl       │ │             │
│       │  │ - mirror():      handle L/R symmetry        │ │             │
│       │  └─────────────────────────────────────────────┘ │             │
│       │  ┌─────────────────────────────────────────────┐ │             │
│       │  │ StateTransformer                            │ │             │
│       │  │ - get_joint_pos(): qpos → normalized        │ │             │
│       │  │ - get_joint_vel(): qvel → normalized        │ │             │
│       │  │ - get_body_state(): body pos/vel/quat       │ │             │
│       │  └─────────────────────────────────────────────┘ │             │
│       └──────────────────────────────────────────────────┘             │
│                              │                                          │
└──────────────────────────────┼──────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                         LOW-LEVEL LAYER                                  │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                    MuJoCo / MJX Backend                            │ │
│  │  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────────────┐│ │
│  │  │  ctrl    │   │  qpos    │   │  qvel    │   │  actuator_force  ││ │
│  │  └──────────┘   └──────────┘   └──────────┘   └──────────────────┘│ │
│  └────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Core Components

#### 3.2.1 JointSpec (Single Source of Truth)

```python
@dataclass(frozen=True)
class JointSpec:
    """Joint specification - single source of truth for joint properties.

    This replaces scattered joint range access throughout the codebase.
    """
    name: str
    joint_id: int
    qpos_addr: int
    qvel_addr: int

    # Physical range (from URDF/XML)
    range_min: float  # radians
    range_max: float  # radians

    # Default position (from keyframe or qpos0)
    default_pos: float

    # Symmetry handling
    mirror_sign: float  # +1.0 for same-sign, -1.0 for inverted
    symmetry_pair: Optional[str]  # e.g., "left_hip_pitch" ↔ "right_hip_pitch"

    @property
    def range_center(self) -> float:
        """Center of joint range."""
        return (self.range_min + self.range_max) / 2

    @property
    def range_span(self) -> float:
        """Half-width of joint range."""
        return (self.range_max - self.range_min) / 2
```

#### 3.2.2 ActuatorSpec

```python
from enum import Enum

class ActuatorType(Enum):
    """Actuator types supported by CAL.

    Currently implemented: POSITION
    Future expansion: VELOCITY, TORQUE, MOTOR
    """
    POSITION = "position"   # ctrl = target angle (rad), MuJoCo PD controller
    # VELOCITY = "velocity" # ctrl = target velocity (rad/s) - future
    # TORQUE = "torque"     # ctrl = direct torque (Nm) - future
    # MOTOR = "motor"       # ctrl = generalized force - future


@dataclass(frozen=True)
class ActuatorSpec:
    """Actuator specification - defines control semantics.

    Naming Convention:
    - action: Normalized policy output in [-1, 1] (what NN produces)
    - ctrl: MuJoCo control signal in physical units (what simulator consumes)

    This naming is consistent with:
    - MuJoCo: data.ctrl
    - Gymnasium: action (normalized) → ctrl
    - Brax: action → ctrl
    - DM Control: action → physics.data.ctrl
    """
    name: str
    actuator_id: int
    joint_spec: JointSpec

    # Actuator type determines how ctrl is interpreted
    # Currently only POSITION is implemented
    actuator_type: ActuatorType = ActuatorType.POSITION

    # Control range (meaning depends on actuator_type)
    # - POSITION: (min_angle, max_angle) in radians
    # - VELOCITY: (min_vel, max_vel) in rad/s (future)
    # - TORQUE: (min_torque, max_torque) in Nm (future)
    ctrl_range: Tuple[float, float]  # from actuator_ctrlrange

    # Force/torque limits (for all types, used in reward normalization)
    force_range: Tuple[float, float]  # from actuator_forcerange

    # Velocity limit (from motor datasheet, used for velocity normalization)
    # TODO: Get actual values from motor datasheets:
    # - HTD-45H servo: max velocity = ??? rad/s (check datasheet)
    # - Different joints may have different motors/gear ratios
    # Default: 10.0 rad/s (~95 RPM) - conservative estimate for small servos
    max_velocity: float = 10.0  # rad/s - PLACEHOLDER, update from motor specs

    # PD gains (only applicable for POSITION type)
    kp: Optional[float] = None
    kv: Optional[float] = None  # Note: kv not kd (MuJoCo convention)

    def normalize_action(self, raw_action: float) -> float:
        """Convert raw action to normalized [-1, 1] range."""
        center = (self.ctrl_range[0] + self.ctrl_range[1]) / 2
        span = (self.ctrl_range[1] - self.ctrl_range[0]) / 2
        return (raw_action - center) / span

    def denormalize_action(self, normalized: float) -> float:
        """Convert normalized [-1, 1] to raw control value."""
        center = (self.ctrl_range[0] + self.ctrl_range[1]) / 2
        span = (self.ctrl_range[1] - self.ctrl_range[0]) / 2
        return normalized * span + center
```

#### 3.2.3 ControlCommand (Sim2Real Trajectory Support)

```python
@dataclass(frozen=True)
class ControlCommand:
    """Complete control command with trajectory information.

    CRITICAL FOR SIM2REAL:
    In simulation, MuJoCo's position actuator instantly starts moving toward
    the target at whatever speed the PD dynamics allow. But real motors need:
    - Target position (where to go)
    - Target velocity (how fast to get there)
    - Duration/time (when to arrive)

    This dataclass bridges the sim-real gap by explicitly specifying trajectory.

    Simulation vs Real Hardware:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ SIMULATION (MuJoCo)                                                     │
    │ - ctrl = target_position only                                           │
    │ - PD controller computes: torque = kp*(target - qpos) - kv*qvel        │
    │ - Velocity is emergent from dynamics (not commanded)                   │
    │ - Time to reach target depends on dynamics, not specified              │
    └─────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────────┐
    │ REAL HARDWARE (Servo Motors)                                            │
    │ - Need target_position + target_velocity (or duration)                  │
    │ - Motor controller interpolates trajectory                              │
    │ - Velocity profile: trapezoidal, S-curve, or linear                    │
    │ - Time to reach is explicitly controlled                               │
    └─────────────────────────────────────────────────────────────────────────┘

    Usage:
        # In simulation - velocity is optional (computed from dynamics)
        cmd = cal.action_to_command(action)
        ctrl = cmd.target_position  # Only this goes to MuJoCo

        # In real hardware - velocity is required
        cmd = cal.action_to_command(action, include_velocity=True)
        motor.set_position(cmd.target_position, velocity=cmd.target_velocity)
    """
    # Target position (always required)
    target_position: jax.Array  # shape (num_actuators,), in radians

    # Target velocity (for Sim2Real trajectory control)
    target_velocity: Optional[jax.Array] = None  # shape (num_actuators,), rad/s

    # Duration to reach target (alternative to velocity)
    duration: Optional[float] = None  # seconds

    # Acceleration limits (for smooth motion)
    max_acceleration: Optional[jax.Array] = None  # rad/s²


class VelocityProfile(Enum):
    """Velocity profile types for trajectory generation.

    Defines how velocity changes over time when moving from position A to B.

    Visual comparison:

    STEP:        ∞│▓▓              Instant (simulation only, unrealistic)
                 0└──── time

    LINEAR:       ┌────┐           Constant velocity (jerky start/stop)
                  │▓▓▓▓│
                 0└──── time

    TRAPEZOIDAL:  /────\           Accel → cruise → decel (most common)
                 /▓▓▓▓▓▓\
                0└──── time

    S_CURVE:     ╭────╮            Smooth jerk-limited (best quality)
                ╱▓▓▓▓▓▓╲
               0└──── time
    """
    STEP = "step"           # Instant (simulation default)
    LINEAR = "linear"       # Constant velocity
    TRAPEZOIDAL = "trapezoidal"  # Accel → cruise → decel
    S_CURVE = "s_curve"     # Smooth jerk-limited
```

#### 3.2.3 ControlAbstractionLayer (Main Interface)

```python
class ControlAbstractionLayer:
    """Central abstraction for all MuJoCo control interactions.

    This class is the ONLY place that should interact with MuJoCo primitives
    for control purposes. All other code should use this interface.

    Design principles:
    1. Actions are always normalized [-1, 1] at the policy level
    2. Joint symmetry is handled transparently
    3. Actuator semantics are abstracted away
    4. State access is consistent and well-defined
    """

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        robot_config: RobotConfig,
    ):
        self._mj_model = mj_model
        self._robot_config = robot_config

        # Build specs from model and config
        self._joint_specs = self._build_joint_specs()
        self._actuator_specs = self._build_actuator_specs()

        # Precompute JAX arrays for efficient transforms
        self._ctrl_centers = jnp.array([a.ctrl_range[0] + a.ctrl_range[1]) / 2
                                        for a in self._actuator_specs])
        self._ctrl_spans = jnp.array([(a.ctrl_range[1] - a.ctrl_range[0]) / 2
                                       for a in self._actuator_specs])
        self._mirror_signs = jnp.array([a.joint_spec.mirror_sign
                                         for a in self._actuator_specs])

    # =========================================================================
    # Action Transformation (Policy → MuJoCo)
    # =========================================================================
    #
    # DESIGN DECISION: Two explicit methods instead of boolean flag
    #
    # We use separate methods for different input types because:
    # 1. Policy actions: normalized [-1, 1], need symmetry correction
    # 2. Physical angles: already in radians (e.g., GMR reference motions)
    #
    # This prevents misuse and makes intent explicit at call site.
    # =========================================================================

    def policy_action_to_ctrl(self, action: jax.Array) -> jax.Array:
        """Transform normalized policy action to MuJoCo ctrl.

        USE FOR: Policy inference, training step, RL rollouts.

        This method:
        1. Applies symmetry correction (mirror_sign) for mirrored joint ranges
        2. Denormalizes from [-1, 1] to physical ctrl range

        Args:
            action: Normalized action in [-1, 1], shape (num_actuators,)
                    Same action value for left/right produces symmetric motion.

        Returns:
            ctrl: MuJoCo control signal (target angle in radians for position actuators)

        Example:
            # Policy outputs action[0] = 0.5 for left hip, action[4] = 0.5 for right hip
            # Both should move forward (symmetric motion)
            policy_action = jnp.array([0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
            ctrl = cal.policy_action_to_ctrl(policy_action)
            # ctrl accounts for mirrored joint ranges automatically:
            # - left_hip_pitch: 0.5 → +1.16 rad (forward)
            # - right_hip_pitch: 0.5 → -1.16 rad (also forward, due to mirror correction)
        """
        # Apply symmetry correction for mirrored joint ranges
        corrected = action * self._mirror_signs

        # Denormalize: [-1, 1] → actual control range
        ctrl = corrected * self._ctrl_spans + self._ctrl_centers
        return ctrl

    def physical_angles_to_ctrl(self, angles: jax.Array) -> jax.Array:
        """Pass physical joint angles directly to MuJoCo ctrl.

        USE FOR: GMR reference motions, mocap playback, direct joint tests.

        This method does NOT apply symmetry correction because:
        - Reference motions are already retargeted to robot skeleton
        - Joint angles are already in physical space (radians)
        - Left/right joints already have correct mirrored values

        Args:
            angles: Physical joint angles in radians, shape (num_actuators,)
                    Already in robot's joint space (e.g., from GMR ref_dof_pos)

        Returns:
            ctrl: MuJoCo control signal (same as input for position actuators)

        Example:
            # GMR reference motion gives physical angles
            ref_dof_pos = gmr_motion['dof_pos'][frame]  # Already in radians
            # ref_dof_pos might be: left_hip=0.8rad, right_hip=-0.8rad (physically correct)

            ctrl = cal.physical_angles_to_ctrl(ref_dof_pos)
            # ctrl = ref_dof_pos (no transformation for position actuators)
        """
        # For position actuators: ctrl = target angle (no transformation needed)
        # Future actuator types may need conversion here
        return angles

    def ctrl_to_policy_action(self, ctrl: jax.Array) -> jax.Array:
        """Transform MuJoCo ctrl to normalized policy action.

        Inverse of policy_action_to_ctrl().

        USE FOR:
        - Converting expert demonstrations to policy action space
        - Behavior cloning from physical trajectories
        - Debugging (what action would produce this ctrl?)

        Args:
            ctrl: MuJoCo control signal (radians for position actuators)

        Returns:
            action: Normalized action in [-1, 1] with symmetry correction reversed
        """
        # Normalize to [-1, 1]
        normalized = (ctrl - self._ctrl_centers) / self._ctrl_spans
        # Reverse mirror correction
        return normalized * self._mirror_signs

    def ctrl_to_physical_angles(self, ctrl: jax.Array) -> jax.Array:
        """Extract physical joint angles from MuJoCo ctrl.

        Inverse of physical_angles_to_ctrl().

        USE FOR:
        - Logging actual joint targets
        - Debugging ctrl values

        Args:
            ctrl: MuJoCo control signal

        Returns:
            angles: Physical joint angles in radians
        """
        # For position actuators: ctrl = target angle (no transformation needed)
        return ctrl

    # =========================================================================
    # State Access (MuJoCo → Observations)
    # =========================================================================
    #
    # DESIGN DECISION: Two explicit methods instead of boolean flag
    #
    # Same pattern as action transformation:
    # 1. Normalized: for policy observations (NN expects [-1, 1] or similar)
    # 2. Physical: for logging, debugging, reference comparison, Sim2Real
    # =========================================================================

    def get_normalized_joint_positions(self, qpos: jax.Array) -> jax.Array:
        """Get joint positions normalized to [-1, 1] based on joint limits.

        USE FOR: Policy observations (NN input).

        Args:
            qpos: Full qpos array from MuJoCo

        Returns:
            Normalized joint positions in [-1, 1], shape (num_actuators,)
            -1 = joint at minimum limit
             0 = joint at center of range
            +1 = joint at maximum limit
        """
        joint_pos = qpos[self._actuator_qpos_addrs]
        return (joint_pos - self._joint_centers) / self._joint_spans

    def get_physical_joint_positions(self, qpos: jax.Array) -> jax.Array:
        """Get joint positions in physical units (radians).

        USE FOR: Logging, debugging, reference motion comparison, Sim2Real.

        Args:
            qpos: Full qpos array from MuJoCo

        Returns:
            Joint positions in radians, shape (num_actuators,)
        """
        return qpos[self._actuator_qpos_addrs]

    def get_normalized_joint_velocities(self, qvel: jax.Array) -> jax.Array:
        """Get joint velocities normalized by per-joint velocity limits.

        USE FOR: Policy observations (NN input).

        Args:
            qvel: Full qvel array from MuJoCo

        Returns:
            Normalized joint velocities, shape (num_actuators,)
            Clipped to [-1, 1] range.

        Note:
            Velocity limits come from ActuatorSpec.max_velocity (motor specs).
            If not specified, defaults to 10.0 rad/s (~95 RPM).
        """
        joint_vel = qvel[self._actuator_qvel_addrs]
        normalized = joint_vel / self._velocity_limits
        return jnp.clip(normalized, -1.0, 1.0)

    def get_physical_joint_velocities(self, qvel: jax.Array) -> jax.Array:
        """Get joint velocities in physical units (rad/s).

        USE FOR: Logging, debugging, reference motion comparison, Sim2Real.

        Args:
            qvel: Full qvel array from MuJoCo

        Returns:
            Joint velocities in rad/s, shape (num_actuators,)
        """
        return qvel[self._actuator_qvel_addrs]

    def get_actuator_torques(
        self,
        data: mjx.Data,
        normalize: bool = True,
    ) -> jax.Array:
        """Get actuator torques/forces.

        Args:
            data: MJX data
            normalize: If True, normalize by force_range

        Returns:
            Actuator forces (num_actuators,), optionally normalized
        """
        torques = data.qfrc_actuator[self._actuator_qvel_addrs]

        if normalize:
            torques = torques / self._force_limits

        return torques

    # =========================================================================
    # Symmetry Utilities
    # =========================================================================

    def get_symmetric_action(
        self,
        action: jax.Array,
        swap_left_right: bool = True,
    ) -> jax.Array:
        """Get action for symmetric (mirrored) pose.

        Used for:
        - Data augmentation (mirror reference motions)
        - Symmetry tests (verify left/right equivalence)

        Args:
            action: Original action
            swap_left_right: Swap left/right joint values

        Returns:
            Mirrored action
        """
        if swap_left_right:
            # Swap left ↔ right indices
            mirrored = action[self._symmetry_swap_indices]
        else:
            mirrored = action

        # Apply sign correction for roll joints (which flip sign on mirror)
        mirrored = mirrored * self._symmetry_flip_signs
        return mirrored

    # =========================================================================
    # Trajectory Control (Sim2Real Bridge)
    # =========================================================================

    def policy_action_to_command(
        self,
        action: jax.Array,
        current_qpos: Optional[jax.Array] = None,
        ctrl_dt: float = 0.02,  # Policy control period (50 Hz typical)
        include_velocity: bool = False,
    ) -> ControlCommand:
        """Transform normalized policy action to control command with trajectory info.

        USE FOR: Policy inference, RL rollouts, Sim2Real deployment.

        This is the PRIMARY method for Sim2Real deployment with policy outputs.
        Applies symmetry correction automatically.

        In Simulation:
            - Use target_position only (velocity is emergent)
            - MuJoCo PD controller handles trajectory

        In Real Hardware:
            - Use target_position + target_velocity
            - Motor controller executes trajectory

        Args:
            action: Normalized action [-1, 1], shape (num_actuators,)
            current_qpos: Current joint positions (needed for velocity calc)
            ctrl_dt: Control period in seconds (time to reach target)
            include_velocity: If True, compute target velocity (linear profile)

        Returns:
            ControlCommand with target_position and optional velocity

        Example (Simulation):
            cmd = cal.policy_action_to_command(action)
            data = data.replace(ctrl=cmd.target_position)

        Example (Real Hardware):
            cmd = cal.policy_action_to_command(
                action,
                current_qpos=encoder_readings,
                ctrl_dt=0.02,
                include_velocity=True
            )
            for i, motor in enumerate(motors):
                motor.set_position(
                    position=cmd.target_position[i],
                    velocity=cmd.target_velocity[i]  # Required for real motors!
                )
        """
        # Get target position with symmetry correction
        target_position = self.policy_action_to_ctrl(action)

        # Compute velocity if requested (required for real hardware)
        target_velocity = None
        if include_velocity and current_qpos is not None:
            target_velocity = self._compute_velocity(
                target_position, current_qpos, ctrl_dt
            )

        return ControlCommand(
            target_position=target_position,
            target_velocity=target_velocity,
            duration=ctrl_dt if include_velocity else None,
        )

    def physical_angles_to_command(
        self,
        angles: jax.Array,
        current_qpos: Optional[jax.Array] = None,
        ctrl_dt: float = 0.02,
        include_velocity: bool = False,
    ) -> ControlCommand:
        """Transform physical joint angles to control command with trajectory info.

        USE FOR: GMR reference motion playback, mocap replay, direct joint tests.

        Does NOT apply symmetry correction (angles are already in physical space).

        Args:
            angles: Physical joint angles in radians, shape (num_actuators,)
            current_qpos: Current joint positions (needed for velocity calc)
            ctrl_dt: Control period in seconds (time to reach target)
            include_velocity: If True, compute target velocity (linear profile)

        Returns:
            ControlCommand with target_position and optional velocity

        Example (GMR Reference Playback):
            for frame in range(num_frames):
                ref_angles = gmr_motion['dof_pos'][frame]
                cmd = cal.physical_angles_to_command(
                    ref_angles,
                    current_qpos=mj_data.qpos,
                    ctrl_dt=0.02,
                    include_velocity=True
                )
                # For simulation: just use position
                mj_data.ctrl[:] = cmd.target_position
                # For real hardware: use both
                # motor.set_position(cmd.target_position, cmd.target_velocity)
        """
        # Get target position (no transformation for position actuators)
        target_position = self.physical_angles_to_ctrl(angles)

        # Compute velocity if requested
        target_velocity = None
        if include_velocity and current_qpos is not None:
            target_velocity = self._compute_velocity(
                target_position, current_qpos, ctrl_dt
            )

        return ControlCommand(
            target_position=target_position,
            target_velocity=target_velocity,
            duration=ctrl_dt if include_velocity else None,
        )

    def _compute_velocity(
        self,
        target_position: jax.Array,
        current_qpos: jax.Array,
        ctrl_dt: float,
    ) -> jax.Array:
        """Compute target velocity for trajectory control.

        Uses linear velocity profile: constant velocity to reach target in ctrl_dt.

        Args:
            target_position: Target joint positions (radians)
            current_qpos: Current full qpos array
            ctrl_dt: Time to reach target (seconds)

        Returns:
            Target velocity (rad/s) for each actuator

        Note:
            Currently only linear profile is implemented.
            Future work may add trapezoidal/S-curve for smoother motion.
        """
        current_pos = current_qpos[self._actuator_qpos_addrs]
        position_delta = target_position - current_pos
        return position_delta / ctrl_dt

    def get_velocity_limits(self) -> jax.Array:
        """Get maximum velocity limits for each actuator.

        These limits come from motor specifications and are critical
        for Sim2Real safety.

        Returns:
            Maximum angular velocity (rad/s) for each actuator
        """
        # TODO: Load from robot_config.yaml or motor specs
        # Default: conservative limit based on typical servos
        default_max_vel = 5.0  # rad/s (approx 50 RPM)
        return jnp.full(len(self._actuator_specs), default_max_vel)

    # =========================================================================
    # Validation & Debugging
    # =========================================================================

    def validate_action(self, action: jax.Array) -> bool:
        """Check if action is within expected bounds."""
        return jnp.all((action >= -1.0) & (action <= 1.0))

    def validate_command(
        self,
        cmd: ControlCommand,
        check_velocity: bool = True,
    ) -> Tuple[bool, List[str]]:
        """Validate control command for safety (useful for Sim2Real).

        Args:
            cmd: ControlCommand to validate
            check_velocity: If True, check velocity limits

        Returns:
            (is_valid, list_of_violations)
        """
        violations = []

        # Check position limits
        ctrl_mins = jnp.array([a.ctrl_range[0] for a in self._actuator_specs])
        ctrl_maxs = jnp.array([a.ctrl_range[1] for a in self._actuator_specs])

        if jnp.any(cmd.target_position < ctrl_mins):
            violations.append("Position below minimum")
        if jnp.any(cmd.target_position > ctrl_maxs):
            violations.append("Position above maximum")

        # Check velocity limits (critical for real hardware safety)
        if check_velocity and cmd.target_velocity is not None:
            vel_limits = self.get_velocity_limits()
            if jnp.any(jnp.abs(cmd.target_velocity) > vel_limits):
                violations.append("Velocity exceeds motor limits")

        return len(violations) == 0, violations

    def get_ctrl_for_default_pose(self) -> jax.Array:
        """Get ctrl values that produce the default (keyframe) pose."""
        # For position actuators, ctrl = target position
        return jnp.array([spec.joint_spec.default_pos
                          for spec in self._actuator_specs])
```

---

## 4. Integration with Existing Components

### 4.1 WildRobotEnv Integration

**Current Flow** (problematic):
```python
# wildrobot_env.py line 524
data = data.replace(ctrl=self._default_joint_qpos)

# wildrobot_env.py line 712
data = state.data.replace(ctrl=filtered_action)
```

**Proposed Flow**:
```python
class WildRobotEnv(mjx_env.MjxEnv):
    def __init__(self, config: TrainingConfig, ...):
        ...
        # Create CAL as the ONLY interface to MuJoCo control
        self._cal = ControlAbstractionLayer(
            mj_model=self._mj_model,
            robot_config=self._robot_config,
        )

    def reset(self, rng: jax.Array) -> WildRobotEnvState:
        ...
        # Use CAL for default ctrl
        default_ctrl = self._cal.get_ctrl_for_default_pose()
        data = data.replace(ctrl=default_ctrl)
        ...

    def step(self, state: WildRobotEnvState, action: jax.Array) -> WildRobotEnvState:
        # Action is normalized [-1, 1] from policy
        # CAL handles denormalization + symmetry correction
        ctrl = self._cal.policy_action_to_ctrl(action)
        data = state.data.replace(ctrl=ctrl)
        ...

    def _get_obs(self, data: mjx.Data, action: jax.Array, ...) -> jax.Array:
        # Use CAL for state access (consistent normalization)
        joint_pos = self._cal.get_normalized_joint_positions(data.qpos)
        joint_vel = self._cal.get_normalized_joint_velocities(data.qvel)
        ...
```

### 4.2 Training Pipeline Integration

**Current Flow**:
```python
# rollout.py
action, raw_action, log_prob = sample_actions(...)
next_env_state = env_step_fn(env_state, action)
```

**Proposed Flow** (no change needed at training level):
The CAL is internal to the environment. Training sees:
- Actions: always normalized [-1, 1]
- Observations: always normalized and consistent
- The CAL handles all simulator-specific details

**Benefit**: Training code is simulator-agnostic. Switch from MuJoCo to another simulator by swapping the CAL implementation.

### 4.3 Test Suite Integration

**Current Flow** (problematic):
```python
# test_physics_validation.py
mj_data.ctrl[act_idx] = 0.1  # What unit? Position? Torque?
```

**Proposed Flow**:
```python
class TestActuatorCorrectness:
    @pytest.fixture
    def cal(self, env):
        """Get CAL from environment for testing."""
        return env._cal

    def test_actuator_isolation(self, cal, mj_model, mj_data):
        # Use CAL for action → ctrl conversion
        action = jnp.zeros(cal.num_actuators)
        action = action.at[act_idx].set(0.5)  # 50% of range, clear semantics

        ctrl = cal.action_to_ctrl(action, use_symmetry_correction=False)
        mj_data.ctrl[:] = np.array(ctrl)
        ...

    def test_symmetry_invariance(self, cal, env):
        """Test that symmetric actions produce symmetric motion."""
        # Left-side action
        action_left = jnp.array([0.5, 0.0, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0])

        # Get symmetric right-side action via CAL
        action_right = cal.get_symmetric_action(action_left, swap_left_right=True)

        # Both should produce symmetric physical motion
        # (This test would FAIL with current code due to range mirroring)
        ...
```

### 4.4 Sim2Real Integration

**Current Challenge**: Policies trained in MuJoCo learn simulator-specific quirks.

**With CAL**:
```python
class RealRobotCAL(ControlAbstractionLayer):
    """CAL implementation for real robot hardware.

    Same interface as simulation CAL, different backend.
    """

    def action_to_ctrl(self, action: jax.Array, ...) -> jax.Array:
        # Same normalization logic
        ctrl = action * self._ctrl_spans + self._ctrl_centers

        # Hardware-specific conversions (e.g., servo protocol)
        return self._convert_to_servo_command(ctrl)

    def get_joint_positions(self, ...) -> jax.Array:
        # Read from real encoders
        raw_positions = self._read_encoder_positions()
        return self._normalize_positions(raw_positions)
```

**Benefit**: Same policy code works in sim and real. Only the CAL implementation changes.

---

## 5. Symmetry Handling Details

### 5.1 Problem: Range Mirroring

For WildRobot, left/right joints have mirrored ranges:

| Joint | Min | Max | Center | Span |
|-------|-----|-----|--------|------|
| left_hip_pitch | -0.087 | 1.57 | 0.74 | 0.83 |
| right_hip_pitch | -1.57 | 0.087 | -0.74 | 0.83 |
| left_hip_roll | -1.57 | 0.17 | -0.70 | 0.87 |
| right_hip_roll | -0.17 | 1.57 | 0.70 | 0.87 |

**Without CAL**: Policy action `0.5` produces:
- Left hip pitch: `0.5 * 0.83 + 0.74 = 1.16 rad` (forward)
- Right hip pitch: `0.5 * 0.83 + (-0.74) = -0.33 rad` (backward!)

**With CAL** (mirror_sign correction):
```python
# In JointSpec initialization
left_hip_pitch.mirror_sign = +1.0
right_hip_pitch.mirror_sign = -1.0  # Inverted!

# In action_to_ctrl:
action_corrected = action * self._mirror_signs
# right_hip_pitch: 0.5 * (-1.0) = -0.5
# ctrl: -0.5 * 0.83 + (-0.74) = -1.16 rad (forward, matching left!)
```

### 5.2 Configuration Schema Refactoring

The current `robot_config.yaml` should be refactored to align with CAL. Key changes:
1. **Merge joint + actuator** - Single definition since they're 1:1 mapped
2. **Add symmetry fields** - `symmetry_pair` and `mirror_sign`
3. **Add motor specs** - `max_velocity` for Sim2Real

#### Current Schema (before CAL):
```yaml
# Redundant: separate actuators and joints sections
actuators:
  details:
  - name: left_hip_pitch
    joint: left_hip_pitch  # 1:1 mapping!
    type: position
    class: htd45hServo

joints:
  details:
  - name: left_hip_pitch  # Duplicated!
    range: [-0.087, 1.571]
```

#### Proposed Schema (CAL-aligned, Simplified):
```yaml
# =============================================================================
# ACTUATED JOINTS (single definition → generates JointSpec + ActuatorSpec)
# =============================================================================
# Each entry defines BOTH the joint AND its actuator (1:1 mapping)
#
# Auto-extracted from MuJoCo model at runtime:
# - ctrl_range, force_range, kp, kv (from actuator defaults)
# - default_pos (from "home" keyframe or qpos0)
# =============================================================================

actuated_joints:
  - name: left_hip_pitch
    type: position                    # ActuatorType.POSITION
    class: htd45hServo
    range: [-0.087, 1.571]            # Joint range (radians)
    # Symmetry
    symmetry_pair: right_hip_pitch    # Paired joint for mirroring
    mirror_sign: 1.0                  # +1 = same sign, -1 = inverted
    # Motor specs (from datasheet)
    max_velocity: 10.0                # rad/s - PLACEHOLDER
    # NOTE: default_pos loaded from MuJoCo keyframe at runtime

  - name: left_hip_roll
    type: position
    class: htd45hServo
    range: [-1.571, 0.175]
    symmetry_pair: right_hip_roll
    mirror_sign: 1.0
    max_velocity: 10.0

  - name: left_knee_pitch
    type: position
    class: htd45hServo
    range: [0.0, 1.396]
    symmetry_pair: right_knee_pitch
    mirror_sign: 1.0                  # Same direction (not mirrored)
    max_velocity: 10.0

  - name: left_ankle_pitch
    type: position
    class: htd45hServo
    range: [-0.785, 0.785]
    symmetry_pair: right_ankle_pitch
    mirror_sign: 1.0
    max_velocity: 10.0

  - name: right_hip_pitch
    type: position
    class: htd45hServo
    range: [-1.571, 0.087]            # MIRRORED range!
    symmetry_pair: left_hip_pitch
    mirror_sign: -1.0                 # INVERTED due to mirrored range
    max_velocity: 10.0

  - name: right_hip_roll
    type: position
    class: htd45hServo
    range: [-0.175, 1.571]            # MIRRORED range!
    symmetry_pair: left_hip_roll
    mirror_sign: -1.0                 # INVERTED
    max_velocity: 10.0

  - name: right_knee_pitch
    type: position
    class: htd45hServo
    range: [0.0, 1.396]
    symmetry_pair: left_knee_pitch
    mirror_sign: 1.0                  # Same direction
    max_velocity: 10.0

  - name: right_ankle_pitch
    type: position
    class: htd45hServo
    range: [-0.785, 0.785]
    symmetry_pair: left_ankle_pitch
    mirror_sign: 1.0
    max_velocity: 10.0
```

#### CAL Initialization (from unified config):
```python
class ControlAbstractionLayer:
    def __init__(self, mj_model, robot_config):
        self._mj_model = mj_model
        self._robot_config = robot_config

        # Build specs from config
        self._build_specs(robot_config)

        # Load default positions from MuJoCo keyframe (NOT from config)
        self._default_qpos = self._load_default_positions()

    def _build_specs(self, config: dict):
        """Build JointSpec and ActuatorSpec from unified config."""
        for i, entry in enumerate(config['actuated_joints']):
            # Create JointSpec (default_pos loaded separately from MuJoCo)
            joint_spec = JointSpec(
                name=entry['name'],
                joint_id=self._get_joint_id(entry['name']),
                qpos_addr=...,
                qvel_addr=...,
                range_min=entry['range'][0],
                range_max=entry['range'][1],
                default_pos=0.0,  # Placeholder, loaded from MuJoCo
                mirror_sign=entry['mirror_sign'],
                symmetry_pair=entry['symmetry_pair'],
            )

            # Create ActuatorSpec (references the JointSpec)
            actuator_spec = ActuatorSpec(
                name=entry['name'],
                actuator_id=i,
                joint_spec=joint_spec,
                actuator_type=ActuatorType(entry['type']),
                ctrl_range=self._mj_model.actuator_ctrlrange[i],  # Auto-extract
                force_range=self._mj_model.actuator_forcerange[i],  # Auto-extract
                max_velocity=entry.get('max_velocity', 10.0),
                kp=...,  # From MuJoCo model
                kv=...,  # From MuJoCo model
            )

            self._joint_specs.append(joint_spec)
            self._actuator_specs.append(actuator_spec)

    def _load_default_positions(self) -> jnp.ndarray:
        """Load default joint positions from MuJoCo model.

        Priority:
        1. "home" keyframe if exists
        2. qpos0 (model default)

        This is the SINGLE SOURCE OF TRUTH for default pose.
        No duplication in robot_config.yaml.
        """
        mj_model = self._mj_model

        # Check if "home" keyframe exists
        if mj_model.nkey > 0:
            # Find "home" keyframe by name
            for key_idx in range(mj_model.nkey):
                # Get keyframe name (MuJoCo stores names in a flat array)
                key_name_adr = mj_model.name_keyadr[key_idx]
                key_name = mj_model.names[key_name_adr:].split(b'\x00')[0].decode()

                if key_name == "home":
                    # Extract actuated joint positions from keyframe
                    full_qpos = mj_model.key_qpos[key_idx]
                    return jnp.array(full_qpos[self._actuator_qpos_addrs])

        # Fallback to qpos0 (model default)
        return jnp.array(mj_model.qpos0[self._actuator_qpos_addrs])

    def get_ctrl_for_default_pose(self) -> jax.Array:
        """Get ctrl values for default (home) pose.

        Loaded from MuJoCo keyframe at initialization.
        """
        return self._default_qpos
```

#### Config Generation: `post_process.py` Integration

Add to `post_process.py` to auto-generate `actuated_joints` section:

```python
def generate_actuated_joints_config(
    robot_xml: str,
) -> List[Dict[str, Any]]:
    """Generate actuated_joints config from MuJoCo XML file.

    Extracts joint/actuator info from robot XML.
    Auto-computes mirror_sign from joint ranges.

    NOTE: default_pos is NOT extracted here - it's loaded from MuJoCo
    keyframe at CAL initialization time (single source of truth).

    Args:
        robot_xml: Path to robot definition (e.g., wildrobot.xml)

    Returns:
        List of actuated joint configs for robot_config.yaml
    """
    import xml.etree.ElementTree as ET
    import numpy as np

    robot_tree = ET.parse(robot_xml)
    robot_root = robot_tree.getroot()

    # =========================================================================
    # Extract joint ranges from robot XML
    # =========================================================================
    joint_ranges = {}
    for joint in robot_root.findall(".//joint"):
        name = joint.get("name")
        if name is None:
            continue
        range_str = joint.get("range")
        if range_str:
            range_vals = [float(x) for x in range_str.split()]
            joint_ranges[name] = range_vals

    # =========================================================================
    # Extract actuator info
    # =========================================================================
    actuators = []
    actuator_section = robot_root.find("actuator")
    if actuator_section is not None:
        for actuator in actuator_section:
            name = actuator.get("name")
            joint = actuator.get("joint")
            act_type = actuator.tag  # "position", "motor", etc.
            act_class = actuator.get("class")
            actuators.append({
                "name": name,
                "joint": joint,
                "type": act_type,
                "class": act_class,
            })

    # =========================================================================
    # Define symmetry pairs (robot-specific, could be config-driven)
    # =========================================================================
    symmetry_pairs = {
        "left_hip_pitch": "right_hip_pitch",
        "left_hip_roll": "right_hip_roll",
        "left_knee_pitch": "right_knee_pitch",
        "left_ankle_pitch": "right_ankle_pitch",
        "right_hip_pitch": "left_hip_pitch",
        "right_hip_roll": "left_hip_roll",
        "right_knee_pitch": "left_knee_pitch",
        "right_ankle_pitch": "left_ankle_pitch",
    }

    # =========================================================================
    # Auto-compute mirror_sign from joint ranges
    # =========================================================================
    def compute_mirror_sign(name: str) -> float:
        """Auto-detect if joint range is mirrored relative to its pair."""
        if name not in symmetry_pairs:
            return 1.0

        pair_name = symmetry_pairs[name]

        if name not in joint_ranges or pair_name not in joint_ranges:
            return 1.0

        my_range = joint_ranges[name]
        pair_range = joint_ranges[pair_name]

        my_center = (my_range[0] + my_range[1]) / 2
        pair_center = (pair_range[0] + pair_range[1]) / 2

        # If centers have opposite signs, ranges are mirrored
        if my_center * pair_center < 0:
            return -1.0
        return 1.0

    # =========================================================================
    # Build actuated_joints config
    # =========================================================================
    actuated_joints = []
    for act in actuators:
        name = act["name"]
        joint_name = act["joint"] or name  # Usually same as actuator name

        entry = {
            "name": name,
            "type": act["type"],
            "class": act["class"],
            "range": joint_ranges.get(joint_name, [-1.57, 1.57]),
            # NOTE: default_pos NOT included - loaded from MuJoCo keyframe at runtime
            "symmetry_pair": symmetry_pairs.get(name),
            "mirror_sign": compute_mirror_sign(name),
            "max_velocity": 10.0,  # PLACEHOLDER - update from motor datasheet
        }
        actuated_joints.append(entry)

    return actuated_joints


def generate_robot_config(
    xml_file: str,
    output_file: str = "robot_config.yaml",
) -> Dict[str, Any]:
    """Generate robot configuration from MuJoCo XML.

    Updated to include actuated_joints section for CAL.

    NOTE: default_pos is loaded from MuJoCo keyframe at CAL runtime,
    not saved to this config file.
    """
    # ... existing code ...

    # NEW: Generate actuated_joints for CAL
    config["actuated_joints"] = generate_actuated_joints_config(
        robot_xml=xml_file,
    )

    # ... rest of existing code ...
```

#### Home Frame Management: Use Dedicated `keyframes.xml`

**Current Problem**: Home frame is in `scene_flat_terrain.xml`, which is:
- Scene-specific (what if we have multiple scenes?)
- Not clearly associated with the robot
- Hard to find

**Solution**: Create dedicated `keyframes.xml` in assets:

```xml
<!-- assets/keyframes.xml -->
<mujoco>
    <!--
    Robot keyframes (default poses).

    These are used by:
    - CAL: default_pos loaded at runtime via _load_default_positions()
    - Environment: reset pose
    - Visualization: reference pose

    qpos layout for WildRobot:
    [0:3]   - root position (x, y, z)
    [3:7]   - root orientation (quat: w, x, y, z)
    [7:15]  - actuated joints (8 DOFs):
              left_hip_pitch, left_hip_roll, left_knee_pitch, left_ankle_pitch,
              right_hip_pitch, right_hip_roll, right_knee_pitch, right_ankle_pitch
    -->

    <keyframe>
        <!-- Standing pose with slight knee bend (stable default) -->
        <key name="home"
             qpos="-0.00669872 -0.000211667 0.469912
                   0.999882 0.00198177 -0.00846523 -0.0126691
                   -0.00513143 0.00469937 0.00609997 0.00672666
                   -0.00524059 0.00264562 -0.000115484 -0.00199531"/>

        <!-- T-pose for visualization/debugging -->
        <key name="t_pose"
             qpos="0 0 0.5
                   1 0 0 0
                   0 0 0 0
                   0 0 0 0"/>

        <!-- Squat pose for sit-to-stand training -->
        <key name="squat"
             qpos="0 0 0.35
                   1 0 0 0
                   0.3 0 0.8 -0.5
                   0.3 0 0.8 -0.5"/>
    </keyframe>
</mujoco>
```

**Update scene files to include keyframes**:

```xml
<!-- assets/scene_flat_terrain.xml -->
<mujoco model="scene">
    <include file="wildrobot.xml" />
    <include file="keyframes.xml" />  <!-- Keyframes now separate -->

    <visual>...</visual>
    <asset>...</asset>
    <worldbody>...</worldbody>

    <!-- No keyframe here anymore - it's in keyframes.xml -->
</mujoco>
```

**Benefits of `keyframes.xml`**:
1. **Clear ownership**: Keyframes are clearly robot config, not scene config
2. **Reusable**: All scenes can include the same keyframes
3. **Easy to edit**: One place to update default poses
4. **Version control**: Changes to poses are clearly tracked
5. **Single source of truth**: CAL loads directly from MuJoCo at runtime

#### Updated `post_process.py` main():

```python
def main() -> None:
    xml_file = "wildrobot.xml"
    keyframe_file = "keyframes.xml"  # NEW: dedicated keyframe file

    print("start post process...")
    add_collision_names(xml_file)
    ensure_root_body_pose(xml_file)
    add_option(xml_file)
    merge_default_blocks(xml_file)
    fix_collision_default_geom(xml_file)
    add_mimic_bodies(xml_file)

    # Generate robot configuration with CAL support
    generate_robot_config(
        xml_file,
        keyframe_file=keyframe_file,
        output_file="robot_config.yaml"
    )

    print("Post process completed")
```

#### Schema Changes Summary:

| Before | After | Benefit |
|--------|-------|---------|
| Separate `actuators` + `joints` | Single `actuated_joints` | No redundancy |
| Joint name duplicated | Single definition | Single source of truth |
| Separate symmetry section | Inline `symmetry_pair`, `mirror_sign` | No redundancy |
| No motor specs | `max_velocity` per joint | Sim2Real ready |

#### Auto-Detection Option:

`mirror_sign` can be **auto-computed** during CAL initialization:

```python
def _compute_mirror_signs(self):
    """Auto-detect mirror_sign from joint ranges."""
    for left_name, right_name in self._get_symmetry_pairs():
        left = self._joint_specs_by_name[left_name]
        right = self._joint_specs_by_name[right_name]

        # If centers have opposite signs, ranges are mirrored
        if left.range_center * right.range_center < 0:
            right.mirror_sign = -1.0
        else:
            right.mirror_sign = 1.0
```

This makes explicit `mirror_sign` config optional - CAL can infer it from ranges.

---

## 6. File Structure

### 6.1 New Files

```
playground_amp/
├── control/                        # NEW: CAL module
│   ├── __init__.py                 # Export CAL, specs, types
│   ├── cal.py                      # ControlAbstractionLayer class
│   ├── specs.py                    # JointSpec, ActuatorSpec, ControlCommand
│   └── types.py                    # ActuatorType, VelocityProfile enums
│
├── envs/
│   └── wildrobot_env.py            # UPDATE: integrate CAL
│
├── configs/
│   └── robot_config.py             # UPDATE: add actuated_joints support
│
├── tests/
│   └── test_cal.py                 # NEW: CAL unit tests
│
└── docs/
    └── CONTROL_ABSTRACTION_LAYER_PROPOSAL.md  # This document

assets/
├── keyframes.xml                   # NEW: dedicated keyframe file
├── post_process.py                 # UPDATE: add generate_actuated_joints_config()
├── robot_config.yaml               # UPDATE: add actuated_joints section
└── ...
```

### 6.2 File Descriptions

| File | Location | Purpose |
|------|----------|---------|
| `cal.py` | `playground_amp/control/` | Main ControlAbstractionLayer class with all transformation methods |
| `specs.py` | `playground_amp/control/` | JointSpec, ActuatorSpec, ControlCommand dataclasses |
| `types.py` | `playground_amp/control/` | ActuatorType, VelocityProfile enums |
| `__init__.py` | `playground_amp/control/` | Module exports |
| `test_cal.py` | `playground_amp/tests/` | Unit tests for CAL |
| `keyframes.xml` | `assets/` | Home pose, t-pose, squat keyframes |
| `post_process.py` | `assets/` | Add `generate_actuated_joints_config()` function |
| `robot_config.yaml` | `assets/` | Add `actuated_joints` section |
| `wildrobot_env.py` | `playground_amp/envs/` | Replace direct ctrl access with CAL |
| `robot_config.py` | `playground_amp/configs/` | Support new actuated_joints schema |

### 6.3 Implementation Order

```
Phase 1: Create CAL Module
├── 1. playground_amp/control/types.py      # Enums first (no dependencies)
├── 2. playground_amp/control/specs.py      # Dataclasses (depends on types)
├── 3. playground_amp/control/cal.py        # Main class (depends on specs)
└── 4. playground_amp/control/__init__.py   # Exports

Phase 2: Create Keyframes
└── 5. assets/keyframes.xml                 # Extract from scene_flat_terrain.xml

Phase 3: Update Config Generation
├── 6. assets/post_process.py               # Add generate_actuated_joints_config()
└── 7. assets/robot_config.yaml             # Regenerate with actuated_joints

Phase 4: Environment Integration
└── 8. playground_amp/envs/wildrobot_env.py # Integrate CAL

Phase 5: Tests
└── 9. playground_amp/tests/test_cal.py     # Unit tests
```

---

## 7. Migration Plan

### Phase 1: Foundation (Week 1)
1. Create `ControlAbstractionLayer` class
2. Create `JointSpec` and `ActuatorSpec` dataclasses
3. Add symmetry configuration to `robot_config.yaml`
4. Unit tests for CAL transforms

### Phase 2: Environment Integration (Week 2)
1. Integrate CAL into `WildRobotEnv.__init__`
2. Replace direct `ctrl` access with CAL methods
3. Replace direct `qpos`/`qvel` access in observations
4. Verify training still works (reward curves match)

### Phase 3: Test Suite Update (Week 3)
1. Update `test_physics_validation.py` to use CAL
2. Add symmetry invariance tests
3. Remove tests that rely on simulator-specific behavior

### Phase 4: Validation (Week 4)
1. Train policy with CAL-based environment
2. Compare with baseline (should be identical or better)
3. Test policy on symmetric poses
4. Document API and migration guide

---

## 8. Success Criteria

1. **No Direct MuJoCo Access**: All `data.ctrl`, `data.qpos`, `data.qvel` access goes through CAL
2. **Symmetry Correctness**: `action[left] = action[right]` produces symmetric motion
3. **Training Parity**: Reward curves with CAL match baseline (within noise)
4. **Test Clarity**: Tests use semantic actions (50% of range), not raw values (0.1 rad)
5. **Sim2Real Ready**: CAL interface can be implemented for real hardware

---

## 9. Appendix: Joint Range Analysis

```
Joint                | Range Min  | Range Max  | Center    | Span     | Mirror Sign
---------------------|------------|------------|-----------|----------|------------
left_hip_pitch       | -0.087     | 1.571      | 0.742     | 0.829    | +1
right_hip_pitch      | -1.571     | 0.087      | -0.742    | 0.829    | -1 (flip!)
left_hip_roll        | -1.571     | 0.175      | -0.698    | 0.873    | +1
right_hip_roll       | -0.175     | 1.571      | 0.698     | 0.873    | -1 (flip!)
left_knee_pitch      | 0.000      | 1.396      | 0.698     | 0.698    | +1
right_knee_pitch     | 0.000      | 1.396      | 0.698     | 0.698    | +1 (same)
left_ankle_pitch     | -0.785     | 0.785      | 0.000     | 0.785    | +1
right_ankle_pitch    | -0.785     | 0.785      | 0.000     | 0.785    | +1 (same)
```

**Observation**: Hip joints have mirrored ranges (physical construction), while knee and ankle are symmetric. CAL handles this automatically.

---

## 10. References

1. **D76312096**: [robot][mujoco] Fixed actuator of type position - Key insight on position actuator semantics
2. **D77383084**: [robot][MuJoCo] Reworked JointState ingestion - Motor vs position actuator handling
3. **Cartpole Benchmark**: Action space normalization issues in RLlib
4. **MetaMend Post (Workplace)**: HAL abstraction pattern
5. **dm_control**: action_spec with bounds
6. **MuJoCo Documentation**: Position actuator PD controller semantics
