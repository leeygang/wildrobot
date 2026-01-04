# Control Abstraction Layer (CAL)

## Executive Summary

This document describes the **Control Abstraction Layer (CAL)** that decouples high-level flows (training, testing, sim2real) from MuJoCo primitives. This addresses the issues of ambiguous actuator semantics, incorrect symmetry assumptions, and training pipelines learning simulator quirks.

**Current Version**: v0.11.0

**Module Location**: `playground_amp/cal/`

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
    # DESIGN DECISION: Single method with REQUIRED normalize parameter
    #
    # Since almost all state access needs both normalized and physical versions,
    # we use a single method with a REQUIRED normalize parameter (no default)
    # to force explicit intent at every call site:
    # - normalize=True: for policy observations (NN expects [-1, 1] or similar)
    # - normalize=False: for logging, debugging, reference comparison, Sim2Real
    # =========================================================================

    def get_joint_positions(
        self,
        qpos: jax.Array,
        normalize: bool,  # REQUIRED - no default for consistency
    ) -> jax.Array:
        """Get joint positions.

        Args:
            qpos: Full qpos array from MuJoCo
            normalize: REQUIRED. True for normalized [-1, 1] based on joint limits,
                      False for physical positions (radians)

        Returns:
            Joint positions, shape (num_actuators,)
            If normalized: -1 = min limit, 0 = center, +1 = max limit
            If physical: radians
        """
        joint_pos = qpos[self._actuator_qpos_addrs]

        if normalize:
            return (joint_pos - self._joint_centers) / self._joint_spans
        else:
            return joint_pos

    def get_joint_velocities(
        self,
        qvel: jax.Array,
        normalize: bool,  # REQUIRED - no default for consistency
    ) -> jax.Array:
        """Get joint velocities.

        Args:
            qvel: Full qvel array from MuJoCo
            normalize: REQUIRED. True for normalized by velocity limits (clipped to [-1, 1]),
                      False for physical velocities (rad/s)

        Returns:
            Joint velocities, shape (num_actuators,)

        Note:
            Velocity limits come from ActuatorSpec.max_velocity (motor specs).
            If not specified, defaults to 10.0 rad/s (~95 RPM).
        """
        joint_vel = qvel[self._actuator_qvel_addrs]

        if normalize:
            normalized = joint_vel / self._velocity_limits
            return jnp.clip(normalized, -1.0, 1.0)
        else:
            return joint_vel

    def get_actuator_torques(
        self,
        qfrc_actuator: jax.Array,
        normalize: bool,  # REQUIRED - no default for consistency
    ) -> jax.Array:
        """Get actuator torques/forces.

        Args:
            qfrc_actuator: Full qfrc_actuator array from MuJoCo data
            normalize: REQUIRED. True to normalize by force_range,
                      False for raw torques (Nm)

        Returns:
            Actuator forces (num_actuators,), optionally normalized
        """
        torques = qfrc_actuator[self._actuator_qvel_addrs]

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

**Concrete Examples**

Hip roll (mirrored ranges, mirror_sign flips):
- left_hip_roll range: [-1.57, 0.17], center ≈ -0.70, span ≈ 0.87
- right_hip_roll range: [-0.17, 1.57], center ≈ +0.70, span ≈ 0.87
- mirror_sign: left = +1, right = -1

Action = +0.5 for both:
- left ctrl ≈ -0.70 + 0.5*0.87 = -0.26 rad
- right ctrl ≈ +0.70 + (-0.5)*0.87 = +0.26 rad
Result: equal magnitude, opposite sign (symmetric roll).

Ankle pitch (identical ranges, no flip):
- left_ankle_pitch range: [-0.785, 0.785], mirror_sign = +1
- right_ankle_pitch range: [-0.785, 0.785], mirror_sign = +1

Action = +0.5 for both:
- left ctrl ≈ 0.5*0.785 = +0.393 rad
- right ctrl ≈ 0.5*0.785 = +0.393 rad
Result: same sign because ranges are identical.

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

### 6.1 Module Files

```
playground_amp/
├── cal/                            # CAL module
│   ├── __init__.py                 # Export CAL, specs, types
│   ├── cal.py                      # ControlAbstractionLayer class
│   ├── specs.py                    # JointSpec, ActuatorSpec, ControlCommand, Pose3D, Velocity3D
│   ├── types.py                    # ActuatorType, VelocityProfile, CoordinateFrame enums
│   └── cal.md                      # This documentation
│
├── envs/
│   └── wildrobot_env.py            # UPDATE: integrate CAL
│
├── configs/
│   └── robot_config.py             # UPDATE: add actuated_joints support
│
├── tests/
│   ├── test_cal.py                 # CAL unit tests
│   └── test_physics_validation.py  # Physics validation tests

assets/
├── keyframes.xml                   # NEW: dedicated keyframe file
├── post_process.py                 # UPDATE: add generate_actuated_joints_config()
├── robot_config.yaml               # UPDATE: add actuated_joints section
└── ...
```

### 6.2 File Descriptions

| File | Location | Purpose |
|------|----------|----------|
| `cal.py` | `playground_amp/cal/` | Main ControlAbstractionLayer class with all transformation methods |
| `specs.py` | `playground_amp/cal/` | JointSpec, ActuatorSpec, ControlCommand, Pose3D, Velocity3D, FootSpec, RootSpec |
| `types.py` | `playground_amp/cal/` | ActuatorType, VelocityProfile, CoordinateFrame enums |
| `__init__.py` | `playground_amp/cal/` | Module exports |
| `cal.md` | `playground_amp/cal/` | This documentation |
| `test_cal.py` | `playground_amp/tests/` | Unit tests for CAL |
| `keyframes.xml` | `assets/` | Home pose, t-pose, squat keyframes |
| `post_process.py` | `assets/` | Add `generate_actuated_joints_config()` function |
| `robot_config.yaml` | `assets/` | Add `actuated_joints` section |
| `wildrobot_env.py` | `playground_amp/envs/` | Uses CAL for all control and state access |
| `robot_config.py` | `playground_amp/configs/` | Support new actuated_joints schema |

### 6.3 Implementation Order

```
Phase 1: Create CAL Module ✅ COMPLETE
├── 1. playground_amp/cal/types.py          # Enums (CoordinateFrame, ActuatorType, VelocityProfile)
├── 2. playground_amp/cal/specs.py          # Dataclasses (JointSpec, ActuatorSpec, Pose3D, Velocity3D, etc.)
├── 3. playground_amp/cal/cal.py            # Main class (ControlAbstractionLayer)
└── 4. playground_amp/cal/__init__.py       # Exports

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

## 10. Future Work: Extended State Abstraction

### 10.1 Motivation

The current CAL (v0.11.0) focuses on **actuator control** (action→ctrl) and **joint state** (qpos/qvel).
However, the environment also accesses other robot state that should be abstracted:

| Current Location | State Type | Used For |
|------------------|------------|----------|
| `wildrobot_env.py` | Foot contacts | AMP discriminator, gait rewards |
| `wildrobot_env.py` | Foot positions | Slip/clearance rewards |
| `wildrobot_env.py` | Foot velocities | Slip penalty |
| `wildrobot_env.py` | Root quat/pos | Observations, AMP features |
| `wildrobot_env.py` | Heading-local vel | Observations, rewards |
| `wildrobot_env.py` | Gravity vector | Observations |

**Problems with current approach:**
1. **Scattered normalization**: Contact forces normalized in env, joint positions in CAL
2. **Inconsistent patterns**: Some methods return normalized, some raw, some both
3. **Methods scattered across env**: Body IDs resolved in env, should be in CAL
4. **Hard to extend**: Adding a new body/sensor requires editing wildrobot_env.py

### 10.2 Consolidation with robot_config

**Key insight**: `robot_config.yaml` already contains all the necessary configuration:

| Already in robot_config.yaml | Used For |
|------------------------------|----------|
| `feet.left_foot_body`, `feet.right_foot_body` | Foot body IDs |
| `feet.left_toe`, `feet.left_heel`, etc. | Contact geom IDs |
| `floating_base.name`, `floating_base.root_body` | Root body |
| `sensors.velocimeter`, `sensors.framezaxis` | Sensor names |
| `actuated_joints` | Already consumed by CAL v0.11.0 |

**Strategy: CAL consumes robot_config, no new config files.**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          CONFIGURATION FLOW                                  │
│                                                                              │
│  ┌──────────────────────┐                                                   │
│  │  robot_config.yaml   │  ← Single source of truth (generated by          │
│  │                      │    post_process.py from MuJoCo XML)               │
│  │  - actuated_joints   │                                                   │
│  │  - feet              │                                                   │
│  │  - floating_base     │                                                   │
│  │  - sensors           │                                                   │
│  └──────────┬───────────┘                                                   │
│             │                                                                │
│             ▼                                                                │
│  ┌──────────────────────┐                                                   │
│  │    RobotConfig       │  ← Python dataclass (robot_config.py)             │
│  │                      │    Parses YAML, provides typed access             │
│  └──────────┬───────────┘                                                   │
│             │                                                                │
│             ▼                                                                │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │              CONTROL ABSTRACTION LAYER (Extended)                     │   │
│  │                                                                       │   │
│  │  __init__(mj_model, robot_config)  ← Takes RobotConfig, not raw YAML │   │
│  │                                                                       │   │
│  │  Builds internal specs from robot_config:                            │   │
│  │  - ActuatorSpec[] from robot_config.actuated_joints                  │   │
│  │  - FootSpec[] from robot_config.feet.*                               │   │
│  │  - RootSpec from robot_config.floating_base + sensors                │   │
│  │                                                                       │   │
│  │  Resolves MuJoCo IDs at init time:                                   │   │
│  │  - body_id = mj_name2id(mj_model, body_name)                         │   │
│  │  - geom_id = mj_name2id(mj_model, geom_name)                         │   │
│  │  - sensor_id = mj_name2id(mj_model, sensor_name)                     │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 10.3 Internal Specs (Built from robot_config)

These are **internal to CAL**, not exposed in config:

```python
from enum import Enum


class CoordinateFrame(Enum):
    """Coordinate frame for position/velocity queries.

    WORLD: Global world frame (inertial, fixed)
    LOCAL: Body-attached frame (rotates with root body)
    HEADING_LOCAL: Root-relative frame rotated by -yaw (heading-invariant)
                   Most common for RL observations and AMP features
    """
    WORLD = "world"
    LOCAL = "local"
    HEADING_LOCAL = "heading_local"


# =========================================================================
# Generic 3D Types (Reusable for root, feet, any body)
# =========================================================================

@dataclass(frozen=True)
class Pose3D:
    """General 3D pose with coordinate frame (JAX-native).

    Self-documenting: frame is stored with the data.
    Reusable for root, feet, or any body.

    Array Type Design (Factory Pattern):
        - Default: JAX arrays for JIT-compiled training
        - Use from_numpy() for visualization with MuJoCo's NumPy arrays

    Example (JAX - training):
        pose = cal.get_root_pose(data, frame=CoordinateFrame.WORLD)
        height = pose.height
        roll, pitch, yaw = pose.euler_angles()

    Example (NumPy - visualization):
        pose = Pose3D.from_numpy(
            position=mj_data.qpos[:3],
            orientation=mj_data.qpos[3:7],
            frame=CoordinateFrame.WORLD,
        )
        sin_yaw, cos_yaw = pose.heading_sincos
    """
    position: jax.Array        # (3,) [x, y, z] in meters
    orientation: jax.Array     # (4,) quaternion [qw, qx, qy, qz]
    frame: CoordinateFrame     # Frame this pose is expressed in

    @classmethod
    def from_numpy(cls, position, orientation, frame) -> Pose3D:
        """Create Pose3D from NumPy arrays (for visualization)."""
        return cls(
            position=jnp.array(position),
            orientation=jnp.array(orientation),
            frame=frame,
        )

    @property
    def height(self) -> float:
        """Z position (height above ground)."""
        return self.position[2]

    @property
    def xyz(self) -> tuple[float, float, float]:
        """Position as (x, y, z) tuple for unpacking."""
        return self.position[0], self.position[1], self.position[2]

    def euler_angles(self) -> tuple[float, float, float]:
        """Extract Euler angles (ZYX convention) from orientation quaternion.

        Returns:
            (roll, pitch, yaw) in radians
            - roll: rotation around X-axis (body lean left/right)
            - pitch: rotation around Y-axis (body lean forward/backward)
            - yaw: rotation around Z-axis (heading direction)

        Raises:
            ValueError: If pose is not in WORLD frame

        Note:
            WORLD frame is required because pitch/roll have physical meaning
            only when measured relative to gravity (world Z-up).
        """
        if self.frame != CoordinateFrame.WORLD:
            raise ValueError(
                f"euler_angles requires WORLD frame, got {self.frame}. "
                f"Pitch/roll are only meaningful relative to gravity (world Z-up)."
            )
        # ... quaternion to euler conversion ...
        return roll, pitch, yaw

    @property
    def heading_sincos(self) -> tuple[float, float]:
        """Heading (yaw) angle as (sin, cos) components.

        Returns sin/cos instead of angle because:
        - Continuous (no discontinuity at ±π)
        - Better for NN observations and frame transformations

        Naming: "heading" for consistency with HEADING_LOCAL frame.
        """
        _, _, yaw = self.euler_angles()
        return jnp.sin(yaw), jnp.cos(yaw)

    def to_heading_local(self, vec_world: jax.Array) -> jax.Array:
        """Transform a world-frame vector to heading-local frame.

        Note: This is a vector transform only. For positions, subtract the root
        position before applying heading-local rotation.
        """
        if self.frame != CoordinateFrame.WORLD:
            raise ValueError(...)
        sin_yaw, cos_yaw = self.heading_sincos
        # ... rotation math ...
        return rotated_vec

    def linvel_fd(self, prev: Pose3D, dt: float, frame: CoordinateFrame) -> jax.Array:
        """Compute finite-difference linear velocity."""
        if self.frame != CoordinateFrame.WORLD:
            raise ValueError(...)
        linvel_world = (self.position - prev.position) / dt
        if frame == CoordinateFrame.HEADING_LOCAL:
            return self.to_heading_local(linvel_world)
        return linvel_world

    def angvel_fd(self, prev: Pose3D, dt: float, frame: CoordinateFrame) -> jax.Array:
        """Compute finite-difference angular velocity from quaternion change."""
        if self.frame != CoordinateFrame.WORLD:
            raise ValueError(...)
        # ... quaternion derivative math ...
        return angvel

    def velocity_fd(
        self,
        prev: Pose3D,
        dt: float,
        frame: CoordinateFrame = CoordinateFrame.HEADING_LOCAL,
    ) -> Velocity3D:
        """Compute finite-difference velocity between two poses.

        This is the PRIMARY method for computing velocity from pose history,
        matching reference data computation for AMP parity.

        Args:
            prev: Previous pose (at time t - dt)
            dt: Time step in seconds
            frame: Coordinate frame for output velocity (default HEADING_LOCAL for AMP)

        Returns:
            Velocity3D with linear and angular velocity in specified frame

        Example:
            curr_pose = cal.get_root_pose(data)
            prev_pose = Pose3D(prev_pos, prev_quat, CoordinateFrame.WORLD)
            vel = curr_pose.velocity_fd(prev_pose, dt=0.02)
        """
        return Velocity3D(
            linear=self.linvel_fd(prev, dt, frame),
            angular=self.angvel_fd(prev, dt, frame),
            frame=frame,
        )


@dataclass(frozen=True)
class Velocity3D:
    """General 3D velocity with coordinate frame (JAX-native).

    Self-documenting: frame is stored with the data.
    Reusable for root, feet, or any body.

    Array Type Design (Factory Pattern):
        - Default: JAX arrays for JIT-compiled training
        - Use from_numpy() for visualization with MuJoCo's NumPy arrays

    Example (JAX - training):
        vel = cal.get_root_velocity(data, frame=CoordinateFrame.WORLD)
        vel_local = vel.to_heading_local(pose)
        assert vel_local.frame == CoordinateFrame.HEADING_LOCAL

    Example (NumPy - visualization):
        vel = Velocity3D.from_numpy(
            linear=mj_data.qvel[:3],
            angular=mj_data.qvel[3:6],
            frame=CoordinateFrame.WORLD,
        )
        vel_local = vel.to_heading_local(pose)
    """
    linear: jax.Array          # (3,) [vx, vy, vz] in m/s
    angular: jax.Array         # (3,) [wx, wy, wz] in rad/s
    frame: CoordinateFrame     # Frame this velocity is expressed in

    @classmethod
    def from_numpy(cls, linear, angular, frame) -> Velocity3D:
        """Create Velocity3D from NumPy arrays (for visualization)."""
        return cls(
            linear=jnp.array(linear),
            angular=jnp.array(angular),
            frame=frame,
        )

    @property
    def linear_xyz(self) -> tuple[float, float, float]:
        """Linear velocity as (vx, vy, vz) tuple for unpacking."""
        return self.linear[0], self.linear[1], self.linear[2]

    @property
    def angular_xyz(self) -> tuple[float, float, float]:
        """Angular velocity as (wx, wy, wz) tuple for unpacking."""
        return self.angular[0], self.angular[1], self.angular[2]

    def to_heading_local(self, pose: Pose3D) -> Velocity3D:
        """Transform velocity from world frame to heading-local frame.

        Note: This is a vector transform only; position origin does not apply.

        Args:
            pose: Pose3D providing the heading direction (must have orientation)

        Returns:
            New Velocity3D in heading-local frame

        Raises:
            ValueError: If velocity is not in WORLD frame
        """
        if self.frame != CoordinateFrame.WORLD:
            raise ValueError(...)
        return Velocity3D(
            linear=pose.to_heading_local(self.linear),
            angular=pose.to_heading_local(self.angular),
            frame=CoordinateFrame.HEADING_LOCAL,
        )


# =========================================================================
# Internal Specs (Built from robot_config at CAL init)
# =========================================================================

@dataclass(frozen=True)
class FootSpec:
    """Internal spec built from robot_config.feet at CAL init."""
    name: str                    # e.g., "left_foot"
    body_name: str               # From robot_config.left_foot_body (e.g., "left_foot")
    body_id: int                 # Resolved via mj_name2id(body_name)
    toe_geom_id: int             # Resolved from robot_config.feet.left_toe
    heel_geom_id: int            # Resolved from robot_config.feet.left_heel
    symmetry_pair: Optional[str] = None


@dataclass(frozen=True)
class RootSpec:
    """Internal spec built from robot_config.floating_base at CAL init.

    Uses chest_imu (BNO085) sensors which exist on the real robot:
    - chest_imu_quat: framequat for orientation
    - chest_imu_gyro: gyroscope for angular velocity
    - chest_imu_accel: accelerometer (can derive gravity direction)

    BNO085 is a 9-DOF IMU (gyro + accel + magnetometer) with onboard sensor fusion:
    - Outputs stable framequat (orientation quaternion)
    - Unlike 6-DOF IMUs (ICM45686), has magnetometer for absolute heading
    - Onboard processor handles sensor fusion - no drift in orientation

    IMU Mounting Position Note:
        The BNO085 is mounted on the chest, not at the center of the root body.
        For WildRobot: ~16.7cm above chest body origin (see wildrobot.xml site pos)

        Effect on readings:
        - Orientation (framequat): NOT affected
          (orientation is same for all points on rigid body)
        - Angular velocity (gyro): NOT affected
          (angular velocity is body-invariant, same for all points)
        - Linear acceleration (accel): SLIGHTLY affected by offset
          - Centripetal: ω²r ≈ (5 rad/s)² × 0.167m ≈ 4.2 m/s²
          - Tangential: α*r ≈ (50 rad/s²) × 0.167m ≈ 8.4 m/s²
          - Compare to gravity (~10 m/s²) and motion (~20 m/s²)
        - Gravity vector: NOT affected (derived from orientation)

        Why it's OK for training:
        1. Orientation/gyro: Body-invariant, no effect from offset
        2. Gravity vector: Derived from orientation, not accelerometer
        3. Accelerometer: Only used for gravity direction (via orientation),
           not for velocity integration (which would accumulate drift)
        4. Sim2Real: MuJoCo sensor site matches real hardware position
    """
    body_name: str               # From robot_config.floating_base.root_body
    body_id: int                 # Resolved via mj_name2id
    qpos_addr: int               # From mj_model.jnt_qposadr
    qvel_addr: int               # From mj_model.jnt_dofadr
    # Chest IMU sensors (BNO085 - real hardware)
    orientation_sensor: str      # chest_imu_quat (framequat)
    gyro_sensor: str             # chest_imu_gyro (angular velocity)
    accel_sensor: str            # chest_imu_accel (can derive gravity)
    # Pelvis sensors
    local_linvel_sensor: str     # pelvis_local_linvel (velocimeter)


@dataclass(frozen=True)
class Imu6DoFSpec:
    """Internal spec for 6-DOF IMU sensors (gyro + accelerometer).

    Used for: knee IMUs, arm IMUs (future), or any other 6-DOF IMU.

    6-DOF IMU provides:
    - 3-axis gyroscope (angular velocity)
    - 3-axis accelerometer (linear acceleration)

    Does NOT provide (unlike 9-DOF IMU like BNO085):
    - Magnetometer (no absolute heading reference)
    - Onboard sensor fusion (no orientation quaternion)

    Example sensors:
    - ICM45686 (knee IMUs): Gyro noise ~0.003°/s/√Hz, Accel noise ~80μg/√Hz

    IMU Mounting Position Note:
        The IMU may not be exactly at the joint center. For WildRobot knee IMUs:
        - Offset: ~2.6cm above knee joint center (see wildrobot.xml site pos)

        Effect on readings:
        - Gyroscope: NOT affected (angular velocity is same for all points on rigid body)
        - Accelerometer: SLIGHTLY affected by offset
          - Centripetal: ω²r ≈ (10 rad/s)² × 0.026m ≈ 2.6 m/s²
          - Tangential: α*r ≈ (100 rad/s²) × 0.026m ≈ 2.6 m/s²
          - Compare to gravity (~10 m/s²) and impacts (~30-50 m/s²)

        Why it's OK for training:
        1. Consistency: Fixed offset = consistent signal = policy can learn
        2. Sim2Real: MuJoCo sensor site matches real hardware position
        3. Signal quality: Offset contribution (~5 m/s²) is small vs signal (~50 m/s²)
        4. Still informative: Captures impacts, swing phase, gravity direction

    Potential uses:
    - Ground impact detection (accelerometer spike = foot strike)
    - Slip detection (unexpected lateral acceleration)
    - Limb swing dynamics (gyro provides angular velocity)
    - Terrain classification (vibration patterns)
    """
    name: str               # e.g., "left_knee", "left_elbow"
    body_name: str          # Body this IMU is attached to
    gyro_sensor: str        # e.g., "left_knee_imu_gyro"
    accel_sensor: str       # e.g., "left_knee_imu_accel"
    symmetry_pair: str      # e.g., "right_knee" or "right_elbow"
```

### 10.4 Extended CAL Interface

```python
class ControlAbstractionLayer:
    """Extended CAL with body state access.

    Consumes RobotConfig - no new config files needed.
    """

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        robot_config: RobotConfig,  # Single config source
    ):
        # Existing: build actuator specs from robot_config.actuated_joints
        self._build_actuator_specs(robot_config.actuated_joints)

        # NEW: build foot specs from robot_config.feet.*
        self._build_foot_specs(robot_config)

        # NEW: build root spec from robot_config.floating_base + sensors
        self._build_root_spec(robot_config)

    def _build_foot_specs(self, robot_config: RobotConfig) -> None:
        """Build FootSpec from existing robot_config.feet section."""
        self._foot_specs = [
            FootSpec(
                name="left_foot",
                body_id=mujoco.mj_name2id(
                    self._mj_model, mujoco.mjtObj.mjOBJ_BODY,
                    robot_config.left_foot_body
                ),
                toe_geom_id=mujoco.mj_name2id(
                    self._mj_model, mujoco.mjtObj.mjOBJ_GEOM,
                    robot_config.get_foot_geom_names()[0]  # left_toe
                ),
                heel_geom_id=mujoco.mj_name2id(
                    self._mj_model, mujoco.mjtObj.mjOBJ_GEOM,
                    robot_config.get_foot_geom_names()[1]  # left_heel
                ),
                symmetry_pair="right_foot",
            ),
            FootSpec(
                name="right_foot",
                body_id=mujoco.mj_name2id(
                    self._mj_model, mujoco.mjtObj.mjOBJ_BODY,
                    robot_config.right_foot_body
                ),
                toe_geom_id=mujoco.mj_name2id(
                    self._mj_model, mujoco.mjtObj.mjOBJ_GEOM,
                    robot_config.get_foot_geom_names()[2]  # right_toe
                ),
                heel_geom_id=mujoco.mj_name2id(
                    self._mj_model, mujoco.mjtObj.mjOBJ_GEOM,
                    robot_config.get_foot_geom_names()[3]  # right_heel
                ),
                symmetry_pair="left_foot",
            ),
        ]

    def _build_root_spec(self, robot_config: RobotConfig) -> None:
        """Build RootSpec from existing robot_config sections.

        Sensor names come from robot_config.yaml, not hardcoded.
        """
        root_body = robot_config.floating_base_body
        self._root_spec = RootSpec(
            body_name=root_body,
            body_id=mujoco.mj_name2id(
                self._mj_model, mujoco.mjtObj.mjOBJ_BODY, root_body
            ),
            qpos_addr=self._get_floating_base_qpos_addr(),
            qvel_addr=self._get_floating_base_qvel_addr(),
            # Chest IMU sensors from robot_config.yaml
            orientation_sensor=robot_config.orientation_sensor,  # chest_imu_quat
            gyro_sensor=robot_config.gyro_sensor,                # chest_imu_gyro
            accel_sensor=robot_config.accel_sensor,              # chest_imu_accel
            # Pelvis sensors from robot_config.yaml
            local_linvel_sensor=robot_config.local_linvel_sensor,
        )

    # =========================================================================
    # Foot State Access (NEW)
    # =========================================================================
    #
    # DESIGN PATTERN: Single method with REQUIRED `normalize` parameter
    #
    # The `normalize` parameter is REQUIRED (no default) to:
    # - Force explicit intent at call site
    # - Prevent bugs from accidentally using wrong version
    # - Make code self-documenting
    #
    # Convention:
    # - normalize=True: For policy observations, AMP discriminator (NN input)
    # - normalize=False: For rewards, debugging, Sim2Real (physical units)
    # =========================================================================

    def get_foot_contacts(
        self,
        data: mjx.Data,
        normalize: bool,  # REQUIRED - no default
    ) -> jax.Array:
        """Get foot contact values.

        Args:
            data: MJX simulation data
            normalize: REQUIRED. True for tanh-normalized [0, 1],
                      False for raw normal forces (Newtons)

        Returns:
            foot_contacts: (num_feet * 2,) - [toe, heel] for each foot
        """
        contacts = []
        for foot in self._foot_specs:
            toe_force = self._get_geom_contact_force(data, foot.toe_geom_id)
            heel_force = self._get_geom_contact_force(data, foot.heel_geom_id)

            if normalize:
                toe_force = jnp.tanh(toe_force / self._contact_scale)
                heel_force = jnp.tanh(heel_force / self._contact_scale)

            contacts.extend([toe_force, heel_force])

        return jnp.array(contacts)

    def get_foot_positions(
        self,
        data: mjx.Data,
        normalize: bool,  # REQUIRED - no default
        frame: CoordinateFrame = CoordinateFrame.WORLD,
    ) -> jax.Array:
        """Get foot body positions.

        Args:
            data: MJX simulation data
            normalize: REQUIRED. True to normalize by root height,
                      False for raw positions (meters)
            frame: Coordinate frame (WORLD or HEADING_LOCAL). HEADING_LOCAL is root-relative.

        Returns:
            positions: (num_feet, 3) foot positions
        """
        root_pos = data.qpos[self._root_spec.qpos_addr : self._root_spec.qpos_addr + 3]
        positions = []
        for foot in self._foot_specs:
            pos = data.xpos[foot.body_id]
            if frame == CoordinateFrame.HEADING_LOCAL:
                pos = self._world_to_heading_local(data, pos - root_pos)
            positions.append(pos)

        result = jnp.stack(positions)

        if normalize:
            root_height = self.get_root_height(data)
            result = result / (root_height + 1e-6)

        return result

    def get_foot_velocities(
        self,
        data: mjx.Data,
        prev_positions: jax.Array,
        dt: float,
        normalize: bool,  # REQUIRED - no default
        frame: CoordinateFrame = CoordinateFrame.WORLD,
        max_foot_velocity: float = 5.0,
    ) -> jax.Array:
        """Get foot velocities via finite difference.

        Args:
            data: Current MJX data
            prev_positions: Previous foot positions from get_foot_positions()
            dt: Time step for finite difference
            normalize: REQUIRED. True to normalize by max_foot_velocity to [-1, 1],
                      False for raw velocities (m/s)
            frame: Coordinate frame
            max_foot_velocity: Max expected foot velocity for normalization

        Returns:
            velocities: (num_feet, 3) foot velocities
        """
        curr_positions = self.get_foot_positions(data, normalize=False, frame=frame)
        velocities = (curr_positions - prev_positions) / dt

        if normalize:
            velocities = jnp.clip(velocities / max_foot_velocity, -1.0, 1.0)

        return velocities

    def get_foot_clearances(
        self,
        data: mjx.Data,
        normalize: bool,  # REQUIRED - no default
    ) -> jax.Array:
        """Get foot height above ground (for swing phase rewards).

        Args:
            data: MJX simulation data
            normalize: REQUIRED. True to normalize by root height,
                      False for raw height (meters)

        Returns:
            clearances: (num_feet,) height values
        """
        positions = self.get_foot_positions(data, normalize=False, frame=CoordinateFrame.WORLD)
        clearances = positions[:, 2]  # Z coordinate

        if normalize:
            root_height = self.get_root_height(data)
            clearances = clearances / (root_height + 1e-6)

        return clearances

    # =========================================================================
    # Knee IMU State Access (NEW - Optional for Enhanced Proprioception)
    # =========================================================================
    #
    # 6-DOF IMUs (ICM45686) provide additional proprioceptive feedback:
    # - Ground impact detection (accelerometer spike = foot strike)
    # - Slip detection (unexpected lateral acceleration)
    # - Limb swing dynamics (gyro provides angular velocity)
    # - Terrain classification (vibration patterns)
    #
    # These sensors exist on real hardware but are NOT currently used in training.
    # An ablation study is recommended to measure their value.
    # =========================================================================

    def _build_6dof_imu_specs(self, robot_config: RobotConfig) -> None:
        """Build Imu6DoFSpec from robot_config.knee_imus section.

        Future: Can be extended to include arm IMUs when added.
        """
        knee_imus = robot_config.get("knee_imus", {})
        self._6dof_imu_specs = [
            Imu6DoFSpec(
                name="left_knee",
                body_name="left_knee",  # Body this IMU is attached to
                gyro_sensor=knee_imus.get("left_gyro", "left_knee_imu_gyro"),
                accel_sensor=knee_imus.get("left_accel", "left_knee_imu_accel"),
                symmetry_pair="right_knee",
            ),
            Imu6DoFSpec(
                name="right_knee",
                body_name="right_knee",  # Body this IMU is attached to
                gyro_sensor=knee_imus.get("right_gyro", "right_knee_imu_gyro"),
                accel_sensor=knee_imus.get("right_accel", "right_knee_imu_accel"),
                symmetry_pair="left_knee",
            ),
        ]

        # Future: Add arm IMUs when available
        # arm_imus = robot_config.get("arm_imus", {})
        # if arm_imus:
        #     self._6dof_imu_specs.extend([
        #         Imu6DoFSpec(name="left_elbow", body_name="left_elbow", ...),
        #         Imu6DoFSpec(name="right_elbow", body_name="right_elbow", ...),
        #     ])

    def get_knee_angular_velocities(
        self,
        data: mjx.Data,
        normalize: bool,  # REQUIRED - no default for consistency
        max_angular_velocity: float = 10.0,  # rad/s
    ) -> jax.Array:
        """Get knee angular velocities from IMU gyros.

        Args:
            data: MJX simulation data
            normalize: REQUIRED. True to normalize by max_angular_velocity to [-1, 1],
                      False for raw angular velocities (rad/s)
            max_angular_velocity: Max expected angular velocity for normalization

        Returns:
            knee_angvel: (2, 3) angular velocities [wx, wy, wz] for [left, right]

        Note - Why jax.Array instead of Velocity3D:
            Velocity3D has .linear (m/s) and .angular (rad/s) fields.
            The ICM45686 gyro outputs angular velocity directly, but the
            accelerometer outputs linear ACCELERATION (m/s²), not velocity.
            These are different physical quantities:

            | Sensor | Measures              | Units  | Velocity3D field |
            |--------|-----------------------|--------|------------------|
            | Gyro   | Angular velocity      | rad/s  | .angular ✓       |
            | Accel  | Linear ACCELERATION   | m/s²   | .linear ✗        |

            To get linear velocity from acceleration would require integration,
            which accumulates drift rapidly (unusable within seconds).

        Note - Why no CoordinateFrame param:
            IMUs are physically attached to knee bodies - they inherently measure
            in LOCAL (sensor) frame. Transforming to world frame would require:
            1. Knowing knee body's world orientation (extra computation)
            2. Not useful for proprioception (local frame is what matters)

        Note - Why no pose from ICM45686:
            ICM45686 is a 6-DOF IMU (gyro + accel only, no magnetometer).
            - Orientation: Could integrate gyro, but yaw drifts without magnetometer
            - Position: Would need to double-integrate accel → huge drift (meters/sec)
            Compare to BNO085 (chest IMU) which has magnetometer + onboard fusion
            processor that outputs stable framequat.

        Potential use cases:
        - Leg swing phase detection
        - Slip detection (unexpected rotation)
        - More accurate joint velocity (vs finite-diff from encoders)
        """
        angvels = []
        for spec in self._6dof_imu_specs:
            angvel = mjx_env.get_sensor_data(
                self._mj_model, data, spec.gyro_sensor
            )
            angvels.append(angvel)

        result = jnp.stack(angvels)

        if normalize:
            result = jnp.clip(result / max_angular_velocity, -1.0, 1.0)

        return result

    def get_knee_accelerations(
        self,
        data: mjx.Data,
        normalize: bool,  # REQUIRED - no default for consistency
        max_acceleration: float = 50.0,  # m/s² (includes gravity ~10 + motion ~40)
    ) -> jax.Array:
        """Get knee accelerations from IMU accelerometers.

        Args:
            data: MJX simulation data
            normalize: REQUIRED. True to normalize by max_acceleration to [-1, 1],
                      False for raw accelerations (m/s²)
            max_acceleration: Max expected acceleration for normalization

        Returns:
            knee_accel: (2, 3) accelerations [ax, ay, az] for [left, right]

        Note - Why jax.Array instead of a dataclass:
            Accelerometers measure linear ACCELERATION (m/s²), not velocity.
            This is a fundamentally different physical quantity from Velocity3D:

            | Field              | Physical Quantity | Units  | ICM45686 Output |
            |--------------------|-------------------|--------|-----------------|
            | Velocity3D.linear  | Linear velocity   | m/s    | ✗ Not measured  |
            | Velocity3D.angular | Angular velocity  | rad/s  | From gyro ✓     |
            | (this method)      | Linear accel      | m/s²   | From accel ✓    |

            To get linear velocity from acceleration would require integration,
            which accumulates drift rapidly (meters of error per second).

        Note - Why no CoordinateFrame param:
            IMUs are physically attached to knee bodies - they inherently measure
            in LOCAL (sensor) frame. Transforming to world frame would require:
            1. Knowing knee body's world orientation (extra computation)
            2. Not useful for proprioception (local frame is what matters)

        Note - Why no pose from ICM45686:
            ICM45686 is a 6-DOF IMU (gyro + accel only, no magnetometer).
            - Orientation: Could integrate gyro, but yaw drifts without magnetometer
            - Position: Would need to double-integrate accel → huge drift (meters/sec)
            Compare to BNO085 (chest IMU) which is 9-DOF with magnetometer +
            onboard sensor fusion processor that outputs stable framequat.

        Potential use cases:
        - Ground impact detection (spike in Z when foot strikes)
        - Slip detection (unexpected lateral acceleration)
        - Terrain classification (vibration patterns differ by surface)
        """
        accels = []
        for spec in self._6dof_imu_specs:
            accel = mjx_env.get_sensor_data(
                self._mj_model, data, spec.accel_sensor
            )
            accels.append(accel)

        result = jnp.stack(accels)

        if normalize:
            result = jnp.clip(result / max_acceleration, -1.0, 1.0)

        return result

    # =========================================================================
    # Root State Access (4 Core APIs using Pose3D/Velocity3D)
    # =========================================================================
    #
    # Consolidated from 10 methods → 4 methods using generic 3D types.
    # Frame is stored with the data for self-documentation.
    #
    # Derivable values (use utility functions on pose.orientation):
    # - height: pose.height (property)
    # - pitch/roll: pitch_roll_from_quat_jax(pose.orientation)
    # - heading: heading_sin_cos_from_quat_jax(pose.orientation)
    # =========================================================================

    def get_root_pose(
        self,
        data: mjx.Data,
        frame: CoordinateFrame = CoordinateFrame.WORLD,
    ) -> Pose3D:
        """Get root body pose with named field access.

        Args:
            data: MJX simulation data
            frame: Coordinate frame for position. HEADING_LOCAL is root-relative.

        Returns:
            Pose3D with:
                .position: (3,) [x, y, z] in meters
                .orientation: (4,) quaternion [qw, qx, qy, qz]
                .frame: CoordinateFrame
                .height: scalar z (convenience property)
                .xy: (2,) [x, y] (convenience property)

        Example:
            pose = cal.get_root_pose(data)
            height = pose.height
            quat = pose.orientation
            pitch, roll = pitch_roll_from_quat_jax(pose.orientation)
        """
        position = data.qpos[self._root_spec.qpos_addr:self._root_spec.qpos_addr + 3]
        orientation = data.qpos[self._root_spec.qpos_addr + 3:self._root_spec.qpos_addr + 7]

        if frame == CoordinateFrame.HEADING_LOCAL:
            root_pos = position
            position = self._world_to_heading_local_vec(orientation, position - root_pos)

        return Pose3D(
            position=position,
            orientation=orientation,
            frame=frame,
        )

    def get_root_velocity(
        self,
        data: mjx.Data,
        frame: CoordinateFrame = CoordinateFrame.WORLD,
    ) -> Velocity3D:
        """Get root body velocity with named field access.

        Args:
            data: MJX simulation data
            frame: Coordinate frame (WORLD, LOCAL, or HEADING_LOCAL)

        Returns:
            Velocity3D with:
                .linear: (3,) [vx, vy, vz] in m/s
                .angular: (3,) [wx, wy, wz] in rad/s
                .frame: CoordinateFrame

        Example:
            vel = cal.get_root_velocity(data, frame=CoordinateFrame.HEADING_LOCAL)
            forward_speed = vel.linear[0]
            yaw_rate = vel.angular[2]
        """
        quat = data.qpos[self._root_spec.qpos_addr + 3:self._root_spec.qpos_addr + 7]

        # Linear velocity
        if self._root_spec.local_linvel_sensor and frame == CoordinateFrame.LOCAL:
            linvel = mjx_env.get_sensor_data(
                self._mj_model, data, self._root_spec.local_linvel_sensor
            )
        else:
            linvel = data.qvel[self._root_spec.qvel_addr:self._root_spec.qvel_addr + 3]
            if frame == CoordinateFrame.HEADING_LOCAL:
                linvel = self._world_to_heading_local_vec(quat, linvel)
            elif frame == CoordinateFrame.LOCAL:
                linvel = self._world_to_local(data, linvel)

        # Angular velocity
        if self._root_spec.global_angvel_sensor and frame == CoordinateFrame.WORLD:
            angvel = mjx_env.get_sensor_data(
                self._mj_model, data, self._root_spec.global_angvel_sensor
            )
        else:
            angvel = data.qvel[self._root_spec.qvel_addr + 3:self._root_spec.qvel_addr + 6]
            if frame == CoordinateFrame.HEADING_LOCAL:
                angvel = self._world_to_heading_local_vec(quat, angvel)
            elif frame == CoordinateFrame.LOCAL:
                angvel = self._world_to_local(data, angvel)

        return Velocity3D(
            linear=linvel,
            angular=angvel,
            frame=frame,
        )

    def get_root_velocity_fd(
        self,
        curr_pose: Pose3D,
        prev_pose: Pose3D,
        dt: float,
        frame: CoordinateFrame = CoordinateFrame.HEADING_LOCAL,
    ) -> Velocity3D:
        """Compute root velocity via finite difference.

        USE FOR: AMP parity (matches reference data computation).

        Args:
            curr_pose: Current root pose from get_root_pose()
            prev_pose: Previous root pose
            dt: Time step
            frame: Coordinate frame (default HEADING_LOCAL for AMP)

        Returns:
            Velocity3D with finite-difference velocities in m/s and rad/s

        Note:
            Returns physical units to match reference data format.
            No normalization - AMP discriminator expects physical scale.

        Example:
            curr_pose = cal.get_root_pose(data)
            prev_pose = cal.get_root_pose(prev_data)
            vel_fd = cal.get_root_velocity_fd(curr_pose, prev_pose, dt)
            linvel = vel_fd.linear
            angvel = vel_fd.angular
        """
        # Linear velocity via finite difference
        if frame == CoordinateFrame.HEADING_LOCAL:
            linvel = heading_local_linvel_fd_from_pos_jax(
                curr_pose.position, prev_pose.position, curr_pose.orientation, dt
            )
        else:
            linvel = (curr_pose.position - prev_pose.position) / dt

        # Angular velocity via finite difference
        angvel = heading_local_angvel_fd_from_quat_jax(
            curr_pose.orientation, prev_pose.orientation, dt
        )

        return Velocity3D(
            linear=linvel,
            angular=angvel,
            frame=frame,
        )

    def get_gravity_vector(self, data: mjx.Data) -> jax.Array:
        """Get gravity vector in local (body) frame.

        Derived from chest_imu_quat (BNO085 orientation sensor).
        This matches the real hardware which has no dedicated gravity sensor.

        Returns:
            gravity: (3,) gravity direction in local frame, normalized
        """
        # Get orientation from chest_imu_quat sensor (BNO085)
        if self._root_spec.orientation_sensor:
            quat = mjx_env.get_sensor_data(
                self._mj_model, data, self._root_spec.orientation_sensor
            )
        else:
            # Fallback to qpos quaternion
            quat = data.qpos[self._root_spec.qpos_addr + 3:self._root_spec.qpos_addr + 7]

        # Rotate world gravity [0, 0, -1] into local frame
        # (normalized direction, not magnitude)
        world_gravity = jnp.array([0.0, 0.0, -1.0])
        return self._rotate_vector_by_quat_inverse(world_gravity, quat)

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _world_to_heading_local_vec(
        self,
        quat: jax.Array,
        vec: jax.Array,
    ) -> jax.Array:
        """Rotate world-frame vector to heading-local frame.

        Note: This is a vector transform only. For positions, subtract the root
        position before applying heading-local rotation.
        """
        sin_yaw, cos_yaw = heading_sin_cos_from_quat_jax(quat)
        return heading_local_from_world_jax(vec, sin_yaw, cos_yaw)

    def _rotate_vector_by_quat_inverse(
        self,
        vec: jax.Array,
        quat: jax.Array,
    ) -> jax.Array:
        """Rotate a vector by the inverse of a quaternion.

        Used to transform world-frame gravity [0, 0, -1] into local body frame.

        Args:
            vec: 3D vector to rotate
            quat: Quaternion in wxyz format [w, x, y, z]

        Returns:
            Rotated vector in body frame
        """
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]

        # Rotation matrix from quaternion (transposed for inverse)
        r00 = 1 - 2 * (y * y + z * z)
        r01 = 2 * (x * y + w * z)
        r02 = 2 * (x * z - w * y)
        r10 = 2 * (x * y - w * z)
        r11 = 1 - 2 * (x * x + z * z)
        r12 = 2 * (y * z + w * x)
        r20 = 2 * (x * z + w * y)
        r21 = 2 * (y * z - w * x)
        r22 = 1 - 2 * (x * x + y * y)

        # Apply transposed rotation (inverse)
        vx, vy, vz = vec[0], vec[1], vec[2]
        return jnp.array([
            r00 * vx + r10 * vy + r20 * vz,
            r01 * vx + r11 * vy + r21 * vz,
            r02 * vx + r12 * vy + r22 * vz,
        ])

    def _get_geom_contact_force(
        self,
        data: mjx.Data,
        geom_id: int,
    ) -> jax.Array:
        """Get normal contact force for a geom (solver-native efc_force)."""
        geom1_match = data.contact.geom1 == geom_id
        geom2_match = data.contact.geom2 == geom_id
        is_our_contact = geom1_match | geom2_match

        normal_forces = data.efc_force[data.contact.efc_address]
        our_forces = jnp.where(is_our_contact, jnp.abs(normal_forces), 0.0)
        return jnp.sum(our_forces)
```

### 10.5 What Stays in Training Config (ppo_walking.yaml)

Normalization and threshold parameters that are **reward-specific** stay in training config, not robot_config:

```yaml
# ppo_walking.yaml - env section
env:
  contact_threshold_force: 5.0   # Force threshold for "loaded" detection (N)
  contact_scale: 10.0            # tanh normalization divisor
  min_height: 0.2                # Termination height (m)
  max_height: 0.8
  max_pitch: 0.6                 # Termination orientation (rad)
  max_roll: 0.6
```

**Rationale**: These are tunable hyperparameters, not robot properties:
- Different training runs may use different thresholds
- Normalization scale affects reward shaping
- Termination thresholds are training-specific

**robot_config.yaml provides**: Body names, geom names, sensor names (static robot properties)
**ppo_walking.yaml provides**: Thresholds, scales, limits (tunable training parameters)

### 10.6 No New Config Files Needed

| Data | Already In | CAL Reads From |
|------|------------|----------------|
| Foot body names | `robot_config.yaml` → `feet.left_foot_body` | `robot_config.left_foot_body` |
| Foot geom names | `robot_config.yaml` → `feet.left_toe` etc. | `robot_config.get_foot_geom_names()` |
| Root body | `robot_config.yaml` → `floating_base.root_body` | `robot_config.floating_base_body` |
| Root IMU sensors | `robot_config.yaml` → `root_imu.*` | `robot_config.orientation_sensor` etc. |
| Linvel sensor | `robot_config.yaml` → `root_imu.local_linvel_sensor` | `robot_config.local_linvel_sensor` |
| Contact scale | `ppo_walking.yaml` → `env.contact_scale` | Passed to CAL or used in env |
| Height limits | `ppo_walking.yaml` → `env.min_height` | Used in env termination |

### 10.6.1 Required Addition to robot_config.yaml

Add a `root_imu` section to specify which sensors to use for root state:

```yaml
# robot_config.yaml - NEW section to add
root_imu:
  # Chest IMU (BNO085) - real hardware sensors
  orientation_sensor: chest_imu_quat    # framequat sensor for orientation
  gyro_sensor: chest_imu_gyro           # gyro sensor for angular velocity
  accel_sensor: chest_imu_accel         # accelerometer
  # Pelvis sensors
  local_linvel_sensor: pelvis_local_linvel  # velocimeter for local linear velocity
```

### 10.6.2 Required Properties in RobotConfig Class

Add properties to `robot_config.py` to expose the root IMU sensors:

```python
# In assets/robot_config.py

class RobotConfig:
    # ... existing code ...

    @property
    def orientation_sensor(self) -> str:
        """Get root orientation sensor name (framequat)."""
        root_imu = self._config.get("root_imu", {})
        return root_imu.get("orientation_sensor", "chest_imu_quat")

    @property
    def gyro_sensor(self) -> str:
        """Get root angular velocity sensor name (gyro)."""
        root_imu = self._config.get("root_imu", {})
        return root_imu.get("gyro_sensor", "chest_imu_gyro")

    @property
    def accel_sensor(self) -> str:
        """Get root accelerometer sensor name."""
        root_imu = self._config.get("root_imu", {})
        return root_imu.get("accel_sensor", "chest_imu_accel")

    @property
    def local_linvel_sensor(self) -> str:
        """Get local linear velocity sensor name (velocimeter)."""
        root_imu = self._config.get("root_imu", {})
        return root_imu.get("local_linvel_sensor", "pelvis_local_linvel")
```

### 10.6.3 Update post_process.py

Add generation of `root_imu` section to `post_process.py`:

```python
def generate_root_imu_config(sensors: dict) -> dict:
    """Generate root_imu config from sensors section.

    Auto-detects chest_imu sensors for real hardware compatibility.
    """
    root_imu = {}

    # Find chest_imu orientation sensor (framequat)
    framequat = sensors.get("framequat", [])
    for s in framequat:
        if "chest_imu" in s.get("name", "").lower():
            root_imu["orientation_sensor"] = s["name"]
            break

    # Find chest_imu gyro sensor
    gyro = sensors.get("gyro", [])
    for s in gyro:
        if "chest_imu" in s.get("name", "").lower():
            root_imu["gyro_sensor"] = s["name"]
            break

    # Find chest_imu accelerometer
    accel = sensors.get("accelerometer", [])
    for s in accel:
        if "chest_imu" in s.get("name", "").lower():
            root_imu["accel_sensor"] = s["name"]
            break

    # Find pelvis velocimeter
    velocimeter = sensors.get("velocimeter", [])
    for s in velocimeter:
        if "pelvis" in s.get("name", "").lower() and "linvel" in s.get("name", "").lower():
            root_imu["local_linvel_sensor"] = s["name"]
            break

    return root_imu
```

### 10.6.4 Knee IMU Configuration (Optional Enhancement)

Add `knee_imus` section to robot_config.yaml for additional proprioception:

```yaml
# robot_config.yaml - Optional section for knee IMUs
knee_imus:
  # Left knee IMU (ICM45686)
  left_gyro: left_knee_imu_gyro
  left_accel: left_knee_imu_accel
  # Right knee IMU (ICM45686)
  right_gyro: right_knee_imu_gyro
  right_accel: right_knee_imu_accel
```

### 10.6.5 Ablation Study: Knee IMU Value

**Hypothesis**: Knee IMUs improve policy performance for:
1. Contact timing precision (accelerometer spike = foot strike)
2. Slip recovery (faster detection via unexpected acceleration)
3. Terrain adaptation (vibration patterns differ by surface)

**Proposed Experiment**:

| Experiment | Observation Dim | Description |
|------------|-----------------|-------------|
| Baseline | 35 | Current observations (no knee IMU) |
| +Knee Gyro | 41 | Add 6D knee angular velocities |
| +Knee Accel | 41 | Add 6D knee accelerations |
| +Both | 47 | Add all 12D knee IMU data |

**Metrics to Compare**:
1. **Sample efficiency**: Timesteps to reach reward threshold
2. **Final performance**: Asymptotic reward
3. **Perturbation recovery**: Push recovery success rate
4. **Slip detection**: Time to detect and recover from foot slip
5. **Terrain transfer**: Performance on unseen terrains (grass, gravel, etc.)

**Implementation Steps**:
1. Add knee IMU observations to env (optional flag in config)
2. Train 4 variants (baseline, +gyro, +accel, +both)
3. Compare metrics at 10M, 50M, 100M timesteps
4. Test perturbation recovery and slip scenarios
5. Document findings

**Observation Integration** (in ppo_walking.yaml):

```yaml
# ppo_walking.yaml
env:
  use_knee_imu: true  # Enable knee IMU observations
  knee_imu_config:
    use_gyro: true    # Include knee angular velocity
    use_accel: true   # Include knee acceleration
    max_angular_velocity: 10.0  # rad/s for normalization
    max_acceleration: 50.0      # m/s² for normalization

dimensions:
  observation_dim: 47  # 35 base + 12 knee IMU
  observation_breakdown:
    # ... existing ...
    knee_imu: 12  # [left_gyro(3), left_accel(3), right_gyro(3), right_accel(3)]
```

### 10.7 Migration Path

**Phase 1: Add New Specs** (non-breaking)
1. Add `FootSpec` and `RootSpec` to `specs.py`
2. Add config loading in `robot_config.py`
3. Existing env code continues to work

**Phase 2: Add CAL Methods** (non-breaking)
1. Add `get_foot_*` and `get_root_*` methods to CAL
2. Add unit tests for new methods
3. Existing env code continues to work

**Phase 3: Migrate Env** (breaking within env only)
1. Replace env's `get_foot_*` methods with `cal.get_foot_*()`
2. Replace env's root state methods with `cal.get_root_*()`
3. Remove duplicate code from env
4. Verify tests pass

**Phase 4: Cleanup**
1. Remove deprecated env methods
2. Update documentation
3. Add migration guide

### 10.7 Benefits

| Benefit | Description |
|---------|-------------|
| **Single source of truth** | All robot state access through CAL |
| **Consistent normalization** | `normalize=True/False` pattern everywhere |
| **Config-driven** | Adding new feet/bodies = config change only |
| **Testable** | CAL methods can be unit-tested in isolation |
| **Sim2Real ready** | Same interface for real robot sensors |
| **AMP compatible** | Finite-diff velocities match reference data |

### 10.8 Open Questions

1. **Should foot contact thresholds be in CAL or reward config?**
   - Current: `contact_threshold_force` in env config
   - Option A: Move to FootSpec (CAL owns all sensing)
   - Option B: Keep in reward config (threshold is reward-specific)
   - **Recommendation**: Keep threshold in reward config, only normalization scale in CAL

2. **Should heading_local be the default frame?**
   - Current: Methods have `frame` parameter
   - Option A: Default to `world`, explicit `heading_local`
   - Option B: Default to `heading_local` (most common for RL)
   - **Recommendation**: Default to `heading_local` for velocities, `world` for positions

3. **Should we add head/torso state?**
   - Could add `BodySpec` for arbitrary bodies
   - Use case: head position for vision, torso for balance
   - **Recommendation**: Future work - add when needed

---

## 11. References

1. **D76312096**: [robot][mujoco] Fixed actuator of type position - Key insight on position actuator semantics
2. **D77383084**: [robot][MuJoCo] Reworked JointState ingestion - Motor vs position actuator handling
3. **Cartpole Benchmark**: Action space normalization issues in RLlib
4. **MetaMend Post (Workplace)**: HAL abstraction pattern
5. **dm_control**: action_spec with bounds
6. **MuJoCo Documentation**: Position actuator PD controller semantics
