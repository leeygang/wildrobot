"""Control Abstraction Layer (CAL) - Main interface for MuJoCo control.

This module provides the ControlAbstractionLayer class, which is the central
abstraction for all MuJoCo control interactions.

Design principles:
1. Actions are always normalized [-1, 1] at the policy level
2. Joint symmetry is handled transparently
3. Actuator semantics are abstracted away
4. State access is consistent and well-defined

Version: v0.11.0
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from playground_amp.cal.specs import (
    ActuatorSpec,
    ControlCommand,
    FootSpec,
    JointSpec,
    Pose3D,
    RootSpec,
    Velocity3D,
)
from playground_amp.cal.types import ActuatorType, CoordinateFrame


class ControlAbstractionLayer:
    """Central abstraction for all MuJoCo control interactions.

    This class is the ONLY place that should interact with MuJoCo primitives
    for control purposes. All other code should use this interface.

    Design principles:
    1. Actions are always normalized [-1, 1] at the policy level
    2. Joint symmetry is handled transparently
    3. Actuator semantics are abstracted away
    4. State access is consistent and well-defined

    Naming Convention:
    - action: Normalized policy output in [-1, 1]
    - ctrl: MuJoCo control signal in physical units (radians for position actuators)

    Example:
        cal = ControlAbstractionLayer(mj_model, robot_config)

        # Policy outputs normalized action
        policy_action = policy(obs)  # shape (8,), range [-1, 1]

        # CAL converts to MuJoCo ctrl with symmetry correction
        ctrl = cal.policy_action_to_ctrl(policy_action)
        data = data.replace(ctrl=ctrl)

        # CAL provides normalized state for observations
        joint_pos = cal.get_normalized_joint_positions(data.qpos)
    """

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        robot_config: Any,
    ):
        """Initialize CAL from MuJoCo model and robot config.

        Args:
            mj_model: MuJoCo model
            robot_config: RobotConfig instance containing:
                - actuated_joints: List of actuated joint configs
                - feet configuration (including contact_scale) for foot state
                - root_imu configuration for root state
        """
        self._mj_model = mj_model
        self._robot_config = robot_config
        self._contact_scale = robot_config.contact_scale
        self._actuated_joints_config = robot_config.actuated_joints

        # Build specs from model and config
        self._joint_specs: List[JointSpec] = []
        self._actuator_specs: List[ActuatorSpec] = []
        self._joint_specs_by_name: Dict[str, JointSpec] = {}
        self._actuator_specs_by_name: Dict[str, ActuatorSpec] = {}

        self._build_specs()

        # Load default positions from MuJoCo keyframe
        self._default_qpos = self._load_default_positions()

        # Update joint specs with default positions
        self._update_default_positions()

        # Precompute JAX arrays for efficient transforms
        self._precompute_arrays()

        # Build extended state specs (foot, root)
        self._build_foot_specs(robot_config)
        self._build_root_spec(robot_config)

    def _build_specs(self) -> None:
        """Build JointSpec and ActuatorSpec from config and MuJoCo model."""
        for i, entry in enumerate(self._actuated_joints_config):
            name = entry["name"]

            # Get joint info from MuJoCo model
            joint_id = mujoco.mj_name2id(
                self._mj_model, mujoco.mjtObj.mjOBJ_JOINT, name
            )
            if joint_id < 0:
                raise ValueError(f"Joint '{name}' not found in MuJoCo model")

            qpos_addr = self._mj_model.jnt_qposadr[joint_id]
            qvel_addr = self._mj_model.jnt_dofadr[joint_id]

            # Get actuator info
            actuator_id = mujoco.mj_name2id(
                self._mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
            )
            if actuator_id < 0:
                raise ValueError(f"Actuator '{name}' not found in MuJoCo model")

            # Get ranges from config
            joint_range = entry.get("range", [-1.57, 1.57])

            # Create JointSpec (default_pos will be updated later)
            joint_spec = JointSpec(
                name=name,
                joint_id=joint_id,
                qpos_addr=qpos_addr,
                qvel_addr=qvel_addr,
                range_min=joint_range[0],
                range_max=joint_range[1],
                default_pos=0.0,  # Will be updated from keyframe
                mirror_sign=entry.get("mirror_sign", 1.0),
                symmetry_pair=entry.get("symmetry_pair"),
            )

            # Get ctrl and force ranges from MuJoCo model
            # Note: ctrl_range is now derived from joint_spec for position actuators
            force_range = tuple(self._mj_model.actuator_forcerange[actuator_id])

            # Get PD gains if available
            kp = None
            kv = None
            # MuJoCo stores gains in actuator_gainprm
            # For position actuators, gainprm[0] is kp
            gainprm = self._mj_model.actuator_gainprm[actuator_id]
            biasprm = self._mj_model.actuator_biasprm[actuator_id]
            if len(gainprm) > 0:
                kp = gainprm[0]
            if len(biasprm) > 2:
                kv = -biasprm[2]  # MuJoCo uses -kv in bias

            # Create ActuatorSpec (ctrl_range is now derived from joint_spec)
            actuator_spec = ActuatorSpec(
                name=name,
                actuator_id=actuator_id,
                joint_spec=joint_spec,
                actuator_type=ActuatorType(entry.get("type", "position")),
                force_range=force_range,
                max_velocity=entry.get("max_velocity", 10.0),
                kp=kp,
                kv=kv,
            )

            self._joint_specs.append(joint_spec)
            self._actuator_specs.append(actuator_spec)
            self._joint_specs_by_name[name] = joint_spec
            self._actuator_specs_by_name[name] = actuator_spec

    def _load_default_positions(self) -> jnp.ndarray:
        """Load default joint positions from MuJoCo model.

        Priority:
        1. "home" keyframe if exists
        2. qpos0 (model default)

        This is the SINGLE SOURCE OF TRUTH for default pose.
        No duplication in robot_config.yaml.
        """
        mj_model = self._mj_model

        # Get qpos addresses for actuated joints
        qpos_addrs = jnp.array([spec.qpos_addr for spec in self._joint_specs])

        # Check if "home" keyframe exists
        if mj_model.nkey > 0:
            # Find "home" keyframe by name
            for key_idx in range(mj_model.nkey):
                # Get keyframe name (MuJoCo stores names in a flat array)
                key_name_adr = mj_model.name_keyadr[key_idx]
                # Read null-terminated string
                key_name = ""
                for c in mj_model.names[key_name_adr:]:
                    if c == 0:
                        break
                    key_name += chr(c)

                if key_name == "home":
                    # Extract actuated joint positions from keyframe
                    full_qpos = mj_model.key_qpos[key_idx]
                    return jnp.array([full_qpos[addr] for addr in qpos_addrs])

        # Fallback to qpos0 (model default)
        return jnp.array([mj_model.qpos0[addr] for addr in qpos_addrs])

    def _update_default_positions(self) -> None:
        """Update JointSpec default_pos from loaded keyframe positions."""
        # Since JointSpec is frozen, we need to recreate them
        updated_joint_specs = []
        updated_actuator_specs = []

        for i, (joint_spec, actuator_spec) in enumerate(
            zip(self._joint_specs, self._actuator_specs)
        ):
            # Create new JointSpec with updated default_pos
            new_joint_spec = JointSpec(
                name=joint_spec.name,
                joint_id=joint_spec.joint_id,
                qpos_addr=joint_spec.qpos_addr,
                qvel_addr=joint_spec.qvel_addr,
                range_min=joint_spec.range_min,
                range_max=joint_spec.range_max,
                default_pos=float(self._default_qpos[i]),
                mirror_sign=joint_spec.mirror_sign,
                symmetry_pair=joint_spec.symmetry_pair,
            )

            # Create new ActuatorSpec with updated JointSpec
            # Note: ctrl_range is derived from joint_spec, not passed as parameter
            new_actuator_spec = ActuatorSpec(
                name=actuator_spec.name,
                actuator_id=actuator_spec.actuator_id,
                joint_spec=new_joint_spec,
                actuator_type=actuator_spec.actuator_type,
                force_range=actuator_spec.force_range,
                max_velocity=actuator_spec.max_velocity,
                kp=actuator_spec.kp,
                kv=actuator_spec.kv,
            )

            updated_joint_specs.append(new_joint_spec)
            updated_actuator_specs.append(new_actuator_spec)
            self._joint_specs_by_name[joint_spec.name] = new_joint_spec
            self._actuator_specs_by_name[actuator_spec.name] = new_actuator_spec

        self._joint_specs = updated_joint_specs
        self._actuator_specs = updated_actuator_specs

    def _precompute_arrays(self) -> None:
        """Precompute JAX arrays for efficient transforms."""
        # Control/joint range centers and spans (for action/position normalization)
        # Note: For position actuators, ctrl_range == joint_range (single source of truth)
        self._ctrl_centers = jnp.array(
            [spec.ctrl_center for spec in self._actuator_specs]
        )
        self._ctrl_spans = jnp.array([spec.ctrl_span for spec in self._actuator_specs])

        # Mirror signs for symmetry correction
        self._mirror_signs = jnp.array([spec.mirror_sign for spec in self._joint_specs])

        # qpos and qvel addresses
        self._actuator_qpos_addrs = jnp.array(
            [spec.qpos_addr for spec in self._joint_specs]
        )
        self._actuator_qvel_addrs = jnp.array(
            [spec.qvel_addr for spec in self._joint_specs]
        )

        # Velocity limits
        self._velocity_limits = jnp.array(
            [spec.max_velocity for spec in self._actuator_specs]
        )

        # Force limits
        self._force_limits = jnp.array(
            [spec.force_limit for spec in self._actuator_specs]
        )

        # Build symmetry swap indices for get_symmetric_action
        self._symmetry_swap_indices = self._build_symmetry_swap_indices()

        # Symmetry flip signs (for joints that flip sign on mirror, e.g., roll joints)
        # For now, use 1.0 for all (no sign flip on swap)
        self._symmetry_flip_signs = jnp.ones(self.num_actuators)

        # Validate configuration
        self._validate_specs()

    def _validate_specs(self) -> None:
        """Validate specs for common configuration errors."""
        errors = []
        warnings = []

        for spec in self._actuator_specs:
            # Check force limits
            if spec.force_limit <= 0:
                errors.append(
                    f"Actuator '{spec.name}' has invalid force_limit={spec.force_limit}. "
                    f"Check forcerange in MuJoCo XML (currently {spec.force_range})."
                )

            # Check joint range
            joint = spec.joint_spec
            raw_span = (joint.range_max - joint.range_min) / 2
            if raw_span <= 0:
                errors.append(
                    f"Joint '{joint.name}' has invalid range [{joint.range_min}, {joint.range_max}]. "
                    f"range_max must be greater than range_min."
                )

            # Check velocity limits
            if spec.max_velocity <= 0:
                warnings.append(
                    f"Actuator '{spec.name}' has max_velocity={spec.max_velocity} <= 0. "
                    f"Using absolute value."
                )

        # Report warnings
        for w in warnings:
            print(f"[CAL WARNING] {w}")

        # Raise on errors
        if errors:
            raise ValueError(
                "CAL configuration errors:\n" + "\n".join(f"  - {e}" for e in errors)
            )

    def _build_symmetry_swap_indices(self) -> jnp.ndarray:
        """Build indices for swapping left/right joints."""
        indices = list(range(self.num_actuators))

        for i, spec in enumerate(self._joint_specs):
            if spec.symmetry_pair and spec.symmetry_pair in self._joint_specs_by_name:
                pair_spec = self._joint_specs_by_name[spec.symmetry_pair]
                pair_idx = self._joint_specs.index(pair_spec)
                indices[i] = pair_idx

        return jnp.array(indices)

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def num_actuators(self) -> int:
        """Number of actuated joints."""
        return len(self._actuator_specs)

    @property
    def joint_specs(self) -> List[JointSpec]:
        """List of joint specifications."""
        return self._joint_specs

    @property
    def actuator_specs(self) -> List[ActuatorSpec]:
        """List of actuator specifications."""
        return self._actuator_specs

    @property
    def qpos_addresses(self) -> jnp.ndarray:
        """Array of qpos addresses for actuated joints."""
        return self._actuator_qpos_addrs

    @property
    def qvel_addresses(self) -> jnp.ndarray:
        """Array of qvel addresses for actuated joints."""
        return self._actuator_qvel_addrs

    # =========================================================================
    # Action Transformation (Policy → MuJoCo)
    # =========================================================================

    def policy_action_to_ctrl(self, action: jax.Array) -> jax.Array:
        """Transform normalized policy action to MuJoCo ctrl.

        USE FOR: Policy inference, training step, RL rollouts.

        This method:
        1. Clips action to [-1, 1] range (defensive - NN can exceed bounds)
        2. Applies symmetry correction (mirror_sign) for mirrored joint ranges
        3. Denormalizes from [-1, 1] to physical ctrl range

        Args:
            action: Normalized action in [-1, 1], shape (num_actuators,)
                    Same action value for left/right produces symmetric motion.
                    Values outside [-1, 1] are clipped for robustness.

        Returns:
            ctrl: MuJoCo control signal (target angle in radians for position actuators)

        Example:
            # Policy outputs action[0] = 0.5 for left hip, action[4] = 0.5 for right hip
            # Both should move forward (symmetric motion)
            policy_action = jnp.array([0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
            ctrl = cal.policy_action_to_ctrl(policy_action)
            # ctrl accounts for mirrored joint ranges automatically
        """
        # Clip action to valid range [-1, 1] for robustness
        # Neural networks can occasionally produce values slightly outside bounds
        action = jnp.clip(action, -1.0, 1.0)

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
        """
        # For position actuators: ctrl = target angle (no transformation needed)
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
        normalized = (ctrl - self._ctrl_centers) / (self._ctrl_spans + 1e-6)
        # Reverse mirror correction
        return normalized * self._mirror_signs

    def ctrl_to_physical_angles(self, ctrl: jax.Array) -> jax.Array:
        """Extract physical joint angles from MuJoCo ctrl.

        Inverse of physical_angles_to_ctrl().

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
    # DESIGN PATTERN: Single method with REQUIRED `normalize` parameter
    #
    # The `normalize` parameter is REQUIRED (no default) to:
    # - Force explicit intent at call site
    # - Prevent bugs from accidentally using wrong version
    # - Make code self-documenting
    #
    # Convention:
    # - normalize=True: For policy observations (NN input)
    # - normalize=False: For logging, debugging, Sim2Real (physical units)
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
            If normalize=True: -1 = min limit, 0 = center, +1 = max limit
            If normalize=False: radians
        """
        joint_pos = qpos[self._actuator_qpos_addrs]

        if normalize:
            return (joint_pos - self._ctrl_centers) / (self._ctrl_spans + 1e-6)
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
            If normalize=True: clipped to [-1, 1]
            If normalize=False: rad/s
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
            qfrc_actuator: Actuator forces from data.qfrc_actuator
            normalize: REQUIRED. True to normalize by force_range,
                      False for raw torques (Nm)

        Returns:
            Actuator forces (num_actuators,), optionally normalized
        """
        torques = qfrc_actuator[self._actuator_qvel_addrs]

        if normalize:
            torques = torques / (self._force_limits + 1e-6)

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
        ctrl_dt: float = 0.02,
    ) -> ControlCommand:
        """Transform normalized policy action to control command with trajectory info.

        USE FOR: Policy inference, RL rollouts, Sim2Real deployment.

        In Simulation:
            - Pass current_qpos=None (velocity not needed)
            - MuJoCo PD controller handles trajectory

        In Real Hardware:
            - Pass current_qpos to compute target velocity
            - Motor controller executes trajectory with velocity profile

        Args:
            action: Normalized action [-1, 1], shape (num_actuators,)
            current_qpos: Current joint positions. If provided, target_velocity
                          is computed for Sim2Real trajectory control.
            ctrl_dt: Control period in seconds (time to reach target)

        Returns:
            ControlCommand with target_position and optional velocity
        """
        # Get target position with symmetry correction
        target_position = self.policy_action_to_ctrl(action)

        # Compute velocity if current_qpos is provided (for Sim2Real)
        target_velocity = None
        if current_qpos is not None:
            target_velocity = self._compute_velocity(
                target_position, current_qpos, ctrl_dt
            )

        return ControlCommand(
            target_position=target_position,
            target_velocity=target_velocity,
            duration=ctrl_dt if target_velocity is not None else None,
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
        return self._velocity_limits

    # =========================================================================
    # Validation & Debugging
    # =========================================================================

    def validate_action(self, action: jax.Array) -> bool:
        """Check if action is within expected bounds."""
        return bool(jnp.all((action >= -1.0) & (action <= 1.0)))

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
        ctrl_mins = jnp.array([spec.ctrl_range[0] for spec in self._actuator_specs])
        ctrl_maxs = jnp.array([spec.ctrl_range[1] for spec in self._actuator_specs])

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
        """Get ctrl values that produce the default (keyframe) pose.

        Loaded from MuJoCo keyframe at initialization.
        """
        return self._default_qpos

    def get_joint_spec(self, name: str) -> Optional[JointSpec]:
        """Get JointSpec by name."""
        return self._joint_specs_by_name.get(name)

    def get_actuator_spec(self, name: str) -> Optional[ActuatorSpec]:
        """Get ActuatorSpec by name."""
        return self._actuator_specs_by_name.get(name)

    # =========================================================================
    # Semantic Joint Position Accessors (for gait shaping rewards)
    # =========================================================================
    # Joint names are defined in robot_config (single source of truth).
    # These methods allow env to access specific joint positions without
    # hardcoding joint names, maintaining flexibility for different robots.

    def get_hip_pitch_positions(
        self,
        qpos: jax.Array,
        normalize: bool = True,
    ) -> tuple[float, float]:
        """Get hip pitch positions (left, right).

        Joint names are defined in robot_config (single source of truth).

        Args:
            qpos: Full qpos array from MJX data
            normalize: If True, return normalized positions [-1, 1]

        Returns:
            (left_hip_pitch, right_hip_pitch) positions
        """
        joint_pos = self.get_joint_positions(qpos, normalize=normalize)
        left_idx, right_idx = self._robot_config.get_hip_pitch_indices()
        return joint_pos[left_idx], joint_pos[right_idx]

    def get_knee_pitch_positions(
        self,
        qpos: jax.Array,
        normalize: bool = True,
    ) -> tuple[float, float]:
        """Get knee pitch positions (left, right).

        Joint names are defined in robot_config (single source of truth).

        Args:
            qpos: Full qpos array from MJX data
            normalize: If True, return normalized positions [-1, 1]

        Returns:
            (left_knee_pitch, right_knee_pitch) positions
        """
        joint_pos = self.get_joint_positions(qpos, normalize=normalize)
        left_idx, right_idx = self._robot_config.get_knee_pitch_indices()
        return joint_pos[left_idx], joint_pos[right_idx]

    def print_specs(self) -> None:
        """Print all joint and actuator specs for debugging."""
        print("\nJoint Specifications:")
        print("-" * 80)
        print(
            f"{'Name':<20} {'Range':<20} {'Center':>8} {'Span':>8} "
            f"{'Mirror':>7} {'Default':>8}"
        )
        print("-" * 80)
        for spec in self._joint_specs:
            print(
                f"{spec.name:<20} [{spec.range_min:>6.3f}, {spec.range_max:>6.3f}] "
                f"{spec.range_center:>8.3f} {spec.range_span:>8.3f} "
                f"{spec.mirror_sign:>7.1f} {spec.default_pos:>8.4f}"
            )

        print("\nActuator Specifications:")
        print("-" * 80)
        print(
            f"{'Name':<20} {'Type':<10} {'Ctrl Range':<20} {'Max Vel':>8} "
            f"{'kp':>8} {'kv':>8}"
        )
        print("-" * 80)
        for spec in self._actuator_specs:
            kp_str = f"{spec.kp:.1f}" if spec.kp is not None else "N/A"
            kv_str = f"{spec.kv:.1f}" if spec.kv is not None else "N/A"
            print(
                f"{spec.name:<20} {spec.actuator_type.value:<10} "
                f"[{spec.ctrl_range[0]:>6.3f}, {spec.ctrl_range[1]:>6.3f}] "
                f"{spec.max_velocity:>8.1f} {kp_str:>8} {kv_str:>8}"
            )

    # =========================================================================
    # Extended State: Internal Build Methods
    # =========================================================================

    def _build_foot_specs(self, robot_config: Any) -> None:
        """Build FootSpec from robot_config.

        Uses robot_config.get_foot_geom_names() for geom names (single source of truth).
        """
        # Get geom names from robot_config method (not raw_config)
        left_toe, left_heel, right_toe, right_heel = robot_config.get_foot_geom_names()

        # Get body names from robot_config properties
        left_body = robot_config.left_foot_body
        right_body = robot_config.right_foot_body

        self._foot_specs = [
            FootSpec(
                name="left_foot",
                body_name=left_body,
                body_id=mujoco.mj_name2id(
                    self._mj_model, mujoco.mjtObj.mjOBJ_BODY, left_body
                ),
                toe_geom_id=mujoco.mj_name2id(
                    self._mj_model, mujoco.mjtObj.mjOBJ_GEOM, left_toe
                ),
                heel_geom_id=mujoco.mj_name2id(
                    self._mj_model, mujoco.mjtObj.mjOBJ_GEOM, left_heel
                ),
                symmetry_pair="right_foot",
            ),
            FootSpec(
                name="right_foot",
                body_name=right_body,
                body_id=mujoco.mj_name2id(
                    self._mj_model, mujoco.mjtObj.mjOBJ_BODY, right_body
                ),
                toe_geom_id=mujoco.mj_name2id(
                    self._mj_model, mujoco.mjtObj.mjOBJ_GEOM, right_toe
                ),
                heel_geom_id=mujoco.mj_name2id(
                    self._mj_model, mujoco.mjtObj.mjOBJ_GEOM, right_heel
                ),
                symmetry_pair="left_foot",
            ),
        ]

    def _build_root_spec(self, robot_config: Any) -> None:
        """Build RootSpec from robot_config.

        Uses robot_config properties (single source of truth):
        - floating_base_name: Joint name
        - floating_base_body: Body name
        - orientation_sensor, root_gyro_sensor, etc.: Sensor names
        """
        root_body = robot_config.floating_base_body
        fb_name = robot_config.floating_base_name

        # Get joint id and addresses
        joint_id = (
            mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_JOINT, fb_name)
            if fb_name
            else -1
        )
        qpos_addr = self._mj_model.jnt_qposadr[joint_id] if joint_id >= 0 else 0
        qvel_addr = self._mj_model.jnt_dofadr[joint_id] if joint_id >= 0 else 0

        # Get IMU sensor names from robot_config properties (single source of truth)
        orientation_sensor = robot_config.orientation_sensor
        gyro_sensor = robot_config.root_gyro_sensor
        accel_sensor = robot_config.root_accel_sensor
        local_linvel_sensor = robot_config.local_linvel_sensor

        self._root_spec = RootSpec(
            body_name=root_body,
            body_id=mujoco.mj_name2id(
                self._mj_model, mujoco.mjtObj.mjOBJ_BODY, root_body
            ),
            qpos_addr=qpos_addr,
            qvel_addr=qvel_addr,
            orientation_sensor=orientation_sensor,
            gyro_sensor=gyro_sensor,
            accel_sensor=accel_sensor,
            local_linvel_sensor=local_linvel_sensor,
        )

    # =========================================================================
    # Foot State Access
    # =========================================================================

    @property
    def foot_specs(self) -> List[FootSpec]:
        """List of foot specifications."""
        return getattr(self, "_foot_specs", [])

    def get_geom_based_foot_contacts(
        self,
        data: mjx.Data,
        normalize: bool,  # REQUIRED - no default
    ) -> jax.Array:
        """Get foot contact values with per-geom granularity.

        Use this for AMP features where per-contact-point detail matters.
        For reward computation (per-foot totals), use get_aggregated_foot_contacts() instead.

        Args:
            data: MJX simulation data
            normalize: REQUIRED. True for tanh-normalized [0, 1],
                      False for raw normal forces (Newtons)

        Returns:
            foot_contacts: (4,) - [left_toe, left_heel, right_toe, right_heel]
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

    def get_aggregated_foot_contacts(
        self,
        data: mjx.Data,
    ) -> tuple[jax.Array, jax.Array]:
        """Get per-foot contact forces (aggregated toe + heel).

        Use this for reward computation where you need per-foot totals
        for threshold comparison (slip gating, clearance, gait periodicity).
        For AMP features (per-geom detail), use get_geom_based_foot_contacts() instead.

        Args:
            data: MJX simulation data

        Returns:
            (left_force, right_force): Scalar normal forces in Newtons
        """
        # Reuse get_geom_based_foot_contacts to avoid duplication
        # Order: [left_toe, left_heel, right_toe, right_heel]
        raw_contacts = self.get_geom_based_foot_contacts(data, normalize=False)
        left_force = raw_contacts[0] + raw_contacts[1]  # toe + heel
        right_force = raw_contacts[2] + raw_contacts[3]  # toe + heel
        return left_force, right_force

    def get_foot_positions(
        self,
        data: mjx.Data,
        normalize: bool,  # REQUIRED - no default
        frame: CoordinateFrame = CoordinateFrame.WORLD,
    ) -> tuple[jax.Array, jax.Array]:
        """Get foot body positions.

        Args:
            data: MJX simulation data
            normalize: REQUIRED. True to normalize by root height,
                      False for raw positions (meters)
            frame: Coordinate frame (WORLD or HEADING_LOCAL)

        Returns:
            (left_foot_pos, right_foot_pos): Each (3,) position vector
        """
        left_foot = self._foot_specs[0]
        right_foot = self._foot_specs[1]

        left_pos = data.xpos[left_foot.body_id]
        right_pos = data.xpos[right_foot.body_id]

        if frame == CoordinateFrame.HEADING_LOCAL:
            quat = data.qpos[
                self._root_spec.qpos_addr + 3 : self._root_spec.qpos_addr + 7
            ]
            left_pos = self._world_to_heading_local_vec(quat, left_pos)
            right_pos = self._world_to_heading_local_vec(quat, right_pos)

        if normalize:
            root_height = self.get_root_height(data)
            left_pos = left_pos / (root_height + 1e-6)
            right_pos = right_pos / (root_height + 1e-6)

        return left_pos, right_pos

    def get_foot_velocities(
        self,
        data: mjx.Data,
        prev_left_pos: jax.Array,
        prev_right_pos: jax.Array,
        dt: float,
        normalize: bool,  # REQUIRED - no default
        frame: CoordinateFrame = CoordinateFrame.WORLD,
        max_foot_velocity: float = 5.0,
    ) -> tuple[jax.Array, jax.Array]:
        """Get foot velocities via finite difference.

        Args:
            data: Current MJX data
            prev_left_pos: Previous left foot position (3,)
            prev_right_pos: Previous right foot position (3,)
            dt: Time step for finite difference
            normalize: REQUIRED. True to normalize by max_foot_velocity to [-1, 1],
                      False for raw velocities (m/s)
            frame: Coordinate frame
            max_foot_velocity: Max expected foot velocity for normalization

        Returns:
            (left_vel, right_vel): Each (3,) velocity vector
        """
        curr_left_pos, curr_right_pos = self.get_foot_positions(
            data, normalize=False, frame=frame
        )

        left_vel = (curr_left_pos - prev_left_pos) / dt
        right_vel = (curr_right_pos - prev_right_pos) / dt

        if normalize:
            left_vel = jnp.clip(left_vel / max_foot_velocity, -1.0, 1.0)
            right_vel = jnp.clip(right_vel / max_foot_velocity, -1.0, 1.0)

        return left_vel, right_vel

    def get_foot_clearances(
        self,
        data: mjx.Data,
        normalize: bool,  # REQUIRED - no default
    ) -> tuple[jax.Array, jax.Array]:
        """Get foot height above ground (for swing phase rewards).

        Args:
            data: MJX simulation data
            normalize: REQUIRED. True to normalize by root height,
                      False for raw height (meters)

        Returns:
            (left_clearance, right_clearance): Scalar height values
        """
        left_pos, right_pos = self.get_foot_positions(
            data, normalize=False, frame=CoordinateFrame.WORLD
        )
        left_clearance = left_pos[2]  # Z coordinate
        right_clearance = right_pos[2]

        if normalize:
            root_height = self.get_root_height(data)
            left_clearance = left_clearance / (root_height + 1e-6)
            right_clearance = right_clearance / (root_height + 1e-6)

        return left_clearance, right_clearance

    def _get_geom_contact_force(
        self,
        data: mjx.Data,
        geom_id: int,
    ) -> jax.Array:
        """Get normal contact force for a geom (solver-native efc_force)."""
        # Use data._impl to avoid deprecation warnings
        contact = data._impl.contact
        geom1_match = contact.geom1 == geom_id
        geom2_match = contact.geom2 == geom_id
        is_our_contact = geom1_match | geom2_match

        # efc_force contains constraint forces; contact.efc_address maps to them
        normal_forces = data._impl.efc_force[contact.efc_address]
        our_forces = jnp.where(is_our_contact, jnp.abs(normal_forces), 0.0)
        return jnp.sum(our_forces)

    # =========================================================================
    # Root State Access (4 Core APIs using Pose3D/Velocity3D)
    # =========================================================================

    @property
    def root_spec(self) -> Optional[RootSpec]:
        """Root body specification."""
        return getattr(self, "_root_spec", None)

    def get_root_height(self, data: mjx.Data) -> jax.Array:
        """Get root body height (Z position).

        Args:
            data: MJX simulation data

        Returns:
            height: Scalar height value in meters
        """
        return data.qpos[self._root_spec.qpos_addr + 2]

    def get_root_pose(
        self,
        data: mjx.Data,
        frame: CoordinateFrame = CoordinateFrame.WORLD,
    ) -> Pose3D:
        """Get root body pose with named field access.

        Args:
            data: MJX simulation data
            frame: Coordinate frame for position

        Returns:
            Pose3D with:
                .position: (3,) [x, y, z] in meters
                .orientation: (4,) quaternion [qw, qx, qy, qz]
                .frame: CoordinateFrame
                .height: scalar z (convenience property)
                .xy: (2,) [x, y] (convenience property)
        """
        position = data.qpos[self._root_spec.qpos_addr : self._root_spec.qpos_addr + 3]
        orientation = data.qpos[
            self._root_spec.qpos_addr + 3 : self._root_spec.qpos_addr + 7
        ]

        if frame == CoordinateFrame.HEADING_LOCAL:
            position = self._world_to_heading_local_vec(orientation, position)

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
        """
        quat = data.qpos[self._root_spec.qpos_addr + 3 : self._root_spec.qpos_addr + 7]

        # Linear velocity
        if frame == CoordinateFrame.LOCAL and self._root_spec.local_linvel_sensor:
            linvel = self._get_sensor_data(data, self._root_spec.local_linvel_sensor)
        else:
            linvel = data.qvel[
                self._root_spec.qvel_addr : self._root_spec.qvel_addr + 3
            ]
            if frame == CoordinateFrame.HEADING_LOCAL:
                linvel = self._world_to_heading_local_vec(quat, linvel)
            elif frame == CoordinateFrame.LOCAL:
                linvel = self._rotate_vector_by_quat_inverse(linvel, quat)

        # Angular velocity
        angvel = data.qvel[
            self._root_spec.qvel_addr + 3 : self._root_spec.qvel_addr + 6
        ]
        if frame == CoordinateFrame.HEADING_LOCAL:
            angvel = self._world_to_heading_local_vec(quat, angvel)
        elif frame == CoordinateFrame.LOCAL:
            angvel = self._rotate_vector_by_quat_inverse(angvel, quat)

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
        # Get orientation from chest_imu_quat sensor (BNO085) or qpos
        if self._root_spec.orientation_sensor:
            quat = self._get_sensor_data(data, self._root_spec.orientation_sensor)
        else:
            quat = data.qpos[
                self._root_spec.qpos_addr + 3 : self._root_spec.qpos_addr + 7
            ]

        # Rotate world gravity [0, 0, -1] into local frame
        world_gravity = jnp.array([0.0, 0.0, -1.0])
        return self._rotate_vector_by_quat_inverse(world_gravity, quat)

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _get_sensor_data(self, data: mjx.Data, sensor_name: str) -> jax.Array:
        """Get sensor data by name.

        Args:
            data: MJX simulation data
            sensor_name: Name of the sensor

        Returns:
            Sensor data array
        """
        sensor_id = mujoco.mj_name2id(
            self._mj_model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name
        )
        if sensor_id < 0:
            raise ValueError(f"Sensor '{sensor_name}' not found in MuJoCo model")

        adr = self._mj_model.sensor_adr[sensor_id]
        dim = self._mj_model.sensor_dim[sensor_id]
        return data.sensordata[adr : adr + dim]

    def _world_to_heading_local_vec(
        self,
        quat: jax.Array,
        vec: jax.Array,
    ) -> jax.Array:
        """Rotate world-frame vector to heading-local frame.

        Heading-local frame: world frame rotated by -yaw (heading direction is +X).

        Args:
            quat: Root quaternion [qw, qx, qy, qz]
            vec: 3D vector in world frame

        Returns:
            Vector in heading-local frame
        """
        # Extract sin/cos of yaw from quaternion (inline for self-containment)
        quat_norm = quat / (jnp.linalg.norm(quat) + 1e-8)
        w, x, y, z = quat_norm[0], quat_norm[1], quat_norm[2], quat_norm[3]
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        norm = jnp.sqrt(siny_cosp * siny_cosp + cosy_cosp * cosy_cosp + 1e-8)
        sin_yaw = siny_cosp / norm
        cos_yaw = cosy_cosp / norm

        # Rotate world vec into heading-local frame (R_z(-yaw))
        vx, vy, vz = vec[0], vec[1], vec[2]
        vx_heading = cos_yaw * vx + sin_yaw * vy
        vy_heading = -sin_yaw * vx + cos_yaw * vy
        return jnp.array([vx_heading, vy_heading, vz])

    def _rotate_vector_by_quat_inverse(
        self,
        vec: jax.Array,
        quat: jax.Array,
    ) -> jax.Array:
        """Rotate a vector by the inverse of a quaternion.

        Used to transform world-frame vectors into local body frame.

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
        return jnp.array(
            [
                r00 * vx + r10 * vy + r20 * vz,
                r01 * vx + r11 * vy + r21 * vz,
                r02 * vx + r12 * vy + r22 * vz,
            ]
        )
