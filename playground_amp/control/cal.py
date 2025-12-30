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

from playground_amp.control.specs import ActuatorSpec, ControlCommand, JointSpec
from playground_amp.control.types import ActuatorType


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
        actuated_joints_config: List[Dict[str, Any]],
    ):
        """Initialize CAL from MuJoCo model and config.

        Args:
            mj_model: MuJoCo model
            actuated_joints_config: List of actuated joint configs from robot_config.yaml
                Each entry should have: name, type, range, symmetry_pair, mirror_sign, max_velocity
        """
        self._mj_model = mj_model
        self._actuated_joints_config = actuated_joints_config

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
