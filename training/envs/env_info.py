"""Typed environment state structures for WildRobot.

This module defines the strict schema for algorithm-critical fields that must
exist at every step. Following industry best practice for JAX-based RL:

1. Algorithm-critical persistent state lives in `info["wr"]` as a typed pytree
2. Wrapper-injected fields (episode_metrics, truncation, etc.) stay at top level
3. Required fields fail loudly when missing, not silently default to zeros

Usage:
    # In env.step():
    wr_info = WildRobotInfo(
        prev_root_pos=curr_root_pos,
        prev_root_quat=curr_root_quat,
        ...
    )
    info = {**wrapper_info, "wr": wr_info}

    # In training code:
    wr = env_state.info["wr"]  # Typed access, fails if missing
    velocity = compute_velocity(wr.prev_root_pos, curr_pos)

See: https://flax.readthedocs.io/en/latest/api_reference/flax.struct.html
"""

from __future__ import annotations

from typing import Any

from training.envs.disturbance import DisturbanceSchedule

import jax.numpy as jnp

# Fixed maximum latency for IMU history buffers (must match EnvConfig.imu_max_latency_steps)
IMU_MAX_LATENCY = 4
IMU_HIST_LEN = IMU_MAX_LATENCY + 1
PRIVILEGED_OBS_DIM = 12

# Use flax.struct for JAX-compatible dataclass with pytree registration
# Falls back to NamedTuple if flax not available
try:
    import flax.struct as struct

    @struct.dataclass
    class WildRobotInfo:
        """Strict schema for WildRobot algorithm-critical info fields.

        These fields are REQUIRED at every step and must not be accessed via
        .get() with defaults. Missing fields should cause hard failures.

        All fields use post-step convention: they reflect state AFTER the
        physics step, to be used as "previous" values in the next step.

        Attributes:
            step_count: Current step count in episode (0 at reset)
            prev_action: Action applied in this step (for action rate penalty)
            truncated: 1.0 if episode ended due to max steps (success metric)
            velocity_cmd: Target velocity for this episode (scalar, m/s)

            # Finite-diff velocity computation (AMP feature parity)
            prev_root_pos: Root position after this step (3,)
            prev_root_quat: Root quaternion (wxyz) after this step (4,)

            # Slip computation
            prev_left_foot_pos: Left foot position after this step (3,)
            prev_right_foot_pos: Right foot position after this step (3,)

            # AMP features
            foot_contacts: Normalized foot contact forces (4,)
            root_height: Root height after this step (scalar)

            # Disturbance pushes (episode-constant schedule)
            push_schedule: DisturbanceSchedule
        """
        # Core bookkeeping
        step_count: jnp.ndarray  # shape=()
        prev_action: jnp.ndarray  # shape=(action_size,), action applied this step
        pending_action: jnp.ndarray  # shape=(action_size,), next filtered action
        truncated: jnp.ndarray  # shape=(), sticky through auto-reset
        velocity_cmd: jnp.ndarray  # shape=(), target velocity for episode

        # Finite-diff velocity (post-step values)
        prev_root_pos: jnp.ndarray  # shape=(3,)
        prev_root_quat: jnp.ndarray  # shape=(4,) wxyz

        # Slip computation (post-step values)
        prev_left_foot_pos: jnp.ndarray  # shape=(3,)
        prev_right_foot_pos: jnp.ndarray  # shape=(3,)

        # IMU history buffers (fixed-size: IMU_MAX_LATENCY + 1)
        # Most-recent IMU at index 0; older entries follow.
        imu_quat_hist: jnp.ndarray  # shape=(IMU_MAX_LATENCY+1, 4)
        imu_gyro_hist: jnp.ndarray  # shape=(IMU_MAX_LATENCY+1, 3)

        # AMP features (post-step values)
        foot_contacts: jnp.ndarray  # shape=(4,)
        root_height: jnp.ndarray  # shape=()
        # Stepping event detection (post-step values)
        prev_left_loaded: jnp.ndarray  # shape=()
        prev_right_loaded: jnp.ndarray  # shape=()
        cycle_start_forward_x: jnp.ndarray  # shape=(), accumulated heading-local cycle progress
        last_touchdown_root_pos: jnp.ndarray  # shape=(3,), world root position at last touchdown
        last_touchdown_foot: jnp.ndarray  # shape=(), -1 none, 0 left, 1 right
        critic_obs: jnp.ndarray  # shape=(PRIVILEGED_OBS_DIM,)
        # v0.19.3: Locomotion reference state + nominal IK joint target
        loc_ref_phase_time: jnp.ndarray  # shape=()
        loc_ref_stance_foot: jnp.ndarray  # shape=(), int32 (0 left, 1 right)
        loc_ref_switch_count: jnp.ndarray  # shape=(), int32
        loc_ref_mode_id: jnp.ndarray  # shape=(), int32 (v2 hybrid mode)
        loc_ref_mode_time: jnp.ndarray  # shape=()
        loc_ref_startup_route_progress: jnp.ndarray  # shape=()
        loc_ref_startup_route_ceiling: jnp.ndarray  # shape=()
        loc_ref_startup_route_stage_id: jnp.ndarray  # shape=(), int32
        loc_ref_startup_route_transition_reason: jnp.ndarray  # shape=(), int32
        loc_ref_gait_phase_sin: jnp.ndarray  # shape=()
        loc_ref_gait_phase_cos: jnp.ndarray  # shape=()
        loc_ref_next_foothold: jnp.ndarray  # shape=(2,)
        loc_ref_swing_pos: jnp.ndarray  # shape=(3,)
        loc_ref_swing_vel: jnp.ndarray  # shape=(3,)
        loc_ref_pelvis_height: jnp.ndarray  # shape=()
        loc_ref_pelvis_roll: jnp.ndarray  # shape=()
        loc_ref_pelvis_pitch: jnp.ndarray  # shape=()
        loc_ref_history: jnp.ndarray  # shape=(4,)
        nominal_q_ref: jnp.ndarray  # shape=(action_size,)
        push_schedule: DisturbanceSchedule

        # -----------------------------------------------------------------------
        # M3: Step FSM state (foot-placement base controller)
        # Stored here so the FSM survives across steps in lax.scan rollouts.
        # All fields initialized to zero (STANCE phase) at reset.
        # -----------------------------------------------------------------------
        #: FSM phase: STANCE=0, SWING=1, TOUCHDOWN_RECOVER=2
        fsm_phase: jnp.ndarray        # shape=(), int32
        #: 0 = left foot swinging, 1 = right foot swinging
        fsm_swing_foot: jnp.ndarray   # shape=(), int32
        #: Number of control ticks spent in current FSM phase
        fsm_phase_ticks: jnp.ndarray  # shape=(), int32
        #: Frozen touchdown target x (heading-local, m)
        fsm_frozen_tx: jnp.ndarray    # shape=()
        #: Frozen touchdown target y (heading-local, m)
        fsm_frozen_ty: jnp.ndarray    # shape=()
        #: X position of swing foot when swing phase began (heading-local, m)
        fsm_swing_sx: jnp.ndarray     # shape=()
        #: Y position of swing foot when swing phase began (heading-local, m)
        fsm_swing_sy: jnp.ndarray     # shape=()
        #: Consecutive ticks the swing foot has been loaded (touchdown detection)
        fsm_touch_hold: jnp.ndarray   # shape=(), int32
        #: Consecutive ticks need_step has exceeded trigger threshold (STANCE)
        fsm_trigger_hold: jnp.ndarray # shape=(), int32

        # -----------------------------------------------------------------------
        # v0.17.1+: Recovery metrics state tracking
        # Enables proper episode-level recovery metric computation.
        # -----------------------------------------------------------------------
        #: Whether recovery window is currently active
        recovery_active: jnp.ndarray  # shape=(), bool
        #: Steps elapsed since push end within the active recovery window
        recovery_age: jnp.ndarray  # shape=(), int32
        #: Support foot at previous step (-1=none, 0=left, 1=right)
        recovery_last_support_foot: jnp.ndarray  # shape=(), int32
        #: Whether first touchdown after push_end has been recorded
        recovery_first_touchdown_recorded: jnp.ndarray  # shape=(), bool
        #: Latched latency from push_end to first touchdown
        recovery_first_step_latency: jnp.ndarray  # shape=(), int32
        #: Accumulated touchdown count during recovery period
        recovery_touchdown_count: jnp.ndarray  # shape=(), int32
        #: Accumulated support-foot changes during recovery period
        recovery_support_foot_changes: jnp.ndarray  # shape=(), int32
        #: Whether first liftoff after push_end has been recorded
        recovery_first_liftoff_recorded: jnp.ndarray  # shape=(), bool
        #: Latched latency from push_end to first liftoff
        recovery_first_liftoff_latency: jnp.ndarray  # shape=(), int32
        #: Age at first touchdown (used for +10 tick diagnostics)
        recovery_first_touchdown_age: jnp.ndarray  # shape=(), int32
        #: Whether a visible recovery step (liftoff then touchdown) was recorded
        recovery_visible_step_recorded: jnp.ndarray  # shape=(), bool
        #: Pitch rate sampled at push end
        recovery_pitch_rate_at_push_end: jnp.ndarray  # shape=()
        #: Pitch rate sampled at first touchdown
        recovery_pitch_rate_at_touchdown: jnp.ndarray  # shape=()
        #: Pitch rate sampled 10 ticks after first touchdown
        recovery_pitch_rate_after_10t: jnp.ndarray  # shape=()
        #: Capture-point error norm sampled at push end
        recovery_capture_error_at_push_end: jnp.ndarray  # shape=()
        #: Capture-point error norm sampled at first touchdown
        recovery_capture_error_at_touchdown: jnp.ndarray  # shape=()
        #: Capture-point error norm sampled 10 ticks after first touchdown
        recovery_capture_error_after_10t: jnp.ndarray  # shape=()
        #: First touchdown foot x in heading-local frame
        recovery_first_step_dx: jnp.ndarray  # shape=()
        #: First touchdown foot y in heading-local frame
        recovery_first_step_dy: jnp.ndarray  # shape=()
        #: First touchdown x-target error
        recovery_first_step_target_err_x: jnp.ndarray  # shape=()
        #: First touchdown y-target error
        recovery_first_step_target_err_y: jnp.ndarray  # shape=()
        #: Minimum root height observed within the active recovery window
        recovery_min_height: jnp.ndarray  # shape=()
        #: Maximum knee flexion magnitude observed within the active recovery window
        recovery_max_knee_flex: jnp.ndarray  # shape=()

        # v0.17.4t: teacher step-target diagnostics (training-time only)
        teacher_active: jnp.ndarray  # shape=()
        teacher_step_required_soft: jnp.ndarray  # shape=()
        teacher_step_required_hard: jnp.ndarray  # shape=()
        teacher_swing_foot: jnp.ndarray  # shape=(), -1 none, 0 left, 1 right
        teacher_target_step_x: jnp.ndarray  # shape=()
        teacher_target_step_y: jnp.ndarray  # shape=()
        teacher_target_reachable: jnp.ndarray  # shape=()

        # v0.17.3: architecture-pivot controller debug state (scalar latches)
        mpc_planner_active: jnp.ndarray  # shape=()
        mpc_controller_active: jnp.ndarray  # shape=()
        mpc_target_com_x: jnp.ndarray  # shape=()
        mpc_target_com_y: jnp.ndarray  # shape=()
        mpc_target_step_x: jnp.ndarray  # shape=()
        mpc_target_step_y: jnp.ndarray  # shape=()
        mpc_support_state: jnp.ndarray  # shape=()
        mpc_step_requested: jnp.ndarray  # shape=()

        # v0.17.3b: episode-constant domain randomization state
        domain_rand_friction_scale: jnp.ndarray  # shape=()
        domain_rand_mass_scales: jnp.ndarray  # shape=(num_bodies,)
        domain_rand_kp_scales: jnp.ndarray  # shape=(action_size,)
        domain_rand_frictionloss_scales: jnp.ndarray  # shape=(action_size,)
        domain_rand_joint_offsets: jnp.ndarray  # shape=(action_size,)

except ImportError:
    # Fallback to NamedTuple if flax not available
    from typing import NamedTuple

    class WildRobotInfo(NamedTuple):
        """Strict schema for WildRobot algorithm-critical info fields."""
        step_count: jnp.ndarray
        prev_action: jnp.ndarray
        pending_action: jnp.ndarray
        truncated: jnp.ndarray
        velocity_cmd: jnp.ndarray
        prev_root_pos: jnp.ndarray
        prev_root_quat: jnp.ndarray
        prev_left_foot_pos: jnp.ndarray
        prev_right_foot_pos: jnp.ndarray
        imu_quat_hist: jnp.ndarray
        imu_gyro_hist: jnp.ndarray
        foot_contacts: jnp.ndarray
        root_height: jnp.ndarray
        prev_left_loaded: jnp.ndarray
        prev_right_loaded: jnp.ndarray
        cycle_start_forward_x: jnp.ndarray
        last_touchdown_root_pos: jnp.ndarray
        last_touchdown_foot: jnp.ndarray
        critic_obs: jnp.ndarray
        loc_ref_phase_time: jnp.ndarray
        loc_ref_stance_foot: jnp.ndarray
        loc_ref_switch_count: jnp.ndarray
        loc_ref_mode_id: jnp.ndarray
        loc_ref_mode_time: jnp.ndarray
        loc_ref_startup_route_progress: jnp.ndarray
        loc_ref_startup_route_ceiling: jnp.ndarray
        loc_ref_startup_route_stage_id: jnp.ndarray
        loc_ref_startup_route_transition_reason: jnp.ndarray
        loc_ref_gait_phase_sin: jnp.ndarray
        loc_ref_gait_phase_cos: jnp.ndarray
        loc_ref_next_foothold: jnp.ndarray
        loc_ref_swing_pos: jnp.ndarray
        loc_ref_swing_vel: jnp.ndarray
        loc_ref_pelvis_height: jnp.ndarray
        loc_ref_pelvis_roll: jnp.ndarray
        loc_ref_pelvis_pitch: jnp.ndarray
        loc_ref_history: jnp.ndarray
        nominal_q_ref: jnp.ndarray
        # M3 FSM state
        fsm_phase: jnp.ndarray
        fsm_swing_foot: jnp.ndarray
        fsm_phase_ticks: jnp.ndarray
        fsm_frozen_tx: jnp.ndarray
        fsm_frozen_ty: jnp.ndarray
        fsm_swing_sx: jnp.ndarray
        fsm_swing_sy: jnp.ndarray
        fsm_touch_hold: jnp.ndarray
        fsm_trigger_hold: jnp.ndarray
        # v0.17.1+: Recovery metrics state
        recovery_active: jnp.ndarray
        recovery_age: jnp.ndarray
        recovery_last_support_foot: jnp.ndarray
        recovery_first_touchdown_recorded: jnp.ndarray
        recovery_first_step_latency: jnp.ndarray
        recovery_touchdown_count: jnp.ndarray
        recovery_support_foot_changes: jnp.ndarray
        recovery_first_liftoff_recorded: jnp.ndarray
        recovery_first_liftoff_latency: jnp.ndarray
        recovery_first_touchdown_age: jnp.ndarray
        recovery_visible_step_recorded: jnp.ndarray
        recovery_pitch_rate_at_push_end: jnp.ndarray
        recovery_pitch_rate_at_touchdown: jnp.ndarray
        recovery_pitch_rate_after_10t: jnp.ndarray
        recovery_capture_error_at_push_end: jnp.ndarray
        recovery_capture_error_at_touchdown: jnp.ndarray
        recovery_capture_error_after_10t: jnp.ndarray
        recovery_first_step_dx: jnp.ndarray
        recovery_first_step_dy: jnp.ndarray
        recovery_first_step_target_err_x: jnp.ndarray
        recovery_first_step_target_err_y: jnp.ndarray
        recovery_min_height: jnp.ndarray
        recovery_max_knee_flex: jnp.ndarray
        teacher_active: jnp.ndarray
        teacher_step_required_soft: jnp.ndarray
        teacher_step_required_hard: jnp.ndarray
        teacher_swing_foot: jnp.ndarray
        teacher_target_step_x: jnp.ndarray
        teacher_target_step_y: jnp.ndarray
        teacher_target_reachable: jnp.ndarray
        mpc_planner_active: jnp.ndarray
        mpc_controller_active: jnp.ndarray
        mpc_target_com_x: jnp.ndarray
        mpc_target_com_y: jnp.ndarray
        mpc_target_step_x: jnp.ndarray
        mpc_target_step_y: jnp.ndarray
        mpc_support_state: jnp.ndarray
        mpc_step_requested: jnp.ndarray
        domain_rand_friction_scale: jnp.ndarray
        domain_rand_mass_scales: jnp.ndarray
        domain_rand_kp_scales: jnp.ndarray
        domain_rand_frictionloss_scales: jnp.ndarray
        domain_rand_joint_offsets: jnp.ndarray
        push_schedule: DisturbanceSchedule


# =============================================================================
# Validation utilities
# =============================================================================

def get_expected_shapes(action_size: int = None) -> dict:
    """Get expected shapes for WildRobotInfo fields.

    Args:
        action_size: Action dimension. If None, reads from robot_config.

    Returns:
        Dict mapping field names to expected shapes.
    """
    if action_size is None:
        from training.configs.training_config import get_robot_config
        action_size = get_robot_config().action_dim

    return {
        "step_count": (),
        "prev_action": (action_size,),
        "pending_action": (action_size,),
        "truncated": (),
        "velocity_cmd": (),
        "prev_root_pos": (3,),
        "prev_root_quat": (4,),
        "prev_left_foot_pos": (3,),
        "prev_right_foot_pos": (3,),
        "imu_quat_hist": (IMU_HIST_LEN, 4),
        "imu_gyro_hist": (IMU_HIST_LEN, 3),
        "foot_contacts": (4,),
        "root_height": (),
        "prev_left_loaded": (),
        "prev_right_loaded": (),
        "cycle_start_forward_x": (),
        "last_touchdown_root_pos": (3,),
        "last_touchdown_foot": (),
        "critic_obs": (PRIVILEGED_OBS_DIM,),
        "loc_ref_phase_time": (),
        "loc_ref_stance_foot": (),
        "loc_ref_switch_count": (),
        "loc_ref_mode_id": (),
        "loc_ref_mode_time": (),
        "loc_ref_startup_route_progress": (),
        "loc_ref_startup_route_ceiling": (),
        "loc_ref_startup_route_stage_id": (),
        "loc_ref_startup_route_transition_reason": (),
        "loc_ref_gait_phase_sin": (),
        "loc_ref_gait_phase_cos": (),
        "loc_ref_next_foothold": (2,),
        "loc_ref_swing_pos": (3,),
        "loc_ref_swing_vel": (3,),
        "loc_ref_pelvis_height": (),
        "loc_ref_pelvis_roll": (),
        "loc_ref_pelvis_pitch": (),
        "loc_ref_history": (4,),
        "nominal_q_ref": (action_size,),
        # M3 FSM state
        "fsm_phase": (),
        "fsm_swing_foot": (),
        "fsm_phase_ticks": (),
        "fsm_frozen_tx": (),
        "fsm_frozen_ty": (),
        "fsm_swing_sx": (),
        "fsm_swing_sy": (),
        "fsm_touch_hold": (),
        "fsm_trigger_hold": (),
        # v0.17.1+: Recovery metrics state
        "recovery_active": (),
        "recovery_age": (),
        "recovery_last_support_foot": (),
        "recovery_first_touchdown_recorded": (),
        "recovery_first_step_latency": (),
        "recovery_touchdown_count": (),
        "recovery_support_foot_changes": (),
        "recovery_first_liftoff_recorded": (),
        "recovery_first_liftoff_latency": (),
        "recovery_first_touchdown_age": (),
        "recovery_visible_step_recorded": (),
        "recovery_pitch_rate_at_push_end": (),
        "recovery_pitch_rate_at_touchdown": (),
        "recovery_pitch_rate_after_10t": (),
        "recovery_capture_error_at_push_end": (),
        "recovery_capture_error_at_touchdown": (),
        "recovery_capture_error_after_10t": (),
        "recovery_first_step_dx": (),
        "recovery_first_step_dy": (),
        "recovery_first_step_target_err_x": (),
        "recovery_first_step_target_err_y": (),
        "recovery_min_height": (),
        "recovery_max_knee_flex": (),
        "teacher_active": (),
        "teacher_step_required_soft": (),
        "teacher_step_required_hard": (),
        "teacher_swing_foot": (),
        "teacher_target_step_x": (),
        "teacher_target_step_y": (),
        "teacher_target_reachable": (),
        "mpc_planner_active": (),
        "mpc_controller_active": (),
        "mpc_target_com_x": (),
        "mpc_target_com_y": (),
        "mpc_target_step_x": (),
        "mpc_target_step_y": (),
        "mpc_support_state": (),
        "mpc_step_requested": (),
        "domain_rand_friction_scale": (),
        "domain_rand_mass_scales": None,
        "domain_rand_kp_scales": (action_size,),
        "domain_rand_frictionloss_scales": (action_size,),
        "domain_rand_joint_offsets": (action_size,),
        "push_schedule": {
            "start_step": (),
            "end_step": (),
            "force_xy": (2,),
            "body_id": (),
            "rng": (2,),
        },
    }


# Default shapes (for documentation, actual validation uses get_expected_shapes)
WILDROBOT_INFO_SHAPES = get_expected_shapes.__doc__  # Placeholder, use get_expected_shapes()

# Fields that must never be all zeros
WILDROBOT_INFO_NONZERO_FIELDS = {
    "prev_root_pos",  # Robot always has position
    "prev_root_quat",  # Quaternion is never all zeros
    "prev_left_foot_pos",  # Foot always has position
    "prev_right_foot_pos",  # Foot always has position
    "root_height",  # Robot always has height > 0
}


def validate_wildrobot_info(
    wr_info: WildRobotInfo,
    context: str = "",
    action_size: int = None,
) -> list[str]:
    """Validate WildRobotInfo has correct shapes and non-zero required fields.

    Args:
        wr_info: The WildRobotInfo to validate
        context: Context string for error messages (e.g., "reset", "step")
        action_size: Expected action size. If None, reads from robot_config.

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    # Get expected shapes (parameterized by action_size)
    expected_shapes = get_expected_shapes(action_size)

    for field_name, expected_shape in expected_shapes.items():
        value = getattr(wr_info, field_name, None)

        if value is None:
            errors.append(f"[{context}] Missing required field: '{field_name}'")
            continue

        if field_name == "push_schedule":
            if not isinstance(value, DisturbanceSchedule):
                errors.append(
                    f"[{context}] Field '{field_name}': expected DisturbanceSchedule, "
                    f"got {type(value)}"
                )
                continue
            for sub_name, sub_shape in expected_shape.items():
                sub_value = getattr(value, sub_name, None)
                if sub_value is None:
                    errors.append(
                        f"[{context}] Field '{field_name}.{sub_name}' is missing"
                    )
                    continue
                if sub_value.shape != sub_shape:
                    errors.append(
                        f"[{context}] Field '{field_name}.{sub_name}': expected "
                        f"shape {sub_shape}, got {sub_value.shape}"
                    )
        else:
            if expected_shape is not None and value.shape != expected_shape:
                errors.append(
                    f"[{context}] Field '{field_name}': expected shape {expected_shape}, "
                    f"got {value.shape}"
                )

        if field_name in WILDROBOT_INFO_NONZERO_FIELDS:
            if jnp.allclose(value, 0.0):
                errors.append(
                    f"[{context}] Field '{field_name}' cannot be all zeros (got {value})"
                )

    return errors


def assert_wildrobot_info_valid(
    wr_info: WildRobotInfo,
    context: str = "",
    action_size: int = None,
):
    """Assert WildRobotInfo is valid, raising detailed error if not."""
    errors = validate_wildrobot_info(wr_info, context, action_size)
    if errors:
        error_msg = "\n".join(errors)
        raise ValueError(f"WildRobotInfo validation failed:\n{error_msg}")


def get_wr_info(info: dict, context: str = "") -> WildRobotInfo:
    """Extract WildRobotInfo from info dict, failing loudly if missing.

    Use this instead of info.get("wr", ...) to ensure hard failure on missing.

    Args:
        info: The info dict from env state
        context: Context for error message

    Returns:
        WildRobotInfo pytree

    Raises:
        KeyError: If "wr" key is missing
    """
    if "wr" not in info:
        raise KeyError(
            f"[{context}] Required 'wr' namespace missing from info dict. "
            "This indicates env state corruption or wrapper misconfiguration."
        )
    return info["wr"]


# =============================================================================
# Reserved key for namespace
# =============================================================================

# The key used to store WildRobotInfo in the info dict
# Using a short key to avoid collision with wrapper fields
WR_INFO_KEY = "wr"
