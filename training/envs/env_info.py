"""Typed environment state structures for WildRobot (v0.20.1 v3-only).

Pruned from the v0.19.5c schema along with the FSM/MPC/teacher/recovery
subsystems removed by the v3 env rewrite (commits ``3a27a57`` /
``ab261a7``).  See ``training/docs/v0201_env_wiring.md`` §10 step 4
for the cleanup contract.

Kept fields are exactly what the v3 env reset/step actually writes
or reads (see the audit doc §B.2).  Wrapper-injected fields
(episode_metrics, truncation, etc.) stay at top level on
``state.info``; this struct holds only algorithm-critical
WildRobot-specific state.

Removed at v0.20.1 step 4 (~50 fields):
  - ``fsm_*`` (M3 step FSM)
  - ``mpc_*`` (v0.17.3 standing pivot)
  - ``teacher_*`` (M3 teacher targets)
  - ``recovery_*`` (push-recovery diagnostics — ~23 fields)
  - ``loc_ref_phase_time``, ``loc_ref_switch_count``,
    ``loc_ref_mode_id``, ``loc_ref_mode_time`` (v2 hybrid mode)
  - ``loc_ref_startup_route_*`` (v2 startup ramp)
  - ``loc_ref_com_x0_at_stance_start``,
    ``loc_ref_com_vx0_at_stance_start`` (v2 LIPM startup)
  - ``loc_ref_stance_margin_smooth`` (v2 support gate)
  - ``cycle_start_forward_x``, ``last_touchdown_root_pos``,
    ``last_touchdown_foot``, ``prev_touchdown_foot_x``
    (v0.19 step-event tracking)
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from training.envs.disturbance import DisturbanceSchedule


# Fixed maximum latency for IMU history buffers (must match
# ``EnvConfig.imu_max_latency_steps``).
IMU_MAX_LATENCY = 4
IMU_HIST_LEN = IMU_MAX_LATENCY + 1
PRIVILEGED_OBS_DIM = 12


# Use flax.struct for JAX-compatible dataclass with pytree registration;
# fall back to NamedTuple if flax is unavailable.
try:
    import flax.struct as struct

    @struct.dataclass
    class WildRobotInfo:
        """Algorithm-critical per-env state for WildRobot v3.

        All fields use post-step convention: they reflect state AFTER
        the physics step, to be used as 'previous' values in the next
        step (or the seed for the next reset).
        """
        # Core bookkeeping
        step_count: jnp.ndarray            # ()
        prev_action: jnp.ndarray           # (action_size,) — action applied this step
        pending_action: jnp.ndarray        # (action_size,) — next filtered action
        truncated: jnp.ndarray             # () — sticky through auto-reset
        velocity_cmd: jnp.ndarray          # () — episode-constant target velocity (m/s)

        # Finite-diff velocity (post-step values used by AMP / privileged critic)
        prev_root_pos: jnp.ndarray         # (3,)
        prev_root_quat: jnp.ndarray        # (4,) wxyz
        prev_left_foot_pos: jnp.ndarray    # (3,)
        prev_right_foot_pos: jnp.ndarray   # (3,)

        # IMU history buffers (most-recent at index 0; rolled each step
        # by ``_apply_imu_noise_and_delay``).
        imu_quat_hist: jnp.ndarray         # (IMU_HIST_LEN, 4)
        imu_gyro_hist: jnp.ndarray         # (IMU_HIST_LEN, 3)

        # Aggregated foot state
        foot_contacts: jnp.ndarray         # (4,) [L_toe, L_heel, R_toe, R_heel]
        root_height: jnp.ndarray           # ()
        prev_left_loaded: jnp.ndarray      # () — last-step contact gate (>= force thresh)
        prev_right_loaded: jnp.ndarray     # ()

        # Privileged critic obs (sim-only fields the actor doesn't see)
        critic_obs: jnp.ndarray            # (PRIVILEGED_OBS_DIM,)

        # v3 offline ReferenceLibrary playback state
        loc_ref_offline_step_idx: jnp.ndarray   # () int32 — increments by 1 per step
        loc_ref_offline_command_id: jnp.ndarray # () int32 — single-command smoke = 0

        # v4-compatibility / v5 obs feed (populated from the offline window)
        loc_ref_stance_foot: jnp.ndarray   # () int32 — 0 left, 1 right
        loc_ref_gait_phase_sin: jnp.ndarray  # ()
        loc_ref_gait_phase_cos: jnp.ndarray  # ()
        loc_ref_next_foothold: jnp.ndarray   # (2,) heading-frame x, y
        loc_ref_swing_pos: jnp.ndarray       # (3,)
        loc_ref_swing_vel: jnp.ndarray       # (3,)
        loc_ref_pelvis_height: jnp.ndarray   # ()
        loc_ref_pelvis_roll: jnp.ndarray     # ()
        loc_ref_pelvis_pitch: jnp.ndarray    # ()
        loc_ref_history: jnp.ndarray         # (4,) — rolled phase_sin_cos buffer
        nominal_q_ref: jnp.ndarray           # (action_size,) — current frame's q_ref

        # Episode-constant disturbance schedule
        push_schedule: DisturbanceSchedule

        # Episode-constant domain-randomization params (applied per step
        # via ``_get_randomized_mjx_model``).
        domain_rand_friction_scale: jnp.ndarray       # ()
        domain_rand_mass_scales: jnp.ndarray          # (n_bodies,)
        domain_rand_kp_scales: jnp.ndarray            # (action_size,)
        domain_rand_frictionloss_scales: jnp.ndarray  # (action_size,)
        domain_rand_joint_offsets: jnp.ndarray        # (action_size,)

except ImportError:
    from typing import NamedTuple

    class WildRobotInfo(NamedTuple):
        """Fallback NamedTuple — fields mirror the flax.struct version."""
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
        critic_obs: jnp.ndarray
        loc_ref_offline_step_idx: jnp.ndarray
        loc_ref_offline_command_id: jnp.ndarray
        loc_ref_stance_foot: jnp.ndarray
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
        push_schedule: DisturbanceSchedule
        domain_rand_friction_scale: jnp.ndarray
        domain_rand_mass_scales: jnp.ndarray
        domain_rand_kp_scales: jnp.ndarray
        domain_rand_frictionloss_scales: jnp.ndarray
        domain_rand_joint_offsets: jnp.ndarray


# =============================================================================
# Validation utilities
# =============================================================================


def get_expected_shapes(action_size: int = None) -> dict:
    """Get expected shapes for ``WildRobotInfo`` fields.

    Args:
        action_size: Action dimension.  If None, reads from ``robot_config``.
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
        "critic_obs": (PRIVILEGED_OBS_DIM,),
        "loc_ref_offline_step_idx": (),
        "loc_ref_offline_command_id": (),
        "loc_ref_stance_foot": (),
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
        "domain_rand_friction_scale": (),
        # mass_scales shape depends on n_bodies (model-dependent); skip strict check
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


# Fields that must never be all zeros (sanity invariants for tests).
WILDROBOT_INFO_NONZERO_FIELDS = {
    "prev_root_pos",        # robot always has position
    "prev_root_quat",       # quaternion is never all zeros
    "prev_left_foot_pos",   # foot always has position
    "prev_right_foot_pos",  # foot always has position
    "root_height",          # robot always has height > 0
}


def validate_wildrobot_info(
    wr_info: WildRobotInfo,
    context: str = "",
    action_size: int = None,
) -> list[str]:
    """Validate ``WildRobotInfo`` shapes and non-zero invariants."""
    errors = []
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
    """Assert ``WildRobotInfo`` is valid, raising a detailed error if not."""
    errors = validate_wildrobot_info(wr_info, context, action_size)
    if errors:
        raise ValueError("WildRobotInfo validation failed:\n" + "\n".join(errors))


def get_wr_info(info: dict, context: str = "") -> WildRobotInfo:
    """Extract ``WildRobotInfo`` from the env info dict, failing loudly if missing.

    Use this instead of ``info.get('wr', ...)`` to surface env-state corruption
    or wrapper misconfiguration as a hard error.
    """
    if "wr" not in info:
        raise KeyError(
            f"[{context}] Required 'wr' namespace missing from info dict. "
            "This indicates env state corruption or wrapper misconfiguration."
        )
    return info["wr"]


# Reserved key for the WildRobotInfo namespace inside ``state.info``.
WR_INFO_KEY = "wr"
