"""WildRobot environment extending mujoco_playground's MjxEnv.

This module provides a fully JAX-compatible WildRobot environment that:
- Extends mjx_env.MjxEnv for native Brax compatibility
- Is fully JIT-able with jax.jit, jax.vmap
- Uses pure JAX arrays (no numpy conversions)
- Supports mujoco_playground.wrapper.wrap_for_brax_training()

Architecture:
    WildRobotEnv(mjx_env.MjxEnv)
        └── reset(rng) -> WildRobotEnvState
        └── step(state, action) -> WildRobotEnvState
        └── Robot utilities (joint access, sensors)

Usage:
    from training.envs.wildrobot_env import WildRobotEnv
    from training.configs.training_config import load_training_config
    from mujoco_playground import wrapper
    from brax.training.agents.ppo import train as ppo

    training_cfg = load_training_config("configs/ppo_walking.yaml")
    runtime_cfg = training_cfg.to_runtime_config()
    env = WildRobotEnv(config=runtime_cfg)

    # Train with Brax PPO
    make_inference_fn, params, _ = ppo.train(
        environment=env,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        ...
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import jax
import jax.numpy as jp
import mujoco
from flax import struct
from ml_collections import config_dict
from mujoco import mjx

from mujoco_playground._src import mjx_env
from policy_contract.jax import frames as jax_frames
from policy_contract.jax.obs import build_observation
from policy_contract.jax.action import postprocess_action
from policy_contract.calib import JaxCalibOps
from policy_contract.jax.state import PolicyState
from policy_contract.layout import get_slices as get_obs_slices
from training.sim_adapter.mjx_signals import MjxSignalsAdapter
from policy_contract.spec import PolicySpec
from training.cal.cal import ControlAbstractionLayer
from training.cal.specs import Pose3D
from training.cal.types import CoordinateFrame

from training.configs.training_config import get_robot_config, TrainingConfig
from training.envs import step_controller as sc
from control.mpc.standing import compute_mpc_standing_action
from control.kinematics.leg_ik import LegIkConfig, solve_leg_sagittal_ik_jax
from training.envs.env_info import (
    IMU_HIST_LEN,
    IMU_MAX_LATENCY,
    PRIVILEGED_OBS_DIM,
    WildRobotInfo,
    WR_INFO_KEY,
)
from control.references.walking_ref_v1 import (
    WalkingRefV1Config,
    LEFT_STANCE as REF_LEFT_STANCE,
    RIGHT_STANCE as REF_RIGHT_STANCE,
)
from control.references.walking_ref_v2 import (
    WalkingRefV2Config,
    WalkingRefV2Mode,
    step_reference_v2_jax,
)
from training.envs.disturbance import (
    DisturbanceSchedule,
    apply_push,
    sample_push_schedule,
)
from training.envs.domain_randomize import (
    nominal_domain_rand_params,
    sample_domain_rand_params,
)
from training.envs.teacher_step_target import (
    compute_teacher_step_target,
    compute_teacher_target_step_xy_reward,
    compute_teacher_step_required_reward,
    compute_teacher_swing_foot_reward,
)
from training.envs.teacher_whole_body_target import (
    compute_teacher_whole_body_target,
    compute_teacher_recovery_height_reward,
    compute_teacher_com_velocity_reduction_reward,
)
from training.core.experiment_tracking import get_initial_env_metrics_jax
from training.core.metrics_registry import (
    build_metrics_vec,
    METRIC_NAMES,
    METRICS_VEC_KEY,
    NUM_METRICS,
)
from policy_contract.spec_builder import build_policy_spec
from training.policy_spec_utils import clamp_home_ctrl, get_home_ctrl_from_mj_model


# =============================================================================
# Helper utilities
# =============================================================================

RECOVERY_WINDOW_STEPS = 50
STANCE_REF_HISTORY_LEN = 2


def _clip_scalar(x: jax.Array, lo: float, hi: float) -> jax.Array:
    return jp.clip(jp.asarray(x, dtype=jp.float32), jp.asarray(lo, dtype=jp.float32), jp.asarray(hi, dtype=jp.float32))


def _step_walking_reference_jax(
    *,
    phase_time_s: jax.Array,
    stance_foot_id: jax.Array,
    stance_switch_count: jax.Array,
    forward_speed_mps: jax.Array,
    com_position_stance_frame: jax.Array,
    com_velocity_stance_frame: jax.Array,
    dt_s: float,
    cfg: WalkingRefV1Config,
    dcm_placement_gain: float = 1.0,
    phase_scale: jax.Array = jp.asarray(1.0, dtype=jp.float32),
    speed_scale: jax.Array = jp.asarray(1.0, dtype=jp.float32),
    support_health: jax.Array = jp.asarray(1.0, dtype=jp.float32),
    support_release_phase_start: jax.Array = jp.asarray(0.35, dtype=jp.float32),
    support_foothold_min_scale: jax.Array = jp.asarray(0.35, dtype=jp.float32),
    support_swing_progress_min_scale: jax.Array = jp.asarray(0.0, dtype=jp.float32),
    support_phase_min_scale: jax.Array = jp.asarray(0.15, dtype=jp.float32),
) -> tuple[dict[str, jax.Array], jax.Array, jax.Array, jax.Array]:
    support_h = jp.clip(jp.asarray(support_health, dtype=jp.float32), 0.0, 1.0)
    phase_support_scale = jp.asarray(support_phase_min_scale, dtype=jp.float32) + (
        1.0 - jp.asarray(support_phase_min_scale, dtype=jp.float32)
    ) * support_h
    dt = jp.asarray(dt_s, dtype=jp.float32) * jp.clip(
        jp.asarray(phase_scale, dtype=jp.float32) * phase_support_scale, 0.02, 1.0
    )
    step_time = jp.asarray(cfg.step_time_s, dtype=jp.float32)
    phase_time = jp.asarray(phase_time_s, dtype=jp.float32) + dt
    stance = jp.asarray(stance_foot_id, dtype=jp.int32)
    switches = jp.asarray(stance_switch_count, dtype=jp.int32)
    switched = phase_time >= step_time
    phase_time = jp.where(switched, phase_time - step_time, phase_time)
    stance = jp.where(switched, 1 - stance, stance)
    switches = jp.where(switched, switches + 1, switches)
    phase = jp.clip(phase_time / jp.maximum(step_time, 1e-6), 0.0, 1.0)

    speed = _clip_scalar(
        forward_speed_mps * jp.clip(jp.asarray(speed_scale, dtype=jp.float32), 0.0, 1.0),
        0.0,
        cfg.max_forward_speed_mps,
    )
    omega = jp.sqrt(jp.asarray(9.81, dtype=jp.float32) / jp.asarray(cfg.nominal_com_height_m, dtype=jp.float32))
    dcm_vel_scale = jp.clip(jp.asarray(speed_scale, dtype=jp.float32), 0.0, 1.0)
    cp_x = com_position_stance_frame[0] + (
        com_velocity_stance_frame[0] * dcm_vel_scale
    ) / jp.maximum(omega, 1e-6)
    b = jp.exp(-omega * step_time)
    nominal_step = _clip_scalar(
        speed * step_time,
        cfg.min_step_length_m,
        cfg.max_step_length_m,
    )
    x_foot_dcm = (nominal_step - b * cp_x) / jp.maximum(1.0 - b, 1e-6)
    dcm_gain = jp.clip(jp.asarray(dcm_placement_gain, dtype=jp.float32), 0.0, 1.0)
    x_foot = (1.0 - dcm_gain) * nominal_step + dcm_gain * x_foot_dcm
    foothold_scale = jp.asarray(support_foothold_min_scale, dtype=jp.float32) + (
        1.0 - jp.asarray(support_foothold_min_scale, dtype=jp.float32)
    ) * support_h
    x_foot = cfg.min_step_length_m + foothold_scale * (x_foot - cfg.min_step_length_m)
    x_foot = _clip_scalar(x_foot, cfg.min_step_length_m, cfg.max_step_length_m)
    y_sign = jp.where(stance == REF_LEFT_STANCE, -1.0, 1.0)
    y_foot = _clip_scalar(
        y_sign * jp.asarray(cfg.nominal_lateral_foot_offset_m, dtype=jp.float32),
        -cfg.max_lateral_step_m,
        cfg.max_lateral_step_m,
    )
    release_start = jp.clip(
        jp.asarray(support_release_phase_start, dtype=jp.float32), 0.0, 0.95
    )
    phase_support = jp.clip(phase - release_start, 0.0, 1.0)
    swing_progress_scale = jp.asarray(
        support_swing_progress_min_scale, dtype=jp.float32
    ) + (1.0 - jp.asarray(support_swing_progress_min_scale, dtype=jp.float32)) * support_h
    swing_x_progress = (
        swing_progress_scale * phase + (1.0 - swing_progress_scale) * phase_support
    )
    swing_x_progress_dphase = swing_progress_scale + (
        1.0 - swing_progress_scale
    ) * (phase > release_start).astype(jp.float32)
    z_swing = jp.asarray(cfg.swing_height_m, dtype=jp.float32) * jp.sin(jp.pi * phase)
    swing_pos = jp.asarray([x_foot * swing_x_progress, y_foot * phase, z_swing], dtype=jp.float32)
    swing_vel = jp.asarray(
        [
            x_foot * swing_x_progress_dphase / jp.maximum(step_time, 1e-6),
            y_foot / jp.maximum(step_time, 1e-6),
            (jp.asarray(cfg.swing_height_m, dtype=jp.float32) * jp.pi / jp.maximum(step_time, 1e-6)) * jp.cos(jp.pi * phase),
        ],
        dtype=jp.float32,
    )
    pelvis_roll = jp.where(
        stance == REF_LEFT_STANCE,
        -jp.asarray(cfg.pelvis_roll_bias_rad, dtype=jp.float32),
        jp.asarray(cfg.pelvis_roll_bias_rad, dtype=jp.float32),
    )
    pelvis_pitch = _clip_scalar(
        jp.asarray(cfg.pelvis_pitch_gain, dtype=jp.float32) * speed,
        -cfg.max_pelvis_pitch_rad,
        cfg.max_pelvis_pitch_rad,
    )
    ref = {
        "gait_phase_sin": jp.sin(2.0 * jp.pi * phase).astype(jp.float32),
        "gait_phase_cos": jp.cos(2.0 * jp.pi * phase).astype(jp.float32),
        "phase_progress": phase.astype(jp.float32),
        "stance_foot_id": stance.astype(jp.int32),
        "next_foothold": jp.asarray([x_foot, y_foot], dtype=jp.float32),
        "swing_pos": swing_pos,
        "swing_vel": swing_vel,
        "pelvis_height": jp.asarray(cfg.nominal_com_height_m, dtype=jp.float32),
        "pelvis_roll": pelvis_roll.astype(jp.float32),
        "pelvis_pitch": pelvis_pitch.astype(jp.float32),
    }
    return ref, phase_time.astype(jp.float32), stance.astype(jp.int32), switches.astype(jp.int32)


def _update_loc_ref_history(prev_hist: jax.Array, phase_sin: jax.Array, phase_cos: jax.Array) -> jax.Array:
    prev = jp.asarray(prev_hist, dtype=jp.float32).reshape(4)
    return jp.asarray([prev[2], prev[3], phase_sin, phase_cos], dtype=jp.float32)


def _loc_ref_brake_scales(
    *,
    velocity_cmd: jax.Array,
    forward_vel_h: jax.Array,
    pitch_rad: jax.Array,
    overspeed_deadband: jax.Array,
    overspeed_brake_gain: jax.Array,
    overspeed_phase_slowdown_gain: jax.Array,
    overspeed_phase_min_scale: jax.Array,
    pitch_brake_start_rad: jax.Array,
    pitch_brake_gain: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    cmd = jp.maximum(jp.asarray(velocity_cmd, dtype=jp.float32), 0.0)
    fwd = jp.asarray(forward_vel_h, dtype=jp.float32)
    pitch = jp.asarray(pitch_rad, dtype=jp.float32)
    overspeed = jp.maximum(fwd - cmd - jp.asarray(overspeed_deadband, dtype=jp.float32), 0.0)
    pitch_excess = jp.maximum(jp.abs(pitch) - jp.asarray(pitch_brake_start_rad, dtype=jp.float32), 0.0)

    speed_scale = 1.0 / (
        1.0
        + jp.asarray(overspeed_brake_gain, dtype=jp.float32) * overspeed
        + jp.asarray(pitch_brake_gain, dtype=jp.float32) * pitch_excess
    )
    phase_scale = 1.0 / (
        1.0
        + jp.asarray(overspeed_phase_slowdown_gain, dtype=jp.float32) * overspeed
        + 0.5 * jp.asarray(pitch_brake_gain, dtype=jp.float32) * pitch_excess
    )
    phase_scale = jp.maximum(
        jp.asarray(overspeed_phase_min_scale, dtype=jp.float32), phase_scale
    )
    return speed_scale.astype(jp.float32), phase_scale.astype(jp.float32), overspeed.astype(jp.float32)


def _loc_ref_support_scales(
    *,
    velocity_cmd: jax.Array,
    forward_vel_h: jax.Array,
    pitch_rad: jax.Array,
    swing_x_brake_pitch_start_rad: jax.Array,
    swing_x_brake_overspeed_deadband: jax.Array,
    swing_x_brake_gain: jax.Array,
    swing_x_min_scale: jax.Array,
    pelvis_pitch_brake_gain: jax.Array,
    pelvis_pitch_min_scale: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Compute support-first nominal scaling for swing-x and pelvis-pitch channels."""
    cmd = jp.maximum(jp.asarray(velocity_cmd, dtype=jp.float32), 0.0)
    fwd = jp.asarray(forward_vel_h, dtype=jp.float32)
    pitch = jp.asarray(pitch_rad, dtype=jp.float32)
    overspeed = jp.maximum(
        fwd - cmd - jp.asarray(swing_x_brake_overspeed_deadband, dtype=jp.float32), 0.0
    )
    pitch_excess = jp.maximum(
        jp.abs(pitch) - jp.asarray(swing_x_brake_pitch_start_rad, dtype=jp.float32), 0.0
    )
    instability = overspeed + pitch_excess

    swing_x_scale = 1.0 / (
        1.0 + jp.asarray(swing_x_brake_gain, dtype=jp.float32) * instability
    )
    swing_x_scale = jp.maximum(
        jp.asarray(swing_x_min_scale, dtype=jp.float32), swing_x_scale
    )

    pelvis_pitch_scale = 1.0 / (
        1.0 + jp.asarray(pelvis_pitch_brake_gain, dtype=jp.float32) * instability
    )
    pelvis_pitch_scale = jp.maximum(
        jp.asarray(pelvis_pitch_min_scale, dtype=jp.float32), pelvis_pitch_scale
    )

    gate_active = (instability > 1e-6).astype(jp.float32)
    return (
        swing_x_scale.astype(jp.float32),
        pelvis_pitch_scale.astype(jp.float32),
        overspeed.astype(jp.float32),
        gate_active,
    )


def _loc_ref_support_health(
    *,
    velocity_cmd: jax.Array,
    forward_vel_h: jax.Array,
    pitch_rad: jax.Array,
    pitch_rate_rad_s: jax.Array,
    overspeed_deadband: jax.Array,
    pitch_start_rad: jax.Array,
    pitch_rate_start_rad_s: jax.Array,
    health_gain: jax.Array,
    left_foot_loaded: jax.Array | None = None,
    right_foot_loaded: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Compute support health helper shared with v1-plumbing and v2 diagnostics.

    For v2 control progression, `step_reference_v2_jax` remains the authoritative owner.
    This helper is kept for legacy/v1 plumbing and lightweight parity checks.
    """
    cmd = jp.maximum(jp.asarray(velocity_cmd, dtype=jp.float32), 0.0)
    fwd = jp.asarray(forward_vel_h, dtype=jp.float32)
    pitch = jp.asarray(pitch_rad, dtype=jp.float32)
    pitch_rate = jp.asarray(pitch_rate_rad_s, dtype=jp.float32)
    overspeed = jp.maximum(fwd - cmd - jp.asarray(overspeed_deadband, dtype=jp.float32), 0.0)
    pitch_excess = jp.maximum(jp.abs(pitch) - jp.asarray(pitch_start_rad, dtype=jp.float32), 0.0)
    pitch_rate_excess = jp.maximum(
        jp.abs(pitch_rate) - jp.asarray(pitch_rate_start_rad_s, dtype=jp.float32), 0.0
    )
    if left_foot_loaded is None or right_foot_loaded is None:
        contact_penalty = jp.asarray(0.0, dtype=jp.float32)
    else:
        single_support = jp.logical_xor(
            jp.asarray(left_foot_loaded, dtype=jp.bool_),
            jp.asarray(right_foot_loaded, dtype=jp.bool_),
        )
        contact_penalty = jp.where(single_support, 0.0, 0.25).astype(jp.float32)
    instability = overspeed + pitch_excess + 0.5 * pitch_rate_excess + contact_penalty
    health = 1.0 / (1.0 + jp.asarray(health_gain, dtype=jp.float32) * instability)
    return health.astype(jp.float32), instability.astype(jp.float32)


def _swing_state_in_stance_frame(
    *,
    left_foot_pos_h: jax.Array,
    right_foot_pos_h: jax.Array,
    left_foot_vel_h: jax.Array,
    right_foot_vel_h: jax.Array,
    stance_foot_id: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    stance_is_left = jp.asarray(stance_foot_id, dtype=jp.int32) == REF_LEFT_STANCE
    stance_pos = jp.where(stance_is_left, left_foot_pos_h, right_foot_pos_h)
    swing_pos_h = jp.where(stance_is_left, right_foot_pos_h, left_foot_pos_h)
    stance_vel = jp.where(stance_is_left, left_foot_vel_h, right_foot_vel_h)
    swing_vel_h = jp.where(stance_is_left, right_foot_vel_h, left_foot_vel_h)
    return swing_pos_h - stance_pos, swing_vel_h - stance_vel


def _update_recovery_tracking(
    *,
    track_recovery: jax.Array,
    push_schedule: DisturbanceSchedule,
    current_step: jax.Array,
    episode_done: jax.Array,
    episode_failed: jax.Array,
    liftoff_left: jax.Array,
    liftoff_right: jax.Array,
    touchdown_left: jax.Array,
    touchdown_right: jax.Array,
    horizontal_speed: jax.Array,
    pitch_rate: jax.Array,
    capture_error_norm: jax.Array,
    first_touchdown_dx: jax.Array,
    first_touchdown_dy: jax.Array,
    first_touchdown_target_err_x: jax.Array,
    first_touchdown_target_err_y: jax.Array,
    current_height: jax.Array,
    current_knee_flex: jax.Array,
    recovery_active: jax.Array,
    recovery_age: jax.Array,
    recovery_last_support_foot: jax.Array,
    recovery_first_liftoff_recorded: jax.Array,
    recovery_first_liftoff_latency: jax.Array,
    recovery_first_touchdown_recorded: jax.Array,
    recovery_first_step_latency: jax.Array,
    recovery_first_touchdown_age: jax.Array,
    recovery_visible_step_recorded: jax.Array,
    recovery_touchdown_count: jax.Array,
    recovery_support_foot_changes: jax.Array,
    recovery_pitch_rate_at_push_end: jax.Array,
    recovery_pitch_rate_at_touchdown: jax.Array,
    recovery_pitch_rate_after_10t: jax.Array,
    recovery_capture_error_at_push_end: jax.Array,
    recovery_capture_error_at_touchdown: jax.Array,
    recovery_capture_error_after_10t: jax.Array,
    recovery_first_step_dx: jax.Array,
    recovery_first_step_dy: jax.Array,
    recovery_first_step_target_err_x: jax.Array,
    recovery_first_step_target_err_y: jax.Array,
    recovery_min_height: jax.Array,
    recovery_max_knee_flex: jax.Array,
) -> tuple[Dict[str, jax.Array], Dict[str, jax.Array]]:
    """Update recovery state and emit finalized summary metrics once per recovery."""
    del recovery_active, recovery_age
    current_step = jp.asarray(current_step, dtype=jp.int32)

    recovery_start = track_recovery & (current_step == push_schedule.end_step)
    in_recovery = (
        track_recovery
        & (current_step >= push_schedule.end_step)
        & (current_step < push_schedule.end_step + RECOVERY_WINDOW_STEPS)
    )
    age = jp.where(
        in_recovery,
        current_step - push_schedule.end_step,
        jp.zeros((), dtype=jp.int32),
    )
    age = age.astype(jp.int32)

    last_support_foot = jp.where(
        recovery_start,
        jp.asarray(-1, dtype=jp.int32),
        recovery_last_support_foot,
    )
    first_touchdown_recorded = jp.where(
        recovery_start,
        jp.asarray(False, dtype=jp.bool_),
        recovery_first_touchdown_recorded,
    )
    first_step_latency = jp.where(
        recovery_start,
        jp.zeros((), dtype=jp.int32),
        recovery_first_step_latency,
    )
    touchdown_count = jp.where(
        recovery_start,
        jp.zeros((), dtype=jp.int32),
        recovery_touchdown_count,
    )
    support_foot_changes = jp.where(
        recovery_start,
        jp.zeros((), dtype=jp.int32),
        recovery_support_foot_changes,
    )
    first_liftoff_recorded = jp.where(
        recovery_start,
        jp.asarray(False, dtype=jp.bool_),
        recovery_first_liftoff_recorded,
    )
    first_liftoff_latency = jp.where(
        recovery_start,
        jp.zeros((), dtype=jp.int32),
        recovery_first_liftoff_latency,
    )
    first_touchdown_age = jp.where(
        recovery_start,
        jp.asarray(-1, dtype=jp.int32),
        recovery_first_touchdown_age,
    )
    visible_step_recorded = jp.where(
        recovery_start,
        jp.asarray(False, dtype=jp.bool_),
        recovery_visible_step_recorded,
    )
    pitch_rate_at_push_end = jp.where(
        recovery_start,
        jp.asarray(pitch_rate, dtype=jp.float32),
        recovery_pitch_rate_at_push_end,
    )
    pitch_rate_at_touchdown = jp.where(
        recovery_start,
        jp.zeros((), dtype=jp.float32),
        recovery_pitch_rate_at_touchdown,
    )
    pitch_rate_after_10t = jp.where(
        recovery_start,
        jp.zeros((), dtype=jp.float32),
        recovery_pitch_rate_after_10t,
    )
    capture_error_at_push_end = jp.where(
        recovery_start,
        jp.asarray(capture_error_norm, dtype=jp.float32),
        recovery_capture_error_at_push_end,
    )
    capture_error_at_touchdown = jp.where(
        recovery_start,
        jp.zeros((), dtype=jp.float32),
        recovery_capture_error_at_touchdown,
    )
    capture_error_after_10t = jp.where(
        recovery_start,
        jp.zeros((), dtype=jp.float32),
        recovery_capture_error_after_10t,
    )
    first_step_dx = jp.where(
        recovery_start,
        jp.zeros((), dtype=jp.float32),
        recovery_first_step_dx,
    )
    first_step_dy = jp.where(
        recovery_start,
        jp.zeros((), dtype=jp.float32),
        recovery_first_step_dy,
    )
    first_step_target_err_x = jp.where(
        recovery_start,
        jp.zeros((), dtype=jp.float32),
        recovery_first_step_target_err_x,
    )
    first_step_target_err_y = jp.where(
        recovery_start,
        jp.zeros((), dtype=jp.float32),
        recovery_first_step_target_err_y,
    )
    min_height = jp.where(
        recovery_start,
        jp.asarray(current_height, dtype=jp.float32),
        recovery_min_height,
    )
    max_knee_flex = jp.where(
        recovery_start,
        jp.asarray(current_knee_flex, dtype=jp.float32),
        recovery_max_knee_flex,
    )

    any_liftoff = liftoff_left | liftoff_right
    any_touchdown = touchdown_left | touchdown_right
    current_support_foot = jp.where(
        touchdown_left & ~touchdown_right,
        jp.asarray(0, dtype=jp.int32),
        jp.where(
            touchdown_right & ~touchdown_left,
            jp.asarray(1, dtype=jp.int32),
            jp.asarray(-1, dtype=jp.int32),
        ),
    )

    first_touchdown_now = in_recovery & any_touchdown & ~first_touchdown_recorded
    first_liftoff_now = in_recovery & any_liftoff & ~first_liftoff_recorded
    first_liftoff_latency = jp.where(first_liftoff_now, age, first_liftoff_latency).astype(
        jp.int32
    )
    liftoff_recorded_before_step = first_liftoff_recorded
    first_liftoff_recorded = first_liftoff_recorded | first_liftoff_now
    first_step_latency = jp.where(
        first_touchdown_now,
        age,
        first_step_latency,
    )
    first_step_latency = first_step_latency.astype(jp.int32)
    first_touchdown_recorded = first_touchdown_recorded | first_touchdown_now
    # Require touchdown after a previously recorded liftoff to avoid same-tick chatter.
    visible_step_now = in_recovery & liftoff_recorded_before_step & first_touchdown_now
    visible_step_recorded = visible_step_recorded | visible_step_now
    first_touchdown_age = jp.where(first_touchdown_now, age, first_touchdown_age).astype(
        jp.int32
    )
    pitch_rate_at_touchdown = jp.where(
        first_touchdown_now, pitch_rate.astype(jp.float32), pitch_rate_at_touchdown
    )
    pitch_rate_after_10t = jp.where(
        first_touchdown_now, pitch_rate.astype(jp.float32), pitch_rate_after_10t
    )
    capture_error_at_touchdown = jp.where(
        first_touchdown_now,
        capture_error_norm.astype(jp.float32),
        capture_error_at_touchdown,
    )
    capture_error_after_10t = jp.where(
        first_touchdown_now,
        capture_error_norm.astype(jp.float32),
        capture_error_after_10t,
    )
    first_step_dx = jp.where(
        first_touchdown_now, first_touchdown_dx.astype(jp.float32), first_step_dx
    )
    first_step_dy = jp.where(
        first_touchdown_now, first_touchdown_dy.astype(jp.float32), first_step_dy
    )
    first_step_target_err_x = jp.where(
        first_touchdown_now,
        first_touchdown_target_err_x.astype(jp.float32),
        first_step_target_err_x,
    )
    first_step_target_err_y = jp.where(
        first_touchdown_now,
        first_touchdown_target_err_y.astype(jp.float32),
        first_step_target_err_y,
    )

    reached_10t_after_touchdown = (
        in_recovery
        & first_touchdown_recorded
        & (first_touchdown_age >= 0)
        & (age == (first_touchdown_age + jp.asarray(10, dtype=jp.int32)))
    )
    pitch_rate_after_10t = jp.where(
        reached_10t_after_touchdown, pitch_rate.astype(jp.float32), pitch_rate_after_10t
    )
    capture_error_after_10t = jp.where(
        reached_10t_after_touchdown,
        capture_error_norm.astype(jp.float32),
        capture_error_after_10t,
    )
    min_height = jp.where(
        in_recovery,
        jp.minimum(min_height, jp.asarray(current_height, dtype=jp.float32)),
        min_height,
    )
    max_knee_flex = jp.where(
        in_recovery,
        jp.maximum(max_knee_flex, jp.asarray(current_knee_flex, dtype=jp.float32)),
        max_knee_flex,
    )

    touchdown_count = jp.where(
        in_recovery & any_touchdown,
        touchdown_count + jp.asarray(1, dtype=jp.int32),
        touchdown_count,
    )
    touchdown_count = touchdown_count.astype(jp.int32)

    support_changed = (
        in_recovery
        & (last_support_foot >= 0)
        & (current_support_foot >= 0)
        & (last_support_foot != current_support_foot)
    )
    support_foot_changes = jp.where(
        support_changed,
        support_foot_changes + jp.asarray(1, dtype=jp.int32),
        support_foot_changes,
    )
    support_foot_changes = support_foot_changes.astype(jp.int32)
    last_support_foot = jp.where(
        in_recovery & (current_support_foot >= 0),
        current_support_foot,
        last_support_foot,
    )

    finalize_recovery = in_recovery & (
        (age == (RECOVERY_WINDOW_STEPS - 1)) | episode_done
    )
    no_touchdown = touchdown_count == 0
    touchdown_then_fail = (touchdown_count > 0) & episode_failed
    touchdown_to_term_steps = jp.where(
        episode_failed & first_touchdown_recorded & (first_touchdown_age >= 0),
        jp.maximum(age - first_touchdown_age, 0).astype(jp.float32),
        jp.zeros((), dtype=jp.float32),
    )
    pitch_rate_reduction_10t = jp.where(
        first_touchdown_recorded,
        jp.abs(pitch_rate_at_touchdown) - jp.abs(pitch_rate_after_10t),
        jp.zeros((), dtype=jp.float32),
    )
    capture_error_reduction_10t = jp.where(
        first_touchdown_recorded,
        capture_error_at_touchdown - capture_error_after_10t,
        jp.zeros((), dtype=jp.float32),
    )
    first_step_dist_abs = jp.sqrt(
        jp.square(first_step_dx.astype(jp.float32))
        + jp.square(first_step_dy.astype(jp.float32))
    )
    summary_metrics = {
        "recovery/first_step_latency": jp.where(
            finalize_recovery,
            first_step_latency.astype(jp.float32),
            jp.zeros((), dtype=jp.float32),
        ),
        "recovery/first_liftoff_latency": jp.where(
            finalize_recovery,
            first_liftoff_latency.astype(jp.float32),
            jp.zeros((), dtype=jp.float32),
        ),
        "recovery/first_touchdown_latency": jp.where(
            finalize_recovery,
            first_step_latency.astype(jp.float32),
            jp.zeros((), dtype=jp.float32),
        ),
        "recovery/touchdown_count": jp.where(
            finalize_recovery,
            touchdown_count.astype(jp.float32),
            jp.zeros((), dtype=jp.float32),
        ),
        "recovery/support_foot_changes": jp.where(
            finalize_recovery,
            support_foot_changes.astype(jp.float32),
            jp.zeros((), dtype=jp.float32),
        ),
        "recovery/post_push_velocity": jp.where(
            finalize_recovery,
            horizontal_speed.astype(jp.float32),
            jp.zeros((), dtype=jp.float32),
        ),
        "recovery/completed": jp.where(
            finalize_recovery,
            jp.asarray(1.0, dtype=jp.float32),
            jp.zeros((), dtype=jp.float32),
        ),
        "recovery/no_touchdown_frac": jp.where(
            finalize_recovery,
            no_touchdown.astype(jp.float32),
            jp.zeros((), dtype=jp.float32),
        ),
        "recovery/touchdown_then_fail_frac": jp.where(
            finalize_recovery,
            touchdown_then_fail.astype(jp.float32),
            jp.zeros((), dtype=jp.float32),
        ),
        "recovery/first_step_dx": jp.where(
            finalize_recovery,
            first_step_dx.astype(jp.float32),
            jp.zeros((), dtype=jp.float32),
        ),
        "recovery/first_step_dy": jp.where(
            finalize_recovery,
            first_step_dy.astype(jp.float32),
            jp.zeros((), dtype=jp.float32),
        ),
        "recovery/first_step_target_err_x": jp.where(
            finalize_recovery,
            first_step_target_err_x.astype(jp.float32),
            jp.zeros((), dtype=jp.float32),
        ),
        "recovery/first_step_target_err_y": jp.where(
            finalize_recovery,
            first_step_target_err_y.astype(jp.float32),
            jp.zeros((), dtype=jp.float32),
        ),
        "recovery/first_step_dist_abs": jp.where(
            finalize_recovery,
            first_step_dist_abs.astype(jp.float32),
            jp.zeros((), dtype=jp.float32),
        ),
        "recovery/pitch_rate_at_push_end": jp.where(
            finalize_recovery,
            pitch_rate_at_push_end.astype(jp.float32),
            jp.zeros((), dtype=jp.float32),
        ),
        "recovery/pitch_rate_at_touchdown": jp.where(
            finalize_recovery,
            pitch_rate_at_touchdown.astype(jp.float32),
            jp.zeros((), dtype=jp.float32),
        ),
        "recovery/pitch_rate_reduction_10t": jp.where(
            finalize_recovery,
            pitch_rate_reduction_10t.astype(jp.float32),
            jp.zeros((), dtype=jp.float32),
        ),
        "recovery/capture_error_at_push_end": jp.where(
            finalize_recovery,
            capture_error_at_push_end.astype(jp.float32),
            jp.zeros((), dtype=jp.float32),
        ),
        "recovery/capture_error_at_touchdown": jp.where(
            finalize_recovery,
            capture_error_at_touchdown.astype(jp.float32),
            jp.zeros((), dtype=jp.float32),
        ),
        "recovery/capture_error_reduction_10t": jp.where(
            finalize_recovery,
            capture_error_reduction_10t.astype(jp.float32),
            jp.zeros((), dtype=jp.float32),
        ),
        "recovery/touchdown_to_term_steps": jp.where(
            finalize_recovery,
            touchdown_to_term_steps.astype(jp.float32),
            jp.zeros((), dtype=jp.float32),
        ),
        "recovery/visible_step_rate": jp.where(
            finalize_recovery,
            visible_step_recorded.astype(jp.float32),
            jp.zeros((), dtype=jp.float32),
        ),
        "recovery/min_height": jp.where(
            finalize_recovery,
            min_height.astype(jp.float32),
            jp.zeros((), dtype=jp.float32),
        ),
        "recovery/max_knee_flex": jp.where(
            finalize_recovery,
            max_knee_flex.astype(jp.float32),
            jp.zeros((), dtype=jp.float32),
        ),
    }

    next_state = {
        "recovery_active": in_recovery & ~finalize_recovery,
        "recovery_age": jp.where(
            in_recovery & ~finalize_recovery,
            age,
            jp.zeros((), dtype=jp.int32),
        ).astype(jp.int32),
        "recovery_last_support_foot": last_support_foot.astype(jp.int32),
        "recovery_first_liftoff_recorded": first_liftoff_recorded,
        "recovery_first_liftoff_latency": first_liftoff_latency.astype(jp.int32),
        "recovery_first_touchdown_recorded": first_touchdown_recorded,
        "recovery_first_step_latency": first_step_latency.astype(jp.int32),
        "recovery_first_touchdown_age": first_touchdown_age.astype(jp.int32),
        "recovery_visible_step_recorded": visible_step_recorded,
        "recovery_touchdown_count": touchdown_count.astype(jp.int32),
        "recovery_support_foot_changes": support_foot_changes.astype(jp.int32),
        "recovery_pitch_rate_at_push_end": pitch_rate_at_push_end.astype(jp.float32),
        "recovery_pitch_rate_at_touchdown": pitch_rate_at_touchdown.astype(jp.float32),
        "recovery_pitch_rate_after_10t": pitch_rate_after_10t.astype(jp.float32),
        "recovery_capture_error_at_push_end": capture_error_at_push_end.astype(jp.float32),
        "recovery_capture_error_at_touchdown": capture_error_at_touchdown.astype(jp.float32),
        "recovery_capture_error_after_10t": capture_error_after_10t.astype(jp.float32),
        "recovery_first_step_dx": first_step_dx.astype(jp.float32),
        "recovery_first_step_dy": first_step_dy.astype(jp.float32),
        "recovery_first_step_target_err_x": first_step_target_err_x.astype(jp.float32),
        "recovery_first_step_target_err_y": first_step_target_err_y.astype(jp.float32),
        "recovery_min_height": min_height.astype(jp.float32),
        "recovery_max_knee_flex": max_knee_flex.astype(jp.float32),
    }
    return next_state, summary_metrics


def _apply_imu_noise_and_delay(signals, rng: jax.Array, cfg, prev_hist_quat, prev_hist_gyro):
    """
    Apply IMU noise and latency using RNG and env config.
    Returns: (signals_override, new_quat_hist, new_gyro_hist, next_rng)

    Note: Avoid consuming RNG when noise/latency are disabled.
    """
    imu_std = float(cfg.env.imu_gyro_noise_std)
    quat_deg = float(cfg.env.imu_quat_noise_deg)
    latency = int(cfg.env.imu_latency_steps)
    max_latency = IMU_MAX_LATENCY
    hist_len = IMU_HIST_LEN

    # Read current raw sensors
    curr_quat = signals.quat_xyzw
    curr_gyro = signals.gyro_rad_s

    use_noise = (imu_std != 0.0) or (quat_deg != 0.0)
    rng_out = rng

    if use_noise:
        rng_out, rng_noise, rng_a, rng_axis = jax.random.split(rng, 4)
        # Sample gyro noise
        gyro_noise = jax.random.normal(rng_noise, shape=curr_gyro.shape) * imu_std
        noisy_gyro = curr_gyro + gyro_noise

        # Sample small random rotation for quaternion noise
        angle_std = quat_deg * (jp.pi / 180.0)
        angle = jax.random.normal(rng_a, shape=()) * angle_std
        axis = jax.random.normal(rng_axis, shape=(3,))
        noise_quat = jax_frames.axis_angle_to_quat(axis, angle)

        # Apply noise: noisy_quat = noise_quat * curr_quat
        noisy_quat = jax_frames.quat_mul(noise_quat, curr_quat)
        # Normalize
        norm = jp.linalg.norm(noisy_quat)
        noisy_quat = noisy_quat / (norm + 1e-12)
    else:
        noisy_gyro = curr_gyro
        noisy_quat = curr_quat

    # Build new histories (most-recent at index 0)
    if prev_hist_quat is None:
        prev_q_hist = jp.repeat(noisy_quat[None, :], hist_len, axis=0)
        prev_g_hist = jp.repeat(noisy_gyro[None, :], hist_len, axis=0)
    else:
        # Roll: put new entry at index 0, drop last
        q_head = noisy_quat[None, :]
        g_head = noisy_gyro[None, :]
        q_rest = prev_hist_quat[: hist_len - 1]
        g_rest = prev_hist_gyro[: hist_len - 1]
        prev_q_hist = jp.concatenate([q_head, q_rest], axis=0)
        prev_g_hist = jp.concatenate([g_head, g_rest], axis=0)

    # Clip latency into valid range and select delayed sample
    latency = jp.clip(latency, 0, max_latency)
    delayed_quat = prev_q_hist[latency]
    delayed_gyro = prev_g_hist[latency]

    # Replace signals with delayed/noisy readings
    signals_override = signals.replace(
        quat_xyzw=delayed_quat.astype(jp.float32),
        gyro_rad_s=delayed_gyro.astype(jp.float32),
    )

    return signals_override, prev_q_hist, prev_g_hist, rng_out


def compute_phase_gated_clearance_reward(
    left_clearance: jax.Array,
    right_clearance: jax.Array,
    left_force: jax.Array,
    right_force: jax.Array,
    contact_threshold: jax.Array,
    phase: jax.Array,
    phase_gate_width: jax.Array,
) -> jax.Array:
    """Compute clock phase-gated clearance reward used by v0.15.9."""
    phase_gate_width = jp.clip(jp.asarray(phase_gate_width, dtype=jp.float32), 1e-3, 0.5)
    phase_half_width = jp.pi * phase_gate_width

    def _phase_gate(theta: jax.Array, center: float) -> jax.Array:
        delta = jp.arctan2(jp.sin(theta - center), jp.cos(theta - center))
        return (jp.abs(delta) <= phase_half_width).astype(jp.float32)

    left_swing_phase = _phase_gate(phase, 0.5 * jp.pi)
    right_swing_phase = _phase_gate(phase, 1.5 * jp.pi)
    left_unloaded = (left_force < contact_threshold).astype(jp.float32)
    right_unloaded = (right_force < contact_threshold).astype(jp.float32)
    return (
        left_clearance * left_swing_phase * left_unloaded
        + right_clearance * right_swing_phase * right_unloaded
    ) * 10.0


def compute_step_length_touchdown_reward(
    left_x: jax.Array,
    right_x: jax.Array,
    touchdown_left: jax.Array,
    touchdown_right: jax.Array,
    velocity_cmd: jax.Array,
    step_reward_gate: jax.Array,
    target_base: jax.Array,
    target_scale: jax.Array,
    sigma: jax.Array,
) -> jax.Array:
    """Compute touchdown step-length reward used by v0.15.9."""
    step_length_target = (
        jp.asarray(target_base, dtype=jp.float32)
        + jp.asarray(target_scale, dtype=jp.float32) * jp.maximum(velocity_cmd, 0.0)
    )
    step_length_sigma = jp.maximum(jp.asarray(sigma, dtype=jp.float32), 1e-6)
    step_len_left = jp.maximum(left_x, 0.0)
    step_len_right = jp.maximum(right_x, 0.0)
    step_len_left_reward = jp.exp(
        -jp.square((step_len_left - step_length_target) / step_length_sigma)
    )
    step_len_right_reward = jp.exp(
        -jp.square((step_len_right - step_length_target) / step_length_sigma)
    )
    reward = jp.where(touchdown_left, step_len_left_reward, 0.0) + jp.where(
        touchdown_right, step_len_right_reward, 0.0
    )
    return jp.asarray(reward).reshape(()) * step_reward_gate


def compute_step_progress_touchdown_reward(
    current_root_pos: jax.Array,
    last_touchdown_root_pos: jax.Array,
    last_touchdown_foot: jax.Array,
    heading_sin: jax.Array,
    heading_cos: jax.Array,
    touchdown_left: jax.Array,
    touchdown_right: jax.Array,
    velocity_cmd: jax.Array,
    velocity_cmd_min: jax.Array,
    step_reward_gate: jax.Array,
    dt: float,
    stride_steps: jax.Array,
    target_scale: jax.Array,
    sigma: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Compute touchdown-to-touchdown structured step progress reward."""
    last_touchdown_foot_i = jp.asarray(last_touchdown_foot, dtype=jp.int32)
    touchdown_any = jp.logical_or(touchdown_left, touchdown_right)
    touchdown_foot = jp.where(
        touchdown_left,
        jp.asarray(0, dtype=jp.int32),
        jp.where(touchdown_right, jp.asarray(1, dtype=jp.int32), last_touchdown_foot_i),
    )
    has_prev_touchdown = last_touchdown_foot_i >= 0
    alternating_touchdown = touchdown_any & has_prev_touchdown & (touchdown_foot != last_touchdown_foot_i)

    delta_world = current_root_pos - last_touchdown_root_pos
    progress_delta = heading_cos * delta_world[0] + heading_sin * delta_world[1]
    step_target = (
        jp.maximum(velocity_cmd, 0.0)
        * dt
        * jp.maximum(jp.asarray(stride_steps, dtype=jp.float32), 1.0)
        * 0.5
        * jp.asarray(target_scale, dtype=jp.float32)
    )
    sigma = jp.maximum(jp.asarray(sigma, dtype=jp.float32), 1e-6)
    reward_raw = jp.exp(-jp.square((progress_delta - step_target) / sigma))
    zero_progress_baseline = jp.exp(-jp.square(step_target / sigma))
    cmd_gate = compute_command_active_gate(velocity_cmd, velocity_cmd_min)
    reward_centered = (1.0 - cmd_gate) * reward_raw + cmd_gate * (reward_raw - zero_progress_baseline)
    reward = jp.where(alternating_touchdown, reward_centered, 0.0) * step_reward_gate

    next_last_touchdown_root_pos = jp.where(
        touchdown_any,
        jp.asarray(current_root_pos, dtype=jp.float32),
        jp.asarray(last_touchdown_root_pos, dtype=jp.float32),
    )
    next_last_touchdown_foot = jp.where(touchdown_any, touchdown_foot, last_touchdown_foot_i)

    return (
        jp.asarray(reward).reshape(()),
        jp.asarray(progress_delta).reshape(()),
        alternating_touchdown.astype(jp.float32).reshape(()),
        next_last_touchdown_root_pos.reshape((3,)),
        next_last_touchdown_foot.reshape(()),
    )


def compute_command_active_gate(velocity_cmd: jax.Array, velocity_cmd_min: jax.Array) -> jax.Array:
    """Return 1 when command magnitude reaches the active-walking threshold."""
    velocity_cmd_min = jp.maximum(jp.asarray(velocity_cmd_min, dtype=jp.float32), 0.0)
    return (jp.abs(velocity_cmd) >= velocity_cmd_min).astype(jp.float32)


def compute_disturbed_window_reward_weights(
    *,
    base_orientation_weight: jax.Array,
    base_height_target_weight: jax.Array,
    base_posture_weight: jax.Array,
    push_active: Optional[jax.Array],
    recovery_active: Optional[jax.Array],
    disturbed_orientation_weight: jax.Array,
    disturbed_height_target_scale: jax.Array,
    disturbed_posture_scale: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Compute effective standing reward weights under disturbed-window gating."""
    disturbed_gate = jp.maximum(
        jp.asarray(push_active if push_active is not None else 0.0, dtype=jp.float32),
        jp.asarray(recovery_active if recovery_active is not None else 0.0, dtype=jp.float32),
    )
    effective_orientation_weight = (
        (1.0 - disturbed_gate) * jp.asarray(base_orientation_weight, dtype=jp.float32)
        + disturbed_gate * jp.asarray(disturbed_orientation_weight, dtype=jp.float32)
    )
    effective_height_target_weight = jp.asarray(base_height_target_weight, dtype=jp.float32) * (
        (1.0 - disturbed_gate)
        + disturbed_gate * jp.asarray(disturbed_height_target_scale, dtype=jp.float32)
    )
    effective_posture_weight = jp.asarray(base_posture_weight, dtype=jp.float32) * (
        (1.0 - disturbed_gate)
        + disturbed_gate * jp.asarray(disturbed_posture_scale, dtype=jp.float32)
    )
    return (
        disturbed_gate,
        effective_orientation_weight,
        effective_height_target_weight,
        effective_posture_weight,
    )


def compute_cycle_progress_terms(
    cycle_progress_accum: jax.Array,
    forward_vel: jax.Array,
    dt: float,
    current_step_count: jax.Array,
    stride_steps: jax.Array,
    velocity_cmd: jax.Array,
    target_scale: jax.Array,
    sigma: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Compute cycle-progress reward and updated accumulator."""
    stride_steps_i = jp.maximum(jp.asarray(stride_steps, dtype=jp.int32), jp.asarray(1, dtype=jp.int32))
    curr_step_i = jp.asarray(current_step_count, dtype=jp.int32)
    cycle_complete = jp.equal(jp.mod(curr_step_i, stride_steps_i), 0)
    cycle_forward_delta = cycle_progress_accum + forward_vel * dt

    cycle_target = (
        jp.maximum(velocity_cmd, 0.0)
        * dt
        * jp.asarray(stride_steps_i, dtype=jp.float32)
        * jp.asarray(target_scale, dtype=jp.float32)
    )
    cycle_sigma = jp.maximum(jp.asarray(sigma, dtype=jp.float32), 1e-6)
    cycle_progress_reward_raw = jp.exp(
        -jp.square((cycle_forward_delta - cycle_target) / cycle_sigma)
    )
    cycle_progress_reward = jp.where(cycle_complete, cycle_progress_reward_raw, 0.0)
    next_cycle_progress_accum = jp.where(cycle_complete, jp.zeros(()), cycle_forward_delta)
    return cycle_progress_reward, cycle_forward_delta, cycle_complete, next_cycle_progress_accum


def compute_zero_baseline_forward_reward(
    forward_vel: jax.Array,
    velocity_cmd: jax.Array,
    scale: jax.Array,
    velocity_cmd_min: jax.Array,
) -> jax.Array:
    """Compute forward tracking reward with standing baseline removed for walking commands."""
    scale = jp.asarray(scale, dtype=jp.float32)
    velocity_cmd_min = jp.asarray(velocity_cmd_min, dtype=jp.float32)
    vel_error = jp.abs(forward_vel - velocity_cmd)
    raw_reward = jp.exp(-vel_error * scale)
    standing_baseline = jp.exp(-jp.abs(velocity_cmd) * scale)
    cmd_gate = compute_command_active_gate(velocity_cmd, velocity_cmd_min)
    reward_zero_baseline = raw_reward - standing_baseline
    return (1.0 - cmd_gate) * raw_reward + cmd_gate * reward_zero_baseline


def compute_height_floor_penalty(
    *,
    height: jax.Array,
    threshold: jax.Array,
    sigma: jax.Array,
) -> jax.Array:
    """Quadratic penalty when height drops below threshold."""
    sigma_safe = jp.maximum(jp.asarray(sigma, dtype=jp.float32), 1e-6)
    normalized_gap = jp.maximum(jp.asarray(threshold, dtype=jp.float32) - height, 0.0) / sigma_safe
    return jp.square(normalized_gap)


def compute_com_velocity_damping_reward(
    *,
    horizontal_speed: jax.Array,
    scale: jax.Array,
) -> jax.Array:
    """Reward low horizontal CoM speed during disturbed recovery windows."""
    scale_safe = jp.maximum(jp.asarray(scale, dtype=jp.float32), 1e-6)
    return jp.exp(-jp.square(horizontal_speed / scale_safe))


def compute_dense_progress_reward(
    forward_vel: jax.Array,
    velocity_cmd: jax.Array,
    velocity_cmd_min: jax.Array,
    left_force: jax.Array,
    right_force: jax.Array,
    contact_threshold: jax.Array,
    pitch: jax.Array,
    pitch_rate: jax.Array,
    upright_pitch_limit: jax.Array,
    upright_pitch_rate_limit: jax.Array,
    upright_gate_sharpness: jax.Array,
) -> jax.Array:
    """Dense heading-local propulsion shaping gated by single-support contact evidence."""
    velocity_cmd_min = jp.maximum(jp.asarray(velocity_cmd_min, dtype=jp.float32), 1e-3)
    cmd_abs = jp.abs(velocity_cmd)
    forward_pos = jp.maximum(forward_vel, 0.0)
    norm_den = jp.maximum(cmd_abs, velocity_cmd_min)
    ratio = jp.clip(forward_pos / norm_den, 0.0, 1.0)
    cmd_gate = compute_command_active_gate(velocity_cmd, velocity_cmd_min)
    left_loaded = left_force > contact_threshold
    right_loaded = right_force > contact_threshold
    support_gate = (left_loaded ^ right_loaded).astype(jp.float32)
    upright_gate = compute_smooth_upright_gate(
        pitch=pitch,
        pitch_rate=pitch_rate,
        upright_pitch_limit=upright_pitch_limit,
        upright_pitch_rate_limit=upright_pitch_rate_limit,
        upright_gate_sharpness=upright_gate_sharpness,
    )
    return ratio * cmd_gate * support_gate * upright_gate


def compute_smooth_upright_gate(
    pitch: jax.Array,
    pitch_rate: jax.Array,
    upright_pitch_limit: jax.Array,
    upright_pitch_rate_limit: jax.Array,
    upright_gate_sharpness: jax.Array,
) -> jax.Array:
    """Smooth uprightness gate using pitch and pitch-rate margins."""
    upright_pitch_limit = jp.maximum(jp.asarray(upright_pitch_limit, dtype=jp.float32), 1e-3)
    upright_pitch_rate_limit = jp.maximum(jp.asarray(upright_pitch_rate_limit, dtype=jp.float32), 1e-3)
    upright_gate_sharpness = jp.maximum(jp.asarray(upright_gate_sharpness, dtype=jp.float32), 1e-3)
    pitch_gate = jax.nn.sigmoid((upright_pitch_limit - jp.abs(pitch)) * upright_gate_sharpness)
    pitch_rate_gate = jax.nn.sigmoid(
        (upright_pitch_rate_limit - jp.abs(pitch_rate)) * upright_gate_sharpness
    )
    return pitch_gate * pitch_rate_gate


def compute_propulsion_quality_gate(
    step_length_reward: jax.Array,
    step_progress_reward: jax.Array,
    step_length_weight: jax.Array,
    step_progress_weight: jax.Array,
) -> jax.Array:
    """Gate forward reward using structured contact-driven step evidence."""
    step_length_weight = jp.maximum(jp.asarray(step_length_weight, dtype=jp.float32), 0.0)
    step_progress_weight = jp.maximum(jp.asarray(step_progress_weight, dtype=jp.float32), 0.0)
    weight_sum = jp.maximum(step_length_weight + step_progress_weight, 1e-6)
    gate_raw = (
        step_length_weight * jp.clip(step_length_reward, 0.0, 1.0)
        + step_progress_weight * jp.clip(step_progress_reward, 0.0, 1.0)
    ) / weight_sum
    return jp.clip(gate_raw, 0.0, 1.0)


# =============================================================================
# Asset Loading
# =============================================================================


def get_assets(root_path: Path) -> Dict[str, bytes]:
    """Load all assets from the WildRobot assets directory.

    Args:
        root_path: Path to assets directory (REQUIRED)

    Returns:
        Dict of asset name to bytes content
    """
    assets = {}

    # Load XML files
    mjx_env.update_assets(assets, root_path, "*.xml")

    # Load STL meshes from assets/assets subdirectory.
    # NOTE: update_assets is non-recursive, so convex_decomposition/* would be
    # missed unless we add recursive loading below.
    meshes_path = root_path / "assets"
    if meshes_path.exists():
        mjx_env.update_assets(assets, meshes_path, "*.stl")
        for stl_path in meshes_path.rglob("*.stl"):
            # Support both plain mesh names ("foo.stl") and nested references
            # produced by convex decomposition ("convex_decomposition/foo.stl").
            rel_from_meshdir = stl_path.relative_to(meshes_path).as_posix()
            assets[rel_from_meshdir] = stl_path.read_bytes()

    return assets


# =============================================================================
# State Definition
# =============================================================================


@struct.dataclass
class WildRobotEnvState(mjx_env.State):
    """WildRobot environment state with Brax compatibility.

    Extends mujoco_playground's State to add:
    - pipeline_state: Required by Brax's training wrapper

    The packed metrics vector is stored in metrics[METRICS_VEC_KEY] following
    the same pattern as info[WR_INFO_KEY]. This avoids adding new fields
    to the state dataclass.
    """

    pipeline_state: mjx.Data = None
    rng: jp.ndarray = None


# =============================================================================
# Environment
# =============================================================================


class WildRobotEnv(mjx_env.MjxEnv):
    """WildRobot walking environment extending MjxEnv.

    This class provides:
    - Fully JAX/JIT compatible reset() and step()
    - Native Brax compatibility via pipeline_state
    - Robot utilities (joint access, sensor reading)

    Example:
        training_cfg = load_training_config("configs/ppo_walking.yaml")
        runtime_cfg = training_cfg.to_runtime_config()
        env = WildRobotEnv(config=runtime_cfg)

        # JIT-compiled reset and step
        reset_fn = jax.jit(env.reset)
        step_fn = jax.jit(env.step)

        # Vectorized environments
        batched_reset = jax.vmap(env.reset)
        batched_step = jax.vmap(env.step)
    """

    def __init__(
        self,
        config: TrainingConfig,
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ) -> None:
        """Initialize WildRobotEnv.

        Args:
            config: TrainingConfig (frozen, JIT-compatible).
                    Call config.freeze() before passing.
            config_overrides: Optional overrides for config values.
        """
        # Create ConfigDict for parent class (MjxEnv expects ConfigDict)
        parent_config = config_dict.ConfigDict()
        parent_config.ctrl_dt = config.env.ctrl_dt
        parent_config.sim_dt = config.env.sim_dt
        super().__init__(parent_config, config_overrides)

        # Store frozen runtime config directly - no extraction needed!
        # All values accessed via self._config.env.*, self._config.ppo.*, etc.
        self._config = config
        self._controller_stack = str(getattr(self._config.env, "controller_stack", "ppo"))
        if self._controller_stack not in ("ppo", "ppo_teacher_standing", "mpc_standing"):
            raise ValueError(
                f"Unsupported controller_stack '{self._controller_stack}'. "
                "Expected one of: ['ppo', 'ppo_teacher_standing', 'mpc_standing']"
            )

        # Ensure IMU history buffer length matches compile-time constant in env_info
        if int(self._config.env.imu_max_latency_steps) != int(IMU_MAX_LATENCY):
            raise ValueError(
                f"Config mismatch: env.imu_max_latency_steps ({self._config.env.imu_max_latency_steps}) "
                f"must equal IMU_MAX_LATENCY ({IMU_MAX_LATENCY}) compiled into env_info.py. "
                "Adjust your config or the code to use a different fixed max latency."
            )
        if int(getattr(self._config.env, "action_delay_steps", 0)) not in (0, 1):
            raise ValueError(
                "Only action_delay_steps in {0, 1} is currently supported."
            )
        self._action_delay_enabled = int(getattr(self._config.env, "action_delay_steps", 0)) == 1

        # Model path (required - must be in config)
        if not config.env.scene_xml_path:
            raise ValueError(
                "scene_xml_path is required in config.env. "
                "Set env.assets_root or env.scene_xml_path in your config."
            )
        self._model_path = Path(config.env.scene_xml_path)

        # Load model and setup robot infrastructure
        self._load_model()

        print(f"WildRobotEnv initialized:")
        print(f"  Actuators: {self._mj_model.nu}")
        print(f"  Floating base: {self._robot_config.floating_base_body}")
        print(f"  Control dt: {self.dt}s, Sim dt: {self.sim_dt}s")

    # =========================================================================
    # Model Loading
    # =========================================================================

    def _load_model(self) -> None:
        """Load MuJoCo model and setup robot infrastructure.

        This method:
        - Loads the XML model file from config path
        - Creates MJX model for JAX compatibility
        - Sets up actuator and joint mappings
        - Extracts initial qpos from keyframe or qpos0
        """
        # Resolve model path relative to project root
        project_root = Path(__file__).parent.parent.parent
        xml_path = project_root / self._model_path

        if not xml_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {xml_path}. "
                f"Check model_path in config: {self._model_path}"
            )

        root_path = xml_path.parent

        print(f"Loading WildRobot model from: {xml_path}")

        self._mj_model = mujoco.MjModel.from_xml_string(
            xml_path.read_text(), assets=get_assets(root_path)
        )
        self._mj_model.opt.timestep = self.sim_dt

        # Create MJX model for JAX compatibility
        self._mjx_model = mjx.put_model(self._mj_model)

        # Get robot config (loaded by train.py at startup)
        self._robot_config = get_robot_config()
        self._signals_adapter = MjxSignalsAdapter(
            self._mj_model,
            self._robot_config,
            foot_switch_threshold=self._config.env.foot_switch_threshold,
        )

        # =====================================================================
        # Initialize Control Abstraction Layer (CAL) - v0.11.0
        # =====================================================================
        # CAL provides the canonical action→ctrl transformation with:
        # - Symmetry correction (policy_action_sign) for left/right joints
        # - Normalization from [-1, 1] to physical joint ranges
        # - Single source of truth for joint specifications
        # - Extended state for foot, root, and knee IMU access
        # - Contact scale is read from robot_config.feet.contact_scale
        self._cal = ControlAbstractionLayer(self._mj_model, self._robot_config)
        print(
            f"  CAL initialized with {self._cal.num_actuators} actuators (extended state included)"
        )

        # Get initial qpos from model's keyframe or qpos0
        # MuJoCo stores keyframe qpos in key_qpos (shape: nkey x nq)
        if self._mj_model.nkey > 0:
            # Use first keyframe (typically "home" pose)
            self._init_qpos = jp.array(self._mj_model.key_qpos[0])
            print(f"  Using keyframe qpos from model (nkey={self._mj_model.nkey})")
            # Debug: print initial height from keyframe
            init_height = float(self._init_qpos[2])
            print(f"  Initial height from keyframe: {init_height:.4f}m")
        else:
            # Fall back to qpos0 (model default)
            self._init_qpos = jp.array(self._mj_model.qpos0)
            print(f"  Using qpos0 from model")
            init_height = float(self._init_qpos[2])
            print(f"  Initial height from qpos0: {init_height:.4f}m")

        # Extract joint-only qpos for control default from CAL (single source of truth)
        self._default_joint_qpos = self._cal.get_ctrl_for_default_pose()

        # Build policy spec after CAL so the action contract uses the exact
        # MJCF home pose everywhere: training, env, eval, and export.
        home_ctrl_list = clamp_home_ctrl(
            home_ctrl=get_home_ctrl_from_mj_model(
                mj_model=self._mj_model,
                actuator_names=[str(item["name"]) for item in self._robot_config.actuated_joints],
            ),
            actuated_joint_specs=self._robot_config.actuated_joints,
            actuator_names=[str(item["name"]) for item in self._robot_config.actuated_joints],
        )
        self._policy_spec = build_policy_spec(
            robot_name=self._robot_config.robot_name,
            actuated_joint_specs=self._robot_config.actuated_joints,
            action_filter_alpha=float(self._config.env.action_filter_alpha),
            layout_id=str(self._config.env.actor_obs_layout_id),
            mapping_id=str(self._config.env.action_mapping_id),
            home_ctrl_rad=home_ctrl_list,
        )
        self._loc_ref_enabled = bool(getattr(self._config.env, "loc_ref_enabled", False)) or (
            self._policy_spec.observation.layout_id == "wr_obs_v4"
        )
        self._loc_ref_version = str(getattr(self._config.env, "loc_ref_version", "v1")).lower()
        if self._loc_ref_version not in ("v1", "v2"):
            raise ValueError("env.loc_ref_version must be 'v1' or 'v2'")
        self._actuator_name_to_index = {
            name: i for i, name in enumerate(self._policy_spec.robot.actuator_names)
        }
        self._walking_ref_cfg = WalkingRefV1Config(
            step_time_s=float(getattr(self._config.env, "loc_ref_step_time_s", 0.36)),
            nominal_com_height_m=float(
                getattr(self._config.env, "loc_ref_nominal_com_height_m", 0.40)
            ),
            nominal_lateral_foot_offset_m=float(
                getattr(self._config.env, "loc_ref_nominal_lateral_foot_offset_m", 0.09)
            ),
            min_step_length_m=float(
                getattr(self._config.env, "loc_ref_min_step_length_m", 0.02)
            ),
            max_step_length_m=float(
                getattr(self._config.env, "loc_ref_max_step_length_m", 0.14)
            ),
            max_lateral_step_m=float(
                getattr(self._config.env, "loc_ref_max_lateral_step_m", 0.14)
            ),
            swing_height_m=float(getattr(self._config.env, "loc_ref_swing_height_m", 0.04)),
            pelvis_roll_bias_rad=float(
                getattr(self._config.env, "loc_ref_pelvis_roll_bias_rad", 0.03)
            ),
            pelvis_pitch_gain=float(
                getattr(self._config.env, "loc_ref_pelvis_pitch_gain", 0.08)
            ),
            max_pelvis_pitch_rad=float(
                getattr(self._config.env, "loc_ref_max_pelvis_pitch_rad", 0.08)
            ),
            dcm_placement_gain=float(
                getattr(self._config.env, "loc_ref_dcm_placement_gain", 1.0)
            ),
        )
        self._walking_ref_v2_cfg = WalkingRefV2Config(
            step_time_s=float(getattr(self._config.env, "loc_ref_step_time_s", 0.36)),
            nominal_com_height_m=float(
                getattr(self._config.env, "loc_ref_nominal_com_height_m", 0.40)
            ),
            nominal_lateral_foot_offset_m=float(
                getattr(self._config.env, "loc_ref_nominal_lateral_foot_offset_m", 0.09)
            ),
            min_step_length_m=float(
                getattr(self._config.env, "loc_ref_min_step_length_m", 0.02)
            ),
            max_step_length_m=float(
                getattr(self._config.env, "loc_ref_max_step_length_m", 0.14)
            ),
            max_lateral_step_m=float(
                getattr(self._config.env, "loc_ref_max_lateral_step_m", 0.14)
            ),
            swing_height_m=float(getattr(self._config.env, "loc_ref_swing_height_m", 0.04)),
            pelvis_roll_bias_rad=float(
                getattr(self._config.env, "loc_ref_pelvis_roll_bias_rad", 0.03)
            ),
            pelvis_pitch_gain=float(
                getattr(self._config.env, "loc_ref_pelvis_pitch_gain", 0.08)
            ),
            max_pelvis_pitch_rad=float(
                getattr(self._config.env, "loc_ref_max_pelvis_pitch_rad", 0.08)
            ),
            max_forward_speed_mps=float(
                max(getattr(self._config.env, "max_velocity", 0.35), 1e-3)
            ),
            dcm_correction_gain=float(
                getattr(self._config.env, "loc_ref_dcm_placement_gain", 1.0)
            ),
            support_pitch_start_rad=float(
                getattr(self._config.env, "loc_ref_swing_x_brake_pitch_start_rad", 0.08)
            ),
            support_pitch_rate_start_rad_s=float(
                getattr(self._config.env, "loc_ref_support_pitch_rate_start_rad_s", 0.8)
            ),
            support_overspeed_deadband=float(
                getattr(self._config.env, "loc_ref_swing_x_brake_overspeed_deadband", 0.03)
            ),
            support_health_gain=float(
                getattr(self._config.env, "loc_ref_support_health_gain", 4.0)
            ),
            support_open_threshold=float(
                getattr(self._config.env, "loc_ref_v2_support_open_threshold", 0.60)
            ),
            support_release_threshold=float(
                getattr(self._config.env, "loc_ref_v2_support_release_threshold", 0.30)
            ),
            support_release_phase_start=float(
                getattr(self._config.env, "loc_ref_support_release_phase_start", 0.35)
            ),
            support_foothold_min_scale=float(
                getattr(self._config.env, "loc_ref_support_foothold_min_scale", 0.35)
            ),
            support_phase_min_scale=float(
                getattr(self._config.env, "loc_ref_support_phase_min_scale", 0.15)
            ),
            support_swing_progress_min_scale=float(
                getattr(self._config.env, "loc_ref_support_swing_progress_min_scale", 0.0)
            ),
            support_stabilize_max_foothold_scale=float(
                getattr(self._config.env, "loc_ref_v2_support_stabilize_max_foothold_scale", 0.35)
            ),
            post_settle_swing_scale=float(
                getattr(self._config.env, "loc_ref_v2_post_settle_swing_scale", 0.15)
            ),
            startup_ramp_s=float(
                getattr(self._config.env, "loc_ref_v2_startup_ramp_s", 0.18)
            ),
            startup_pelvis_height_offset_m=float(
                getattr(self._config.env, "loc_ref_v2_startup_pelvis_height_offset_m", 0.035)
            ),
            startup_support_open_health=float(
                getattr(self._config.env, "loc_ref_v2_startup_support_open_health", 0.25)
            ),
            startup_handoff_pitch_max_rad=float(
                getattr(self._config.env, "loc_ref_v2_startup_handoff_pitch_max_rad", 0.30)
            ),
            startup_handoff_pitch_rate_max_rad_s=float(
                getattr(
                    self._config.env,
                    "loc_ref_v2_startup_handoff_pitch_rate_max_rad_s",
                    1.50,
                )
            ),
            startup_readiness_knee_err_good_rad=float(
                getattr(self._config.env, "loc_ref_v2_startup_readiness_knee_err_good_rad", 0.10)
            ),
            startup_readiness_knee_err_bad_rad=float(
                getattr(self._config.env, "loc_ref_v2_startup_readiness_knee_err_bad_rad", 0.45)
            ),
            startup_readiness_ankle_err_good_rad=float(
                getattr(self._config.env, "loc_ref_v2_startup_readiness_ankle_err_good_rad", 0.08)
            ),
            startup_readiness_ankle_err_bad_rad=float(
                getattr(self._config.env, "loc_ref_v2_startup_readiness_ankle_err_bad_rad", 0.35)
            ),
            startup_readiness_pitch_good_rad=float(
                getattr(self._config.env, "loc_ref_v2_startup_readiness_pitch_good_rad", 0.05)
            ),
            startup_readiness_pitch_bad_rad=float(
                getattr(self._config.env, "loc_ref_v2_startup_readiness_pitch_bad_rad", 0.25)
            ),
            startup_readiness_pitch_rate_good_rad_s=float(
                getattr(
                    self._config.env,
                    "loc_ref_v2_startup_readiness_pitch_rate_good_rad_s",
                    0.60,
                )
            ),
            startup_readiness_pitch_rate_bad_rad_s=float(
                getattr(
                    self._config.env,
                    "loc_ref_v2_startup_readiness_pitch_rate_bad_rad_s",
                    2.00,
                )
            ),
            startup_readiness_min_health=float(
                getattr(self._config.env, "loc_ref_v2_startup_readiness_min_health", 0.20)
            ),
            startup_progress_min_scale=float(
                getattr(self._config.env, "loc_ref_v2_startup_progress_min_scale", 0.20)
            ),
            startup_realization_lead_alpha=float(
                getattr(self._config.env, "loc_ref_v2_startup_realization_lead_alpha", 0.25)
            ),
            startup_handoff_min_readiness=float(
                getattr(self._config.env, "loc_ref_v2_startup_handoff_min_readiness", 0.10)
            ),
            startup_handoff_min_pelvis_realization=float(
                getattr(
                    self._config.env,
                    "loc_ref_v2_startup_handoff_min_pelvis_realization",
                    0.60,
                )
            ),
            startup_handoff_min_alpha=float(
                getattr(self._config.env, "loc_ref_v2_startup_handoff_min_alpha", 0.85)
            ),
            startup_handoff_timeout_s=float(
                getattr(self._config.env, "loc_ref_v2_startup_handoff_timeout_s", 1.20)
            ),
            startup_target_rate_design_rad_s=float(
                getattr(
                    self._config.env,
                    "loc_ref_v2_startup_target_rate_design_rad_s",
                    0.5235987756,
                )
            ),
            startup_target_rate_hard_cap_rad_s=float(
                getattr(
                    self._config.env,
                    "loc_ref_v2_startup_target_rate_hard_cap_rad_s",
                    1.7453292520,
                )
            ),
            startup_route_w1_alpha=float(
                getattr(self._config.env, "loc_ref_v2_startup_route_w1_alpha", 0.25)
            ),
            startup_route_w2_alpha=float(
                getattr(self._config.env, "loc_ref_v2_startup_route_w2_alpha", 0.50)
            ),
            startup_route_w3_alpha=float(
                getattr(self._config.env, "loc_ref_v2_startup_route_w3_alpha", 0.75)
            ),
            startup_route_w1_scale=float(
                getattr(self._config.env, "loc_ref_v2_startup_route_w1_scale", 0.20)
            ),
            startup_route_w2_scale=float(
                getattr(self._config.env, "loc_ref_v2_startup_route_w2_scale", 0.45)
            ),
            startup_route_w3_scale=float(
                getattr(self._config.env, "loc_ref_v2_startup_route_w3_scale", 0.75)
            ),
            startup_route_w1_support_y_scale=float(
                getattr(self._config.env, "loc_ref_v2_startup_route_w1_support_y_scale", 0.25)
            ),
            startup_route_w2_support_y_scale=float(
                getattr(self._config.env, "loc_ref_v2_startup_route_w2_support_y_scale", 0.60)
            ),
            startup_route_w3_support_y_scale=float(
                getattr(self._config.env, "loc_ref_v2_startup_route_w3_support_y_scale", 0.85)
            ),
            startup_route_w1_pelvis_roll_scale=float(
                getattr(self._config.env, "loc_ref_v2_startup_route_w1_pelvis_roll_scale", 0.45)
            ),
            startup_route_w2_pelvis_roll_scale=float(
                getattr(self._config.env, "loc_ref_v2_startup_route_w2_pelvis_roll_scale", 0.75)
            ),
            startup_route_w3_pelvis_roll_scale=float(
                getattr(self._config.env, "loc_ref_v2_startup_route_w3_pelvis_roll_scale", 0.90)
            ),
            startup_route_w1_pelvis_pitch_scale=float(
                getattr(self._config.env, "loc_ref_v2_startup_route_w1_pelvis_pitch_scale", 0.20)
            ),
            startup_route_w2_pelvis_pitch_scale=float(
                getattr(self._config.env, "loc_ref_v2_startup_route_w2_pelvis_pitch_scale", 0.50)
            ),
            startup_route_w3_pelvis_pitch_scale=float(
                getattr(self._config.env, "loc_ref_v2_startup_route_w3_pelvis_pitch_scale", 0.80)
            ),
            startup_route_w1_pelvis_height_scale=float(
                getattr(self._config.env, "loc_ref_v2_startup_route_w1_pelvis_height_scale", 0.15)
            ),
            startup_route_w2_pelvis_height_scale=float(
                getattr(self._config.env, "loc_ref_v2_startup_route_w2_pelvis_height_scale", 0.45)
            ),
            startup_route_w3_pelvis_height_scale=float(
                getattr(self._config.env, "loc_ref_v2_startup_route_w3_pelvis_height_scale", 0.80)
            ),
            startup_route_w2_min_pelvis_realization=float(
                getattr(
                    self._config.env,
                    "loc_ref_v2_startup_route_w2_min_pelvis_realization",
                    0.30,
                )
            ),
            startup_route_w3_min_pelvis_realization=float(
                getattr(
                    self._config.env,
                    "loc_ref_v2_startup_route_w3_min_pelvis_realization",
                    0.55,
                )
            ),
            startup_route_w2_pitch_relax=float(
                getattr(self._config.env, "loc_ref_v2_startup_route_w2_pitch_relax", 1.25)
            ),
            startup_route_w2_pitch_rate_relax=float(
                getattr(
                    self._config.env,
                    "loc_ref_v2_startup_route_w2_pitch_rate_relax",
                    1.25,
                )
            ),
            support_entry_shaping_window_s=float(
                getattr(self._config.env, "loc_ref_v2_support_entry_shaping_window_s", 0.12)
            ),
            max_lateral_release_m=float(
                getattr(self._config.env, "loc_ref_max_lateral_release_m", 0.02)
            ),
            touchdown_phase_min=float(
                getattr(self._config.env, "loc_ref_v2_touchdown_phase_min", 0.55)
            ),
            capture_hold_s=float(
                getattr(self._config.env, "loc_ref_v2_capture_hold_s", 0.04)
            ),
            settle_hold_s=float(
                getattr(self._config.env, "loc_ref_v2_settle_hold_s", 0.04)
            ),
            support_pelvis_height_offset_m=float(
                getattr(self._config.env, "loc_ref_v2_support_pelvis_height_offset_m", 0.050)
            ),
        )
        self._leg_ik_cfg = LegIkConfig()
        self._residual_q_scale = jp.asarray(
            getattr(self._config.env, "loc_ref_residual_scale", 0.18), dtype=jp.float32
        )
        self._loc_ref_swing_target_blend = jp.asarray(
            getattr(self._config.env, "loc_ref_swing_target_blend", 0.65), dtype=jp.float32
        )
        self._loc_ref_stance_height_blend = jp.asarray(
            getattr(self._config.env, "loc_ref_stance_height_blend", 0.25), dtype=jp.float32
        )
        self._loc_ref_stance_extension_margin_m = jp.asarray(
            getattr(self._config.env, "loc_ref_stance_extension_margin_m", 0.015),
            dtype=jp.float32,
        )
        self._loc_ref_max_swing_x_delta_m = jp.asarray(
            getattr(self._config.env, "loc_ref_max_swing_x_delta_m", 0.04), dtype=jp.float32
        )
        self._loc_ref_max_swing_z_delta_m = jp.asarray(
            getattr(self._config.env, "loc_ref_max_swing_z_delta_m", 0.03), dtype=jp.float32
        )
        self._loc_ref_swing_y_to_hip_roll = jp.asarray(
            getattr(self._config.env, "loc_ref_swing_y_to_hip_roll", 0.30), dtype=jp.float32
        )
        self._loc_ref_overspeed_deadband = jp.asarray(
            getattr(self._config.env, "loc_ref_overspeed_deadband", 0.05), dtype=jp.float32
        )
        self._loc_ref_overspeed_brake_gain = jp.asarray(
            getattr(self._config.env, "loc_ref_overspeed_brake_gain", 1.5), dtype=jp.float32
        )
        self._loc_ref_overspeed_phase_slowdown_gain = jp.asarray(
            getattr(self._config.env, "loc_ref_overspeed_phase_slowdown_gain", 2.5),
            dtype=jp.float32,
        )
        self._loc_ref_overspeed_phase_min_scale = jp.asarray(
            getattr(self._config.env, "loc_ref_overspeed_phase_min_scale", 0.2),
            dtype=jp.float32,
        )
        self._loc_ref_pitch_brake_start_rad = jp.asarray(
            getattr(self._config.env, "loc_ref_pitch_brake_start_rad", 0.12), dtype=jp.float32
        )
        self._loc_ref_pitch_brake_gain = jp.asarray(
            getattr(self._config.env, "loc_ref_pitch_brake_gain", 1.0), dtype=jp.float32
        )
        self._loc_ref_swing_x_brake_pitch_start_rad = jp.asarray(
            getattr(self._config.env, "loc_ref_swing_x_brake_pitch_start_rad", 0.08),
            dtype=jp.float32,
        )
        self._loc_ref_swing_x_brake_overspeed_deadband = jp.asarray(
            getattr(self._config.env, "loc_ref_swing_x_brake_overspeed_deadband", 0.03),
            dtype=jp.float32,
        )
        self._loc_ref_swing_x_brake_gain = jp.asarray(
            getattr(self._config.env, "loc_ref_swing_x_brake_gain", 5.0), dtype=jp.float32
        )
        self._loc_ref_swing_x_min_scale = jp.asarray(
            getattr(self._config.env, "loc_ref_swing_x_min_scale", 0.05), dtype=jp.float32
        )
        self._loc_ref_pelvis_pitch_brake_gain = jp.asarray(
            getattr(self._config.env, "loc_ref_pelvis_pitch_brake_gain", 6.0),
            dtype=jp.float32,
        )
        self._loc_ref_pelvis_pitch_min_scale = jp.asarray(
            getattr(self._config.env, "loc_ref_pelvis_pitch_min_scale", 0.0),
            dtype=jp.float32,
        )
        self._loc_ref_support_pitch_rate_start_rad_s = jp.asarray(
            getattr(self._config.env, "loc_ref_support_pitch_rate_start_rad_s", 0.8),
            dtype=jp.float32,
        )
        self._loc_ref_support_health_gain = jp.asarray(
            getattr(self._config.env, "loc_ref_support_health_gain", 4.0),
            dtype=jp.float32,
        )
        self._loc_ref_support_release_phase_start = jp.asarray(
            getattr(self._config.env, "loc_ref_support_release_phase_start", 0.35),
            dtype=jp.float32,
        )
        self._loc_ref_support_foothold_min_scale = jp.asarray(
            getattr(self._config.env, "loc_ref_support_foothold_min_scale", 0.35),
            dtype=jp.float32,
        )
        self._loc_ref_support_swing_progress_min_scale = jp.asarray(
            getattr(self._config.env, "loc_ref_support_swing_progress_min_scale", 0.0),
            dtype=jp.float32,
        )
        self._loc_ref_support_phase_min_scale = jp.asarray(
            getattr(self._config.env, "loc_ref_support_phase_min_scale", 0.15),
            dtype=jp.float32,
        )
        self._loc_ref_v2_open_threshold = jp.asarray(
            getattr(self._config.env, "loc_ref_v2_support_open_threshold", 0.60),
            dtype=jp.float32,
        )
        self._loc_ref_v2_release_threshold = jp.asarray(
            getattr(self._config.env, "loc_ref_v2_support_release_threshold", 0.30),
            dtype=jp.float32,
        )
        self._loc_ref_v2_touchdown_phase_min = jp.asarray(
            getattr(self._config.env, "loc_ref_v2_touchdown_phase_min", 0.55),
            dtype=jp.float32,
        )
        self._loc_ref_v2_capture_hold_s = jp.asarray(
            getattr(self._config.env, "loc_ref_v2_capture_hold_s", 0.04),
            dtype=jp.float32,
        )
        self._loc_ref_v2_settle_hold_s = jp.asarray(
            getattr(self._config.env, "loc_ref_v2_settle_hold_s", 0.04),
            dtype=jp.float32,
        )
        self._loc_ref_v2_support_stabilize_max_foothold_scale = jp.asarray(
            getattr(self._config.env, "loc_ref_v2_support_stabilize_max_foothold_scale", 0.35),
            dtype=jp.float32,
        )
        self._loc_ref_v2_post_settle_swing_scale = jp.asarray(
            getattr(self._config.env, "loc_ref_v2_post_settle_swing_scale", 0.15),
            dtype=jp.float32,
        )
        self._loc_ref_v2_support_pelvis_height_offset_m = jp.asarray(
            getattr(self._config.env, "loc_ref_v2_support_pelvis_height_offset_m", 0.050),
            dtype=jp.float32,
        )
        self._loc_ref_v2_support_stance_extension_margin_m = jp.asarray(
            getattr(self._config.env, "loc_ref_v2_support_stance_extension_margin_m", 0.020),
            dtype=jp.float32,
        )
        self._idx_left_hip_pitch = self._actuator_name_to_index.get("left_hip_pitch", -1)
        self._idx_right_hip_pitch = self._actuator_name_to_index.get("right_hip_pitch", -1)
        self._idx_left_hip_roll = self._actuator_name_to_index.get("left_hip_roll", -1)
        self._idx_right_hip_roll = self._actuator_name_to_index.get("right_hip_roll", -1)
        self._idx_left_knee_pitch = self._actuator_name_to_index.get("left_knee_pitch", -1)
        self._idx_right_knee_pitch = self._actuator_name_to_index.get("right_knee_pitch", -1)
        self._idx_left_ankle_pitch = self._actuator_name_to_index.get("left_ankle_pitch", -1)
        self._idx_right_ankle_pitch = self._actuator_name_to_index.get("right_ankle_pitch", -1)
        self._idx_left_shoulder_pitch = self._actuator_name_to_index.get(
            "left_shoulder_pitch", -1
        )
        self._idx_right_shoulder_pitch = self._actuator_name_to_index.get(
            "right_shoulder_pitch", -1
        )
        self._idx_waist_yaw = self._actuator_name_to_index.get("waist_yaw", -1)

        self._default_pose_action = JaxCalibOps.ctrl_to_policy_action(
            spec=self._policy_spec,
            ctrl_rad=self._default_joint_qpos,
        ).astype(jp.float32)
        self._joint_range_mins = jp.asarray(
            [float(item["range"][0]) for item in self._robot_config.actuated_joints],
            dtype=jp.float32,
        )
        self._joint_range_maxs = jp.asarray(
            [float(item["range"][1]) for item in self._robot_config.actuated_joints],
            dtype=jp.float32,
        )
        self._joint_half_spans = 0.5 * (self._joint_range_maxs - self._joint_range_mins)
        actuator_qpos_addrs = []
        actuator_dof_addrs = []
        for name in self._policy_spec.robot.actuator_names:
            act_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            joint_id = int(self._mj_model.actuator_trnid[act_id][0])
            actuator_qpos_addrs.append(int(self._mj_model.jnt_qposadr[joint_id]))
            actuator_dof_addrs.append(int(self._mj_model.jnt_dofadr[joint_id]))
        self._actuator_qpos_addrs = jp.asarray(actuator_qpos_addrs, dtype=jp.int32)
        self._actuator_dof_addrs = jp.asarray(actuator_dof_addrs, dtype=jp.int32)
        self._base_geom_friction = self._mjx_model.geom_friction
        self._base_body_mass = self._mjx_model.body_mass
        self._base_actuator_gainprm = self._mjx_model.actuator_gainprm
        self._base_actuator_biasprm = self._mjx_model.actuator_biasprm
        self._base_dof_frictionloss = self._mjx_model.dof_frictionloss

        # Disturbance configuration
        self._push_body_ids = jp.asarray([-1], dtype=jp.int32)
        if self._config.env.push_enabled:
            push_body_names = list(self._config.env.push_bodies)
            if not push_body_names:
                push_body_names = [self._config.env.push_body]

            push_body_ids = []
            for body_name in push_body_names:
                body_id = mujoco.mj_name2id(
                    self._mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name
                )
                if body_id < 0:
                    raise ValueError(f"Push body '{body_name}' not found in model.")
                push_body_ids.append(int(body_id))
            self._push_body_ids = jp.asarray(push_body_ids, dtype=jp.int32)

    # =========================================================================
    # MjxEnv Interface
    # =========================================================================

    def _sample_domain_rand_params(self, rng: jax.Array) -> dict[str, jax.Array]:
        if not bool(getattr(self._config.env, "domain_randomization_enabled", False)):
            return nominal_domain_rand_params(
                num_bodies=self._mj_model.nbody,
                num_actuators=self.action_size,
            )
        return sample_domain_rand_params(
            rng,
            num_bodies=self._mj_model.nbody,
            num_actuators=self.action_size,
            friction_range=tuple(self._config.env.domain_rand_friction_range),
            mass_scale_range=tuple(self._config.env.domain_rand_mass_scale_range),
            kp_scale_range=tuple(self._config.env.domain_rand_kp_scale_range),
            frictionloss_scale_range=tuple(
                self._config.env.domain_rand_frictionloss_scale_range
            ),
            joint_offset_rad=float(self._config.env.domain_rand_joint_offset_rad),
        )

    def _get_randomized_mjx_model(self, wr: WildRobotInfo | dict[str, jax.Array]) -> mjx.Model:
        if not bool(getattr(self._config.env, "domain_randomization_enabled", False)):
            return self._mjx_model
        if isinstance(wr, dict):
            friction_scale = wr["friction_scale"]
            mass_scales = wr["mass_scales"]
            kp_scales = wr["kp_scales"]
            frictionloss_scales = wr["frictionloss_scales"]
        else:
            friction_scale = wr.domain_rand_friction_scale
            mass_scales = wr.domain_rand_mass_scales
            kp_scales = wr.domain_rand_kp_scales
            frictionloss_scales = wr.domain_rand_frictionloss_scales

        geom_friction = self._base_geom_friction * friction_scale
        body_mass = self._base_body_mass * mass_scales
        actuator_gainprm = self._base_actuator_gainprm.at[:, 0].set(
            self._base_actuator_gainprm[:, 0] * kp_scales
        )
        actuator_biasprm = self._base_actuator_biasprm.at[:, 1].set(
            self._base_actuator_biasprm[:, 1] * kp_scales
        )
        dof_frictionloss = self._base_dof_frictionloss.at[self._actuator_dof_addrs].set(
            self._base_dof_frictionloss[self._actuator_dof_addrs] * frictionloss_scales
        )
        return self._mjx_model.replace(
            geom_friction=geom_friction,
            body_mass=body_mass,
            actuator_gainprm=actuator_gainprm,
            actuator_biasprm=actuator_biasprm,
            dof_frictionloss=dof_frictionloss,
        )

    def _make_initial_state(
        self,
        qpos: jax.Array,
        velocity_cmd: jax.Array,
        base_info: dict | None = None,
        push_schedule: DisturbanceSchedule | None = None,
        domain_rand_params: dict[str, jax.Array] | None = None,
    ) -> WildRobotEnvState:
        """Create initial environment state from qpos (shared by reset and auto-reset).

        Args:
            qpos: Initial joint positions (including floating base)
            velocity_cmd: Velocity command for this episode
            base_info: Optional base info dict to preserve (for auto-reset).
                       When provided, env fields are merged on top of base_info
                       to preserve wrapper-injected fields (truncation, episode_done, etc.)

        Returns:
            Initial WildRobotEnvState
        """
        qvel = jp.zeros(self._mj_model.nv)
        if domain_rand_params is None:
            domain_rand_params = nominal_domain_rand_params(
                num_bodies=self._mj_model.nbody,
                num_actuators=self.action_size,
            )
        randomized_model = self._get_randomized_mjx_model(domain_rand_params)

        # Create MJX data and run forward kinematics
        data = mjx.make_data(randomized_model)
        data = data.replace(qpos=qpos, qvel=qvel, ctrl=self._default_joint_qpos)
        data = mjx.forward(randomized_model, data)

        if push_schedule is None:
            push_schedule = DisturbanceSchedule(
                start_step=jp.zeros(()),
                end_step=jp.zeros(()),
                force_xy=jp.zeros((2,)),
                body_id=jp.asarray(-1, dtype=jp.int32),
                rng=jp.zeros((2,), dtype=jp.uint32),
            )

        # Build observation using default pose action (matches ctrl at reset)
        default_action = JaxCalibOps.ctrl_to_policy_action(
            spec=self._policy_spec,
            ctrl_rad=self._default_joint_qpos,
        )

        # Read raw sensors and initialize IMU history using push_schedule.rng
        raw_signals = self._signals_adapter.read(data)
        signals_override, imu_quat_hist, imu_gyro_hist, next_rng = _apply_imu_noise_and_delay(
            raw_signals, push_schedule.rng, self._config, None, None
        )

        root_pose = self._cal.get_root_pose(data)
        root_vel_h = self._cal.get_root_velocity(data, frame=CoordinateFrame.HEADING_LOCAL)
        _, pitch_reset, _ = root_pose.euler_angles()
        left_foot_h, right_foot_h = self._cal.get_foot_positions(
            data, normalize=False, frame=CoordinateFrame.HEADING_LOCAL
        )
        left_force, right_force = self._cal.get_aggregated_foot_contacts(data)
        left_loaded_reset = left_force > self._config.env.contact_threshold_force
        right_loaded_reset = right_force > self._config.env.contact_threshold_force
        stance_pos_h = left_foot_h
        if self._loc_ref_version == "v2":
            (
                ref_state,
                ref_phase_time,
                ref_stance_foot,
                ref_switch_count,
                ref_mode_id,
                ref_mode_time,
                ref_support_health,
                ref_support_instability,
                ref_progression_permission,
            ) = step_reference_v2_jax(
                config=self._walking_ref_v2_cfg,
                phase_time_s=jp.zeros((), dtype=jp.float32),
                stance_foot_id=jp.asarray(REF_LEFT_STANCE, dtype=jp.int32),
                stance_switch_count=jp.zeros((), dtype=jp.int32),
                mode_id=jp.asarray(int(WalkingRefV2Mode.STARTUP_SUPPORT_RAMP), dtype=jp.int32),
                mode_time_s=jp.zeros((), dtype=jp.float32),
                forward_speed_mps=velocity_cmd,
                com_position_stance_frame=jp.asarray(
                    [-stance_pos_h[0], -stance_pos_h[1]], dtype=jp.float32
                ),
                com_velocity_stance_frame=jp.asarray(
                    [root_vel_h.linear_xyz[0], root_vel_h.linear_xyz[1]], dtype=jp.float32
                ),
                root_pitch_rad=pitch_reset,
                root_pitch_rate_rad_s=root_vel_h.angular_xyz[1],
                root_height_m=root_pose.height,
                left_foot_loaded=left_loaded_reset,
                right_foot_loaded=right_loaded_reset,
                dt_s=self.dt,
                stance_knee_tracking_error_rad=jp.zeros((), dtype=jp.float32),
                stance_ankle_tracking_error_rad=jp.zeros((), dtype=jp.float32),
            )
            ref_overspeed = jp.maximum(
                jp.asarray(root_vel_h.linear_xyz[0], dtype=jp.float32)
                - jp.maximum(jp.asarray(velocity_cmd, dtype=jp.float32), 0.0)
                - jp.asarray(self._loc_ref_swing_x_brake_overspeed_deadband, dtype=jp.float32),
                0.0,
            ).astype(jp.float32)
            ref_mode_id_dbg = ref_mode_id.astype(jp.float32)
            startup_route_progress = jp.asarray(
                ref_state.get("startup_route_progress", jp.asarray(1.0, dtype=jp.float32)),
                dtype=jp.float32,
            )
            startup_route_ceiling = jp.asarray(
                ref_state.get("startup_route_ceiling", jp.asarray(1.0, dtype=jp.float32)),
                dtype=jp.float32,
            )
            startup_route_stage_id = jp.asarray(
                ref_state.get(
                    "startup_route_stage_id",
                    jp.asarray(4, dtype=jp.int32),
                ),
                dtype=jp.int32,
            )
            startup_route_transition_reason = jp.asarray(
                ref_state.get(
                    "startup_route_transition_reason",
                    jp.asarray(0, dtype=jp.int32),
                ),
                dtype=jp.int32,
            )
        else:
            ref_speed_scale, ref_phase_scale, ref_overspeed = _loc_ref_brake_scales(
                velocity_cmd=velocity_cmd,
                forward_vel_h=root_vel_h.linear_xyz[0],
                pitch_rad=pitch_reset,
                overspeed_deadband=self._loc_ref_overspeed_deadband,
                overspeed_brake_gain=self._loc_ref_overspeed_brake_gain,
                overspeed_phase_slowdown_gain=self._loc_ref_overspeed_phase_slowdown_gain,
                overspeed_phase_min_scale=self._loc_ref_overspeed_phase_min_scale,
                pitch_brake_start_rad=self._loc_ref_pitch_brake_start_rad,
                pitch_brake_gain=self._loc_ref_pitch_brake_gain,
            )
            ref_support_health, ref_support_instability = _loc_ref_support_health(
                velocity_cmd=velocity_cmd,
                forward_vel_h=root_vel_h.linear_xyz[0],
                pitch_rad=pitch_reset,
                pitch_rate_rad_s=root_vel_h.angular_xyz[1],
                overspeed_deadband=self._loc_ref_swing_x_brake_overspeed_deadband,
                pitch_start_rad=self._loc_ref_swing_x_brake_pitch_start_rad,
                pitch_rate_start_rad_s=self._loc_ref_support_pitch_rate_start_rad_s,
                health_gain=self._loc_ref_support_health_gain,
            )
            ref_state, ref_phase_time, ref_stance_foot, ref_switch_count = _step_walking_reference_jax(
                phase_time_s=jp.zeros((), dtype=jp.float32),
                stance_foot_id=jp.asarray(REF_LEFT_STANCE, dtype=jp.int32),
                stance_switch_count=jp.zeros((), dtype=jp.int32),
                forward_speed_mps=velocity_cmd,
                com_position_stance_frame=jp.asarray(
                    [-stance_pos_h[0], -stance_pos_h[1]], dtype=jp.float32
                ),
                com_velocity_stance_frame=jp.asarray(
                    [root_vel_h.linear_xyz[0], root_vel_h.linear_xyz[1]], dtype=jp.float32
                ),
                dt_s=self.dt,
                cfg=self._walking_ref_cfg,
                dcm_placement_gain=jp.asarray(
                    getattr(self._config.env, "loc_ref_dcm_placement_gain", 1.0),
                    dtype=jp.float32,
                ),
                phase_scale=ref_phase_scale,
                speed_scale=ref_speed_scale,
                support_health=ref_support_health,
                support_release_phase_start=self._loc_ref_support_release_phase_start,
                support_foothold_min_scale=self._loc_ref_support_foothold_min_scale,
                support_swing_progress_min_scale=self._loc_ref_support_swing_progress_min_scale,
                support_phase_min_scale=self._loc_ref_support_phase_min_scale,
            )
            ref_progression_permission = ref_support_health
            ref_mode_id = jp.asarray(int(WalkingRefV2Mode.SUPPORT_STABILIZE), dtype=jp.int32)
            ref_mode_time = jp.asarray(0.0, dtype=jp.float32)
            ref_mode_id_dbg = ref_mode_id.astype(jp.float32)
            startup_route_progress = jp.asarray(1.0, dtype=jp.float32)
            startup_route_ceiling = jp.asarray(1.0, dtype=jp.float32)
            startup_route_stage_id = jp.asarray(4, dtype=jp.int32)
            startup_route_transition_reason = jp.asarray(0, dtype=jp.int32)
        loc_ref_phase_sin = ref_state["gait_phase_sin"]
        loc_ref_phase_cos = ref_state["gait_phase_cos"]
        loc_ref_phase_progress = ref_state["phase_progress"]
        loc_ref_next_foothold = ref_state["next_foothold"]
        loc_ref_swing_pos = ref_state["swing_pos"]
        loc_ref_swing_vel = ref_state["swing_vel"]
        loc_ref_pelvis_height = ref_state["pelvis_height"]
        loc_ref_pelvis_roll = ref_state["pelvis_roll"]
        loc_ref_pelvis_pitch = ref_state["pelvis_pitch"]
        loc_ref_history = _update_loc_ref_history(
            jp.zeros((STANCE_REF_HISTORY_LEN * 2,), dtype=jp.float32),
            loc_ref_phase_sin,
            loc_ref_phase_cos,
        )
        (
            nominal_q_ref,
            left_ref_reachable,
            right_ref_reachable,
            nominal_swing_x_target,
            nominal_pelvis_pitch_target,
            nominal_stance_sagittal_target,
            nominal_swing_x_scale,
            nominal_pelvis_pitch_scale,
            nominal_support_gate_active,
        ) = (
            self._compute_nominal_q_ref_from_loc_ref(
                stance_foot_id=ref_stance_foot,
                swing_pos=loc_ref_swing_pos,
                swing_vel=loc_ref_swing_vel,
                pelvis_height=loc_ref_pelvis_height,
                pelvis_roll=loc_ref_pelvis_roll,
                pelvis_pitch=loc_ref_pelvis_pitch,
                left_foot_pos_h=left_foot_h,
                right_foot_pos_h=right_foot_h,
                root_pitch=pitch_reset,
                root_pitch_rate=root_vel_h.angular_xyz[1],
                forward_vel_h=root_vel_h.linear_xyz[0],
                velocity_cmd=velocity_cmd,
                support_health=ref_support_health,
                mode_id=ref_mode_id,
            )
        )
        nominal_q_ref = self._apply_startup_support_rate_limiter(
            nominal_q_ref=nominal_q_ref,
            prev_nominal_q_ref=jp.asarray(
                qpos[self._actuator_qpos_addrs],
                dtype=jp.float32,
            ),
            stance_foot_id=ref_stance_foot,
            mode_id=ref_mode_id,
            mode_time_s=ref_mode_time,
        )
        nominal_swing_y_target = jp.asarray(loc_ref_swing_pos[1], dtype=jp.float32)
        nominal_pelvis_roll_target = jp.asarray(loc_ref_pelvis_roll, dtype=jp.float32)
        if self._idx_left_hip_roll >= 0:
            nominal_left_hip_roll_target = jp.asarray(
                nominal_q_ref[self._idx_left_hip_roll], dtype=jp.float32
            )
        else:
            nominal_left_hip_roll_target = jp.zeros((), dtype=jp.float32)
        if self._idx_right_hip_roll >= 0:
            nominal_right_hip_roll_target = jp.asarray(
                nominal_q_ref[self._idx_right_hip_roll], dtype=jp.float32
            )
        else:
            nominal_right_hip_roll_target = jp.zeros((), dtype=jp.float32)
        left_foot_y_reset = jp.asarray(left_foot_h[1], dtype=jp.float32)
        right_foot_y_reset = jp.asarray(right_foot_h[1], dtype=jp.float32)
        support_width_y_actual_reset = jp.abs(right_foot_y_reset - left_foot_y_reset)
        support_width_y_nominal_reset = jp.abs(
            jp.asarray(loc_ref_next_foothold[1], dtype=jp.float32)
        )
        support_width_y_commanded_reset = jp.abs(nominal_swing_y_target)
        base_support_y_reset = jp.asarray(loc_ref_next_foothold[1], dtype=jp.float32)
        lateral_release_y_reset = nominal_swing_y_target - base_support_y_reset
        swing_pos_reset, _ = _swing_state_in_stance_frame(
            left_foot_pos_h=left_foot_h,
            right_foot_pos_h=right_foot_h,
            left_foot_vel_h=jp.zeros((3,), dtype=jp.float32),
            right_foot_vel_h=jp.zeros((3,), dtype=jp.float32),
            stance_foot_id=ref_stance_foot,
        )

        obs = self._get_obs(
            data,
            default_action,
            velocity_cmd,
            step_count=jp.zeros((), dtype=jp.int32),
            loc_ref_phase_sin_cos=jp.asarray(
                [loc_ref_phase_sin, loc_ref_phase_cos], dtype=jp.float32
            ),
            loc_ref_stance_foot=jp.asarray(
                [ref_stance_foot.astype(jp.float32)], dtype=jp.float32
            ),
            loc_ref_next_foothold=loc_ref_next_foothold,
            loc_ref_swing_pos=loc_ref_swing_pos,
            loc_ref_swing_vel=loc_ref_swing_vel,
            loc_ref_pelvis_targets=jp.asarray(
                [loc_ref_pelvis_height, loc_ref_pelvis_roll, loc_ref_pelvis_pitch],
                dtype=jp.float32,
            ),
            loc_ref_history=loc_ref_history,
            signals=signals_override,
        )

        # Initial reward and done
        reward = jp.zeros(())
        done = jp.zeros(())

        # Cache root pose (used for height, pitch/roll, position, orientation)
        height = root_pose.height

        # v0.15.10: Keep reset forward reward semantics aligned with step reward.
        forward_reward = compute_zero_baseline_forward_reward(
            forward_vel=jp.zeros(()),
            velocity_cmd=velocity_cmd,
            scale=self._config.reward_weights.forward_velocity_scale,
            velocity_cmd_min=self._config.reward_weights.velocity_cmd_min,
        )
        _, pitch, _ = root_pose.euler_angles()
        forward_upright_gate = compute_smooth_upright_gate(
            pitch=pitch,
            pitch_rate=jp.zeros(()),
            upright_pitch_limit=jp.asarray(
                getattr(self._config.reward_weights, "dense_progress_upright_pitch", 0.25),
                dtype=jp.float32,
            ),
            upright_pitch_rate_limit=jp.asarray(
                getattr(self._config.reward_weights, "dense_progress_upright_pitch_rate", 0.9),
                dtype=jp.float32,
            ),
            upright_gate_sharpness=jp.asarray(
                getattr(self._config.reward_weights, "dense_progress_upright_sharpness", 10.0),
                dtype=jp.float32,
            ),
        )
        cmd_gate_fwd = compute_command_active_gate(
            velocity_cmd, getattr(self._config.reward_weights, "velocity_cmd_min", 0.2)
        )
        forward_gate_strength = jp.asarray(
            getattr(self._config.reward_weights, "forward_upright_gate_strength", 1.0),
            dtype=jp.float32,
        )
        forward_reward = forward_reward * (
            (1.0 - cmd_gate_fwd)
            + cmd_gate_fwd
            * ((1.0 - forward_gate_strength) + forward_gate_strength * forward_upright_gate)
        )

        # Healthy reward: robot is upright at reset
        healthy_reward = jp.where(
            (height > self._config.env.min_height)
            & (height < self._config.env.max_height),
            1.0,
            0.0,
        )

        # Action rate: no previous action, so no penalty
        action_rate = jp.zeros(())

        # v0.10.1: Initialize all metric keys to match step() for JAX pytree compatibility
        # At reset, robot is stationary so most values are zero
        roll, pitch, _ = root_pose.euler_angles()
        left_force, right_force = self._cal.get_aggregated_foot_contacts(data)
        foot_switches = self._cal.get_foot_switches(
            data, threshold=self._config.env.foot_switch_threshold
        )
        contact_threshold = self._config.env.contact_threshold_force
        left_loaded = left_force > contact_threshold
        right_loaded = right_force > contact_threshold

        # Compute total reward with weights (from config)
        weights = self._config.reward_weights
        total_reward = (
            weights.tracking_lin_vel * forward_reward
            + weights.base_height * healthy_reward
            + weights.action_rate * action_rate
        )

        # This is the single source of truth for env metrics keys
        metrics = get_initial_env_metrics_jax(
            velocity_cmd=velocity_cmd,
            height=height,
            pitch=pitch,
            roll=roll,
            left_force=left_force,
            right_force=right_force,
            left_toe_switch=foot_switches[0],
            left_heel_switch=foot_switches[1],
            right_toe_switch=foot_switches[2],
            right_heel_switch=foot_switches[3],
            forward_reward=forward_reward,
            healthy_reward=healthy_reward,
            action_rate=action_rate,
            total_reward=total_reward,
        )

        # Info - use typed WildRobotInfo under "wr" namespace
        # This separates algorithm-critical fields from wrapper-injected fields
        left_foot_pos, right_foot_pos = self._cal.get_foot_positions(
            data, normalize=False
        )
        critic_obs = self._get_privileged_critic_obs(
            data, jp.zeros((), dtype=jp.int32), push_schedule
        )
        wr_info = WildRobotInfo(
            step_count=jp.zeros(()),
            prev_action=default_action,
            pending_action=default_action,
            truncated=jp.zeros(()),  # No truncation at reset
            velocity_cmd=velocity_cmd,  # Target velocity for this episode
            prev_root_pos=root_pose.position,  # (3,)
            prev_root_quat=root_pose.orientation,  # (4,) wxyz
            prev_left_foot_pos=left_foot_pos,
            prev_right_foot_pos=right_foot_pos,
            imu_quat_hist=imu_quat_hist,
            imu_gyro_hist=imu_gyro_hist,
            foot_contacts=self._cal.get_geom_based_foot_contacts(
                data, normalize=True
            ),  # For AMP
            root_height=root_pose.height,  # Actual root height for AMP features
            prev_left_loaded=left_loaded.astype(jp.float32),
            prev_right_loaded=right_loaded.astype(jp.float32),
            cycle_start_forward_x=jp.zeros(()),
            last_touchdown_root_pos=root_pose.position,
            last_touchdown_foot=jp.asarray(-1, dtype=jp.int32),
            critic_obs=critic_obs,
            loc_ref_phase_time=ref_phase_time,
            loc_ref_stance_foot=ref_stance_foot.astype(jp.int32),
            loc_ref_switch_count=ref_switch_count.astype(jp.int32),
            loc_ref_mode_id=ref_mode_id.astype(jp.int32),
            loc_ref_mode_time=ref_mode_time,
            loc_ref_startup_route_progress=startup_route_progress,
            loc_ref_startup_route_ceiling=startup_route_ceiling,
            loc_ref_startup_route_stage_id=startup_route_stage_id,
            loc_ref_startup_route_transition_reason=startup_route_transition_reason,
            loc_ref_gait_phase_sin=loc_ref_phase_sin,
            loc_ref_gait_phase_cos=loc_ref_phase_cos,
            loc_ref_next_foothold=loc_ref_next_foothold,
            loc_ref_swing_pos=loc_ref_swing_pos,
            loc_ref_swing_vel=loc_ref_swing_vel,
            loc_ref_pelvis_height=loc_ref_pelvis_height,
            loc_ref_pelvis_roll=loc_ref_pelvis_roll,
            loc_ref_pelvis_pitch=loc_ref_pelvis_pitch,
            loc_ref_history=loc_ref_history,
            nominal_q_ref=nominal_q_ref,
            push_schedule=push_schedule,
            # M3 FSM state: initialised to STANCE (all zeros)
            fsm_phase=jp.zeros((), dtype=jp.int32),
            fsm_swing_foot=jp.zeros((), dtype=jp.int32),
            fsm_phase_ticks=jp.zeros((), dtype=jp.int32),
            fsm_frozen_tx=jp.zeros((), dtype=jp.float32),
            fsm_frozen_ty=jp.zeros((), dtype=jp.float32),
            fsm_swing_sx=jp.zeros((), dtype=jp.float32),
            fsm_swing_sy=jp.zeros((), dtype=jp.float32),
            fsm_touch_hold=jp.zeros((), dtype=jp.int32),
            fsm_trigger_hold=jp.zeros((), dtype=jp.int32),
            # v0.17.1+: Recovery metrics state
            recovery_active=jp.asarray(False, dtype=jp.bool_),
            recovery_age=jp.zeros((), dtype=jp.int32),
            recovery_last_support_foot=jp.asarray(-1, dtype=jp.int32),
            recovery_first_liftoff_recorded=jp.asarray(False, dtype=jp.bool_),
            recovery_first_liftoff_latency=jp.zeros((), dtype=jp.int32),
            recovery_first_touchdown_recorded=jp.asarray(False, dtype=jp.bool_),
            recovery_first_step_latency=jp.zeros((), dtype=jp.int32),
            recovery_first_touchdown_age=jp.asarray(-1, dtype=jp.int32),
            recovery_visible_step_recorded=jp.asarray(False, dtype=jp.bool_),
            recovery_touchdown_count=jp.zeros((), dtype=jp.int32),
            recovery_support_foot_changes=jp.zeros((), dtype=jp.int32),
            recovery_pitch_rate_at_push_end=jp.zeros((), dtype=jp.float32),
            recovery_pitch_rate_at_touchdown=jp.zeros((), dtype=jp.float32),
            recovery_pitch_rate_after_10t=jp.zeros((), dtype=jp.float32),
            recovery_capture_error_at_push_end=jp.zeros((), dtype=jp.float32),
            recovery_capture_error_at_touchdown=jp.zeros((), dtype=jp.float32),
            recovery_capture_error_after_10t=jp.zeros((), dtype=jp.float32),
            recovery_first_step_dx=jp.zeros((), dtype=jp.float32),
            recovery_first_step_dy=jp.zeros((), dtype=jp.float32),
            recovery_first_step_target_err_x=jp.zeros((), dtype=jp.float32),
            recovery_first_step_target_err_y=jp.zeros((), dtype=jp.float32),
            recovery_min_height=jp.zeros((), dtype=jp.float32),
            recovery_max_knee_flex=jp.zeros((), dtype=jp.float32),
            teacher_active=jp.zeros((), dtype=jp.float32),
            teacher_step_required_soft=jp.zeros((), dtype=jp.float32),
            teacher_step_required_hard=jp.zeros((), dtype=jp.float32),
            teacher_swing_foot=jp.asarray(-1, dtype=jp.int32),
            teacher_target_step_x=jp.zeros((), dtype=jp.float32),
            teacher_target_step_y=jp.zeros((), dtype=jp.float32),
            teacher_target_reachable=jp.zeros((), dtype=jp.float32),
            mpc_planner_active=jp.zeros((), dtype=jp.float32),
            mpc_controller_active=jp.zeros((), dtype=jp.float32),
            mpc_target_com_x=jp.zeros((), dtype=jp.float32),
            mpc_target_com_y=jp.zeros((), dtype=jp.float32),
            mpc_target_step_x=jp.zeros((), dtype=jp.float32),
            mpc_target_step_y=jp.zeros((), dtype=jp.float32),
            mpc_support_state=jp.zeros((), dtype=jp.float32),
            mpc_step_requested=jp.zeros((), dtype=jp.float32),
            domain_rand_friction_scale=domain_rand_params["friction_scale"],
            domain_rand_mass_scales=domain_rand_params["mass_scales"],
            domain_rand_kp_scales=domain_rand_params["kp_scales"],
            domain_rand_frictionloss_scales=domain_rand_params["frictionloss_scales"],
            domain_rand_joint_offsets=domain_rand_params["joint_offsets"],
        )
        metrics["tracking/loc_ref_left_reachable"] = left_ref_reachable
        metrics["tracking/loc_ref_right_reachable"] = right_ref_reachable
        metrics["tracking/loc_ref_phase_progress"] = jp.asarray(
            loc_ref_phase_progress, dtype=jp.float32
        )
        metrics["tracking/loc_ref_stance_foot"] = ref_stance_foot.astype(jp.float32)
        metrics["tracking/loc_ref_mode_id"] = ref_mode_id.astype(jp.float32)
        metrics["tracking/loc_ref_progression_permission"] = ref_progression_permission
        metrics["tracking/nominal_q_abs_mean"] = jp.mean(jp.abs(nominal_q_ref))
        metrics["tracking/residual_q_abs_mean"] = jp.zeros((), dtype=jp.float32)
        metrics["tracking/residual_q_abs_max"] = jp.zeros((), dtype=jp.float32)
        if self._loc_ref_version == "v1":
            metrics["debug/loc_ref_speed_scale"] = ref_speed_scale
            metrics["debug/loc_ref_phase_scale"] = ref_phase_scale
            metrics["debug/loc_ref_swing_x_scale_active"] = ref_speed_scale
            metrics["debug/loc_ref_phase_scale_active"] = ref_phase_scale
        else:
            metrics["debug/loc_ref_speed_scale"] = jp.zeros((), dtype=jp.float32)
            metrics["debug/loc_ref_phase_scale"] = jp.zeros((), dtype=jp.float32)
            metrics["debug/loc_ref_swing_x_scale_active"] = nominal_swing_x_scale
            metrics["debug/loc_ref_phase_scale_active"] = ref_progression_permission
        metrics["debug/loc_ref_overspeed"] = ref_overspeed
        metrics["debug/loc_ref_support_health"] = ref_support_health
        metrics["debug/loc_ref_support_instability"] = ref_support_instability
        metrics["debug/loc_ref_startup_route_progress"] = startup_route_progress
        metrics["debug/loc_ref_startup_route_ceiling"] = startup_route_ceiling
        metrics["debug/loc_ref_startup_route_stage_id"] = startup_route_stage_id.astype(jp.float32)
        metrics["debug/loc_ref_startup_route_transition_reason"] = startup_route_transition_reason.astype(
            jp.float32
        )
        metrics["debug/m3_pelvis_orientation_error"] = jp.zeros((), dtype=jp.float32)
        metrics["debug/m3_pelvis_height_error"] = jp.zeros((), dtype=jp.float32)
        metrics["debug/m3_swing_pos_error"] = jp.zeros((), dtype=jp.float32)
        metrics["debug/m3_swing_vel_error"] = jp.zeros((), dtype=jp.float32)
        metrics["debug/m3_foothold_error"] = jp.zeros((), dtype=jp.float32)
        metrics["debug/m3_impact_force"] = jp.zeros((), dtype=jp.float32)
        metrics["debug/loc_ref_swing_x_target"] = nominal_swing_x_target
        metrics["debug/loc_ref_swing_x_actual"] = jp.asarray(
            swing_pos_reset[0], dtype=jp.float32
        )
        metrics["debug/loc_ref_swing_x_error"] = jp.asarray(
            swing_pos_reset[0] - nominal_swing_x_target, dtype=jp.float32
        )
        metrics["debug/loc_ref_swing_y_target"] = nominal_swing_y_target
        metrics["debug/loc_ref_swing_y_actual"] = jp.asarray(
            swing_pos_reset[1], dtype=jp.float32
        )
        metrics["debug/loc_ref_swing_y_error"] = jp.asarray(
            swing_pos_reset[1] - nominal_swing_y_target, dtype=jp.float32
        )
        metrics["debug/loc_ref_left_foot_y_actual"] = left_foot_y_reset
        metrics["debug/loc_ref_right_foot_y_actual"] = right_foot_y_reset
        metrics["debug/loc_ref_support_width_y_actual"] = support_width_y_actual_reset
        metrics["debug/loc_ref_support_width_y_nominal"] = support_width_y_nominal_reset
        metrics["debug/loc_ref_support_width_y_commanded"] = support_width_y_commanded_reset
        metrics["debug/loc_ref_base_support_y"] = base_support_y_reset
        metrics["debug/loc_ref_lateral_release_y"] = lateral_release_y_reset
        metrics["debug/loc_ref_pelvis_roll_target"] = nominal_pelvis_roll_target
        metrics["debug/loc_ref_hip_roll_left_target"] = nominal_left_hip_roll_target
        metrics["debug/loc_ref_hip_roll_right_target"] = nominal_right_hip_roll_target
        metrics["debug/loc_ref_pelvis_pitch_target"] = nominal_pelvis_pitch_target
        metrics["debug/loc_ref_root_pitch"] = jp.asarray(pitch_reset, dtype=jp.float32)
        metrics["debug/loc_ref_root_pitch_rate"] = jp.asarray(
            root_vel_h.angular_xyz[1], dtype=jp.float32
        )
        metrics["debug/loc_ref_stance_sagittal_target"] = nominal_stance_sagittal_target
        metrics["debug/loc_ref_swing_x_scale"] = nominal_swing_x_scale
        metrics["debug/loc_ref_pelvis_pitch_scale"] = nominal_pelvis_pitch_scale
        metrics["debug/loc_ref_support_gate_active"] = nominal_support_gate_active
        metrics["debug/loc_ref_hybrid_mode_id"] = ref_mode_id_dbg
        metrics["debug/loc_ref_progression_permission"] = ref_progression_permission

        # Merge: base_info (wrapper fields) + wr namespace (env fields)
        if base_info is not None:
            info = dict(base_info)
            info[WR_INFO_KEY] = wr_info
        else:
            info = {WR_INFO_KEY: wr_info}

        # Build packed metrics vector and store in metrics dict
        # This follows the same pattern as info[WR_INFO_KEY]
        metrics_vec = build_metrics_vec(metrics)
        metrics[METRICS_VEC_KEY] = metrics_vec

        return WildRobotEnvState(
            data=data,
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics,
            info=info,
            pipeline_state=data,
            rng=next_rng,
        )

    def reset(self, rng: jax.Array) -> WildRobotEnvState:
        """Reset environment to initial state.

        Args:
            rng: JAX random key for randomization.

        Returns:
            Initial WildRobotEnvState.
        """
        rng, key1, key2, key3, key4 = jax.random.split(rng, 5)

        # Sample velocity command
        velocity_cmd = jax.random.uniform(
            key1,
            shape=(),
            minval=self._config.env.min_velocity,
            maxval=self._config.env.max_velocity,
        )

        domain_rand_params = self._sample_domain_rand_params(key4)

        # Use initial qpos from model (keyframe or qpos0)
        qpos = self._init_qpos.copy()

        # Add small random noise to joint positions (not floating base)
        joint_noise = jax.random.uniform(
            key2, shape=self._default_joint_qpos.shape, minval=-0.05, maxval=0.05
        )
        joint_qpos = (
            self._default_joint_qpos
            + joint_noise
            + domain_rand_params["joint_offsets"]
        )
        joint_qpos = jp.clip(joint_qpos, self._joint_range_mins, self._joint_range_maxs)
        qpos = qpos.at[self._actuator_qpos_addrs].set(joint_qpos)

        schedule = sample_push_schedule(key3, self._config.env, self._push_body_ids)

        return self._make_initial_state(
            qpos,
            velocity_cmd,
            push_schedule=schedule,
            domain_rand_params=domain_rand_params,
        )

    def step(
        self,
        state: WildRobotEnvState,
        action: jax.Array,
        disable_pushes: bool = False,
    ) -> WildRobotEnvState:
        """Step environment forward with auto-reset on termination.

        Args:
            state: Current environment state.
            action: Action to apply (joint position targets).

        Returns:
            New WildRobotEnvState. If episode terminated, state is auto-reset
            but done=True is still returned so the algorithm knows.
        """
        # Read from typed WildRobotInfo namespace (fail loudly if missing)
        wr = state.info[WR_INFO_KEY]
        velocity_cmd = wr.velocity_cmd
        step_count = wr.step_count
        prev_action = wr.prev_action
        pending_action = wr.pending_action

        policy_action = action
        root_pose_pre = self._cal.get_root_pose(state.data)
        root_vel_pre_h = self._cal.get_root_velocity(
            state.data, frame=CoordinateFrame.HEADING_LOCAL
        )
        _, pitch_pre, _ = root_pose_pre.euler_angles()
        left_foot_pre_h, right_foot_pre_h = self._cal.get_foot_positions(
            state.data, normalize=False, frame=CoordinateFrame.HEADING_LOCAL
        )
        left_force_pre, right_force_pre = self._cal.get_aggregated_foot_contacts(state.data)
        left_loaded_pre = left_force_pre > self._config.env.contact_threshold_force
        right_loaded_pre = right_force_pre > self._config.env.contact_threshold_force
        stance_pre_is_left = jp.asarray(wr.loc_ref_stance_foot, dtype=jp.int32) == jp.asarray(
            REF_LEFT_STANCE, dtype=jp.int32
        )
        actuated_qpos_pre = jp.asarray(
            state.data.qpos[self._actuator_qpos_addrs], dtype=jp.float32
        )
        stance_knee_err_pre, stance_ankle_err_pre = self._stance_leg_tracking_errors(
            actual_q=actuated_qpos_pre,
            target_q=jp.asarray(wr.nominal_q_ref, dtype=jp.float32),
            stance_foot_id=wr.loc_ref_stance_foot,
        )
        stance_pre_pos_h = jp.where(stance_pre_is_left, left_foot_pre_h, right_foot_pre_h)
        if self._loc_ref_version == "v2":
            (
                ref_state,
                ref_phase_time,
                ref_stance_foot,
                ref_switch_count,
                ref_mode_id,
                ref_mode_time,
                ref_support_health,
                ref_support_instability,
                ref_progression_permission,
            ) = step_reference_v2_jax(
                config=self._walking_ref_v2_cfg,
                phase_time_s=wr.loc_ref_phase_time,
                stance_foot_id=wr.loc_ref_stance_foot,
                stance_switch_count=wr.loc_ref_switch_count,
                mode_id=wr.loc_ref_mode_id,
                mode_time_s=wr.loc_ref_mode_time,
                forward_speed_mps=velocity_cmd,
                com_position_stance_frame=jp.asarray(
                    [-stance_pre_pos_h[0], -stance_pre_pos_h[1]], dtype=jp.float32
                ),
                com_velocity_stance_frame=jp.asarray(
                    [root_vel_pre_h.linear_xyz[0], root_vel_pre_h.linear_xyz[1]], dtype=jp.float32
                ),
                root_pitch_rad=pitch_pre,
                root_pitch_rate_rad_s=root_vel_pre_h.angular_xyz[1],
                root_height_m=root_pose_pre.height,
                left_foot_loaded=left_loaded_pre,
                right_foot_loaded=right_loaded_pre,
                dt_s=self.dt,
                stance_knee_tracking_error_rad=stance_knee_err_pre,
                stance_ankle_tracking_error_rad=stance_ankle_err_pre,
                startup_route_progress_prev=wr.loc_ref_startup_route_progress,
                startup_route_ceiling_prev=wr.loc_ref_startup_route_ceiling,
                startup_route_stage_id_prev=wr.loc_ref_startup_route_stage_id,
                startup_route_transition_reason_prev=wr.loc_ref_startup_route_transition_reason,
            )
            ref_overspeed = jp.maximum(
                jp.asarray(root_vel_pre_h.linear_xyz[0], dtype=jp.float32)
                - jp.maximum(jp.asarray(velocity_cmd, dtype=jp.float32), 0.0)
                - jp.asarray(self._loc_ref_swing_x_brake_overspeed_deadband, dtype=jp.float32),
                0.0,
            ).astype(jp.float32)
            ref_mode_id_dbg = ref_mode_id.astype(jp.float32)
            startup_route_progress = jp.asarray(
                ref_state.get("startup_route_progress", jp.asarray(1.0, dtype=jp.float32)),
                dtype=jp.float32,
            )
            startup_route_ceiling = jp.asarray(
                ref_state.get("startup_route_ceiling", jp.asarray(1.0, dtype=jp.float32)),
                dtype=jp.float32,
            )
            startup_route_stage_id = jp.asarray(
                ref_state.get(
                    "startup_route_stage_id",
                    jp.asarray(4, dtype=jp.int32),
                ),
                dtype=jp.int32,
            )
            startup_route_transition_reason = jp.asarray(
                ref_state.get(
                    "startup_route_transition_reason",
                    jp.asarray(0, dtype=jp.int32),
                ),
                dtype=jp.int32,
            )
        else:
            ref_speed_scale, ref_phase_scale, ref_overspeed = _loc_ref_brake_scales(
                velocity_cmd=velocity_cmd,
                forward_vel_h=root_vel_pre_h.linear_xyz[0],
                pitch_rad=pitch_pre,
                overspeed_deadband=self._loc_ref_overspeed_deadband,
                overspeed_brake_gain=self._loc_ref_overspeed_brake_gain,
                overspeed_phase_slowdown_gain=self._loc_ref_overspeed_phase_slowdown_gain,
                overspeed_phase_min_scale=self._loc_ref_overspeed_phase_min_scale,
                pitch_brake_start_rad=self._loc_ref_pitch_brake_start_rad,
                pitch_brake_gain=self._loc_ref_pitch_brake_gain,
            )
            ref_support_health, ref_support_instability = _loc_ref_support_health(
                velocity_cmd=velocity_cmd,
                forward_vel_h=root_vel_pre_h.linear_xyz[0],
                pitch_rad=pitch_pre,
                pitch_rate_rad_s=root_vel_pre_h.angular_xyz[1],
                overspeed_deadband=self._loc_ref_swing_x_brake_overspeed_deadband,
                pitch_start_rad=self._loc_ref_swing_x_brake_pitch_start_rad,
                pitch_rate_start_rad_s=self._loc_ref_support_pitch_rate_start_rad_s,
                health_gain=self._loc_ref_support_health_gain,
            )
            ref_state, ref_phase_time, ref_stance_foot, ref_switch_count = _step_walking_reference_jax(
                phase_time_s=wr.loc_ref_phase_time,
                stance_foot_id=wr.loc_ref_stance_foot,
                stance_switch_count=wr.loc_ref_switch_count,
                forward_speed_mps=velocity_cmd,
                com_position_stance_frame=jp.asarray(
                    [-stance_pre_pos_h[0], -stance_pre_pos_h[1]], dtype=jp.float32
                ),
                com_velocity_stance_frame=jp.asarray(
                    [root_vel_pre_h.linear_xyz[0], root_vel_pre_h.linear_xyz[1]], dtype=jp.float32
                ),
                dt_s=self.dt,
                cfg=self._walking_ref_cfg,
                dcm_placement_gain=jp.asarray(
                    getattr(self._config.env, "loc_ref_dcm_placement_gain", 1.0),
                    dtype=jp.float32,
                ),
                phase_scale=ref_phase_scale,
                speed_scale=ref_speed_scale,
                support_health=ref_support_health,
                support_release_phase_start=self._loc_ref_support_release_phase_start,
                support_foothold_min_scale=self._loc_ref_support_foothold_min_scale,
                support_swing_progress_min_scale=self._loc_ref_support_swing_progress_min_scale,
                support_phase_min_scale=self._loc_ref_support_phase_min_scale,
            )
            ref_progression_permission = ref_support_health
            ref_mode_id = jp.asarray(wr.loc_ref_mode_id, dtype=jp.int32)
            ref_mode_time = jp.asarray(wr.loc_ref_mode_time, dtype=jp.float32)
            ref_mode_id_dbg = ref_mode_id.astype(jp.float32)
            startup_route_progress = jp.asarray(1.0, dtype=jp.float32)
            startup_route_ceiling = jp.asarray(1.0, dtype=jp.float32)
            startup_route_stage_id = jp.asarray(4, dtype=jp.int32)
            startup_route_transition_reason = jp.asarray(0, dtype=jp.int32)
        loc_ref_phase_sin = ref_state["gait_phase_sin"]
        loc_ref_phase_cos = ref_state["gait_phase_cos"]
        loc_ref_phase_progress = ref_state["phase_progress"]
        loc_ref_next_foothold = ref_state["next_foothold"]
        loc_ref_swing_pos = ref_state["swing_pos"]
        loc_ref_swing_vel = ref_state["swing_vel"]
        loc_ref_pelvis_height = ref_state["pelvis_height"]
        loc_ref_pelvis_roll = ref_state["pelvis_roll"]
        loc_ref_pelvis_pitch = ref_state["pelvis_pitch"]
        loc_ref_history = _update_loc_ref_history(
            wr.loc_ref_history, loc_ref_phase_sin, loc_ref_phase_cos
        )
        (
            nominal_q_ref,
            left_ref_reachable,
            right_ref_reachable,
            nominal_swing_x_target,
            nominal_pelvis_pitch_target,
            nominal_stance_sagittal_target,
            nominal_swing_x_scale,
            nominal_pelvis_pitch_scale,
            nominal_support_gate_active,
        ) = (
            self._compute_nominal_q_ref_from_loc_ref(
                stance_foot_id=ref_stance_foot,
                swing_pos=loc_ref_swing_pos,
                swing_vel=loc_ref_swing_vel,
                pelvis_height=loc_ref_pelvis_height,
                pelvis_roll=loc_ref_pelvis_roll,
                pelvis_pitch=loc_ref_pelvis_pitch,
                left_foot_pos_h=left_foot_pre_h,
                right_foot_pos_h=right_foot_pre_h,
                root_pitch=pitch_pre,
                root_pitch_rate=root_vel_pre_h.angular_xyz[1],
                forward_vel_h=root_vel_pre_h.linear_xyz[0],
                velocity_cmd=velocity_cmd,
                support_health=ref_support_health,
                mode_id=ref_mode_id,
            )
        )
        nominal_q_ref = self._apply_startup_support_rate_limiter(
            nominal_q_ref=nominal_q_ref,
            prev_nominal_q_ref=jp.asarray(wr.nominal_q_ref, dtype=jp.float32),
            stance_foot_id=ref_stance_foot,
            mode_id=ref_mode_id,
            mode_time_s=ref_mode_time,
        )
        nominal_swing_y_target = jp.asarray(loc_ref_swing_pos[1], dtype=jp.float32)
        nominal_pelvis_roll_target = jp.asarray(loc_ref_pelvis_roll, dtype=jp.float32)
        if self._idx_left_hip_roll >= 0:
            nominal_left_hip_roll_target = jp.asarray(
                nominal_q_ref[self._idx_left_hip_roll], dtype=jp.float32
            )
        else:
            nominal_left_hip_roll_target = jp.zeros((), dtype=jp.float32)
        if self._idx_right_hip_roll >= 0:
            nominal_right_hip_roll_target = jp.asarray(
                nominal_q_ref[self._idx_right_hip_roll], dtype=jp.float32
            )
        else:
            nominal_right_hip_roll_target = jp.zeros((), dtype=jp.float32)
        residual_delta_q = jp.zeros((self.action_size,), dtype=jp.float32)

        controller_stack = self._controller_stack
        mpc_debug = None

        if controller_stack == "mpc_standing":
            raw_action, mpc_debug = self._mpc_standing_compute_ctrl(state.data, policy_action)
            new_fsm = (
                wr.fsm_phase, wr.fsm_swing_foot, wr.fsm_phase_ticks,
                wr.fsm_frozen_tx, wr.fsm_frozen_ty, wr.fsm_swing_sx, wr.fsm_swing_sy,
                wr.fsm_touch_hold, wr.fsm_trigger_hold,
            )
        elif controller_stack == "ppo_teacher_standing":
            raw_action = policy_action
            new_fsm = (
                wr.fsm_phase, wr.fsm_swing_foot, wr.fsm_phase_ticks,
                wr.fsm_frozen_tx, wr.fsm_frozen_ty, wr.fsm_swing_sx, wr.fsm_swing_sy,
                wr.fsm_touch_hold, wr.fsm_trigger_hold,
            )
        elif bool(getattr(self._config.env, "fsm_enabled", False)):
            raw_action, new_fsm = self._fsm_compute_ctrl(state.data, wr, policy_action)
        elif bool(getattr(self._config.env, "base_ctrl_enabled", False)):
            raw_action = (
                self._mix_base_and_residual_action(state.data, policy_action)
            )
            # M3 not active: carry FSM state forward unchanged (all zeros at startup)
            new_fsm = (
                wr.fsm_phase, wr.fsm_swing_foot, wr.fsm_phase_ticks,
                wr.fsm_frozen_tx, wr.fsm_frozen_ty, wr.fsm_swing_sx, wr.fsm_swing_sy,
                wr.fsm_touch_hold, wr.fsm_trigger_hold,
            )
        else:
            if self._loc_ref_enabled:
                raw_action, residual_delta_q = self._compose_loc_ref_residual_action(
                    policy_action=policy_action,
                    nominal_q_ref=nominal_q_ref,
                )
            else:
                raw_action = policy_action
            new_fsm = (
                wr.fsm_phase, wr.fsm_swing_foot, wr.fsm_phase_ticks,
                wr.fsm_frozen_tx, wr.fsm_frozen_ty, wr.fsm_swing_sx, wr.fsm_swing_sy,
                wr.fsm_touch_hold, wr.fsm_trigger_hold,
            )
        policy_state = PolicyState(prev_action=pending_action)
        filtered_action, policy_state = postprocess_action(
            spec=self._policy_spec,
            state=policy_state,
            action_raw=raw_action,
        )
        applied_action = pending_action if self._action_delay_enabled else filtered_action
        applied_target_q = JaxCalibOps.action_to_ctrl(
            spec=self._policy_spec, action=applied_action
        ).astype(jp.float32)

        # =====================================================================
        # Apply action as control via CAL (v0.11.0)
        # =====================================================================
        # CAL applies symmetry correction (policy_action_sign) for left/right joints
        # This ensures same action values produce symmetric motion on both sides
        ctrl = JaxCalibOps.action_to_ctrl(spec=self._policy_spec, action=applied_action)
        data = state.data.replace(ctrl=ctrl)

        # Apply external push disturbance (JIT-safe boolean gating)
        push_enabled = jp.logical_and(
            jp.asarray(self._config.env.push_enabled, dtype=jp.bool_),
            jp.logical_not(jp.asarray(disable_pushes, dtype=jp.bool_)),
        )
        data = apply_push(data, wr.push_schedule, step_count, enabled=push_enabled)

        # Physics simulation (multiple substeps)
        randomized_model = self._get_randomized_mjx_model(wr)

        def substep_fn(data, _):
            return mjx.step(randomized_model, data), None

        n_substeps = int(self.dt / self.sim_dt)
        data, _ = jax.lax.scan(substep_fn, data, None, length=n_substeps)

        # Compute observation with finite-diff velocities
        # Use previous root pose from WildRobotInfo for FD velocity computation
        prev_root_pos = wr.prev_root_pos
        prev_root_quat = wr.prev_root_quat

        # IMU noise + latency handling (training-only). Use state.rng for determinism.
        raw_signals = self._signals_adapter.read(data)
        signals_override, new_imu_quat_hist, new_imu_gyro_hist, rng_after = _apply_imu_noise_and_delay(
            raw_signals, state.rng, self._config, wr.imu_quat_hist, wr.imu_gyro_hist
        )
        # Build observation using possibly delayed/noisy IMU signals
        obs = self._get_obs(
            data,
            applied_action,
            velocity_cmd,
            step_count=step_count + 1,
            prev_root_pos=prev_root_pos,
            prev_root_quat=prev_root_quat,
            loc_ref_phase_sin_cos=jp.asarray(
                [loc_ref_phase_sin, loc_ref_phase_cos], dtype=jp.float32
            ),
            loc_ref_stance_foot=jp.asarray(
                [ref_stance_foot.astype(jp.float32)], dtype=jp.float32
            ),
            loc_ref_next_foothold=loc_ref_next_foothold,
            loc_ref_swing_pos=loc_ref_swing_pos,
            loc_ref_swing_vel=loc_ref_swing_vel,
            loc_ref_pelvis_targets=jp.asarray(
                [loc_ref_pelvis_height, loc_ref_pelvis_roll, loc_ref_pelvis_pitch],
                dtype=jp.float32,
            ),
            loc_ref_history=loc_ref_history,
            signals=signals_override,
        )

        # Get previous foot positions for slip computation from WildRobotInfo
        prev_left_foot_pos = wr.prev_left_foot_pos
        prev_right_foot_pos = wr.prev_right_foot_pos
        prev_left_loaded = wr.prev_left_loaded
        prev_right_loaded = wr.prev_right_loaded

        current_push_active = (
            (step_count >= wr.push_schedule.start_step)
            & (step_count < wr.push_schedule.end_step)
            & push_enabled
        )

        # Compute reward
        reward, reward_components = self._get_reward(
            data,
            applied_action,
            policy_action,
            prev_action,
            velocity_cmd,
            prev_left_foot_pos,
            prev_right_foot_pos,
            prev_left_loaded,
            prev_right_loaded,
            wr.fsm_phase,
            wr.fsm_swing_foot,
            current_step_count=step_count + 1,
            cycle_start_forward_x=wr.cycle_start_forward_x,
            last_touchdown_root_pos=wr.last_touchdown_root_pos,
            last_touchdown_foot=wr.last_touchdown_foot,
            push_active=current_push_active,
            recovery_active=wr.recovery_active,
            recovery_first_touchdown_recorded=wr.recovery_first_touchdown_recorded,
            recovery_pitch_rate_at_touchdown=wr.recovery_pitch_rate_at_touchdown,
            recovery_capture_error_at_touchdown=wr.recovery_capture_error_at_touchdown,
            nominal_q_ref=nominal_q_ref,
            residual_delta_q=residual_delta_q,
            loc_ref_pelvis_height=loc_ref_pelvis_height,
            loc_ref_pelvis_roll=loc_ref_pelvis_roll,
            loc_ref_pelvis_pitch=loc_ref_pelvis_pitch,
            loc_ref_swing_pos=loc_ref_swing_pos,
            loc_ref_swing_vel=loc_ref_swing_vel,
            loc_ref_next_foothold=loc_ref_next_foothold,
            loc_ref_stance_foot=ref_stance_foot,
        )

        # Check termination (get done, terminated, truncated, and diagnostics)
        done, terminated, truncated, term_info = self._get_termination(
            data, step_count + 1
        )

        # Cache root pose and velocity (used for metrics and WildRobotInfo update)
        curr_root_pose = self._cal.get_root_pose(data)
        forward_vel, _, _ = self._cal.get_root_velocity(
            data, frame=CoordinateFrame.HEADING_LOCAL
        ).linear_xyz
        curr_root_vel_h = self._cal.get_root_velocity(
            data, frame=CoordinateFrame.HEADING_LOCAL
        )
        _, curr_pitch, _ = curr_root_pose.euler_angles()
        left_foot_post_h, right_foot_post_h = self._cal.get_foot_positions(
            data, normalize=False, frame=CoordinateFrame.HEADING_LOCAL
        )
        swing_pos_post, _ = _swing_state_in_stance_frame(
            left_foot_pos_h=left_foot_post_h,
            right_foot_pos_h=right_foot_post_h,
            left_foot_vel_h=jp.zeros((3,), dtype=jp.float32),
            right_foot_vel_h=jp.zeros((3,), dtype=jp.float32),
            stance_foot_id=ref_stance_foot,
        )
        left_foot_y_actual = jp.asarray(left_foot_post_h[1], dtype=jp.float32)
        right_foot_y_actual = jp.asarray(right_foot_post_h[1], dtype=jp.float32)
        support_width_y_actual = jp.abs(right_foot_y_actual - left_foot_y_actual)
        support_width_y_nominal = jp.abs(jp.asarray(loc_ref_next_foothold[1], dtype=jp.float32))
        support_width_y_commanded = jp.abs(nominal_swing_y_target)
        base_support_y = jp.asarray(loc_ref_next_foothold[1], dtype=jp.float32)
        lateral_release_y = nominal_swing_y_target - base_support_y
        _, _, _, cycle_start_forward_x = compute_cycle_progress_terms(
            cycle_progress_accum=wr.cycle_start_forward_x,
            forward_vel=forward_vel,
            dt=self.dt,
            current_step_count=step_count + 1,
            stride_steps=jp.asarray(self._config.env.clock_stride_period_steps, dtype=jp.int32),
            velocity_cmd=velocity_cmd,
            target_scale=jp.asarray(1.0, dtype=jp.float32),
            sigma=jp.asarray(1.0, dtype=jp.float32),
        )
        # v0.10.4: Store step count in metrics for episode length calculation
        # This gets preserved through auto-reset, unlike wr_info.step_count which resets
        episode_step_count = step_count + 1

        metrics = {
            "velocity_command": velocity_cmd,
            "height": curr_root_pose.height,
            "forward_velocity": forward_vel,
            "episode_step_count": episode_step_count,  # v0.10.4: For ep_len logging
            "reward/total": reward,
            **reward_components,
            # v0.10.2: Termination diagnostics
            **term_info,
            # v0.14.x: M3 base-controller FSM debug metrics
            "debug/bc_phase": new_fsm[0].astype(jp.float32),
            "debug/bc_in_swing": (new_fsm[0] == sc.SWING).astype(jp.float32),
            "debug/bc_in_recover": (new_fsm[0] == sc.TOUCHDOWN_RECOVER).astype(jp.float32),
            "debug/bc_swing_foot": new_fsm[1].astype(jp.float32),
            "debug/bc_phase_ticks": new_fsm[2].astype(jp.float32),
            "debug/mpc_planner_active": jp.zeros((), dtype=jp.float32),
            "debug/mpc_controller_active": jp.zeros((), dtype=jp.float32),
            "debug/mpc_target_com_x": jp.zeros((), dtype=jp.float32),
            "debug/mpc_target_com_y": jp.zeros((), dtype=jp.float32),
            "debug/mpc_target_step_x": jp.zeros((), dtype=jp.float32),
            "debug/mpc_target_step_y": jp.zeros((), dtype=jp.float32),
            "debug/mpc_support_state": jp.zeros((), dtype=jp.float32),
            "debug/mpc_step_requested": jp.zeros((), dtype=jp.float32),
            "tracking/cmd_vs_achieved_forward": jp.abs(forward_vel - velocity_cmd),
            "tracking/loc_ref_phase_progress": jp.asarray(
                loc_ref_phase_progress, dtype=jp.float32
            ),
            "tracking/loc_ref_stance_foot": ref_stance_foot.astype(jp.float32),
            "tracking/loc_ref_mode_id": ref_mode_id.astype(jp.float32),
            "tracking/loc_ref_progression_permission": ref_progression_permission,
            "tracking/nominal_q_abs_mean": jp.mean(jp.abs(nominal_q_ref)),
            "tracking/residual_q_abs_mean": jp.mean(jp.abs(residual_delta_q)),
            "tracking/residual_q_abs_max": jp.max(jp.abs(residual_delta_q)),
            "tracking/loc_ref_left_reachable": left_ref_reachable,
            "tracking/loc_ref_right_reachable": right_ref_reachable,
            "debug/loc_ref_overspeed": ref_overspeed,
            "debug/loc_ref_support_health": ref_support_health,
            "debug/loc_ref_support_instability": ref_support_instability,
            "debug/loc_ref_nominal_vs_applied_q_l1": jp.mean(
                jp.abs(applied_target_q - nominal_q_ref)
            ),
            "debug/loc_ref_applied_q_abs_mean": jp.mean(jp.abs(applied_target_q)),
            "debug/loc_ref_nominal_q_abs_mean": jp.mean(jp.abs(nominal_q_ref)),
            "debug/loc_ref_swing_x_target": nominal_swing_x_target,
            "debug/loc_ref_swing_x_actual": jp.asarray(swing_pos_post[0], dtype=jp.float32),
            "debug/loc_ref_swing_x_error": jp.asarray(
                swing_pos_post[0] - nominal_swing_x_target, dtype=jp.float32
            ),
            "debug/loc_ref_swing_y_target": nominal_swing_y_target,
            "debug/loc_ref_swing_y_actual": jp.asarray(swing_pos_post[1], dtype=jp.float32),
            "debug/loc_ref_swing_y_error": jp.asarray(
                swing_pos_post[1] - nominal_swing_y_target, dtype=jp.float32
            ),
            "debug/loc_ref_left_foot_y_actual": left_foot_y_actual,
            "debug/loc_ref_right_foot_y_actual": right_foot_y_actual,
            "debug/loc_ref_support_width_y_actual": support_width_y_actual,
            "debug/loc_ref_support_width_y_nominal": support_width_y_nominal,
            "debug/loc_ref_support_width_y_commanded": support_width_y_commanded,
            "debug/loc_ref_base_support_y": base_support_y,
            "debug/loc_ref_lateral_release_y": lateral_release_y,
            "debug/loc_ref_pelvis_roll_target": nominal_pelvis_roll_target,
            "debug/loc_ref_hip_roll_left_target": nominal_left_hip_roll_target,
            "debug/loc_ref_hip_roll_right_target": nominal_right_hip_roll_target,
            "debug/loc_ref_pelvis_pitch_target": nominal_pelvis_pitch_target,
            "debug/loc_ref_root_pitch": jp.asarray(curr_pitch, dtype=jp.float32),
            "debug/loc_ref_root_pitch_rate": jp.asarray(
                curr_root_vel_h.angular_xyz[1], dtype=jp.float32
            ),
            "debug/loc_ref_stance_sagittal_target": nominal_stance_sagittal_target,
            "debug/loc_ref_swing_x_scale": nominal_swing_x_scale,
            "debug/loc_ref_pelvis_pitch_scale": nominal_pelvis_pitch_scale,
            "debug/loc_ref_support_gate_active": nominal_support_gate_active,
            "debug/loc_ref_hybrid_mode_id": ref_mode_id_dbg,
            "debug/loc_ref_progression_permission": ref_progression_permission,
            "debug/loc_ref_startup_route_progress": startup_route_progress,
            "debug/loc_ref_startup_route_ceiling": startup_route_ceiling,
            "debug/loc_ref_startup_route_stage_id": startup_route_stage_id.astype(jp.float32),
            "debug/loc_ref_startup_route_transition_reason": startup_route_transition_reason.astype(
                jp.float32
            ),
        }
        if self._loc_ref_version == "v1":
            metrics["debug/loc_ref_speed_scale"] = ref_speed_scale
            metrics["debug/loc_ref_phase_scale"] = ref_phase_scale
            metrics["debug/loc_ref_swing_x_scale_active"] = ref_speed_scale
            metrics["debug/loc_ref_phase_scale_active"] = ref_phase_scale
        else:
            metrics["debug/loc_ref_speed_scale"] = jp.zeros((), dtype=jp.float32)
            metrics["debug/loc_ref_phase_scale"] = jp.zeros((), dtype=jp.float32)
            metrics["debug/loc_ref_swing_x_scale_active"] = nominal_swing_x_scale
            metrics["debug/loc_ref_phase_scale_active"] = ref_progression_permission
        if mpc_debug is not None:
            metrics["debug/mpc_planner_active"] = mpc_debug.planner_active
            metrics["debug/mpc_controller_active"] = mpc_debug.controller_active
            metrics["debug/mpc_target_com_x"] = mpc_debug.target_com_x
            metrics["debug/mpc_target_com_y"] = mpc_debug.target_com_y
            metrics["debug/mpc_target_step_x"] = mpc_debug.target_step_x
            metrics["debug/mpc_target_step_y"] = mpc_debug.target_step_y
            metrics["debug/mpc_support_state"] = mpc_debug.support_state
            metrics["debug/mpc_step_requested"] = mpc_debug.step_requested
        # Add placeholder for metrics_vec (will be rebuilt after cond)
        # This ensures pytree structure matches preserved_metrics
        metrics[METRICS_VEC_KEY] = build_metrics_vec(metrics)

        # Compute current foot positions for WildRobotInfo update
        curr_left_foot_pos, curr_right_foot_pos = self._cal.get_foot_positions(
            data, normalize=False
        )
        # Cache contact loads for touchdown/step-event detection in next step.
        left_force_now, right_force_now = self._cal.get_aggregated_foot_contacts(data)
        contact_threshold = self._config.env.contact_threshold_force
        left_loaded_now = left_force_now > contact_threshold
        right_loaded_now = right_force_now > contact_threshold
        prev_left_loaded_b = wr.prev_left_loaded > 0.5
        prev_right_loaded_b = wr.prev_right_loaded > 0.5
        liftoff_left_now = jp.logical_and(prev_left_loaded_b, jp.logical_not(left_loaded_now))
        liftoff_right_now = jp.logical_and(prev_right_loaded_b, jp.logical_not(right_loaded_now))
        touchdown_left_now = jp.logical_and(jp.logical_not(prev_left_loaded_b), left_loaded_now)
        touchdown_right_now = jp.logical_and(jp.logical_not(prev_right_loaded_b), right_loaded_now)
        if bool(getattr(self._config.env, "fsm_enabled", False)):
            in_swing = wr.fsm_phase == sc.SWING
            touchdown_left_now = jp.logical_and(
                touchdown_left_now, in_swing & (wr.fsm_swing_foot == 0)
            )
            touchdown_right_now = jp.logical_and(
                touchdown_right_now, in_swing & (wr.fsm_swing_foot == 1)
            )
        touchdown_any_now = jp.logical_or(touchdown_left_now, touchdown_right_now)
        touchdown_foot_now = jp.where(
            touchdown_left_now,
            jp.asarray(0, dtype=jp.int32),
            jp.where(
                touchdown_right_now,
                jp.asarray(1, dtype=jp.int32),
                wr.last_touchdown_foot,
            ),
        )
        next_last_touchdown_root_pos = jp.where(
            touchdown_any_now, curr_root_pose.position, wr.last_touchdown_root_pos
        )
        next_last_touchdown_foot = jp.where(
            touchdown_any_now, touchdown_foot_now, wr.last_touchdown_foot
        )
        critic_obs = self._get_privileged_critic_obs(
            data, step_count + 1, wr.push_schedule
        )
        root_velocity_world = self._cal.get_root_velocity(
            data, frame=CoordinateFrame.WORLD
        )
        post_push_horizontal_speed = jp.linalg.norm(root_velocity_world.linear[:2])
        root_vel_heading_local = self._cal.get_root_velocity(
            data, frame=CoordinateFrame.HEADING_LOCAL
        )
        pitch_rate_now = root_vel_heading_local.angular_xyz[1]
        capture_error_xy = self._get_capture_point_error(data)
        capture_error_norm = jp.linalg.norm(capture_error_xy)
        root_pose_now = self._cal.get_root_pose(data)
        left_foot_h, right_foot_h = self._cal.get_foot_positions(
            data, normalize=False, frame=CoordinateFrame.HEADING_LOCAL
        )
        first_touchdown_dx = jp.where(
            touchdown_left_now, left_foot_h[0], jp.where(touchdown_right_now, right_foot_h[0], 0.0)
        )
        first_touchdown_dy = jp.where(
            touchdown_left_now, left_foot_h[1], jp.where(touchdown_right_now, right_foot_h[1], 0.0)
        )
        root_vel_for_target = root_vel_heading_local.linear_xyz
        fwd_vel_now, lat_vel_now, _ = root_vel_for_target
        roll_now, pitch_now, _ = root_pose_now.euler_angles()
        weights = self._config.reward_weights
        base_y = 0.5 * jp.asarray(weights.stance_width_target)
        y_corr = (
            jp.asarray(getattr(weights, "foot_place_k_lat_vel", 0.15)) * lat_vel_now
            + jp.asarray(getattr(weights, "foot_place_k_roll", 0.10)) * roll_now
        )
        x_corr = (
            jp.asarray(getattr(weights, "foot_place_k_cmd_vel", 0.0)) * velocity_cmd
            + jp.asarray(getattr(weights, "foot_place_k_fwd_vel", 0.05)) * fwd_vel_now
            + jp.asarray(getattr(weights, "foot_place_k_pitch", 0.05)) * pitch_now
        )
        target_x = x_corr
        target_y = jp.where(
            touchdown_left_now, base_y + y_corr, jp.where(touchdown_right_now, -base_y + y_corr, 0.0)
        )
        first_touchdown_target_err_x = first_touchdown_dx - target_x
        first_touchdown_target_err_y = first_touchdown_dy - target_y
        left_knee_pitch_now, right_knee_pitch_now = self._cal.get_knee_pitch_positions(
            data.qpos, normalize=False
        )
        recovery_knee_flex_now = jp.maximum(
            jp.abs(left_knee_pitch_now), jp.abs(right_knee_pitch_now)
        )
        recovery_state_updates, recovery_metrics = _update_recovery_tracking(
            track_recovery=jp.logical_and(
                jp.asarray(self._config.env.push_enabled, dtype=jp.bool_),
                jp.logical_not(jp.asarray(disable_pushes, dtype=jp.bool_)),
            ),
            push_schedule=wr.push_schedule,
            current_step=step_count + 1,
            episode_done=done > 0.5,
            episode_failed=(done > 0.5) & (truncated < 0.5),
            liftoff_left=liftoff_left_now,
            liftoff_right=liftoff_right_now,
            touchdown_left=touchdown_left_now,
            touchdown_right=touchdown_right_now,
            horizontal_speed=post_push_horizontal_speed,
            pitch_rate=pitch_rate_now,
            capture_error_norm=capture_error_norm,
            first_touchdown_dx=first_touchdown_dx,
            first_touchdown_dy=first_touchdown_dy,
            first_touchdown_target_err_x=first_touchdown_target_err_x,
            first_touchdown_target_err_y=first_touchdown_target_err_y,
            current_height=root_pose_now.height,
            current_knee_flex=recovery_knee_flex_now,
            recovery_active=wr.recovery_active,
            recovery_age=wr.recovery_age,
            recovery_last_support_foot=wr.recovery_last_support_foot,
            recovery_first_liftoff_recorded=wr.recovery_first_liftoff_recorded,
            recovery_first_liftoff_latency=wr.recovery_first_liftoff_latency,
            recovery_first_touchdown_recorded=wr.recovery_first_touchdown_recorded,
            recovery_first_step_latency=wr.recovery_first_step_latency,
            recovery_first_touchdown_age=wr.recovery_first_touchdown_age,
            recovery_visible_step_recorded=wr.recovery_visible_step_recorded,
            recovery_touchdown_count=wr.recovery_touchdown_count,
            recovery_support_foot_changes=wr.recovery_support_foot_changes,
            recovery_pitch_rate_at_push_end=wr.recovery_pitch_rate_at_push_end,
            recovery_pitch_rate_at_touchdown=wr.recovery_pitch_rate_at_touchdown,
            recovery_pitch_rate_after_10t=wr.recovery_pitch_rate_after_10t,
            recovery_capture_error_at_push_end=wr.recovery_capture_error_at_push_end,
            recovery_capture_error_at_touchdown=wr.recovery_capture_error_at_touchdown,
            recovery_capture_error_after_10t=wr.recovery_capture_error_after_10t,
            recovery_first_step_dx=wr.recovery_first_step_dx,
            recovery_first_step_dy=wr.recovery_first_step_dy,
            recovery_first_step_target_err_x=wr.recovery_first_step_target_err_x,
            recovery_first_step_target_err_y=wr.recovery_first_step_target_err_y,
            recovery_min_height=wr.recovery_min_height,
            recovery_max_knee_flex=wr.recovery_max_knee_flex,
        )
        metrics.update(recovery_metrics)
        push_active_now = (
            (step_count + 1 >= wr.push_schedule.start_step)
            & (step_count + 1 < wr.push_schedule.end_step)
            & jp.asarray(self._config.env.push_enabled, dtype=jp.bool_)
            & jp.logical_not(jp.asarray(disable_pushes, dtype=jp.bool_))
        )
        teacher_enabled_now = bool(getattr(self._config.env, "teacher_enabled", False)) or (
            controller_stack == "ppo_teacher_standing"
        )
        teacher = compute_teacher_step_target(
            teacher_enabled=teacher_enabled_now,
            push_active=push_active_now,
            recovery_active=wr.recovery_active,
            need_step=metrics.get("debug/need_step", jp.zeros(())),
            pitch=pitch_now,
            roll=roll_now,
            pitch_rate=pitch_rate_now,
            forward_vel=fwd_vel_now,
            lateral_vel=lat_vel_now,
            capture_error_xy=capture_error_xy,
            hard_threshold=float(getattr(self._config.env, "teacher_hard_threshold", 0.60)),
            x_min=float(getattr(self._config.env, "teacher_target_x_min", -0.10)),
            x_max=float(getattr(self._config.env, "teacher_target_x_max", 0.10)),
            left_y_min=float(getattr(self._config.env, "teacher_target_y_left_min", 0.02)),
            left_y_max=float(getattr(self._config.env, "teacher_target_y_left_max", 0.06)),
            right_y_min=float(getattr(self._config.env, "teacher_target_y_right_min", -0.06)),
            right_y_max=float(getattr(self._config.env, "teacher_target_y_right_max", -0.02)),
        )
        teacher_xy_reward, teacher_xy_error = compute_teacher_target_step_xy_reward(
            touchdown_x=first_touchdown_dx,
            touchdown_y=first_touchdown_dy,
            target_x=teacher.target_step_x,
            target_y=teacher.target_step_y,
            teacher_active=teacher.teacher_active,
            first_recovery_touchdown=(
                wr.recovery_active & (~wr.recovery_first_touchdown_recorded) & touchdown_any_now
            ).astype(jp.float32),
            sigma=float(getattr(self._config.reward_weights, "teacher_target_step_xy_sigma", 0.06)),
        )
        first_recovery_liftoff_event = (
            wr.recovery_active & (~wr.recovery_first_liftoff_recorded) & (liftoff_left_now | liftoff_right_now)
        ).astype(jp.float32)
        first_recovery_touchdown_event = (
            wr.recovery_active & (~wr.recovery_first_touchdown_recorded) & touchdown_any_now
        ).astype(jp.float32)
        teacher_step_required_reward, teacher_step_required_event = compute_teacher_step_required_reward(
            teacher_active=teacher.teacher_active,
            step_required_soft=teacher.step_required_soft,
            first_recovery_liftoff=first_recovery_liftoff_event,
        )
        teacher_swing_reward, _teacher_swing_event = compute_teacher_swing_foot_reward(
            touchdown_foot=touchdown_foot_now,
            teacher_swing_foot=teacher.swing_foot,
            teacher_active=teacher.teacher_active,
            teacher_step_required_hard=teacher.step_required_hard,
            first_recovery_touchdown=first_recovery_touchdown_event,
        )
        whole_body_teacher = compute_teacher_whole_body_target(
            whole_body_teacher_enabled=bool(
                getattr(self._config.env, "whole_body_teacher_enabled", False)
            ),
            push_active=push_active_now,
            recovery_active=wr.recovery_active,
            step_required_hard=teacher.step_required_hard,
            height=root_pose_now.height,
            horizontal_speed=post_push_horizontal_speed,
            height_target_min=float(
                getattr(self._config.env, "whole_body_teacher_height_target_min", 0.39)
            ),
            height_target_max=float(
                getattr(self._config.env, "whole_body_teacher_height_target_max", 0.42)
            ),
            height_hard_gate=bool(
                getattr(self._config.env, "whole_body_teacher_height_hard_gate", True)
            ),
            com_vel_target=float(
                getattr(self._config.env, "whole_body_teacher_com_vel_target", 0.12)
            ),
        )
        teacher_recovery_height_reward = compute_teacher_recovery_height_reward(
            height_error=whole_body_teacher.recovery_height_error,
            teacher_active=whole_body_teacher.whole_body_active,
            sigma=float(
                getattr(self._config.reward_weights, "teacher_recovery_height_sigma", 0.03)
            ),
        )
        teacher_com_velocity_reduction_reward = compute_teacher_com_velocity_reduction_reward(
            com_velocity_error=whole_body_teacher.com_velocity_error,
            horizontal_speed=post_push_horizontal_speed,
            teacher_active=whole_body_teacher.whole_body_active,
            active_speed_min=float(
                getattr(
                    self._config.env, "whole_body_teacher_com_vel_active_speed_min", 0.10
                )
            ),
            sigma=float(
                getattr(
                    self._config.reward_weights,
                    "teacher_com_velocity_reduction_sigma",
                    0.08,
                )
            ),
        )
        teacher_xy_event = (
            teacher.teacher_active
            * (wr.recovery_active & (~wr.recovery_first_touchdown_recorded) & touchdown_any_now).astype(
                jp.float32
            )
        )
        reward += (
            jp.asarray(getattr(self._config.reward_weights, "teacher_target_step_xy", 0.0), dtype=jp.float32)
            * teacher_xy_reward
        )
        reward += (
            jp.asarray(getattr(self._config.reward_weights, "teacher_step_required", 0.0), dtype=jp.float32)
            * teacher_step_required_reward
        )
        reward += (
            jp.asarray(getattr(self._config.reward_weights, "teacher_swing_foot", 0.0), dtype=jp.float32)
            * teacher_swing_reward
        )
        reward += (
            jp.asarray(
                getattr(self._config.reward_weights, "teacher_recovery_height", 0.0),
                dtype=jp.float32,
            )
            * teacher_recovery_height_reward
        )
        reward += (
            jp.asarray(
                getattr(self._config.reward_weights, "teacher_com_velocity_reduction", 0.0),
                dtype=jp.float32,
            )
            * teacher_com_velocity_reduction_reward
        )
        metrics["reward/total"] = reward
        reward_components["reward/total"] = reward
        reward_components["reward/teacher_target_step_xy"] = teacher_xy_reward
        reward_components["reward/teacher_step_required"] = teacher_step_required_reward
        reward_components["reward/teacher_swing_foot"] = teacher_swing_reward
        reward_components["reward/teacher_recovery_height"] = teacher_recovery_height_reward
        reward_components["reward/teacher_com_velocity_reduction"] = (
            teacher_com_velocity_reduction_reward
        )
        reward_components["teacher/target_xy_error"] = teacher_xy_error * teacher_xy_event
        reward_components["teacher/target_xy_error_events"] = teacher_xy_event
        reward_components["teacher/step_required_event_rate"] = teacher_step_required_event
        reward_components["teacher/swing_foot_match_frac"] = teacher_swing_reward
        reward_components["teacher/swing_foot_match_events"] = _teacher_swing_event
        reward_components["recovery/visible_step_rate"] = recovery_metrics.get(
            "recovery/visible_step_rate", jp.zeros(())
        )
        reward_components["visible_step_rate_hard"] = recovery_metrics.get(
            "recovery/visible_step_rate", jp.zeros(())
        )
        metrics["reward/teacher_target_step_xy"] = teacher_xy_reward
        metrics["reward/teacher_step_required"] = teacher_step_required_reward
        metrics["reward/teacher_swing_foot"] = teacher_swing_reward
        metrics["reward/teacher_recovery_height"] = teacher_recovery_height_reward
        metrics["reward/teacher_com_velocity_reduction"] = (
            teacher_com_velocity_reduction_reward
        )
        metrics["teacher/active_frac"] = teacher.teacher_active
        metrics["teacher/active_count"] = teacher.active_count
        metrics["teacher/step_required_mean"] = teacher.step_required_soft * teacher.teacher_active
        metrics["teacher/step_required_hard_frac"] = teacher.step_required_hard * teacher.teacher_active
        metrics["teacher/target_step_x_mean"] = teacher.target_step_x * teacher.teacher_active
        metrics["teacher/target_step_y_mean"] = teacher.target_step_y * teacher.teacher_active
        metrics["teacher/swing_left_frac"] = (
            (teacher.swing_foot == jp.asarray(0, dtype=jp.int32)).astype(jp.float32)
            * teacher.teacher_active
        )
        metrics["teacher/target_xy_error"] = teacher_xy_error * teacher_xy_event
        metrics["teacher/target_xy_error_events"] = teacher_xy_event
        clean_now = (~jp.logical_or(push_active_now, wr.recovery_active)).astype(jp.float32)
        metrics["teacher/raw_step_required_during_clean_mean"] = (
            teacher.raw_step_required_soft * clean_now
        )
        metrics["teacher/teacher_active_during_clean_frac"] = teacher.teacher_active * clean_now
        metrics["teacher/clean_step_count"] = clean_now
        push_now = push_active_now.astype(jp.float32)
        metrics["teacher/push_step_count"] = push_now
        metrics["teacher/teacher_active_during_push_frac"] = teacher.teacher_active * push_now
        metrics["teacher/reachable_frac"] = teacher.target_reachable * teacher.teacher_active
        metrics["teacher/target_step_x_min"] = jp.where(
            teacher.teacher_active > 0.5, teacher.target_step_x, jp.asarray(jp.inf, dtype=jp.float32)
        )
        metrics["teacher/target_step_x_max"] = jp.where(
            teacher.teacher_active > 0.5, teacher.target_step_x, jp.asarray(-jp.inf, dtype=jp.float32)
        )
        metrics["teacher/target_step_y_min"] = jp.where(
            teacher.teacher_active > 0.5, teacher.target_step_y, jp.asarray(jp.inf, dtype=jp.float32)
        )
        metrics["teacher/target_step_y_max"] = jp.where(
            teacher.teacher_active > 0.5, teacher.target_step_y, jp.asarray(-jp.inf, dtype=jp.float32)
        )
        metrics["teacher/target_step_x_sq_mean"] = (
            teacher.target_step_x * teacher.target_step_x * teacher.teacher_active
        )
        metrics["teacher/target_step_y_sq_mean"] = (
            teacher.target_step_y * teacher.target_step_y * teacher.teacher_active
        )
        metrics["teacher/swing_foot"] = teacher.swing_foot.astype(jp.float32)
        metrics["teacher/target_xy_error_events"] = teacher_xy_event
        metrics["teacher/step_required_event_rate"] = teacher_step_required_event
        metrics["teacher/swing_foot_match_frac"] = teacher_swing_reward
        metrics["teacher/swing_foot_match_events"] = _teacher_swing_event
        metrics["teacher/whole_body_active_frac"] = whole_body_teacher.whole_body_active
        metrics["teacher/whole_body_active_count"] = whole_body_teacher.whole_body_active_count
        metrics["teacher/recovery_height_target_mean"] = whole_body_teacher.recovery_height_target
        metrics["teacher/recovery_height_error"] = whole_body_teacher.recovery_height_error
        metrics["teacher/recovery_height_in_band_frac"] = whole_body_teacher.recovery_height_in_band
        metrics["teacher/com_velocity_target_mean"] = whole_body_teacher.com_velocity_target
        metrics["teacher/com_velocity_error"] = whole_body_teacher.com_velocity_error
        metrics["teacher/com_velocity_target_hit_frac"] = whole_body_teacher.com_velocity_target_hit
        metrics["recovery/visible_step_rate"] = recovery_metrics.get("recovery/visible_step_rate", jp.zeros(()))
        metrics["visible_step_rate_hard"] = recovery_metrics.get("recovery/visible_step_rate", jp.zeros(()))
        metrics["recovery/min_height"] = recovery_metrics.get("recovery/min_height", jp.zeros(()))
        metrics["recovery/max_knee_flex"] = recovery_metrics.get("recovery/max_knee_flex", jp.zeros(()))
        metrics["recovery/first_step_dist_abs"] = recovery_metrics.get(
            "recovery/first_step_dist_abs", jp.zeros(())
        )
        metrics["teacher/target_step_x_std"] = jp.zeros(())
        metrics["teacher/target_step_y_std"] = jp.zeros(())
        reward_components["teacher/active_frac"] = teacher.teacher_active
        reward_components["teacher/active_count"] = teacher.active_count
        reward_components["teacher/step_required_mean"] = (
            teacher.step_required_soft * teacher.teacher_active
        )
        reward_components["teacher/step_required_hard_frac"] = (
            teacher.step_required_hard * teacher.teacher_active
        )
        reward_components["teacher/target_step_x_mean"] = (
            teacher.target_step_x * teacher.teacher_active
        )
        reward_components["teacher/target_step_y_mean"] = (
            teacher.target_step_y * teacher.teacher_active
        )
        reward_components["teacher/swing_left_frac"] = (
            (teacher.swing_foot == jp.asarray(0, dtype=jp.int32)).astype(jp.float32)
            * teacher.teacher_active
        )
        reward_components["teacher/raw_step_required_during_clean_mean"] = (
            teacher.raw_step_required_soft * clean_now
        )
        reward_components["teacher/teacher_active_during_clean_frac"] = (
            teacher.teacher_active * clean_now
        )
        reward_components["teacher/clean_step_count"] = clean_now
        reward_components["teacher/push_step_count"] = push_now
        reward_components["teacher/teacher_active_during_push_frac"] = (
            teacher.teacher_active * push_now
        )
        reward_components["teacher/reachable_frac"] = (
            teacher.target_reachable * teacher.teacher_active
        )
        reward_components["teacher/target_step_x_min"] = jp.where(
            teacher.teacher_active > 0.5, teacher.target_step_x, jp.asarray(jp.inf, dtype=jp.float32)
        )
        reward_components["teacher/target_step_x_max"] = jp.where(
            teacher.teacher_active > 0.5, teacher.target_step_x, jp.asarray(-jp.inf, dtype=jp.float32)
        )
        reward_components["teacher/target_step_y_min"] = jp.where(
            teacher.teacher_active > 0.5, teacher.target_step_y, jp.asarray(jp.inf, dtype=jp.float32)
        )
        reward_components["teacher/target_step_y_max"] = jp.where(
            teacher.teacher_active > 0.5, teacher.target_step_y, jp.asarray(-jp.inf, dtype=jp.float32)
        )
        reward_components["teacher/target_step_x_sq_mean"] = (
            teacher.target_step_x * teacher.target_step_x * teacher.teacher_active
        )
        reward_components["teacher/target_step_y_sq_mean"] = (
            teacher.target_step_y * teacher.target_step_y * teacher.teacher_active
        )
        reward_components["teacher/swing_foot"] = teacher.swing_foot.astype(jp.float32)
        reward_components["teacher/swing_foot_match_events"] = _teacher_swing_event
        reward_components["teacher/target_step_x_std"] = jp.zeros(())
        reward_components["teacher/target_step_y_std"] = jp.zeros(())
        reward_components["teacher/whole_body_active_frac"] = whole_body_teacher.whole_body_active
        reward_components["teacher/whole_body_active_count"] = whole_body_teacher.whole_body_active_count
        reward_components["teacher/recovery_height_target_mean"] = (
            whole_body_teacher.recovery_height_target
        )
        reward_components["teacher/recovery_height_error"] = whole_body_teacher.recovery_height_error
        reward_components["teacher/recovery_height_in_band_frac"] = (
            whole_body_teacher.recovery_height_in_band
        )
        reward_components["teacher/com_velocity_target_mean"] = whole_body_teacher.com_velocity_target
        reward_components["teacher/com_velocity_error"] = whole_body_teacher.com_velocity_error
        reward_components["teacher/com_velocity_target_hit_frac"] = (
            whole_body_teacher.com_velocity_target_hit
        )
        metrics["teacher/whole_body_active_during_clean_frac"] = (
            whole_body_teacher.whole_body_active * clean_now
        )
        metrics["teacher/whole_body_active_during_push_frac"] = (
            whole_body_teacher.whole_body_active * push_now
        )
        reward_components["teacher/whole_body_active_during_clean_frac"] = (
            whole_body_teacher.whole_body_active * clean_now
        )
        reward_components["teacher/whole_body_active_during_push_frac"] = (
            whole_body_teacher.whole_body_active * push_now
        )
        reward_components["recovery/min_height"] = recovery_metrics.get(
            "recovery/min_height", jp.zeros(())
        )
        reward_components["recovery/max_knee_flex"] = recovery_metrics.get(
            "recovery/max_knee_flex", jp.zeros(())
        )
        reward_components["recovery/first_step_dist_abs"] = recovery_metrics.get(
            "recovery/first_step_dist_abs", jp.zeros(())
        )
        unnecessary_need_step_threshold = jp.asarray(0.2, dtype=jp.float32)
        unnecessary_touchdown = (
            touchdown_any_now
            & (~push_active_now)
            & (~wr.recovery_active)
            & (metrics.get("debug/need_step", jp.zeros(())) < unnecessary_need_step_threshold)
        )
        metrics["unnecessary_step_rate"] = unnecessary_touchdown.astype(jp.float32)

        # Create new WildRobotInfo with updated values
        new_wr_info = WildRobotInfo(
            step_count=step_count + 1,
            prev_action=applied_action,
            pending_action=policy_state.prev_action,
            truncated=truncated,  # 1.0 if reached max steps (success), 0.0 otherwise
            velocity_cmd=velocity_cmd,  # Preserved through episode
            prev_root_pos=curr_root_pose.position,
            prev_root_quat=curr_root_pose.orientation,
            prev_left_foot_pos=curr_left_foot_pos,
            prev_right_foot_pos=curr_right_foot_pos,
            imu_quat_hist=new_imu_quat_hist,
            imu_gyro_hist=new_imu_gyro_hist,
            foot_contacts=self._cal.get_geom_based_foot_contacts(
                data, normalize=True
            ),  # For AMP
            root_height=curr_root_pose.height,  # Actual root height for AMP features
            prev_left_loaded=left_loaded_now.astype(jp.float32),
            prev_right_loaded=right_loaded_now.astype(jp.float32),
            cycle_start_forward_x=cycle_start_forward_x,
            last_touchdown_root_pos=next_last_touchdown_root_pos,
            last_touchdown_foot=next_last_touchdown_foot,
            critic_obs=critic_obs,
            loc_ref_phase_time=ref_phase_time,
            loc_ref_stance_foot=ref_stance_foot.astype(jp.int32),
            loc_ref_switch_count=ref_switch_count.astype(jp.int32),
            loc_ref_mode_id=ref_mode_id.astype(jp.int32),
            loc_ref_mode_time=ref_mode_time,
            loc_ref_startup_route_progress=startup_route_progress,
            loc_ref_startup_route_ceiling=startup_route_ceiling,
            loc_ref_startup_route_stage_id=startup_route_stage_id,
            loc_ref_startup_route_transition_reason=startup_route_transition_reason,
            loc_ref_gait_phase_sin=loc_ref_phase_sin,
            loc_ref_gait_phase_cos=loc_ref_phase_cos,
            loc_ref_next_foothold=loc_ref_next_foothold,
            loc_ref_swing_pos=loc_ref_swing_pos,
            loc_ref_swing_vel=loc_ref_swing_vel,
            loc_ref_pelvis_height=loc_ref_pelvis_height,
            loc_ref_pelvis_roll=loc_ref_pelvis_roll,
            loc_ref_pelvis_pitch=loc_ref_pelvis_pitch,
            loc_ref_history=loc_ref_history,
            nominal_q_ref=nominal_q_ref,
            push_schedule=wr.push_schedule,
            # M3 FSM state (updated by _fsm_compute_ctrl or carried from wr)
            fsm_phase=new_fsm[0],
            fsm_swing_foot=new_fsm[1],
            fsm_phase_ticks=new_fsm[2],
            fsm_frozen_tx=new_fsm[3],
            fsm_frozen_ty=new_fsm[4],
            fsm_swing_sx=new_fsm[5],
            fsm_swing_sy=new_fsm[6],
            fsm_touch_hold=new_fsm[7],
            fsm_trigger_hold=new_fsm[8],
            # v0.17.1+: Recovery metrics state
            recovery_active=recovery_state_updates["recovery_active"],
            recovery_age=recovery_state_updates["recovery_age"],
            recovery_last_support_foot=recovery_state_updates["recovery_last_support_foot"],
            recovery_first_liftoff_recorded=recovery_state_updates["recovery_first_liftoff_recorded"],
            recovery_first_liftoff_latency=recovery_state_updates["recovery_first_liftoff_latency"],
            recovery_first_touchdown_recorded=recovery_state_updates["recovery_first_touchdown_recorded"],
            recovery_first_step_latency=recovery_state_updates["recovery_first_step_latency"],
            recovery_first_touchdown_age=recovery_state_updates["recovery_first_touchdown_age"],
            recovery_visible_step_recorded=recovery_state_updates["recovery_visible_step_recorded"],
            recovery_touchdown_count=recovery_state_updates["recovery_touchdown_count"],
            recovery_support_foot_changes=recovery_state_updates["recovery_support_foot_changes"],
            recovery_pitch_rate_at_push_end=recovery_state_updates["recovery_pitch_rate_at_push_end"],
            recovery_pitch_rate_at_touchdown=recovery_state_updates["recovery_pitch_rate_at_touchdown"],
            recovery_pitch_rate_after_10t=recovery_state_updates["recovery_pitch_rate_after_10t"],
            recovery_capture_error_at_push_end=recovery_state_updates["recovery_capture_error_at_push_end"],
            recovery_capture_error_at_touchdown=recovery_state_updates["recovery_capture_error_at_touchdown"],
            recovery_capture_error_after_10t=recovery_state_updates["recovery_capture_error_after_10t"],
            recovery_first_step_dx=recovery_state_updates["recovery_first_step_dx"],
            recovery_first_step_dy=recovery_state_updates["recovery_first_step_dy"],
            recovery_first_step_target_err_x=recovery_state_updates["recovery_first_step_target_err_x"],
            recovery_first_step_target_err_y=recovery_state_updates["recovery_first_step_target_err_y"],
            recovery_min_height=recovery_state_updates["recovery_min_height"],
            recovery_max_knee_flex=recovery_state_updates["recovery_max_knee_flex"],
            teacher_active=metrics.get("teacher/active_frac", jp.zeros(())),
            teacher_step_required_soft=metrics.get("teacher/step_required_mean", jp.zeros(())),
            teacher_step_required_hard=metrics.get(
                "teacher/step_required_hard_frac", jp.zeros(())
            ),
            teacher_swing_foot=metrics.get(
                "teacher/swing_foot", jp.asarray(-1, dtype=jp.float32)
            ).astype(jp.int32),
            teacher_target_step_x=metrics.get("teacher/target_step_x_mean", jp.zeros(())),
            teacher_target_step_y=metrics.get("teacher/target_step_y_mean", jp.zeros(())),
            teacher_target_reachable=metrics.get("teacher/reachable_frac", jp.zeros(())),
            mpc_planner_active=metrics["debug/mpc_planner_active"],
            mpc_controller_active=metrics["debug/mpc_controller_active"],
            mpc_target_com_x=metrics["debug/mpc_target_com_x"],
            mpc_target_com_y=metrics["debug/mpc_target_com_y"],
            mpc_target_step_x=metrics["debug/mpc_target_step_x"],
            mpc_target_step_y=metrics["debug/mpc_target_step_y"],
            mpc_support_state=metrics["debug/mpc_support_state"],
            mpc_step_requested=metrics["debug/mpc_step_requested"],
            domain_rand_friction_scale=wr.domain_rand_friction_scale,
            domain_rand_mass_scales=wr.domain_rand_mass_scales,
            domain_rand_kp_scales=wr.domain_rand_kp_scales,
            domain_rand_frictionloss_scales=wr.domain_rand_frictionloss_scales,
            domain_rand_joint_offsets=wr.domain_rand_joint_offsets,
        )

        # Preserve wrapper fields, update wr namespace
        # The Brax training wrapper adds fields like 'truncation', 'episode_done',
        # 'episode_metrics', 'first_obs', 'steps', 'first_state'. We preserve these.
        info = dict(state.info)  # Copy existing wrapper fields
        info[WR_INFO_KEY] = new_wr_info  # Update env-owned namespace

        # =================================================================
        # Auto-reset on termination
        # When done=True, reset physics state to initial keyframe but keep
        # done=True so the algorithm knows an episode ended.
        # v0.10.2: Preserve termination metrics for diagnostics
        # =================================================================
        # Pass state.info as base_info to preserve wrapper fields during auto-reset
        # Use updated RNG after consuming IMU noise
        rng_after, reset_push_rng, reset_domain_rng = jax.random.split(rng_after, 3)
        reset_schedule = sample_push_schedule(
            reset_push_rng, self._config.env, self._push_body_ids
        )
        reset_domain_rand_params = self._sample_domain_rand_params(reset_domain_rng)
        reset_qpos = self._init_qpos.at[self._actuator_qpos_addrs].set(
            jp.clip(
                self._default_joint_qpos + reset_domain_rand_params["joint_offsets"],
                self._joint_range_mins,
                self._joint_range_maxs,
            )
        )
        reset_state = self._make_initial_state(
            reset_qpos,
            velocity_cmd,
            base_info=state.info,
            push_schedule=reset_schedule,
            domain_rand_params=reset_domain_rand_params,
        )

        # v0.10.2: Preserve termination diagnostics in metrics even after reset
        # The reset_state.metrics has all zeros for term/*, but we want to keep
        # the actual termination reason for logging
        # v0.10.4: Also preserve episode_step_count for accurate ep_len
        # v0.10.4: Also preserve tracking metrics for accurate vel_err/torque logging
        preserved_metrics = {
            **reset_state.metrics,
            "episode_step_count": metrics[
                "episode_step_count"
            ],  # Terminal step count for ep_len
            # Tracking metrics (for topline monitoring)
            "tracking/vel_error": metrics["tracking/vel_error"],
            "tracking/max_torque": metrics["tracking/max_torque"],
            # Core metrics (for logging)
            "forward_velocity": metrics["forward_velocity"],
            "velocity_command": metrics["velocity_command"],
            # Termination diagnostics
            "term/height_low": metrics["term/height_low"],
            "term/height_high": metrics["term/height_high"],
            "term/pitch": metrics["term/pitch"],
            "term/roll": metrics["term/roll"],
            "term/truncated": metrics["term/truncated"],
            "term/pitch_val": metrics["term/pitch_val"],
            "term/roll_val": metrics["term/roll_val"],
            "term/height_val": metrics["term/height_val"],
            # Keep dict structure stable when adding new reward/debug keys.
            "reward/posture": metrics.get("reward/posture", jp.zeros(())),
            "reward/pitch_rate": metrics.get("reward/pitch_rate", jp.zeros(())),
            "reward/height_floor": metrics.get("reward/height_floor", jp.zeros(())),
            "reward/com_velocity_damping": metrics.get("reward/com_velocity_damping", jp.zeros(())),
            "reward/teacher_target_step_xy": metrics.get(
                "reward/teacher_target_step_xy", jp.zeros(())
            ),
            "reward/teacher_step_required": metrics.get("reward/teacher_step_required", jp.zeros(())),
            "reward/teacher_swing_foot": metrics.get("reward/teacher_swing_foot", jp.zeros(())),
            "reward/teacher_recovery_height": metrics.get(
                "reward/teacher_recovery_height", jp.zeros(())
            ),
            "reward/teacher_com_velocity_reduction": metrics.get(
                "reward/teacher_com_velocity_reduction", jp.zeros(())
            ),
            "reward/arrest_pitch_rate": metrics.get("reward/arrest_pitch_rate", jp.zeros(())),
            "reward/arrest_capture_error": metrics.get(
                "reward/arrest_capture_error", jp.zeros(())
            ),
            "reward/post_touchdown_survival": metrics.get(
                "reward/post_touchdown_survival", jp.zeros(())
            ),
            "debug/posture_mse": metrics.get("debug/posture_mse", jp.zeros(())),
            "reward/step_event": metrics.get("reward/step_event", jp.zeros(())),
            "reward/foot_place": metrics.get("reward/foot_place", jp.zeros(())),
            "reward/step_length": metrics.get("reward/step_length", jp.zeros(())),
            "reward/step_progress": metrics.get("reward/step_progress", jp.zeros(())),
            "reward/teacher_target_step_xy": metrics.get(
                "reward/teacher_target_step_xy", jp.zeros(())
            ),
            "reward/dense_progress": metrics.get("reward/dense_progress", jp.zeros(())),
            "reward/cycle_progress": metrics.get("reward/cycle_progress", jp.zeros(())),
            "debug/need_step": metrics.get("debug/need_step", jp.zeros(())),
            "debug/touchdown_left": metrics.get("debug/touchdown_left", jp.zeros(())),
            "debug/touchdown_right": metrics.get("debug/touchdown_right", jp.zeros(())),
            "debug/velocity_step_gate": metrics.get("debug/velocity_step_gate", jp.zeros(())),
            "debug/propulsion_gate": metrics.get("debug/propulsion_gate", jp.zeros(())),
            "debug/cycle_complete": metrics.get("debug/cycle_complete", jp.zeros(())),
            "debug/cycle_forward_delta": metrics.get("debug/cycle_forward_delta", jp.zeros(())),
            "debug/step_progress_delta": metrics.get("debug/step_progress_delta", jp.zeros(())),
            "debug/step_progress_event": metrics.get("debug/step_progress_event", jp.zeros(())),
            "debug/recovery_step_gate": metrics.get("debug/recovery_step_gate", jp.zeros(())),
            "debug/post_touchdown_arrest_gate": metrics.get(
                "debug/post_touchdown_arrest_gate", jp.zeros(())
            ),
            "debug/disturbed_reward_gate": metrics.get("debug/disturbed_reward_gate", jp.zeros(())),
            "debug/effective_orientation_weight": metrics.get(
                "debug/effective_orientation_weight", jp.zeros(())
            ),
            "debug/effective_height_target_weight": metrics.get(
                "debug/effective_height_target_weight", jp.zeros(())
            ),
            "debug/effective_posture_weight": metrics.get(
                "debug/effective_posture_weight", jp.zeros(())
            ),
            # v0.14.x: M3 FSM debug metrics – preserve on auto-reset for accurate logging
            "debug/bc_phase": metrics.get("debug/bc_phase", jp.zeros(())),
            "debug/bc_swing_foot": metrics.get("debug/bc_swing_foot", jp.zeros(())),
            "debug/bc_phase_ticks": metrics.get("debug/bc_phase_ticks", jp.zeros(())),
            "debug/bc_in_swing": metrics.get("debug/bc_in_swing", jp.zeros(())),
            "debug/bc_in_recover": metrics.get("debug/bc_in_recover", jp.zeros(())),
            # v0.17.2: Finalized recovery summaries must survive auto-reset.
            "recovery/first_step_latency": metrics.get("recovery/first_step_latency", jp.zeros(())),
            "recovery/first_liftoff_latency": metrics.get("recovery/first_liftoff_latency", jp.zeros(())),
            "recovery/first_touchdown_latency": metrics.get("recovery/first_touchdown_latency", jp.zeros(())),
            "recovery/touchdown_count": metrics.get("recovery/touchdown_count", jp.zeros(())),
            "recovery/support_foot_changes": metrics.get("recovery/support_foot_changes", jp.zeros(())),
            "recovery/post_push_velocity": metrics.get("recovery/post_push_velocity", jp.zeros(())),
            "recovery/completed": metrics.get("recovery/completed", jp.zeros(())),
            "recovery/no_touchdown_frac": metrics.get("recovery/no_touchdown_frac", jp.zeros(())),
            "recovery/touchdown_then_fail_frac": metrics.get(
                "recovery/touchdown_then_fail_frac", jp.zeros(())
            ),
            "recovery/first_step_dx": metrics.get("recovery/first_step_dx", jp.zeros(())),
            "recovery/first_step_dy": metrics.get("recovery/first_step_dy", jp.zeros(())),
            "recovery/first_step_target_err_x": metrics.get(
                "recovery/first_step_target_err_x", jp.zeros(())
            ),
            "recovery/first_step_target_err_y": metrics.get(
                "recovery/first_step_target_err_y", jp.zeros(())
            ),
            "recovery/first_step_dist_abs": metrics.get("recovery/first_step_dist_abs", jp.zeros(())),
            "recovery/pitch_rate_at_push_end": metrics.get(
                "recovery/pitch_rate_at_push_end", jp.zeros(())
            ),
            "recovery/pitch_rate_at_touchdown": metrics.get(
                "recovery/pitch_rate_at_touchdown", jp.zeros(())
            ),
            "recovery/pitch_rate_reduction_10t": metrics.get(
                "recovery/pitch_rate_reduction_10t", jp.zeros(())
            ),
            "recovery/capture_error_at_push_end": metrics.get(
                "recovery/capture_error_at_push_end", jp.zeros(())
            ),
            "recovery/capture_error_at_touchdown": metrics.get(
                "recovery/capture_error_at_touchdown", jp.zeros(())
            ),
            "recovery/capture_error_reduction_10t": metrics.get(
                "recovery/capture_error_reduction_10t", jp.zeros(())
            ),
            "recovery/touchdown_to_term_steps": metrics.get(
                "recovery/touchdown_to_term_steps", jp.zeros(())
            ),
            "recovery/visible_step_rate": metrics.get("recovery/visible_step_rate", jp.zeros(())),
            "recovery/min_height": metrics.get("recovery/min_height", jp.zeros(())),
            "recovery/max_knee_flex": metrics.get("recovery/max_knee_flex", jp.zeros(())),
            "visible_step_rate_hard": metrics.get("visible_step_rate_hard", jp.zeros(())),
            "unnecessary_step_rate": metrics.get("unnecessary_step_rate", jp.zeros(())),
            "teacher/active_frac": metrics.get("teacher/active_frac", jp.zeros(())),
            "teacher/active_count": metrics.get("teacher/active_count", jp.zeros(())),
            "teacher/step_required_mean": metrics.get("teacher/step_required_mean", jp.zeros(())),
            "teacher/step_required_hard_frac": metrics.get(
                "teacher/step_required_hard_frac", jp.zeros(())
            ),
            "teacher/target_step_x_mean": metrics.get("teacher/target_step_x_mean", jp.zeros(())),
            "teacher/target_step_y_mean": metrics.get("teacher/target_step_y_mean", jp.zeros(())),
            "teacher/swing_left_frac": metrics.get("teacher/swing_left_frac", jp.zeros(())),
            "teacher/target_xy_error": metrics.get("teacher/target_xy_error", jp.zeros(())),
            "teacher/teacher_active_during_clean_frac": metrics.get(
                "teacher/teacher_active_during_clean_frac", jp.zeros(())
            ),
            "teacher/raw_step_required_during_clean_mean": metrics.get(
                "teacher/raw_step_required_during_clean_mean", jp.zeros(())
            ),
            "teacher/clean_step_count": metrics.get("teacher/clean_step_count", jp.zeros(())),
            "teacher/push_step_count": metrics.get("teacher/push_step_count", jp.zeros(())),
            "teacher/teacher_active_during_push_frac": metrics.get(
                "teacher/teacher_active_during_push_frac", jp.zeros(())
            ),
            "teacher/reachable_frac": metrics.get("teacher/reachable_frac", jp.zeros(())),
            "teacher/target_step_x_min": metrics.get("teacher/target_step_x_min", jp.zeros(())),
            "teacher/target_step_x_max": metrics.get("teacher/target_step_x_max", jp.zeros(())),
            "teacher/target_step_y_min": metrics.get("teacher/target_step_y_min", jp.zeros(())),
            "teacher/target_step_y_max": metrics.get("teacher/target_step_y_max", jp.zeros(())),
            "teacher/target_step_x_sq_mean": metrics.get(
                "teacher/target_step_x_sq_mean", jp.zeros(())
            ),
            "teacher/target_step_y_sq_mean": metrics.get(
                "teacher/target_step_y_sq_mean", jp.zeros(())
            ),
            "teacher/swing_foot": metrics.get(
                "teacher/swing_foot", jp.asarray(-1, dtype=jp.int32)
            ).astype(jp.float32),
            "teacher/target_xy_error_events": metrics.get(
                "teacher/target_xy_error_events", jp.zeros(())
            ),
            "teacher/target_step_x_std": metrics.get("teacher/target_step_x_std", jp.zeros(())),
            "teacher/target_step_y_std": metrics.get("teacher/target_step_y_std", jp.zeros(())),
            "teacher/step_required_event_rate": metrics.get(
                "teacher/step_required_event_rate", jp.zeros(())
            ),
            "teacher/swing_foot_match_frac": metrics.get("teacher/swing_foot_match_frac", jp.zeros(())),
            "teacher/swing_foot_match_events": metrics.get(
                "teacher/swing_foot_match_events", jp.zeros(())
            ),
            "teacher/whole_body_active_frac": metrics.get(
                "teacher/whole_body_active_frac", jp.zeros(())
            ),
            "teacher/whole_body_active_count": metrics.get(
                "teacher/whole_body_active_count", jp.zeros(())
            ),
            "teacher/whole_body_active_during_clean_frac": metrics.get(
                "teacher/whole_body_active_during_clean_frac", jp.zeros(())
            ),
            "teacher/whole_body_active_during_push_frac": metrics.get(
                "teacher/whole_body_active_during_push_frac", jp.zeros(())
            ),
            "teacher/recovery_height_target_mean": metrics.get(
                "teacher/recovery_height_target_mean", jp.zeros(())
            ),
            "teacher/recovery_height_error": metrics.get(
                "teacher/recovery_height_error", jp.zeros(())
            ),
            "teacher/recovery_height_in_band_frac": metrics.get(
                "teacher/recovery_height_in_band_frac", jp.zeros(())
            ),
            "teacher/com_velocity_target_mean": metrics.get(
                "teacher/com_velocity_target_mean", jp.zeros(())
            ),
            "teacher/com_velocity_error": metrics.get(
                "teacher/com_velocity_error", jp.zeros(())
            ),
            "teacher/com_velocity_target_hit_frac": metrics.get(
                "teacher/com_velocity_target_hit_frac", jp.zeros(())
            ),
            "visible_step_rate_hard": metrics.get("visible_step_rate_hard", jp.zeros(())),
            # v0.17.3: preserve architecture-pivot controller debug metrics
            "debug/mpc_planner_active": metrics.get("debug/mpc_planner_active", jp.zeros(())),
            "debug/mpc_controller_active": metrics.get("debug/mpc_controller_active", jp.zeros(())),
            "debug/mpc_target_com_x": metrics.get("debug/mpc_target_com_x", jp.zeros(())),
            "debug/mpc_target_com_y": metrics.get("debug/mpc_target_com_y", jp.zeros(())),
            "debug/mpc_target_step_x": metrics.get("debug/mpc_target_step_x", jp.zeros(())),
            "debug/mpc_target_step_y": metrics.get("debug/mpc_target_step_y", jp.zeros(())),
            "debug/mpc_support_state": metrics.get("debug/mpc_support_state", jp.zeros(())),
            "debug/mpc_step_requested": metrics.get("debug/mpc_step_requested", jp.zeros(())),
            "tracking/cmd_vs_achieved_forward": metrics.get(
                "tracking/cmd_vs_achieved_forward", jp.zeros(())
            ),
            "tracking/loc_ref_phase_progress": metrics.get(
                "tracking/loc_ref_phase_progress", jp.zeros(())
            ),
            "tracking/loc_ref_stance_foot": metrics.get(
                "tracking/loc_ref_stance_foot", jp.zeros(())
            ),
            "tracking/loc_ref_mode_id": metrics.get(
                "tracking/loc_ref_mode_id", jp.zeros(())
            ),
            "tracking/loc_ref_progression_permission": metrics.get(
                "tracking/loc_ref_progression_permission", jp.zeros(())
            ),
            "tracking/nominal_q_abs_mean": metrics.get(
                "tracking/nominal_q_abs_mean", jp.zeros(())
            ),
            "tracking/residual_q_abs_mean": metrics.get(
                "tracking/residual_q_abs_mean", jp.zeros(())
            ),
            "tracking/residual_q_abs_max": metrics.get(
                "tracking/residual_q_abs_max", jp.zeros(())
            ),
            "tracking/loc_ref_left_reachable": metrics.get(
                "tracking/loc_ref_left_reachable", jp.zeros(())
            ),
            "tracking/loc_ref_right_reachable": metrics.get(
                "tracking/loc_ref_right_reachable", jp.zeros(())
            ),
            "debug/m3_pelvis_orientation_error": metrics.get(
                "debug/m3_pelvis_orientation_error", jp.zeros(())
            ),
            "debug/m3_pelvis_height_error": metrics.get(
                "debug/m3_pelvis_height_error", jp.zeros(())
            ),
            "debug/m3_swing_pos_error": metrics.get(
                "debug/m3_swing_pos_error", jp.zeros(())
            ),
            "debug/m3_swing_vel_error": metrics.get(
                "debug/m3_swing_vel_error", jp.zeros(())
            ),
            "debug/m3_foothold_error": metrics.get(
                "debug/m3_foothold_error", jp.zeros(())
            ),
            "debug/m3_impact_force": metrics.get(
                "debug/m3_impact_force", jp.zeros(())
            ),
            "debug/loc_ref_speed_scale": metrics.get(
                "debug/loc_ref_speed_scale", jp.zeros(())
            ),
            "debug/loc_ref_phase_scale": metrics.get(
                "debug/loc_ref_phase_scale", jp.zeros(())
            ),
            "debug/loc_ref_overspeed": metrics.get(
                "debug/loc_ref_overspeed", jp.zeros(())
            ),
            "debug/loc_ref_support_health": metrics.get(
                "debug/loc_ref_support_health", jp.zeros(())
            ),
            "debug/loc_ref_support_instability": metrics.get(
                "debug/loc_ref_support_instability", jp.zeros(())
            ),
            "debug/loc_ref_nominal_vs_applied_q_l1": metrics.get(
                "debug/loc_ref_nominal_vs_applied_q_l1", jp.zeros(())
            ),
            "debug/loc_ref_applied_q_abs_mean": metrics.get(
                "debug/loc_ref_applied_q_abs_mean", jp.zeros(())
            ),
            "debug/loc_ref_nominal_q_abs_mean": metrics.get(
                "debug/loc_ref_nominal_q_abs_mean", jp.zeros(())
            ),
            "debug/loc_ref_swing_x_target": metrics.get(
                "debug/loc_ref_swing_x_target", jp.zeros(())
            ),
            "debug/loc_ref_swing_x_actual": metrics.get(
                "debug/loc_ref_swing_x_actual", jp.zeros(())
            ),
            "debug/loc_ref_swing_x_error": metrics.get(
                "debug/loc_ref_swing_x_error", jp.zeros(())
            ),
            "debug/loc_ref_swing_y_target": metrics.get(
                "debug/loc_ref_swing_y_target", jp.zeros(())
            ),
            "debug/loc_ref_swing_y_actual": metrics.get(
                "debug/loc_ref_swing_y_actual", jp.zeros(())
            ),
            "debug/loc_ref_swing_y_error": metrics.get(
                "debug/loc_ref_swing_y_error", jp.zeros(())
            ),
            "debug/loc_ref_left_foot_y_actual": metrics.get(
                "debug/loc_ref_left_foot_y_actual", jp.zeros(())
            ),
            "debug/loc_ref_right_foot_y_actual": metrics.get(
                "debug/loc_ref_right_foot_y_actual", jp.zeros(())
            ),
            "debug/loc_ref_support_width_y_actual": metrics.get(
                "debug/loc_ref_support_width_y_actual", jp.zeros(())
            ),
            "debug/loc_ref_support_width_y_nominal": metrics.get(
                "debug/loc_ref_support_width_y_nominal", jp.zeros(())
            ),
            "debug/loc_ref_support_width_y_commanded": metrics.get(
                "debug/loc_ref_support_width_y_commanded", jp.zeros(())
            ),
            "debug/loc_ref_base_support_y": metrics.get(
                "debug/loc_ref_base_support_y", jp.zeros(())
            ),
            "debug/loc_ref_lateral_release_y": metrics.get(
                "debug/loc_ref_lateral_release_y", jp.zeros(())
            ),
            "debug/loc_ref_pelvis_roll_target": metrics.get(
                "debug/loc_ref_pelvis_roll_target", jp.zeros(())
            ),
            "debug/loc_ref_hip_roll_left_target": metrics.get(
                "debug/loc_ref_hip_roll_left_target", jp.zeros(())
            ),
            "debug/loc_ref_hip_roll_right_target": metrics.get(
                "debug/loc_ref_hip_roll_right_target", jp.zeros(())
            ),
            "debug/loc_ref_pelvis_pitch_target": metrics.get(
                "debug/loc_ref_pelvis_pitch_target", jp.zeros(())
            ),
            "debug/loc_ref_root_pitch": metrics.get(
                "debug/loc_ref_root_pitch", jp.zeros(())
            ),
            "debug/loc_ref_root_pitch_rate": metrics.get(
                "debug/loc_ref_root_pitch_rate", jp.zeros(())
            ),
            "debug/loc_ref_stance_sagittal_target": metrics.get(
                "debug/loc_ref_stance_sagittal_target", jp.zeros(())
            ),
            "debug/loc_ref_swing_x_scale": metrics.get(
                "debug/loc_ref_swing_x_scale", jp.zeros(())
            ),
            "debug/loc_ref_pelvis_pitch_scale": metrics.get(
                "debug/loc_ref_pelvis_pitch_scale", jp.zeros(())
            ),
            "debug/loc_ref_support_gate_active": metrics.get(
                "debug/loc_ref_support_gate_active", jp.zeros(())
            ),
            "debug/loc_ref_hybrid_mode_id": metrics.get(
                "debug/loc_ref_hybrid_mode_id", jp.zeros(())
            ),
            "debug/loc_ref_progression_permission": metrics.get(
                "debug/loc_ref_progression_permission", jp.zeros(())
            ),
            "debug/loc_ref_swing_x_scale_active": metrics.get(
                "debug/loc_ref_swing_x_scale_active", jp.zeros(())
            ),
            "debug/loc_ref_phase_scale_active": metrics.get(
                "debug/loc_ref_phase_scale_active", jp.zeros(())
            ),
        }

        # v0.10.2: Also preserve truncated flag in wr info for success rate calculation
        # We need to preserve the original wr_info.truncated (1.0 if success) through auto-reset
        #
        # v0.10.4 FIX: step_count must RESET to 0 for new episode, not accumulate!
        # The terminal step count is passed separately for logging.
        # Key insight: preserved_wr_info is what the NEXT episode sees, so step_count=0.
        reset_wr_info = reset_state.info[WR_INFO_KEY]
        preserved_wr_info = WildRobotInfo(
            step_count=reset_wr_info.step_count,  # Reset to 0 for new episode (NOT new_wr_info.step_count!)
            prev_action=reset_wr_info.prev_action,
            pending_action=reset_wr_info.pending_action,
            truncated=new_wr_info.truncated,  # Preserve original truncated flag for success rate
            velocity_cmd=reset_wr_info.velocity_cmd,  # New velocity for new episode
            prev_root_pos=reset_wr_info.prev_root_pos,
            prev_root_quat=reset_wr_info.prev_root_quat,
            prev_left_foot_pos=reset_wr_info.prev_left_foot_pos,
            prev_right_foot_pos=reset_wr_info.prev_right_foot_pos,
            imu_quat_hist=reset_wr_info.imu_quat_hist,
            imu_gyro_hist=reset_wr_info.imu_gyro_hist,
            foot_contacts=reset_wr_info.foot_contacts,
            root_height=reset_wr_info.root_height,
            prev_left_loaded=reset_wr_info.prev_left_loaded,
            prev_right_loaded=reset_wr_info.prev_right_loaded,
            cycle_start_forward_x=reset_wr_info.cycle_start_forward_x,
            last_touchdown_root_pos=reset_wr_info.last_touchdown_root_pos,
            last_touchdown_foot=reset_wr_info.last_touchdown_foot,
            critic_obs=reset_wr_info.critic_obs,
            loc_ref_phase_time=reset_wr_info.loc_ref_phase_time,
            loc_ref_stance_foot=reset_wr_info.loc_ref_stance_foot,
            loc_ref_switch_count=reset_wr_info.loc_ref_switch_count,
            loc_ref_mode_id=reset_wr_info.loc_ref_mode_id,
            loc_ref_mode_time=reset_wr_info.loc_ref_mode_time,
            loc_ref_startup_route_progress=reset_wr_info.loc_ref_startup_route_progress,
            loc_ref_startup_route_ceiling=reset_wr_info.loc_ref_startup_route_ceiling,
            loc_ref_startup_route_stage_id=reset_wr_info.loc_ref_startup_route_stage_id,
            loc_ref_startup_route_transition_reason=reset_wr_info.loc_ref_startup_route_transition_reason,
            loc_ref_gait_phase_sin=reset_wr_info.loc_ref_gait_phase_sin,
            loc_ref_gait_phase_cos=reset_wr_info.loc_ref_gait_phase_cos,
            loc_ref_next_foothold=reset_wr_info.loc_ref_next_foothold,
            loc_ref_swing_pos=reset_wr_info.loc_ref_swing_pos,
            loc_ref_swing_vel=reset_wr_info.loc_ref_swing_vel,
            loc_ref_pelvis_height=reset_wr_info.loc_ref_pelvis_height,
            loc_ref_pelvis_roll=reset_wr_info.loc_ref_pelvis_roll,
            loc_ref_pelvis_pitch=reset_wr_info.loc_ref_pelvis_pitch,
            loc_ref_history=reset_wr_info.loc_ref_history,
            nominal_q_ref=reset_wr_info.nominal_q_ref,
            push_schedule=reset_wr_info.push_schedule,
            # M3 FSM: reset to STANCE on new episode
            fsm_phase=reset_wr_info.fsm_phase,
            fsm_swing_foot=reset_wr_info.fsm_swing_foot,
            fsm_phase_ticks=reset_wr_info.fsm_phase_ticks,
            fsm_frozen_tx=reset_wr_info.fsm_frozen_tx,
            fsm_frozen_ty=reset_wr_info.fsm_frozen_ty,
            fsm_swing_sx=reset_wr_info.fsm_swing_sx,
            fsm_swing_sy=reset_wr_info.fsm_swing_sy,
            fsm_touch_hold=reset_wr_info.fsm_touch_hold,
            fsm_trigger_hold=reset_wr_info.fsm_trigger_hold,
            # v0.17.1+: Recovery metrics state (reset to initial values)
            recovery_active=reset_wr_info.recovery_active,
            recovery_age=reset_wr_info.recovery_age,
            recovery_last_support_foot=reset_wr_info.recovery_last_support_foot,
            recovery_first_liftoff_recorded=reset_wr_info.recovery_first_liftoff_recorded,
            recovery_first_liftoff_latency=reset_wr_info.recovery_first_liftoff_latency,
            recovery_first_touchdown_recorded=reset_wr_info.recovery_first_touchdown_recorded,
            recovery_first_step_latency=reset_wr_info.recovery_first_step_latency,
            recovery_first_touchdown_age=reset_wr_info.recovery_first_touchdown_age,
            recovery_visible_step_recorded=reset_wr_info.recovery_visible_step_recorded,
            recovery_touchdown_count=reset_wr_info.recovery_touchdown_count,
            recovery_support_foot_changes=reset_wr_info.recovery_support_foot_changes,
            recovery_pitch_rate_at_push_end=reset_wr_info.recovery_pitch_rate_at_push_end,
            recovery_pitch_rate_at_touchdown=reset_wr_info.recovery_pitch_rate_at_touchdown,
            recovery_pitch_rate_after_10t=reset_wr_info.recovery_pitch_rate_after_10t,
            recovery_capture_error_at_push_end=reset_wr_info.recovery_capture_error_at_push_end,
            recovery_capture_error_at_touchdown=reset_wr_info.recovery_capture_error_at_touchdown,
            recovery_capture_error_after_10t=reset_wr_info.recovery_capture_error_after_10t,
            recovery_first_step_dx=reset_wr_info.recovery_first_step_dx,
            recovery_first_step_dy=reset_wr_info.recovery_first_step_dy,
            recovery_first_step_target_err_x=reset_wr_info.recovery_first_step_target_err_x,
            recovery_first_step_target_err_y=reset_wr_info.recovery_first_step_target_err_y,
            recovery_min_height=reset_wr_info.recovery_min_height,
            recovery_max_knee_flex=reset_wr_info.recovery_max_knee_flex,
            teacher_active=reset_wr_info.teacher_active,
            teacher_step_required_soft=reset_wr_info.teacher_step_required_soft,
            teacher_step_required_hard=reset_wr_info.teacher_step_required_hard,
            teacher_swing_foot=reset_wr_info.teacher_swing_foot,
            teacher_target_step_x=reset_wr_info.teacher_target_step_x,
            teacher_target_step_y=reset_wr_info.teacher_target_step_y,
            teacher_target_reachable=reset_wr_info.teacher_target_reachable,
            mpc_planner_active=reset_wr_info.mpc_planner_active,
            mpc_controller_active=reset_wr_info.mpc_controller_active,
            mpc_target_com_x=reset_wr_info.mpc_target_com_x,
            mpc_target_com_y=reset_wr_info.mpc_target_com_y,
            mpc_target_step_x=reset_wr_info.mpc_target_step_x,
            mpc_target_step_y=reset_wr_info.mpc_target_step_y,
            mpc_support_state=reset_wr_info.mpc_support_state,
            mpc_step_requested=reset_wr_info.mpc_step_requested,
            domain_rand_friction_scale=reset_wr_info.domain_rand_friction_scale,
            domain_rand_mass_scales=reset_wr_info.domain_rand_mass_scales,
            domain_rand_kp_scales=reset_wr_info.domain_rand_kp_scales,
            domain_rand_frictionloss_scales=reset_wr_info.domain_rand_frictionloss_scales,
            domain_rand_joint_offsets=reset_wr_info.domain_rand_joint_offsets,
        )
        preserved_info = dict(reset_state.info)  # Copy wrapper fields
        preserved_info[WR_INFO_KEY] = preserved_wr_info  # Update wr namespace

        final_data, final_obs, final_metrics, final_info, final_rng = jax.lax.cond(
            done > 0.5,
            lambda: (
                reset_state.data,
                reset_state.obs,
                preserved_metrics,  # Use preserved metrics with term/* intact
                preserved_info,  # Use preserved info with truncated intact
                reset_state.rng,
            ),
            lambda: (data, obs, metrics, info, rng_after),
        )

        # Build packed metrics vector and store in metrics dict
        final_metrics_vec = build_metrics_vec(final_metrics)
        final_metrics[METRICS_VEC_KEY] = final_metrics_vec

        return WildRobotEnvState(
            data=final_data,
            obs=final_obs,
            reward=reward,  # Keep original reward (from before reset)
            done=done,  # Keep done=True so algorithm knows episode ended
            metrics=final_metrics,
            info=final_info,
            pipeline_state=final_data,
            rng=final_rng,
        )

    def _get_privileged_critic_obs(
        self,
        data: mjx.Data,
        step_count: jax.Array,
        push_schedule: DisturbanceSchedule,
    ) -> jax.Array:
        """Build sim-only critic features without changing the actor contract."""
        root_pose = self._cal.get_root_pose(data)
        roll, pitch, _ = root_pose.euler_angles()
        root_vel = self._cal.get_root_velocity(
            data, frame=CoordinateFrame.HEADING_LOCAL
        )
        push_active = (
            (step_count >= push_schedule.start_step)
            & (step_count < push_schedule.end_step)
            & jp.asarray(self._config.env.push_enabled, dtype=jp.bool_)
        ).astype(jp.float32)
        push_force_xy = jp.where(push_active > 0.5, push_schedule.force_xy, 0.0)
        critic_obs = jp.array(
            [
                root_vel.linear_xyz[0],
                root_vel.linear_xyz[1],
                root_vel.linear_xyz[2],
                root_vel.angular_xyz[0],
                root_vel.angular_xyz[1],
                root_vel.angular_xyz[2],
                roll,
                pitch,
                root_pose.height,
                push_force_xy[0],
                push_force_xy[1],
                push_active,
            ],
            dtype=jp.float32,
        )
        return jp.reshape(critic_obs, (PRIVILEGED_OBS_DIM,))

    # Exposed for focused tests.
    _loc_ref_support_scales = staticmethod(_loc_ref_support_scales)
    _loc_ref_support_health = staticmethod(_loc_ref_support_health)

    def _joint_tracking_error(
        self, actual_q: jax.Array, target_q: jax.Array, idx: jax.Array | int
    ) -> jax.Array:
        """Absolute tracking error for one actuator index (safe when idx is missing)."""
        idx_arr = jp.asarray(idx, dtype=jp.int32)
        idx_safe = jp.maximum(idx_arr, jp.asarray(0, dtype=jp.int32))
        err = jp.abs(
            jp.asarray(actual_q[idx_safe], dtype=jp.float32)
            - jp.asarray(target_q[idx_safe], dtype=jp.float32)
        )
        return jp.where(idx_arr >= 0, err, jp.asarray(0.0, dtype=jp.float32))

    def _stance_leg_tracking_errors(
        self,
        actual_q: jax.Array,
        target_q: jax.Array,
        stance_foot_id: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Stance knee/ankle target-vs-actual errors used by startup readiness governor."""
        stance_is_left = jp.asarray(stance_foot_id, dtype=jp.int32) == jp.asarray(
            REF_LEFT_STANCE, dtype=jp.int32
        )
        knee_idx = jp.where(
            stance_is_left,
            jp.asarray(self._idx_left_knee_pitch, dtype=jp.int32),
            jp.asarray(self._idx_right_knee_pitch, dtype=jp.int32),
        )
        ankle_idx = jp.where(
            stance_is_left,
            jp.asarray(self._idx_left_ankle_pitch, dtype=jp.int32),
            jp.asarray(self._idx_right_ankle_pitch, dtype=jp.int32),
        )
        knee_err = self._joint_tracking_error(actual_q, target_q, knee_idx)
        ankle_err = self._joint_tracking_error(actual_q, target_q, ankle_idx)
        return knee_err, ankle_err

    def _limit_joint_target_step(
        self,
        target_q: jax.Array,
        prev_target_q: jax.Array,
        idx: jax.Array | int,
        max_delta_rad: jax.Array,
    ) -> jax.Array:
        """Clamp per-step target change for one joint index."""
        idx_arr = jp.asarray(idx, dtype=jp.int32)
        idx_safe = jp.maximum(idx_arr, jp.asarray(0, dtype=jp.int32))
        desired_delta = jp.asarray(target_q[idx_safe] - prev_target_q[idx_safe], dtype=jp.float32)
        clipped_delta = jp.clip(
            desired_delta,
            -jp.asarray(max_delta_rad, dtype=jp.float32),
            jp.asarray(max_delta_rad, dtype=jp.float32),
        )
        limited_value = jp.asarray(prev_target_q[idx_safe], dtype=jp.float32) + clipped_delta
        return jax.lax.cond(
            idx_arr >= 0,
            lambda q: q.at[idx_safe].set(limited_value),
            lambda q: q,
            target_q,
        )

    def _apply_startup_support_rate_limiter(
        self,
        nominal_q_ref: jax.Array,
        prev_nominal_q_ref: jax.Array,
        stance_foot_id: jax.Array,
        mode_id: jax.Array,
        mode_time_s: jax.Array,
    ) -> jax.Array:
        """Limit startup and support stance hip/knee/ankle target rates.

        Uses the design rate (30 deg/s) during startup and early support entry,
        and the hard-cap rate (100 deg/s) for the remainder of SUPPORT_STABILIZE.

        During startup (double support), BOTH legs are rate-limited because
        both are load-bearing.  After handoff to SUPPORT_STABILIZE, only the
        stance leg is rate-limited.
        """
        if self._walking_ref_v2_cfg.debug_force_support_only:
            return nominal_q_ref
        mode_i = jp.asarray(mode_id, dtype=jp.int32)
        in_startup = mode_i == jp.asarray(
            int(WalkingRefV2Mode.STARTUP_SUPPORT_RAMP), dtype=jp.int32
        )
        in_support = mode_i == jp.asarray(
            int(WalkingRefV2Mode.SUPPORT_STABILIZE), dtype=jp.int32
        )
        in_support_entry = in_support & (
            jp.asarray(mode_time_s, dtype=jp.float32)
            < jp.asarray(self._walking_ref_v2_cfg.support_entry_shaping_window_s, dtype=jp.float32)
        )
        limiter_active = jp.logical_or(in_startup, in_support)
        use_design_rate = jp.logical_or(in_startup, in_support_entry)

        stance_is_left = jp.asarray(stance_foot_id, dtype=jp.int32) == jp.asarray(
            REF_LEFT_STANCE, dtype=jp.int32
        )
        # Stance leg indices (used in support mode).
        stance_hip_idx = jp.where(
            stance_is_left,
            jp.asarray(self._idx_left_hip_pitch, dtype=jp.int32),
            jp.asarray(self._idx_right_hip_pitch, dtype=jp.int32),
        )
        stance_knee_idx = jp.where(
            stance_is_left,
            jp.asarray(self._idx_left_knee_pitch, dtype=jp.int32),
            jp.asarray(self._idx_right_knee_pitch, dtype=jp.int32),
        )
        stance_ankle_idx = jp.where(
            stance_is_left,
            jp.asarray(self._idx_left_ankle_pitch, dtype=jp.int32),
            jp.asarray(self._idx_right_ankle_pitch, dtype=jp.int32),
        )
        design_step = jp.asarray(
            self._walking_ref_v2_cfg.startup_target_rate_design_rad_s * self.dt,
            dtype=jp.float32,
        )
        hard_step = jp.asarray(
            self._walking_ref_v2_cfg.startup_target_rate_hard_cap_rad_s * self.dt,
            dtype=jp.float32,
        )
        max_step = jp.where(use_design_rate, jp.minimum(design_step, hard_step), hard_step)
        # During startup, rate-limit BOTH legs (both are load-bearing).
        # After handoff to support, only rate-limit the stance leg.
        limited = self._limit_joint_target_step(
            target_q=nominal_q_ref,
            prev_target_q=prev_nominal_q_ref,
            idx=stance_hip_idx,
            max_delta_rad=max_step,
        )
        limited = self._limit_joint_target_step(
            target_q=limited,
            prev_target_q=prev_nominal_q_ref,
            idx=stance_knee_idx,
            max_delta_rad=max_step,
        )
        limited = self._limit_joint_target_step(
            target_q=limited,
            prev_target_q=prev_nominal_q_ref,
            idx=stance_ankle_idx,
            max_delta_rad=max_step,
        )
        # During startup (double support): also rate-limit the swing leg.
        swing_hip_idx = jp.where(
            stance_is_left,
            jp.asarray(self._idx_right_hip_pitch, dtype=jp.int32),
            jp.asarray(self._idx_left_hip_pitch, dtype=jp.int32),
        )
        swing_knee_idx = jp.where(
            stance_is_left,
            jp.asarray(self._idx_right_knee_pitch, dtype=jp.int32),
            jp.asarray(self._idx_left_knee_pitch, dtype=jp.int32),
        )
        swing_ankle_idx = jp.where(
            stance_is_left,
            jp.asarray(self._idx_right_ankle_pitch, dtype=jp.int32),
            jp.asarray(self._idx_left_ankle_pitch, dtype=jp.int32),
        )
        swing_limited = self._limit_joint_target_step(
            target_q=limited,
            prev_target_q=prev_nominal_q_ref,
            idx=swing_hip_idx,
            max_delta_rad=max_step,
        )
        swing_limited = self._limit_joint_target_step(
            target_q=swing_limited,
            prev_target_q=prev_nominal_q_ref,
            idx=swing_knee_idx,
            max_delta_rad=max_step,
        )
        swing_limited = self._limit_joint_target_step(
            target_q=swing_limited,
            prev_target_q=prev_nominal_q_ref,
            idx=swing_ankle_idx,
            max_delta_rad=max_step,
        )
        limited = jp.where(in_startup, swing_limited, limited)
        return jax.lax.cond(limiter_active, lambda q: q, lambda _q: nominal_q_ref, limited)

    def _compute_nominal_q_ref_from_loc_ref(
        self,
        *,
        stance_foot_id: jax.Array,
        swing_pos: jax.Array,
        swing_vel: jax.Array,
        pelvis_height: jax.Array,
        pelvis_roll: jax.Array,
        pelvis_pitch: jax.Array,
        left_foot_pos_h: jax.Array,
        right_foot_pos_h: jax.Array,
        root_pitch: jax.Array,
        root_pitch_rate: jax.Array,
        forward_vel_h: jax.Array,
        velocity_cmd: jax.Array,
        support_health: jax.Array,
        mode_id: jax.Array,
    ) -> tuple[
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
    ]:
        """Compute nominal joint-space target from locomotion reference fields."""
        support_health = jp.clip(jp.asarray(support_health, dtype=jp.float32), 0.0, 1.0)
        # Use support-specific margin for STARTUP_SUPPORT_RAMP (0) and SUPPORT_STABILIZE (1).
        # Larger margin → less negative nominal_stance_z → shorter effective leg → MORE knee flexion.
        is_support_mode = jp.logical_or(
            jp.asarray(mode_id, dtype=jp.int32) == jp.asarray(int(WalkingRefV2Mode.STARTUP_SUPPORT_RAMP), dtype=jp.int32),
            jp.asarray(mode_id, dtype=jp.int32) == jp.asarray(int(WalkingRefV2Mode.SUPPORT_STABILIZE), dtype=jp.int32),
        )
        base_margin = jp.where(
            is_support_mode,
            self._loc_ref_v2_support_stance_extension_margin_m,
            self._loc_ref_stance_extension_margin_m,
        )
        stance_margin = base_margin * (0.2 + 0.8 * support_health)
        nominal_stance_z = -jp.maximum(
            jp.asarray(0.0, dtype=jp.float32),
            jp.asarray(pelvis_height, dtype=jp.float32) - stance_margin,
        )
        swing_x_ref = jp.asarray(swing_pos[0], dtype=jp.float32)
        swing_z_ref = jp.asarray(swing_pos[2], dtype=jp.float32)
        swing_vx_ref = jp.asarray(swing_vel[0], dtype=jp.float32)

        stance_is_left = jp.asarray(stance_foot_id, dtype=jp.int32) == jp.asarray(
            REF_LEFT_STANCE, dtype=jp.int32
        )
        stance_pos = jp.where(stance_is_left, left_foot_pos_h, right_foot_pos_h)
        swing_pos_actual = jp.where(stance_is_left, right_foot_pos_h, left_foot_pos_h)
        swing_pos_stance = swing_pos_actual - stance_pos
        swing_x_actual = jp.asarray(swing_pos_stance[0], dtype=jp.float32)
        swing_z_actual = jp.asarray(swing_pos_stance[2], dtype=jp.float32)
        swing_x_scale, pelvis_pitch_scale, _, support_gate_active = _loc_ref_support_scales(
            velocity_cmd=velocity_cmd,
            forward_vel_h=forward_vel_h,
            pitch_rad=root_pitch,
            swing_x_brake_pitch_start_rad=self._loc_ref_swing_x_brake_pitch_start_rad,
            swing_x_brake_overspeed_deadband=self._loc_ref_swing_x_brake_overspeed_deadband,
            swing_x_brake_gain=self._loc_ref_swing_x_brake_gain,
            swing_x_min_scale=self._loc_ref_swing_x_min_scale,
            pelvis_pitch_brake_gain=self._loc_ref_pelvis_pitch_brake_gain,
            pelvis_pitch_min_scale=self._loc_ref_pelvis_pitch_min_scale,
        )
        swing_x_scale = jp.maximum(
            self._loc_ref_swing_x_min_scale, swing_x_scale * support_health
        )
        pelvis_pitch_scale = jp.maximum(
            self._loc_ref_pelvis_pitch_min_scale, pelvis_pitch_scale * support_health
        )
        support_gate_active = jp.maximum(
            support_gate_active, (support_health < 0.999).astype(jp.float32)
        )
        swing_x_blended = (
            (1.0 - self._loc_ref_swing_target_blend) * swing_x_actual
            + self._loc_ref_swing_target_blend * swing_x_ref
        )
        swing_x_support_target = swing_x_actual + swing_x_scale * (
            swing_x_blended - swing_x_actual
        )
        swing_x_target = jp.clip(
            swing_x_support_target,
            swing_x_actual - self._loc_ref_max_swing_x_delta_m,
            swing_x_actual + self._loc_ref_max_swing_x_delta_m,
        )
        swing_z_blended = (
            (1.0 - self._loc_ref_swing_target_blend) * swing_z_actual
            + self._loc_ref_swing_target_blend * swing_z_ref
        )
        swing_z_target = jp.clip(
            swing_z_blended,
            swing_z_actual - self._loc_ref_max_swing_z_delta_m,
            swing_z_actual + self._loc_ref_max_swing_z_delta_m,
        )
        swing_target_z = nominal_stance_z + swing_z_target
        stance_z_actual = jp.asarray(
            jp.where(stance_is_left, left_foot_pos_h[2], right_foot_pos_h[2]),
            dtype=jp.float32,
        )
        stance_target_z = (
            (1.0 - self._loc_ref_stance_height_blend) * nominal_stance_z
            + self._loc_ref_stance_height_blend * stance_z_actual
        )

        left_target_x = jp.where(stance_is_left, 0.0, swing_x_target)
        left_target_z = jp.where(stance_is_left, stance_target_z, swing_target_z)
        right_target_x = jp.where(stance_is_left, swing_x_target, 0.0)
        right_target_z = jp.where(stance_is_left, swing_target_z, stance_target_z)

        left_hip, left_knee, left_ankle, left_reachable = solve_leg_sagittal_ik_jax(
            target_x_m=left_target_x, target_z_m=left_target_z, config=self._leg_ik_cfg
        )
        right_hip, right_knee, right_ankle, right_reachable = solve_leg_sagittal_ik_jax(
            target_x_m=right_target_x, target_z_m=right_target_z, config=self._leg_ik_cfg
        )

        q_ref = jp.asarray(self._default_joint_qpos, dtype=jp.float32)
        if self._idx_left_hip_pitch >= 0:
            q_ref = q_ref.at[self._idx_left_hip_pitch].set(-left_hip)
        if self._idx_left_knee_pitch >= 0:
            q_ref = q_ref.at[self._idx_left_knee_pitch].set(left_knee)
        if self._idx_left_ankle_pitch >= 0:
            q_ref = q_ref.at[self._idx_left_ankle_pitch].set(left_ankle)
        if self._idx_right_hip_pitch >= 0:
            q_ref = q_ref.at[self._idx_right_hip_pitch].set(right_hip)
        if self._idx_right_knee_pitch >= 0:
            q_ref = q_ref.at[self._idx_right_knee_pitch].set(right_knee)
        if self._idx_right_ankle_pitch >= 0:
            q_ref = q_ref.at[self._idx_right_ankle_pitch].set(right_ankle)

        roll = jp.asarray(pelvis_roll, dtype=jp.float32)
        roll = roll + self._loc_ref_swing_y_to_hip_roll * jp.asarray(swing_pos[1], dtype=jp.float32)
        pitch = jp.asarray(pelvis_pitch, dtype=jp.float32) * pelvis_pitch_scale
        if self._idx_left_hip_roll >= 0:
            q_ref = q_ref.at[self._idx_left_hip_roll].set(-roll)
        if self._idx_right_hip_roll >= 0:
            q_ref = q_ref.at[self._idx_right_hip_roll].set(roll)
        if self._idx_waist_yaw >= 0:
            q_ref = q_ref.at[self._idx_waist_yaw].set(jp.asarray(0.0, dtype=jp.float32))
        if self._idx_left_shoulder_pitch >= 0:
            q_ref = q_ref.at[self._idx_left_shoulder_pitch].set(-0.10 * pitch)
        if self._idx_right_shoulder_pitch >= 0:
            q_ref = q_ref.at[self._idx_right_shoulder_pitch].set(-0.10 * pitch)

        q_ref = jp.clip(q_ref, self._joint_range_mins, self._joint_range_maxs).astype(jp.float32)
        _ = root_pitch_rate  # Reserved for future sagittal gating terms.
        return (
            q_ref,
            left_reachable.astype(jp.float32),
            right_reachable.astype(jp.float32),
            swing_x_target.astype(jp.float32),
            pitch.astype(jp.float32),
            stance_target_z.astype(jp.float32),
            swing_x_scale.astype(jp.float32),
            pelvis_pitch_scale.astype(jp.float32),
            support_gate_active.astype(jp.float32),
        )

    def _compose_loc_ref_residual_action(
        self,
        *,
        policy_action: jax.Array,
        nominal_q_ref: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Compose policy residual around nominal q_ref and map back to action space."""
        residual_delta_q = (
            jp.clip(jp.asarray(policy_action, dtype=jp.float32), -1.0, 1.0)
            * self._residual_q_scale
            * self._joint_half_spans
        )
        target_q = jp.clip(
            jp.asarray(nominal_q_ref, dtype=jp.float32) + residual_delta_q,
            self._joint_range_mins,
            self._joint_range_maxs,
        )
        raw_action = JaxCalibOps.ctrl_to_policy_action(
            spec=self._policy_spec,
            ctrl_rad=target_q,
        ).astype(jp.float32)
        return raw_action, residual_delta_q

    def _mix_base_and_residual_action(
        self, data: mjx.Data, policy_action: jax.Array
    ) -> jax.Array:
        """M2: Apply a joint-heuristic base action and gate residual authority.

        action_applied = action_base + residual_scale(need_step) * policy_action

        Notes:
        - This operates in policy-action space so CAL symmetry correction is preserved.
        - The gating uses the same need_step computation as stepping-trait rewards.
        """
        env_cfg = self._config.env
        weights = self._config.reward_weights

        root_pose = self._cal.get_root_pose(data)
        roll, pitch, _ = root_pose.euler_angles()
        height = root_pose.height
        root_vel = self._cal.get_root_velocity(data, frame=CoordinateFrame.HEADING_LOCAL)
        _, lateral_vel, _ = root_vel.linear_xyz
        roll_rate, pitch_rate, _ = root_vel.angular_xyz

        healthy = jp.where(
            (height > env_cfg.min_height) & (height < env_cfg.max_height), 1.0, 0.0
        )

        need_step = self._compute_need_step(
            pitch=pitch,
            roll=roll,
            lateral_vel=lateral_vel,
            pitch_rate=pitch_rate,
            healthy=healthy,
        )

        # Base action: default pose (often near 0 in policy-action space)
        base_action = self._default_pose_action

        # Conservative uprightness feedback (in policy-action units)
        clip = jp.maximum(jp.asarray(env_cfg.base_ctrl_action_clip, dtype=jp.float32), 0.0)
        pitch_delta = -(
            jp.asarray(env_cfg.base_ctrl_pitch_kp, dtype=jp.float32) * pitch
            + jp.asarray(env_cfg.base_ctrl_pitch_kd, dtype=jp.float32) * pitch_rate
        )
        roll_delta = -(
            jp.asarray(env_cfg.base_ctrl_roll_kp, dtype=jp.float32) * roll
            + jp.asarray(env_cfg.base_ctrl_roll_kd, dtype=jp.float32) * roll_rate
        )
        pitch_delta = jp.clip(pitch_delta, -clip, clip)
        roll_delta = jp.clip(roll_delta, -clip, clip)

        hip_pitch_gain = jp.asarray(env_cfg.base_ctrl_hip_pitch_gain, dtype=jp.float32)
        ankle_pitch_gain = jp.asarray(env_cfg.base_ctrl_ankle_pitch_gain, dtype=jp.float32)
        hip_roll_gain = jp.asarray(env_cfg.base_ctrl_hip_roll_gain, dtype=jp.float32)

        # Apply feedback to a small set of stabilizing joints (if present).
        action = base_action
        action = self._maybe_add_action_delta(action, "left_hip_pitch", hip_pitch_gain * pitch_delta)
        action = self._maybe_add_action_delta(action, "right_hip_pitch", hip_pitch_gain * pitch_delta)
        action = self._maybe_add_action_delta(action, "left_ankle_pitch", ankle_pitch_gain * pitch_delta)
        action = self._maybe_add_action_delta(action, "right_ankle_pitch", ankle_pitch_gain * pitch_delta)
        action = self._maybe_add_action_delta(action, "left_hip_roll", hip_roll_gain * roll_delta)
        action = self._maybe_add_action_delta(action, "right_hip_roll", hip_roll_gain * roll_delta)

        # Residual authority increases when need_step is high.
        residual_min = jp.asarray(env_cfg.residual_scale_min, dtype=jp.float32)
        residual_max = jp.asarray(env_cfg.residual_scale_max, dtype=jp.float32)
        gate_power = jp.maximum(jp.asarray(env_cfg.residual_gate_power, dtype=jp.float32), 0.0)
        t = jp.power(need_step, gate_power)
        residual_scale = residual_min + t * (residual_max - residual_min)

        mixed = action + residual_scale * policy_action
        return jp.clip(mixed, -1.0, 1.0).astype(jp.float32)

    def _compute_need_step(
        self,
        *,
        pitch: jax.Array,
        roll: jax.Array,
        lateral_vel: jax.Array,
        pitch_rate: jax.Array,
        healthy: jax.Array,
    ) -> jax.Array:
        """Shared 0..1 gate for stepping rewards and residual authority.

        Keep this active near collapse. A binary healthy gate suppresses the
        FSM exactly when a rescue step is most needed.
        """
        weights = self._config.reward_weights
        g_pitch = jp.maximum(getattr(weights, "step_need_pitch", 0.35), 1e-6)
        g_roll = jp.maximum(getattr(weights, "step_need_roll", 0.35), 1e-6)
        g_lat = jp.maximum(getattr(weights, "step_need_lat_vel", 0.30), 1e-6)
        g_pr = jp.maximum(getattr(weights, "step_need_pitch_rate", 1.00), 1e-6)
        need_step = (
            jp.abs(pitch) / g_pitch
            + jp.abs(roll) / g_roll
            + jp.abs(lateral_vel) / g_lat
            + jp.abs(pitch_rate) / g_pr
        ) / 4.0
        return jp.clip(need_step, 0.0, 1.0)

    def _mpc_standing_compute_ctrl(
        self,
        data: "mjx.Data",
        policy_action: jax.Array,
    ):
        """v0.17.3 standing bring-up controller path.

        This is a conservative scaffold for architecture pivot validation:
        - initializes/runs in JAX env loop
        - emits inspectable planner/controller debug signals
        - does not claim full MPC/contact-phase optimization
        """
        env_cfg = self._config.env

        root_pose = self._cal.get_root_pose(data)
        roll, pitch, _ = root_pose.euler_angles()
        root_vel = self._cal.get_root_velocity(data, frame=CoordinateFrame.HEADING_LOCAL)
        forward_vel, lateral_vel, _ = root_vel.linear_xyz
        roll_rate, pitch_rate, _ = root_vel.angular_xyz
        height_error = jp.asarray(env_cfg.target_height, dtype=jp.float32) - root_pose.height

        a = self._actuator_name_to_index
        idx_lhp = a.get("left_hip_pitch", -1)
        idx_rhp = a.get("right_hip_pitch", -1)
        idx_lhr = a.get("left_hip_roll", -1)
        idx_rhr = a.get("right_hip_roll", -1)
        idx_lank = a.get("left_ankle_pitch", a.get("left_ankle", -1))
        idx_rank = a.get("right_ankle_pitch", a.get("right_ankle", -1))

        raw_action, debug = compute_mpc_standing_action(
            pitch=pitch,
            roll=roll,
            pitch_rate=pitch_rate,
            roll_rate=roll_rate,
            height_error=height_error,
            forward_vel=forward_vel,
            lateral_vel=lateral_vel,
            default_action=self._default_pose_action,
            policy_action=policy_action,
            hip_pitch_gain=jp.asarray(env_cfg.base_ctrl_hip_pitch_gain, dtype=jp.float32),
            ankle_pitch_gain=jp.asarray(env_cfg.base_ctrl_ankle_pitch_gain, dtype=jp.float32),
            hip_roll_gain=jp.asarray(env_cfg.base_ctrl_hip_roll_gain, dtype=jp.float32),
            idx_left_hip_pitch=idx_lhp,
            idx_right_hip_pitch=idx_rhp,
            idx_left_ankle_pitch=idx_lank,
            idx_right_ankle_pitch=idx_rank,
            idx_left_hip_roll=idx_lhr,
            idx_right_hip_roll=idx_rhr,
            pitch_kp=float(env_cfg.mpc_pitch_kp),
            pitch_kd=float(env_cfg.mpc_pitch_kd),
            roll_kp=float(env_cfg.mpc_roll_kp),
            roll_kd=float(env_cfg.mpc_roll_kd),
            height_kp=float(env_cfg.mpc_height_kp),
            residual_scale=float(env_cfg.mpc_residual_scale),
            action_clip=float(env_cfg.mpc_action_clip),
            step_trigger_threshold=float(env_cfg.mpc_step_trigger_threshold),
        )
        return raw_action, debug

    def _maybe_add_action_delta(
        self, action: jax.Array, actuator_name: str, delta: jax.Array
    ) -> jax.Array:
        idx = self._actuator_name_to_index.get(actuator_name)
        if idx is None:
            return action
        return action.at[idx].add(delta)

    def _fsm_compute_ctrl(
        self,
        data: "mjx.Data",
        wr: "WildRobotInfo",
        policy_action: jax.Array,
    ) -> tuple[jax.Array, tuple]:
        """M3: Foot-placement + arms base controller with gated residual RL.

        Reads FSM state from wr, advances the FSM, computes a stabilising base
        action, then blends in the policy residual.

        Returns:
            (mixed_action, new_fsm_tuple) where new_fsm_tuple is a 9-element
            tuple matching the FSM field order in WildRobotInfo.
        """
        env_cfg = self._config.env
        weights = self._config.reward_weights

        # -- Signals ---------------------------------------------------------
        root_pose = self._cal.get_root_pose(data)
        roll, pitch, _ = root_pose.euler_angles()
        height = root_pose.height
        root_vel = self._cal.get_root_velocity(
            data, frame=CoordinateFrame.HEADING_LOCAL
        )
        forward_vel, lateral_vel, _ = root_vel.linear_xyz
        roll_rate, pitch_rate, _ = root_vel.angular_xyz

        healthy = jp.where(
            (height > env_cfg.min_height) & (height < env_cfg.max_height),
            1.0, 0.0,
        )

        need_step = sc.compute_need_step(
            pitch=pitch,
            roll=roll,
            lateral_vel=lateral_vel,
            pitch_rate=pitch_rate,
            healthy=healthy,
            g_pitch=float(getattr(weights, "step_need_pitch", 0.35)),
            g_roll=float(getattr(weights, "step_need_roll", 0.35)),
            g_lat=float(getattr(weights, "step_need_lat_vel", 0.30)),
            g_pr=float(getattr(weights, "step_need_pitch_rate", 1.00)),
        )

        # -- Foot positions (heading-local) -----------------------------------
        left_foot_pos, right_foot_pos = self._cal.get_foot_positions(
            data, normalize=False, frame=CoordinateFrame.HEADING_LOCAL
        )
        left_force, right_force = self._cal.get_aggregated_foot_contacts(data)
        contact_threshold = env_cfg.contact_threshold_force
        loaded_left = (left_force > contact_threshold).astype(jp.int32)
        loaded_right = (right_force > contact_threshold).astype(jp.int32)

        # -- FSM state update ------------------------------------------------
        new_fsm = sc.update_fsm(
            phase=wr.fsm_phase,
            swing_foot=wr.fsm_swing_foot,
            phase_ticks=wr.fsm_phase_ticks,
            frozen_tx=wr.fsm_frozen_tx,
            frozen_ty=wr.fsm_frozen_ty,
            swing_sx=wr.fsm_swing_sx,
            swing_sy=wr.fsm_swing_sy,
            touch_hold=wr.fsm_touch_hold,
            trigger_hold=wr.fsm_trigger_hold,
            need_step=need_step,
            loaded_left=loaded_left,
            loaded_right=loaded_right,
            left_foot_x=left_foot_pos[0],
            left_foot_y=left_foot_pos[1],
            right_foot_x=right_foot_pos[0],
            right_foot_y=right_foot_pos[1],
            lateral_vel=lateral_vel,
            roll=roll,
            forward_vel=forward_vel,
            pitch=pitch,
            trigger_threshold=float(env_cfg.fsm_trigger_threshold),
            recover_threshold=float(env_cfg.fsm_recover_threshold),
            trigger_hold_ticks=int(env_cfg.fsm_trigger_hold_ticks),
            touch_hold_ticks=int(env_cfg.fsm_touch_hold_ticks),
            swing_timeout_ticks=int(env_cfg.fsm_swing_timeout_ticks),
            y_nominal_m=float(env_cfg.fsm_y_nominal_m),
            k_lat_vel=float(env_cfg.fsm_k_lat_vel),
            k_roll=float(env_cfg.fsm_k_roll),
            k_fwd_vel=float(env_cfg.fsm_k_fwd_vel),
            k_pitch=float(env_cfg.fsm_k_pitch),
            x_nominal_m=float(env_cfg.fsm_x_nominal_m),
            x_step_min_m=float(env_cfg.fsm_x_step_min_m),
            x_step_max_m=float(env_cfg.fsm_x_step_max_m),
            y_step_inner_m=float(env_cfg.fsm_y_step_inner_m),
            y_step_outer_m=float(env_cfg.fsm_y_step_outer_m),
            step_max_delta_m=float(env_cfg.fsm_step_max_delta_m),
        )
        new_phase, new_swing_foot = new_fsm[0], new_fsm[1]
        new_phase_ticks = new_fsm[2]

        # -- Actuator index mapping (looked up at Python trace time) ----------
        a = self._actuator_name_to_index
        idx_lhp  = a.get("left_hip_pitch",  -1)
        idx_rhp  = a.get("right_hip_pitch", -1)
        idx_lhr  = a.get("left_hip_roll",   -1)
        idx_rhr  = a.get("right_hip_roll",  -1)
        idx_lk   = a.get("left_knee",       -1)
        idx_rk   = a.get("right_knee",      -1)
        idx_la   = a.get("left_ankle",      -1)
        idx_ra   = a.get("right_ankle",     -1)
        idx_w    = a.get("waist",           -1)

        # -- Base action from step controller --------------------------------
        ctrl_base = sc.compute_ctrl_base(
            phase=new_phase,
            swing_foot=new_swing_foot,
            phase_ticks=new_phase_ticks,
            frozen_tx=new_fsm[3],
            frozen_ty=new_fsm[4],
            swing_sx=new_fsm[5],
            swing_sy=new_fsm[6],
            pitch=pitch,
            roll=roll,
            pitch_rate=pitch_rate,
            roll_rate=roll_rate,
            need_step=need_step,
            left_foot_x=left_foot_pos[0],
            left_foot_y=left_foot_pos[1],
            right_foot_x=right_foot_pos[0],
            right_foot_y=right_foot_pos[1],
            default_action=self._default_pose_action,
            idx_left_hip_pitch=idx_lhp,
            idx_right_hip_pitch=idx_rhp,
            idx_left_hip_roll=idx_lhr,
            idx_right_hip_roll=idx_rhr,
            idx_left_knee=idx_lk,
            idx_right_knee=idx_rk,
            idx_left_ankle=idx_la,
            idx_right_ankle=idx_ra,
            idx_waist=idx_w,
            base_pitch_kp=float(env_cfg.base_ctrl_pitch_kp),
            base_pitch_kd=float(env_cfg.base_ctrl_pitch_kd),
            base_roll_kp=float(env_cfg.base_ctrl_roll_kp),
            base_roll_kd=float(env_cfg.base_ctrl_roll_kd),
            hip_pitch_gain=float(env_cfg.base_ctrl_hip_pitch_gain),
            ankle_pitch_gain=float(env_cfg.base_ctrl_ankle_pitch_gain),
            hip_roll_gain=float(env_cfg.base_ctrl_hip_roll_gain),
            base_action_clip=float(env_cfg.base_ctrl_action_clip),
            swing_x_to_hip_pitch=float(env_cfg.fsm_swing_x_to_hip_pitch),
            swing_y_to_hip_roll=float(env_cfg.fsm_swing_y_to_hip_roll),
            swing_z_to_knee=float(env_cfg.fsm_swing_z_to_knee),
            swing_z_to_ankle=float(env_cfg.fsm_swing_z_to_ankle),
            swing_height_m=float(env_cfg.fsm_swing_height_m),
            swing_duration_ticks=int(env_cfg.fsm_swing_duration_ticks),
            swing_height_need_step_mult=float(env_cfg.fsm_swing_height_need_step_mult),
            arm_enabled=bool(env_cfg.fsm_arm_enabled),
            arm_need_step_threshold=float(env_cfg.fsm_arm_need_step_threshold),
            arm_k_roll=float(env_cfg.fsm_arm_k_roll),
            arm_k_roll_rate=float(env_cfg.fsm_arm_k_roll_rate),
            arm_k_pitch_rate=float(env_cfg.fsm_arm_k_pitch_rate),
            arm_max_delta_rad=float(env_cfg.fsm_arm_max_delta_rad),
        )

        # -- Gated residual authority ----------------------------------------
        # Scale residual authority by FSM phase.
        resid_swing   = jp.asarray(env_cfg.fsm_resid_scale_swing,   jp.float32)
        resid_stance  = jp.asarray(env_cfg.fsm_resid_scale_stance,  jp.float32)
        resid_recover = jp.asarray(env_cfg.fsm_resid_scale_recover, jp.float32)

        in_swing    = (new_phase == sc.SWING).astype(jp.float32)
        in_recover  = (new_phase == sc.TOUCHDOWN_RECOVER).astype(jp.float32)
        in_stance   = (1.0 - in_swing - in_recover)
        resid_scale = (
            in_stance * resid_stance
            + in_swing * resid_swing
            + in_recover * resid_recover
        )

        mixed = ctrl_base + resid_scale * policy_action
        mixed = jp.clip(mixed, -1.0, 1.0).astype(jp.float32)

        return mixed, new_fsm

    # =========================================================================
    # Observation, Reward, Done
    # =========================================================================

    def _get_obs(
        self,
        data: mjx.Data,
        action: jax.Array,
        velocity_cmd: jax.Array,
        step_count: Optional[jax.Array] = None,
        prev_root_pos: Optional[jax.Array] = None,
        prev_root_quat: Optional[jax.Array] = None,
        loc_ref_phase_sin_cos: Optional[jax.Array] = None,
        loc_ref_stance_foot: Optional[jax.Array] = None,
        loc_ref_next_foothold: Optional[jax.Array] = None,
        loc_ref_swing_pos: Optional[jax.Array] = None,
        loc_ref_swing_vel: Optional[jax.Array] = None,
        loc_ref_pelvis_targets: Optional[jax.Array] = None,
        loc_ref_history: Optional[jax.Array] = None,
        signals=None,
    ) -> jax.Array:
        """Build observation vector.

        v0.12.x: Policy observations now come from contract Signals + frame helpers.
        Angular velocity is derived from the IMU gyro sensor (hardware-consistent).
        Linear velocity is privileged-only (sim reward/metrics), not part of the actor observation.

        Observation includes:
        - Base orientation (gravity vector in local frame): 3
        - Base angular velocity (heading-local, from FD or qvel): 3
        - Joint positions (actuated): 8
        - Joint velocities (actuated): 8
        - Foot switches (toe/heel L/R): 4
        - Previous action: 8
        - Velocity command: 1
        - Padding: 1

        Total: 36

        Args:
            data: MJX simulation data
            action: Previous action
            velocity_cmd: Current velocity command
            prev_root_pos: Previous root position for FD velocity (optional)
            prev_root_quat: Previous root quaternion for FD velocity (optional)

        Returns:
            Observation vector (36,)
        """
        if signals is None:
            signals = self._signals_adapter.read(data)
        policy_state = PolicyState(prev_action=action)
        capture_point_error = None
        gait_clock = None
        if self._policy_spec.observation.layout_id == "wr_obs_v2":
            capture_point_error = self._get_capture_point_error(data)
        elif self._policy_spec.observation.layout_id == "wr_obs_v3":
            gait_clock = self._get_gait_clock(step_count)
        return build_observation(
            spec=self._policy_spec,
            state=policy_state,
            signals=signals,
            velocity_cmd=velocity_cmd,
            capture_point_error=capture_point_error,
            gait_clock=gait_clock,
            loc_ref_phase_sin_cos=loc_ref_phase_sin_cos,
            loc_ref_stance_foot=loc_ref_stance_foot,
            loc_ref_next_foothold=loc_ref_next_foothold,
            loc_ref_swing_pos=loc_ref_swing_pos,
            loc_ref_swing_vel=loc_ref_swing_vel,
            loc_ref_pelvis_targets=loc_ref_pelvis_targets,
            loc_ref_history=loc_ref_history,
        )

    def _get_capture_point_error(self, data: mjx.Data) -> jax.Array:
        """Approximate heading-local capture-point error from CoM height and velocity."""
        root_pose = self._cal.get_root_pose(data)
        root_vel = self._cal.get_root_velocity(
            data, frame=CoordinateFrame.HEADING_LOCAL
        )
        vx, vy, _ = root_vel.linear_xyz
        com_height = jp.maximum(root_pose.height, jp.asarray(1e-3, dtype=jp.float32))
        omega0 = jp.sqrt(jp.asarray(9.81, dtype=jp.float32) / com_height)
        return jp.asarray(
            [
                -vx / omega0,
                -vy / omega0,
            ],
            dtype=jp.float32,
        )

    def _get_gait_clock(self, step_count: Optional[jax.Array]) -> jax.Array:
        """Return left/right sinusoidal gait clock features for `wr_obs_v3`."""
        if step_count is None:
            step_count = jp.zeros((), dtype=jp.float32)
        step = jp.asarray(step_count, dtype=jp.float32)
        stride_steps = jp.maximum(
            jp.asarray(self._config.env.clock_stride_period_steps, dtype=jp.float32),
            1.0,
        )
        phase = 2.0 * jp.pi * (step / stride_steps)
        phase_right = phase + jp.pi
        return jp.asarray(
            [
                jp.sin(phase),
                jp.cos(phase),
                jp.sin(phase_right),
                jp.cos(phase_right),
            ],
            dtype=jp.float32,
        )

    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        raw_action: jax.Array,
        prev_action: jax.Array,
        velocity_cmd: jax.Array,
        prev_left_foot_pos: Optional[jax.Array] = None,
        prev_right_foot_pos: Optional[jax.Array] = None,
        prev_left_loaded: Optional[jax.Array] = None,
        prev_right_loaded: Optional[jax.Array] = None,
        fsm_phase: Optional[jax.Array] = None,
        fsm_swing_foot: Optional[jax.Array] = None,
        current_step_count: Optional[jax.Array] = None,
        cycle_start_forward_x: Optional[jax.Array] = None,
        last_touchdown_root_pos: Optional[jax.Array] = None,
        last_touchdown_foot: Optional[jax.Array] = None,
        push_active: Optional[jax.Array] = None,
        recovery_active: Optional[jax.Array] = None,
        recovery_first_touchdown_recorded: Optional[jax.Array] = None,
        recovery_pitch_rate_at_touchdown: Optional[jax.Array] = None,
        recovery_capture_error_at_touchdown: Optional[jax.Array] = None,
        nominal_q_ref: Optional[jax.Array] = None,
        residual_delta_q: Optional[jax.Array] = None,
        loc_ref_pelvis_height: Optional[jax.Array] = None,
        loc_ref_pelvis_roll: Optional[jax.Array] = None,
        loc_ref_pelvis_pitch: Optional[jax.Array] = None,
        loc_ref_swing_pos: Optional[jax.Array] = None,
        loc_ref_swing_vel: Optional[jax.Array] = None,
        loc_ref_next_foothold: Optional[jax.Array] = None,
        loc_ref_stance_foot: Optional[jax.Array] = None,
    ) -> tuple[jax.Array, Dict[str, jax.Array]]:
        """Compute reward following gold standard for bipedal locomotion.

        v0.10.1: Implements industry best practices:
        - Primary: velocity tracking, alive bonus, orientation, slip penalty
        - Secondary: torque penalty, saturation, action smoothness, clearance

        Returns:
            (total_reward, reward_components_dict)
        """

        # =====================================================================
        # PRIMARY REWARDS (Tier 1)
        # =====================================================================

        # Cache root pose (used for height + orientation)
        root_pose = self._cal.get_root_pose(data)

        # Cache root velocity (used for forward/lateral vel + angular vel penalty)
        # HEADING_LOCAL: horizontal velocity in heading direction (robust to pitch/roll)
        root_vel = self._cal.get_root_velocity(
            data, frame=CoordinateFrame.HEADING_LOCAL
        )

        # 1. Forward velocity tracking (exp shaping)
        forward_vel, lateral_vel, v_z = root_vel.linear_xyz
        vel_error = jp.abs(forward_vel - velocity_cmd)
        forward_reward = compute_zero_baseline_forward_reward(
            forward_vel=forward_vel,
            velocity_cmd=velocity_cmd,
            scale=self._config.reward_weights.forward_velocity_scale,
            velocity_cmd_min=self._config.reward_weights.velocity_cmd_min,
        )

        # 2. Lateral velocity penalty (penalize sideways drift)
        lateral_penalty = jp.square(lateral_vel)

        # 3. Alive/healthy reward (height-based)
        height = root_pose.height
        healthy = jp.where(
            (height > self._config.env.min_height)
            & (height < self._config.env.max_height),
            1.0,
            0.0,
        )

        # 4. Orientation penalty (pitch² + roll²)
        roll, pitch, _ = root_pose.euler_angles()
        orientation_penalty = jp.square(pitch) + jp.square(roll)
        com_horizontal_speed = jp.sqrt(jp.square(forward_vel) + jp.square(lateral_vel))

        # 5. Angular velocity penalty (reduce body rotation)
        # Note: Using cached root_vel (LOCAL frame) - reuse from above
        angvel_x, angvel_y, angvel_z = root_vel.angular_xyz
        # Only penalize yaw rate (z-axis rotation), not pitch/roll which are handled by orientation penalty
        angvel_penalty = jp.square(angvel_z)
        pitch_rate = angvel_y
        pitch_rate_penalty = jp.square(pitch_rate)

        # =====================================================================
        # EFFORT REWARDS (Tier 1 - critical for sim2real)
        # =====================================================================

        # 6. Torque penalty - penalize normalized torque usage
        normalized_torques = self._cal.get_actuator_torques(
            data.qfrc_actuator, normalize=True
        )
        torque_penalty = jp.sum(jp.square(normalized_torques))

        # 7. Saturation penalty - penalize actuators near limits
        torque_abs = jp.abs(normalized_torques)
        saturation = torque_abs > 0.95
        saturation_penalty = jp.sum(saturation.astype(jp.float32))
        torque_sat_frac = jp.mean(saturation.astype(jp.float32))
        torque_abs_max = jp.max(torque_abs)

        # =====================================================================
        # SMOOTHNESS REWARDS (Tier 2)
        # =====================================================================

        # 8. Action rate penalty (smoothness)
        action_diff = action - prev_action
        action_rate_penalty = jp.sum(jp.square(action_diff))
        action_abs = jp.abs(action)
        action_abs_max = jp.max(action_abs)
        action_sat_frac = jp.mean((action_abs > 0.95).astype(jp.float32))
        raw_action_abs = jp.abs(raw_action)
        raw_action_abs_max = jp.max(raw_action_abs)
        raw_action_sat_frac = jp.mean((raw_action_abs > 0.95).astype(jp.float32))

        # 9. Joint velocity penalty (efficiency) - use normalized for consistent scaling
        joint_vel = self._cal.get_joint_velocities(data.qvel, normalize=True)
        joint_vel_penalty = jp.sum(jp.square(joint_vel))

        # =====================================================================
        # FOOT STABILITY REWARDS (Tier 2)
        # =====================================================================

        # Get foot contact forces for slip/clearance gating
        left_force, right_force = self._cal.get_aggregated_foot_contacts(data)
        # Robot weight for contact threshold (approximate: mass * gravity)
        # ~2kg robot * 9.81 ≈ 20N total, so ~5N per foot is "loaded"
        # v0.10.1: Read from config instead of hard-coding
        contact_threshold = self._config.env.contact_threshold_force

        # 10. Slip penalty (when foot is loaded, penalize tangential velocity)
        # Use finite-diff velocity if previous positions available
        if prev_left_foot_pos is not None and prev_right_foot_pos is not None:
            left_vel, right_vel = self._cal.get_foot_velocities(
                data,
                prev_left_foot_pos,
                prev_right_foot_pos,
                self.dt,
                normalize=False,
                frame=CoordinateFrame.HEADING_LOCAL,
            )
            # Tangential (xy) velocity magnitude
            left_slip = jp.sqrt(jp.square(left_vel[0]) + jp.square(left_vel[1]))
            right_slip = jp.sqrt(jp.square(right_vel[0]) + jp.square(right_vel[1]))

            # Gate by contact force (only penalize when loaded)
            left_loaded = left_force > contact_threshold
            right_loaded = right_force > contact_threshold
            slip_penalty = jp.where(left_loaded, jp.square(left_slip), 0.0) + jp.where(
                right_loaded, jp.square(right_slip), 0.0
            )
        else:
            slip_penalty = jp.zeros(())
            left_slip = jp.zeros(())
            right_slip = jp.zeros(())
            left_vel = jp.zeros((3,), dtype=jp.float32)
            right_vel = jp.zeros((3,), dtype=jp.float32)

        # TODO(M3 cleanup): Remove deprecated gait-style reward computations entirely
        # after the first M3 validation run confirms they are not needed for debugging.
        # 11. Swing clearance reward (phase-gated by clock; active in total reward)
        # Reward foot height above ground during swing phase
        min_clearance = 0.02  # 2cm minimum clearance during swing
        left_clearance_raw, right_clearance_raw = self._cal.get_foot_clearances(
            data, normalize=False
        )
        left_clearance = left_clearance_raw
        right_clearance = right_clearance_raw
        left_clearance = jp.clip(
            left_clearance - min_clearance, 0.0, 0.05
        )  # cap at 5cm
        right_clearance = jp.clip(right_clearance - min_clearance, 0.0, 0.05)

        # Gate by clock phase and unloaded status so clearance only pays during
        # the expected swing window for each foot.
        if current_step_count is None:
            step_f = jp.zeros((), dtype=jp.float32)
        else:
            step_f = jp.asarray(current_step_count, dtype=jp.float32)
        stride_steps_f = jp.maximum(
            jp.asarray(self._config.env.clock_stride_period_steps, dtype=jp.float32),
            1.0,
        )
        phase = 2.0 * jp.pi * (step_f / stride_steps_f)

        phase_gate_width = jp.asarray(
            getattr(self._config.env, "clock_phase_gate_width", 0.20), dtype=jp.float32
        )
        clearance_reward = compute_phase_gated_clearance_reward(
            left_clearance=left_clearance,
            right_clearance=right_clearance,
            left_force=left_force,
            right_force=right_force,
            contact_threshold=contact_threshold,
            phase=phase,
            phase_gate_width=phase_gate_width,
        )
        left_swing = left_force < contact_threshold
        right_swing = right_force < contact_threshold

        # 12. Gait periodicity (deprecated for M3 total reward; kept for metrics)
        left_loaded = left_force > contact_threshold
        right_loaded = right_force > contact_threshold
        gait_periodicity = jp.where(left_loaded ^ right_loaded, 1.0, 0.0)
        gait_periodicity = jp.asarray(gait_periodicity).reshape(())

        # Get weights from frozen config (type-safe access)
        weights = self._config.reward_weights

        # 13. Hip/knee swing (deprecated for M3 total reward; kept for metrics)
        # Use CAL semantic accessors - joint names defined in robot_config (single source)
        left_hip_pitch, right_hip_pitch = self._cal.get_hip_pitch_positions(
            data.qpos, normalize=True
        )
        left_knee_pitch, right_knee_pitch = self._cal.get_knee_pitch_positions(
            data.qpos, normalize=True
        )

        hip_min = jp.maximum(weights.hip_swing_min, 1e-6)
        knee_min = jp.maximum(weights.knee_swing_min, 1e-6)

        def _swing_reward(angle: jax.Array, min_amp: jax.Array) -> jax.Array:
            amp = jp.abs(angle)
            return jp.clip((amp - min_amp) / min_amp, 0.0, 1.0)

        hip_swing = 0.5 * (
            _swing_reward(left_hip_pitch, hip_min) * left_swing
            + _swing_reward(right_hip_pitch, hip_min) * right_swing
        )
        knee_swing = 0.5 * (
            _swing_reward(left_knee_pitch, knee_min) * left_swing
            + _swing_reward(right_knee_pitch, knee_min) * right_swing
        )
        hip_swing = jp.asarray(hip_swing).reshape(())
        knee_swing = jp.asarray(knee_swing).reshape(())

        # 14. Flight phase penalty (deprecated for M3 total reward; kept for metrics)
        flight_phase = jp.where((~left_loaded) & (~right_loaded), 1.0, 0.0)
        flight_phase = jp.asarray(flight_phase).reshape(())

        # 15. Height target shaping
        # Default is one-sided shaping (penalize only being below target height).
        # If enabled, two-sided shaping encourages staying near target_height.
        height_target = self._config.env.target_height
        height_sigma = jp.maximum(weights.height_target_sigma, 1e-6)
        if bool(getattr(self._config.env, "height_target_two_sided", False)):
            height_error = (height - height_target) / height_sigma
        else:
            height_error = jp.maximum(height_target - height, 0.0) / height_sigma
        height_target_reward = jp.exp(-jp.square(height_error))

        # 16. Pre-collapse shaping: height margin and downward velocity near min height
        collapse_height_sigma = jp.maximum(self._config.env.collapse_height_sigma, 1e-6)
        collapse_vz_gate_band = jp.maximum(self._config.env.collapse_vz_gate_band, 1e-6)
        h_margin = height - (
            self._config.env.min_height + self._config.env.collapse_height_buffer
        )
        collapse_height_pen = jp.square(jp.maximum(-h_margin / collapse_height_sigma, 0.0))
        collapse_gate = jax.nn.sigmoid(
            (self._config.env.min_height + collapse_vz_gate_band - height)
            / collapse_vz_gate_band
        )
        collapse_vz_pen = collapse_gate * jp.square(jp.maximum(-v_z, 0.0))

        # 17. Stance width penalty (deprecated for M3 total reward; kept for metrics)
        left_foot_pos, right_foot_pos = self._cal.get_foot_positions(
            data, normalize=False, frame=CoordinateFrame.HEADING_LOCAL
        )
        stance_width = jp.abs(left_foot_pos[1] - right_foot_pos[1])
        stance_sigma = jp.maximum(weights.stance_width_sigma, 1e-6)
        stance_excess = jp.maximum(stance_width - weights.stance_width_target, 0.0)
        stance_width_penalty = jp.square(stance_excess / stance_sigma)

        # 18. Posture return shaping (encourage returning to default joint pose when upright)
        posture_sigma = jp.maximum(getattr(weights, "posture_sigma", 0.35), 1e-6)
        joint_pos_rad = self._cal.get_joint_positions(data.qpos, normalize=False)
        posture_mse = jp.mean(jp.square(joint_pos_rad - self._default_joint_qpos))
        posture_reward = jp.exp(-posture_mse / (posture_sigma * posture_sigma))
        posture_gate_pitch = getattr(weights, "posture_gate_pitch", 0.35)
        posture_gate_roll = getattr(weights, "posture_gate_roll", 0.35)
        posture_gate = (
            healthy
            * (jp.abs(pitch) < posture_gate_pitch).astype(jp.float32)
            * (jp.abs(roll) < posture_gate_roll).astype(jp.float32)
        )
        posture_reward = posture_reward * posture_gate

        (
            disturbed_gate,
            effective_orientation_weight,
            effective_height_target_weight,
            effective_posture_weight,
        ) = compute_disturbed_window_reward_weights(
            base_orientation_weight=weights.orientation,
            base_height_target_weight=weights.height_target,
            base_posture_weight=getattr(weights, "posture", 0.0),
            push_active=push_active,
            recovery_active=recovery_active,
            disturbed_orientation_weight=getattr(
                weights, "disturbed_orientation", weights.orientation
            ),
            disturbed_height_target_scale=getattr(weights, "disturbed_height_target_scale", 1.0),
            disturbed_posture_scale=getattr(weights, "disturbed_posture_scale", 1.0),
        )
        height_floor_penalty = disturbed_gate * compute_height_floor_penalty(
            height=height,
            threshold=jp.asarray(
                getattr(weights, "height_floor_threshold", 0.20), dtype=jp.float32
            ),
            sigma=jp.asarray(getattr(weights, "height_floor_sigma", 0.03), dtype=jp.float32),
        )
        com_velocity_damping_reward = disturbed_gate * compute_com_velocity_damping_reward(
            horizontal_speed=com_horizontal_speed,
            scale=jp.asarray(
                getattr(weights, "com_velocity_damping_scale", 1.0), dtype=jp.float32
            ),
        )

        # 19. Step event + foot placement shaping (gated by "need to step")
        # Detect touchdown events from contact transitions (prev_loaded -> loaded).
        if prev_left_loaded is None or prev_right_loaded is None:
            prev_left_loaded_b = left_loaded
            prev_right_loaded_b = right_loaded
        else:
            prev_left_loaded_b = prev_left_loaded > 0.5
            prev_right_loaded_b = prev_right_loaded > 0.5

        liftoff_left = jp.logical_and(prev_left_loaded_b, jp.logical_not(left_loaded))
        liftoff_right = jp.logical_and(prev_right_loaded_b, jp.logical_not(right_loaded))
        touchdown_left = jp.logical_and(jp.logical_not(prev_left_loaded_b), left_loaded)
        touchdown_right = jp.logical_and(jp.logical_not(prev_right_loaded_b), right_loaded)
        if (
            bool(getattr(self._config.env, "fsm_enabled", False))
            and fsm_phase is not None
            and fsm_swing_foot is not None
        ):
            in_swing = fsm_phase == sc.SWING
            liftoff_left = jp.logical_and(liftoff_left, in_swing & (fsm_swing_foot == 0))
            liftoff_right = jp.logical_and(liftoff_right, in_swing & (fsm_swing_foot == 1))
            touchdown_left = jp.logical_and(touchdown_left, in_swing & (fsm_swing_foot == 0))
            touchdown_right = jp.logical_and(touchdown_right, in_swing & (fsm_swing_foot == 1))
        step_event = jp.where(touchdown_left | touchdown_right, 1.0, 0.0)
        step_event = jp.asarray(step_event).reshape(())
        liftoff_event = jp.where(liftoff_left | liftoff_right, 1.0, 0.0)
        liftoff_event = jp.asarray(liftoff_event).reshape(())

        need_step = self._compute_need_step(
            pitch=pitch,
            roll=roll,
            lateral_vel=lateral_vel,
            pitch_rate=pitch_rate,
            healthy=healthy,
        )
        command_step_gate = compute_command_active_gate(
            velocity_cmd, getattr(weights, "velocity_cmd_min", 0.2)
        )
        recovery_step_gate = jp.maximum(
            jp.asarray(push_active if push_active is not None else 0.0, dtype=jp.float32),
            jp.asarray(recovery_active if recovery_active is not None else 0.0, dtype=jp.float32),
        )
        step_reward_gate = jp.maximum(recovery_step_gate, command_step_gate)

        # Foot placement reward at touchdown (Raibert-style heuristic in heading-local frame).
        base_y = 0.5 * jp.asarray(weights.stance_width_target)
        y_corr = (
            jp.asarray(getattr(weights, "foot_place_k_lat_vel", 0.15)) * lateral_vel
            + jp.asarray(getattr(weights, "foot_place_k_roll", 0.10)) * roll
        )
        x_corr = (
            jp.asarray(getattr(weights, "foot_place_k_cmd_vel", 0.0)) * velocity_cmd
            + jp.asarray(getattr(weights, "foot_place_k_fwd_vel", 0.05)) * forward_vel
            + jp.asarray(getattr(weights, "foot_place_k_pitch", 0.05)) * pitch
        )
        sigma_xy = jp.maximum(getattr(weights, "foot_place_sigma", 0.12), 1e-6)
        inv_sigma2 = 1.0 / (sigma_xy * sigma_xy)

        left_x, left_y = left_foot_pos[0], left_foot_pos[1]
        right_x, right_y = right_foot_pos[0], right_foot_pos[1]
        left_y_des = base_y + y_corr
        right_y_des = -base_y + y_corr
        x_des = x_corr

        err_left = (left_x - x_des) * (left_x - x_des) + (left_y - left_y_des) * (left_y - left_y_des)
        err_right = (right_x - x_des) * (right_x - x_des) + (right_y - right_y_des) * (right_y - right_y_des)
        foot_place_left = jp.exp(-err_left * inv_sigma2)
        foot_place_right = jp.exp(-err_right * inv_sigma2)
        foot_place_reward = jp.where(touchdown_left, foot_place_left, 0.0) + jp.where(
            touchdown_right, foot_place_right, 0.0
        )
        foot_place_reward = jp.asarray(foot_place_reward).reshape(())
        foot_place_reward = foot_place_reward * step_reward_gate

        # v0.19.3: reference-guided task-space rewards (forward + stop only).
        if loc_ref_pelvis_height is None:
            loc_ref_pelvis_height = jp.asarray(height, dtype=jp.float32)
        if loc_ref_pelvis_roll is None:
            loc_ref_pelvis_roll = jp.asarray(0.0, dtype=jp.float32)
        if loc_ref_pelvis_pitch is None:
            loc_ref_pelvis_pitch = jp.asarray(0.0, dtype=jp.float32)
        if loc_ref_swing_pos is None:
            loc_ref_swing_pos = jp.zeros((3,), dtype=jp.float32)
        if loc_ref_swing_vel is None:
            loc_ref_swing_vel = jp.zeros((3,), dtype=jp.float32)
        if loc_ref_next_foothold is None:
            loc_ref_next_foothold = jp.zeros((2,), dtype=jp.float32)
        if loc_ref_stance_foot is None:
            loc_ref_stance_foot = jp.asarray(0, dtype=jp.int32)
        if residual_delta_q is None:
            residual_delta_q = jp.zeros((self.action_size,), dtype=jp.float32)
        if nominal_q_ref is None:
            nominal_q_ref = self._default_joint_qpos

        pelvis_roll_err = roll - jp.asarray(loc_ref_pelvis_roll, dtype=jp.float32)
        pelvis_pitch_err = pitch - jp.asarray(loc_ref_pelvis_pitch, dtype=jp.float32)
        pelvis_orientation_err = jp.sqrt(
            jp.square(pelvis_roll_err) + jp.square(pelvis_pitch_err)
        )
        pelvis_orientation_sigma = jp.maximum(
            jp.asarray(getattr(weights, "m3_pelvis_orientation_sigma", 0.20), dtype=jp.float32),
            1e-6,
        )
        m3_pelvis_orientation_reward = jp.exp(
            -jp.square(pelvis_orientation_err / pelvis_orientation_sigma)
        )

        pelvis_height_err = height - jp.asarray(loc_ref_pelvis_height, dtype=jp.float32)
        pelvis_height_sigma = jp.maximum(
            jp.asarray(getattr(weights, "m3_pelvis_height_sigma", 0.04), dtype=jp.float32),
            1e-6,
        )
        m3_pelvis_height_reward = jp.exp(-jp.square(pelvis_height_err / pelvis_height_sigma))

        swing_pos_actual, swing_vel_actual = _swing_state_in_stance_frame(
            left_foot_pos_h=left_foot_pos,
            right_foot_pos_h=right_foot_pos,
            left_foot_vel_h=left_vel,
            right_foot_vel_h=right_vel,
            stance_foot_id=loc_ref_stance_foot,
        )
        swing_pos_err = jp.linalg.norm(
            swing_pos_actual - jp.asarray(loc_ref_swing_pos, dtype=jp.float32)
        )
        swing_vel_err = jp.linalg.norm(
            swing_vel_actual - jp.asarray(loc_ref_swing_vel, dtype=jp.float32)
        )
        swing_pos_sigma = jp.maximum(
            jp.asarray(getattr(weights, "m3_swing_pos_sigma", 0.08), dtype=jp.float32),
            1e-6,
        )
        swing_vel_sigma = jp.maximum(
            jp.asarray(getattr(weights, "m3_swing_vel_sigma", 0.80), dtype=jp.float32),
            1e-6,
        )
        m3_swing_tracking_reward = jp.exp(-jp.square(swing_pos_err / swing_pos_sigma)) * jp.exp(
            -jp.square(swing_vel_err / swing_vel_sigma)
        )

        touchdown_any = touchdown_left | touchdown_right
        touchdown_pos = jp.where(
            touchdown_left,
            left_foot_pos[:2],
            jp.where(touchdown_right, right_foot_pos[:2], jp.zeros((2,), dtype=jp.float32)),
        )
        stance_xy = jp.where(
            jp.asarray(loc_ref_stance_foot, dtype=jp.int32) == REF_LEFT_STANCE,
            left_foot_pos[:2],
            right_foot_pos[:2],
        )
        touchdown_pos_stance_frame = touchdown_pos - stance_xy
        foothold_err = jp.linalg.norm(
            touchdown_pos_stance_frame - jp.asarray(loc_ref_next_foothold, dtype=jp.float32)
        )
        foothold_sigma = jp.maximum(
            jp.asarray(getattr(weights, "m3_foothold_sigma", 0.10), dtype=jp.float32),
            1e-6,
        )
        m3_foothold_consistency_reward = jp.where(
            touchdown_any,
            jp.exp(-jp.square(foothold_err / foothold_sigma)),
            jp.zeros((), dtype=jp.float32),
        )

        m3_residual_mag_penalty = jp.mean(jp.square(jp.asarray(residual_delta_q, dtype=jp.float32)))

        impact_force = jp.maximum(left_force, right_force)
        impact_threshold = jp.asarray(
            getattr(weights, "m3_impact_force_threshold", 40.0), dtype=jp.float32
        )
        impact_sigma = jp.maximum(
            jp.asarray(getattr(weights, "m3_impact_force_sigma", 20.0), dtype=jp.float32),
            1e-6,
        )
        m3_excessive_impact_penalty = jp.square(
            jp.maximum(impact_force - impact_threshold, 0.0) / impact_sigma
        )

        step_length_reward = compute_step_length_touchdown_reward(
            left_x=left_x,
            right_x=right_x,
            touchdown_left=touchdown_left,
            touchdown_right=touchdown_right,
            velocity_cmd=velocity_cmd,
            step_reward_gate=step_reward_gate,
            target_base=jp.asarray(getattr(weights, "step_length_target_base", 0.03), dtype=jp.float32),
            target_scale=jp.asarray(getattr(weights, "step_length_target_scale", 0.25), dtype=jp.float32),
            sigma=jp.asarray(getattr(weights, "step_length_sigma", 0.04), dtype=jp.float32),
        )

        # Structured touchdown-to-touchdown progress reward (contact-driven).
        if last_touchdown_root_pos is None:
            last_touchdown_root_pos = root_pose.position
        if last_touchdown_foot is None:
            last_touchdown_foot = jp.asarray(-1, dtype=jp.int32)
        heading_sin, heading_cos = root_pose.heading_sincos
        (
            step_progress_reward,
            step_progress_delta,
            step_progress_event,
            _,
            _,
        ) = compute_step_progress_touchdown_reward(
            current_root_pos=root_pose.position,
            last_touchdown_root_pos=last_touchdown_root_pos,
            last_touchdown_foot=last_touchdown_foot,
            heading_sin=heading_sin,
            heading_cos=heading_cos,
            touchdown_left=touchdown_left,
            touchdown_right=touchdown_right,
            velocity_cmd=velocity_cmd,
            velocity_cmd_min=jp.asarray(getattr(weights, "velocity_cmd_min", 0.08), dtype=jp.float32),
            step_reward_gate=step_reward_gate,
            dt=self.dt,
            stride_steps=jp.asarray(self._config.env.clock_stride_period_steps, dtype=jp.int32),
            target_scale=jp.asarray(getattr(weights, "step_progress_target_scale", 1.0), dtype=jp.float32),
            sigma=jp.asarray(getattr(weights, "step_progress_sigma", 0.05), dtype=jp.float32),
        )
        capture_error_norm = jp.linalg.norm(self._get_capture_point_error(data))
        post_touchdown_arrest_gate = (
            jp.asarray(recovery_active if recovery_active is not None else 0.0, dtype=jp.float32)
            * jp.asarray(
                recovery_first_touchdown_recorded
                if recovery_first_touchdown_recorded is not None
                else 0.0,
                dtype=jp.float32,
            )
        )
        touchdown_pitch_rate_abs = jp.abs(
            jp.asarray(
                recovery_pitch_rate_at_touchdown
                if recovery_pitch_rate_at_touchdown is not None
                else 0.0,
                dtype=jp.float32,
            )
        )
        touchdown_capture_error = jp.asarray(
            recovery_capture_error_at_touchdown
            if recovery_capture_error_at_touchdown is not None
            else 0.0,
            dtype=jp.float32,
        )
        arrest_pitch_rate_scale = jp.maximum(
            jp.asarray(getattr(weights, "arrest_pitch_rate_scale", 3.0), dtype=jp.float32),
            1e-6,
        )
        arrest_capture_error_scale = jp.maximum(
            jp.asarray(getattr(weights, "arrest_capture_error_scale", 0.15), dtype=jp.float32),
            1e-6,
        )
        arrest_pitch_rate_reward = post_touchdown_arrest_gate * jp.clip(
            (touchdown_pitch_rate_abs - jp.abs(pitch_rate)) / arrest_pitch_rate_scale,
            0.0,
            1.0,
        )
        arrest_capture_error_reward = post_touchdown_arrest_gate * jp.clip(
            (touchdown_capture_error - capture_error_norm) / arrest_capture_error_scale,
            0.0,
            1.0,
        )
        post_touchdown_survival_reward = post_touchdown_arrest_gate

        # Per-step-cycle progress reward (retained as optional auxiliary).
        if current_step_count is None:
            curr_step_i = jp.zeros((), dtype=jp.int32)
        else:
            curr_step_i = jp.asarray(current_step_count, dtype=jp.int32)
        if cycle_start_forward_x is None:
            cycle_start_forward_x = jp.zeros(())
        cycle_progress_reward, cycle_forward_delta, cycle_complete, _ = compute_cycle_progress_terms(
            cycle_progress_accum=cycle_start_forward_x,
            forward_vel=forward_vel,
            dt=self.dt,
            current_step_count=curr_step_i,
            stride_steps=jp.asarray(self._config.env.clock_stride_period_steps, dtype=jp.int32),
            velocity_cmd=velocity_cmd,
            target_scale=jp.asarray(getattr(weights, "cycle_progress_target_scale", 1.0), dtype=jp.float32),
            sigma=jp.asarray(getattr(weights, "cycle_progress_sigma", 0.08), dtype=jp.float32),
        )
        dense_progress_reward = compute_dense_progress_reward(
            forward_vel=forward_vel,
            velocity_cmd=velocity_cmd,
            velocity_cmd_min=jp.asarray(getattr(weights, "velocity_cmd_min", 0.2), dtype=jp.float32),
            left_force=left_force,
            right_force=right_force,
            contact_threshold=contact_threshold,
            pitch=pitch,
            pitch_rate=pitch_rate,
            upright_pitch_limit=jp.asarray(
                getattr(weights, "dense_progress_upright_pitch", 0.25), dtype=jp.float32
            ),
            upright_pitch_rate_limit=jp.asarray(
                getattr(weights, "dense_progress_upright_pitch_rate", 0.9), dtype=jp.float32
            ),
            upright_gate_sharpness=jp.asarray(
                getattr(weights, "dense_progress_upright_sharpness", 10.0), dtype=jp.float32
            ),
        )
        forward_upright_gate = compute_smooth_upright_gate(
            pitch=pitch,
            pitch_rate=pitch_rate,
            upright_pitch_limit=jp.asarray(
                getattr(weights, "dense_progress_upright_pitch", 0.25), dtype=jp.float32
            ),
            upright_pitch_rate_limit=jp.asarray(
                getattr(weights, "dense_progress_upright_pitch_rate", 0.9), dtype=jp.float32
            ),
            upright_gate_sharpness=jp.asarray(
                getattr(weights, "dense_progress_upright_sharpness", 10.0), dtype=jp.float32
            ),
        )
        forward_upright_gate_strength = jp.asarray(
            getattr(weights, "forward_upright_gate_strength", 1.0), dtype=jp.float32
        )
        cmd_gate_fwd = compute_command_active_gate(
            velocity_cmd, getattr(weights, "velocity_cmd_min", 0.2)
        )
        forward_reward = forward_reward * (
            (1.0 - cmd_gate_fwd)
            + cmd_gate_fwd
            * (
                (1.0 - forward_upright_gate_strength)
                + forward_upright_gate_strength * forward_upright_gate
            )
        )
        propulsion_gate = compute_propulsion_quality_gate(
            step_length_reward=step_length_reward,
            step_progress_reward=step_progress_reward,
            step_length_weight=jp.asarray(
                getattr(weights, "propulsion_gate_step_length_weight", 0.5), dtype=jp.float32
            ),
            step_progress_weight=jp.asarray(
                getattr(weights, "propulsion_gate_step_progress_weight", 0.5), dtype=jp.float32
            ),
        )
        forward_reward = forward_reward * (
            (1.0 - getattr(weights, "velocity_step_gate", 0.0))
            + getattr(weights, "velocity_step_gate", 0.0) * propulsion_gate
        )

        # =====================================================================
        # COMBINE REWARDS
        # =====================================================================

        # v0.10.4: Smooth, command-gated standing penalty
        # Only apply when |velocity_cmd| >= velocity_cmd_min (don't penalize if asked to stop)
        # Only apply when healthy (prevents noise during falling/recovery)
        # Penalty is smooth: relu(threshold - |vel|) / threshold, ranges from 0 to 1
        standing_threshold = weights.velocity_standing_threshold
        standing_penalty_weight = weights.velocity_standing_penalty
        velocity_cmd_min = weights.velocity_cmd_min

        # Smooth penalty: how far below threshold (0 if at/above threshold)
        cmd_sign = jp.where(velocity_cmd >= 0.0, 1.0, -1.0)
        signed_forward_vel = cmd_sign * forward_vel
        velocity_deficit = jp.maximum(standing_threshold - signed_forward_vel, 0.0)
        standing_penalty_raw = velocity_deficit / (standing_threshold + 1e-6)

        # Gate by command magnitude: only penalize if asked to move (supports future backward cmds)
        cmd_gate = compute_command_active_gate(velocity_cmd, velocity_cmd_min)

        # Gate by healthy: don't penalize during falling/recovery (prevents twitch behaviors)
        healthy_gate = healthy

        standing_penalty = standing_penalty_raw * cmd_gate * healthy_gate

        # v0.17.3a: alive bonus — constant 1.0 every step, dominant survival signal
        alive_reward = jp.ones(())

        total = (
            # Primary (Tier 1)
            getattr(weights, "alive", 0.0) * alive_reward
            + weights.tracking_lin_vel * forward_reward
            + weights.lateral_velocity * lateral_penalty
            + weights.base_height * healthy
            + effective_orientation_weight * orientation_penalty
            + weights.angular_velocity * angvel_penalty
            + getattr(weights, "pitch_rate", 0.0) * pitch_rate_penalty
            + weights.collapse_height * collapse_height_pen
            + weights.collapse_vz * collapse_vz_pen
            # v0.10.4: Smooth standing penalty (negative reward for standing still)
            - standing_penalty_weight * standing_penalty
            # Effort (Tier 1)
            + weights.torque * torque_penalty
            + weights.saturation * saturation_penalty
            # Smoothness (Tier 2)
            + weights.action_rate * action_rate_penalty
            + weights.joint_velocity * joint_vel_penalty
            # Foot stability / recovery (M3)
            + weights.slip * slip_penalty
            + weights.clearance * clearance_reward
            + effective_height_target_weight * height_target_reward
            + effective_posture_weight * posture_reward
            + getattr(weights, "height_floor", 0.0) * height_floor_penalty
            + getattr(weights, "com_velocity_damping", 0.0) * com_velocity_damping_reward
            + getattr(weights, "step_event", 0.0) * (step_event * step_reward_gate)
            + getattr(weights, "foot_place", 0.0) * foot_place_reward
            + getattr(weights, "step_length", 0.0) * step_length_reward
            + getattr(weights, "step_progress", 0.0) * step_progress_reward
            + getattr(weights, "m3_pelvis_orientation_tracking", 0.0)
            * m3_pelvis_orientation_reward
            + getattr(weights, "m3_pelvis_height_tracking", 0.0)
            * m3_pelvis_height_reward
            + getattr(weights, "m3_swing_foot_tracking", 0.0)
            * m3_swing_tracking_reward
            + getattr(weights, "m3_foothold_consistency", 0.0)
            * m3_foothold_consistency_reward
            + getattr(weights, "m3_residual_magnitude", 0.0)
            * m3_residual_mag_penalty
            + getattr(weights, "arrest_pitch_rate", 0.0) * arrest_pitch_rate_reward
            + getattr(weights, "arrest_capture_error", 0.0) * arrest_capture_error_reward
            + getattr(weights, "post_touchdown_survival", 0.0)
            * post_touchdown_survival_reward
            + getattr(weights, "m3_excessive_impact", 0.0)
            * m3_excessive_impact_penalty
            + getattr(weights, "dense_progress", 0.0) * dense_progress_reward
            + getattr(weights, "cycle_progress", 0.0) * cycle_progress_reward
        )

        # =====================================================================
        # REWARD COMPONENTS FOR LOGGING
        # =====================================================================

        # v0.10.3: Compute tracking metrics for walking exit criteria
        # Max normalized torque (peak torque as fraction of limit, across all actuators)
        max_torque_ratio = jp.max(jp.abs(normalized_torques))
        avg_torque = jp.mean(
            jp.abs(self._cal.get_actuator_torques(data.qfrc_actuator, normalize=False))
        )

        foot_switches = self._cal.get_foot_switches(
            data, threshold=self._config.env.foot_switch_threshold
        )

        components = {
            # Primary
            "reward/alive": alive_reward,
            "reward/forward": forward_reward,
            "reward/lateral": lateral_penalty,
            "reward/healthy": healthy,
            "reward/orientation": orientation_penalty,
            "reward/angvel": angvel_penalty,
            "reward/pitch_rate": pitch_rate_penalty,
            "reward/com_velocity_damping": com_velocity_damping_reward,
            # v0.10.4: Standing penalty
            "reward/standing": standing_penalty,
            "reward/collapse_height_pen": collapse_height_pen,
            "reward/collapse_vz_pen": collapse_vz_pen,
            "reward/height_floor": height_floor_penalty,
            # Effort
            "reward/torque": torque_penalty,
            "reward/saturation": saturation_penalty,
            # Smoothness
            "reward/action_rate": action_rate_penalty,
            "reward/joint_vel": joint_vel_penalty,
            # Foot stability
            "reward/slip": slip_penalty,
            "reward/clearance": clearance_reward,
            "reward/gait_periodicity": gait_periodicity,
            "reward/hip_swing": hip_swing,
            "reward/knee_swing": knee_swing,
            "reward/flight_phase": flight_phase,
            "reward/height_target": height_target_reward,
            "reward/stance_width_penalty": stance_width_penalty,
            "reward/posture": posture_reward,
            "reward/step_event": step_event * step_reward_gate,
            "reward/foot_place": foot_place_reward,
            "reward/step_length": step_length_reward,
            "reward/step_progress": step_progress_reward,
            "reward/m3_pelvis_orientation_tracking": m3_pelvis_orientation_reward,
            "reward/m3_pelvis_height_tracking": m3_pelvis_height_reward,
            "reward/m3_swing_foot_tracking": m3_swing_tracking_reward,
            "reward/m3_foothold_consistency": m3_foothold_consistency_reward,
            "reward/m3_residual_magnitude": m3_residual_mag_penalty,
            "reward/m3_excessive_impact": m3_excessive_impact_penalty,
            "reward/teacher_target_step_xy": jp.zeros(()),
            "reward/teacher_step_required": jp.zeros(()),
            "reward/teacher_swing_foot": jp.zeros(()),
            "reward/teacher_recovery_height": jp.zeros(()),
            "reward/teacher_com_velocity_reduction": jp.zeros(()),
            "reward/arrest_pitch_rate": arrest_pitch_rate_reward,
            "reward/arrest_capture_error": arrest_capture_error_reward,
            "reward/post_touchdown_survival": post_touchdown_survival_reward,
            "reward/dense_progress": dense_progress_reward,
            "reward/cycle_progress": cycle_progress_reward,
            "debug/posture_mse": posture_mse,
            "debug/need_step": need_step,
            "debug/touchdown_left": touchdown_left.astype(jp.float32),
            "debug/touchdown_right": touchdown_right.astype(jp.float32),
            "debug/m3_pelvis_orientation_error": pelvis_orientation_err,
            "debug/m3_pelvis_height_error": jp.abs(pelvis_height_err),
            "debug/m3_swing_pos_error": swing_pos_err,
            "debug/m3_swing_vel_error": swing_vel_err,
            "debug/m3_foothold_error": foothold_err,
            "debug/m3_impact_force": impact_force,
            "debug/cycle_complete": cycle_complete.astype(jp.float32),
            "debug/cycle_forward_delta": cycle_forward_delta,
            "debug/step_progress_delta": step_progress_delta,
            "debug/step_progress_event": step_progress_event,
            "debug/recovery_step_gate": recovery_step_gate,
            "debug/post_touchdown_arrest_gate": post_touchdown_arrest_gate,
            "debug/disturbed_reward_gate": disturbed_gate,
            "debug/effective_orientation_weight": effective_orientation_weight,
            "debug/effective_height_target_weight": effective_height_target_weight,
            "debug/effective_posture_weight": effective_posture_weight,
            # Debug metrics
            "debug/pitch": pitch,
            "debug/roll": roll,
            "debug/forward_vel": forward_vel,
            "debug/lateral_vel": lateral_vel,
            "debug/left_force": left_force,
            "debug/right_force": right_force,
            "debug/left_toe_switch": foot_switches[0],
            "debug/left_heel_switch": foot_switches[1],
            "debug/right_toe_switch": foot_switches[2],
            "debug/right_heel_switch": foot_switches[3],
            "debug/pitch_rate": pitch_rate,
            "debug/velocity_step_gate": propulsion_gate,
            "debug/propulsion_gate": propulsion_gate,
            "debug/action_abs_max": action_abs_max,
            "debug/action_sat_frac": action_sat_frac,
            "debug/raw_action_abs_max": raw_action_abs_max,
            "debug/raw_action_sat_frac": raw_action_sat_frac,
            "debug/torque_abs_max": torque_abs_max,
            "debug/torque_sat_frac": torque_sat_frac,
            "teacher/active_frac": jp.zeros(()),
            "teacher/active_count": jp.zeros(()),
            "teacher/step_required_mean": jp.zeros(()),
            "teacher/step_required_hard_frac": jp.zeros(()),
            "teacher/target_step_x_mean": jp.zeros(()),
            "teacher/target_step_y_mean": jp.zeros(()),
            "teacher/swing_left_frac": jp.zeros(()),
            "teacher/target_xy_error": jp.zeros(()),
            "teacher/target_xy_error_events": jp.zeros(()),
            "teacher/teacher_active_during_clean_frac": jp.zeros(()),
            "teacher/raw_step_required_during_clean_mean": jp.zeros(()),
            "teacher/clean_step_count": jp.zeros(()),
            "teacher/push_step_count": jp.zeros(()),
            "teacher/teacher_active_during_push_frac": jp.zeros(()),
            "teacher/reachable_frac": jp.zeros(()),
            "teacher/target_step_x_min": jp.asarray(jp.inf, dtype=jp.float32),
            "teacher/target_step_x_max": jp.asarray(-jp.inf, dtype=jp.float32),
            "teacher/target_step_y_min": jp.asarray(jp.inf, dtype=jp.float32),
            "teacher/target_step_y_max": jp.asarray(-jp.inf, dtype=jp.float32),
            "teacher/target_step_x_sq_mean": jp.zeros(()),
            "teacher/target_step_y_sq_mean": jp.zeros(()),
            "teacher/swing_foot": jp.asarray(-1.0, dtype=jp.float32),
            "teacher/step_required_event_rate": jp.zeros(()),
            "teacher/swing_foot_match_frac": jp.zeros(()),
            "teacher/swing_foot_match_events": jp.zeros(()),
            "teacher/whole_body_active_frac": jp.zeros(()),
            "teacher/whole_body_active_count": jp.zeros(()),
            "teacher/whole_body_active_during_clean_frac": jp.zeros(()),
            "teacher/whole_body_active_during_push_frac": jp.zeros(()),
            "teacher/recovery_height_target_mean": jp.zeros(()),
            "teacher/recovery_height_error": jp.zeros(()),
            "teacher/recovery_height_in_band_frac": jp.zeros(()),
            "teacher/com_velocity_target_mean": jp.zeros(()),
            "teacher/com_velocity_error": jp.zeros(()),
            "teacher/com_velocity_target_hit_frac": jp.zeros(()),
            "recovery/visible_step_rate": jp.zeros(()),
            "recovery/min_height": jp.zeros(()),
            "recovery/max_knee_flex": jp.zeros(()),
            "recovery/first_step_dist_abs": jp.zeros(()),
            "visible_step_rate_hard": jp.zeros(()),
            # v0.10.3: Tracking metrics for walking exit criteria
            "tracking/vel_error": vel_error,  # |forward_vel - velocity_cmd|
            "tracking/max_torque": max_torque_ratio,  # max(|torque|/limit)
            "tracking/avg_torque": avg_torque,  # mean(|torque|) in Nm
            "unnecessary_step_rate": jp.zeros(()),  # Overwritten in step() using touchdown context
        }

        return total, components

    def _get_termination(
        self, data: mjx.Data, step_count: int
    ) -> tuple[jax.Array, jax.Array, jax.Array, Dict[str, jax.Array]]:
        """Check termination conditions.

        v0.10.2: Added termination diagnostics for debugging.

        Returns:
            (done, terminated, truncated, termination_info) where:
            - done: True if episode ended (terminated OR truncated)
            - terminated: True if ended due to failure (falling or bad orientation)
            - truncated: True if ended due to time limit (success)
            - termination_info: Dict with per-condition flags for diagnostics
        """
        # Cache root pose (used for height + orientation)
        root_pose = self._cal.get_root_pose(data)
        height = root_pose.height

        # Height termination (failure - robot fell)
        height_too_low = height < self._config.env.min_height
        height_too_high = height > self._config.env.max_height
        height_fail = height_too_low | height_too_high

        # v0.10.1: Orientation termination (extreme pitch/roll)
        roll, pitch, _ = root_pose.euler_angles()
        pitch_fail = jp.abs(pitch) > self._config.env.max_pitch
        roll_fail = jp.abs(roll) > self._config.env.max_roll
        orientation_fail = pitch_fail | roll_fail

        # Combined failure termination
        # v0.17.3a: relaxed termination — only terminate on height, not orientation
        if self._config.env.use_relaxed_termination:
            terminated = height_fail
        else:
            terminated = height_fail | orientation_fail

        # Episode length truncation (success - reached max steps without failing)
        truncated = (step_count >= self._config.env.max_episode_steps) & ~terminated

        # Combined done flag
        done = terminated | truncated

        # v0.10.2: Termination diagnostics
        termination_info = {
            "term/height_low": jp.where(height_too_low, 1.0, 0.0),
            "term/height_high": jp.where(height_too_high, 1.0, 0.0),
            "term/pitch": jp.where(pitch_fail, 1.0, 0.0),
            "term/roll": jp.where(roll_fail, 1.0, 0.0),
            "term/truncated": jp.where(truncated, 1.0, 0.0),
            # Raw values for debugging
            "term/pitch_val": pitch,
            "term/roll_val": roll,
            "term/height_val": height,
        }

        return (
            jp.where(done, 1.0, 0.0),
            jp.where(terminated, 1.0, 0.0),
            jp.where(truncated, 1.0, 0.0),
            termination_info,
        )

    # =========================================================================
    # Properties (MjxEnv interface)
    # =========================================================================

    @property
    def xml_path(self) -> str:
        """Path to the xml file for the environment (required by MjxEnv)."""
        return str(self._model_path)

    @property
    def action_size(self) -> int:
        return self._mjx_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    @property
    def cal(self) -> ControlAbstractionLayer:
        """Access the Control Abstraction Layer for joint/actuator operations."""
        return self._cal

    @property
    def observation_size(self) -> int:
        """Return observation size using shared ObsLayout.

        v0.7.0: Velocities are now in heading-local frame for AMP parity.
        The observation layout is unchanged.

        See ObsLayout class for component breakdown.
        """
        return self._policy_spec.model.obs_dim
