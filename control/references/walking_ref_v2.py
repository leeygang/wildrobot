"""Support-first hybrid walking reference generator v2 (M2.5)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from math import cos, pi, sin
from typing import Tuple

import jax.numpy as jnp

from .locomotion_contract import LocomotionReferenceState


Vec2 = Tuple[float, float]
Vec3 = Tuple[float, float, float]

LEFT_STANCE = 0
RIGHT_STANCE = 1


class WalkingRefV2Mode(IntEnum):
    """Hybrid nominal reference mode."""

    STARTUP_SUPPORT_RAMP = 0
    SUPPORT_STABILIZE = 1
    SWING_RELEASE = 2
    TOUCHDOWN_CAPTURE = 3
    POST_TOUCHDOWN_SETTLE = 4


@dataclass(frozen=True)
class WalkingRefV2Config:
    step_time_s: float = 0.50
    nominal_com_height_m: float = 0.39
    nominal_lateral_foot_offset_m: float = 0.08
    min_step_length_m: float = 0.01
    max_step_length_m: float = 0.06
    max_lateral_step_m: float = 0.10
    swing_height_m: float = 0.025
    pelvis_roll_bias_rad: float = 0.02
    pelvis_pitch_gain: float = 0.010
    max_pelvis_pitch_rad: float = 0.02
    max_forward_speed_mps: float = 0.35
    dcm_correction_gain: float = 0.12
    support_pitch_start_rad: float = 0.06
    support_pitch_rate_start_rad_s: float = 0.60
    support_overspeed_deadband: float = 0.01
    support_health_gain: float = 10.0
    support_open_threshold: float = 0.60
    support_release_threshold: float = 0.30
    support_release_phase_start: float = 0.45
    support_foothold_min_scale: float = 0.15
    support_phase_min_scale: float = 0.05
    support_swing_progress_min_scale: float = 0.0
    support_stabilize_max_foothold_scale: float = 0.35
    post_settle_swing_scale: float = 0.15
    startup_ramp_s: float = 0.18
    startup_pelvis_height_offset_m: float = 0.035
    startup_support_open_health: float = 0.45
    max_lateral_release_m: float = 0.02
    touchdown_phase_min: float = 0.55
    capture_hold_s: float = 0.04
    settle_hold_s: float = 0.04
    support_pelvis_height_offset_m: float = 0.050
    debug_force_support_only: bool = False

    def __post_init__(self) -> None:
        if self.step_time_s <= 0.0:
            raise ValueError("step_time_s must be > 0")
        if self.nominal_com_height_m <= 0.0:
            raise ValueError("nominal_com_height_m must be > 0")
        if self.min_step_length_m < 0.0 or self.max_step_length_m <= self.min_step_length_m:
            raise ValueError("step length bounds are invalid")
        if self.max_lateral_step_m <= 0.0:
            raise ValueError("max_lateral_step_m must be > 0")
        if self.max_forward_speed_mps <= 0.0:
            raise ValueError("max_forward_speed_mps must be > 0")
        if not 0.0 <= self.dcm_correction_gain <= 1.0:
            raise ValueError("dcm_correction_gain must be in [0, 1]")
        if not 0.0 <= self.support_release_threshold <= self.support_open_threshold <= 1.0:
            raise ValueError("support thresholds must satisfy 0<=release<=open<=1")
        if not 0.0 <= self.support_release_phase_start <= 0.95:
            raise ValueError("support_release_phase_start must be in [0, 0.95]")
        if not 0.0 <= self.support_foothold_min_scale <= 1.0:
            raise ValueError("support_foothold_min_scale must be in [0, 1]")
        if not 0.0 < self.support_phase_min_scale <= 1.0:
            raise ValueError("support_phase_min_scale must be in (0, 1]")
        if not 0.0 <= self.support_swing_progress_min_scale <= 1.0:
            raise ValueError("support_swing_progress_min_scale must be in [0, 1]")
        if not 0.0 <= self.support_stabilize_max_foothold_scale <= 1.0:
            raise ValueError("support_stabilize_max_foothold_scale must be in [0, 1]")
        if not 0.0 <= self.post_settle_swing_scale <= 1.0:
            raise ValueError("post_settle_swing_scale must be in [0, 1]")
        if self.startup_ramp_s < 0.0:
            raise ValueError("startup_ramp_s must be >= 0")
        if self.startup_pelvis_height_offset_m < 0.0:
            raise ValueError("startup_pelvis_height_offset_m must be >= 0")
        if not 0.0 <= self.startup_support_open_health <= 1.0:
            raise ValueError("startup_support_open_health must be in [0, 1]")
        if self.support_pelvis_height_offset_m < 0.0:
            raise ValueError("support_pelvis_height_offset_m must be >= 0")
        if not 0.0 <= self.max_lateral_release_m <= self.max_lateral_step_m:
            raise ValueError("max_lateral_release_m must be in [0, max_lateral_step_m]")
        if not 0.0 <= self.touchdown_phase_min <= 1.0:
            raise ValueError("touchdown_phase_min must be in [0, 1]")
        if self.capture_hold_s < 0.0 or self.settle_hold_s < 0.0:
            raise ValueError("capture_hold_s/settle_hold_s must be >= 0")


@dataclass(frozen=True)
class WalkingRefV2State:
    phase_time_s: float = 0.0
    stance_foot_id: int = LEFT_STANCE
    stance_switch_count: int = 0
    mode_id: int = int(WalkingRefV2Mode.STARTUP_SUPPORT_RAMP)
    mode_time_s: float = 0.0

    def __post_init__(self) -> None:
        if self.stance_foot_id not in (LEFT_STANCE, RIGHT_STANCE):
            raise ValueError("stance_foot_id must be 0(left) or 1(right)")
        if self.phase_time_s < 0.0 or self.mode_time_s < 0.0:
            raise ValueError("phase_time_s/mode_time_s must be >= 0")
        if self.mode_id not in {int(m) for m in WalkingRefV2Mode}:
            raise ValueError("mode_id is invalid")


@dataclass(frozen=True)
class WalkingRefV2Input:
    forward_speed_mps: float
    com_position_stance_frame: Vec2 = (0.0, 0.0)
    com_velocity_stance_frame: Vec2 = (0.0, 0.0)
    root_pitch_rad: float = 0.0
    root_pitch_rate_rad_s: float = 0.0
    left_foot_loaded: bool = True
    right_foot_loaded: bool = False


@dataclass(frozen=True)
class WalkingReferenceV2Output:
    reference: LocomotionReferenceState
    next_state: WalkingRefV2State
    support_health: float
    support_instability: float
    progression_permission: float
    hybrid_mode_id: int


def _clip(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, value)))


def compute_support_health_v2(*, config: WalkingRefV2Config, inputs: WalkingRefV2Input) -> tuple[float, float]:
    """Compute support health and instability scalar."""
    cmd = _clip(inputs.forward_speed_mps, 0.0, config.max_forward_speed_mps)
    fwd = float(inputs.com_velocity_stance_frame[0])
    overspeed = max(fwd - cmd - config.support_overspeed_deadband, 0.0)
    pitch_excess = max(abs(float(inputs.root_pitch_rad)) - config.support_pitch_start_rad, 0.0)
    pitch_rate_excess = max(
        abs(float(inputs.root_pitch_rate_rad_s)) - config.support_pitch_rate_start_rad_s, 0.0
    )
    single_support = bool(inputs.left_foot_loaded) ^ bool(inputs.right_foot_loaded)
    contact_penalty = 0.0 if single_support else 0.25
    instability = overspeed + pitch_excess + 0.5 * pitch_rate_excess + contact_penalty
    health = 1.0 / (1.0 + config.support_health_gain * instability)
    return _clip(health, 0.0, 1.0), float(instability)


def _permission_from_health(config: WalkingRefV2Config, health: float) -> float:
    denom = max(config.support_open_threshold - config.support_release_threshold, 1e-6)
    permission = (health - config.support_release_threshold) / denom
    return _clip(permission, 0.0, 1.0)


def compute_support_health_v2_jax(
    *,
    config: WalkingRefV2Config,
    forward_speed_mps: jnp.ndarray,
    root_pitch_rad: jnp.ndarray,
    root_pitch_rate_rad_s: jnp.ndarray,
    com_velocity_stance_frame: jnp.ndarray,
    left_foot_loaded: jnp.ndarray,
    right_foot_loaded: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """JAX-friendly support health computation for env reset/step paths."""
    cmd = jnp.clip(
        jnp.asarray(forward_speed_mps, dtype=jnp.float32),
        0.0,
        jnp.asarray(config.max_forward_speed_mps, dtype=jnp.float32),
    )
    fwd = jnp.asarray(com_velocity_stance_frame[0], dtype=jnp.float32)
    overspeed = jnp.maximum(
        fwd - cmd - jnp.asarray(config.support_overspeed_deadband, dtype=jnp.float32), 0.0
    )
    pitch_excess = jnp.maximum(
        jnp.abs(jnp.asarray(root_pitch_rad, dtype=jnp.float32))
        - jnp.asarray(config.support_pitch_start_rad, dtype=jnp.float32),
        0.0,
    )
    pitch_rate_excess = jnp.maximum(
        jnp.abs(jnp.asarray(root_pitch_rate_rad_s, dtype=jnp.float32))
        - jnp.asarray(config.support_pitch_rate_start_rad_s, dtype=jnp.float32),
        0.0,
    )
    single_support = jnp.logical_xor(
        jnp.asarray(left_foot_loaded, dtype=jnp.bool_),
        jnp.asarray(right_foot_loaded, dtype=jnp.bool_),
    )
    contact_penalty = jnp.where(single_support, 0.0, 0.25).astype(jnp.float32)
    instability = overspeed + pitch_excess + 0.5 * pitch_rate_excess + contact_penalty
    health = 1.0 / (
        1.0 + jnp.asarray(config.support_health_gain, dtype=jnp.float32) * instability
    )
    return jnp.clip(health, 0.0, 1.0).astype(jnp.float32), instability.astype(jnp.float32)


def step_reference_v2_jax(
    *,
    config: WalkingRefV2Config,
    phase_time_s: jnp.ndarray,
    stance_foot_id: jnp.ndarray,
    stance_switch_count: jnp.ndarray,
    mode_id: jnp.ndarray,
    mode_time_s: jnp.ndarray,
    forward_speed_mps: jnp.ndarray,
    com_position_stance_frame: jnp.ndarray,
    com_velocity_stance_frame: jnp.ndarray,
    root_pitch_rad: jnp.ndarray,
    root_pitch_rate_rad_s: jnp.ndarray,
    left_foot_loaded: jnp.ndarray,
    right_foot_loaded: jnp.ndarray,
    dt_s: float,
) -> tuple[dict[str, jnp.ndarray], jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """JAX-friendly walking_ref_v2 stepping used by training env rollouts."""
    support_health, support_instability = compute_support_health_v2_jax(
        config=config,
        forward_speed_mps=forward_speed_mps,
        root_pitch_rad=root_pitch_rad,
        root_pitch_rate_rad_s=root_pitch_rate_rad_s,
        com_velocity_stance_frame=com_velocity_stance_frame,
        left_foot_loaded=left_foot_loaded,
        right_foot_loaded=right_foot_loaded,
    )
    denom = jnp.maximum(
        jnp.asarray(config.support_open_threshold - config.support_release_threshold, dtype=jnp.float32),
        jnp.asarray(1e-6, dtype=jnp.float32),
    )
    progression_permission = jnp.clip(
        (support_health - jnp.asarray(config.support_release_threshold, dtype=jnp.float32))
        / denom,
        0.0,
        1.0,
    ).astype(jnp.float32)
    phase_scale = jnp.asarray(config.support_phase_min_scale, dtype=jnp.float32) + (
        1.0 - jnp.asarray(config.support_phase_min_scale, dtype=jnp.float32)
    ) * progression_permission

    startup_mode = jnp.asarray(int(WalkingRefV2Mode.STARTUP_SUPPORT_RAMP), dtype=jnp.int32)
    phase_time = jnp.where(
        jnp.asarray(mode_id, dtype=jnp.int32) == startup_mode,
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(phase_time_s, dtype=jnp.float32)
        + jnp.asarray(dt_s, dtype=jnp.float32) * phase_scale,
    )
    stance = jnp.asarray(stance_foot_id, dtype=jnp.int32)
    switches = jnp.asarray(stance_switch_count, dtype=jnp.int32)
    mode = jnp.asarray(mode_id, dtype=jnp.int32)
    mode_time = jnp.asarray(mode_time_s, dtype=jnp.float32) + jnp.asarray(
        dt_s, dtype=jnp.float32
    )
    step_time = jnp.asarray(config.step_time_s, dtype=jnp.float32)
    phase = jnp.clip(phase_time / jnp.maximum(step_time, 1e-6), 0.0, 1.0)
    stance_is_left = stance == jnp.asarray(LEFT_STANCE, dtype=jnp.int32)
    swing_loaded = jnp.where(
        stance_is_left,
        jnp.asarray(right_foot_loaded, dtype=jnp.bool_),
        jnp.asarray(left_foot_loaded, dtype=jnp.bool_),
    )

    support_mode = jnp.asarray(int(WalkingRefV2Mode.SUPPORT_STABILIZE), dtype=jnp.int32)
    swing_mode = jnp.asarray(int(WalkingRefV2Mode.SWING_RELEASE), dtype=jnp.int32)
    capture_mode = jnp.asarray(int(WalkingRefV2Mode.TOUCHDOWN_CAPTURE), dtype=jnp.int32)
    settle_mode = jnp.asarray(int(WalkingRefV2Mode.POST_TOUCHDOWN_SETTLE), dtype=jnp.int32)

    startup_denom = jnp.maximum(jnp.asarray(config.startup_ramp_s, dtype=jnp.float32), 1e-6)
    startup_alpha = jnp.clip(mode_time / startup_denom, 0.0, 1.0)
    startup_stable = (
        (support_health >= jnp.asarray(config.startup_support_open_health, dtype=jnp.float32))
        & (
            jnp.abs(jnp.asarray(root_pitch_rad, dtype=jnp.float32))
            <= jnp.asarray(config.support_pitch_start_rad, dtype=jnp.float32)
        )
        & (
            jnp.abs(jnp.asarray(root_pitch_rate_rad_s, dtype=jnp.float32))
            <= jnp.asarray(config.support_pitch_rate_start_rad_s, dtype=jnp.float32)
        )
    )

    from_startup = mode == startup_mode
    from_support = mode == support_mode
    from_swing = mode == swing_mode
    from_capture = mode == capture_mode
    from_settle = mode == settle_mode

    to_support_from_startup = from_startup & (
        (mode_time >= jnp.asarray(config.startup_ramp_s, dtype=jnp.float32))
        | ((startup_alpha >= jnp.asarray(0.85, dtype=jnp.float32)) & startup_stable)
    )
    to_swing = (
        from_support
        & (progression_permission >= jnp.asarray(config.support_open_threshold, dtype=jnp.float32))
        & (phase >= jnp.asarray(config.support_release_phase_start, dtype=jnp.float32))
    )
    to_capture = (
        from_swing
        & swing_loaded
        & (phase >= jnp.asarray(config.touchdown_phase_min, dtype=jnp.float32))
    )
    to_settle = from_capture & (mode_time >= jnp.asarray(config.capture_hold_s, dtype=jnp.float32))
    to_support = from_settle & (mode_time >= jnp.asarray(config.settle_hold_s, dtype=jnp.float32))

    mode = jnp.where(
        to_support_from_startup,
        support_mode,
        jnp.where(
            to_swing,
            swing_mode,
            jnp.where(
                to_capture,
                capture_mode,
                jnp.where(to_settle, settle_mode, jnp.where(to_support, support_mode, mode)),
            ),
        ),
    )
    transitioned = to_support_from_startup | to_swing | to_capture | to_settle | to_support
    mode_time = jnp.where(transitioned, 0.0, mode_time)

    switched = phase_time >= step_time
    phase_time = jnp.where(switched, 0.0, phase_time)
    stance = jnp.where(switched, 1 - stance, stance)
    switches = jnp.where(switched, switches + 1, switches)
    mode = jnp.where(switched, settle_mode, mode)
    mode_time = jnp.where(switched, 0.0, mode_time)
    phase = jnp.where(switched, 0.0, phase)

    if config.debug_force_support_only:
        mode = support_mode
        mode_time = jnp.asarray(0.0, dtype=jnp.float32)
        phase_time = jnp.asarray(0.0, dtype=jnp.float32)
        phase = jnp.asarray(0.0, dtype=jnp.float32)

    speed = jnp.clip(
        jnp.asarray(forward_speed_mps, dtype=jnp.float32),
        0.0,
        jnp.asarray(config.max_forward_speed_mps, dtype=jnp.float32),
    )
    nominal_step = jnp.clip(
        speed * step_time,
        jnp.asarray(config.min_step_length_m, dtype=jnp.float32),
        jnp.asarray(config.max_step_length_m, dtype=jnp.float32),
    )
    omega = jnp.sqrt(
        jnp.asarray(9.81, dtype=jnp.float32)
        / jnp.maximum(jnp.asarray(config.nominal_com_height_m, dtype=jnp.float32), 1e-6)
    )
    cp_x = jnp.asarray(com_position_stance_frame[0], dtype=jnp.float32) + jnp.asarray(
        com_velocity_stance_frame[0], dtype=jnp.float32
    ) / jnp.maximum(omega, 1e-6)
    b = jnp.exp(-omega * step_time)
    x_foot_dcm = (nominal_step - b * cp_x) / jnp.maximum(1.0 - b, 1e-6)
    x_foot_nom = (1.0 - jnp.asarray(config.dcm_correction_gain, dtype=jnp.float32)) * nominal_step + jnp.asarray(
        config.dcm_correction_gain, dtype=jnp.float32
    ) * x_foot_dcm
    x_foot_nom = jnp.clip(
        x_foot_nom,
        jnp.asarray(config.min_step_length_m, dtype=jnp.float32),
        jnp.asarray(config.max_step_length_m, dtype=jnp.float32),
    )

    foothold_scale = jnp.asarray(config.support_foothold_min_scale, dtype=jnp.float32) + (
        1.0 - jnp.asarray(config.support_foothold_min_scale, dtype=jnp.float32)
    ) * progression_permission
    foothold_scale = jnp.where(
        mode == support_mode,
        jnp.minimum(
            foothold_scale,
            jnp.asarray(config.support_stabilize_max_foothold_scale, dtype=jnp.float32),
        ),
        foothold_scale,
    )
    x_foot = jnp.asarray(config.min_step_length_m, dtype=jnp.float32) + foothold_scale * (
        x_foot_nom - jnp.asarray(config.min_step_length_m, dtype=jnp.float32)
    )
    x_foot = jnp.where(
        mode == startup_mode,
        jnp.asarray(config.min_step_length_m, dtype=jnp.float32)
        + startup_alpha
        * (x_foot - jnp.asarray(config.min_step_length_m, dtype=jnp.float32)),
        x_foot,
    )
    x_foot = jnp.clip(
        x_foot,
        jnp.asarray(config.min_step_length_m, dtype=jnp.float32),
        jnp.asarray(config.max_step_length_m, dtype=jnp.float32),
    )

    y_sign = jnp.where(stance == jnp.asarray(LEFT_STANCE, dtype=jnp.int32), -1.0, 1.0)
    y_foot = jnp.clip(
        y_sign * jnp.asarray(config.nominal_lateral_foot_offset_m, dtype=jnp.float32),
        -jnp.asarray(config.max_lateral_step_m, dtype=jnp.float32),
        jnp.asarray(config.max_lateral_step_m, dtype=jnp.float32),
    )

    release_prog = jnp.clip(
        (phase - jnp.asarray(config.support_release_phase_start, dtype=jnp.float32))
        / jnp.maximum(1.0 - jnp.asarray(config.support_release_phase_start, dtype=jnp.float32), 1e-6),
        0.0,
        1.0,
    )
    base_release = jnp.asarray(config.support_swing_progress_min_scale, dtype=jnp.float32) + (
        1.0 - jnp.asarray(config.support_swing_progress_min_scale, dtype=jnp.float32)
    ) * release_prog
    swing_release = jnp.where(
        (mode == startup_mode) | (mode == support_mode),
        0.0,
        progression_permission * base_release,
    )
    swing_release = jnp.where(mode == capture_mode, 1.0, swing_release)
    swing_release = jnp.where(
        mode == settle_mode,
        jnp.asarray(config.post_settle_swing_scale, dtype=jnp.float32) * progression_permission,
        swing_release,
    )

    base_support_y = y_foot
    lateral_release_y = y_sign * jnp.asarray(config.max_lateral_release_m, dtype=jnp.float32) * swing_release
    swing_x = x_foot * swing_release
    swing_y = jnp.clip(
        base_support_y + lateral_release_y,
        -jnp.asarray(config.max_lateral_step_m, dtype=jnp.float32),
        jnp.asarray(config.max_lateral_step_m, dtype=jnp.float32),
    )
    swing_z = jnp.asarray(config.swing_height_m, dtype=jnp.float32) * jnp.sin(
        jnp.asarray(pi, dtype=jnp.float32) * phase
    )
    swing_release_vel_scale = jnp.where(mode == swing_mode, progression_permission * base_release, 0.0)
    swing_vel = jnp.asarray(
        [
            x_foot * swing_release / jnp.maximum(step_time, 1e-6),
            y_sign
            * jnp.asarray(config.max_lateral_release_m, dtype=jnp.float32)
            * swing_release_vel_scale
            / jnp.maximum(step_time, 1e-6),
            (jnp.asarray(config.swing_height_m, dtype=jnp.float32) * jnp.asarray(pi, dtype=jnp.float32) / jnp.maximum(step_time, 1e-6))
            * jnp.cos(jnp.asarray(pi, dtype=jnp.float32) * phase),
        ],
        dtype=jnp.float32,
    )
    pelvis_roll_target = jnp.where(
        stance == jnp.asarray(LEFT_STANCE, dtype=jnp.int32),
        -jnp.asarray(config.pelvis_roll_bias_rad, dtype=jnp.float32),
        jnp.asarray(config.pelvis_roll_bias_rad, dtype=jnp.float32),
    )
    pelvis_pitch_target = jnp.clip(
        jnp.asarray(config.pelvis_pitch_gain, dtype=jnp.float32) * speed * progression_permission,
        -jnp.asarray(config.max_pelvis_pitch_rad, dtype=jnp.float32),
        jnp.asarray(config.max_pelvis_pitch_rad, dtype=jnp.float32),
    )
    pelvis_roll = jnp.where(mode == startup_mode, startup_alpha * pelvis_roll_target, pelvis_roll_target)
    pelvis_pitch = jnp.where(mode == startup_mode, startup_alpha * pelvis_pitch_target, pelvis_pitch_target)
    
    pelvis_height_startup = jnp.asarray(config.nominal_com_height_m, dtype=jnp.float32) + jnp.asarray(
        config.startup_pelvis_height_offset_m, dtype=jnp.float32
    )
    pelvis_height_support = jnp.asarray(config.nominal_com_height_m, dtype=jnp.float32) + jnp.asarray(
        config.support_pelvis_height_offset_m, dtype=jnp.float32
    )
    pelvis_height = jnp.where(
        mode == startup_mode,
        pelvis_height_startup
        + startup_alpha * (pelvis_height_support - pelvis_height_startup),
        jnp.where(
            mode == support_mode,
            pelvis_height_support,
            jnp.asarray(config.nominal_com_height_m, dtype=jnp.float32),
        ),
    )

    ref = {
        "gait_phase_sin": jnp.sin(2.0 * jnp.asarray(pi, dtype=jnp.float32) * phase).astype(jnp.float32),
        "gait_phase_cos": jnp.cos(2.0 * jnp.asarray(pi, dtype=jnp.float32) * phase).astype(jnp.float32),
        "phase_progress": phase.astype(jnp.float32),
        "stance_foot_id": stance.astype(jnp.int32),
        "next_foothold": jnp.asarray([x_foot, y_foot], dtype=jnp.float32),
        "swing_pos": jnp.asarray([swing_x, swing_y, swing_z], dtype=jnp.float32),
        "swing_vel": swing_vel,
        "pelvis_height": pelvis_height.astype(jnp.float32),
        "pelvis_roll": pelvis_roll.astype(jnp.float32),
        "pelvis_pitch": pelvis_pitch.astype(jnp.float32),
    }
    return (
        ref,
        phase_time.astype(jnp.float32),
        stance.astype(jnp.int32),
        switches.astype(jnp.int32),
        mode.astype(jnp.int32),
        mode_time.astype(jnp.float32),
        support_health.astype(jnp.float32),
        support_instability.astype(jnp.float32),
        progression_permission.astype(jnp.float32),
    )


def step_reference_v2(
    *,
    config: WalkingRefV2Config,
    state: WalkingRefV2State,
    inputs: WalkingRefV2Input,
    dt_s: float,
) -> WalkingReferenceV2Output:
    """Advance support-first hybrid reference state and emit locomotion reference."""
    if dt_s <= 0.0:
        raise ValueError("dt_s must be > 0")

    support_health, support_instability = compute_support_health_v2(config=config, inputs=inputs)
    progression_permission = _permission_from_health(config, support_health)
    phase_scale = config.support_phase_min_scale + (1.0 - config.support_phase_min_scale) * progression_permission

    mode = WalkingRefV2Mode(state.mode_id)
    phase_time = 0.0 if mode == WalkingRefV2Mode.STARTUP_SUPPORT_RAMP else state.phase_time_s + dt_s * phase_scale
    stance = state.stance_foot_id
    switch_count = state.stance_switch_count
    mode_time = state.mode_time_s + dt_s

    phase = _clip(phase_time / max(config.step_time_s, 1e-6), 0.0, 1.0)
    stance_is_left = stance == LEFT_STANCE
    swing_loaded = bool(inputs.right_foot_loaded if stance_is_left else inputs.left_foot_loaded)

    startup_alpha = _clip(mode_time / max(config.startup_ramp_s, 1e-6), 0.0, 1.0)
    startup_stable = (
        support_health >= config.startup_support_open_health
        and abs(float(inputs.root_pitch_rad)) <= config.support_pitch_start_rad
        and abs(float(inputs.root_pitch_rate_rad_s)) <= config.support_pitch_rate_start_rad_s
    )

    if mode == WalkingRefV2Mode.STARTUP_SUPPORT_RAMP:
        if mode_time >= config.startup_ramp_s or (startup_alpha >= 0.85 and startup_stable):
            mode = WalkingRefV2Mode.SUPPORT_STABILIZE
            mode_time = 0.0
    elif mode == WalkingRefV2Mode.SUPPORT_STABILIZE:
        if progression_permission >= config.support_open_threshold and phase >= config.support_release_phase_start:
            mode = WalkingRefV2Mode.SWING_RELEASE
            mode_time = 0.0
    elif mode == WalkingRefV2Mode.SWING_RELEASE:
        if swing_loaded and phase >= config.touchdown_phase_min:
            mode = WalkingRefV2Mode.TOUCHDOWN_CAPTURE
            mode_time = 0.0
    elif mode == WalkingRefV2Mode.TOUCHDOWN_CAPTURE:
        if mode_time >= config.capture_hold_s:
            mode = WalkingRefV2Mode.POST_TOUCHDOWN_SETTLE
            mode_time = 0.0
    elif mode == WalkingRefV2Mode.POST_TOUCHDOWN_SETTLE:
        if mode_time >= config.settle_hold_s:
            mode = WalkingRefV2Mode.SUPPORT_STABILIZE
            mode_time = 0.0

    if phase_time >= config.step_time_s:
        phase_time = 0.0
        stance = RIGHT_STANCE if stance == LEFT_STANCE else LEFT_STANCE
        switch_count += 1
        mode = WalkingRefV2Mode.POST_TOUCHDOWN_SETTLE
        mode_time = 0.0
        phase = 0.0

    speed = _clip(inputs.forward_speed_mps, 0.0, config.max_forward_speed_mps)
    nominal_step = _clip(speed * config.step_time_s, config.min_step_length_m, config.max_step_length_m)
    omega = (9.81 / max(config.nominal_com_height_m, 1e-6)) ** 0.5
    cp_x = float(inputs.com_position_stance_frame[0]) + float(inputs.com_velocity_stance_frame[0]) / max(omega, 1e-6)
    b = float(pow(2.718281828459045, -omega * config.step_time_s))
    x_foot_dcm = (nominal_step - b * cp_x) / max(1.0 - b, 1e-6)
    x_foot_nom = (1.0 - config.dcm_correction_gain) * nominal_step + config.dcm_correction_gain * x_foot_dcm
    x_foot_nom = _clip(x_foot_nom, config.min_step_length_m, config.max_step_length_m)

    foothold_scale = config.support_foothold_min_scale + (1.0 - config.support_foothold_min_scale) * progression_permission
    if mode == WalkingRefV2Mode.SUPPORT_STABILIZE:
        foothold_scale = min(foothold_scale, config.support_stabilize_max_foothold_scale)
    x_foot = config.min_step_length_m + foothold_scale * (x_foot_nom - config.min_step_length_m)
    if mode == WalkingRefV2Mode.STARTUP_SUPPORT_RAMP:
        x_foot = config.min_step_length_m + startup_alpha * (x_foot - config.min_step_length_m)
    x_foot = _clip(x_foot, config.min_step_length_m, config.max_step_length_m)

    y_sign = -1.0 if stance == LEFT_STANCE else 1.0
    y_foot = _clip(y_sign * config.nominal_lateral_foot_offset_m, -config.max_lateral_step_m, config.max_lateral_step_m)

    release_prog = _clip(
        (phase - config.support_release_phase_start) / max(1.0 - config.support_release_phase_start, 1e-6),
        0.0,
        1.0,
    )
    if mode in (WalkingRefV2Mode.STARTUP_SUPPORT_RAMP, WalkingRefV2Mode.SUPPORT_STABILIZE):
        swing_release = 0.0
    elif mode == WalkingRefV2Mode.SWING_RELEASE:
        base = config.support_swing_progress_min_scale + (
            1.0 - config.support_swing_progress_min_scale
        ) * release_prog
        swing_release = progression_permission * base
    elif mode == WalkingRefV2Mode.TOUCHDOWN_CAPTURE:
        swing_release = 1.0
    else:
        swing_release = config.post_settle_swing_scale * progression_permission

    base_support_y = y_foot
    lateral_release_y = y_sign * config.max_lateral_release_m * swing_release
    swing_x = x_foot * swing_release
    swing_y = _clip(base_support_y + lateral_release_y, -config.max_lateral_step_m, config.max_lateral_step_m)
    swing_z = config.swing_height_m * sin(pi * phase)
    swing_release_vel_scale = progression_permission * base if mode == WalkingRefV2Mode.SWING_RELEASE else 0.0
    swing_vel = (
        x_foot * swing_release / max(config.step_time_s, 1e-6),
        y_sign * config.max_lateral_release_m * swing_release_vel_scale / max(config.step_time_s, 1e-6),
        (config.swing_height_m * pi / max(config.step_time_s, 1e-6)) * cos(pi * phase),
    )

    pelvis_roll_target = (-config.pelvis_roll_bias_rad if stance == LEFT_STANCE else config.pelvis_roll_bias_rad)
    pelvis_pitch_target = _clip(
        config.pelvis_pitch_gain * speed * progression_permission,
        -config.max_pelvis_pitch_rad,
        config.max_pelvis_pitch_rad,
    )
    pelvis_height_startup = config.nominal_com_height_m + config.startup_pelvis_height_offset_m
    pelvis_height_support = config.nominal_com_height_m + config.support_pelvis_height_offset_m
    
    if mode == WalkingRefV2Mode.STARTUP_SUPPORT_RAMP:
        pelvis_roll = startup_alpha * pelvis_roll_target
        pelvis_pitch = startup_alpha * pelvis_pitch_target
        pelvis_height = pelvis_height_startup + startup_alpha * (pelvis_height_support - pelvis_height_startup)
    elif mode == WalkingRefV2Mode.SUPPORT_STABILIZE:
        pelvis_roll = pelvis_roll_target
        pelvis_pitch = pelvis_pitch_target
        pelvis_height = pelvis_height_support
    else:
        pelvis_roll = pelvis_roll_target
        pelvis_pitch = pelvis_pitch_target
        pelvis_height = config.nominal_com_height_m

    ref = LocomotionReferenceState(
        gait_phase_sin=float(sin(2.0 * pi * phase)),
        gait_phase_cos=float(cos(2.0 * pi * phase)),
        stance_foot_id=stance,
        desired_next_foothold_stance_frame=(x_foot, y_foot),
        desired_swing_foot_position=(swing_x, swing_y, swing_z),
        desired_swing_foot_velocity=swing_vel,
        desired_pelvis_height_m=pelvis_height,
        desired_pelvis_roll_rad=pelvis_roll,
        desired_pelvis_pitch_rad=pelvis_pitch,
    )
    next_state = WalkingRefV2State(
        phase_time_s=phase_time,
        stance_foot_id=stance,
        stance_switch_count=switch_count,
        mode_id=int(mode),
        mode_time_s=mode_time,
    )
    return WalkingReferenceV2Output(
        reference=ref,
        next_state=next_state,
        support_health=support_health,
        support_instability=support_instability,
        progression_permission=progression_permission,
        hybrid_mode_id=int(mode),
    )
