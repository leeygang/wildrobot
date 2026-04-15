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


class StartupRouteStage(IntEnum):
    """Fixed startup/support route stages (A -> W1 -> W2 -> W3 -> B)."""

    A = 0
    W1 = 1
    W2 = 2
    W3 = 3
    B = 4


class StartupRouteReason(IntEnum):
    """Reason code for staged-route progression/hold decisions."""

    NONE = 0
    PELVIS_NOT_REALIZED = 1
    JOINT_TRACKING_LAG = 2
    ROOT_PITCH_LIMIT = 3
    ROOT_PITCH_RATE_LIMIT = 4
    SUPPORT_HEALTH_LOW = 5
    ROUTE_COMPLETE = 6
    TIMEOUT_FALLBACK = 7


@dataclass(frozen=True)
class WalkingRefV2Config:
    step_time_s: float = 0.50
    walking_pelvis_height_m: float = 0.39
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
    startup_ramp_s: float = 1.20
    startup_pelvis_height_offset_m: float = 0.07
    startup_support_open_health: float = 0.25
    startup_handoff_pitch_max_rad: float = 0.30
    startup_handoff_pitch_rate_max_rad_s: float = 1.50
    startup_readiness_knee_err_good_rad: float = 0.10
    startup_readiness_knee_err_bad_rad: float = 0.45
    startup_readiness_ankle_err_good_rad: float = 0.08
    startup_readiness_ankle_err_bad_rad: float = 0.35
    startup_readiness_pitch_good_rad: float = 0.05
    startup_readiness_pitch_bad_rad: float = 0.25
    startup_readiness_pitch_rate_good_rad_s: float = 0.60
    startup_readiness_pitch_rate_bad_rad_s: float = 2.00
    startup_readiness_min_health: float = 0.20
    startup_progress_min_scale: float = 0.20
    startup_realization_lead_alpha: float = 0.25
    startup_handoff_min_readiness: float = 0.10
    startup_handoff_min_pelvis_realization: float = 0.60
    startup_handoff_min_alpha: float = 0.85
    startup_handoff_timeout_s: float = 1.20
    startup_target_rate_design_rad_s: float = 0.5235987756
    startup_target_rate_hard_cap_rad_s: float = 1.7453292520
    startup_route_w1_alpha: float = 0.20
    startup_route_w2_alpha: float = 0.50
    startup_route_w3_alpha: float = 0.80
    # Legacy startup route scale triple (used for swing-x / support-chain demand).
    startup_route_w1_scale: float = 0.15
    startup_route_w2_scale: float = 0.45
    startup_route_w3_scale: float = 0.80
    # Stage-specific route geometry channels (A -> W1 -> W2 -> W3 -> B).
    startup_route_w1_support_y_scale: float = 0.25
    startup_route_w2_support_y_scale: float = 0.60
    startup_route_w3_support_y_scale: float = 0.85
    startup_route_w1_pelvis_roll_scale: float = 0.45
    startup_route_w2_pelvis_roll_scale: float = 0.75
    startup_route_w3_pelvis_roll_scale: float = 0.90
    startup_route_w1_pelvis_pitch_scale: float = 0.20
    startup_route_w2_pelvis_pitch_scale: float = 0.50
    startup_route_w3_pelvis_pitch_scale: float = 0.80
    startup_route_w1_pelvis_height_scale: float = 0.15
    startup_route_w2_pelvis_height_scale: float = 0.45
    startup_route_w3_pelvis_height_scale: float = 0.80
    # Config-backed route admission thresholds.
    startup_route_w2_min_pelvis_realization: float = 0.30
    startup_route_w3_min_pelvis_realization: float = 0.55
    startup_route_w2_pitch_relax: float = 1.25
    startup_route_w2_pitch_rate_relax: float = 1.25
    support_entry_shaping_window_s: float = 0.12
    max_lateral_release_m: float = 0.02
    touchdown_phase_min: float = 0.55
    capture_hold_s: float = 0.04
    settle_hold_s: float = 0.04
    support_pelvis_height_offset_m: float = 0.00
    debug_force_support_only: bool = False
    # DCM COM trajectory: let the body fall forward over the stance foot
    # following LIPM dynamics.  The IK automatically produces stance-leg
    # hip extension and ankle rocker from the time-varying pelvis_x.
    com_trajectory_enabled: bool = False
    com_trajectory_max_behind_m: float = 0.01
    com_trajectory_max_ahead_m: float = 0.03
    # v0.19.4-C: COM trajectory mode — "linear" (phase-proportional ramp)
    # or "lipm" (full LIPM cosh/sinh dynamics using stance-start COM state).
    com_trajectory_mode: str = "linear"
    # v0.19.4-C: ankle push-off — explicit plantarflexion during terminal
    # stance, on top of IK-derived ankle angle.
    ankle_pushoff_enabled: bool = False
    ankle_pushoff_phase_start: float = 0.70
    ankle_pushoff_max_rad: float = 0.15

    def __post_init__(self) -> None:
        if self.step_time_s <= 0.0:
            raise ValueError("step_time_s must be > 0")
        if self.walking_pelvis_height_m <= 0.0:
            raise ValueError("walking_pelvis_height_m must be > 0")
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
        if self.startup_handoff_pitch_max_rad < 0.0:
            raise ValueError("startup_handoff_pitch_max_rad must be >= 0")
        if self.startup_handoff_pitch_rate_max_rad_s < 0.0:
            raise ValueError("startup_handoff_pitch_rate_max_rad_s must be >= 0")
        if self.startup_readiness_knee_err_good_rad < 0.0:
            raise ValueError("startup_readiness_knee_err_good_rad must be >= 0")
        if self.startup_readiness_knee_err_bad_rad < self.startup_readiness_knee_err_good_rad:
            raise ValueError("startup_readiness_knee_err_bad_rad must be >= knee_err_good")
        if self.startup_readiness_ankle_err_good_rad < 0.0:
            raise ValueError("startup_readiness_ankle_err_good_rad must be >= 0")
        if self.startup_readiness_ankle_err_bad_rad < self.startup_readiness_ankle_err_good_rad:
            raise ValueError("startup_readiness_ankle_err_bad_rad must be >= ankle_err_good")
        if self.startup_readiness_pitch_good_rad < 0.0:
            raise ValueError("startup_readiness_pitch_good_rad must be >= 0")
        if self.startup_readiness_pitch_bad_rad < self.startup_readiness_pitch_good_rad:
            raise ValueError("startup_readiness_pitch_bad_rad must be >= pitch_good")
        if self.startup_readiness_pitch_rate_good_rad_s < 0.0:
            raise ValueError("startup_readiness_pitch_rate_good_rad_s must be >= 0")
        if (
            self.startup_readiness_pitch_rate_bad_rad_s
            < self.startup_readiness_pitch_rate_good_rad_s
        ):
            raise ValueError("startup_readiness_pitch_rate_bad_rad_s must be >= pitch_rate_good")
        if not 0.0 <= self.startup_readiness_min_health <= 1.0:
            raise ValueError("startup_readiness_min_health must be in [0, 1]")
        if not 0.0 <= self.startup_progress_min_scale <= 1.0:
            raise ValueError("startup_progress_min_scale must be in [0, 1]")
        if not 0.0 <= self.startup_realization_lead_alpha <= 1.0:
            raise ValueError("startup_realization_lead_alpha must be in [0, 1]")
        if not 0.0 <= self.startup_handoff_min_readiness <= 1.0:
            raise ValueError("startup_handoff_min_readiness must be in [0, 1]")
        if not 0.0 <= self.startup_handoff_min_pelvis_realization <= 1.0:
            raise ValueError("startup_handoff_min_pelvis_realization must be in [0, 1]")
        if not 0.0 <= self.startup_handoff_min_alpha <= 1.0:
            raise ValueError("startup_handoff_min_alpha must be in [0, 1]")
        if self.startup_handoff_timeout_s < 0.0:
            raise ValueError("startup_handoff_timeout_s must be >= 0")
        if self.startup_target_rate_design_rad_s < 0.0:
            raise ValueError("startup_target_rate_design_rad_s must be >= 0")
        if self.startup_target_rate_hard_cap_rad_s < self.startup_target_rate_design_rad_s:
            raise ValueError("startup_target_rate_hard_cap_rad_s must be >= design rate")
        if not 0.0 <= self.startup_route_w1_alpha <= self.startup_route_w2_alpha <= self.startup_route_w3_alpha <= 1.0:
            raise ValueError("startup_route_w*_alpha must satisfy 0<=w1<=w2<=w3<=1")
        if not 0.0 <= self.startup_route_w1_scale <= self.startup_route_w2_scale <= self.startup_route_w3_scale <= 1.0:
            raise ValueError("startup_route_w*_scale must satisfy 0<=w1<=w2<=w3<=1")
        if not 0.0 <= self.startup_route_w1_support_y_scale <= self.startup_route_w2_support_y_scale <= self.startup_route_w3_support_y_scale <= 1.0:
            raise ValueError("startup_route_w*_support_y_scale must satisfy 0<=w1<=w2<=w3<=1")
        if not 0.0 <= self.startup_route_w1_pelvis_roll_scale <= self.startup_route_w2_pelvis_roll_scale <= self.startup_route_w3_pelvis_roll_scale <= 1.0:
            raise ValueError("startup_route_w*_pelvis_roll_scale must satisfy 0<=w1<=w2<=w3<=1")
        if not 0.0 <= self.startup_route_w1_pelvis_pitch_scale <= self.startup_route_w2_pelvis_pitch_scale <= self.startup_route_w3_pelvis_pitch_scale <= 1.0:
            raise ValueError("startup_route_w*_pelvis_pitch_scale must satisfy 0<=w1<=w2<=w3<=1")
        if not 0.0 <= self.startup_route_w1_pelvis_height_scale <= self.startup_route_w2_pelvis_height_scale <= self.startup_route_w3_pelvis_height_scale <= 1.0:
            raise ValueError("startup_route_w*_pelvis_height_scale must satisfy 0<=w1<=w2<=w3<=1")
        if not 0.0 <= self.startup_route_w2_min_pelvis_realization <= self.startup_route_w3_min_pelvis_realization <= 1.0:
            raise ValueError(
                "startup_route_w2/w3_min_pelvis_realization must satisfy 0<=w2<=w3<=1"
            )
        if self.startup_route_w2_pitch_relax < 1.0:
            raise ValueError("startup_route_w2_pitch_relax must be >= 1.0")
        if self.startup_route_w2_pitch_rate_relax < 1.0:
            raise ValueError("startup_route_w2_pitch_rate_relax must be >= 1.0")
        if self.support_entry_shaping_window_s < 0.0:
            raise ValueError("support_entry_shaping_window_s must be >= 0")
        if self.support_pelvis_height_offset_m < 0.0:
            raise ValueError("support_pelvis_height_offset_m must be >= 0")
        if not 0.0 <= self.max_lateral_release_m <= self.max_lateral_step_m:
            raise ValueError("max_lateral_release_m must be in [0, max_lateral_step_m]")
        if not 0.0 <= self.touchdown_phase_min <= 1.0:
            raise ValueError("touchdown_phase_min must be in [0, 1]")
        if self.capture_hold_s < 0.0 or self.settle_hold_s < 0.0:
            raise ValueError("capture_hold_s/settle_hold_s must be >= 0")
        if self.com_trajectory_mode not in ("linear", "lipm"):
            raise ValueError("com_trajectory_mode must be 'linear' or 'lipm'")
        if self.ankle_pushoff_enabled:
            if not 0.0 <= self.ankle_pushoff_phase_start <= 1.0:
                raise ValueError("ankle_pushoff_phase_start must be in [0, 1]")
            if self.ankle_pushoff_max_rad < 0.0:
                raise ValueError("ankle_pushoff_max_rad must be >= 0")


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
    com_x_planned: float = 0.0
    ankle_pushoff_rad: float = 0.0


def _clip(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, value)))


def _readiness_component(value_abs: float, good: float, bad: float) -> float:
    """Map abs(error) to [0,1] readiness where good->1 and bad-or-worse->0."""
    if bad <= good:
        return 1.0 if value_abs <= good else 0.0
    return _clip((bad - value_abs) / (bad - good), 0.0, 1.0)


def _startup_pelvis_realization(
    *,
    root_height_m: float,
    startup_height_m: float,
    support_height_m: float,
) -> float:
    """Directional pelvis realization in [0,1]: 0 at startup height, 1 at support height.

    Uses signed progress so overshooting past support clips to 1.0 instead of
    regressing.  Works for both downward ramps (startup > support, normal) and
    upward ramps.
    """
    span = max(abs(support_height_m - startup_height_m), 1e-6)
    direction = 1.0 if startup_height_m >= support_height_m else -1.0
    progress = direction * (startup_height_m - root_height_m)
    return _clip(progress / span, 0.0, 1.0)


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


def _readiness_component_jax(
    value_abs: jnp.ndarray, *, good: float, bad: float
) -> jnp.ndarray:
    good_v = jnp.asarray(good, dtype=jnp.float32)
    bad_v = jnp.asarray(bad, dtype=jnp.float32)
    denom = jnp.maximum(bad_v - good_v, jnp.asarray(1e-6, dtype=jnp.float32))
    comp = (bad_v - jnp.asarray(value_abs, dtype=jnp.float32)) / denom
    return jnp.clip(comp, 0.0, 1.0).astype(jnp.float32)


def _startup_pelvis_realization_jax(
    *,
    root_height_m: jnp.ndarray,
    startup_height_m: jnp.ndarray,
    support_height_m: jnp.ndarray,
) -> jnp.ndarray:
    """Directional pelvis realization: 0 at startup height, 1 at support height.

    Uses signed progress so overshooting past support clips to 1.0.
    """
    span = jnp.maximum(jnp.abs(support_height_m - startup_height_m), jnp.asarray(1e-6, dtype=jnp.float32))
    direction = jnp.sign(startup_height_m - support_height_m)
    progress = direction * (startup_height_m - jnp.asarray(root_height_m, dtype=jnp.float32))
    return jnp.clip(progress / span, 0.0, 1.0).astype(jnp.float32)


def _piecewise_route_scale_jax(
    progress: jnp.ndarray,
    *,
    w1_alpha: float,
    w2_alpha: float,
    w3_alpha: float,
    w1_scale: float,
    w2_scale: float,
    w3_scale: float,
) -> jnp.ndarray:
    """Map startup route progress [0,1] into fixed A/W1/W2/W3/B waypoint scale."""
    p = jnp.clip(jnp.asarray(progress, dtype=jnp.float32), 0.0, 1.0)
    w1_a = jnp.asarray(w1_alpha, dtype=jnp.float32)
    w2_a = jnp.asarray(w2_alpha, dtype=jnp.float32)
    w3_a = jnp.asarray(w3_alpha, dtype=jnp.float32)
    s1 = jnp.asarray(w1_scale, dtype=jnp.float32)
    s2 = jnp.asarray(w2_scale, dtype=jnp.float32)
    s3 = jnp.asarray(w3_scale, dtype=jnp.float32)

    seg1 = s1 * p / jnp.maximum(w1_a, jnp.asarray(1e-6, dtype=jnp.float32))
    seg2 = s1 + (s2 - s1) * (p - w1_a) / jnp.maximum(
        w2_a - w1_a, jnp.asarray(1e-6, dtype=jnp.float32)
    )
    seg3 = s2 + (s3 - s2) * (p - w2_a) / jnp.maximum(
        w3_a - w2_a, jnp.asarray(1e-6, dtype=jnp.float32)
    )
    seg4 = s3 + (1.0 - s3) * (p - w3_a) / jnp.maximum(
        1.0 - w3_a, jnp.asarray(1e-6, dtype=jnp.float32)
    )
    out = jnp.where(p < w1_a, seg1, jnp.where(p < w2_a, seg2, jnp.where(p < w3_a, seg3, seg4)))
    return jnp.clip(out, 0.0, 1.0).astype(jnp.float32)


def _stage_id_from_progress_jax(
    progress: jnp.ndarray, *, w1_alpha: float, w2_alpha: float, w3_alpha: float
) -> jnp.ndarray:
    """Convert route progress into discrete fixed-route stage id."""
    p = jnp.clip(jnp.asarray(progress, dtype=jnp.float32), 0.0, 1.0)
    return jnp.where(
        p >= jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray(int(StartupRouteStage.B), dtype=jnp.int32),
        jnp.where(
            p >= jnp.asarray(w3_alpha, dtype=jnp.float32),
            jnp.asarray(int(StartupRouteStage.W3), dtype=jnp.int32),
            jnp.where(
                p >= jnp.asarray(w2_alpha, dtype=jnp.float32),
                jnp.asarray(int(StartupRouteStage.W2), dtype=jnp.int32),
                jnp.where(
                    p >= jnp.asarray(w1_alpha, dtype=jnp.float32),
                    jnp.asarray(int(StartupRouteStage.W1), dtype=jnp.int32),
                    jnp.asarray(int(StartupRouteStage.A), dtype=jnp.int32),
                ),
            ),
        ),
    ).astype(jnp.int32)


def _first_failed_reason_jax(
    *,
    pelvis_ok: jnp.ndarray,
    joints_ok: jnp.ndarray,
    pitch_ok: jnp.ndarray,
    pitch_rate_ok: jnp.ndarray,
    health_ok: jnp.ndarray,
) -> jnp.ndarray:
    """Return first blocking reason for route progression gate."""
    return jnp.where(
        pelvis_ok,
        jnp.where(
            joints_ok,
            jnp.where(
                pitch_ok,
                jnp.where(
                    pitch_rate_ok,
                    jnp.where(
                        health_ok,
                        jnp.asarray(int(StartupRouteReason.NONE), dtype=jnp.int32),
                        jnp.asarray(int(StartupRouteReason.SUPPORT_HEALTH_LOW), dtype=jnp.int32),
                    ),
                    jnp.asarray(int(StartupRouteReason.ROOT_PITCH_RATE_LIMIT), dtype=jnp.int32),
                ),
                jnp.asarray(int(StartupRouteReason.ROOT_PITCH_LIMIT), dtype=jnp.int32),
            ),
            jnp.asarray(int(StartupRouteReason.JOINT_TRACKING_LAG), dtype=jnp.int32),
        ),
        jnp.asarray(int(StartupRouteReason.PELVIS_NOT_REALIZED), dtype=jnp.int32),
    ).astype(jnp.int32)


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
    root_height_m: jnp.ndarray | None = None,
    left_foot_loaded: jnp.ndarray,
    right_foot_loaded: jnp.ndarray,
    dt_s: float,
    stance_knee_tracking_error_rad: jnp.ndarray | None = None,
    stance_ankle_tracking_error_rad: jnp.ndarray | None = None,
    startup_route_progress_prev: jnp.ndarray | None = None,
    startup_route_ceiling_prev: jnp.ndarray | None = None,
    startup_route_stage_id_prev: jnp.ndarray | None = None,
    startup_route_transition_reason_prev: jnp.ndarray | None = None,
    com_x0_at_stance_start: jnp.ndarray | None = None,
    com_vx0_at_stance_start: jnp.ndarray | None = None,
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
    # During startup and support-stabilize, double-support (both feet loaded)
    # is the expected state.  The health function penalises non-single-support
    # by 0.25, which unfairly prevents swing release.  Remove that penalty
    # while in startup or support-stabilize modes.
    _startup_mode_cst = jnp.asarray(int(WalkingRefV2Mode.STARTUP_SUPPORT_RAMP), dtype=jnp.int32)
    _support_mode_cst = jnp.asarray(int(WalkingRefV2Mode.SUPPORT_STABILIZE), dtype=jnp.int32)
    _is_double_support = jnp.logical_not(jnp.logical_xor(
        jnp.asarray(left_foot_loaded, dtype=jnp.bool_),
        jnp.asarray(right_foot_loaded, dtype=jnp.bool_),
    ))
    _in_ds_exempt_mode = (
        (jnp.asarray(mode_id, dtype=jnp.int32) == _startup_mode_cst)
        | (jnp.asarray(mode_id, dtype=jnp.int32) == _support_mode_cst)
    )
    _ds_exempt = _in_ds_exempt_mode & _is_double_support
    _adj_instab = jnp.maximum(support_instability - jnp.asarray(0.25, dtype=jnp.float32), 0.0)
    _adj_health = jnp.clip(
        1.0 / (1.0 + jnp.asarray(config.support_health_gain, dtype=jnp.float32) * _adj_instab),
        0.0, 1.0,
    ).astype(jnp.float32)
    support_health = jnp.where(_ds_exempt, _adj_health, support_health)
    support_instability = jnp.where(_ds_exempt, _adj_instab, support_instability)
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
    startup_mode = jnp.asarray(int(WalkingRefV2Mode.STARTUP_SUPPORT_RAMP), dtype=jnp.int32)
    support_mode = jnp.asarray(int(WalkingRefV2Mode.SUPPORT_STABILIZE), dtype=jnp.int32)
    swing_mode = jnp.asarray(int(WalkingRefV2Mode.SWING_RELEASE), dtype=jnp.int32)
    capture_mode = jnp.asarray(int(WalkingRefV2Mode.TOUCHDOWN_CAPTURE), dtype=jnp.int32)
    settle_mode = jnp.asarray(int(WalkingRefV2Mode.POST_TOUCHDOWN_SETTLE), dtype=jnp.int32)

    mode = jnp.asarray(mode_id, dtype=jnp.int32)
    route_progress_prev = (
        jnp.asarray(startup_route_progress_prev, dtype=jnp.float32)
        if startup_route_progress_prev is not None
        else jnp.asarray(1.0, dtype=jnp.float32)
    )
    route_ceiling_prev = (
        jnp.asarray(startup_route_ceiling_prev, dtype=jnp.float32)
        if startup_route_ceiling_prev is not None
        else jnp.asarray(1.0, dtype=jnp.float32)
    )
    route_stage_prev = (
        jnp.asarray(startup_route_stage_id_prev, dtype=jnp.int32)
        if startup_route_stage_id_prev is not None
        else jnp.asarray(int(StartupRouteStage.B), dtype=jnp.int32)
    )
    route_reason_prev = (
        jnp.asarray(startup_route_transition_reason_prev, dtype=jnp.int32)
        if startup_route_transition_reason_prev is not None
        else jnp.asarray(int(StartupRouteReason.NONE), dtype=jnp.int32)
    )
    phase_scale = jnp.asarray(config.support_phase_min_scale, dtype=jnp.float32) + (
        1.0 - jnp.asarray(config.support_phase_min_scale, dtype=jnp.float32)
    ) * progression_permission
    startup_height = jnp.asarray(config.walking_pelvis_height_m, dtype=jnp.float32) + jnp.asarray(
        config.startup_pelvis_height_offset_m, dtype=jnp.float32
    )
    support_height = jnp.asarray(config.walking_pelvis_height_m, dtype=jnp.float32) + jnp.asarray(
        config.support_pelvis_height_offset_m, dtype=jnp.float32
    )
    startup_pelvis_realization = _startup_pelvis_realization_jax(
        root_height_m=(
            jnp.asarray(root_height_m, dtype=jnp.float32)
            if root_height_m is not None
            else startup_height
        ),
        startup_height_m=startup_height,
        support_height_m=support_height,
    )
    knee_err = (
        jnp.asarray(stance_knee_tracking_error_rad, dtype=jnp.float32)
        if stance_knee_tracking_error_rad is not None
        else jnp.asarray(0.0, dtype=jnp.float32)
    )
    ankle_err = (
        jnp.asarray(stance_ankle_tracking_error_rad, dtype=jnp.float32)
        if stance_ankle_tracking_error_rad is not None
        else jnp.asarray(0.0, dtype=jnp.float32)
    )
    readiness_knee = _readiness_component_jax(
        jnp.abs(knee_err),
        good=config.startup_readiness_knee_err_good_rad,
        bad=config.startup_readiness_knee_err_bad_rad,
    )
    readiness_ankle = _readiness_component_jax(
        jnp.abs(ankle_err),
        good=config.startup_readiness_ankle_err_good_rad,
        bad=config.startup_readiness_ankle_err_bad_rad,
    )
    readiness_pitch = _readiness_component_jax(
        jnp.abs(jnp.asarray(root_pitch_rad, dtype=jnp.float32)),
        good=config.startup_readiness_pitch_good_rad,
        bad=config.startup_readiness_pitch_bad_rad,
    )
    readiness_pitch_rate = _readiness_component_jax(
        jnp.abs(jnp.asarray(root_pitch_rate_rad_s, dtype=jnp.float32)),
        good=config.startup_readiness_pitch_rate_good_rad_s,
        bad=config.startup_readiness_pitch_rate_bad_rad_s,
    )
    readiness_health = jnp.clip(
        (
            support_health
            - jnp.asarray(config.startup_readiness_min_health, dtype=jnp.float32)
        )
        / jnp.maximum(
            jnp.asarray(1.0 - config.startup_readiness_min_health, dtype=jnp.float32),
            jnp.asarray(1e-6, dtype=jnp.float32),
        ),
        0.0,
        1.0,
    ).astype(jnp.float32)
    startup_readiness = jnp.minimum(
        jnp.minimum(readiness_knee, readiness_ankle),
        jnp.minimum(readiness_pitch, jnp.minimum(readiness_pitch_rate, readiness_health)),
    ).astype(jnp.float32)
    startup_readiness_scale = (
        jnp.asarray(config.startup_progress_min_scale, dtype=jnp.float32)
        + (1.0 - jnp.asarray(config.startup_progress_min_scale, dtype=jnp.float32))
        * startup_readiness
    ).astype(jnp.float32)
    startup_alpha = jnp.clip(
        startup_pelvis_realization
        + jnp.asarray(config.startup_realization_lead_alpha, dtype=jnp.float32)
        * startup_readiness_scale,
        0.0,
        1.0,
    ).astype(jnp.float32)
    joint_max_err = jnp.maximum(jnp.abs(knee_err), jnp.abs(ankle_err)).astype(jnp.float32)
    pitch_abs = jnp.abs(jnp.asarray(root_pitch_rad, dtype=jnp.float32))
    pitch_rate_abs = jnp.abs(jnp.asarray(root_pitch_rate_rad_s, dtype=jnp.float32))

    # Fixed staged route: A -> W1 -> W2 -> W3 -> B.
    w1_alpha = jnp.asarray(config.startup_route_w1_alpha, dtype=jnp.float32)
    w2_alpha = jnp.asarray(config.startup_route_w2_alpha, dtype=jnp.float32)
    w3_alpha = jnp.asarray(config.startup_route_w3_alpha, dtype=jnp.float32)

    w2_joint_max = jnp.asarray(
        max(config.startup_readiness_knee_err_bad_rad, config.startup_readiness_ankle_err_bad_rad),
        dtype=jnp.float32,
    )
    w3_joint_max = jnp.asarray(
        max(
            0.5 * (config.startup_readiness_knee_err_good_rad + config.startup_readiness_knee_err_bad_rad),
            0.5 * (config.startup_readiness_ankle_err_good_rad + config.startup_readiness_ankle_err_bad_rad),
        ),
        dtype=jnp.float32,
    )
    b_joint_max = jnp.asarray(
        max(config.startup_readiness_knee_err_good_rad, config.startup_readiness_ankle_err_good_rad),
        dtype=jnp.float32,
    )
    w2_pelvis_ok = startup_pelvis_realization >= jnp.asarray(
        config.startup_route_w2_min_pelvis_realization, dtype=jnp.float32
    )
    w2_joints_ok = joint_max_err <= w2_joint_max
    w2_pitch_ok = pitch_abs <= jnp.asarray(
        config.startup_handoff_pitch_max_rad * config.startup_route_w2_pitch_relax,
        dtype=jnp.float32,
    )
    w2_pitch_rate_ok = pitch_rate_abs <= jnp.asarray(
        config.startup_handoff_pitch_rate_max_rad_s * config.startup_route_w2_pitch_rate_relax,
        dtype=jnp.float32,
    )
    w2_health_ok = support_health >= jnp.asarray(config.startup_readiness_min_health, dtype=jnp.float32)
    allow_w2 = w2_pelvis_ok & w2_joints_ok & w2_pitch_ok & w2_pitch_rate_ok & w2_health_ok

    w3_pelvis_ok = startup_pelvis_realization >= jnp.asarray(
        config.startup_route_w3_min_pelvis_realization, dtype=jnp.float32
    )
    w3_joints_ok = joint_max_err <= w3_joint_max
    w3_pitch_ok = pitch_abs <= jnp.asarray(config.startup_handoff_pitch_max_rad, dtype=jnp.float32)
    w3_pitch_rate_ok = pitch_rate_abs <= jnp.asarray(
        config.startup_handoff_pitch_rate_max_rad_s, dtype=jnp.float32
    )
    w3_health_ok = support_health >= jnp.asarray(
        max(config.startup_readiness_min_health, config.startup_support_open_health),
        dtype=jnp.float32,
    )
    allow_w3 = allow_w2 & w3_pelvis_ok & w3_joints_ok & w3_pitch_ok & w3_pitch_rate_ok & w3_health_ok

    b_pelvis_ok = startup_pelvis_realization >= jnp.asarray(
        config.startup_handoff_min_pelvis_realization, dtype=jnp.float32
    )
    b_joints_ok = joint_max_err <= b_joint_max
    b_pitch_ok = pitch_abs <= jnp.asarray(config.startup_handoff_pitch_max_rad, dtype=jnp.float32)
    b_pitch_rate_ok = pitch_rate_abs <= jnp.asarray(
        config.startup_handoff_pitch_rate_max_rad_s, dtype=jnp.float32
    )
    b_health_ok = support_health >= jnp.asarray(config.startup_support_open_health, dtype=jnp.float32)
    allow_b = allow_w3 & b_pelvis_ok & b_joints_ok & b_pitch_ok & b_pitch_rate_ok & b_health_ok

    route_ceiling = jnp.where(
        allow_b,
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.where(allow_w3, w3_alpha, jnp.where(allow_w2, w2_alpha, w1_alpha)),
    )
    # Ratchet: during startup the ceiling can only advance, never drop back.
    # This prevents transient pitch spikes from resetting progress.
    route_ceiling = jnp.where(
        mode == startup_mode,
        jnp.maximum(route_ceiling, route_ceiling_prev),
        route_ceiling,
    )
    route_progress = jnp.clip(
        jnp.minimum(startup_alpha, route_ceiling), 0.0, 1.0
    ).astype(jnp.float32)
    # Ratchet progress too: once a stage is reached, don't regress.
    route_progress = jnp.where(
        mode == startup_mode,
        jnp.maximum(route_progress, route_progress_prev),
        route_progress,
    )
    route_stage_id = _stage_id_from_progress_jax(
        route_progress,
        w1_alpha=config.startup_route_w1_alpha,
        w2_alpha=config.startup_route_w2_alpha,
        w3_alpha=config.startup_route_w3_alpha,
    )
    reason_w2 = _first_failed_reason_jax(
        pelvis_ok=w2_pelvis_ok,
        joints_ok=w2_joints_ok,
        pitch_ok=w2_pitch_ok,
        pitch_rate_ok=w2_pitch_rate_ok,
        health_ok=w2_health_ok,
    )
    reason_w3 = _first_failed_reason_jax(
        pelvis_ok=w3_pelvis_ok,
        joints_ok=w3_joints_ok,
        pitch_ok=w3_pitch_ok,
        pitch_rate_ok=w3_pitch_rate_ok,
        health_ok=w3_health_ok,
    )
    reason_b = _first_failed_reason_jax(
        pelvis_ok=b_pelvis_ok,
        joints_ok=b_joints_ok,
        pitch_ok=b_pitch_ok,
        pitch_rate_ok=b_pitch_rate_ok,
        health_ok=b_health_ok,
    )
    blocked_reason = jnp.where(
        allow_w3,
        reason_b,
        jnp.where(allow_w2, reason_w3, reason_w2),
    )
    route_blocked = startup_alpha > (route_ceiling + jnp.asarray(1e-6, dtype=jnp.float32))
    route_reason = jnp.where(
        allow_b,
        jnp.asarray(int(StartupRouteReason.ROUTE_COMPLETE), dtype=jnp.int32),
        jnp.where(
            route_blocked,
            blocked_reason,
            jnp.asarray(int(StartupRouteReason.NONE), dtype=jnp.int32),
        ),
    )
    route_swing_x_scale = _piecewise_route_scale_jax(
        route_progress,
        w1_alpha=config.startup_route_w1_alpha,
        w2_alpha=config.startup_route_w2_alpha,
        w3_alpha=config.startup_route_w3_alpha,
        w1_scale=config.startup_route_w1_scale,
        w2_scale=config.startup_route_w2_scale,
        w3_scale=config.startup_route_w3_scale,
    )
    route_support_y_scale = _piecewise_route_scale_jax(
        route_progress,
        w1_alpha=config.startup_route_w1_alpha,
        w2_alpha=config.startup_route_w2_alpha,
        w3_alpha=config.startup_route_w3_alpha,
        w1_scale=config.startup_route_w1_support_y_scale,
        w2_scale=config.startup_route_w2_support_y_scale,
        w3_scale=config.startup_route_w3_support_y_scale,
    )
    route_pelvis_roll_scale = _piecewise_route_scale_jax(
        route_progress,
        w1_alpha=config.startup_route_w1_alpha,
        w2_alpha=config.startup_route_w2_alpha,
        w3_alpha=config.startup_route_w3_alpha,
        w1_scale=config.startup_route_w1_pelvis_roll_scale,
        w2_scale=config.startup_route_w2_pelvis_roll_scale,
        w3_scale=config.startup_route_w3_pelvis_roll_scale,
    )
    route_pelvis_pitch_scale = _piecewise_route_scale_jax(
        route_progress,
        w1_alpha=config.startup_route_w1_alpha,
        w2_alpha=config.startup_route_w2_alpha,
        w3_alpha=config.startup_route_w3_alpha,
        w1_scale=config.startup_route_w1_pelvis_pitch_scale,
        w2_scale=config.startup_route_w2_pelvis_pitch_scale,
        w3_scale=config.startup_route_w3_pelvis_pitch_scale,
    )
    route_pelvis_height_scale = _piecewise_route_scale_jax(
        route_progress,
        w1_alpha=config.startup_route_w1_alpha,
        w2_alpha=config.startup_route_w2_alpha,
        w3_alpha=config.startup_route_w3_alpha,
        w1_scale=config.startup_route_w1_pelvis_height_scale,
        w2_scale=config.startup_route_w2_pelvis_height_scale,
        w3_scale=config.startup_route_w3_pelvis_height_scale,
    )

    phase_time = jnp.where(
        mode == startup_mode,
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(phase_time_s, dtype=jnp.float32)
        + jnp.asarray(dt_s, dtype=jnp.float32) * phase_scale,
    )
    stance = jnp.asarray(stance_foot_id, dtype=jnp.int32)
    switches = jnp.asarray(stance_switch_count, dtype=jnp.int32)
    mode_time = (
        jnp.asarray(mode_time_s, dtype=jnp.float32) + jnp.asarray(dt_s, dtype=jnp.float32)
    ).astype(jnp.float32)
    step_time = jnp.asarray(config.step_time_s, dtype=jnp.float32)
    phase = jnp.where(
        mode == startup_mode,
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.clip(phase_time / jnp.maximum(step_time, 1e-6), 0.0, 1.0),
    ).astype(jnp.float32)
    stance_is_left = stance == jnp.asarray(LEFT_STANCE, dtype=jnp.int32)
    swing_loaded = jnp.where(
        stance_is_left,
        jnp.asarray(right_foot_loaded, dtype=jnp.bool_),
        jnp.asarray(left_foot_loaded, dtype=jnp.bool_),
    )

    startup_stable = (
        (support_health >= jnp.asarray(config.startup_support_open_health, dtype=jnp.float32))
        & (
            jnp.abs(jnp.asarray(root_pitch_rad, dtype=jnp.float32))
            <= jnp.asarray(config.startup_handoff_pitch_max_rad, dtype=jnp.float32)
        )
        & (
            jnp.abs(jnp.asarray(root_pitch_rate_rad_s, dtype=jnp.float32))
            <= jnp.asarray(config.startup_handoff_pitch_rate_max_rad_s, dtype=jnp.float32)
        )
    )
    startup_ready_for_handoff = (
        startup_readiness >= jnp.asarray(config.startup_handoff_min_readiness, dtype=jnp.float32)
    ) & (
        startup_pelvis_realization
        >= jnp.asarray(config.startup_handoff_min_pelvis_realization, dtype=jnp.float32)
    ) & (
        route_progress >= jnp.asarray(config.startup_handoff_min_alpha, dtype=jnp.float32)
    )
    startup_route_complete = route_stage_id == jnp.asarray(int(StartupRouteStage.B), dtype=jnp.int32)
    startup_timeout_reached = mode_time >= jnp.asarray(
        config.startup_handoff_timeout_s, dtype=jnp.float32
    )

    from_startup = mode == startup_mode
    from_support = mode == support_mode
    from_swing = mode == swing_mode
    from_capture = mode == capture_mode
    from_settle = mode == settle_mode

    to_support_from_startup = from_startup & (
        (startup_stable & startup_ready_for_handoff & startup_route_complete)
        | startup_timeout_reached  # unconditional safety valve
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

    switched = (mode != startup_mode) & (phase_time >= step_time)
    phase_time = jnp.where(switched, 0.0, phase_time)
    stance = jnp.where(switched, 1 - stance, stance)
    switches = jnp.where(switched, switches + 1, switches)
    mode = jnp.where(switched, settle_mode, mode)
    mode_time = jnp.where(switched, 0.0, mode_time)
    phase = jnp.where(switched, 0.0, phase)

    if config.debug_force_support_only:
        # Keep mode_time progressing so support-entry shaping stays bounded in debug runs.
        mode = support_mode
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
        / jnp.maximum(jnp.asarray(config.walking_pelvis_height_m, dtype=jnp.float32), 1e-6)
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
        + route_swing_x_scale
        * (x_foot - jnp.asarray(config.min_step_length_m, dtype=jnp.float32)),
        x_foot,
    )
    x_foot = jnp.clip(
        x_foot,
        jnp.asarray(config.min_step_length_m, dtype=jnp.float32),
        jnp.asarray(config.max_step_length_m, dtype=jnp.float32),
    )

    # ── DCM COM trajectory ─────────────────────────────────────────────
    # Two modes:
    #   "linear" — phase-proportional ramp (v0.19.4-B baseline).
    #   "lipm"   — full LIPM cosh/sinh using stance-start COM state
    #              (v0.19.4-C propulsion upgrade).
    _com_x0 = (
        jnp.asarray(com_x0_at_stance_start, dtype=jnp.float32)
        if com_x0_at_stance_start is not None
        else jnp.asarray(0.0, dtype=jnp.float32)
    )
    _com_vx0 = (
        jnp.asarray(com_vx0_at_stance_start, dtype=jnp.float32)
        if com_vx0_at_stance_start is not None
        else jnp.asarray(0.0, dtype=jnp.float32)
    )
    _com_drive_linear = phase * speed * step_time
    _t_stance = phase * step_time
    _com_drive_lipm = (
        _com_x0 * jnp.cosh(omega * _t_stance)
        + (_com_vx0 / jnp.maximum(omega, jnp.asarray(1e-6, dtype=jnp.float32)))
        * jnp.sinh(omega * _t_stance)
    )
    _use_lipm = jnp.asarray(config.com_trajectory_mode == "lipm", dtype=jnp.bool_)
    _com_drive = jnp.where(_use_lipm, _com_drive_lipm, _com_drive_linear)
    com_x_planned = jnp.where(
        jnp.asarray(config.com_trajectory_enabled, dtype=jnp.bool_),
        jnp.clip(
            _com_drive,
            -jnp.asarray(config.com_trajectory_max_behind_m, dtype=jnp.float32),
            jnp.asarray(config.com_trajectory_max_ahead_m, dtype=jnp.float32),
        ),
        jnp.asarray(0.0, dtype=jnp.float32),
    )
    # Suppress COM drive during startup/support (double support).
    com_x_planned = jnp.where(
        (mode == startup_mode) | (mode == support_mode),
        jnp.asarray(0.0, dtype=jnp.float32),
        com_x_planned,
    )
    # Scale COM drive by progression_permission so forward torso drive
    # is reduced when stepping is suppressed.  Prevents pitch-without-step.
    com_x_planned = com_x_planned * progression_permission
    # State tracking: record COM state at each stance transition.
    _stance_switched = switches > jnp.asarray(stance_switch_count, dtype=jnp.int32)
    com_x0_out = jnp.where(
        _stance_switched,
        jnp.asarray(com_position_stance_frame[0], dtype=jnp.float32),
        _com_x0,
    )
    com_vx0_out = jnp.where(
        _stance_switched,
        jnp.asarray(com_velocity_stance_frame[0], dtype=jnp.float32),
        _com_vx0,
    )

    # ── Ankle push-off ────────────────────────────────────────────────
    # During terminal stance, add explicit plantarflexion on top of IK.
    # This is the primary push-off mechanism in human gait.
    _pushoff_phase_start = jnp.asarray(config.ankle_pushoff_phase_start, dtype=jnp.float32)
    _pushoff_max = jnp.asarray(config.ankle_pushoff_max_rad, dtype=jnp.float32)
    _pushoff_progress = jnp.clip(
        (phase - _pushoff_phase_start)
        / jnp.maximum(1.0 - _pushoff_phase_start, jnp.asarray(1e-6, dtype=jnp.float32)),
        0.0,
        1.0,
    )
    # Ramp up linearly during terminal stance, scale by speed.
    _speed_scale = jnp.clip(
        speed / jnp.maximum(jnp.asarray(config.max_forward_speed_mps, dtype=jnp.float32), jnp.asarray(1e-6, dtype=jnp.float32)),
        0.0,
        1.0,
    )
    ankle_pushoff = jnp.where(
        jnp.asarray(config.ankle_pushoff_enabled, dtype=jnp.bool_)
        & (mode == swing_mode),
        _pushoff_progress * _pushoff_max * _speed_scale,
        jnp.asarray(0.0, dtype=jnp.float32),
    )

    y_sign = jnp.where(stance == jnp.asarray(LEFT_STANCE, dtype=jnp.int32), -1.0, 1.0)
    y_foot = jnp.clip(
        y_sign * jnp.asarray(config.nominal_lateral_foot_offset_m, dtype=jnp.float32),
        -jnp.asarray(config.max_lateral_step_m, dtype=jnp.float32),
        jnp.asarray(config.max_lateral_step_m, dtype=jnp.float32),
    )
    y_foot = jnp.where(
        mode == startup_mode,
        route_support_y_scale * y_foot,
        y_foot,
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
    pelvis_roll = jnp.where(
        mode == startup_mode,
        route_pelvis_roll_scale * pelvis_roll_target,
        pelvis_roll_target,
    )
    pelvis_pitch = jnp.where(
        mode == startup_mode,
        route_pelvis_pitch_scale * pelvis_pitch_target,
        pelvis_pitch_target,
    )
    
    pelvis_height_startup = jnp.asarray(config.walking_pelvis_height_m, dtype=jnp.float32) + jnp.asarray(
        config.startup_pelvis_height_offset_m, dtype=jnp.float32
    )
    pelvis_height_support = jnp.asarray(config.walking_pelvis_height_m, dtype=jnp.float32) + jnp.asarray(
        config.support_pelvis_height_offset_m, dtype=jnp.float32
    )
    pelvis_height = jnp.where(
        mode == startup_mode,
        pelvis_height_startup
        + route_pelvis_height_scale * (pelvis_height_support - pelvis_height_startup),
        jnp.where(
            mode == support_mode,
            pelvis_height_support,
            jnp.asarray(config.walking_pelvis_height_m, dtype=jnp.float32),
        ),
    )
    startup_route_transition_reason = jnp.where(
        from_startup & to_support_from_startup & jnp.logical_not(startup_route_complete),
        jnp.asarray(int(StartupRouteReason.TIMEOUT_FALLBACK), dtype=jnp.int32),
        route_reason,
    )
    route_state_update = (mode == startup_mode) | (from_startup & to_support_from_startup)
    startup_route_stage_id_out = jnp.where(
        route_state_update,
        route_stage_id,
        route_stage_prev,
    ).astype(jnp.int32)
    startup_route_progress_out = jnp.where(
        route_state_update,
        route_progress,
        route_progress_prev,
    ).astype(jnp.float32)
    startup_route_ceiling_out = jnp.where(
        route_state_update,
        route_ceiling,
        route_ceiling_prev,
    ).astype(jnp.float32)
    startup_route_transition_reason_out = jnp.where(
        route_state_update,
        startup_route_transition_reason,
        route_reason_prev,
    ).astype(jnp.int32)

    ref = {
        "gait_phase_sin": jnp.sin(2.0 * jnp.asarray(pi, dtype=jnp.float32) * phase).astype(jnp.float32),
        "gait_phase_cos": jnp.cos(2.0 * jnp.asarray(pi, dtype=jnp.float32) * phase).astype(jnp.float32),
        "phase_progress": phase.astype(jnp.float32),
        "stance_foot_id": stance.astype(jnp.int32),
        "next_foothold": jnp.asarray([x_foot, y_foot], dtype=jnp.float32),
        "next_foothold_raw": jnp.asarray([x_foot_nom, y_foot], dtype=jnp.float32),
        "nominal_step_length": nominal_step.astype(jnp.float32),
        "swing_pos": jnp.asarray([swing_x, swing_y, swing_z], dtype=jnp.float32),
        "swing_vel": swing_vel,
        "pelvis_height": pelvis_height.astype(jnp.float32),
        "pelvis_roll": pelvis_roll.astype(jnp.float32),
        "pelvis_pitch": pelvis_pitch.astype(jnp.float32),
        "startup_route_progress": startup_route_progress_out,
        "startup_route_ceiling": startup_route_ceiling_out,
        "startup_route_stage_id": startup_route_stage_id_out,
        "startup_route_transition_reason": startup_route_transition_reason_out,
        "com_x_planned": com_x_planned.astype(jnp.float32),
        "com_x0_at_stance_start": com_x0_out.astype(jnp.float32),
        "com_vx0_at_stance_start": com_vx0_out.astype(jnp.float32),
        "ankle_pushoff_rad": ankle_pushoff.astype(jnp.float32),
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
    root_height_m: float | None = None,
    stance_knee_tracking_error_rad: float = 0.0,
    stance_ankle_tracking_error_rad: float = 0.0,
) -> WalkingReferenceV2Output:
    """Advance support-first hybrid reference state and emit locomotion reference."""
    if dt_s <= 0.0:
        raise ValueError("dt_s must be > 0")

    support_health, support_instability = compute_support_health_v2(config=config, inputs=inputs)
    # During startup and support-stabilize, double-support is expected —
    # remove the contact penalty so progression_permission is not blocked.
    if state.mode_id in (int(WalkingRefV2Mode.STARTUP_SUPPORT_RAMP), int(WalkingRefV2Mode.SUPPORT_STABILIZE)):
        is_double = not (bool(inputs.left_foot_loaded) ^ bool(inputs.right_foot_loaded))
        if is_double:
            adj_instab = max(support_instability - 0.25, 0.0)
            support_instability = adj_instab
            support_health = _clip(1.0 / (1.0 + config.support_health_gain * adj_instab), 0.0, 1.0)
    progression_permission = _permission_from_health(config, support_health)
    phase_scale = config.support_phase_min_scale + (1.0 - config.support_phase_min_scale) * progression_permission
    readiness_knee = _readiness_component(
        abs(float(stance_knee_tracking_error_rad)),
        config.startup_readiness_knee_err_good_rad,
        config.startup_readiness_knee_err_bad_rad,
    )
    readiness_ankle = _readiness_component(
        abs(float(stance_ankle_tracking_error_rad)),
        config.startup_readiness_ankle_err_good_rad,
        config.startup_readiness_ankle_err_bad_rad,
    )
    readiness_pitch = _readiness_component(
        abs(float(inputs.root_pitch_rad)),
        config.startup_readiness_pitch_good_rad,
        config.startup_readiness_pitch_bad_rad,
    )
    readiness_pitch_rate = _readiness_component(
        abs(float(inputs.root_pitch_rate_rad_s)),
        config.startup_readiness_pitch_rate_good_rad_s,
        config.startup_readiness_pitch_rate_bad_rad_s,
    )
    readiness_health = _clip(
        (support_health - config.startup_readiness_min_health)
        / max(1.0 - config.startup_readiness_min_health, 1e-6),
        0.0,
        1.0,
    )
    startup_readiness = min(
        readiness_knee,
        readiness_ankle,
        readiness_pitch,
        readiness_pitch_rate,
        readiness_health,
    )
    startup_readiness_scale = config.startup_progress_min_scale + (
        1.0 - config.startup_progress_min_scale
    ) * startup_readiness
    startup_height = config.walking_pelvis_height_m + config.startup_pelvis_height_offset_m
    support_height = config.walking_pelvis_height_m + config.support_pelvis_height_offset_m
    pelvis_height_now = startup_height if root_height_m is None else float(root_height_m)
    startup_pelvis_realization = _startup_pelvis_realization(
        root_height_m=pelvis_height_now,
        startup_height_m=startup_height,
        support_height_m=support_height,
    )
    startup_alpha = _clip(
        startup_pelvis_realization
        + config.startup_realization_lead_alpha * startup_readiness_scale,
        0.0,
        1.0,
    )

    mode = WalkingRefV2Mode(state.mode_id)
    phase_time = 0.0 if mode == WalkingRefV2Mode.STARTUP_SUPPORT_RAMP else state.phase_time_s + dt_s * phase_scale
    stance = state.stance_foot_id
    switch_count = state.stance_switch_count
    mode_time = state.mode_time_s + dt_s

    phase = _clip(phase_time / max(config.step_time_s, 1e-6), 0.0, 1.0)
    stance_is_left = stance == LEFT_STANCE
    swing_loaded = bool(inputs.right_foot_loaded if stance_is_left else inputs.left_foot_loaded)

    startup_stable = (
        support_health >= config.startup_support_open_health
        and abs(float(inputs.root_pitch_rad)) <= config.startup_handoff_pitch_max_rad
        and abs(float(inputs.root_pitch_rate_rad_s))
        <= config.startup_handoff_pitch_rate_max_rad_s
    )
    startup_ready_for_handoff = (
        startup_readiness >= config.startup_handoff_min_readiness
        and startup_pelvis_realization >= config.startup_handoff_min_pelvis_realization
        and startup_alpha >= config.startup_handoff_min_alpha
    )
    startup_timeout_reached = mode_time >= config.startup_handoff_timeout_s

    if mode == WalkingRefV2Mode.STARTUP_SUPPORT_RAMP:
        if (startup_stable and startup_ready_for_handoff) or startup_timeout_reached:
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
    omega = (9.81 / max(config.walking_pelvis_height_m, 1e-6)) ** 0.5
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
    pelvis_height_startup = config.walking_pelvis_height_m + config.startup_pelvis_height_offset_m
    pelvis_height_support = config.walking_pelvis_height_m + config.support_pelvis_height_offset_m
    
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
        pelvis_height = config.walking_pelvis_height_m

    # DCM COM trajectory (matches JAX implementation)
    from math import cosh, sinh, exp as math_exp
    if config.com_trajectory_enabled:
        if config.com_trajectory_mode == "lipm":
            t_stance = phase * config.step_time_s
            com_drive = (
                float(inputs.com_position_stance_frame[0]) * cosh(omega * t_stance)
                + float(inputs.com_velocity_stance_frame[0]) / max(omega, 1e-6)
                * sinh(omega * t_stance)
            )
        else:
            com_drive = phase * speed * config.step_time_s
        com_x_planned_val = _clip(
            com_drive,
            -config.com_trajectory_max_behind_m,
            config.com_trajectory_max_ahead_m,
        )
    else:
        com_x_planned_val = 0.0
    # Suppress during startup/support (double support)
    if mode in (WalkingRefV2Mode.STARTUP_SUPPORT_RAMP, WalkingRefV2Mode.SUPPORT_STABILIZE):
        com_x_planned_val = 0.0
    # Scale COM drive by progression_permission so forward torso drive
    # is reduced when stepping is suppressed.
    com_x_planned_val *= progression_permission

    # Ankle push-off: explicit plantarflexion during terminal stance.
    ankle_pushoff_val = 0.0
    if config.ankle_pushoff_enabled and mode == WalkingRefV2Mode.SWING_RELEASE:
        pushoff_progress = _clip(
            (phase - config.ankle_pushoff_phase_start)
            / max(1.0 - config.ankle_pushoff_phase_start, 1e-6),
            0.0,
            1.0,
        )
        speed_scale = _clip(speed / max(config.max_forward_speed_mps, 1e-6), 0.0, 1.0)
        ankle_pushoff_val = pushoff_progress * config.ankle_pushoff_max_rad * speed_scale

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
        com_x_planned=com_x_planned_val,
        ankle_pushoff_rad=ankle_pushoff_val,
    )
