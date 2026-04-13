"""Forward-only walking reference generator v1 (v0.19.2)."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin
from typing import Tuple

from .locomotion_contract import LocomotionReferenceState
from ..reduced_model.lipm import LipmConfig, capture_point, natural_frequency
from ..reduced_model.dcm import DcmStepConfig, compute_next_foothold_1d


Vec2 = Tuple[float, float]
Vec3 = Tuple[float, float, float]

LEFT_STANCE = 0
RIGHT_STANCE = 1


@dataclass(frozen=True)
class WalkingRefV1Config:
    step_time_s: float = 0.36
    walking_pelvis_height_m: float = 0.40
    nominal_lateral_foot_offset_m: float = 0.09
    min_step_length_m: float = 0.02
    max_step_length_m: float = 0.14
    max_lateral_step_m: float = 0.14
    swing_height_m: float = 0.04
    pelvis_roll_bias_rad: float = 0.03
    pelvis_pitch_gain: float = 0.08
    max_pelvis_pitch_rad: float = 0.08
    max_forward_speed_mps: float = 0.35
    dcm_placement_gain: float = 1.0

    def __post_init__(self) -> None:
        if self.step_time_s <= 0.0:
            raise ValueError("step_time_s must be > 0")
        if self.walking_pelvis_height_m <= 0.0:
            raise ValueError("walking_pelvis_height_m must be > 0")
        if self.swing_height_m < 0.0:
            raise ValueError("swing_height_m must be >= 0")
        if self.min_step_length_m < 0.0 or self.max_step_length_m <= self.min_step_length_m:
            raise ValueError("step length bounds are invalid")
        if self.max_lateral_step_m <= 0.0:
            raise ValueError("max_lateral_step_m must be > 0")
        if self.max_forward_speed_mps <= 0.0:
            raise ValueError("max_forward_speed_mps must be > 0")
        if self.dcm_placement_gain < 0.0 or self.dcm_placement_gain > 1.0:
            raise ValueError("dcm_placement_gain must be in [0, 1]")


@dataclass(frozen=True)
class WalkingRefV1State:
    phase_time_s: float = 0.0
    stance_foot_id: int = LEFT_STANCE
    stance_switch_count: int = 0

    def __post_init__(self) -> None:
        if self.stance_foot_id not in (LEFT_STANCE, RIGHT_STANCE):
            raise ValueError("stance_foot_id must be 0(left) or 1(right)")
        if self.phase_time_s < 0.0:
            raise ValueError("phase_time_s must be >= 0")


@dataclass(frozen=True)
class WalkingRefV1Input:
    forward_speed_mps: float
    com_position_stance_frame: Vec2 = (0.0, 0.0)
    com_velocity_stance_frame: Vec2 = (0.0, 0.0)


@dataclass(frozen=True)
class WalkingReferenceOutput:
    reference: LocomotionReferenceState
    next_state: WalkingRefV1State


def _clip(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, value)))


def _swing_trajectory(
    *,
    phase: float,
    step_length_m: float,
    lateral_target_m: float,
    config: WalkingRefV1Config,
) -> tuple[Vec3, Vec3]:
    p = _clip(phase, 0.0, 1.0)
    x = step_length_m * p
    y = lateral_target_m * p
    z = config.swing_height_m * sin(pi * p)
    xdot = step_length_m / config.step_time_s
    ydot = lateral_target_m / config.step_time_s
    zdot = (config.swing_height_m * pi / config.step_time_s) * cos(pi * p)
    return (x, y, z), (xdot, ydot, zdot)


def step_reference(
    *,
    config: WalkingRefV1Config,
    state: WalkingRefV1State,
    inputs: WalkingRefV1Input,
    dt_s: float,
) -> WalkingReferenceOutput:
    """Advance reference state by one control step and emit locomotion reference."""
    if dt_s <= 0.0:
        raise ValueError("dt_s must be > 0")

    speed = _clip(inputs.forward_speed_mps, 0.0, config.max_forward_speed_mps)
    phase_time = state.phase_time_s + float(dt_s)
    stance = state.stance_foot_id
    switch_count = state.stance_switch_count
    if phase_time >= config.step_time_s:
        phase_time = phase_time - config.step_time_s
        stance = RIGHT_STANCE if stance == LEFT_STANCE else LEFT_STANCE
        switch_count += 1
    phase = _clip(phase_time / config.step_time_s, 0.0, 1.0)

    lipm_cfg = LipmConfig(com_height_m=config.walking_pelvis_height_m)
    omega = natural_frequency(lipm_cfg)
    cp = capture_point(
        com_position_stance_frame=inputs.com_position_stance_frame,
        com_velocity_stance_frame=inputs.com_velocity_stance_frame,
        config=lipm_cfg,
    )

    nominal_step = _clip(
        speed * config.step_time_s,
        config.min_step_length_m,
        config.max_step_length_m,
    )
    dcm_cfg = DcmStepConfig(step_time_s=config.step_time_s, omega=omega)
    x_foot_dcm = compute_next_foothold_1d(
        dcm_now=cp[0],
        dcm_target_next=nominal_step,
        config=dcm_cfg,
    )
    x_foot = (1.0 - config.dcm_placement_gain) * nominal_step + config.dcm_placement_gain * x_foot_dcm
    x_foot = _clip(x_foot, config.min_step_length_m, config.max_step_length_m)

    nominal_lateral = config.nominal_lateral_foot_offset_m
    y_sign = -1.0 if stance == LEFT_STANCE else 1.0  # swing foot moves to opposite side
    y_foot = _clip(y_sign * nominal_lateral, -config.max_lateral_step_m, config.max_lateral_step_m)

    swing_pos, swing_vel = _swing_trajectory(
        phase=phase,
        step_length_m=x_foot,
        lateral_target_m=y_foot,
        config=config,
    )

    pelvis_roll = -config.pelvis_roll_bias_rad if stance == LEFT_STANCE else config.pelvis_roll_bias_rad
    pelvis_pitch = _clip(config.pelvis_pitch_gain * speed, -config.max_pelvis_pitch_rad, config.max_pelvis_pitch_rad)

    ref = LocomotionReferenceState(
        gait_phase_sin=float(sin(2.0 * pi * phase)),
        gait_phase_cos=float(cos(2.0 * pi * phase)),
        stance_foot_id=stance,
        desired_next_foothold_stance_frame=(x_foot, y_foot),
        desired_swing_foot_position=swing_pos,
        desired_swing_foot_velocity=swing_vel,
        desired_pelvis_height_m=config.walking_pelvis_height_m,
        desired_pelvis_roll_rad=pelvis_roll,
        desired_pelvis_pitch_rad=pelvis_pitch,
    )
    next_state = WalkingRefV1State(
        phase_time_s=phase_time,
        stance_foot_id=stance,
        stance_switch_count=switch_count,
    )
    return WalkingReferenceOutput(reference=ref, next_state=next_state)
