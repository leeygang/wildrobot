"""Small observable-state planner for reactive standing recovery."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np


HOLD = 0
SWING = 1
SETTLE = 2


@dataclass(frozen=True)
class StandingRecoveryPlannerConfig:
    enabled: bool = False
    trigger_angle_rad: float = math.radians(5.0)
    lookahead_s: float = 0.25
    com_height_m: float = 0.46
    capture_gain: float = 1.0
    max_step_m: float = 0.10
    swing_duration_steps: int = 20
    settle_min_steps: int = 25
    settle_max_steps: int = 75
    settle_angle_rad: float = math.radians(3.0)
    settle_rate_rad_s: float = 0.10

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, Any] | None
    ) -> "StandingRecoveryPlannerConfig":
        values = dict(data or {})
        return cls(
            enabled=bool(values.get("enabled", False)),
            trigger_angle_rad=float(values.get("trigger_angle_rad", math.radians(5.0))),
            lookahead_s=float(values.get("lookahead_s", 0.25)),
            com_height_m=float(values.get("com_height_m", 0.46)),
            capture_gain=float(values.get("capture_gain", 1.0)),
            max_step_m=float(values.get("max_step_m", 0.10)),
            swing_duration_steps=int(values.get("swing_duration_steps", 20)),
            settle_min_steps=int(values.get("settle_min_steps", 25)),
            settle_max_steps=int(values.get("settle_max_steps", 75)),
            settle_angle_rad=float(values.get("settle_angle_rad", math.radians(3.0))),
            settle_rate_rad_s=float(values.get("settle_rate_rad_s", 0.10)),
        )


@dataclass(frozen=True)
class StandingRecoveryPlannerState:
    phase: int = HOLD
    swing_foot: int = -1
    phase_step: int = 0
    settle_count: int = 0
    target_xy_m: tuple[float, float] = (0.0, 0.0)
    step_count: int = 0


def roll_pitch_from_quat_wxyz(quat_wxyz: np.ndarray) -> tuple[float, float]:
    w, x, y, z = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(float(np.clip(2.0 * (w * y - z * x), -1.0, 1.0)))
    return roll, pitch


def encode_recovery_command(
    state: StandingRecoveryPlannerState,
    cfg: StandingRecoveryPlannerConfig,
) -> np.ndarray:
    if not cfg.enabled or state.phase == HOLD:
        return np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    swing = np.zeros(2, dtype=np.float32)
    if state.swing_foot in (0, 1):
        swing[state.swing_foot] = 1.0
    if state.phase == SWING:
        progress = min(1.0, state.phase_step / max(1, cfg.swing_duration_steps))
        phase_angle = math.pi * progress
    else:
        phase_angle = math.pi
    scale = max(cfg.max_step_m, 1e-6)
    target = np.clip(np.asarray(state.target_xy_m) / scale, -1.0, 1.0)
    return np.asarray(
        [1.0, swing[0], swing[1], math.sin(phase_angle),
         math.cos(phase_angle), target[0], target[1]],
        dtype=np.float32,
    )


def advance_recovery_planner(
    state: StandingRecoveryPlannerState,
    cfg: StandingRecoveryPlannerConfig,
    *,
    roll_rad: float,
    pitch_rad: float,
    roll_rate_rad_s: float,
    pitch_rate_rad_s: float,
    left_foot_x_m: float,
    right_foot_x_m: float,
    left_loaded: bool,
    right_loaded: bool,
) -> StandingRecoveryPlannerState:
    if not cfg.enabled:
        return StandingRecoveryPlannerState()

    predicted_roll = roll_rad + cfg.lookahead_s * roll_rate_rad_s
    predicted_pitch = pitch_rad + cfg.lookahead_s * pitch_rate_rad_s
    severity = max(abs(predicted_roll), abs(predicted_pitch))
    stable = (
        abs(roll_rad) < cfg.settle_angle_rad
        and abs(pitch_rad) < cfg.settle_angle_rad
        and abs(roll_rate_rad_s) < cfg.settle_rate_rad_s
        and abs(pitch_rate_rad_s) < cfg.settle_rate_rad_s
        and left_loaded
        and right_loaded
    )

    start_step = state.phase == HOLD and severity >= cfg.trigger_angle_rad
    retry_step = (
        state.phase == SETTLE
        and state.phase_step >= cfg.settle_max_steps
        and not stable
    )
    if start_step or retry_step:
        omega0 = math.sqrt(9.81 / max(cfg.com_height_m, 1e-6))
        dx = cfg.capture_gain * cfg.com_height_m * (
            math.tan(pitch_rad) + pitch_rate_rad_s / omega0
        )
        dy = cfg.capture_gain * cfg.com_height_m * (
            math.tan(roll_rad) + roll_rate_rad_s / omega0
        )
        dx = float(np.clip(dx, -cfg.max_step_m, cfg.max_step_m))
        dy = float(np.clip(dy, -cfg.max_step_m, cfg.max_step_m))
        if abs(dy) > abs(dx):
            swing_foot = 0 if dy >= 0.0 else 1
        elif dx >= 0.0:
            swing_foot = 0 if left_foot_x_m <= right_foot_x_m else 1
        else:
            swing_foot = 0 if left_foot_x_m >= right_foot_x_m else 1
        return StandingRecoveryPlannerState(
            phase=SWING,
            swing_foot=swing_foot,
            target_xy_m=(dx, dy),
            step_count=state.step_count + 1,
        )

    if state.phase == SWING:
        next_step = state.phase_step + 1
        if next_step >= cfg.swing_duration_steps:
            return StandingRecoveryPlannerState(
                phase=SETTLE,
                swing_foot=state.swing_foot,
                target_xy_m=state.target_xy_m,
                step_count=state.step_count,
            )
        return StandingRecoveryPlannerState(
            phase=SWING,
            swing_foot=state.swing_foot,
            phase_step=next_step,
            target_xy_m=state.target_xy_m,
            step_count=state.step_count,
        )

    if state.phase == SETTLE:
        settle_count = state.settle_count + 1 if stable else 0
        if settle_count >= cfg.settle_min_steps:
            return StandingRecoveryPlannerState(step_count=state.step_count)
        return StandingRecoveryPlannerState(
            phase=SETTLE,
            swing_foot=state.swing_foot,
            phase_step=state.phase_step + 1,
            settle_count=settle_count,
            target_xy_m=state.target_xy_m,
            step_count=state.step_count,
        )

    return state
