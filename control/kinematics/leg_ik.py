"""Simple sagittal leg IK for nominal stepping targets (v0.19.2)."""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, atan2, cos, pi, sin, sqrt


@dataclass(frozen=True)
class LegIkConfig:
    upper_leg_length_m: float = 0.21
    lower_leg_length_m: float = 0.21
    min_reach_margin_m: float = 1e-4

    def __post_init__(self) -> None:
        if self.upper_leg_length_m <= 0.0 or self.lower_leg_length_m <= 0.0:
            raise ValueError("leg lengths must be > 0")


@dataclass(frozen=True)
class LegIkResult:
    hip_pitch_rad: float
    knee_pitch_rad: float
    ankle_pitch_rad: float
    reachable: bool
    clipped_target_x_m: float
    clipped_target_z_m: float


def _clip(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, value)))


def solve_leg_sagittal_ik(
    *,
    target_x_m: float,
    target_z_m: float,
    config: LegIkConfig,
) -> LegIkResult:
    """Solve 2-link sagittal IK with ankle orientation keeping foot level."""
    l1 = config.upper_leg_length_m
    l2 = config.lower_leg_length_m
    r = sqrt(float(target_x_m) ** 2 + float(target_z_m) ** 2)
    r_min = abs(l1 - l2) + config.min_reach_margin_m
    r_max = (l1 + l2) - config.min_reach_margin_m
    reachable = True
    if r < r_min:
        reachable = False
        scale = r_min / max(r, 1e-9)
        tx, tz = float(target_x_m) * scale, float(target_z_m) * scale
    elif r > r_max:
        reachable = False
        scale = r_max / max(r, 1e-9)
        tx, tz = float(target_x_m) * scale, float(target_z_m) * scale
    else:
        tx, tz = float(target_x_m), float(target_z_m)

    rr = sqrt(tx * tx + tz * tz)
    cos_knee = _clip((l1 * l1 + l2 * l2 - rr * rr) / (2.0 * l1 * l2), -1.0, 1.0)
    knee_internal = acos(cos_knee)
    knee_pitch = pi - knee_internal  # 0 = straight, positive = bent

    cos_hip_aux = _clip((l1 * l1 + rr * rr - l2 * l2) / (2.0 * l1 * rr), -1.0, 1.0)
    hip_aux = acos(cos_hip_aux)
    hip_line = atan2(tx, -tz)
    hip_pitch = hip_line - hip_aux

    # Keep foot pitch near level in sagittal plane.
    ankle_pitch = -(hip_pitch + knee_pitch)

    return LegIkResult(
        hip_pitch_rad=float(hip_pitch),
        knee_pitch_rad=float(max(0.0, knee_pitch)),
        ankle_pitch_rad=float(ankle_pitch),
        reachable=reachable,
        clipped_target_x_m=tx,
        clipped_target_z_m=tz,
    )


def forward_leg_sagittal(
    *,
    hip_pitch_rad: float,
    knee_pitch_rad: float,
    config: LegIkConfig,
) -> tuple[float, float]:
    """Forward kinematics for tests/verification."""
    l1 = config.upper_leg_length_m
    l2 = config.lower_leg_length_m
    thigh_x = l1 * sin(hip_pitch_rad)
    thigh_z = -l1 * cos(hip_pitch_rad)
    shank_x = l2 * sin(hip_pitch_rad + knee_pitch_rad)
    shank_z = -l2 * cos(hip_pitch_rad + knee_pitch_rad)
    return float(thigh_x + shank_x), float(thigh_z + shank_z)
