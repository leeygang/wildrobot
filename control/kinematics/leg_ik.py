"""Simple sagittal leg IK for nominal stepping targets (v0.19.2)."""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, atan2, cos, pi, sin, sqrt

import jax.numpy as jnp


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


def solve_leg_sagittal_ik_jax(
    *,
    target_x_m: jnp.ndarray,
    target_z_m: jnp.ndarray,
    config: LegIkConfig,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """JAX-friendly sagittal IK used by training env rollout code.

    Returns:
        (hip_pitch_rad, knee_pitch_rad, ankle_pitch_rad, reachable_float)
    """
    l1 = jnp.asarray(config.upper_leg_length_m, dtype=jnp.float32)
    l2 = jnp.asarray(config.lower_leg_length_m, dtype=jnp.float32)
    eps = jnp.asarray(1e-6, dtype=jnp.float32)
    tx = jnp.asarray(target_x_m, dtype=jnp.float32)
    tz = jnp.asarray(target_z_m, dtype=jnp.float32)
    r = jnp.sqrt(tx * tx + tz * tz + eps)
    r_min = jnp.asarray(abs(config.upper_leg_length_m - config.lower_leg_length_m) + config.min_reach_margin_m, dtype=jnp.float32)
    r_max = jnp.asarray((config.upper_leg_length_m + config.lower_leg_length_m) - config.min_reach_margin_m, dtype=jnp.float32)
    reachable = ((r >= r_min) & (r <= r_max)).astype(jnp.float32)
    r_clipped = jnp.clip(r, r_min, r_max)
    scale = r_clipped / jnp.maximum(r, eps)
    tx = tx * scale
    tz = tz * scale
    rr = jnp.sqrt(tx * tx + tz * tz + eps)

    cos_knee = jnp.clip((l1 * l1 + l2 * l2 - rr * rr) / (2.0 * l1 * l2 + eps), -1.0, 1.0)
    knee_internal = jnp.arccos(cos_knee)
    knee_pitch = jnp.maximum(0.0, jnp.asarray(pi, dtype=jnp.float32) - knee_internal)

    cos_hip_aux = jnp.clip((l1 * l1 + rr * rr - l2 * l2) / (2.0 * l1 * rr + eps), -1.0, 1.0)
    hip_aux = jnp.arccos(cos_hip_aux)
    hip_line = jnp.arctan2(tx, -tz)
    hip_pitch = hip_line - hip_aux
    ankle_pitch = -(hip_pitch + knee_pitch)
    return (
        hip_pitch.astype(jnp.float32),
        knee_pitch.astype(jnp.float32),
        ankle_pitch.astype(jnp.float32),
        reachable.astype(jnp.float32),
    )
