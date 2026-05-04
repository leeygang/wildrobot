"""Minimal LIPM helpers for walking reference generation (v0.19.2)."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Tuple


Vec2 = Tuple[float, float]


@dataclass(frozen=True)
class LipmConfig:
    com_height_m: float = 0.40
    gravity_m_s2: float = 9.81

    def __post_init__(self) -> None:
        if self.com_height_m <= 0.0:
            raise ValueError("com_height_m must be > 0")
        if self.gravity_m_s2 <= 0.0:
            raise ValueError("gravity_m_s2 must be > 0")


def natural_frequency(config: LipmConfig) -> float:
    return sqrt(config.gravity_m_s2 / config.com_height_m)


def capture_point(
    *,
    com_position_stance_frame: Vec2,
    com_velocity_stance_frame: Vec2,
    config: LipmConfig,
) -> Vec2:
    """Compute capture-point in stance-foot frame."""
    omega = natural_frequency(config)
    cp_x = float(com_position_stance_frame[0]) + float(com_velocity_stance_frame[0]) / omega
    cp_y = float(com_position_stance_frame[1]) + float(com_velocity_stance_frame[1]) / omega
    return (cp_x, cp_y)
