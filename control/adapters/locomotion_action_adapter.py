"""Locomotion action composition helper (schema-level)."""

from __future__ import annotations

from typing import Tuple


def compose_final_joint_target(
    nominal_joint_target: Tuple[float, ...],
    residual_joint_delta: Tuple[float, ...],
) -> Tuple[float, ...]:
    """Compose final target from nominal + residual policy delta."""
    if len(nominal_joint_target) != len(residual_joint_delta):
        raise ValueError("nominal_joint_target and residual_joint_delta must have same length")
    return tuple(a + b for a, b in zip(nominal_joint_target, residual_joint_delta))
