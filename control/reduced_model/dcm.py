"""Minimal DCM helpers for bounded step placement (v0.19.2)."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp


@dataclass(frozen=True)
class DcmStepConfig:
    step_time_s: float
    omega: float

    def __post_init__(self) -> None:
        if self.step_time_s <= 0.0:
            raise ValueError("step_time_s must be > 0")
        if self.omega <= 0.0:
            raise ValueError("omega must be > 0")


def dcm_terminal_scale(config: DcmStepConfig) -> float:
    return exp(-config.omega * config.step_time_s)


def compute_next_foothold_1d(
    *,
    dcm_now: float,
    dcm_target_next: float,
    config: DcmStepConfig,
) -> float:
    """One-dimensional bounded DCM placement relation."""
    b = dcm_terminal_scale(config)
    return float((dcm_target_next - b * dcm_now) / max(1.0 - b, 1e-6))
