"""Phase utilities for reference-motion teacher training."""

from __future__ import annotations

import numpy as np


def wrap_phase(phase: np.ndarray | float) -> np.ndarray:
    """Wrap phase values into [0, 1)."""
    return np.mod(np.asarray(phase, dtype=np.float32), 1.0)


def phase_distance(phase_a: np.ndarray | float, phase_b: np.ndarray | float) -> np.ndarray:
    """Shortest wrapped distance between phases (range [0, 0.5])."""
    a = wrap_phase(phase_a)
    b = wrap_phase(phase_b)
    delta = np.mod(a - b + 0.5, 1.0) - 0.5
    return np.abs(delta).astype(np.float32)


def phase_to_index(phase: np.ndarray | float, frame_count: int) -> np.ndarray:
    """Convert normalized phase in [0, 1) to frame indices."""
    if frame_count <= 0:
        raise ValueError(f"frame_count must be > 0, got {frame_count}")
    wrapped = wrap_phase(phase)
    idx = np.floor(wrapped * float(frame_count)).astype(np.int32)
    return np.mod(idx, frame_count)


def phase_to_sin_cos(phase: np.ndarray | float) -> np.ndarray:
    """Encode phase as [sin(2πp), cos(2πp)]."""
    wrapped = wrap_phase(phase)
    theta = 2.0 * np.pi * wrapped
    return np.stack([np.sin(theta), np.cos(theta)], axis=-1).astype(np.float32)

