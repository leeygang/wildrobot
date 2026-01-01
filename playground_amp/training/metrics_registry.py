"""Metrics Registry for WildRobot Training.

This module defines a centralized registry for all training metrics.
It enables a "packed array" approach where metrics are stored as a single
fixed-shape tensor, making the rollout fully JIT-compatible with lax.scan.

Design Principles:
1. Single source of truth for metric names and indices
2. Fixed pytree structure for lax.scan compatibility
3. No .get() fallbacks - missing metrics cause hard failures
4. Adding new metrics requires only registry + env changes

Usage:
    from playground_amp.training.metrics_registry import (
        METRIC_NAMES,
        METRIC_INDEX,
        build_metrics_vec,
        unpack_metrics,
    )

    # In env: build packed vector
    metrics_vec = build_metrics_vec({
        "forward_velocity": forward_vel,
        "height": height,
        ...
    })

    # In training: unpack for logging
    metrics_dict = unpack_metrics(metrics_mean)
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, NamedTuple

import jax.numpy as jnp


class Reducer(Enum):
    """Aggregation method for metrics across rollout."""
    MEAN = "mean"           # Standard mean over (T, N)
    MAX = "max"             # Max over (T, N) - for peak values
    SUM = "sum"             # Sum over (T, N) - for counts
    LAST = "last"           # Last value only - for episode-end metrics


class MetricSpec(NamedTuple):
    """Specification for a single metric."""
    name: str                           # Unique identifier (e.g., "forward_velocity")
    reducer: Reducer = Reducer.MEAN     # How to aggregate over rollout
    log_prefix: str = "env"             # W&B prefix (e.g., "env/forward_velocity")
    topline: bool = False               # Also log as topline/ metric
    description: str = ""               # Documentation


# =============================================================================
# METRIC REGISTRY - Single source of truth
# =============================================================================
# Order matters! Index in this list = index in packed vector.
# When adding metrics, append to end to avoid breaking checkpoints.

METRIC_SPECS: List[MetricSpec] = [
    # =========================================================================
    # Core locomotion metrics (indices 0-2)
    # =========================================================================
    MetricSpec(
        name="forward_velocity",
        reducer=Reducer.MEAN,
        topline=True,
        description="Forward velocity in m/s",
    ),
    MetricSpec(
        name="height",
        reducer=Reducer.MEAN,
        description="Robot base height in m",
    ),
    MetricSpec(
        name="velocity_command",
        reducer=Reducer.MEAN,
        topline=True,
        description="Commanded velocity in m/s",
    ),
    # v0.10.4: Episode step count for accurate ep_len calculation
    MetricSpec(
        name="episode_step_count",
        reducer=Reducer.MEAN,  # Will be weighted by dones in training_loop
        description="Step count within episode (for ep_len)",
    ),

    # =========================================================================
    # v0.10.3: Walking tracking metrics (indices 3-4)
    # =========================================================================
    MetricSpec(
        name="tracking/vel_error",
        reducer=Reducer.MEAN,
        topline=True,
        description="|forward_vel - velocity_cmd| tracking error",
    ),
    MetricSpec(
        name="tracking/max_torque",
        reducer=Reducer.MEAN,
        topline=True,
        description="Peak torque as fraction of limit (0-1)",
    ),
    MetricSpec(
        name="tracking/avg_torque",
        reducer=Reducer.MEAN,
        description="Mean absolute actuator torque (Nm)",
    ),

    # =========================================================================
    # Reward components (indices 5-20)
    # =========================================================================
    MetricSpec(
        name="reward/total",
        reducer=Reducer.MEAN,
        description="Total reward",
    ),
    MetricSpec(
        name="reward/forward",
        reducer=Reducer.MEAN,
        description="Forward velocity tracking reward",
    ),
    MetricSpec(
        name="reward/lateral",
        reducer=Reducer.MEAN,
        description="Lateral velocity penalty",
    ),
    MetricSpec(
        name="reward/healthy",
        reducer=Reducer.MEAN,
        description="Alive/healthy bonus",
    ),
    MetricSpec(
        name="reward/orientation",
        reducer=Reducer.MEAN,
        description="Orientation penalty (pitch² + roll²)",
    ),
    MetricSpec(
        name="reward/angvel",
        reducer=Reducer.MEAN,
        description="Angular velocity penalty",
    ),
    MetricSpec(
        name="reward/torque",
        reducer=Reducer.MEAN,
        description="Torque penalty",
    ),
    MetricSpec(
        name="reward/saturation",
        reducer=Reducer.MEAN,
        description="Actuator saturation penalty",
    ),
    MetricSpec(
        name="reward/action_rate",
        reducer=Reducer.MEAN,
        description="Action rate penalty (smoothness)",
    ),
    MetricSpec(
        name="reward/joint_vel",
        reducer=Reducer.MEAN,
        description="Joint velocity penalty",
    ),
    MetricSpec(
        name="reward/slip",
        reducer=Reducer.MEAN,
        description="Foot slip penalty",
    ),
    MetricSpec(
        name="reward/clearance",
        reducer=Reducer.MEAN,
        description="Swing foot clearance reward",
    ),
    # v0.10.4: Standing penalty
    MetricSpec(
        name="reward/standing",
        reducer=Reducer.MEAN,
        description="Standing still penalty (1.0 when |vel| < threshold)",
    ),
    MetricSpec(
        name="reward/gait_periodicity",
        reducer=Reducer.MEAN,
        description="Alternating support reward (left XOR right contact)",
    ),
    MetricSpec(
        name="reward/hip_swing",
        reducer=Reducer.MEAN,
        description="Hip swing amplitude reward",
    ),
    MetricSpec(
        name="reward/knee_swing",
        reducer=Reducer.MEAN,
        description="Knee swing amplitude reward",
    ),
    MetricSpec(
        name="reward/flight_phase",
        reducer=Reducer.MEAN,
        description="Flight phase penalty (both feet unloaded)",
    ),

    # =========================================================================
    # Debug metrics
    # =========================================================================
    MetricSpec(
        name="debug/pitch",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Body pitch angle",
    ),
    MetricSpec(
        name="debug/roll",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Body roll angle",
    ),
    MetricSpec(
        name="debug/forward_vel",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Raw forward velocity",
    ),
    MetricSpec(
        name="debug/lateral_vel",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Raw lateral velocity",
    ),
    MetricSpec(
        name="debug/left_force",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Left foot contact force",
    ),
    MetricSpec(
        name="debug/right_force",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Right foot contact force",
    ),
    MetricSpec(
        name="debug/left_toe_switch",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Left toe switch state",
    ),
    MetricSpec(
        name="debug/left_heel_switch",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Left heel switch state",
    ),
    MetricSpec(
        name="debug/right_toe_switch",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Right toe switch state",
    ),
    MetricSpec(
        name="debug/right_heel_switch",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Right heel switch state",
    ),

    # =========================================================================
    # Termination diagnostics (indices 23-30)
    # =========================================================================
    MetricSpec(
        name="term/height_low",
        reducer=Reducer.MEAN,
        description="Height too low termination flag",
    ),
    MetricSpec(
        name="term/height_high",
        reducer=Reducer.MEAN,
        description="Height too high termination flag",
    ),
    MetricSpec(
        name="term/pitch",
        reducer=Reducer.MEAN,
        description="Pitch limit termination flag",
    ),
    MetricSpec(
        name="term/roll",
        reducer=Reducer.MEAN,
        description="Roll limit termination flag",
    ),
    MetricSpec(
        name="term/truncated",
        reducer=Reducer.MEAN,
        description="Successful truncation (max steps reached)",
    ),
    MetricSpec(
        name="term/pitch_val",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Pitch value at termination",
    ),
    MetricSpec(
        name="term/roll_val",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Roll value at termination",
    ),
    MetricSpec(
        name="term/height_val",
        reducer=Reducer.MEAN,
        log_prefix="debug",
        description="Height value at termination",
    ),
]

# =============================================================================
# Derived constants
# =============================================================================

# Number of metrics in packed vector
NUM_METRICS: int = len(METRIC_SPECS)

# Ordered list of metric names (for iteration)
METRIC_NAMES: List[str] = [spec.name for spec in METRIC_SPECS]

# Name -> index mapping (for fast lookup)
METRIC_INDEX: Dict[str, int] = {spec.name: i for i, spec in enumerate(METRIC_SPECS)}

# Name -> spec mapping
METRIC_SPEC_BY_NAME: Dict[str, MetricSpec] = {spec.name: spec for spec in METRIC_SPECS}

# Topline metrics (for dashboard)
TOPLINE_METRICS: List[str] = [spec.name for spec in METRIC_SPECS if spec.topline]

# Key used to store packed metrics vector in state.metrics dict
# This follows the same pattern as WR_INFO_KEY for info dict
METRICS_VEC_KEY: str = "wr_metrics_vec"


# =============================================================================
# Helper functions
# =============================================================================

def build_metrics_vec(metrics_dict: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Build packed metrics vector from dict.

    Missing metrics default to zeros for backward compatibility with older
    checkpoints/configs that lack newly added metrics.

    Args:
        metrics_dict: Dict mapping metric names to scalar values.

    Returns:
        Packed vector of shape (NUM_METRICS,) or (N, NUM_METRICS) if batched.
    """
    zeros = jnp.zeros((), dtype=jnp.float32)
    values = [metrics_dict.get(name, zeros) for name in METRIC_NAMES]
    return jnp.stack(values, axis=-1)


def unpack_metrics(metrics_vec: jnp.ndarray) -> Dict[str, jnp.ndarray]:
    """Unpack metrics vector to dict.

    Args:
        metrics_vec: Packed vector of shape (..., NUM_METRICS).

    Returns:
        Dict mapping metric names to values.
    """
    return {name: metrics_vec[..., i] for i, name in enumerate(METRIC_NAMES)}


def aggregate_metrics(
    metrics_vec: jnp.ndarray,
    dones: jnp.ndarray = None,
) -> Dict[str, jnp.ndarray]:
    """Aggregate metrics over rollout using registered reducers.

    Args:
        metrics_vec: Shape (T, N, NUM_METRICS) - metrics over rollout.
        dones: Shape (T, N) - episode done flags (unused currently).

    Returns:
        Dict mapping metric names to aggregated scalar values.
    """
    result = {}

    for i, spec in enumerate(METRIC_SPECS):
        values = metrics_vec[..., i]  # (T, N)

        if spec.reducer == Reducer.MEAN:
            result[spec.name] = jnp.mean(values)
        elif spec.reducer == Reducer.MAX:
            result[spec.name] = jnp.max(values)
        elif spec.reducer == Reducer.SUM:
            result[spec.name] = jnp.sum(values)
        elif spec.reducer == Reducer.LAST:
            result[spec.name] = values[-1].mean()  # Last timestep, mean over envs
        else:
            result[spec.name] = jnp.mean(values)

    return result


def format_for_wandb(
    aggregated_metrics: Dict[str, jnp.ndarray],
) -> Dict[str, float]:
    """Format aggregated metrics for W&B logging.

    Applies log_prefix from registry and creates topline duplicates.

    Args:
        aggregated_metrics: Dict from aggregate_metrics().

    Returns:
        Dict with prefixed keys ready for wandb.log().
    """
    result = {}

    for name, value in aggregated_metrics.items():
        spec = METRIC_SPEC_BY_NAME.get(name)
        if spec is None:
            # Unknown metric, log with default prefix
            result[f"env/{name}"] = float(value)
            continue

        # Primary logging with prefix
        result[f"{spec.log_prefix}/{name}"] = float(value)

        # Topline duplicate
        if spec.topline:
            result[f"topline/{name}"] = float(value)

    return result


def validate_metrics_vec(metrics_vec: jnp.ndarray) -> bool:
    """Check for NaN values indicating missing metrics.

    Args:
        metrics_vec: Packed metrics vector.

    Returns:
        True if all values are valid (no NaN).
    """
    return not jnp.any(jnp.isnan(metrics_vec))


def get_missing_metrics(metrics_dict: Dict[str, jnp.ndarray]) -> List[str]:
    """Get list of registered metrics missing from dict.

    Args:
        metrics_dict: Dict to check.

    Returns:
        List of missing metric names.
    """
    return [name for name in METRIC_NAMES if name not in metrics_dict]
