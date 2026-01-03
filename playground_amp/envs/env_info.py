"""Typed environment state structures for WildRobot.

This module defines the strict schema for algorithm-critical fields that must
exist at every step. Following industry best practice for JAX-based RL:

1. Algorithm-critical persistent state lives in `info["wr"]` as a typed pytree
2. Wrapper-injected fields (episode_metrics, truncation, etc.) stay at top level
3. Required fields fail loudly when missing, not silently default to zeros

Usage:
    # In env.step():
    wr_info = WildRobotInfo(
        prev_root_pos=curr_root_pos,
        prev_root_quat=curr_root_quat,
        ...
    )
    info = {**wrapper_info, "wr": wr_info}

    # In training code:
    wr = env_state.info["wr"]  # Typed access, fails if missing
    velocity = compute_velocity(wr.prev_root_pos, curr_pos)

See: https://flax.readthedocs.io/en/latest/api_reference/flax.struct.html
"""

from __future__ import annotations

from typing import Any

from playground_amp.envs.disturbance import DisturbanceSchedule

import jax.numpy as jnp

# Use flax.struct for JAX-compatible dataclass with pytree registration
# Falls back to NamedTuple if flax not available
try:
    import flax.struct as struct

    @struct.dataclass
    class WildRobotInfo:
        """Strict schema for WildRobot algorithm-critical info fields.

        These fields are REQUIRED at every step and must not be accessed via
        .get() with defaults. Missing fields should cause hard failures.

        All fields use post-step convention: they reflect state AFTER the
        physics step, to be used as "previous" values in the next step.

        Attributes:
            step_count: Current step count in episode (0 at reset)
            prev_action: Action applied in this step (for action rate penalty)
            truncated: 1.0 if episode ended due to max steps (success metric)
            velocity_cmd: Target velocity for this episode (scalar, m/s)

            # Finite-diff velocity computation (AMP feature parity)
            prev_root_pos: Root position after this step (3,)
            prev_root_quat: Root quaternion (wxyz) after this step (4,)

            # Slip computation
            prev_left_foot_pos: Left foot position after this step (3,)
            prev_right_foot_pos: Right foot position after this step (3,)

            # AMP features
            foot_contacts: Normalized foot contact forces (4,)
            root_height: Root height after this step (scalar)

            # Observation masking (Sim2Real)
            linvel_obs_mask: 1.0 if linvel is valid, 0.0 if forced to zero

            # Disturbance pushes (episode-constant schedule)
            push_schedule: DisturbanceSchedule
        """
        # Core bookkeeping
        step_count: jnp.ndarray  # shape=()
        prev_action: jnp.ndarray  # shape=(action_size,)
        truncated: jnp.ndarray  # shape=(), sticky through auto-reset
        velocity_cmd: jnp.ndarray  # shape=(), target velocity for episode

        # Finite-diff velocity (post-step values)
        prev_root_pos: jnp.ndarray  # shape=(3,)
        prev_root_quat: jnp.ndarray  # shape=(4,) wxyz

        # Slip computation (post-step values)
        prev_left_foot_pos: jnp.ndarray  # shape=(3,)
        prev_right_foot_pos: jnp.ndarray  # shape=(3,)

        # AMP features (post-step values)
        foot_contacts: jnp.ndarray  # shape=(4,)
        root_height: jnp.ndarray  # shape=()
        linvel_obs_mask: jnp.ndarray  # shape=()
        push_schedule: DisturbanceSchedule

except ImportError:
    # Fallback to NamedTuple if flax not available
    from typing import NamedTuple

    class WildRobotInfo(NamedTuple):
        """Strict schema for WildRobot algorithm-critical info fields."""
        step_count: jnp.ndarray
        prev_action: jnp.ndarray
        truncated: jnp.ndarray
        velocity_cmd: jnp.ndarray
        prev_root_pos: jnp.ndarray
        prev_root_quat: jnp.ndarray
        prev_left_foot_pos: jnp.ndarray
        prev_right_foot_pos: jnp.ndarray
        foot_contacts: jnp.ndarray
        root_height: jnp.ndarray
        linvel_obs_mask: jnp.ndarray
        push_schedule: DisturbanceSchedule


# =============================================================================
# Validation utilities
# =============================================================================

def get_expected_shapes(action_size: int = None) -> dict:
    """Get expected shapes for WildRobotInfo fields.

    Args:
        action_size: Action dimension. If None, reads from robot_config.

    Returns:
        Dict mapping field names to expected shapes.
    """
    if action_size is None:
        from playground_amp.configs.training_config import get_robot_config
        action_size = get_robot_config().action_dim

    return {
        "step_count": (),
        "prev_action": (action_size,),
        "truncated": (),
        "velocity_cmd": (),
        "prev_root_pos": (3,),
        "prev_root_quat": (4,),
        "prev_left_foot_pos": (3,),
        "prev_right_foot_pos": (3,),
        "foot_contacts": (4,),
        "root_height": (),
        "linvel_obs_mask": (),
        "push_schedule": {
            "start_step": (),
            "end_step": (),
            "force_xy": (2,),
            "rng": (2,),
        },
    }


# Default shapes (for documentation, actual validation uses get_expected_shapes)
WILDROBOT_INFO_SHAPES = get_expected_shapes.__doc__  # Placeholder, use get_expected_shapes()

# Fields that must never be all zeros
WILDROBOT_INFO_NONZERO_FIELDS = {
    "prev_root_pos",  # Robot always has position
    "prev_root_quat",  # Quaternion is never all zeros
    "prev_left_foot_pos",  # Foot always has position
    "prev_right_foot_pos",  # Foot always has position
    "root_height",  # Robot always has height > 0
}


def validate_wildrobot_info(
    wr_info: WildRobotInfo,
    context: str = "",
    action_size: int = None,
) -> list[str]:
    """Validate WildRobotInfo has correct shapes and non-zero required fields.

    Args:
        wr_info: The WildRobotInfo to validate
        context: Context string for error messages (e.g., "reset", "step")
        action_size: Expected action size. If None, reads from robot_config.

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    # Get expected shapes (parameterized by action_size)
    expected_shapes = get_expected_shapes(action_size)

    for field_name, expected_shape in expected_shapes.items():
        value = getattr(wr_info, field_name, None)

        if value is None:
            errors.append(f"[{context}] Missing required field: '{field_name}'")
            continue

        if field_name == "push_schedule":
            if not isinstance(value, DisturbanceSchedule):
                errors.append(
                    f"[{context}] Field '{field_name}': expected DisturbanceSchedule, "
                    f"got {type(value)}"
                )
                continue
            for sub_name, sub_shape in expected_shape.items():
                sub_value = getattr(value, sub_name, None)
                if sub_value is None:
                    errors.append(
                        f"[{context}] Field '{field_name}.{sub_name}' is missing"
                    )
                    continue
                if sub_value.shape != sub_shape:
                    errors.append(
                        f"[{context}] Field '{field_name}.{sub_name}': expected "
                        f"shape {sub_shape}, got {sub_value.shape}"
                    )
        else:
            if value.shape != expected_shape:
                errors.append(
                    f"[{context}] Field '{field_name}': expected shape {expected_shape}, "
                    f"got {value.shape}"
                )

        if field_name in WILDROBOT_INFO_NONZERO_FIELDS:
            if jnp.allclose(value, 0.0):
                errors.append(
                    f"[{context}] Field '{field_name}' cannot be all zeros (got {value})"
                )

    return errors


def assert_wildrobot_info_valid(
    wr_info: WildRobotInfo,
    context: str = "",
    action_size: int = None,
):
    """Assert WildRobotInfo is valid, raising detailed error if not."""
    errors = validate_wildrobot_info(wr_info, context, action_size)
    if errors:
        error_msg = "\n".join(errors)
        raise ValueError(f"WildRobotInfo validation failed:\n{error_msg}")


def get_wr_info(info: dict, context: str = "") -> WildRobotInfo:
    """Extract WildRobotInfo from info dict, failing loudly if missing.

    Use this instead of info.get("wr", ...) to ensure hard failure on missing.

    Args:
        info: The info dict from env state
        context: Context for error message

    Returns:
        WildRobotInfo pytree

    Raises:
        KeyError: If "wr" key is missing
    """
    if "wr" not in info:
        raise KeyError(
            f"[{context}] Required 'wr' namespace missing from info dict. "
            "This indicates env state corruption or wrapper misconfiguration."
        )
    return info["wr"]


# =============================================================================
# Reserved key for namespace
# =============================================================================

# The key used to store WildRobotInfo in the info dict
# Using a short key to avoid collision with wrapper fields
WR_INFO_KEY = "wr"
