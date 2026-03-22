"""Interfaces for the architecture-pivot control stack.

These interfaces define contracts between:
- reduced-order model adapter
- planner
- execution layer

They are intentionally lightweight and do not claim a full MPC implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import jax


@dataclass
class ControllerDebugSignals:
    """Inspectable planner/controller signals for logging."""

    planner_active: jax.Array
    controller_active: jax.Array
    target_com_x: jax.Array
    target_com_y: jax.Array
    target_step_x: jax.Array
    target_step_y: jax.Array
    support_state: jax.Array
    step_requested: jax.Array


class StandingController(Protocol):
    """Minimal standing-controller contract for v0.17.3 bring-up."""

    def compute_action(
        self,
        *,
        pitch: jax.Array,
        roll: jax.Array,
        pitch_rate: jax.Array,
        roll_rate: jax.Array,
        height_error: jax.Array,
        forward_vel: jax.Array,
        lateral_vel: jax.Array,
        default_action: jax.Array,
        policy_action: jax.Array,
    ) -> tuple[jax.Array, ControllerDebugSignals]:
        """Return action in policy-action space plus debug signals."""

