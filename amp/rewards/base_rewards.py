"""Base interfaces for modular reward components."""

from typing import Dict, Protocol
import jax
import jax.numpy as jp
from mujoco import mjx


class RewardComponent(Protocol):
    """Protocol for modular reward components.

    Each reward component is responsible for computing one aspect of the total reward
    (e.g., velocity tracking, foot contact, energy efficiency).
    """

    def compute(
        self,
        data: mjx.Data,
        action: jax.Array,
        **kwargs
    ) -> jax.Array:
        """Compute the reward value.

        Args:
            data: MuJoCo data containing current state
            action: Action taken at current step
            **kwargs: Additional context (e.g., velocity_cmd, phase, prev_action)

        Returns:
            Scalar reward value
        """
        ...

    def get_info(self, data: mjx.Data) -> Dict[str, float]:
        """Get debugging info for logging (optional).

        Args:
            data: MuJoCo data containing current state

        Returns:
            Dictionary of debug metrics (e.g., {'foot_contact_left': 0.8})
        """
        return {}
