"""Smoothness reward components for natural motion."""

from typing import Dict, Optional
import jax
import jax.numpy as jp


class SmoothnessReward:
    """Reward smooth, natural joint motion.

    Penalizes:
    - Deviation from nominal pose
    - Joint limit violations
    - High accelerations (jerky motion)
    - Action discontinuities

    Tracks:
    - Previous joint velocities (for acceleration)
    - Previous actions (for action rate)
    """

    def __init__(
        self,
        default_qpos: jax.Array,
        joint_limits: float = 1.5,
        dt: float = 0.02,
    ):
        """Initialize smoothness reward.

        Args:
            default_qpos: Nominal joint positions (standing pose)
            joint_limits: Joint position limits (radians)
            dt: Control timestep for acceleration calculation
        """
        self.default_qpos = default_qpos
        self.joint_limits = joint_limits
        self.dt = dt

    def compute(
        self,
        qpos: jax.Array,
        qvel: jax.Array,
        action: jax.Array,
        prev_qvel: Optional[jax.Array] = None,
        prev_action: Optional[jax.Array] = None,
    ) -> Dict[str, jax.Array]:
        """Compute smoothness reward components.

        Args:
            qpos: Joint positions
            qvel: Joint velocities
            action: Current action
            prev_qvel: Previous joint velocities (for acceleration)
            prev_action: Previous action (for action rate)

        Returns:
            Dictionary of reward components:
                - nominal_joint_position: Deviation from default pose
                - joint_position_limit: Joint limit violations
                - joint_velocity: Velocity magnitude (metric only)
                - joint_acceleration: Jerkiness penalty
                - action_rate: Action smoothness
        """
        # Nominal pose penalty
        nominal_penalty = -jp.sum(jp.square(qpos - self.default_qpos))

        # Joint limit penalty
        limit_penalty = -jp.sum(
            jp.square(jp.clip(jp.abs(qpos) - self.joint_limits, 0.0, jp.inf))
        )

        # Joint velocity metric (for monitoring only, not penalty)
        velocity_metric = -jp.sum(jp.square(qvel))

        # Joint acceleration penalty (proper smoothness measure)
        if prev_qvel is not None:
            acceleration = (qvel - prev_qvel) / self.dt
            acceleration_penalty = -jp.sum(jp.square(acceleration))
        else:
            acceleration_penalty = jp.array(0.0)

        # Action rate penalty (action smoothness)
        if prev_action is not None:
            action_rate_penalty = -jp.sum(jp.square(action - prev_action))
        else:
            action_rate_penalty = -jp.sum(jp.square(action))

        return {
            "nominal_joint_position": nominal_penalty,
            "joint_position_limit": limit_penalty,
            "joint_velocity": velocity_metric,  # Metric only
            "joint_acceleration": acceleration_penalty,
            "action_rate": action_rate_penalty,
        }

    def get_info(
        self,
        qpos: jax.Array,
        qvel: jax.Array,
        prev_qvel: Optional[jax.Array] = None,
    ) -> Dict[str, float]:
        """Get smoothness metrics for debugging.

        Args:
            qpos: Joint positions
            qvel: Joint velocities
            prev_qvel: Previous joint velocities

        Returns:
            Debug information
        """
        info = {
            "max_joint_pos_deviation": float(jp.max(jp.abs(qpos - self.default_qpos))),
            "max_joint_velocity": float(jp.max(jp.abs(qvel))),
            "rms_joint_velocity": float(jp.sqrt(jp.mean(jp.square(qvel)))),
        }

        if prev_qvel is not None:
            acceleration = (qvel - prev_qvel) / self.dt
            info["max_joint_acceleration"] = float(jp.max(jp.abs(acceleration)))
            info["rms_joint_acceleration"] = float(jp.sqrt(jp.mean(jp.square(acceleration))))

        return info
