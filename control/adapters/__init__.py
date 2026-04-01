"""Adapters connecting references, kinematics, and policy contracts."""

from .locomotion_action_adapter import compose_final_joint_target

__all__ = ["compose_final_joint_target"]
