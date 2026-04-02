"""Adapters connecting references, kinematics, and policy contracts."""

from .locomotion_action_adapter import compose_final_joint_target
from .reference_to_joint_targets import NominalTargetOutput, reference_to_nominal_joint_targets

__all__ = [
    "compose_final_joint_target",
    "NominalTargetOutput",
    "reference_to_nominal_joint_targets",
]
