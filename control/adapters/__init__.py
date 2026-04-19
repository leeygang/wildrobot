"""Adapters connecting references, kinematics, and policy contracts.

The v0.19.x ``reference_to_joint_targets`` module was deleted at
v0.20.1 along with ``locomotion_contract``.  The remaining
``locomotion_action_adapter`` is policy-contract-side and is kept.
"""

from .locomotion_action_adapter import compose_final_joint_target

__all__ = [
    "compose_final_joint_target",
]
