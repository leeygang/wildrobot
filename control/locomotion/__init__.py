"""Locomotion controller package (v0.19.5).

Provides a unified walking controller with two deployment modes:
- nominal_only: walking reference + IK → joint targets (no neural network)
- residual_ppo: walking reference + IK → q_ref + policy delta → joint targets
"""

from control.locomotion.nominal_ik_adapter import NominalIkConfig, compute_nominal_q_ref
from control.locomotion.walking_controller import WalkingController, WalkingControllerConfig

__all__ = [
    "NominalIkConfig",
    "compute_nominal_q_ref",
    "WalkingController",
    "WalkingControllerConfig",
]
