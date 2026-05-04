"""Bounded leg IK utility (v0.20.1).

The v1/v2 ``NominalKinematicsAdapter`` interface was deleted with the
parametric reference stack at v0.20.1; the offline ``ReferenceLibrary``
does IK once at trajectory build time and the v3 env doesn't need a
runtime adapter."""

from .leg_ik import LegIkConfig, LegIkResult, solve_leg_sagittal_ik, forward_leg_sagittal

__all__ = [
    "LegIkConfig",
    "LegIkResult",
    "solve_leg_sagittal_ik",
    "forward_leg_sagittal",
]
