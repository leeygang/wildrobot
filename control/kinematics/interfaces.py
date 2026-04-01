"""Minimal kinematics interfaces for the v0.19.0 platform pivot."""

from __future__ import annotations

from typing import Protocol, Tuple

from control.references.locomotion_contract import LocomotionReferenceState


class NominalKinematicsAdapter(Protocol):
    """Converts reference state into nominal joint targets.

    Milestone 0 contract only: implementations are intentionally deferred.
    """

    def nominal_joint_target(
        self,
        reference_state: LocomotionReferenceState,
    ) -> Tuple[float, ...]:
        """Return nominal joint targets in radians."""
