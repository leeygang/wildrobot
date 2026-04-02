"""Locomotion reference contracts for the v0.19 platform pivot."""

from .locomotion_contract import (
    LocomotionCommand,
    LocomotionReferenceState,
    LocomotionObservationContract,
    LocomotionActionContract,
    validate_locomotion_contract,
)
from .walking_ref_v1 import (
    WalkingRefV1Config,
    WalkingRefV1Input,
    WalkingRefV1State,
    WalkingReferenceOutput,
    step_reference,
)

__all__ = [
    "LocomotionCommand",
    "LocomotionReferenceState",
    "LocomotionObservationContract",
    "LocomotionActionContract",
    "validate_locomotion_contract",
    "WalkingRefV1Config",
    "WalkingRefV1Input",
    "WalkingRefV1State",
    "WalkingReferenceOutput",
    "step_reference",
]
