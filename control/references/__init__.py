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
from .walking_ref_v2 import (
    WalkingRefV2Config,
    WalkingRefV2Input,
    WalkingRefV2State,
    WalkingRefV2Mode,
    WalkingReferenceV2Output,
    compute_support_health_v2,
    step_reference_v2,
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
    "WalkingRefV2Config",
    "WalkingRefV2Input",
    "WalkingRefV2State",
    "WalkingRefV2Mode",
    "WalkingReferenceV2Output",
    "compute_support_health_v2",
    "step_reference_v2",
]
