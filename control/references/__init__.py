"""Locomotion reference contracts and library for WildRobot walking.

v0.19.x: Runtime walking FSM (walking_ref_v1, walking_ref_v2)
v0.20.x: Offline reference library (reference_library)
"""

from .locomotion_contract import (
    LocomotionCommand,
    LocomotionReferenceState,
    LocomotionObservationContract,
    LocomotionActionContract,
    validate_locomotion_contract,
)
from .reference_library import (
    ReferenceLibrary,
    ReferenceLibraryMeta,
    ReferencePreviewWindow,
    ReferenceTrajectory,
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
    # v0.20 offline reference library (active path)
    "ReferenceLibrary",
    "ReferenceLibraryMeta",
    "ReferencePreviewWindow",
    "ReferenceTrajectory",
    # v0.19 contracts (backward compatibility)
    "LocomotionCommand",
    "LocomotionReferenceState",
    "LocomotionObservationContract",
    "LocomotionActionContract",
    "validate_locomotion_contract",
    # v0.19 runtime FSM (deprecated for runtime, allowed for offline generation)
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
