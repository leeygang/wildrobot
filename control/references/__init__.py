"""Locomotion reference library for WildRobot walking.

v0.20.x architecture (active): the offline ``ReferenceLibrary`` is the
single source of truth for locomotion semantics; runtime is a thin
lookup over it.  The previous runtime-FSM modules
(``walking_ref_v1``, ``walking_ref_v2``, ``locomotion_contract``)
were deleted at v0.20.1 since they had been deprecated since v0.20.0
and were not on the active path.  See ``training/docs/walking_training.md``
and ``training/docs/reference_design.md``.
"""

from .reference_library import (
    ReferenceLibrary,
    ReferenceLibraryMeta,
    ReferencePreviewWindow,
    ReferenceTrajectory,
)
from .runtime_reference_service import (
    RuntimeReferenceService,
    RuntimeReferenceWindow,
)

__all__ = [
    # v0.20 offline reference library (active path)
    "ReferenceLibrary",
    "ReferenceLibraryMeta",
    "ReferencePreviewWindow",
    "ReferenceTrajectory",
    # v0.20.1 runtime lookup service (Layer 2)
    "RuntimeReferenceService",
    "RuntimeReferenceWindow",
]
