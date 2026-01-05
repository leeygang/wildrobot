"""Policy contract: shared policy I/O specification and validation."""

from __future__ import annotations

from .spec import (
    ActionSpec,
    JointSpec,
    ModelSpec,
    ObservationSpec,
    ObsFieldSpec,
    PolicyBundle,
    PolicySpec,
    RobotSpec,
    validate_spec,
    validate_runtime_compat,
)
from .layout import get_slices

__all__ = [
    "PolicySpec",
    "ModelSpec",
    "RobotSpec",
    "ObservationSpec",
    "ObsFieldSpec",
    "ActionSpec",
    "JointSpec",
    "PolicyBundle",
    "validate_spec",
    "validate_runtime_compat",
    "get_slices",
]
