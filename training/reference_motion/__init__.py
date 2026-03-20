"""Reference motion loading, phase utilities, and WildRobot retarget helpers."""

from .loader import (
    ReferenceMotionClip,
    load_reference_motion,
    save_reference_motion_clip,
)
from .phase import phase_distance, phase_to_index, phase_to_sin_cos, wrap_phase
from .retarget import (
    generate_bootstrap_nominal_walk_clip,
    generate_real_retarget_nominal_walk_clip,
    retarget_wildrobot_clip,
)

__all__ = [
    "ReferenceMotionClip",
    "load_reference_motion",
    "save_reference_motion_clip",
    "wrap_phase",
    "phase_distance",
    "phase_to_index",
    "phase_to_sin_cos",
    "retarget_wildrobot_clip",
    "generate_bootstrap_nominal_walk_clip",
    "generate_real_retarget_nominal_walk_clip",
]
