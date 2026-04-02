"""Reduced-order locomotion helpers (LIPM/DCM) for walking references."""

from .lipm import LipmConfig, capture_point, natural_frequency
from .dcm import DcmStepConfig, compute_next_foothold_1d, dcm_terminal_scale

__all__ = [
    "LipmConfig",
    "capture_point",
    "natural_frequency",
    "DcmStepConfig",
    "compute_next_foothold_1d",
    "dcm_terminal_scale",
]
