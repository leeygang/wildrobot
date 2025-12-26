"""Offline AMP feature extraction from reference data (NumPy).

This module processes reference motion data (GMR, physics) into
AMP features for discriminator training.

For online policy feature extraction, see policy_features.py.
For shared feature configuration, see feature_config.py.

IMPORTANT: Uses FeatureLayout from feature_config.py as the SINGLE SOURCE
OF TRUTH for feature ordering and dimensions.

Used by:
- scripts/convert_ref_data_*.py
- scripts/generate_physics_reference_dataset.py
- playground_amp/data/gmr_to_physics_ref_data.py
- train.py (load_reference_features)
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Union

import numpy as np

from playground_amp.configs.feature_config import FeatureConfig, get_feature_config
from playground_amp.configs.training_config import get_training_config


# =============================================================================
# Reference Data Loading
# =============================================================================


def load_reference_features(
    pickle_file_path: Union[str, Path],
) -> np.ndarray:
    """Load reference motion data and extract AMP features.

    Handles multiple data formats:
    - Pre-extracted features (2D array)
    - Dict with 'features' key
    - Dict with raw motion data

    Mirror augmentation setting is read from training_config.

    Args:
        pickle_file_path: Path to reference data pickle file

    Returns:
        Reference features array (N, feature_dim)
    """
    with open(pickle_file_path, "rb") as f:
        ref_data = pickle.load(f)

    # Extract features based on data format
    if isinstance(ref_data, np.ndarray) and ref_data.ndim == 2:
        # Already extracted features (2D array: num_samples x feature_dim)
        ref_features = ref_data
    elif isinstance(ref_data, dict):
        if "features" in ref_data:
            ref_features = np.array(ref_data["features"])
        else:
            raise ValueError(f"Unknown reference data dict format: {ref_data.keys()}")
    else:
        ref_features = np.array(ref_data)

    # Read enable_mirror from training config
    training_config = get_training_config()
    enable_mirror = training_config.raw_config.get("amp", {}).get(
        "enable_mirror_augmentation", True
    )

    # Apply gait mirroring augmentation
    if enable_mirror:
        from playground_amp.amp.amp_mirror import augment_with_mirror

        ref_features = augment_with_mirror(ref_features)

    return ref_features


# =============================================================================
# Preprocessing Utilities (NumPy)
# =============================================================================
def normalize_velocity(
    root_linvel: np.ndarray,
    min_speed_threshold: float = 0.05,
) -> np.ndarray:
    """Normalize root linear velocity to unit direction.

    When speed is below threshold, returns zero vector to avoid noisy
    normalized direction from near-stationary states.

    Args:
        root_linvel: Root linear velocity (..., 3)
        min_speed_threshold: Below this speed (m/s), return zero vector

    Returns:
        Normalized velocity direction (..., 3), unit vectors or zeros
    """
    speed = np.linalg.norm(root_linvel, axis=-1, keepdims=True)
    is_moving = speed > min_speed_threshold
    return np.where(
        is_moving, root_linvel / (speed + 1e-8), np.zeros_like(root_linvel)
    )


# =============================================================================
# Feature Extraction
# =============================================================================


def extract_ref_features(
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    root_linvel: np.ndarray,
    root_angvel: np.ndarray,
    root_height: np.ndarray,
    foot_contacts: np.ndarray,
    config: FeatureConfig,
    normalize_vel: bool = False,
    min_speed_threshold: float = 0.05,
) -> np.ndarray:
    """Extract AMP features from reference motion data (NumPy).

    Uses FeatureLayout.get_feature_list() for consistent ordering with online extraction.
    Offline extraction always produces full features (no dropping).

    Args:
        joint_pos: Joint positions (N, num_joints)
        joint_vel: Joint velocities (N, num_joints)
        root_linvel: Root linear velocity (N, 3)
        root_angvel: Root angular velocity (N, 3)
        root_height: Root height (N,) or (N, 1) - auto-normalized to (N, 1)
        foot_contacts: Foot contacts (N, 4)
        config: FeatureConfig with num_actuated_joints
        normalize_vel: If True, normalize root_linvel to unit direction
        min_speed_threshold: Below this speed (m/s), return zero vector

    Returns:
        features: AMP features (N, full_dim) - always full features for offline
    """
    # Normalize velocity if requested
    if normalize_vel:
        root_linvel = normalize_velocity(root_linvel, min_speed_threshold)

    # Root height: ensure shape is (N, 1) for concatenation
    if root_height.ndim == 1:
        root_height = root_height[:, np.newaxis]

    # Get layout and order features (SINGLE SOURCE OF TRUTH)
    layout = config.get_layout()
    ordered = layout.get_features(
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        root_linvel=root_linvel,
        root_angvel=root_angvel,
        root_height=root_height,
        foot_contacts=foot_contacts,
    )
    return np.concatenate(ordered, axis=-1)
