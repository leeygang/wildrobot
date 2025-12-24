"""Gait Mirroring for AMP Reference Data.

This module implements the "Gold Rule" of data augmentation for AMP training:
For every motion clip, create a mirrored version by swapping left/right channels.

Why this is critical:
- Raw AMASS data often has asymmetric gait (e.g., right foot planted 86% vs left 66%)
- The discriminator learns this bias and penalizes symmetric gaits
- This forces the robot to limp to satisfy the discriminator
- Mirroring doubles the data and forces discriminator to judge motion QUALITY, not leg preference

Implementation:
1. Swap left/right joint positions and velocities
2. Negate lateral (hip_roll) joints to preserve direction
3. Negate lateral velocity components (Y direction)
4. Swap left/right foot contacts

References:
- NVIDIA IsaacGym locomotion
- DeepMind's Locomotion papers
- AMP (Adversarial Motion Priors) best practices
"""

from __future__ import annotations

import numpy as np


def mirror_amp_features_27dim(features: np.ndarray) -> np.ndarray:
    """Mirror 27-dim AMP features (with waist masked).

    27-dim feature layout:
    - Joint positions (0-7): [left_hip_pitch, left_hip_roll, left_knee_pitch, left_ankle_pitch,
                               right_hip_pitch, right_hip_roll, right_knee_pitch, right_ankle_pitch]
    - Joint velocities (8-15): same order as positions
    - Root velocities (16-21): [linvel_x, linvel_y, linvel_z, angvel_x, angvel_y, angvel_z]
    - Root height (22)
    - Foot contacts (23-26): [left_toe, left_heel, right_toe, right_heel]

    Args:
        features: Array of shape (N, 27) with AMP features

    Returns:
        Mirrored features of shape (N, 27)
    """
    assert features.shape[1] == 27, f"Expected 27-dim features, got {features.shape[1]}"

    mirrored = features.copy()

    # === Joint Positions (indices 0-7) ===
    # Swap left (0-3) with right (4-7)
    left_pos = features[:, 0:4].copy()
    right_pos = features[:, 4:8].copy()
    mirrored[:, 0:4] = right_pos
    mirrored[:, 4:8] = left_pos

    # Negate hip_roll (lateral joint) - indices 1 and 5 after swap
    # hip_roll is index 1 (left) and 5 (right) in original
    # After swap: original right_hip_roll is now at index 1, original left_hip_roll is now at index 5
    mirrored[:, 1] = -mirrored[:, 1]  # Was right_hip_roll, now left
    mirrored[:, 5] = -mirrored[:, 5]  # Was left_hip_roll, now right

    # === Joint Velocities (indices 8-15) ===
    # Swap left (8-11) with right (12-15)
    left_vel = features[:, 8:12].copy()
    right_vel = features[:, 12:16].copy()
    mirrored[:, 8:12] = right_vel
    mirrored[:, 12:16] = left_vel

    # Negate hip_roll velocities (indices 9 and 13 after swap)
    mirrored[:, 9] = -mirrored[:, 9]  # Was right_hip_roll_vel, now left
    mirrored[:, 13] = -mirrored[:, 13]  # Was left_hip_roll_vel, now right

    # === Root Velocities (indices 16-21) ===
    # linvel_y (index 17): negate (lateral direction flips in mirror)
    mirrored[:, 17] = -features[:, 17]

    # angvel_x (index 19): negate (roll flips in mirror)
    mirrored[:, 19] = -features[:, 19]

    # angvel_z (index 21): negate (yaw flips in mirror)
    mirrored[:, 21] = -features[:, 21]

    # === Root Height (index 22) ===
    # No change needed

    # === Foot Contacts (indices 23-26) ===
    # Swap left (23-24) with right (25-26)
    left_contacts = features[:, 23:25].copy()
    right_contacts = features[:, 25:27].copy()
    mirrored[:, 23:25] = right_contacts
    mirrored[:, 25:27] = left_contacts

    return mirrored


def mirror_amp_features_29dim(features: np.ndarray) -> np.ndarray:
    """Mirror 29-dim AMP features (waist included).

    29-dim feature layout:
    - Joint positions (0-8): [waist_yaw, left_hip_pitch, left_hip_roll, left_knee_pitch, left_ankle_pitch,
                               right_hip_pitch, right_hip_roll, right_knee_pitch, right_ankle_pitch]
    - Joint velocities (9-17): same order as positions
    - Root velocities (18-23): [linvel_x, linvel_y, linvel_z, angvel_x, angvel_y, angvel_z]
    - Root height (24)
    - Foot contacts (25-28): [left_toe, left_heel, right_toe, right_heel]

    Args:
        features: Array of shape (N, 29) with AMP features

    Returns:
        Mirrored features of shape (N, 29)
    """
    assert features.shape[1] == 29, f"Expected 29-dim features, got {features.shape[1]}"

    mirrored = features.copy()

    # === Waist Yaw (index 0) ===
    # Negate (yaw flips in mirror)
    mirrored[:, 0] = -features[:, 0]

    # === Joint Positions (indices 1-8) ===
    # Swap left (1-4) with right (5-8)
    left_pos = features[:, 1:5].copy()
    right_pos = features[:, 5:9].copy()
    mirrored[:, 1:5] = right_pos
    mirrored[:, 5:9] = left_pos

    # Negate hip_roll (indices 2 and 6 after swap)
    mirrored[:, 2] = -mirrored[:, 2]  # Was right_hip_roll, now left
    mirrored[:, 6] = -mirrored[:, 6]  # Was left_hip_roll, now right

    # === Waist Yaw Velocity (index 9) ===
    mirrored[:, 9] = -features[:, 9]

    # === Joint Velocities (indices 10-17) ===
    # Swap left (10-13) with right (14-17)
    left_vel = features[:, 10:14].copy()
    right_vel = features[:, 14:18].copy()
    mirrored[:, 10:14] = right_vel
    mirrored[:, 14:18] = left_vel

    # Negate hip_roll velocities (indices 11 and 15 after swap)
    mirrored[:, 11] = -mirrored[:, 11]
    mirrored[:, 15] = -mirrored[:, 15]

    # === Root Velocities (indices 18-23) ===
    mirrored[:, 19] = -features[:, 19]  # linvel_y
    mirrored[:, 21] = -features[:, 21]  # angvel_x
    mirrored[:, 23] = -features[:, 23]  # angvel_z

    # === Root Height (index 24) ===
    # No change needed

    # === Foot Contacts (indices 25-28) ===
    left_contacts = features[:, 25:27].copy()
    right_contacts = features[:, 27:29].copy()
    mirrored[:, 25:27] = right_contacts
    mirrored[:, 27:29] = left_contacts

    return mirrored


def augment_with_mirror(features: np.ndarray) -> np.ndarray:
    """Augment reference data by adding mirrored versions.

    This is the main function to call for data augmentation.
    Automatically detects feature dimension and applies appropriate mirroring.

    Args:
        features: Array of shape (N, D) where D is 27 or 29

    Returns:
        Augmented features of shape (2*N, D) with original + mirrored data
    """
    feature_dim = features.shape[1]

    if feature_dim == 27:
        mirrored = mirror_amp_features_27dim(features)
    elif feature_dim == 29:
        mirrored = mirror_amp_features_29dim(features)
    else:
        raise ValueError(
            f"Unsupported feature dimension: {feature_dim}. Expected 27 or 29."
        )

    # Concatenate original and mirrored
    augmented = np.concatenate([features, mirrored], axis=0)

    return augmented


def verify_mirror_symmetry(features: np.ndarray) -> dict:
    """Verify that mirroring produces symmetric statistics.

    After augmentation, left and right features should have identical means.

    Args:
        features: Augmented features (should include mirrored data)

    Returns:
        Dictionary with symmetry statistics
    """
    feature_dim = features.shape[1]

    if feature_dim == 27:
        # Joint positions
        left_pos_mean = np.mean(features[:, 0:4], axis=0)
        right_pos_mean = np.mean(features[:, 4:8], axis=0)

        # Joint velocities
        left_vel_mean = np.mean(features[:, 8:12], axis=0)
        right_vel_mean = np.mean(features[:, 12:16], axis=0)

        # Foot contacts
        left_contact_mean = np.mean(features[:, 23:25], axis=0)
        right_contact_mean = np.mean(features[:, 25:27], axis=0)

    elif feature_dim == 29:
        # Joint positions (skip waist at index 0)
        left_pos_mean = np.mean(features[:, 1:5], axis=0)
        right_pos_mean = np.mean(features[:, 5:9], axis=0)

        # Joint velocities (skip waist vel at index 9)
        left_vel_mean = np.mean(features[:, 10:14], axis=0)
        right_vel_mean = np.mean(features[:, 14:18], axis=0)

        # Foot contacts
        left_contact_mean = np.mean(features[:, 25:27], axis=0)
        right_contact_mean = np.mean(features[:, 27:29], axis=0)
    else:
        raise ValueError(f"Unsupported feature dimension: {feature_dim}")

    # For hip_roll (index 1 in left, 5 in right for 27-dim),
    # the means should be NEGATIVES of each other after proper mirroring
    # But for hip_pitch, knee_pitch, ankle_pitch, they should match

    return {
        "left_pos_mean": left_pos_mean,
        "right_pos_mean": right_pos_mean,
        "left_vel_mean": left_vel_mean,
        "right_vel_mean": right_vel_mean,
        "left_contact_mean": left_contact_mean,
        "right_contact_mean": right_contact_mean,
        "pos_symmetry_error": np.abs(
            left_pos_mean[[0, 2, 3]] - right_pos_mean[[0, 2, 3]]
        ).max(),  # pitch joints
        "contact_symmetry_error": np.abs(left_contact_mean - right_contact_mean).max(),
    }
