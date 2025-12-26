#!/usr/bin/env python3
"""Convert reference motion data to use normalized velocity direction.

The AMP discriminator uses features with normalized velocity direction
(unit vector) instead of raw velocity. This preserves directional
information while removing speed magnitude that causes mismatch.

Supports both:
- 27-dim features (8 joints): root_linvel at indices 16-18
- 29-dim features (9 joints): root_linvel at indices 18-20

Usage:
    python scripts/convert_ref_data_normalized_velocity.py
"""

import pickle
import numpy as np
from pathlib import Path


def convert_reference_data(input_path: str, output_path: str):
    """Convert reference data to use normalized velocity direction."""

    print(f"Loading reference data from: {input_path}")
    with open(input_path, "rb") as f:
        data = pickle.load(f)

    # Extract features
    if isinstance(data, dict) and "features" in data:
        features = np.array(data["features"])
    else:
        features = np.array(data)

    print(f"Original features shape: {features.shape}")

    feature_dim = features.shape[-1]

    # Determine feature layout based on dimension
    if feature_dim == 27:
        # 27-dim layout (8 joints):
        # - joint_pos: 0-7 (8 dims)
        # - joint_vel: 8-15 (8 dims)
        # - root_linvel: 16-18 (3 dims) ← NORMALIZE THIS
        # - root_angvel: 19-21 (3 dims)
        # - root_height: 22 (1 dim)
        # - foot_contacts: 23-26 (4 dims)
        linvel_start, linvel_end = 16, 19
        num_joints = 8
        print(f"Detected 27-dim features (8 joints)")
    elif feature_dim == 29:
        # 29-dim layout (9 joints):
        # - joint_pos: 0-8 (9 dims)
        # - joint_vel: 9-17 (9 dims)
        # - root_linvel: 18-20 (3 dims) ← NORMALIZE THIS
        # - root_angvel: 21-23 (3 dims)
        # - root_height: 24 (1 dim)
        # - foot_contacts: 25-28 (4 dims)
        linvel_start, linvel_end = 18, 21
        num_joints = 9
        print(f"Detected 29-dim features (9 joints)")
    else:
        print(f"⚠️  Unsupported feature dimension: {feature_dim}")
        print(f"   Expected 27 (8 joints) or 29 (9 joints)")
        return

    # Extract root linear velocity
    root_linvel = features[:, linvel_start:linvel_end].copy()

    # Print original velocity stats
    print(f"\nOriginal root_linvel stats:")
    print(f"  Mean: {np.mean(root_linvel, axis=0)}")
    print(f"  Std:  {np.std(root_linvel, axis=0)}")
    speed = np.linalg.norm(root_linvel, axis=1)
    print(f"  Speed mean: {speed.mean():.3f} m/s")
    print(f"  Speed std:  {speed.std():.3f} m/s")

    # Normalize to unit direction
    # SAFETY: When velocity is very small (standing), use zero vector
    # to avoid noisy normalized direction
    min_speed_threshold = 0.1  # m/s - below this, consider stationary
    linvel_norm = np.linalg.norm(root_linvel, axis=1, keepdims=True)

    # Only normalize if moving, else zero
    is_moving = linvel_norm > min_speed_threshold
    root_linvel_dir = np.where(
        is_moving,
        root_linvel / (linvel_norm + 1e-8),
        np.zeros_like(root_linvel)
    )

    # Count how many frames are stationary
    num_stationary = np.sum(~is_moving)
    print(f"  Stationary frames (speed < {min_speed_threshold} m/s): {num_stationary} ({100*num_stationary/len(root_linvel):.1f}%)")

    # Create new features with normalized velocity direction
    new_features = features.copy()
    new_features[:, linvel_start:linvel_end] = root_linvel_dir

    # Verify normalization
    new_norms = np.linalg.norm(new_features[:, linvel_start:linvel_end], axis=1)
    print(f"\nNormalized velocity direction:")
    print(f"  Norm mean: {new_norms.mean():.6f} (should be ~1.0)")
    print(f"  Norm std:  {new_norms.std():.6f} (should be ~0.0)")
    print(f"  Direction mean: {np.mean(new_features[:, linvel_start:linvel_end], axis=0)}")

    print(f"\nNew features shape: {new_features.shape}")

    # Save
    if isinstance(data, dict):
        data["features"] = new_features
        data["feature_dim"] = feature_dim
        data["velocity_normalized"] = True
        data["notes"] = f"root_linvel (indices {linvel_start}-{linvel_end-1}) normalized to unit direction"
        output_data = data
    else:
        output_data = {
            "features": new_features,
            "feature_dim": feature_dim,
            "velocity_normalized": True,
        }

    with open(output_path, "wb") as f:
        pickle.dump(output_data, f)

    print(f"\n✓ Saved to: {output_path}")


if __name__ == "__main__":
    input_path = "playground_amp/data/walking_motions_merged.pkl"
    output_path = "playground_amp/data/walking_motions_normalized_vel.pkl"

    if not Path(input_path).exists():
        print(f"Error: {input_path} not found")
        exit(1)

    convert_reference_data(input_path, output_path)

    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("=" * 60)
    print("1. Update ppo_amass_training.yaml:")
    print(f"   amp.dataset_path: {output_path}")
    print("2. Verify robot_config.yaml:")
    print("   dimensions.amp_feature_dim: 29")
    print("=" * 60)
