#!/usr/bin/env python3
"""Convert reference motion data to use normalized velocity direction.

The AMP discriminator now uses 29-dim features with normalized velocity
direction (unit vector) instead of raw velocity. This preserves directional
information while removing speed magnitude that causes mismatch.

This script normalizes root_linvel (indices 18-20) to unit direction.

Usage:
    python scripts/convert_ref_data_normalized_velocity.py
"""

import pickle
import numpy as np
from pathlib import Path


def convert_reference_data(input_path: str, output_path: str):
    """Convert 29-dim reference data to use normalized velocity direction."""

    print(f"Loading reference data from: {input_path}")
    with open(input_path, "rb") as f:
        data = pickle.load(f)

    # Extract features
    if isinstance(data, dict) and "features" in data:
        features = np.array(data["features"])
    else:
        features = np.array(data)

    print(f"Original features shape: {features.shape}")

    if features.shape[-1] != 29:
        print(f"⚠️  Expected 29-dim features, got {features.shape[-1]}-dim")
        return

    # 29-dim layout:
    # - joint_pos: 0-8 (9 dims)
    # - joint_vel: 9-17 (9 dims)
    # - root_linvel: 18-20 (3 dims) ← NORMALIZE THIS
    # - root_angvel: 21-23 (3 dims)
    # - root_height: 24 (1 dim)
    # - foot_contacts: 25-28 (4 dims)

    # Extract root linear velocity
    root_linvel = features[:, 18:21].copy()

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
    new_features[:, 18:21] = root_linvel_dir

    # Verify normalization
    new_norms = np.linalg.norm(new_features[:, 18:21], axis=1)
    print(f"\nNormalized velocity direction:")
    print(f"  Norm mean: {new_norms.mean():.6f} (should be ~1.0)")
    print(f"  Norm std:  {new_norms.std():.6f} (should be ~0.0)")
    print(f"  Direction mean: {np.mean(new_features[:, 18:21], axis=0)}")

    print(f"\nNew features shape: {new_features.shape}")

    # Save
    if isinstance(data, dict):
        data["features"] = new_features
        data["feature_dim"] = 29
        data["velocity_normalized"] = True
        data["notes"] = "root_linvel (indices 18-20) normalized to unit direction"
        output_data = data
    else:
        output_data = {
            "features": new_features,
            "feature_dim": 29,
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
