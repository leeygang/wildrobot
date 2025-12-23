#!/usr/bin/env python3
"""Convert reference motion data to exclude root linear velocity.

The AMP discriminator now uses 26-dim features (without root_linvel)
to avoid penalizing the policy for following velocity commands.

This script removes root_linvel (indices 18-20) from existing 29-dim data.

Usage:
    python scripts/convert_ref_data_no_velocity.py
"""

import pickle
import numpy as np
from pathlib import Path


def convert_reference_data(input_path: str, output_path: str):
    """Convert 29-dim reference data to 26-dim (no root_linvel)."""

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
        if features.shape[-1] == 26:
            print("Already 26-dim, nothing to do!")
            return

    # Original 29-dim layout:
    # - joint_pos: 0-8 (9 dims) ✅ keep
    # - joint_vel: 9-17 (9 dims) ✅ keep
    # - root_linvel: 18-20 (3 dims) ❌ REMOVE
    # - root_angvel: 21-23 (3 dims) ✅ keep
    # - root_height: 24 (1 dim) ✅ keep
    # - foot_contacts: 25-28 (4 dims) ✅ keep

    # New 26-dim layout:
    # - joint_pos: 0-8 (9 dims)
    # - joint_vel: 9-17 (9 dims)
    # - root_angvel: 18-20 (3 dims)
    # - root_height: 21 (1 dim)
    # - foot_contacts: 22-25 (4 dims)

    # Remove indices 18, 19, 20 (root_linvel)
    keep_indices = list(range(0, 18)) + list(range(21, 29))
    new_features = features[:, keep_indices]

    print(f"New features shape: {new_features.shape}")

    # Verify the conversion
    assert new_features.shape[-1] == 26, f"Expected 26-dim, got {new_features.shape[-1]}"

    # Verify specific components match
    np.testing.assert_array_equal(new_features[:, 0:18], features[:, 0:18],
                                   err_msg="Joint pos/vel mismatch")
    np.testing.assert_array_equal(new_features[:, 18:21], features[:, 21:24],
                                   err_msg="Root angvel mismatch")
    np.testing.assert_array_equal(new_features[:, 21:22], features[:, 24:25],
                                   err_msg="Root height mismatch")
    np.testing.assert_array_equal(new_features[:, 22:26], features[:, 25:29],
                                   err_msg="Foot contacts mismatch")

    print("✓ Conversion verified")

    # Save
    if isinstance(data, dict):
        data["features"] = new_features
        data["feature_dim"] = 26
        data["removed_features"] = ["root_linvel (indices 18-20)"]
        output_data = data
    else:
        output_data = new_features

    with open(output_path, "wb") as f:
        pickle.dump(output_data, f)

    print(f"✓ Saved to: {output_path}")

    # Print velocity stats that were removed (for reference)
    removed_linvel = features[:, 18:21]
    print(f"\nRemoved root_linvel stats:")
    print(f"  Mean: {np.mean(removed_linvel, axis=0)}")
    print(f"  Std:  {np.std(removed_linvel, axis=0)}")


if __name__ == "__main__":
    input_path = "playground_amp/data/walking_motions_merged.pkl"
    output_path = "playground_amp/data/walking_motions_merged_no_vel.pkl"

    if not Path(input_path).exists():
        print(f"Error: {input_path} not found")
        exit(1)

    convert_reference_data(input_path, output_path)

    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("=" * 60)
    print("1. Update ppo_amass_training.yaml:")
    print(f"   amp.dataset_path: {output_path}")
    print("2. Update robot_config.yaml:")
    print("   dimensions.amp_feature_dim: 26")
    print("=" * 60)
