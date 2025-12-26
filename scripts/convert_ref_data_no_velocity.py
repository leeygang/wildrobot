#!/usr/bin/env python3
"""Convert reference motion data to exclude ALL velocity features.

The AMP discriminator should only learn from position-based features because:
- GMR velocities are computed from finite differences of mocap
- Policy velocities come from MuJoCo qvel
- These are fundamentally different sources, creating distribution mismatch

This script removes ALL velocity features:
- joint_vel (finite diff of joint positions)
- root_linvel (finite diff of root position)
- root_angvel (finite diff of root orientation)

Keeping only:
- joint_pos (positions from mocap/retargeting)
- root_height (from mocap)
- foot_contacts (from FK estimation)

Usage:
    python scripts/convert_ref_data_no_velocity.py
"""

import pickle
import numpy as np
from pathlib import Path


def convert_reference_data(input_path: str, output_path: str):
    """Convert reference data to position-only features (no velocities)."""

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

    if feature_dim == 27:
        # 27-dim layout (8 joints):
        # - joint_pos: 0-7 (8 dims) ✅ keep
        # - joint_vel: 8-15 (8 dims) ❌ remove
        # - root_linvel: 16-18 (3 dims) ❌ remove
        # - root_angvel: 19-21 (3 dims) ❌ remove
        # - root_height: 22 (1 dim) ✅ keep
        # - foot_contacts: 23-26 (4 dims) ✅ keep
        num_joints = 8
        joint_pos_idx = list(range(0, 8))           # 8 dims
        root_height_idx = [22]                       # 1 dim
        foot_contacts_idx = list(range(23, 27))     # 4 dims
        keep_indices = joint_pos_idx + root_height_idx + foot_contacts_idx
        expected_output_dim = 13  # 8 + 1 + 4

        # Velocity indices being removed (for stats)
        joint_vel_idx = list(range(8, 16))
        root_linvel_idx = list(range(16, 19))
        root_angvel_idx = list(range(19, 22))

    elif feature_dim == 29:
        # 29-dim layout (9 joints):
        # - joint_pos: 0-8 (9 dims) ✅ keep
        # - joint_vel: 9-17 (9 dims) ❌ remove
        # - root_linvel: 18-20 (3 dims) ❌ remove
        # - root_angvel: 21-23 (3 dims) ❌ remove
        # - root_height: 24 (1 dim) ✅ keep
        # - foot_contacts: 25-28 (4 dims) ✅ keep
        num_joints = 9
        joint_pos_idx = list(range(0, 9))           # 9 dims
        root_height_idx = [24]                       # 1 dim
        foot_contacts_idx = list(range(25, 29))     # 4 dims
        keep_indices = joint_pos_idx + root_height_idx + foot_contacts_idx
        expected_output_dim = 14  # 9 + 1 + 4

        # Velocity indices being removed (for stats)
        joint_vel_idx = list(range(9, 18))
        root_linvel_idx = list(range(18, 21))
        root_angvel_idx = list(range(21, 24))

    else:
        print(f"❌ Unsupported feature dimension: {feature_dim}")
        print("   Expected 27-dim (8 joints) or 29-dim (9 joints)")
        return

    # Extract new features
    new_features = features[:, keep_indices]

    print(f"New features shape: {new_features.shape}")
    print(f"Removed: joint_vel ({num_joints}d), root_linvel (3d), root_angvel (3d)")

    # Verify the conversion
    assert new_features.shape[-1] == expected_output_dim, \
        f"Expected {expected_output_dim}-dim, got {new_features.shape[-1]}"

    # New layout (for 8 joints = 13-dim):
    # - joint_pos: 0-7 (8 dims)
    # - root_height: 8 (1 dim)
    # - foot_contacts: 9-12 (4 dims)
    print(f"\nNew {expected_output_dim}-dim layout ({num_joints} joints):")
    print(f"  joint_pos: 0-{num_joints-1} ({num_joints} dims)")
    print(f"  root_height: {num_joints} (1 dim)")
    print(f"  foot_contacts: {num_joints+1}-{num_joints+4} (4 dims)")

    print("\n✓ Conversion verified")

    # Print velocity stats that were removed (for reference)
    print("\n" + "-" * 40)
    print("Removed velocity statistics:")
    print("-" * 40)

    removed_joint_vel = features[:, joint_vel_idx]
    print(f"joint_vel ({num_joints}d):")
    print(f"  Mean: {np.mean(removed_joint_vel):.4f}")
    print(f"  Std:  {np.std(removed_joint_vel):.4f}")
    print(f"  Range: [{np.min(removed_joint_vel):.4f}, {np.max(removed_joint_vel):.4f}]")

    removed_linvel = features[:, root_linvel_idx]
    print(f"root_linvel (3d):")
    print(f"  Mean: {np.mean(removed_linvel, axis=0)}")
    print(f"  Std:  {np.std(removed_linvel, axis=0)}")

    removed_angvel = features[:, root_angvel_idx]
    print(f"root_angvel (3d):")
    print(f"  Mean: {np.mean(removed_angvel, axis=0)}")
    print(f"  Std:  {np.std(removed_angvel, axis=0)}")

    # Save
    if isinstance(data, dict):
        data["features"] = new_features
        data["feature_dim"] = expected_output_dim
        data["removed_features"] = [
            f"joint_vel (indices {joint_vel_idx[0]}-{joint_vel_idx[-1]})",
            f"root_linvel (indices {root_linvel_idx[0]}-{root_linvel_idx[-1]})",
            f"root_angvel (indices {root_angvel_idx[0]}-{root_angvel_idx[-1]})"
        ]
        data["kept_features"] = ["joint_pos", "root_height", "foot_contacts"]
        output_data = data
    else:
        output_data = new_features

    with open(output_path, "wb") as f:
        pickle.dump(output_data, f)

    print(f"\n✓ Saved to: {output_path}")

    return expected_output_dim


if __name__ == "__main__":
    input_path = "playground_amp/data/walking_motions_merged.pkl"
    output_path = "playground_amp/data/walking_motions_no_vel.pkl"

    if not Path(input_path).exists():
        print(f"Error: {input_path} not found")
        exit(1)

    output_dim = convert_reference_data(input_path, output_path)

    if output_dim:
        print("\n" + "=" * 60)
        print("NEXT STEPS:")
        print("=" * 60)
        print("1. Update ppo_amass_training.yaml:")
        print(f"   amp.dataset_path: {output_path}")
        print("2. Update robot_config.yaml:")
        print(f"   dimensions.amp_feature_dim: {output_dim}")
        print("=" * 60)
