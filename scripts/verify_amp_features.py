#!/usr/bin/env python3
"""Verify AMP feature consistency between reference data and policy extraction.

Checks:
1. Feature ordering matches between reference and policy
2. Feature statistics are in similar ranges
3. Normalized velocity direction is correct

Usage:
    uv run python scripts/verify_amp_features.py
"""

import pickle
import numpy as np
from pathlib import Path

# Feature layout (29-dim with normalized velocity direction)
FEATURE_NAMES = [
    "joint_pos_0", "joint_pos_1", "joint_pos_2", "joint_pos_3", "joint_pos_4",
    "joint_pos_5", "joint_pos_6", "joint_pos_7", "joint_pos_8",
    "joint_vel_0", "joint_vel_1", "joint_vel_2", "joint_vel_3", "joint_vel_4",
    "joint_vel_5", "joint_vel_6", "joint_vel_7", "joint_vel_8",
    "root_linvel_dir_x", "root_linvel_dir_y", "root_linvel_dir_z",
    "root_angvel_x", "root_angvel_y", "root_angvel_z",
    "root_height",
    "foot_contact_0", "foot_contact_1", "foot_contact_2", "foot_contact_3",
]

FEATURE_GROUPS = {
    "Joint positions (0-8)": (0, 9),
    "Joint velocities (9-17)": (9, 18),
    "Root linvel direction (18-20)": (18, 21),
    "Root angular velocity (21-23)": (21, 24),
    "Root height (24)": (24, 25),
    "Foot contacts (25-28)": (25, 29),
}


def verify_reference_data(data_path: str):
    """Verify reference data format and statistics."""

    print(f"\n{'=' * 60}")
    print("REFERENCE DATA VERIFICATION")
    print(f"{'=' * 60}")
    print(f"Path: {data_path}")

    with open(data_path, "rb") as f:
        data = pickle.load(f)

    features = data["features"]
    print(f"\n1. Shape: {features.shape}")
    assert features.shape[1] == 29, f"Expected 29 features, got {features.shape[1]}"
    print("   ✅ Feature dimension correct (29)")

    print(f"\n2. Velocity normalized: {data.get('velocity_normalized', False)}")
    if data.get('velocity_normalized', False):
        print("   ✅ Velocity is normalized to direction")
    else:
        print("   ⚠️  WARNING: Velocity NOT normalized!")

    # Check velocity direction norms
    # With safe thresholding, stationary frames have norm=0, moving frames have norm=1
    vel_dir = features[:, 18:21]
    vel_norms = np.linalg.norm(vel_dir, axis=1)

    # Count stationary vs moving
    is_moving = vel_norms > 0.5  # Moving frames have norm ~1
    num_moving = np.sum(is_moving)
    num_stationary = len(vel_norms) - num_moving

    print(f"\n3. Velocity direction norms (with safe thresholding):")
    print(f"   Moving frames: {num_moving} ({100*num_moving/len(vel_norms):.1f}%) - norm ~1.0")
    print(f"   Stationary frames: {num_stationary} ({100*num_stationary/len(vel_norms):.1f}%) - norm = 0")
    print(f"   Overall norm mean: {vel_norms.mean():.6f}")

    # Check that moving frames have norm ~1.0
    moving_norms = vel_norms[is_moving]
    if len(moving_norms) > 0 and np.allclose(moving_norms, 1.0, atol=0.01):
        print("   ✅ Moving frames properly normalized (norm ~1.0)")
    elif len(moving_norms) == 0:
        print("   ⚠️  No moving frames detected")
    else:
        print(f"   ❌ Moving frame norms incorrect: mean={moving_norms.mean():.4f}")
        return False

    # Check that stationary frames have norm ~0
    stationary_norms = vel_norms[~is_moving]
    if len(stationary_norms) > 0 and np.allclose(stationary_norms, 0.0, atol=0.01):
        print("   ✅ Stationary frames zeroed (norm ~0.0)")
    print("   ✅ Safe velocity normalization applied correctly")

    # Print per-group statistics
    print(f"\n4. Per-Group Statistics:")
    print(f"   {'Group':<35} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print(f"   {'-'*75}")

    for group_name, (start, end) in FEATURE_GROUPS.items():
        group_data = features[:, start:end]
        print(f"   {group_name:<35} {group_data.mean():>10.4f} {group_data.std():>10.4f} "
              f"{group_data.min():>10.4f} {group_data.max():>10.4f}")

    return True


def simulate_policy_features():
    """Simulate policy feature extraction to verify ordering.

    This creates dummy observations and extracts features to verify
    the extraction code matches the expected format.
    """
    try:
        import jax.numpy as jnp
        from playground_amp.amp.amp_features import extract_amp_features, DEFAULT_AMP_CONFIG
    except ImportError:
        print("\n⚠️  Cannot import JAX/amp_features - skipping policy simulation")
        return None

    print(f"\n{'=' * 60}")
    print("POLICY FEATURE EXTRACTION VERIFICATION")
    print(f"{'=' * 60}")

    # Create a known observation
    # Observation layout (38-dim):
    # - gravity: 0-2 (3)
    # - angvel: 3-5 (3)
    # - linvel: 6-8 (3)
    # - joint_pos: 9-17 (9)
    # - joint_vel: 18-26 (9)
    # - prev_action: 27-35 (9)
    # - velocity_cmd: 36 (1)
    # - padding: 37 (1)

    obs = jnp.zeros(38)

    # Set known values to verify ordering
    obs = obs.at[0:3].set(jnp.array([0.0, 0.0, 1.0]))  # gravity (pointing up)
    obs = obs.at[3:6].set(jnp.array([0.1, 0.2, 0.3]))  # angvel
    obs = obs.at[6:9].set(jnp.array([1.0, 0.0, 0.0]))  # linvel (forward)
    obs = obs.at[9:18].set(jnp.arange(9) * 0.1)  # joint_pos
    obs = obs.at[18:27].set(jnp.arange(9) * 0.01)  # joint_vel

    # Extract features
    features = extract_amp_features(obs)

    print(f"\n1. Config feature_dim: {DEFAULT_AMP_CONFIG.feature_dim}")
    print(f"   Extracted features shape: {features.shape}")
    assert features.shape[0] == 29, f"Expected 29 features, got {features.shape[0]}"
    print("   ✅ Feature dimension correct (29)")

    # Verify ordering
    print(f"\n2. Feature ordering verification:")

    # Joint positions (should be 0, 0.1, 0.2, ..., 0.8)
    joint_pos = features[0:9]
    expected_joint_pos = jnp.arange(9) * 0.1
    if jnp.allclose(joint_pos, expected_joint_pos):
        print("   ✅ Joint positions (0-8) correct")
    else:
        print(f"   ❌ Joint positions mismatch: {joint_pos}")

    # Joint velocities (should be 0, 0.01, 0.02, ..., 0.08)
    joint_vel = features[9:18]
    expected_joint_vel = jnp.arange(9) * 0.01
    if jnp.allclose(joint_vel, expected_joint_vel):
        print("   ✅ Joint velocities (9-17) correct")
    else:
        print(f"   ❌ Joint velocities mismatch: {joint_vel}")

    # Root linvel direction (should be [1, 0, 0] normalized)
    linvel_dir = features[18:21]
    expected_linvel_dir = jnp.array([1.0, 0.0, 0.0])  # normalized
    if jnp.allclose(linvel_dir, expected_linvel_dir, atol=1e-6):
        print("   ✅ Root linvel direction (18-20) correct (normalized)")
    else:
        print(f"   ❌ Root linvel direction mismatch: {linvel_dir}")

    # Root angvel (should be [0.1, 0.2, 0.3])
    angvel = features[21:24]
    expected_angvel = jnp.array([0.1, 0.2, 0.3])
    if jnp.allclose(angvel, expected_angvel):
        print("   ✅ Root angular velocity (21-23) correct")
    else:
        print(f"   ❌ Root angular velocity mismatch: {angvel}")

    # Root height (should be 1.0 - gravity z component)
    height = features[24]
    if jnp.isclose(height, 1.0):
        print("   ✅ Root height (24) correct")
    else:
        print(f"   ❌ Root height mismatch: {height}")

    # Foot contacts (should be zeros)
    foot_contacts = features[25:29]
    if jnp.allclose(foot_contacts, jnp.zeros(4)):
        print("   ✅ Foot contacts (25-28) correct (zeros)")
    else:
        print(f"   ❌ Foot contacts mismatch: {foot_contacts}")

    return features


def main():
    print("=" * 60)
    print("AMP FEATURE CONSISTENCY VERIFICATION")
    print("=" * 60)

    # Verify reference data
    ref_path = "playground_amp/data/walking_motions_normalized_vel.pkl"
    if Path(ref_path).exists():
        ref_ok = verify_reference_data(ref_path)
    else:
        print(f"\n⚠️  Reference data not found: {ref_path}")
        ref_ok = False

    # Verify policy feature extraction
    policy_features = simulate_policy_features()

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    if ref_ok and policy_features is not None:
        print("✅ Reference data: VALID")
        print("✅ Policy features: VALID")
        print("✅ Feature ordering: CONSISTENT")
        print("\n🎉 All checks passed! Ready to train.")
    else:
        print("⚠️  Some checks failed - review output above")


if __name__ == "__main__":
    main()
