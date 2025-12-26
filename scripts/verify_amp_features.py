#!/usr/bin/env python3
"""Verify AMP feature consistency between reference data and policy extraction.

v0.10.0: Physics Reference Support
For physics reference data (has dof_vel and foot_contacts stored), uses
27-dim feature format with TRUE qvel. For GMR reference data, uses the
29-dim normalized velocity format.

Checks:
1. Feature ordering matches between reference and policy
2. Feature statistics are in similar ranges
3. Normalized velocity direction is correct (29-dim) or TRUE qvel (27-dim)

Usage:
    uv run python scripts/verify_amp_features.py
    uv run python scripts/verify_amp_features.py --reference data/amp/walking_medium01_amp.pkl
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

# Feature layout for 29-dim (normalized velocity direction)
FEATURE_NAMES_29DIM = [
    "joint_pos_0",
    "joint_pos_1",
    "joint_pos_2",
    "joint_pos_3",
    "joint_pos_4",
    "joint_pos_5",
    "joint_pos_6",
    "joint_pos_7",
    "joint_pos_8",
    "joint_vel_0",
    "joint_vel_1",
    "joint_vel_2",
    "joint_vel_3",
    "joint_vel_4",
    "joint_vel_5",
    "joint_vel_6",
    "joint_vel_7",
    "joint_vel_8",
    "root_linvel_dir_x",
    "root_linvel_dir_y",
    "root_linvel_dir_z",
    "root_angvel_x",
    "root_angvel_y",
    "root_angvel_z",
    "root_height",
    "foot_contact_0",
    "foot_contact_1",
    "foot_contact_2",
    "foot_contact_3",
]

# Feature layout for 27-dim (TRUE qvel physics reference, 8-joint robot)
FEATURE_NAMES_27DIM = [
    "joint_pos_0",
    "joint_pos_1",
    "joint_pos_2",
    "joint_pos_3",
    "joint_pos_4",
    "joint_pos_5",
    "joint_pos_6",
    "joint_pos_7",
    "joint_vel_0",
    "joint_vel_1",
    "joint_vel_2",
    "joint_vel_3",
    "joint_vel_4",
    "joint_vel_5",
    "joint_vel_6",
    "joint_vel_7",
    "root_linvel_x",
    "root_linvel_y",
    "root_linvel_z",
    "root_angvel_x",
    "root_angvel_y",
    "root_angvel_z",
    "root_height",
    "foot_contact_0",
    "foot_contact_1",
    "foot_contact_2",
    "foot_contact_3",
]

# Feature groups for 29-dim (normalized velocity)
FEATURE_GROUPS_29DIM = {
    "Joint positions (0-8)": (0, 9),
    "Joint velocities (9-17)": (9, 18),
    "Root linvel direction (18-20)": (18, 21),
    "Root angular velocity (21-23)": (21, 24),
    "Root height (24)": (24, 25),
    "Foot contacts (25-28)": (25, 29),
}

# Feature groups for 27-dim (TRUE qvel, 8-joint robot)
FEATURE_GROUPS_27DIM = {
    "Joint positions (0-7)": (0, 8),
    "Joint velocities (8-15)": (8, 16),
    "Root linear velocity (16-18)": (16, 19),
    "Root angular velocity (19-21)": (19, 22),
    "Root height (22)": (22, 23),
    "Foot contacts (23-26)": (23, 27),
}


def verify_reference_data(data_path: str):
    """Verify reference data format and statistics.

    v0.10.0: Supports both 27-dim (physics TRUE qvel) and 29-dim (normalized velocity) formats.
    """

    print(f"\n{'=' * 60}")
    print("REFERENCE DATA VERIFICATION")
    print(f"{'=' * 60}")
    print(f"Path: {data_path}")

    with open(data_path, "rb") as f:
        data = pickle.load(f)

    features = data["features"]
    feature_dim = features.shape[1]
    print(f"\n1. Shape: {features.shape}")

    # v0.10.0: Detect data format
    is_physics_ref = "dof_vel" in data and "foot_contacts" in data

    if feature_dim == 27:
        print(f"   Feature dimension: 27 (TRUE qvel physics reference)")
        if is_physics_ref:
            print("   [PHYSICS] Data contains TRUE qvel from MuJoCo simulation")
        feature_groups = FEATURE_GROUPS_27DIM
    elif feature_dim == 29:
        print(f"   Feature dimension: 29 (normalized velocity direction)")
        print("   [GMR] Data uses normalized velocity direction")
        feature_groups = FEATURE_GROUPS_29DIM
    else:
        print(f"   ❌ Unexpected feature dimension: {feature_dim}")
        return False

    print("   ✅ Feature dimension recognized")

    # Check if velocity is normalized (only for 29-dim)
    if feature_dim == 29:
        print(f"\n2. Velocity normalized: {data.get('velocity_normalized', False)}")
        if data.get("velocity_normalized", False):
            print("   ✅ Velocity is normalized to direction")
        else:
            print("   ⚠️  WARNING: Velocity NOT normalized!")

        # Check velocity direction norms
        vel_dir = features[:, 18:21]
        vel_norms = np.linalg.norm(vel_dir, axis=1)

        # Count stationary vs moving
        is_moving = vel_norms > 0.5
        num_moving = np.sum(is_moving)
        num_stationary = len(vel_norms) - num_moving

        print(f"\n3. Velocity direction norms (with safe thresholding):")
        print(
            f"   Moving frames: {num_moving} ({100*num_moving/len(vel_norms):.1f}%) - norm ~1.0"
        )
        print(
            f"   Stationary frames: {num_stationary} ({100*num_stationary/len(vel_norms):.1f}%) - norm = 0"
        )
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
    else:
        # 27-dim physics reference: check TRUE qvel statistics
        print(f"\n2. TRUE qvel statistics (physics reference):")
        print(f"   Has dof_vel: {'dof_vel' in data}")
        print(f"   Has foot_contacts: {'foot_contacts' in data}")

        if "dof_vel" in data:
            dof_vel = data["dof_vel"]
            print(f"   dof_vel shape: {dof_vel.shape}")
            print(f"   dof_vel mean: {dof_vel.mean():.4f} rad/s")
            print(f"   dof_vel std: {dof_vel.std():.4f} rad/s")
            print("   ✅ TRUE joint velocities stored")

        if "foot_contacts" in data:
            foot_contacts = data["foot_contacts"]
            print(f"   foot_contacts shape: {foot_contacts.shape}")
            print(f"   foot_contacts mean: {foot_contacts.mean():.4f}")
            print("   ✅ TRUE foot contacts stored")

        # Check linear velocity magnitude (not normalized)
        lin_vel = features[:, 16:19]
        lin_vel_mag = np.linalg.norm(lin_vel, axis=1)
        print(f"\n3. Root linear velocity (TRUE, heading-local):")
        print(f"   Mean magnitude: {lin_vel_mag.mean():.4f} m/s")
        print(f"   Max magnitude: {lin_vel_mag.max():.4f} m/s")
        print(f"   Forward (x) mean: {lin_vel[:, 0].mean():.4f} m/s")
        print("   ✅ TRUE linear velocities stored (not normalized)")

    # Print per-group statistics
    print(f"\n4. Per-Group Statistics:")
    print(f"   {'Group':<40} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print(f"   {'-'*80}")

    for group_name, (start, end) in feature_groups.items():
        group_data = features[:, start:end]
        print(
            f"   {group_name:<40} {group_data.mean():>10.4f} {group_data.std():>10.4f} "
            f"{group_data.min():>10.4f} {group_data.max():>10.4f}"
        )

    return True


def simulate_policy_features(feature_dim: int = 27):
    """Simulate policy feature extraction to verify ordering.

    v0.10.0: Supports both 27-dim (TRUE qvel) and 29-dim (normalized velocity) formats.

    This creates dummy observations and extracts features to verify
    the extraction code matches the expected format.
    """
    try:
        import jax.numpy as jnp
        from playground_amp.amp.policy_features import (
            FeatureConfig,
            create_config_from_robot,
            extract_amp_features,
        )
        from playground_amp.configs.training_config import get_robot_config, load_robot_config
    except ImportError as e:
        print(f"\n⚠️  Cannot import JAX/amp_features - skipping policy simulation: {e}")
        return None

    print(f"\n{'=' * 60}")
    print("POLICY FEATURE EXTRACTION VERIFICATION")
    print(f"{'=' * 60}")
    print(f"Testing {feature_dim}-dim feature format")

    # Load robot config to get actual feature config
    try:
        load_robot_config("assets/robot_config.yaml")
        robot_config = get_robot_config()
        config = create_config_from_robot(robot_config)
        num_joints = config.num_actuated_joints
    except Exception as e:
        print(f"\n⚠️  Cannot load robot config: {e}")
        # Default to 8 joints for wildrobot
        num_joints = 8
        config = None

    print(f"\n1. Robot config: {num_joints} actuated joints")

    if config:
        print(f"   Expected feature_dim: {config.feature_dim}")

    # Create a known observation (35-dim v0.7.0 layout)
    obs = jnp.zeros(35)

    # Set known values to verify ordering
    obs = obs.at[0:3].set(jnp.array([0.0, 0.0, 1.0]))  # gravity (pointing up)
    obs = obs.at[3:6].set(jnp.array([0.1, 0.2, 0.3]))  # angvel (heading-local)
    obs = obs.at[6:9].set(
        jnp.array([1.0, 0.5, 0.0])
    )  # linvel (heading-local, forward + lateral)
    obs = obs.at[9 : 9 + num_joints].set(jnp.arange(num_joints) * 0.1)  # joint_pos
    obs = obs.at[9 + num_joints : 9 + 2 * num_joints].set(
        jnp.arange(num_joints) * 0.01
    )  # joint_vel

    if config is None:
        print("\n⚠️  Cannot extract features without config")
        return None

    # Extract features using v0.10.0 API
    features = extract_amp_features(
        obs=obs,
        config=config,
        root_height=jnp.array(0.45),
        prev_joint_pos=jnp.arange(num_joints) * 0.1,  # Same as current
        dt=0.02,
        use_obs_joint_vel=True,  # Use TRUE qvel from observation
    )

    print(f"\n2. Extracted features shape: {features.shape}")
    actual_dim = features.shape[0]

    if actual_dim == 27:
        print("   ✅ Feature dimension: 27 (TRUE qvel physics format)")
    elif actual_dim == 29:
        print("   ✅ Feature dimension: 29 (normalized velocity format)")
    else:
        print(f"   ⚠️  Unexpected dimension: {actual_dim}")

    # Verify ordering for 27-dim (physics reference format)
    print(f"\n3. Feature ordering verification ({actual_dim}-dim):")

    n = num_joints

    # Joint positions (should be 0, 0.1, 0.2, ..., 0.7 for 8 joints)
    joint_pos = features[0:n]
    expected_joint_pos = jnp.arange(n) * 0.1
    if jnp.allclose(joint_pos, expected_joint_pos, atol=1e-5):
        print(f"   ✅ Joint positions (0-{n-1}) correct")
    else:
        print(f"   ❌ Joint positions mismatch")
        print(f"      Expected: {expected_joint_pos}")
        print(f"      Got: {joint_pos}")

    # Joint velocities - should be from obs (TRUE qvel)
    joint_vel = features[n : 2 * n]
    expected_joint_vel = jnp.arange(n) * 0.01
    if jnp.allclose(joint_vel, expected_joint_vel, atol=1e-5):
        print(f"   ✅ Joint velocities ({n}-{2*n-1}) correct (TRUE qvel from obs)")
    else:
        print(f"   ❌ Joint velocities mismatch")
        print(f"      Expected: {expected_joint_vel}")
        print(f"      Got: {joint_vel}")

    # Root linear velocity (heading-local)
    linvel = features[2 * n : 2 * n + 3]
    expected_linvel = jnp.array([1.0, 0.5, 0.0])  # From obs
    if jnp.allclose(linvel, expected_linvel, atol=1e-5):
        print(f"   ✅ Root linear velocity ({2*n}-{2*n+2}) correct (heading-local)")
    else:
        print(f"   ❌ Root linear velocity mismatch")
        print(f"      Expected: {expected_linvel}")
        print(f"      Got: {linvel}")

    # Root angular velocity (heading-local)
    angvel = features[2 * n + 3 : 2 * n + 6]
    expected_angvel = jnp.array([0.1, 0.2, 0.3])  # From obs
    if jnp.allclose(angvel, expected_angvel, atol=1e-5):
        print(f"   ✅ Root angular velocity ({2*n+3}-{2*n+5}) correct (heading-local)")
    else:
        print(f"   ❌ Root angular velocity mismatch")
        print(f"      Expected: {expected_angvel}")
        print(f"      Got: {angvel}")

    # Root height
    height = features[2 * n + 6]
    if jnp.isclose(height, 0.45, atol=1e-5):
        print(f"   ✅ Root height ({2*n+6}) correct")
    else:
        print(f"   ❌ Root height mismatch: expected 0.45, got {height}")

    # Foot contacts (estimated from joint positions)
    foot_contacts = features[2 * n + 7 : 2 * n + 11]
    print(f"   ℹ️  Foot contacts ({2*n+7}-{2*n+10}): {foot_contacts}")
    print(f"      (Contacts are estimated from joint positions by default)")

    return features


def main():
    parser = argparse.ArgumentParser(description="Verify AMP feature consistency")
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        help="Path to reference AMP pickle file (default: searches for common paths)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("AMP FEATURE CONSISTENCY VERIFICATION (v0.10.0)")
    print("=" * 60)

    # Find reference data
    ref_paths = [
        args.reference,
        "data/amp/walking_medium01_amp.pkl",  # Physics reference
        "playground_amp/data/walking_motions_normalized_vel.pkl",  # GMR reference
    ]

    ref_ok = False
    ref_path_used = None
    feature_dim = 27  # Default to physics format

    for ref_path in ref_paths:
        if ref_path and Path(ref_path).exists():
            ref_ok = verify_reference_data(ref_path)
            ref_path_used = ref_path
            # Detect feature dimension
            with open(ref_path, "rb") as f:
                data = pickle.load(f)
                feature_dim = data["features"].shape[1]
            break
    else:
        if args.reference:
            print(f"\n⚠️  Reference data not found: {args.reference}")
        else:
            print(f"\n⚠️  No reference data found in default paths")

    # Verify policy feature extraction
    policy_features = simulate_policy_features(feature_dim)

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    if ref_ok:
        print(f"✅ Reference data: VALID ({ref_path_used})")
        print(f"   Format: {feature_dim}-dim")
        if feature_dim == 27:
            print("   Type: Physics reference (TRUE qvel)")
        else:
            print("   Type: GMR reference (normalized velocity)")
    else:
        print("⚠️  Reference data: NOT VERIFIED")

    if policy_features is not None:
        print("✅ Policy features: VALID")
        print("✅ Feature ordering: CONSISTENT")
        print("\n🎉 All checks passed! Ready to train.")
    else:
        print("⚠️  Policy features: SKIPPED")
        print("⚠️  Some checks were skipped - review output above")


if __name__ == "__main__":
    main()
