#!/usr/bin/env python3
"""Validate that reference data and policy feature extraction produce IDENTICAL features.

This script verifies end-to-end consistency by:
1. Creating synthetic observations with known values
2. Extracting features via policy code (JAX)
3. Computing what reference data conversion would produce (NumPy)
4. Comparing the two to ensure they are IDENTICAL

This catches any discrepancy in:
- Feature ordering
- Velocity normalization logic
- Safe thresholding behavior
- Numerical precision

Usage:
    uv run python scripts/validate_feature_consistency.py
"""

import sys
import numpy as np
import pickle
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def compute_reference_style_features(obs: np.ndarray) -> np.ndarray:
    """Compute AMP features using the same logic as reference data conversion.

    This replicates the NumPy logic from convert_ref_data_normalized_velocity.py
    to ensure reference data and policy extraction are consistent.

    Args:
        obs: Observation array (38-dim) or batch (N, 38)

    Returns:
        AMP features (29-dim) or batch (N, 29)
    """
    single = obs.ndim == 1
    if single:
        obs = obs[np.newaxis, :]

    # Observation layout (38-dim):
    # - gravity: 0-2 (3)
    # - angvel: 3-5 (3)
    # - linvel: 6-8 (3)
    # - joint_pos: 9-17 (9)
    # - joint_vel: 18-26 (9)
    # - prev_action: 27-35 (9)
    # - velocity_cmd: 36 (1)
    # - padding: 37 (1)

    joint_pos = obs[:, 9:18]      # 9 dims
    joint_vel = obs[:, 18:27]     # 9 dims
    root_linvel = obs[:, 6:9]     # 3 dims
    root_angvel = obs[:, 3:6]     # 3 dims
    root_height = obs[:, 2:3]     # 1 dim (gravity z)

    # Normalize velocity to direction (same logic as conversion script)
    min_speed_threshold = 0.1
    linvel_norm = np.linalg.norm(root_linvel, axis=1, keepdims=True)

    is_moving = linvel_norm > min_speed_threshold
    root_linvel_dir = np.where(
        is_moving,
        root_linvel / (linvel_norm + 1e-8),
        np.zeros_like(root_linvel)
    )

    # Foot contacts (zeros placeholder)
    foot_contacts = np.zeros((obs.shape[0], 4))

    # Concatenate in AMP feature order
    features = np.concatenate([
        joint_pos,        # 0-8
        joint_vel,        # 9-17
        root_linvel_dir,  # 18-20 (normalized)
        root_angvel,      # 21-23
        root_height,      # 24
        foot_contacts,    # 25-28
    ], axis=1)

    if single:
        features = features[0]

    return features


def compute_policy_style_features(obs: np.ndarray) -> np.ndarray:
    """Compute AMP features using the policy extraction code (JAX).

    This imports and uses the actual policy feature extraction code
    to ensure it matches the reference data format.
    """
    try:
        import jax.numpy as jnp
        from playground_amp.amp.amp_features import extract_amp_features
    except ImportError as e:
        raise ImportError(f"Cannot import JAX/amp_features: {e}")

    # Convert to JAX array
    obs_jax = jnp.array(obs)

    # Extract features using policy code
    features_jax = extract_amp_features(obs_jax)

    # Convert back to NumPy
    return np.array(features_jax)


def test_stationary_case():
    """Test case: Robot standing still (velocity near zero)."""
    print("\n" + "=" * 60)
    print("TEST 1: Stationary Case (velocity ≈ 0)")
    print("=" * 60)

    obs = np.zeros(38, dtype=np.float32)
    obs[2] = 1.0  # gravity z (upright)
    obs[9:18] = np.linspace(0.1, 0.9, 9)  # joint positions
    obs[18:27] = np.linspace(0.01, 0.09, 9)  # joint velocities
    obs[6:9] = np.array([0.05, 0.02, 0.01])  # Very slow velocity (< 0.1 m/s threshold)
    obs[3:6] = np.array([0.1, 0.2, 0.3])  # angular velocity

    speed = np.linalg.norm(obs[6:9])
    print(f"Input velocity: {obs[6:9]} (speed = {speed:.4f} m/s)")
    print(f"Expected: STATIONARY (speed < 0.1) → zero direction vector")

    ref_features = compute_reference_style_features(obs)
    policy_features = compute_policy_style_features(obs)

    print(f"\nReference velocity direction:  {ref_features[18:21]}")
    print(f"Policy velocity direction:     {policy_features[18:21]}")

    # Check if identical
    if np.allclose(ref_features, policy_features, atol=1e-6):
        print("\n✅ PASS: Reference and policy features are IDENTICAL")
        return True
    else:
        diff = np.abs(ref_features - policy_features)
        max_diff_idx = np.argmax(diff)
        print(f"\n❌ FAIL: Features differ!")
        print(f"   Max difference: {diff.max():.8f} at index {max_diff_idx}")
        print(f"   Reference[{max_diff_idx}]: {ref_features[max_diff_idx]}")
        print(f"   Policy[{max_diff_idx}]:    {policy_features[max_diff_idx]}")
        return False


def test_moving_forward_case():
    """Test case: Robot walking forward."""
    print("\n" + "=" * 60)
    print("TEST 2: Moving Forward Case")
    print("=" * 60)

    obs = np.zeros(38, dtype=np.float32)
    obs[2] = 1.0  # gravity z
    obs[9:18] = np.linspace(0.1, 0.9, 9)  # joint positions
    obs[18:27] = np.linspace(0.01, 0.09, 9)  # joint velocities
    obs[6:9] = np.array([0.8, 0.0, 0.0])  # Forward velocity
    obs[3:6] = np.array([0.1, 0.2, 0.3])  # angular velocity

    speed = np.linalg.norm(obs[6:9])
    print(f"Input velocity: {obs[6:9]} (speed = {speed:.4f} m/s)")
    print(f"Expected: MOVING → normalized to [1, 0, 0]")

    ref_features = compute_reference_style_features(obs)
    policy_features = compute_policy_style_features(obs)

    print(f"\nReference velocity direction:  {ref_features[18:21]}")
    print(f"Policy velocity direction:     {policy_features[18:21]}")

    # Check if identical
    if np.allclose(ref_features, policy_features, atol=1e-6):
        print("\n✅ PASS: Reference and policy features are IDENTICAL")
        return True
    else:
        diff = np.abs(ref_features - policy_features)
        max_diff_idx = np.argmax(diff)
        print(f"\n❌ FAIL: Features differ!")
        print(f"   Max difference: {diff.max():.8f} at index {max_diff_idx}")
        return False


def test_moving_diagonal_case():
    """Test case: Robot walking diagonally."""
    print("\n" + "=" * 60)
    print("TEST 3: Moving Diagonal Case")
    print("=" * 60)

    obs = np.zeros(38, dtype=np.float32)
    obs[2] = 1.0  # gravity z
    obs[9:18] = np.linspace(-0.5, 0.5, 9)  # joint positions
    obs[18:27] = np.linspace(-0.1, 0.1, 9)  # joint velocities
    obs[6:9] = np.array([0.5, 0.5, 0.0])  # Diagonal velocity
    obs[3:6] = np.array([-0.1, 0.1, 0.0])  # angular velocity

    speed = np.linalg.norm(obs[6:9])
    expected_dir = obs[6:9] / speed
    print(f"Input velocity: {obs[6:9]} (speed = {speed:.4f} m/s)")
    print(f"Expected direction: {expected_dir}")

    ref_features = compute_reference_style_features(obs)
    policy_features = compute_policy_style_features(obs)

    print(f"\nReference velocity direction:  {ref_features[18:21]}")
    print(f"Policy velocity direction:     {policy_features[18:21]}")

    # Check if identical
    if np.allclose(ref_features, policy_features, atol=1e-6):
        print("\n✅ PASS: Reference and policy features are IDENTICAL")
        return True
    else:
        diff = np.abs(ref_features - policy_features)
        max_diff_idx = np.argmax(diff)
        print(f"\n❌ FAIL: Features differ!")
        print(f"   Max difference: {diff.max():.8f} at index {max_diff_idx}")
        return False


def test_boundary_case():
    """Test case: Robot at exactly the threshold speed."""
    print("\n" + "=" * 60)
    print("TEST 4: Boundary Case (speed ≈ 0.1 m/s)")
    print("=" * 60)

    obs = np.zeros(38, dtype=np.float32)
    obs[2] = 1.0
    obs[9:18] = np.zeros(9)
    obs[18:27] = np.zeros(9)

    # Test just below threshold
    obs[6:9] = np.array([0.099, 0.0, 0.0])  # Just below 0.1 threshold
    ref_below = compute_reference_style_features(obs)
    policy_below = compute_policy_style_features(obs)

    # Test just above threshold
    obs[6:9] = np.array([0.101, 0.0, 0.0])  # Just above 0.1 threshold
    ref_above = compute_reference_style_features(obs)
    policy_above = compute_policy_style_features(obs)

    print(f"Speed 0.099 m/s (below threshold):")
    print(f"  Reference:  {ref_below[18:21]}")
    print(f"  Policy:     {policy_below[18:21]}")

    print(f"\nSpeed 0.101 m/s (above threshold):")
    print(f"  Reference:  {ref_above[18:21]}")
    print(f"  Policy:     {policy_above[18:21]}")

    below_match = np.allclose(ref_below, policy_below, atol=1e-6)
    above_match = np.allclose(ref_above, policy_above, atol=1e-6)

    if below_match and above_match:
        print("\n✅ PASS: Boundary behavior is IDENTICAL")
        return True
    else:
        print(f"\n❌ FAIL: Boundary behavior differs!")
        return False


def test_batch_case():
    """Test case: Batch of observations."""
    print("\n" + "=" * 60)
    print("TEST 5: Batch Processing")
    print("=" * 60)

    # Create batch with mixed stationary/moving
    batch_size = 100
    obs_batch = np.random.randn(batch_size, 38).astype(np.float32) * 0.5
    obs_batch[:, 2] = 1.0  # gravity z

    # Mix of speeds
    speeds = np.random.uniform(0.0, 1.0, batch_size)
    directions = np.random.randn(batch_size, 3)
    directions = directions / (np.linalg.norm(directions, axis=1, keepdims=True) + 1e-8)
    obs_batch[:, 6:9] = directions * speeds[:, np.newaxis]

    ref_features = compute_reference_style_features(obs_batch)
    policy_features = compute_policy_style_features(obs_batch)

    # Count stationary vs moving
    actual_speeds = np.linalg.norm(obs_batch[:, 6:9], axis=1)
    num_stationary = np.sum(actual_speeds < 0.1)
    num_moving = batch_size - num_stationary

    print(f"Batch size: {batch_size}")
    print(f"Stationary samples: {num_stationary}")
    print(f"Moving samples: {num_moving}")

    # Check if identical
    if np.allclose(ref_features, policy_features, atol=1e-6):
        print(f"\n✅ PASS: All {batch_size} samples are IDENTICAL")
        return True
    else:
        diff = np.abs(ref_features - policy_features)
        max_diff = diff.max()
        num_mismatch = np.sum(np.any(diff > 1e-6, axis=1))
        print(f"\n❌ FAIL: {num_mismatch}/{batch_size} samples differ!")
        print(f"   Max difference: {max_diff:.8f}")
        return False


def validate_against_actual_reference_data():
    """Validate by sampling from actual reference data file."""
    print("\n" + "=" * 60)
    print("TEST 6: Actual Reference Data Consistency")
    print("=" * 60)

    ref_path = Path("playground_amp/data/walking_motions_normalized_vel.pkl")
    if not ref_path.exists():
        print(f"⚠️  Reference data not found: {ref_path}")
        return None

    with open(ref_path, "rb") as f:
        data = pickle.load(f)

    features = data["features"]
    print(f"Reference data shape: {features.shape}")

    # Check velocity direction consistency
    vel_dir = features[:, 18:21]
    vel_norms = np.linalg.norm(vel_dir, axis=1)

    # Moving frames should have norm ~1, stationary ~0
    is_moving = vel_norms > 0.5
    moving_norms = vel_norms[is_moving]
    stationary_norms = vel_norms[~is_moving]

    print(f"\nMoving frames ({np.sum(is_moving)}):")
    if len(moving_norms) > 0:
        print(f"  Norm mean: {moving_norms.mean():.6f} (should be ~1.0)")
        print(f"  Norm std:  {moving_norms.std():.6f} (should be ~0.0)")

    print(f"\nStationary frames ({np.sum(~is_moving)}):")
    if len(stationary_norms) > 0:
        print(f"  Norm mean: {stationary_norms.mean():.6f} (should be ~0.0)")
        print(f"  Norm std:  {stationary_norms.std():.6f} (should be ~0.0)")

    # Validate norms
    moving_ok = len(moving_norms) == 0 or np.allclose(moving_norms, 1.0, atol=0.01)
    stationary_ok = len(stationary_norms) == 0 or np.allclose(stationary_norms, 0.0, atol=0.01)

    if moving_ok and stationary_ok:
        print("\n✅ PASS: Reference data velocity normalization is correct")
        return True
    else:
        print("\n❌ FAIL: Reference data has inconsistent velocity norms")
        return False


def main():
    print("=" * 60)
    print("REFERENCE vs POLICY FEATURE CONSISTENCY VALIDATION")
    print("=" * 60)
    print("\nThis validates that reference data and policy feature")
    print("extraction produce IDENTICAL features for the same input.")

    results = []

    # Run all tests
    results.append(("Stationary", test_stationary_case()))
    results.append(("Moving Forward", test_moving_forward_case()))
    results.append(("Moving Diagonal", test_moving_diagonal_case()))
    results.append(("Boundary", test_boundary_case()))
    results.append(("Batch", test_batch_case()))
    results.append(("Reference Data", validate_against_actual_reference_data()))

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        if passed is None:
            status = "⚠️  SKIPPED"
        elif passed:
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
            all_passed = False
        print(f"  {name:<20} {status}")

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL VALIDATIONS PASSED!")
        print("Reference data and policy features are IDENTICAL.")
        print("=" * 60)
        return 0
    else:
        print("⚠️  SOME VALIDATIONS FAILED!")
        print("Review the output above for details.")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    exit(main())
