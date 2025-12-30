#!/usr/bin/env python3
"""Debug script to analyze AMP feature distribution mismatch.

This script diagnoses why the discriminator achieves 100% accuracy
by comparing reference motion features with policy rollout features.

Usage:
    cd playground_amp
    uv run python scripts/debug_amp_features.py
"""

import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# Load robot config first
from playground_amp.configs.training_config import load_robot_config

robot_config_path = Path("assets/robot_config.yaml")
load_robot_config(robot_config_path)

from playground_amp.amp.policy_features import get_feature_config


def analyze_reference_data():
    """Analyze reference motion data distribution."""
    print("=" * 70)
    print("REFERENCE DATA ANALYSIS")
    print("=" * 70)

    # Load reference data
    ref_path = Path("playground_amp/data/walking_motions_normalized_vel.pkl")
    with open(ref_path, "rb") as f:
        ref_data = pickle.load(f)

    ref_features = np.array(ref_data)
    print(f"\nReference data shape: {ref_features.shape}")
    print(f"  - {ref_features.shape[0]} samples")
    print(f"  - {ref_features.shape[1]} feature dimensions")

    # Feature names (29-dim format)
    feature_names = [
        # Joint positions (0-8)
        "hip_x_pos",
        "hip_y_pos",
        "hip_z_pos",
        "knee_x_pos",
        "knee_y_pos",
        "knee_z_pos",
        "ankle_x_pos",
        "ankle_y_pos",
        "ankle_z_pos",
        # Joint velocities (9-17)
        "hip_x_vel",
        "hip_y_vel",
        "hip_z_vel",
        "knee_x_vel",
        "knee_y_vel",
        "knee_z_vel",
        "ankle_x_vel",
        "ankle_y_vel",
        "ankle_z_vel",
        # Root velocities (18-23)
        "linvel_x",
        "linvel_y",
        "linvel_z",
        "angvel_x",
        "angvel_y",
        "angvel_z",
        # Root height (24)
        "root_height",
        # Foot contacts (25-28)
        "foot_1",
        "foot_2",
        "foot_3",
        "foot_4",
    ]

    # Per-feature statistics
    print("\n" + "-" * 70)
    print("Per-Feature Statistics (Reference Data):")
    print("-" * 70)
    print(f"{'Feature':<20} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 70)

    for i in range(min(ref_features.shape[1], len(feature_names))):
        col = ref_features[:, i]
        name = feature_names[i] if i < len(feature_names) else f"feat_{i}"
        print(
            f"{name:<20} {col.mean():>10.4f} {col.std():>10.4f} {col.min():>10.4f} {col.max():>10.4f}"
        )

    # Highlight potential issues
    print("\n" + "=" * 70)
    print("POTENTIAL ISSUES:")
    print("=" * 70)

    # Check velocity normalization (indices 18-20)
    linvel = ref_features[:, 18:21]
    linvel_norms = np.linalg.norm(linvel, axis=1)
    print(f"\n1. Linear Velocity Direction (should be unit vectors if normalized):")
    print(f"   Mean norm: {linvel_norms.mean():.4f} (expect ~1.0 if normalized)")
    print(f"   Std norm:  {linvel_norms.std():.4f}")
    print(f"   Min norm:  {linvel_norms.min():.4f}")
    print(f"   Max norm:  {linvel_norms.max():.4f}")

    # Check foot contacts (indices 25-28)
    foot_contacts = ref_features[:, 25:29]
    print(f"\n2. Foot Contacts (should have binary-like distribution):")
    print(f"   Mean: {foot_contacts.mean(axis=0)}")
    print(f"   % non-zero per foot: {(foot_contacts > 0).mean(axis=0) * 100}")
    print(f"   Overall % non-zero: {(foot_contacts > 0).mean() * 100:.1f}%")

    # Check root height (index 24)
    root_height = ref_features[:, 24]
    print(f"\n3. Root Height (gravity z-component, expect ~-0.9 to -1.0 for upright):")
    print(f"   Mean: {root_height.mean():.4f}")
    print(f"   Std:  {root_height.std():.4f}")
    print(f"   Range: [{root_height.min():.4f}, {root_height.max():.4f}]")

    # Check for any NaN or Inf
    print(f"\n4. Data Quality:")
    print(f"   NaN count: {np.isnan(ref_features).sum()}")
    print(f"   Inf count: {np.isinf(ref_features).sum()}")

    return ref_features


def analyze_policy_features():
    """Generate some policy features and compare with reference."""
    print("\n" + "=" * 70)
    print("POLICY FEATURE ANALYSIS (from random initial states)")
    print("=" * 70)

    from playground_amp.amp.policy_features import (
        extract_amp_features,
        get_feature_config,
    )
    from playground_amp.envs.wildrobot_env import EnvConfig, WildRobotEnv

    # Create environment
    env = WildRobotEnv(config=EnvConfig())
    amp_config = get_feature_config()

    # Collect features from random policy
    rng = jax.random.PRNGKey(42)
    features_list = []

    num_episodes = 10
    steps_per_episode = 100

    for ep in range(num_episodes):
        rng, reset_rng, step_rng = jax.random.split(rng, 3)
        state = env.reset(reset_rng)

        for step in range(steps_per_episode):
            # Random action
            step_rng, action_rng = jax.random.split(step_rng)
            action = jax.random.uniform(action_rng, (9,), minval=-1, maxval=1)

            state = env.step(state, action)

            # Extract features
            # Access via typed WR namespace
            from playground_amp.envs.env_info import WR_INFO_KEY

            obs = state.obs
            wr_info = state.info[WR_INFO_KEY]
            foot_contacts = wr_info.foot_contacts

            features = extract_amp_features(obs, amp_config, foot_contacts)
            features_list.append(np.array(features))

            if state.done:
                break

    policy_features = np.stack(features_list)
    print(f"\nPolicy features shape: {policy_features.shape}")

    # Per-feature statistics
    feature_names = [
        "hip_x_pos",
        "hip_y_pos",
        "hip_z_pos",
        "knee_x_pos",
        "knee_y_pos",
        "knee_z_pos",
        "ankle_x_pos",
        "ankle_y_pos",
        "ankle_z_pos",
        "hip_x_vel",
        "hip_y_vel",
        "hip_z_vel",
        "knee_x_vel",
        "knee_y_vel",
        "knee_z_vel",
        "ankle_x_vel",
        "ankle_y_vel",
        "ankle_z_vel",
        "linvel_x",
        "linvel_y",
        "linvel_z",
        "angvel_x",
        "angvel_y",
        "angvel_z",
        "root_height",
        "foot_1",
        "foot_2",
        "foot_3",
        "foot_4",
    ]

    print("\n" + "-" * 70)
    print("Per-Feature Statistics (Policy Rollouts):")
    print("-" * 70)
    print(f"{'Feature':<20} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 70)

    for i in range(min(policy_features.shape[1], len(feature_names))):
        col = policy_features[:, i]
        name = feature_names[i] if i < len(feature_names) else f"feat_{i}"
        print(
            f"{name:<20} {col.mean():>10.4f} {col.std():>10.4f} {col.min():>10.4f} {col.max():>10.4f}"
        )

    # Velocity analysis
    linvel = policy_features[:, 18:21]
    linvel_norms = np.linalg.norm(linvel, axis=1)
    print(f"\nLinear Velocity Direction Norms:")
    print(f"   Mean: {linvel_norms.mean():.4f}")
    print(f"   % zero (stationary): {(linvel_norms == 0).mean() * 100:.1f}%")

    # Foot contacts
    foot_contacts = policy_features[:, 25:29]
    print(f"\nFoot Contacts:")
    print(f"   % non-zero: {(foot_contacts > 0).mean() * 100:.1f}%")

    return policy_features


def compare_distributions(ref_features, policy_features):
    """Compare reference and policy feature distributions."""
    print("\n" + "=" * 70)
    print("DISTRIBUTION COMPARISON (Reference vs Policy)")
    print("=" * 70)

    feature_names = [
        "hip_x_pos",
        "hip_y_pos",
        "hip_z_pos",
        "knee_x_pos",
        "knee_y_pos",
        "knee_z_pos",
        "ankle_x_pos",
        "ankle_y_pos",
        "ankle_z_pos",
        "hip_x_vel",
        "hip_y_vel",
        "hip_z_vel",
        "knee_x_vel",
        "knee_y_vel",
        "knee_z_vel",
        "ankle_x_vel",
        "ankle_y_vel",
        "ankle_z_vel",
        "linvel_x",
        "linvel_y",
        "linvel_z",
        "angvel_x",
        "angvel_y",
        "angvel_z",
        "root_height",
        "foot_1",
        "foot_2",
        "foot_3",
        "foot_4",
    ]

    print(
        f"\n{'Feature':<20} {'Ref Mean':>10} {'Pol Mean':>10} {'Diff':>10} {'Separable?':>12}"
    )
    print("-" * 70)

    separable_features = []

    for i in range(
        min(ref_features.shape[1], policy_features.shape[1], len(feature_names))
    ):
        ref_col = ref_features[:, i]
        pol_col = policy_features[:, i]

        ref_mean = ref_col.mean()
        pol_mean = pol_col.mean()
        diff = abs(ref_mean - pol_mean)

        # Check if distributions overlap
        ref_std = ref_col.std()
        pol_std = pol_col.std()

        # Simple separability heuristic: difference > 2 * max(std)
        max_std = max(ref_std, pol_std)
        is_separable = diff > 2 * max_std if max_std > 0.001 else False

        name = feature_names[i]
        sep_str = "⚠️ YES" if is_separable else "no"
        print(
            f"{name:<20} {ref_mean:>10.4f} {pol_mean:>10.4f} {diff:>10.4f} {sep_str:>12}"
        )

        if is_separable:
            separable_features.append(name)

    print("\n" + "=" * 70)
    print("DIAGNOSIS:")
    print("=" * 70)

    if separable_features:
        print(f"\n⚠️  Found {len(separable_features)} easily separable features:")
        for f in separable_features:
            print(f"   - {f}")
        print(
            "\nThese features alone allow the discriminator to achieve ~100% accuracy."
        )
        print("The discriminator doesn't need to learn style - it just needs to detect")
        print("these obvious distribution differences.")
    else:
        print("\n✓ No obviously separable features found.")
        print(
            "  The discriminator accuracy may be due to subtle multi-feature correlations."
        )


if __name__ == "__main__":
    ref_features = analyze_reference_data()
    policy_features = analyze_policy_features()
    compare_distributions(ref_features, policy_features)
