#!/usr/bin/env python3
"""Diagnostic script to detect AMP feature leaks.

A "feature leak" is when the discriminator can achieve high accuracy
by exploiting systematic differences between reference and policy
features that don't relate to motion style.

Common leaks:
1. Global position (policy at X=0, reference at X=10)
2. Different value ranges (overflow/underflow)
3. Missing or constant features in one distribution
4. Different normalization

Usage:
    cd playground_amp
    uv run python scripts/diagnose_amp_features.py
"""

import pickle
from pathlib import Path

import numpy as np

# Load robot config first
from playground_amp.configs.config import load_robot_config

robot_config_path = Path("assets/robot_config.yaml")
load_robot_config(robot_config_path)


def load_reference_features():
    """Load reference features from pickle file."""
    ref_path = Path("playground_amp/data/walking_motions_normalized_vel.pkl")
    with open(ref_path, "rb") as f:
        data = pickle.load(f)
    return np.array(data["features"])


def collect_policy_features(num_samples=1000):
    """Collect policy features from random rollouts."""
    import jax
    import jax.numpy as jnp
    
    from playground_amp.train import create_env
    from playground_amp.amp.amp_features import extract_amp_features, get_default_amp_config
    
    env = create_env()
    amp_config = get_default_amp_config()
    
    rng = jax.random.PRNGKey(42)
    features_list = []
    
    steps = 0
    while steps < num_samples:
        rng, reset_rng, step_rng = jax.random.split(rng, 3)
        state = env.reset(reset_rng)
        
        for _ in range(100):
            step_rng, action_rng = jax.random.split(step_rng)
            action = jax.random.uniform(action_rng, (9,), minval=-1, maxval=1)
            
            state = env.step(state, action)
            
            obs = state.obs
            foot_contacts = state.info["foot_contacts"]
            root_height = state.info["root_height"]
            
            features = extract_amp_features(
                obs, amp_config, foot_contacts, root_height
            )
            features_list.append(np.array(features))
            steps += 1
            
            if state.done or steps >= num_samples:
                break
    
    return np.stack(features_list[:num_samples])


def analyze_feature_leak(ref_features, policy_features):
    """Analyze potential feature leaks."""
    
    feature_names = [
        # Joint positions (0-8)
        "joint_pos[0] (waist_yaw)",
        "joint_pos[1] (left_hip_pitch)",
        "joint_pos[2] (left_hip_roll)",
        "joint_pos[3] (left_knee_pitch)",
        "joint_pos[4] (left_ankle_pitch)",
        "joint_pos[5] (right_hip_pitch)",
        "joint_pos[6] (right_hip_roll)",
        "joint_pos[7] (right_knee_pitch)",
        "joint_pos[8] (right_ankle_pitch)",
        # Joint velocities (9-17)
        "joint_vel[0] (waist_yaw)",
        "joint_vel[1] (left_hip_pitch)",
        "joint_vel[2] (left_hip_roll)",
        "joint_vel[3] (left_knee_pitch)",
        "joint_vel[4] (left_ankle_pitch)",
        "joint_vel[5] (right_hip_pitch)",
        "joint_vel[6] (right_hip_roll)",
        "joint_vel[7] (right_knee_pitch)",
        "joint_vel[8] (right_ankle_pitch)",
        # Root velocities (18-23)
        "linvel_dir_x",
        "linvel_dir_y",
        "linvel_dir_z",
        "angvel_x",
        "angvel_y",
        "angvel_z",
        # Root height (24)
        "root_height",
        # Foot contacts (25-28)
        "foot_contact[0] (left_toe)",
        "foot_contact[1] (left_heel)",
        "foot_contact[2] (right_toe)",
        "foot_contact[3] (right_heel)",
    ]
    
    print("=" * 80)
    print("AMP FEATURE LEAK DIAGNOSTIC")
    print("=" * 80)
    print(f"Reference samples: {ref_features.shape[0]}")
    print(f"Policy samples: {policy_features.shape[0]}")
    print()
    
    # =========================================================================
    # Check 1: Constant Features (zero variance)
    # =========================================================================
    print("-" * 80)
    print("CHECK 1: Constant Features (zero variance → discriminator can memorize)")
    print("-" * 80)
    
    ref_var = np.var(ref_features, axis=0)
    pol_var = np.var(policy_features, axis=0)
    
    constant_in_ref = ref_var < 1e-6
    constant_in_pol = pol_var < 1e-6
    
    if np.any(constant_in_ref):
        print("⚠️  CONSTANT in REFERENCE:")
        for i in np.where(constant_in_ref)[0]:
            name = feature_names[i] if i < len(feature_names) else f"feat[{i}]"
            print(f"    [{i}] {name}: value = {ref_features[0, i]:.6f}")
    
    if np.any(constant_in_pol):
        print("⚠️  CONSTANT in POLICY:")
        for i in np.where(constant_in_pol)[0]:
            name = feature_names[i] if i < len(feature_names) else f"feat[{i}]"
            print(f"    [{i}] {name}: value = {policy_features[0, i]:.6f}")
    
    if not np.any(constant_in_ref) and not np.any(constant_in_pol):
        print("✓ No constant features found")
    print()
    
    # =========================================================================
    # Check 2: Mean Separation (easy to distinguish by mean alone)
    # =========================================================================
    print("-" * 80)
    print("CHECK 2: Mean Separation (discriminator can cheat on mean difference)")
    print("-" * 80)
    
    ref_mean = np.mean(ref_features, axis=0)
    pol_mean = np.mean(policy_features, axis=0)
    ref_std = np.std(ref_features, axis=0)
    pol_std = np.std(policy_features, axis=0)
    
    # Compute standardized mean difference (effect size)
    pooled_std = np.sqrt((ref_std**2 + pol_std**2) / 2 + 1e-8)
    effect_size = np.abs(ref_mean - pol_mean) / pooled_std
    
    print(f"{'Feature':<35} {'Ref μ':>10} {'Pol μ':>10} {'Effect':>10} {'Status':<10}")
    print("-" * 80)
    
    severe_leaks = []
    moderate_leaks = []
    
    for i in range(ref_features.shape[1]):
        name = feature_names[i] if i < len(feature_names) else f"feat[{i}]"
        es = effect_size[i]
        
        if es > 2.0:
            status = "🔴 SEVERE"
            severe_leaks.append((i, name, es))
        elif es > 1.0:
            status = "🟡 MODERATE"
            moderate_leaks.append((i, name, es))
        elif es > 0.5:
            status = "🟢 small"
        else:
            status = "✓ ok"
        
        print(f"{name:<35} {ref_mean[i]:>10.4f} {pol_mean[i]:>10.4f} {es:>10.2f} {status:<10}")
    
    print()
    
    # =========================================================================
    # Check 3: Value Range Issues (overflow/NaN/Inf)
    # =========================================================================
    print("-" * 80)
    print("CHECK 3: Value Range Issues (overflow warning source)")
    print("-" * 80)
    
    ref_min = np.min(ref_features, axis=0)
    ref_max = np.max(ref_features, axis=0)
    pol_min = np.min(policy_features, axis=0)
    pol_max = np.max(policy_features, axis=0)
    
    # Check for extreme values
    extreme_threshold = 100.0
    
    ref_extreme = (np.abs(ref_min) > extreme_threshold) | (np.abs(ref_max) > extreme_threshold)
    pol_extreme = (np.abs(pol_min) > extreme_threshold) | (np.abs(pol_max) > extreme_threshold)
    
    if np.any(ref_extreme):
        print("⚠️  EXTREME values in REFERENCE:")
        for i in np.where(ref_extreme)[0]:
            name = feature_names[i] if i < len(feature_names) else f"feat[{i}]"
            print(f"    [{i}] {name}: range [{ref_min[i]:.2f}, {ref_max[i]:.2f}]")
    
    if np.any(pol_extreme):
        print("⚠️  EXTREME values in POLICY:")
        for i in np.where(pol_extreme)[0]:
            name = feature_names[i] if i < len(feature_names) else f"feat[{i}]"
            print(f"    [{i}] {name}: range [{pol_min[i]:.2f}, {pol_max[i]:.2f}]")
    
    # Check for NaN/Inf
    ref_nan = np.any(np.isnan(ref_features))
    pol_nan = np.any(np.isnan(policy_features))
    ref_inf = np.any(np.isinf(ref_features))
    pol_inf = np.any(np.isinf(policy_features))
    
    if ref_nan or ref_inf:
        print(f"⚠️  REFERENCE contains NaN={ref_nan}, Inf={ref_inf}")
    if pol_nan or pol_inf:
        print(f"⚠️  POLICY contains NaN={pol_nan}, Inf={pol_inf}")
    
    if not np.any(ref_extreme) and not np.any(pol_extreme) and not ref_nan and not pol_nan:
        print("✓ No extreme values or NaN/Inf found")
    print()
    
    # =========================================================================
    # Check 4: Distribution Overlap (KL divergence proxy)
    # =========================================================================
    print("-" * 80)
    print("CHECK 4: Distribution Overlap (range overlap between ref and policy)")
    print("-" * 80)
    
    no_overlap = []
    for i in range(ref_features.shape[1]):
        # Check if ranges overlap
        overlap = min(ref_max[i], pol_max[i]) - max(ref_min[i], pol_min[i])
        ref_range = ref_max[i] - ref_min[i]
        pol_range = pol_max[i] - pol_min[i]
        
        if overlap < 0:
            name = feature_names[i] if i < len(feature_names) else f"feat[{i}]"
            no_overlap.append((i, name, ref_min[i], ref_max[i], pol_min[i], pol_max[i]))
    
    if no_overlap:
        print("🔴 FEATURES WITH NO DISTRIBUTION OVERLAP (100% separable):")
        for i, name, rmin, rmax, pmin, pmax in no_overlap:
            print(f"    [{i}] {name}")
            print(f"        Reference: [{rmin:.4f}, {rmax:.4f}]")
            print(f"        Policy:    [{pmin:.4f}, {pmax:.4f}]")
    else:
        print("✓ All features have overlapping ranges")
    print()
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if severe_leaks or no_overlap:
        print("🔴 CRITICAL FEATURE LEAKS DETECTED!")
        print()
        print("The discriminator can achieve near-100% accuracy by exploiting:")
        if severe_leaks:
            print("\n  Severe mean differences (effect size > 2.0):")
            for i, name, es in severe_leaks:
                print(f"    - {name}: effect size = {es:.2f}")
        if no_overlap:
            print("\n  Non-overlapping distributions:")
            for i, name, rmin, rmax, pmin, pmax in no_overlap:
                print(f"    - {name}: ref=[{rmin:.2f},{rmax:.2f}], pol=[{pmin:.2f},{pmax:.2f}]")
        
        print("\n  RECOMMENDED FIXES:")
        print("  1. Regenerate reference data with same feature extraction as policy")
        print("  2. Or remove/mask problematic features from AMP")
        print("  3. Or add noise to make distributions overlap")
    elif moderate_leaks:
        print("🟡 MODERATE LEAKS - may cause high discriminator accuracy")
        for i, name, es in moderate_leaks:
            print(f"    - {name}: effect size = {es:.2f}")
    else:
        print("✓ No significant feature leaks detected!")
        print("  If discriminator still hits 100%, the issue may be:")
        print("  - Multi-feature correlations (harder to detect)")
        print("  - Discriminator overfitting to noise")
        print("  - Insufficient regularization (try increasing r1_gamma)")
    
    return severe_leaks, moderate_leaks, no_overlap


if __name__ == "__main__":
    print("Loading reference features...")
    ref_features = load_reference_features()
    
    print("Collecting policy features (this may take a minute)...")
    policy_features = collect_policy_features(num_samples=1000)
    
    analyze_feature_leak(ref_features, policy_features)
