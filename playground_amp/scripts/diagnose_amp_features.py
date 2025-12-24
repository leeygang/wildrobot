#!/usr/bin/env python3
"""Diagnostic script to detect AMP feature leaks.

A "feature leak" is when the discriminator can achieve high accuracy
by exploiting systematic differences between reference and policy
features that don't relate to motion style.

=== THE GOLDEN RULES OF AMP FEATURES ===

1. INVARIANCE TO GLOBAL TRANSLATION: Features must be root-relative.
   Never use absolute positions.

2. INVARIANCE TO GLOBAL HEADING: All positions and velocities must be
   in the robot's local frame. The discriminator should not know if
   the robot is walking North or South.

3. MATHEMATICAL PARITY: The code that processes reference data must be
   the EXACT SAME code path as live simulation data.

Common leaks this script detects:
- Global position (policy at X=0, reference at X=10)
- Different value ranges (overflow/underflow)
- Missing or constant features in one distribution
- Different normalization
- Temporal/frame rate mismatches
- Root height offsets

Usage:
    cd wildrobot
    uv run python playground_amp/scripts/diagnose_amp_features.py
"""

import pickle
import sys
from pathlib import Path

# Add project root to path so imports work without PYTHONPATH
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

# Load robot config first
from playground_amp.configs.config import load_robot_config

robot_config_path = project_root / "assets" / "robot_config.yaml"
load_robot_config(robot_config_path)

# Feature names for 29-dim AMP features
FEATURE_NAMES = [
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
    # Root height (24) - CRITICAL CHECK
    "root_height",
    # Foot contacts (25-28)
    "foot_contact[0] (left_toe)",
    "foot_contact[1] (left_heel)",
    "foot_contact[2] (right_toe)",
    "foot_contact[3] (right_heel)",
]

# Critical feature indices that commonly cause leaks
CRITICAL_FEATURES = {
    24: "root_height",  # Most common leak source
}


def load_reference_features():
    """Load reference features from pickle file specified in training config."""
    from playground_amp.configs.config import load_training_config

    # Load training config to get dataset path
    training_config_path = (
        project_root / "playground_amp" / "configs" / "ppo_amass_training.yaml"
    )
    training_config = load_training_config(training_config_path)

    # Get dataset path from config (ref_motion_path is where it's stored)
    dataset_path = training_config.ref_motion_path
    if dataset_path is None:
        raise ValueError("ref_motion_path not set in training config")

    ref_path = project_root / dataset_path

    print(f"  Loading from: {ref_path}")

    with open(ref_path, "rb") as f:
        data = pickle.load(f)
    return np.array(data["features"])


def collect_policy_features(num_samples=1000):
    """Collect policy features from random rollouts."""
    import jax
    import jax.numpy as jnp
    from playground_amp.amp.amp_features import extract_amp_features, get_amp_config

    from playground_amp.train import create_env

    env = create_env()
    amp_config = get_amp_config()

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

            features = extract_amp_features(obs, amp_config, foot_contacts, root_height)
            features_list.append(np.array(features))
            steps += 1

            if state.done or steps >= num_samples:
                break

    return np.stack(features_list[:num_samples])


def check_constant_features(ref_features, policy_features):
    """Check 1: Constant Features (zero variance → discriminator can memorize)."""
    print("-" * 80)
    print("CHECK 1: Constant Features (zero variance → discriminator can memorize)")
    print("-" * 80)

    ref_var = np.var(ref_features, axis=0)
    pol_var = np.var(policy_features, axis=0)

    constant_in_ref = ref_var < 1e-6
    constant_in_pol = pol_var < 1e-6

    issues = []

    if np.any(constant_in_ref):
        print("⚠️  CONSTANT in REFERENCE:")
        for i in np.where(constant_in_ref)[0]:
            name = FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f"feat[{i}]"
            print(f"    [{i}] {name}: value = {ref_features[0, i]:.6f}")
            issues.append((i, name, "constant_in_ref"))

    if np.any(constant_in_pol):
        print("⚠️  CONSTANT in POLICY:")
        for i in np.where(constant_in_pol)[0]:
            name = FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f"feat[{i}]"
            print(f"    [{i}] {name}: value = {policy_features[0, i]:.6f}")
            issues.append((i, name, "constant_in_pol"))

    if not issues:
        print("✓ No constant features found")
    print()

    return issues


def check_mean_separation(ref_features, policy_features):
    """Check 2: Mean Separation (easy to distinguish by mean alone)."""
    print("-" * 80)
    print("CHECK 2: Mean Separation (discriminator can cheat on mean difference)")
    print("-" * 80)

    ref_mean = np.mean(ref_features, axis=0)
    pol_mean = np.mean(policy_features, axis=0)
    ref_std = np.std(ref_features, axis=0)
    pol_std = np.std(policy_features, axis=0)

    # Compute standardized mean difference (Cohen's d effect size)
    pooled_std = np.sqrt((ref_std**2 + pol_std**2) / 2 + 1e-8)
    effect_size = np.abs(ref_mean - pol_mean) / pooled_std

    print(f"{'Feature':<35} {'Ref μ':>10} {'Pol μ':>10} {'Effect':>10} {'Status':<10}")
    print("-" * 80)

    severe_leaks = []
    moderate_leaks = []

    for i in range(ref_features.shape[1]):
        name = FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f"feat[{i}]"
        es = effect_size[i]

        if es > 2.0:
            status = "🔴 SEVERE"
            severe_leaks.append((i, name, es, ref_mean[i], pol_mean[i]))
        elif es > 1.0:
            status = "🟡 MODERATE"
            moderate_leaks.append((i, name, es, ref_mean[i], pol_mean[i]))
        elif es > 0.5:
            status = "🟢 small"
        else:
            status = "✓ ok"

        print(
            f"{name:<35} {ref_mean[i]:>10.4f} {pol_mean[i]:>10.4f} {es:>10.2f} {status:<10}"
        )

    print()
    return severe_leaks, moderate_leaks


def check_value_range(ref_features, policy_features):
    """Check 3: Value Range Issues (overflow/NaN/Inf)."""
    print("-" * 80)
    print("CHECK 3: Value Range Issues (overflow warning source)")
    print("-" * 80)

    ref_min = np.min(ref_features, axis=0)
    ref_max = np.max(ref_features, axis=0)
    pol_min = np.min(policy_features, axis=0)
    pol_max = np.max(policy_features, axis=0)

    # Check for extreme values
    extreme_threshold = 100.0
    issues = []

    ref_extreme = (np.abs(ref_min) > extreme_threshold) | (
        np.abs(ref_max) > extreme_threshold
    )
    pol_extreme = (np.abs(pol_min) > extreme_threshold) | (
        np.abs(pol_max) > extreme_threshold
    )

    if np.any(ref_extreme):
        print("⚠️  EXTREME values in REFERENCE:")
        for i in np.where(ref_extreme)[0]:
            name = FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f"feat[{i}]"
            print(f"    [{i}] {name}: range [{ref_min[i]:.2f}, {ref_max[i]:.2f}]")
            issues.append((i, name, "extreme_ref"))

    if np.any(pol_extreme):
        print("⚠️  EXTREME values in POLICY:")
        for i in np.where(pol_extreme)[0]:
            name = FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f"feat[{i}]"
            print(f"    [{i}] {name}: range [{pol_min[i]:.2f}, {pol_max[i]:.2f}]")
            issues.append((i, name, "extreme_pol"))

    # Check for NaN/Inf
    ref_nan = np.any(np.isnan(ref_features))
    pol_nan = np.any(np.isnan(policy_features))
    ref_inf = np.any(np.isinf(ref_features))
    pol_inf = np.any(np.isinf(policy_features))

    if ref_nan or ref_inf:
        print(f"⚠️  REFERENCE contains NaN={ref_nan}, Inf={ref_inf}")
        issues.append((-1, "reference", "nan_inf"))
    if pol_nan or pol_inf:
        print(f"⚠️  POLICY contains NaN={pol_nan}, Inf={pol_inf}")
        issues.append((-1, "policy", "nan_inf"))

    if not issues:
        print("✓ No extreme values or NaN/Inf found")
    print()

    return issues


def check_distribution_overlap(ref_features, policy_features):
    """Check 4: Distribution Overlap (KL divergence proxy)."""
    print("-" * 80)
    print("CHECK 4: Distribution Overlap (range overlap between ref and policy)")
    print("-" * 80)

    ref_min = np.min(ref_features, axis=0)
    ref_max = np.max(ref_features, axis=0)
    pol_min = np.min(policy_features, axis=0)
    pol_max = np.max(policy_features, axis=0)

    no_overlap = []
    for i in range(ref_features.shape[1]):
        # Check if ranges overlap
        overlap = min(ref_max[i], pol_max[i]) - max(ref_min[i], pol_min[i])

        if overlap < 0:
            name = FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f"feat[{i}]"
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

    return no_overlap


def check_root_height(ref_features, policy_features):
    """Check 5: Root Height (most common Golden Rule violation)."""
    print("-" * 80)
    print("CHECK 5: Root Height (Critical - most common Golden Rule violation)")
    print("-" * 80)

    ROOT_HEIGHT_IDX = 24

    ref_height = ref_features[:, ROOT_HEIGHT_IDX]
    pol_height = policy_features[:, ROOT_HEIGHT_IDX]

    ref_mean = np.mean(ref_height)
    pol_mean = np.mean(pol_height)
    ref_std = np.std(ref_height)
    pol_std = np.std(pol_height)
    ref_min, ref_max = np.min(ref_height), np.max(ref_height)
    pol_min, pol_max = np.min(pol_height), np.max(pol_height)

    print(f"  Reference root_height:")
    print(f"    Mean: {ref_mean:.4f}, Std: {ref_std:.4f}")
    print(f"    Range: [{ref_min:.4f}, {ref_max:.4f}]")
    print()
    print(f"  Policy root_height:")
    print(f"    Mean: {pol_mean:.4f}, Std: {pol_std:.4f}")
    print(f"    Range: [{pol_min:.4f}, {pol_max:.4f}]")
    print()

    # Check for common issues
    issues = []

    # Issue 1: Negative height in policy (gravity leak - the v0.5.0 bug!)
    if pol_mean < 0:
        print("  🔴 CRITICAL: Policy root_height is NEGATIVE!")
        print(
            "     This likely means gravity[2] is being used instead of actual height."
        )
        print("     Fix: Use env.info['root_height'] instead of obs extraction.")
        issues.append("negative_policy_height")

    # Issue 2: Large offset between reference and policy
    offset = abs(ref_mean - pol_mean)
    if offset > 0.1:
        print(f"  🔴 CRITICAL: Large height offset detected: {offset:.4f}m")
        print("     Reference and policy are measuring from different baselines.")
        print(
            "     Fix: Ensure both use the same height definition (e.g., pelvis from ground)."
        )
        issues.append("height_offset")

    # Issue 3: No overlap
    overlap = min(ref_max, pol_max) - max(ref_min, pol_min)
    if overlap < 0:
        print("  🔴 CRITICAL: Root height distributions DO NOT OVERLAP!")
        print("     Discriminator can achieve 100% accuracy on this feature alone.")
        issues.append("no_overlap")

    if not issues:
        print("  ✓ Root height looks consistent between reference and policy")

    print()
    return issues


def check_temporal_correlation(ref_features, policy_features):
    """Check 6: Temporal Correlation (frame rate mismatch detection)."""
    print("-" * 80)
    print("CHECK 6: Temporal Correlation (frame rate / delta mismatch)")
    print("-" * 80)

    # Compute deltas (frame-to-frame differences)
    ref_deltas = np.diff(ref_features, axis=0)
    pol_deltas = np.diff(policy_features, axis=0)

    # Focus on velocity features (9-17) where frame rate matters most
    VELOCITY_START = 9
    VELOCITY_END = 18

    ref_vel_deltas = ref_deltas[:, VELOCITY_START:VELOCITY_END]
    pol_vel_deltas = pol_deltas[:, VELOCITY_START:VELOCITY_END]

    ref_delta_std = np.std(ref_vel_deltas, axis=0)
    pol_delta_std = np.std(pol_vel_deltas, axis=0)

    # Compute ratio of delta magnitudes
    ratio = pol_delta_std / (ref_delta_std + 1e-8)

    print(f"  Joint velocity delta (frame-to-frame) ratios (Policy/Reference):")
    print(f"  {'Feature':<35} {'Ref Δσ':>10} {'Pol Δσ':>10} {'Ratio':>10}")
    print("  " + "-" * 70)

    issues = []
    for i in range(VELOCITY_END - VELOCITY_START):
        feat_idx = VELOCITY_START + i
        name = (
            FEATURE_NAMES[feat_idx]
            if feat_idx < len(FEATURE_NAMES)
            else f"feat[{feat_idx}]"
        )
        r = ratio[i]

        if r > 2.0 or r < 0.5:
            status = "⚠️  MISMATCH"
            issues.append((feat_idx, name, r))
        else:
            status = "✓"

        print(
            f"  {name:<35} {ref_delta_std[i]:>10.4f} {pol_delta_std[i]:>10.4f} {r:>10.2f} {status}"
        )

    print()

    if issues:
        print("  ⚠️  Frame rate mismatch detected!")
        print("     The magnitude of frame-to-frame changes differs significantly.")
        print("     Common causes:")
        print("       - Reference data at 120Hz, simulation at 50Hz")
        print("       - Different dt values in velocity calculation")
        print(
            "       - Reference uses finite differences, policy uses analytical velocity"
        )
    else:
        print("  ✓ Temporal correlations appear consistent")

    print()
    return issues


def plot_severe_leaks(
    ref_features, policy_features, severe_leaks, no_overlap, output_dir=None
):
    """Generate histogram plots for severe leaks."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️  matplotlib not installed, skipping histogram plots")
        return

    if not severe_leaks and not no_overlap:
        print("No severe leaks to plot.")
        return

    # Combine all problem features
    problem_features = []
    for i, name, es, ref_mean, pol_mean in severe_leaks:
        problem_features.append((i, name, f"Effect size: {es:.2f}"))
    for i, name, rmin, rmax, pmin, pmax in no_overlap:
        if not any(pf[0] == i for pf in problem_features):
            problem_features.append((i, name, "No overlap"))

    if not problem_features:
        return

    # Create output directory
    if output_dir is None:
        output_dir = project_root / "playground_amp" / "logs" / "diagnostics"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("-" * 80)
    print("GENERATING HISTOGRAM PLOTS")
    print("-" * 80)

    num_plots = min(len(problem_features), 6)  # Max 6 plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (i, name, issue) in enumerate(problem_features[:num_plots]):
        ax = axes[idx]

        ref_data = ref_features[:, i]
        pol_data = policy_features[:, i]

        # Plot histograms
        ax.hist(
            ref_data,
            bins=50,
            alpha=0.6,
            label="Reference (AMASS)",
            color="blue",
            density=True,
        )
        ax.hist(
            pol_data,
            bins=50,
            alpha=0.6,
            label="Policy (Live)",
            color="red",
            density=True,
        )

        ax.set_title(f"[{i}] {name}\n{issue}", fontsize=10)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(num_plots, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("AMP Feature Leak Detection - Distribution Comparison", fontsize=14)
    plt.tight_layout()

    # Save plot
    plot_path = output_dir / "feature_leak_histograms.png"
    plt.savefig(plot_path, dpi=150)
    print(f"  Saved histogram plot to: {plot_path}")

    # Also save individual plots for each problem feature
    for i, name, issue in problem_features[:6]:
        fig_single, ax = plt.subplots(figsize=(8, 5))

        ref_data = ref_features[:, i]
        pol_data = policy_features[:, i]

        ax.hist(
            ref_data,
            bins=50,
            alpha=0.6,
            label="Reference (AMASS)",
            color="blue",
            density=True,
        )
        ax.hist(
            pol_data,
            bins=50,
            alpha=0.6,
            label="Policy (Live)",
            color="red",
            density=True,
        )

        safe_name = (
            name.replace("[", "_")
            .replace("]", "_")
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
        )
        ax.set_title(f"Feature Leak: {name}\n{issue}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True, alpha=0.3)

        single_path = output_dir / f"leak_{i:02d}_{safe_name}.png"
        fig_single.savefig(single_path, dpi=150)
        plt.close(fig_single)
        print(f"  Saved: {single_path}")

    plt.close(fig)
    print()


def analyze_feature_leak(ref_features, policy_features, save_plots=True):
    """Main analysis function - runs all checks."""
    print("=" * 80)
    print("AMP FEATURE LEAK DIAGNOSTIC")
    print("=" * 80)
    print(f"Reference samples: {ref_features.shape[0]}")
    print(f"Policy samples: {policy_features.shape[0]}")
    print(f"Feature dimension: {ref_features.shape[1]}")
    print()

    # Run all checks
    constant_issues = check_constant_features(ref_features, policy_features)
    severe_leaks, moderate_leaks = check_mean_separation(ref_features, policy_features)
    range_issues = check_value_range(ref_features, policy_features)
    no_overlap = check_distribution_overlap(ref_features, policy_features)
    height_issues = check_root_height(ref_features, policy_features)
    temporal_issues = check_temporal_correlation(ref_features, policy_features)

    # Generate plots for severe leaks
    if save_plots:
        plot_severe_leaks(ref_features, policy_features, severe_leaks, no_overlap)

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    total_critical = len(severe_leaks) + len(no_overlap) + len(height_issues)

    if total_critical > 0:
        print("🔴 CRITICAL FEATURE LEAKS DETECTED!")
        print()
        print("The discriminator can achieve near-100% accuracy by exploiting:")

        if severe_leaks:
            print("\n  Severe mean differences (effect size > 2.0):")
            for i, name, es, ref_mean, pol_mean in severe_leaks:
                print(
                    f"    - [{i}] {name}: effect={es:.2f}, ref_μ={ref_mean:.4f}, pol_μ={pol_mean:.4f}"
                )

        if no_overlap:
            print("\n  Non-overlapping distributions:")
            for i, name, rmin, rmax, pmin, pmax in no_overlap:
                print(
                    f"    - [{i}] {name}: ref=[{rmin:.3f},{rmax:.3f}], pol=[{pmin:.3f},{pmax:.3f}]"
                )

        if height_issues:
            print("\n  Root height issues:")
            for issue in height_issues:
                print(f"    - {issue}")

        print("\n" + "=" * 80)
        print("RECOMMENDED FIXES:")
        print("=" * 80)
        print("  1. Ensure reference data uses SAME feature extraction code as policy")
        print(
            "  2. Check that root_height comes from env.info, not obs gravity component"
        )
        print("  3. Verify frame rates match (reference Hz == simulation Hz)")
        print("  4. Re-generate reference data with corrected feature extraction")

    elif moderate_leaks or temporal_issues:
        print("🟡 MODERATE LEAKS - may cause elevated discriminator accuracy")
        if moderate_leaks:
            for i, name, es, ref_mean, pol_mean in moderate_leaks:
                print(f"    - [{i}] {name}: effect={es:.2f}")
        if temporal_issues:
            print(
                "\n  Temporal correlation mismatches detected (possible frame rate issue)"
            )

    else:
        print("✓ No significant feature leaks detected!")
        print()
        print("  If discriminator still hits 100% accuracy, the issue may be:")
        print("  - Multi-feature correlations (harder to detect)")
        print("  - Discriminator overfitting to noise")
        print("  - Insufficient regularization (try increasing r1_gamma)")
        print("  - Discriminator architecture too powerful")

    return {
        "severe_leaks": severe_leaks,
        "moderate_leaks": moderate_leaks,
        "no_overlap": no_overlap,
        "height_issues": height_issues,
        "temporal_issues": temporal_issues,
        "constant_issues": constant_issues,
        "range_issues": range_issues,
    }


if __name__ == "__main__":
    print("=" * 80)
    print("AMP FEATURE DIAGNOSTIC TOOL")
    print("Checking for Golden Rule violations...")
    print("=" * 80)
    print()

    print("Loading reference features...")
    ref_features = load_reference_features()

    print("Collecting policy features (this may take a minute)...")
    policy_features = collect_policy_features(num_samples=1000)

    results = analyze_feature_leak(ref_features, policy_features, save_plots=True)

    # Exit code based on severity
    if results["severe_leaks"] or results["no_overlap"] or results["height_issues"]:
        sys.exit(1)  # Critical issues found
    elif results["moderate_leaks"] or results["temporal_issues"]:
        sys.exit(0)  # Warnings only
    else:
        sys.exit(0)  # All clear
