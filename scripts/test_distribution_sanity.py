#!/usr/bin/env python3
"""Test 6: Distribution Sanity Test.

v0.11.0: Validates that reference data distribution is plausible and
reachable by policy.

This test ensures:
1. Reference features are not degenerate (constant values)
2. Contact rates are physically plausible (not always on/off)
3. Root height is in reasonable range
4. Velocities are bounded (no numerical explosions)
5. Joint positions are within limits
6. Reference vs random policy distributions overlap reasonably

Usage:
    cd ~/projects/wildrobot
    uv run python scripts/test_distribution_sanity.py
    uv run python scripts/test_distribution_sanity.py --reference data/amp/walking_medium01_amp.pkl
    uv run python scripts/test_distribution_sanity.py --verbose
"""

import argparse
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mujoco
import numpy as np
from rich import print
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

from playground_amp.amp.policy_features import (
    FeatureConfig,
    create_config_from_robot,
)
from playground_amp.configs.training_config import get_robot_config, load_robot_config

console = Console()


# =============================================================================
# Test Configuration
# =============================================================================


@dataclass
class DistributionSanityConfig:
    """Configuration for distribution sanity tests."""

    # Feature dimension layout (27-dim physics reference)
    n_joints: int = 8

    # Feature indices (0-indexed)
    joint_pos_start: int = 0
    joint_pos_end: int = 8
    joint_vel_start: int = 8
    joint_vel_end: int = 16
    root_lin_vel_start: int = 16
    root_lin_vel_end: int = 19
    root_ang_vel_start: int = 19
    root_ang_vel_end: int = 22
    root_height_idx: int = 22
    foot_contacts_start: int = 23
    foot_contacts_end: int = 27

    # Sanity gate thresholds
    min_std_threshold: float = 0.001  # No dimension should be near-constant
    contact_rate_min: float = 0.02  # Contact rate lower bound (relaxed)
    contact_rate_max: float = 0.98  # Contact rate upper bound (relaxed)

    # Root height range (meters)
    height_min: float = 0.30
    height_max: float = 0.60

    # Velocity bounds
    lin_vel_max: float = 3.0  # m/s (relaxed for dynamic motions)
    ang_vel_max: float = 15.0  # rad/s (relaxed for dynamic motions)
    joint_vel_max: float = 25.0  # rad/s

    # Joint position limits (approximate for wildrobot)
    joint_pos_min: float = -3.14  # rad
    joint_pos_max: float = 3.14  # rad

    # Random rollout parameters
    num_random_frames: int = 1000
    random_seed: int = 42

    # Wasserstein distance thresholds (warning level)
    wasserstein_warn_threshold: float = 2.0


@dataclass
class SanityTestResult:
    """Result from a single sanity test."""

    test_name: str
    passed: bool
    severity: str  # "PASS", "WARN", "FAIL"
    details: str
    metrics: Dict[str, float] = field(default_factory=dict)


# =============================================================================
# Feature Group Names
# =============================================================================

FEATURE_NAMES = {
    0: "joint_pos[0] (L_hip_pitch)",
    1: "joint_pos[1] (L_hip_roll)",
    2: "joint_pos[2] (L_knee)",
    3: "joint_pos[3] (L_ankle)",
    4: "joint_pos[4] (R_hip_pitch)",
    5: "joint_pos[5] (R_hip_roll)",
    6: "joint_pos[6] (R_knee)",
    7: "joint_pos[7] (R_ankle)",
    8: "joint_vel[0]",
    9: "joint_vel[1]",
    10: "joint_vel[2]",
    11: "joint_vel[3]",
    12: "joint_vel[4]",
    13: "joint_vel[5]",
    14: "joint_vel[6]",
    15: "joint_vel[7]",
    16: "root_lin_vel_x",
    17: "root_lin_vel_y",
    18: "root_lin_vel_z",
    19: "root_ang_vel_x",
    20: "root_ang_vel_y",
    21: "root_ang_vel_z",
    22: "root_height",
    23: "foot_contact[0] (L_toe)",
    24: "foot_contact[1] (L_heel)",
    25: "foot_contact[2] (R_toe)",
    26: "foot_contact[3] (R_heel)",
}


# =============================================================================
# Helper Functions
# =============================================================================


def load_reference_features(ref_path: Path) -> np.ndarray:
    """Load features from a physics reference file.

    Args:
        ref_path: Path to reference .pkl file

    Returns:
        Features array of shape (N, D)
    """
    with open(ref_path, "rb") as f:
        data = pickle.load(f)

    if "features" in data:
        return data["features"]
    else:
        raise ValueError(f"No 'features' key found in {ref_path}")


def load_all_reference_features(amp_dir: Path) -> Tuple[np.ndarray, List[str]]:
    """Load and concatenate features from all reference files.

    Args:
        amp_dir: Directory containing *_amp.pkl files

    Returns:
        Tuple of (concatenated features, list of clip names)
    """
    all_features = []
    clip_names = []

    for pkl_path in sorted(amp_dir.glob("*_amp.pkl")):
        try:
            features = load_reference_features(pkl_path)
            all_features.append(features)
            clip_names.append(pkl_path.stem)
        except Exception as e:
            print(f"[yellow]Warning: Failed to load {pkl_path}: {e}[/yellow]")

    if not all_features:
        raise ValueError(f"No valid reference files found in {amp_dir}")

    return np.concatenate(all_features, axis=0), clip_names


def run_random_policy_rollout(
    mj_model: mujoco.MjModel,
    config: DistributionSanityConfig,
    amp_config: FeatureConfig,
) -> np.ndarray:
    """Run random policy rollout to collect feature samples.

    This simulates what a random/untrained policy might produce,
    providing a baseline for distribution comparison.

    Args:
        mj_model: MuJoCo model
        config: Test configuration
        amp_config: AMP feature configuration

    Returns:
        Features array of shape (N, D)
    """
    np.random.seed(config.random_seed)

    mj_data = mujoco.MjData(mj_model)
    n = amp_config.num_actuated_joints

    sim_dt = 0.002
    control_dt = 0.02
    n_substeps = int(control_dt / sim_dt)

    # Get foot geom IDs
    left_toe_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "left_toe")
    right_toe_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "right_toe")

    all_features = []

    # Reset to home position
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
    mj_data.qvel[:] = 0
    mujoco.mj_forward(mj_model, mj_data)

    for _ in range(config.num_random_frames):
        # Random control action (within actuator range)
        ctrl_range = mj_model.actuator_ctrlrange[:n]
        random_ctrl = np.random.uniform(ctrl_range[:, 0], ctrl_range[:, 1])
        mj_data.ctrl[:n] = random_ctrl

        # Step physics
        for _ in range(n_substeps):
            mujoco.mj_step(mj_model, mj_data)

        # Check for simulation instability
        if np.any(np.isnan(mj_data.qpos)) or np.any(np.abs(mj_data.qpos) > 100):
            # Reset and continue
            mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
            mj_data.qvel[:] = 0
            mujoco.mj_forward(mj_model, mj_data)
            continue

        # Extract contacts
        left_fn, right_fn = 0.0, 0.0
        for contact_idx in range(mj_data.ncon):
            contact = mj_data.contact[contact_idx]
            geom1, geom2 = contact.geom1, contact.geom2

            foot_geom = None
            if geom1 == 0 and geom2 in [left_toe_id, right_toe_id]:
                foot_geom = geom2
            elif geom2 == 0 and geom1 in [left_toe_id, right_toe_id]:
                foot_geom = geom1

            if foot_geom is not None and contact.efc_address >= 0:
                fn = mj_data.efc_force[contact.efc_address]
                if fn > 0:
                    if foot_geom == left_toe_id:
                        left_fn += fn
                    else:
                        right_fn += fn

        # Contact states (simplified: toe-only for now)
        contact_threshold = 0.5
        contacts = [
            1.0 if left_fn > contact_threshold else 0.0,
            1.0 if left_fn > contact_threshold else 0.0,  # heel same as toe
            1.0 if right_fn > contact_threshold else 0.0,
            1.0 if right_fn > contact_threshold else 0.0,  # heel same as toe
        ]

        # Extract state
        qpos = mj_data.qpos.copy()
        qvel = mj_data.qvel.copy()

        # Convert to heading-local velocities
        root_quat_wxyz = qpos[3:7]
        x, y, z, w = (
            root_quat_wxyz[1],
            root_quat_wxyz[2],
            root_quat_wxyz[3],
            root_quat_wxyz[0],
        )
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)

        root_lin_vel_world = qvel[0:3]
        root_ang_vel_world = qvel[3:6]

        lin_vel_heading = np.array(
            [
                cos_yaw * root_lin_vel_world[0] + sin_yaw * root_lin_vel_world[1],
                -sin_yaw * root_lin_vel_world[0] + cos_yaw * root_lin_vel_world[1],
                root_lin_vel_world[2],
            ]
        )
        ang_vel_heading = np.array(
            [
                cos_yaw * root_ang_vel_world[0] + sin_yaw * root_ang_vel_world[1],
                -sin_yaw * root_ang_vel_world[0] + cos_yaw * root_ang_vel_world[1],
                root_ang_vel_world[2],
            ]
        )

        # Assemble features (27-dim)
        features = np.concatenate(
            [
                qpos[7 : 7 + n],  # joint_pos
                qvel[6 : 6 + n],  # joint_vel
                lin_vel_heading,  # root_lin_vel
                ang_vel_heading,  # root_ang_vel
                [qpos[2]],  # root_height
                contacts,  # foot_contacts
            ]
        )
        all_features.append(features)

    return np.array(all_features, dtype=np.float32)


def compute_wasserstein_1d(p: np.ndarray, q: np.ndarray) -> float:
    """Compute 1D Wasserstein distance between two distributions.

    Uses sorted samples to compute the Earth Mover's Distance.

    Args:
        p: Samples from distribution P
        q: Samples from distribution Q

    Returns:
        Wasserstein-1 distance
    """
    p_sorted = np.sort(p)
    q_sorted = np.sort(q)

    # Interpolate to same size if needed
    n = min(len(p_sorted), len(q_sorted))
    p_interp = np.interp(
        np.linspace(0, 1, n), np.linspace(0, 1, len(p_sorted)), p_sorted
    )
    q_interp = np.interp(
        np.linspace(0, 1, n), np.linspace(0, 1, len(q_sorted)), q_sorted
    )

    return float(np.mean(np.abs(p_interp - q_interp)))


# =============================================================================
# Sanity Tests
# =============================================================================


def test_feature_variance(
    features: np.ndarray,
    config: DistributionSanityConfig,
) -> SanityTestResult:
    """Test 6.1: Check that no feature dimension is near-constant.

    Near-constant features indicate degenerate data that won't train well.
    """
    stds = features.std(axis=0)

    # Find problematic dimensions
    constant_dims = []
    for i, std in enumerate(stds):
        if std < config.min_std_threshold:
            constant_dims.append((i, FEATURE_NAMES.get(i, f"dim[{i}]"), std))

    # Skip contact dimensions (they can legitimately be near-constant for some motions)
    contact_start = config.foot_contacts_start
    non_contact_constant = [d for d in constant_dims if d[0] < contact_start]

    passed = len(non_contact_constant) == 0

    if passed:
        details = f"All {features.shape[1]} dimensions have sufficient variance (min_std={stds.min():.4f})"
        severity = "PASS"
    else:
        details = f"{len(non_contact_constant)} dimensions near-constant:\n"
        for idx, name, std in non_contact_constant[:5]:
            details += f"  - {name}: std={std:.6f}\n"
        severity = "FAIL"

    return SanityTestResult(
        test_name="Feature Variance",
        passed=passed,
        severity=severity,
        details=details.strip(),
        metrics={
            "min_std": float(stds.min()),
            "max_std": float(stds.max()),
            "mean_std": float(stds.mean()),
            "num_constant_dims": len(non_contact_constant),
        },
    )


def test_contact_rates(
    features: np.ndarray,
    config: DistributionSanityConfig,
) -> SanityTestResult:
    """Test 6.2: Check that contact rates are not degenerate.

    Contacts that are always on or always off indicate unrealistic data.
    """
    contacts = features[:, config.foot_contacts_start : config.foot_contacts_end]
    contact_rates = contacts.mean(axis=0)

    issues = []
    warnings = []

    channel_names = ["L_toe", "L_heel", "R_toe", "R_heel"]

    for i, (rate, name) in enumerate(zip(contact_rates, channel_names)):
        if rate < config.contact_rate_min:
            if rate < 0.01:
                issues.append(f"{name}: {rate:.1%} (always off)")
            else:
                warnings.append(f"{name}: {rate:.1%} (low)")
        elif rate > config.contact_rate_max:
            if rate > 0.99:
                issues.append(f"{name}: {rate:.1%} (always on)")
            else:
                warnings.append(f"{name}: {rate:.1%} (high)")

    if issues:
        passed = False
        severity = "FAIL"
        details = "Degenerate contact rates:\n  " + "\n  ".join(issues)
        if warnings:
            details += "\n  Warnings: " + ", ".join(warnings)
    elif warnings:
        passed = True
        severity = "WARN"
        details = "Contact rates borderline:\n  " + "\n  ".join(warnings)
    else:
        passed = True
        severity = "PASS"
        details = f"Contact rates healthy: L_toe={contact_rates[0]:.1%}, L_heel={contact_rates[1]:.1%}, R_toe={contact_rates[2]:.1%}, R_heel={contact_rates[3]:.1%}"

    return SanityTestResult(
        test_name="Contact Rates",
        passed=passed,
        severity=severity,
        details=details,
        metrics={
            "L_toe_rate": float(contact_rates[0]),
            "L_heel_rate": float(contact_rates[1]),
            "R_toe_rate": float(contact_rates[2]),
            "R_heel_rate": float(contact_rates[3]),
        },
    )


def test_root_height(
    features: np.ndarray,
    config: DistributionSanityConfig,
) -> SanityTestResult:
    """Test 6.3: Check that root height is in reasonable range.

    Heights outside [0.30, 0.60] m suggest unrealistic poses.
    """
    heights = features[:, config.root_height_idx]

    height_mean = heights.mean()
    height_std = heights.std()
    height_min = heights.min()
    height_max = heights.max()

    issues = []

    # Check if mean is in reasonable range
    if height_mean < config.height_min:
        issues.append(f"Mean height {height_mean:.3f}m below {config.height_min}m")
    if height_mean > config.height_max:
        issues.append(f"Mean height {height_mean:.3f}m above {config.height_max}m")

    # Check for extreme values
    pct_below = (heights < config.height_min).mean()
    pct_above = (heights > config.height_max).mean()

    if pct_below > 0.1:
        issues.append(f"{pct_below:.1%} frames below {config.height_min}m")
    if pct_above > 0.1:
        issues.append(f"{pct_above:.1%} frames above {config.height_max}m")

    passed = len(issues) == 0

    if passed:
        details = f"Height healthy: mean={height_mean:.3f}m, std={height_std:.3f}m, range=[{height_min:.3f}, {height_max:.3f}]"
        severity = "PASS"
    else:
        details = "Height issues:\n  " + "\n  ".join(issues)
        severity = "FAIL"

    return SanityTestResult(
        test_name="Root Height",
        passed=passed,
        severity=severity,
        details=details,
        metrics={
            "height_mean": float(height_mean),
            "height_std": float(height_std),
            "height_min": float(height_min),
            "height_max": float(height_max),
            "pct_below_min": float(pct_below),
            "pct_above_max": float(pct_above),
        },
    )


def test_velocity_bounds(
    features: np.ndarray,
    config: DistributionSanityConfig,
) -> SanityTestResult:
    """Test 6.4: Check that velocities are bounded (no numerical explosions).

    Extremely high velocities indicate simulation instability or data errors.
    """
    joint_vel = features[:, config.joint_vel_start : config.joint_vel_end]
    root_lin_vel = features[:, config.root_lin_vel_start : config.root_lin_vel_end]
    root_ang_vel = features[:, config.root_ang_vel_start : config.root_ang_vel_end]

    issues = []
    metrics = {}

    # Joint velocities
    joint_vel_max = np.abs(joint_vel).max()
    metrics["joint_vel_max"] = float(joint_vel_max)
    if joint_vel_max > config.joint_vel_max:
        issues.append(
            f"Joint vel max {joint_vel_max:.2f} rad/s exceeds {config.joint_vel_max}"
        )

    # Root linear velocity
    lin_vel_mag = np.linalg.norm(root_lin_vel, axis=1)
    lin_vel_max = lin_vel_mag.max()
    metrics["lin_vel_max"] = float(lin_vel_max)
    if lin_vel_max > config.lin_vel_max:
        issues.append(f"Lin vel max {lin_vel_max:.2f} m/s exceeds {config.lin_vel_max}")

    # Root angular velocity
    ang_vel_mag = np.linalg.norm(root_ang_vel, axis=1)
    ang_vel_max = ang_vel_mag.max()
    metrics["ang_vel_max"] = float(ang_vel_max)
    if ang_vel_max > config.ang_vel_max:
        issues.append(
            f"Ang vel max {ang_vel_max:.2f} rad/s exceeds {config.ang_vel_max}"
        )

    # Check for NaN/Inf
    if np.any(np.isnan(features)):
        issues.append("Features contain NaN values")
        metrics["has_nan"] = 1.0
    else:
        metrics["has_nan"] = 0.0

    if np.any(np.isinf(features)):
        issues.append("Features contain Inf values")
        metrics["has_inf"] = 1.0
    else:
        metrics["has_inf"] = 0.0

    passed = len(issues) == 0

    if passed:
        details = f"Velocities bounded: joint<{joint_vel_max:.1f}, lin<{lin_vel_max:.2f}, ang<{ang_vel_max:.1f}"
        severity = "PASS"
    else:
        details = "Velocity issues:\n  " + "\n  ".join(issues)
        severity = "FAIL"

    return SanityTestResult(
        test_name="Velocity Bounds",
        passed=passed,
        severity=severity,
        details=details,
        metrics=metrics,
    )


def test_joint_limits(
    features: np.ndarray,
    config: DistributionSanityConfig,
) -> SanityTestResult:
    """Test 6.5: Check that joint positions are within limits.

    Positions outside joint limits suggest data errors or unrealistic poses.
    """
    joint_pos = features[:, config.joint_pos_start : config.joint_pos_end]

    joint_min = joint_pos.min(axis=0)
    joint_max = joint_pos.max(axis=0)

    issues = []

    for i in range(config.n_joints):
        if joint_min[i] < config.joint_pos_min:
            issues.append(
                f"Joint {i} min {joint_min[i]:.2f} below limit {config.joint_pos_min}"
            )
        if joint_max[i] > config.joint_pos_max:
            issues.append(
                f"Joint {i} max {joint_max[i]:.2f} above limit {config.joint_pos_max}"
            )

    passed = len(issues) == 0

    if passed:
        details = f"Joint positions within limits: range=[{joint_pos.min():.2f}, {joint_pos.max():.2f}]"
        severity = "PASS"
    else:
        details = "Joint limit violations:\n  " + "\n  ".join(issues[:5])
        if len(issues) > 5:
            details += f"\n  ... and {len(issues)-5} more"
        severity = "FAIL"

    return SanityTestResult(
        test_name="Joint Limits",
        passed=passed,
        severity=severity,
        details=details,
        metrics={
            "joint_pos_min": float(joint_pos.min()),
            "joint_pos_max": float(joint_pos.max()),
            "num_violations": len(issues),
        },
    )


def test_distribution_overlap(
    ref_features: np.ndarray,
    random_features: np.ndarray,
    config: DistributionSanityConfig,
) -> SanityTestResult:
    """Test 6.6: Check distribution overlap between reference and random policy.

    Large Wasserstein distances indicate distributions that may be hard to match.
    This is informational - we expect some difference but not extreme.
    """
    # Compute per-group Wasserstein distances
    groups = {
        "joint_pos": (config.joint_pos_start, config.joint_pos_end),
        "joint_vel": (config.joint_vel_start, config.joint_vel_end),
        "root_lin_vel": (config.root_lin_vel_start, config.root_lin_vel_end),
        "root_ang_vel": (config.root_ang_vel_start, config.root_ang_vel_end),
        "root_height": (config.root_height_idx, config.root_height_idx + 1),
    }

    metrics = {}
    warnings = []

    for group_name, (start, end) in groups.items():
        ref_group = ref_features[:, start:end].flatten()
        rand_group = random_features[:, start:end].flatten()

        # Normalize by reference std to make distances comparable
        ref_std = ref_group.std()
        if ref_std > 0.001:
            w_dist = compute_wasserstein_1d(ref_group, rand_group) / ref_std
        else:
            w_dist = 0.0

        metrics[f"wasserstein_{group_name}"] = float(w_dist)

        if w_dist > config.wasserstein_warn_threshold:
            warnings.append(f"{group_name}: W={w_dist:.2f}")

    # This test is informational - always passes but may warn
    passed = True

    if warnings:
        severity = "WARN"
        details = (
            f"Large distribution gaps (W > {config.wasserstein_warn_threshold}):\n  "
            + "\n  ".join(warnings)
        )
        details += "\n  Note: Some gap expected between reference and random policy"
    else:
        severity = "PASS"
        details = "Distribution overlap reasonable for all feature groups"

    return SanityTestResult(
        test_name="Distribution Overlap",
        passed=passed,
        severity=severity,
        details=details,
        metrics=metrics,
    )


# =============================================================================
# Main Test Runner
# =============================================================================


def run_sanity_tests(
    ref_features: np.ndarray,
    random_features: Optional[np.ndarray],
    config: DistributionSanityConfig,
    verbose: bool = False,
) -> Tuple[List[SanityTestResult], bool]:
    """Run all distribution sanity tests.

    Args:
        ref_features: Reference features (N, D)
        random_features: Random policy features (N, D), optional
        config: Test configuration
        verbose: Print detailed output

    Returns:
        Tuple of (list of results, overall pass/fail)
    """
    print(f"\n[bold cyan]Test 6: Distribution Sanity[/bold cyan]")
    print(f"  Reference frames: {len(ref_features)}")
    print(f"  Feature dimensions: {ref_features.shape[1]}")
    if random_features is not None:
        print(f"  Random policy frames: {len(random_features)}")

    results = []

    # Test 6.1: Feature variance
    result = test_feature_variance(ref_features, config)
    results.append(result)

    # Test 6.2: Contact rates
    result = test_contact_rates(ref_features, config)
    results.append(result)

    # Test 6.3: Root height
    result = test_root_height(ref_features, config)
    results.append(result)

    # Test 6.4: Velocity bounds
    result = test_velocity_bounds(ref_features, config)
    results.append(result)

    # Test 6.5: Joint limits
    result = test_joint_limits(ref_features, config)
    results.append(result)

    # Test 6.6: Distribution overlap (only if random features available)
    if random_features is not None:
        result = test_distribution_overlap(ref_features, random_features, config)
        results.append(result)

    # Print results table
    table = Table(title="Test 6: Distribution Sanity Results")
    table.add_column("Test", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Details")

    all_passed = True
    for result in results:
        if result.severity == "PASS":
            status = "[green]✓ PASS[/green]"
        elif result.severity == "WARN":
            status = "[yellow]⚠ WARN[/yellow]"
        else:
            status = "[red]✗ FAIL[/red]"
            all_passed = False

        # Truncate long details for table
        details_short = result.details.split("\n")[0]
        if len(details_short) > 60:
            details_short = details_short[:57] + "..."

        table.add_row(result.test_name, status, details_short)

    console.print(table)

    # Verbose output
    if verbose:
        print("\n[bold]Detailed Metrics:[/bold]")
        for result in results:
            print(f"\n  {result.test_name}:")
            print(f"    Status: {result.severity}")
            print(f"    Details: {result.details}")
            for key, val in result.metrics.items():
                if isinstance(val, float):
                    print(f"    {key}: {val:.6f}")
                else:
                    print(f"    {key}: {val}")

    # Summary
    num_pass = sum(1 for r in results if r.severity == "PASS")
    num_warn = sum(1 for r in results if r.severity == "WARN")
    num_fail = sum(1 for r in results if r.severity == "FAIL")

    if all_passed:
        if num_warn > 0:
            print(
                f"\n[bold yellow]⚠ Test 6 PASSED with {num_warn} warning(s)[/bold yellow]"
            )
        else:
            print(
                f"\n[bold green]✓ Test 6 PASSED: Reference distribution is sane[/bold green]"
            )
    else:
        print(
            f"\n[bold red]✗ Test 6 FAILED: {num_fail} sanity check(s) failed[/bold red]"
        )
        print("  Review reference data generation for issues.")

    return results, all_passed


def run_per_clip_analysis(
    amp_dir: Path,
    config: DistributionSanityConfig,
) -> Dict[str, List[SanityTestResult]]:
    """Run sanity tests on each clip individually.

    Args:
        amp_dir: Directory containing *_amp.pkl files
        config: Test configuration

    Returns:
        Dict mapping clip name to test results
    """
    print(f"\n[bold cyan]Test 6: Per-Clip Analysis[/bold cyan]")

    all_results = {}

    for pkl_path in sorted(amp_dir.glob("*_amp.pkl")):
        try:
            features = load_reference_features(pkl_path)
            results, passed = run_sanity_tests(features, None, config, verbose=False)
            all_results[pkl_path.stem] = results
        except Exception as e:
            print(f"[red]Error processing {pkl_path}: {e}[/red]")

    # Summary table
    print(f"\n[bold]Per-Clip Summary:[/bold]")
    table = Table(title="Test 6: Per-Clip Results")
    table.add_column("Clip", style="cyan")
    table.add_column("Frames")
    table.add_column("Variance")
    table.add_column("Contacts")
    table.add_column("Height")
    table.add_column("Velocity")
    table.add_column("Joints")

    for clip_name, results in all_results.items():
        row = [clip_name]

        # Get frame count
        pkl_path = amp_dir / f"{clip_name}.pkl"
        try:
            features = load_reference_features(pkl_path)
            row.append(str(len(features)))
        except:
            row.append("?")

        for result in results[:5]:  # First 5 tests (not distribution overlap)
            if result.severity == "PASS":
                row.append("[green]✓[/green]")
            elif result.severity == "WARN":
                row.append("[yellow]⚠[/yellow]")
            else:
                row.append("[red]✗[/red]")

        table.add_row(*row)

    console.print(table)

    return all_results


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Test 6: Distribution Sanity")
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        help="Single reference file to test (default: batch test all in data/amp)",
    )
    parser.add_argument(
        "--amp-dir",
        type=str,
        default="data/amp",
        help="Directory containing *_amp.pkl reference files",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="assets/scene_flat_terrain.xml",
        help="MuJoCo scene XML for random rollouts",
    )
    parser.add_argument(
        "--robot-config",
        type=str,
        default="assets/robot_config.yaml",
        help="Robot config YAML",
    )
    parser.add_argument(
        "--skip-random",
        action="store_true",
        help="Skip random policy rollout comparison",
    )
    parser.add_argument(
        "--per-clip",
        action="store_true",
        help="Run analysis on each clip individually",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed metrics",
    )

    args = parser.parse_args()

    # Load robot config
    print(f"[bold blue]Loading robot config:[/bold blue] {args.robot_config}")
    load_robot_config(args.robot_config)
    robot_config = get_robot_config()
    amp_config = create_config_from_robot(robot_config)

    # Test configuration
    config = DistributionSanityConfig()

    # Per-clip analysis mode
    if args.per_clip:
        amp_dir = Path(args.amp_dir)
        all_results = run_per_clip_analysis(amp_dir, config)

        # Count overall pass/fail
        num_clips = len(all_results)
        num_fail_clips = sum(
            1
            for results in all_results.values()
            if any(r.severity == "FAIL" for r in results)
        )

        if num_fail_clips > 0:
            print(
                f"\n[bold red]✗ {num_fail_clips}/{num_clips} clips have sanity issues[/bold red]"
            )
            sys.exit(1)
        else:
            print(
                f"\n[bold green]✓ All {num_clips} clips pass sanity checks[/bold green]"
            )
            sys.exit(0)

    # Load reference features
    if args.reference:
        ref_path = Path(args.reference)
        print(f"\n[bold blue]Loading reference:[/bold blue] {ref_path}")
        ref_features = load_reference_features(ref_path)
        clip_names = [ref_path.stem]
    else:
        amp_dir = Path(args.amp_dir)
        print(f"\n[bold blue]Loading all references from:[/bold blue] {amp_dir}")
        ref_features, clip_names = load_all_reference_features(amp_dir)
        print(
            f"  Loaded {len(clip_names)} clips: {', '.join(clip_names[:5])}{'...' if len(clip_names) > 5 else ''}"
        )

    # Run random policy rollouts for comparison (optional)
    random_features = None
    if not args.skip_random:
        print(f"\n[bold blue]Running random policy rollout...[/bold blue]")
        model_path = Path(args.model)

        from playground_amp.envs.wildrobot_env import get_assets

        mj_model = mujoco.MjModel.from_xml_string(
            model_path.read_text(), assets=get_assets(model_path.parent)
        )

        random_features = run_random_policy_rollout(mj_model, config, amp_config)
        print(f"  Collected {len(random_features)} random frames")

    # Run sanity tests
    results, passed = run_sanity_tests(
        ref_features, random_features, config, verbose=args.verbose
    )

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
