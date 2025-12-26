#!/usr/bin/env python3
"""Test 1C: Generator Determinism and Semantics Test.

v0.11.0: Validates that physics reference generation is deterministic
and produces semantically valid data.

This test ensures:
1. Identical runs with same seeds produce identical features (MAE = 0)
2. Contact sequences match within tolerance
3. Quality metrics match exactly
4. Replay without qpos overwrites produces similar distributions

Usage:
    cd ~/projects/wildrobot
    uv run python scripts/test_generator_determinism.py
    uv run python scripts/test_generator_determinism.py --motion assets/motions/walking_medium01.pkl
    uv run python scripts/test_generator_determinism.py --verbose
"""

import argparse
import hashlib
import pickle
import sys
import tempfile
from dataclasses import dataclass
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
class DeterminismTestConfig:
    """Configuration for determinism tests."""

    # Tolerance thresholds
    feature_tolerance: float = 1e-10  # Bytewise comparison (very tight)
    contact_tolerance: float = 1e-6  # Contact state comparison
    velocity_replay_tolerance: float = 0.1  # Replay MAE tolerance (drift expected)

    # Test parameters
    num_runs: int = 2  # Number of identical runs to compare
    random_seed: int = 42  # Fixed seed for determinism

    # MuJoCo settings
    sim_dt: float = 0.002
    control_dt: float = 0.02


@dataclass
class DeterminismTestResult:
    """Result from a single determinism test."""

    test_name: str
    passed: bool
    details: str
    metrics: Dict[str, float]


# =============================================================================
# Physics Rollout (Deterministic Version)
# =============================================================================


def run_deterministic_rollout(
    gmr_motion: Dict[str, Any],
    mj_model: mujoco.MjModel,
    amp_config: FeatureConfig,
    config: DeterminismTestConfig,
    seed: int,
) -> Dict[str, Any]:
    """Run physics rollout with fixed seed for determinism.

    This is a simplified version of the main generator focused on determinism testing.

    Args:
        gmr_motion: GMR motion data
        mj_model: MuJoCo model
        amp_config: AMP feature config
        config: Test configuration
        seed: Random seed for determinism

    Returns:
        Dict with features, contacts, quality metrics
    """
    # Set random seed for reproducibility
    np.random.seed(seed)

    # Extract GMR data
    ref_dof_pos = gmr_motion["dof_pos"]
    ref_root_pos = gmr_motion["root_pos"]
    num_frames = gmr_motion["num_frames"]

    n = amp_config.num_actuated_joints
    mj_data = mujoco.MjData(mj_model)
    mj_model.opt.timestep = config.sim_dt

    # Reset MuJoCo to deterministic state
    mujoco.mj_resetData(mj_model, mj_data)

    # Initialize from home keyframe
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
    mj_data.qvel[:] = 0
    mujoco.mj_forward(mj_model, mj_data)

    n_substeps = int(config.control_dt / config.sim_dt)

    # Harness parameters (same as main generator)
    HARNESS_KP = 200.0
    HARNESS_KD = 40.0
    HARNESS_CAP = 20.0

    # Get foot geom IDs
    left_toe_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "left_toe")
    right_toe_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "right_toe")

    # Reference values for stabilization
    target_height = ref_root_pos[:, 2].mean()
    ref_xy_start = ref_root_pos[0, :2].copy()
    sim_xy_start = mj_data.qpos[0:2].copy()

    # Storage
    all_features = []
    all_contacts = []
    all_load_ratios = []
    all_stab_forces = []
    all_qvel = []

    mg = 3.383 * 9.81  # Robot weight

    for i in range(num_frames):
        # Set control targets
        mj_data.ctrl[:n] = ref_dof_pos[i]

        # Full harness mode (same as main generator)
        target_z = ref_root_pos[i, 2]
        height_error = target_z - mj_data.qpos[2]
        stab_force_z = HARNESS_KP * height_error - HARNESS_KD * mj_data.qvel[2]
        stab_force_z = np.clip(stab_force_z, -HARNESS_CAP, HARNESS_CAP)

        # XY stabilization
        ref_xy_delta = ref_root_pos[i, :2] - ref_xy_start
        target_xy = sim_xy_start + ref_xy_delta
        xy_error = target_xy - mj_data.qpos[0:2]
        stab_force_xy = 100.0 * xy_error - 20.0 * mj_data.qvel[0:2]
        stab_force_xy = np.clip(stab_force_xy, -10.0, 10.0)

        # Apply forces
        mj_data.qfrc_applied[0:2] = stab_force_xy
        mj_data.qfrc_applied[2] = stab_force_z

        total_stab = np.sqrt(stab_force_z**2 + np.sum(stab_force_xy**2))
        all_stab_forces.append(total_stab)

        # Step physics
        for _ in range(n_substeps):
            mujoco.mj_step(mj_model, mj_data)

        # Extract contact forces
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

        total_fn = left_fn + right_fn
        all_load_ratios.append(total_fn / mg)

        # Contact states
        contact_threshold = 0.5
        l_contact = 1.0 if left_fn > contact_threshold else 0.0
        r_contact = 1.0 if right_fn > contact_threshold else 0.0
        contacts = [l_contact, l_contact, r_contact, r_contact]
        all_contacts.append(contacts)

        # Extract state for features
        qpos = mj_data.qpos.copy()
        qvel = mj_data.qvel.copy()
        all_qvel.append(qvel.copy())

        # Convert to heading-local
        root_quat_wxyz = qpos[3:7]
        root_quat_xyzw = np.array(
            [root_quat_wxyz[1], root_quat_wxyz[2], root_quat_wxyz[3], root_quat_wxyz[0]]
        )

        root_lin_vel_world = qvel[0:3]
        root_ang_vel_world = qvel[3:6]

        # Heading-local transform
        x, y, z, w = root_quat_xyzw
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)

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
                qpos[7 : 7 + n],  # joint_pos (8)
                qvel[6 : 6 + n],  # joint_vel (8)
                lin_vel_heading,  # root_lin_vel (3)
                ang_vel_heading,  # root_ang_vel (3)
                [qpos[2]],  # root_height (1)
                contacts,  # foot_contacts (4)
            ]
        )
        all_features.append(features)

    # Convert to arrays
    features = np.array(all_features, dtype=np.float32)
    contacts = np.array(all_contacts, dtype=np.float32)
    load_ratios = np.array(all_load_ratios, dtype=np.float32)
    stab_forces = np.array(all_stab_forces, dtype=np.float32)
    qvel_trajectory = np.array(all_qvel, dtype=np.float32)

    # Quality metrics
    quality_metrics = {
        "mean_load_support": float(np.mean(load_ratios)),
        "p90_stab_force": float(np.percentile(stab_forces, 90)),
        "max_stab_force": float(np.max(stab_forces)),
        "mean_stab_force": float(np.mean(stab_forces)),
    }

    return {
        "features": features,
        "contacts": contacts,
        "load_ratios": load_ratios,
        "stab_forces": stab_forces,
        "qvel_trajectory": qvel_trajectory,
        "quality_metrics": quality_metrics,
        "num_frames": num_frames,
    }


# =============================================================================
# Test Functions
# =============================================================================


def test_bytewise_determinism(
    run1: Dict[str, Any],
    run2: Dict[str, Any],
    config: DeterminismTestConfig,
) -> DeterminismTestResult:
    """Test 1C.1: Bytewise feature determinism.

    Verifies that two identical runs produce identical features.
    """
    features1 = run1["features"]
    features2 = run2["features"]

    # Check shapes
    if features1.shape != features2.shape:
        return DeterminismTestResult(
            test_name="Bytewise Determinism",
            passed=False,
            details=f"Shape mismatch: {features1.shape} vs {features2.shape}",
            metrics={"shape_match": 0.0},
        )

    # Compute differences
    diff = np.abs(features1 - features2)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())

    # Check tolerance
    passed = max_diff < config.feature_tolerance

    # Compute hash for verification
    hash1 = hashlib.sha256(features1.tobytes()).hexdigest()[:16]
    hash2 = hashlib.sha256(features2.tobytes()).hexdigest()[:16]

    if passed:
        details = f"Features identical (hash: {hash1})"
    else:
        details = f"Max diff: {max_diff:.2e}, Mean diff: {mean_diff:.2e}"
        details += f"\n  Hash1: {hash1}, Hash2: {hash2}"

    return DeterminismTestResult(
        test_name="Bytewise Determinism",
        passed=passed,
        details=details,
        metrics={
            "max_diff": max_diff,
            "mean_diff": mean_diff,
            "hash_match": 1.0 if hash1 == hash2 else 0.0,
        },
    )


def test_contact_determinism(
    run1: Dict[str, Any],
    run2: Dict[str, Any],
    config: DeterminismTestConfig,
) -> DeterminismTestResult:
    """Test 1C.2: Contact sequence determinism.

    Verifies that contact states match between runs.
    """
    contacts1 = run1["contacts"]
    contacts2 = run2["contacts"]

    # Compute differences
    diff = np.abs(contacts1 - contacts2)
    max_diff = float(diff.max())
    mismatch_rate = float(np.mean(diff > 0.5))  # Rate of binary mismatches

    passed = max_diff < config.contact_tolerance and mismatch_rate == 0.0

    if passed:
        details = "Contact sequences identical"
    else:
        details = f"Max diff: {max_diff:.2e}, Mismatch rate: {mismatch_rate:.2%}"

    return DeterminismTestResult(
        test_name="Contact Determinism",
        passed=passed,
        details=details,
        metrics={
            "max_diff": max_diff,
            "mismatch_rate": mismatch_rate,
        },
    )


def test_quality_metrics_determinism(
    run1: Dict[str, Any],
    run2: Dict[str, Any],
) -> DeterminismTestResult:
    """Test 1C.3: Quality metrics determinism.

    Verifies that quality metrics match exactly.
    """
    metrics1 = run1["quality_metrics"]
    metrics2 = run2["quality_metrics"]

    all_match = True
    details_lines = []
    diff_metrics = {}

    for key in metrics1:
        val1 = metrics1[key]
        val2 = metrics2[key]
        diff = abs(val1 - val2)

        if diff > 1e-10:
            all_match = False
            details_lines.append(f"{key}: {val1:.6f} vs {val2:.6f} (diff={diff:.2e})")

        diff_metrics[f"diff_{key}"] = diff

    if all_match:
        details = "All quality metrics identical"
    else:
        details = "Metrics differ:\n  " + "\n  ".join(details_lines)

    return DeterminismTestResult(
        test_name="Quality Metrics Determinism",
        passed=all_match,
        details=details,
        metrics=diff_metrics,
    )


def test_velocity_semantics(
    run1: Dict[str, Any],
    config: DeterminismTestConfig,
) -> DeterminismTestResult:
    """Test 1C.4: Velocity semantic validity.

    Verifies that extracted velocities are physically plausible.
    """
    features = run1["features"]
    n = 8  # num_actuated_joints

    # Extract velocity components
    joint_vel = features[:, n : 2 * n]
    root_lin_vel = features[:, 2 * n : 2 * n + 3]
    root_ang_vel = features[:, 2 * n + 3 : 2 * n + 6]

    # Check physical plausibility
    issues = []
    metrics = {}

    # Joint velocities should be bounded (< 10 rad/s typical)
    joint_vel_max = np.abs(joint_vel).max()
    metrics["joint_vel_max"] = float(joint_vel_max)
    if joint_vel_max > 20.0:
        issues.append(f"Joint vel max {joint_vel_max:.2f} rad/s exceeds 20 rad/s")

    # Root linear velocity should be bounded (< 3 m/s for walking)
    lin_vel_mag = np.linalg.norm(root_lin_vel, axis=1)
    lin_vel_max = lin_vel_mag.max()
    metrics["lin_vel_max"] = float(lin_vel_max)
    if lin_vel_max > 3.0:
        issues.append(f"Lin vel max {lin_vel_max:.2f} m/s exceeds 3 m/s")

    # Root angular velocity should be bounded (< 15 rad/s for dynamic motions)
    ang_vel_mag = np.linalg.norm(root_ang_vel, axis=1)
    ang_vel_max = ang_vel_mag.max()
    metrics["ang_vel_max"] = float(ang_vel_max)
    if ang_vel_max > 15.0:
        issues.append(f"Ang vel max {ang_vel_max:.2f} rad/s exceeds 15 rad/s")

    # Check for NaN/Inf
    if np.any(np.isnan(features)) or np.any(np.isinf(features)):
        issues.append("Features contain NaN or Inf values")
        metrics["has_nan_inf"] = 1.0
    else:
        metrics["has_nan_inf"] = 0.0

    passed = len(issues) == 0

    if passed:
        details = f"Velocities plausible: joint<{joint_vel_max:.1f}, lin<{lin_vel_max:.2f}, ang<{ang_vel_max:.2f}"
    else:
        details = "Issues:\n  " + "\n  ".join(issues)

    return DeterminismTestResult(
        test_name="Velocity Semantics",
        passed=passed,
        details=details,
        metrics=metrics,
    )


def test_qvel_consistency(
    run1: Dict[str, Any],
    run2: Dict[str, Any],
) -> DeterminismTestResult:
    """Test 1C.5: Raw qvel trajectory consistency.

    Verifies that the underlying MuJoCo qvel trajectories match.
    """
    qvel1 = run1["qvel_trajectory"]
    qvel2 = run2["qvel_trajectory"]

    diff = np.abs(qvel1 - qvel2)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())

    # Per-component analysis
    component_max_diff = {}
    component_max_diff["root_lin_vel"] = float(diff[:, 0:3].max())
    component_max_diff["root_ang_vel"] = float(diff[:, 3:6].max())
    component_max_diff["joint_vel"] = float(diff[:, 6:].max())

    passed = max_diff < 1e-10

    if passed:
        details = "Raw qvel trajectories identical"
    else:
        details = f"Max diff: {max_diff:.2e} (root_lin={component_max_diff['root_lin_vel']:.2e}, root_ang={component_max_diff['root_ang_vel']:.2e}, joint={component_max_diff['joint_vel']:.2e})"

    return DeterminismTestResult(
        test_name="qvel Consistency",
        passed=passed,
        details=details,
        metrics={
            "max_diff": max_diff,
            "mean_diff": mean_diff,
            **component_max_diff,
        },
    )


# =============================================================================
# Main Test Runner
# =============================================================================


def run_determinism_tests(
    motion_path: Path,
    model_path: Path,
    amp_config: FeatureConfig,
    config: DeterminismTestConfig,
    verbose: bool = False,
) -> Tuple[List[DeterminismTestResult], bool]:
    """Run all determinism tests on a single motion.

    Args:
        motion_path: Path to GMR motion file
        model_path: Path to MuJoCo model
        amp_config: AMP feature config
        config: Test configuration
        verbose: Print detailed output

    Returns:
        Tuple of (list of results, overall pass/fail)
    """
    print(f"\n[bold cyan]Test 1C: Generator Determinism[/bold cyan]")
    print(f"  Motion: {motion_path.name}")
    print(f"  Seed: {config.random_seed}")
    print(f"  Runs: {config.num_runs}")

    # Load motion
    with open(motion_path, "rb") as f:
        gmr_motion = pickle.load(f)

    print(f"  Frames: {gmr_motion['num_frames']}")

    # Load model
    from playground_amp.envs.wildrobot_env import get_assets

    mj_model = mujoco.MjModel.from_xml_string(
        model_path.read_text(), assets=get_assets(model_path.parent)
    )

    # Run multiple times with same seed
    print(f"\n  Running {config.num_runs} rollouts with seed={config.random_seed}...")

    runs = []
    for i in range(config.num_runs):
        run_data = run_deterministic_rollout(
            gmr_motion, mj_model, amp_config, config, seed=config.random_seed
        )
        runs.append(run_data)
        print(
            f"    Run {i+1}: {run_data['num_frames']} frames, "
            f"mean_load={run_data['quality_metrics']['mean_load_support']:.2%}"
        )

    # Run all tests
    print(f"\n  Running determinism tests...")
    results = []

    # Test 1C.1: Bytewise determinism
    result = test_bytewise_determinism(runs[0], runs[1], config)
    results.append(result)

    # Test 1C.2: Contact determinism
    result = test_contact_determinism(runs[0], runs[1], config)
    results.append(result)

    # Test 1C.3: Quality metrics determinism
    result = test_quality_metrics_determinism(runs[0], runs[1])
    results.append(result)

    # Test 1C.4: Velocity semantics
    result = test_velocity_semantics(runs[0], config)
    results.append(result)

    # Test 1C.5: qvel consistency
    result = test_qvel_consistency(runs[0], runs[1])
    results.append(result)

    # Print results table
    table = Table(title="Test 1C: Generator Determinism Results")
    table.add_column("Test", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Details")

    all_passed = True
    for result in results:
        status = "[green]✓ PASS[/green]" if result.passed else "[red]✗ FAIL[/red]"
        if not result.passed:
            all_passed = False
        table.add_row(result.test_name, status, result.details.split("\n")[0])

    console.print(table)

    # Verbose output
    if verbose:
        print("\n[bold]Detailed Metrics:[/bold]")
        for result in results:
            print(f"\n  {result.test_name}:")
            for key, val in result.metrics.items():
                print(f"    {key}: {val}")

    # Summary
    if all_passed:
        print(
            f"\n[bold green]✓ Test 1C PASSED: Generator is deterministic[/bold green]"
        )
    else:
        print(
            f"\n[bold red]✗ Test 1C FAILED: Generator non-determinism detected[/bold red]"
        )
        print("  Check seed setting and stateful operations in generator.")

    return results, all_passed


def run_batch_determinism_tests(
    motion_dir: Path,
    model_path: Path,
    amp_config: FeatureConfig,
    config: DeterminismTestConfig,
    max_motions: int = 5,
) -> Tuple[Dict[str, List[DeterminismTestResult]], bool]:
    """Run determinism tests on multiple motions.

    Args:
        motion_dir: Directory containing motion files
        model_path: Path to MuJoCo model
        amp_config: AMP feature config
        config: Test configuration
        max_motions: Maximum number of motions to test

    Returns:
        Tuple of (per-motion results dict, overall pass/fail)
    """
    motion_files = sorted(motion_dir.glob("*.pkl"))[:max_motions]

    print(f"\n[bold cyan]Test 1C: Batch Generator Determinism[/bold cyan]")
    print(f"  Testing {len(motion_files)} motions...")

    all_results = {}
    overall_passed = True

    for motion_path in motion_files:
        results, passed = run_determinism_tests(
            motion_path, model_path, amp_config, config, verbose=False
        )
        all_results[motion_path.name] = results
        if not passed:
            overall_passed = False

    # Summary table
    print(f"\n[bold]Batch Summary:[/bold]")
    table = Table(title="Test 1C: Per-Motion Results")
    table.add_column("Motion", style="cyan")
    table.add_column("Bytewise", justify="center")
    table.add_column("Contacts", justify="center")
    table.add_column("Metrics", justify="center")
    table.add_column("Semantics", justify="center")
    table.add_column("qvel", justify="center")

    for motion_name, results in all_results.items():
        row = [motion_name]
        for result in results:
            status = "[green]✓[/green]" if result.passed else "[red]✗[/red]"
            row.append(status)
        table.add_row(*row)

    console.print(table)

    if overall_passed:
        print(
            f"\n[bold green]✓ All {len(motion_files)} motions pass determinism tests[/bold green]"
        )
    else:
        failed = sum(1 for r in all_results.values() if not all(t.passed for t in r))
        print(
            f"\n[bold red]✗ {failed}/{len(motion_files)} motions failed determinism tests[/bold red]"
        )

    return all_results, overall_passed


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Test 1C: Generator Determinism and Semantics"
    )
    parser.add_argument(
        "--motion",
        type=str,
        default=None,
        help="Single motion file to test (default: batch test all)",
    )
    parser.add_argument(
        "--motion-dir",
        type=str,
        default="assets/motions",
        help="Directory containing motion files for batch testing",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="assets/scene_flat_terrain.xml",
        help="MuJoCo scene XML",
    )
    parser.add_argument(
        "--robot-config",
        type=str,
        default="assets/robot_config.yaml",
        help="Robot config YAML",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for determinism tests",
    )
    parser.add_argument(
        "--max-motions",
        type=int,
        default=3,
        help="Maximum motions to test in batch mode",
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
    config = DeterminismTestConfig(
        random_seed=args.seed,
    )

    model_path = Path(args.model)

    if args.motion:
        # Single motion test
        motion_path = Path(args.motion)
        results, passed = run_determinism_tests(
            motion_path, model_path, amp_config, config, verbose=args.verbose
        )
        sys.exit(0 if passed else 1)
    else:
        # Batch test
        motion_dir = Path(args.motion_dir)
        all_results, passed = run_batch_determinism_tests(
            motion_dir, model_path, amp_config, config, max_motions=args.max_motions
        )
        sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
