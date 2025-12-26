#!/usr/bin/env python3
"""Test 7: Harness Contribution Gate Test.

v0.11.0: Validates that reference data meets Tier 0 load support requirements.

This test ensures:
1. Feet carry majority of body weight (not harness)
2. Harness contribution is bounded
3. Clips are physically grounded, not "puppeted"

Tier 0 Quality Gates (for m=3.383kg, mg≈33.2N):
- mean(ΣF_n) / mg >= 0.85  (contact forces carry 85%+ of weight)
- p90(|F_stab_z|) / mg <= 0.15 (harness contributes ≤15% at p90)
- unloaded_frames <= 10% (frames with ΣF_n < 0.6*mg are rare)

Usage:
    cd ~/projects/wildrobot
    uv run python scripts/test_harness_contribution.py
    uv run python scripts/test_harness_contribution.py --motion assets/motions/walking_medium01.pkl
    uv run python scripts/test_harness_contribution.py --verbose
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
class HarnessGateConfig:
    """Configuration for harness contribution tests."""

    # Robot parameters
    robot_mass: float = 3.383  # kg
    gravity: float = 9.81  # m/s^2

    @property
    def mg(self) -> float:
        return self.robot_mass * self.gravity

    # Tier 0 thresholds (strict - for training data)
    tier0_min_load_support: float = 0.85  # mean(ΣF_n)/mg >= 85%
    tier0_max_stab_force_p90: float = 0.15  # p90(|F_stab|)/mg <= 15%
    tier0_max_unloaded_rate: float = 0.10  # unloaded frames <= 10%

    # Tier 1 thresholds (relaxed - for debugging/curriculum)
    tier1_min_load_support: float = 0.60  # mean(ΣF_n)/mg >= 60%
    tier1_max_stab_force_p90: float = 0.40  # p90(|F_stab|)/mg <= 40%
    tier1_max_unloaded_rate: float = 0.30  # unloaded frames <= 30%

    # Load detection threshold
    unloaded_threshold: float = 0.6  # Frame is "unloaded" if ΣF_n < 0.6*mg

    # Simulation parameters
    sim_dt: float = 0.002
    control_dt: float = 0.02

    # Harness parameters (weak assist for Tier 0)
    harness_kp: float = 60.0  # N/m
    harness_kd: float = 10.0  # N·s/m
    harness_cap: float = 4.0  # N (~12% of mg)
    orient_gate: float = 0.436  # rad (25°) - disable if tilted


@dataclass
class HarnessGateResult:
    """Result from harness contribution test for a single clip."""

    clip_name: str
    tier: int  # 0, 1, or -1 (rejected)
    tier_reason: str

    # Load support metrics
    mean_load_support: float  # mean(ΣF_n) / mg
    p90_stab_force: float  # p90(|F_stab|) / mg
    max_stab_force: float  # max(|F_stab|) / mg
    mean_stab_force: float  # mean(|F_stab|) / mg
    unloaded_frame_rate: float  # fraction of frames with low contact force

    # Contact metrics
    left_contact_rate: float
    right_contact_rate: float
    both_contact_rate: float
    no_contact_rate: float

    # Height metrics
    height_mean: float
    height_std: float
    height_min: float
    height_max: float

    # Frame counts
    num_frames: int

    # Gate results
    gate_results: Dict[str, bool] = field(default_factory=dict)


# =============================================================================
# Physics Rollout with Quality Metrics
# =============================================================================


def run_physics_rollout_with_metrics(
    motion_path: Path,
    mj_model: mujoco.MjModel,
    config: HarnessGateConfig,
    amp_config: FeatureConfig,
) -> Dict[str, Any]:
    """Run physics rollout and collect quality metrics.

    Uses weak harness (Tier 0 settings) to allow feet to naturally
    support body weight.

    Args:
        motion_path: Path to GMR motion file
        mj_model: MuJoCo model
        config: Test configuration
        amp_config: AMP feature configuration

    Returns:
        Dict with quality metrics
    """
    # Load motion
    with open(motion_path, "rb") as f:
        motion = pickle.load(f)

    ref_dof_pos = motion["dof_pos"]
    ref_root_pos = motion["root_pos"]
    num_frames = motion["num_frames"]

    n = amp_config.num_actuated_joints
    mj_data = mujoco.MjData(mj_model)
    mj_model.opt.timestep = config.sim_dt

    # Get foot geom IDs
    left_toe_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "left_toe")
    right_toe_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "right_toe")
    ground_id = 0  # Ground plane is typically geom 0

    # Initialize from home keyframe
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
    mj_data.qvel[:] = 0
    mujoco.mj_forward(mj_model, mj_data)

    n_substeps = int(config.control_dt / config.sim_dt)

    # Reference values for stabilization
    target_height = ref_root_pos[:, 2].mean()
    ref_xy_start = ref_root_pos[0, :2].copy()
    sim_xy_start = mj_data.qpos[0:2].copy()

    # Storage for metrics
    all_sum_fn = []  # Total normal force per frame
    all_stab_force = []  # Harness force per frame
    all_left_contact = []  # Left foot contact per frame
    all_right_contact = []  # Right foot contact per frame
    all_heights = []  # Root height per frame

    mg = config.mg

    for i in range(num_frames):
        # Set control targets
        mj_data.ctrl[:n] = ref_dof_pos[i]

        # Compute orientation for gating
        w, x, y, z = mj_data.qpos[3:7]
        pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))

        # Apply soft height assist (only if upright)
        if abs(pitch) < config.orient_gate and abs(roll) < config.orient_gate:
            height_error = target_height - mj_data.qpos[2]
            f_stab = (
                config.harness_kp * height_error - config.harness_kd * mj_data.qvel[2]
            )
            f_stab = np.clip(f_stab, -config.harness_cap, config.harness_cap)
        else:
            f_stab = 0.0

        # Apply XY tracking (weak, to keep robot on trajectory)
        ref_xy_delta = ref_root_pos[i, :2] - ref_xy_start
        target_xy = sim_xy_start + ref_xy_delta
        xy_error = target_xy - mj_data.qpos[0:2]
        stab_force_xy = 100.0 * xy_error - 20.0 * mj_data.qvel[0:2]
        stab_force_xy = np.clip(stab_force_xy, -10.0, 10.0)

        # Apply forces
        mj_data.qfrc_applied[0:2] = stab_force_xy
        mj_data.qfrc_applied[2] = f_stab

        # Record stabilization force (Z component only for Tier gates)
        all_stab_force.append(abs(f_stab))

        # Step physics
        for _ in range(n_substeps):
            mujoco.mj_step(mj_model, mj_data)

        # Extract contact normal forces
        left_fn, right_fn = 0.0, 0.0
        for contact_idx in range(mj_data.ncon):
            contact = mj_data.contact[contact_idx]
            geom1, geom2 = contact.geom1, contact.geom2

            # Check for foot-ground contact
            foot_geom = None
            if geom1 == ground_id and geom2 in [left_toe_id, right_toe_id]:
                foot_geom = geom2
            elif geom2 == ground_id and geom1 in [left_toe_id, right_toe_id]:
                foot_geom = geom1

            if foot_geom is not None and contact.efc_address >= 0:
                fn = mj_data.efc_force[contact.efc_address]
                if fn > 0:
                    if foot_geom == left_toe_id:
                        left_fn += fn
                    else:
                        right_fn += fn

        # Record contact forces
        sum_fn = left_fn + right_fn
        all_sum_fn.append(sum_fn)
        all_left_contact.append(1.0 if left_fn > 0.5 else 0.0)
        all_right_contact.append(1.0 if right_fn > 0.5 else 0.0)

        # Record height
        all_heights.append(mj_data.qpos[2])

    # Convert to arrays
    sum_fn_arr = np.array(all_sum_fn)
    stab_force_arr = np.array(all_stab_force)
    left_contact_arr = np.array(all_left_contact)
    right_contact_arr = np.array(all_right_contact)
    heights_arr = np.array(all_heights)

    # Compute metrics
    mean_load_support = sum_fn_arr.mean() / mg
    p90_stab_force = np.percentile(stab_force_arr, 90) / mg
    max_stab_force = stab_force_arr.max() / mg
    mean_stab_force = stab_force_arr.mean() / mg

    # Unloaded frame rate
    unloaded_threshold = config.unloaded_threshold * mg
    unloaded_frames = (sum_fn_arr < unloaded_threshold).sum()
    unloaded_frame_rate = unloaded_frames / num_frames

    # Contact rates
    left_contact_rate = left_contact_arr.mean()
    right_contact_rate = right_contact_arr.mean()
    both_contact_rate = ((left_contact_arr > 0.5) & (right_contact_arr > 0.5)).mean()
    no_contact_rate = ((left_contact_arr < 0.5) & (right_contact_arr < 0.5)).mean()

    return {
        "num_frames": num_frames,
        "mean_load_support": mean_load_support,
        "p90_stab_force": p90_stab_force,
        "max_stab_force": max_stab_force,
        "mean_stab_force": mean_stab_force,
        "unloaded_frame_rate": unloaded_frame_rate,
        "left_contact_rate": left_contact_rate,
        "right_contact_rate": right_contact_rate,
        "both_contact_rate": both_contact_rate,
        "no_contact_rate": no_contact_rate,
        "height_mean": heights_arr.mean(),
        "height_std": heights_arr.std(),
        "height_min": heights_arr.min(),
        "height_max": heights_arr.max(),
        "sum_fn_per_frame": sum_fn_arr,
        "stab_force_per_frame": stab_force_arr,
    }


# =============================================================================
# Tier Classification
# =============================================================================


def classify_tier(
    metrics: Dict[str, Any],
    config: HarnessGateConfig,
) -> Tuple[int, str, Dict[str, bool]]:
    """Classify clip into Tier 0, Tier 1, or Rejected.

    Args:
        metrics: Quality metrics from rollout
        config: Test configuration

    Returns:
        Tuple of (tier, reason, gate_results)
    """
    gate_results = {}

    # Tier 0 gates
    gate_results["tier0_load_support"] = (
        metrics["mean_load_support"] >= config.tier0_min_load_support
    )
    gate_results["tier0_stab_force"] = (
        metrics["p90_stab_force"] <= config.tier0_max_stab_force_p90
    )
    gate_results["tier0_unloaded"] = (
        metrics["unloaded_frame_rate"] <= config.tier0_max_unloaded_rate
    )

    # Tier 1 gates
    gate_results["tier1_load_support"] = (
        metrics["mean_load_support"] >= config.tier1_min_load_support
    )
    gate_results["tier1_stab_force"] = (
        metrics["p90_stab_force"] <= config.tier1_max_stab_force_p90
    )
    gate_results["tier1_unloaded"] = (
        metrics["unloaded_frame_rate"] <= config.tier1_max_unloaded_rate
    )

    # Classify
    tier0_pass = all(
        [
            gate_results["tier0_load_support"],
            gate_results["tier0_stab_force"],
            gate_results["tier0_unloaded"],
        ]
    )

    tier1_pass = all(
        [
            gate_results["tier1_load_support"],
            gate_results["tier1_stab_force"],
            gate_results["tier1_unloaded"],
        ]
    )

    if tier0_pass:
        return 0, "All Tier 0 gates passed", gate_results
    elif tier1_pass:
        # Determine which Tier 0 gate failed
        failures = []
        if not gate_results["tier0_load_support"]:
            failures.append(
                f"load_support={metrics['mean_load_support']:.1%}<{config.tier0_min_load_support:.0%}"
            )
        if not gate_results["tier0_stab_force"]:
            failures.append(
                f"p90_stab={metrics['p90_stab_force']:.1%}>{config.tier0_max_stab_force_p90:.0%}"
            )
        if not gate_results["tier0_unloaded"]:
            failures.append(
                f"unloaded={metrics['unloaded_frame_rate']:.1%}>{config.tier0_max_unloaded_rate:.0%}"
            )
        return 1, f"Tier 1 (failed: {', '.join(failures)})", gate_results
    else:
        # Determine which Tier 1 gate failed
        failures = []
        if not gate_results["tier1_load_support"]:
            failures.append(
                f"load_support={metrics['mean_load_support']:.1%}<{config.tier1_min_load_support:.0%}"
            )
        if not gate_results["tier1_stab_force"]:
            failures.append(
                f"p90_stab={metrics['p90_stab_force']:.1%}>{config.tier1_max_stab_force_p90:.0%}"
            )
        if not gate_results["tier1_unloaded"]:
            failures.append(
                f"unloaded={metrics['unloaded_frame_rate']:.1%}>{config.tier1_max_unloaded_rate:.0%}"
            )
        return -1, f"Rejected ({', '.join(failures)})", gate_results


# =============================================================================
# Test Runner
# =============================================================================


def run_harness_gate_test(
    motion_path: Path,
    mj_model: mujoco.MjModel,
    config: HarnessGateConfig,
    amp_config: FeatureConfig,
    verbose: bool = False,
) -> HarnessGateResult:
    """Run harness contribution gate test on a single motion.

    Args:
        motion_path: Path to motion file
        mj_model: MuJoCo model
        config: Test configuration
        amp_config: AMP feature configuration
        verbose: Print detailed output

    Returns:
        HarnessGateResult with metrics and tier classification
    """
    clip_name = motion_path.stem

    if verbose:
        print(f"\n  Processing: {clip_name}")

    # Run physics rollout
    metrics = run_physics_rollout_with_metrics(
        motion_path, mj_model, config, amp_config
    )

    # Classify tier
    tier, tier_reason, gate_results = classify_tier(metrics, config)

    if verbose:
        print(f"    Frames: {metrics['num_frames']}")
        print(f"    Load support: {metrics['mean_load_support']:.1%}")
        print(f"    p90 stab force: {metrics['p90_stab_force']:.1%}")
        print(f"    Unloaded rate: {metrics['unloaded_frame_rate']:.1%}")
        print(f"    Tier: {tier} ({tier_reason})")

    return HarnessGateResult(
        clip_name=clip_name,
        tier=tier,
        tier_reason=tier_reason,
        mean_load_support=metrics["mean_load_support"],
        p90_stab_force=metrics["p90_stab_force"],
        max_stab_force=metrics["max_stab_force"],
        mean_stab_force=metrics["mean_stab_force"],
        unloaded_frame_rate=metrics["unloaded_frame_rate"],
        left_contact_rate=metrics["left_contact_rate"],
        right_contact_rate=metrics["right_contact_rate"],
        both_contact_rate=metrics["both_contact_rate"],
        no_contact_rate=metrics["no_contact_rate"],
        height_mean=metrics["height_mean"],
        height_std=metrics["height_std"],
        height_min=metrics["height_min"],
        height_max=metrics["height_max"],
        num_frames=metrics["num_frames"],
        gate_results=gate_results,
    )


def run_batch_harness_gate_test(
    motion_dir: Path,
    mj_model: mujoco.MjModel,
    config: HarnessGateConfig,
    amp_config: FeatureConfig,
    verbose: bool = False,
) -> Tuple[List[HarnessGateResult], bool]:
    """Run harness contribution gate test on all motions.

    Args:
        motion_dir: Directory containing motion files
        mj_model: MuJoCo model
        config: Test configuration
        amp_config: AMP feature configuration
        verbose: Print detailed output

    Returns:
        Tuple of (list of results, overall pass for Tier 0)
    """
    print(f"\n[bold cyan]Test 7: Harness Contribution Gate[/bold cyan]")
    print(f"  Motion directory: {motion_dir}")
    print(f"  Robot mass: {config.robot_mass} kg (mg = {config.mg:.1f} N)")
    print(f"\n  Tier 0 gates:")
    print(f"    mean(ΣF_n)/mg >= {config.tier0_min_load_support:.0%}")
    print(f"    p90(|F_stab|)/mg <= {config.tier0_max_stab_force_p90:.0%}")
    print(f"    unloaded_frames <= {config.tier0_max_unloaded_rate:.0%}")

    motion_files = sorted(motion_dir.glob("*.pkl"))
    print(f"\n  Found {len(motion_files)} motion files")

    results = []

    for motion_path in motion_files:
        try:
            result = run_harness_gate_test(
                motion_path, mj_model, config, amp_config, verbose=verbose
            )
            results.append(result)
        except Exception as e:
            print(f"[red]  Error processing {motion_path}: {e}[/red]")

    # Print results table
    print(f"\n")
    table = Table(title="Test 7: Harness Contribution Results")
    table.add_column("Clip", style="cyan")
    table.add_column("Tier", justify="center")
    table.add_column("Load Support", justify="right")
    table.add_column("p90 Stab", justify="right")
    table.add_column("Unloaded", justify="right")
    table.add_column("Status", justify="center")

    for result in results:
        if result.tier == 0:
            tier_str = "[green]Tier 0[/green]"
            status = "[green]✓ PASS[/green]"
        elif result.tier == 1:
            tier_str = "[yellow]Tier 1[/yellow]"
            status = "[yellow]⚠ WARN[/yellow]"
        else:
            tier_str = "[red]Rejected[/red]"
            status = "[red]✗ FAIL[/red]"

        # Format metrics with color coding
        load_str = f"{result.mean_load_support:.1%}"
        if result.gate_results.get("tier0_load_support"):
            load_str = f"[green]{load_str}[/green]"
        elif result.gate_results.get("tier1_load_support"):
            load_str = f"[yellow]{load_str}[/yellow]"
        else:
            load_str = f"[red]{load_str}[/red]"

        stab_str = f"{result.p90_stab_force:.1%}"
        if result.gate_results.get("tier0_stab_force"):
            stab_str = f"[green]{stab_str}[/green]"
        elif result.gate_results.get("tier1_stab_force"):
            stab_str = f"[yellow]{stab_str}[/yellow]"
        else:
            stab_str = f"[red]{stab_str}[/red]"

        unloaded_str = f"{result.unloaded_frame_rate:.1%}"
        if result.gate_results.get("tier0_unloaded"):
            unloaded_str = f"[green]{unloaded_str}[/green]"
        elif result.gate_results.get("tier1_unloaded"):
            unloaded_str = f"[yellow]{unloaded_str}[/yellow]"
        else:
            unloaded_str = f"[red]{unloaded_str}[/red]"

        table.add_row(
            result.clip_name,
            tier_str,
            load_str,
            stab_str,
            unloaded_str,
            status,
        )

    console.print(table)

    # Summary statistics
    tier0_count = sum(1 for r in results if r.tier == 0)
    tier1_count = sum(1 for r in results if r.tier == 1)
    rejected_count = sum(1 for r in results if r.tier == -1)

    print(f"\n[bold]Summary:[/bold]")
    print(f"  Tier 0 (training): {tier0_count}/{len(results)} clips")
    print(f"  Tier 1 (curriculum): {tier1_count}/{len(results)} clips")
    print(f"  Rejected: {rejected_count}/{len(results)} clips")

    # Overall pass/fail
    all_tier0 = tier0_count == len(results)
    any_tier0 = tier0_count > 0

    if all_tier0:
        print(
            f"\n[bold green]✓ Test 7 PASSED: All {len(results)} clips meet Tier 0 requirements[/bold green]"
        )
    elif any_tier0:
        print(
            f"\n[bold yellow]⚠ Test 7 PARTIAL: {tier0_count}/{len(results)} clips meet Tier 0 requirements[/bold yellow]"
        )
        print(f"  Training can proceed with Tier 0 clips only.")
    else:
        print(
            f"\n[bold red]✗ Test 7 FAILED: No clips meet Tier 0 requirements[/bold red]"
        )
        print(f"  Harness parameters may need adjustment.")

    return results, all_tier0


def print_detailed_metrics(results: List[HarnessGateResult]):
    """Print detailed metrics for all clips."""
    print(f"\n[bold]Detailed Metrics:[/bold]")

    for result in results:
        tier_color = (
            "green" if result.tier == 0 else ("yellow" if result.tier == 1 else "red")
        )
        print(
            f"\n  [{tier_color}]{result.clip_name}[/{tier_color}] (Tier {result.tier})"
        )
        print(f"    {result.tier_reason}")
        print(f"    Load support: {result.mean_load_support:.2%} (min: 85%)")
        print(f"    Stab force p90: {result.p90_stab_force:.2%} (max: 15%)")
        print(f"    Stab force mean: {result.mean_stab_force:.2%}")
        print(f"    Stab force max: {result.max_stab_force:.2%}")
        print(f"    Unloaded frames: {result.unloaded_frame_rate:.2%} (max: 10%)")
        print(
            f"    Contact rates: L={result.left_contact_rate:.1%}, R={result.right_contact_rate:.1%}"
        )
        print(
            f"    Both feet: {result.both_contact_rate:.1%}, No feet: {result.no_contact_rate:.1%}"
        )
        print(
            f"    Height: mean={result.height_mean:.3f}m, std={result.height_std:.3f}m"
        )


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Test 7: Harness Contribution Gate")
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
        "--verbose",
        action="store_true",
        help="Print detailed metrics",
    )
    parser.add_argument(
        "--tier0-only",
        action="store_true",
        help="Exit with error if any clip is not Tier 0",
    )

    args = parser.parse_args()

    # Load robot config
    print(f"[bold blue]Loading robot config:[/bold blue] {args.robot_config}")
    load_robot_config(args.robot_config)
    robot_config = get_robot_config()
    amp_config = create_config_from_robot(robot_config)

    # Test configuration
    config = HarnessGateConfig()

    # Load MuJoCo model
    model_path = Path(args.model)
    from playground_amp.envs.wildrobot_env import get_assets

    mj_model = mujoco.MjModel.from_xml_string(
        model_path.read_text(), assets=get_assets(model_path.parent)
    )

    if args.motion:
        # Single motion test
        motion_path = Path(args.motion)
        print(f"\n[bold cyan]Test 7: Harness Contribution Gate[/bold cyan]")
        print(f"  Motion: {motion_path}")

        result = run_harness_gate_test(
            motion_path, mj_model, config, amp_config, verbose=True
        )

        # Print result
        if result.tier == 0:
            print(f"\n[bold green]✓ Tier 0: {result.tier_reason}[/bold green]")
            sys.exit(0)
        elif result.tier == 1:
            print(f"\n[bold yellow]⚠ Tier 1: {result.tier_reason}[/bold yellow]")
            sys.exit(0 if not args.tier0_only else 1)
        else:
            print(f"\n[bold red]✗ Rejected: {result.tier_reason}[/bold red]")
            sys.exit(1)
    else:
        # Batch test
        motion_dir = Path(args.motion_dir)
        results, all_tier0 = run_batch_harness_gate_test(
            motion_dir, mj_model, config, amp_config, verbose=args.verbose
        )

        if args.verbose:
            print_detailed_metrics(results)

        # Exit code based on tier0_only flag
        if args.tier0_only:
            sys.exit(0 if all_tier0 else 1)
        else:
            any_tier0 = any(r.tier == 0 for r in results)
            sys.exit(0 if any_tier0 else 1)


if __name__ == "__main__":
    main()
