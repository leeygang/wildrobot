#!/usr/bin/env python3
"""Tune PD gains for Tier 0 stable walking with weak harness.

This script experiments with different actuator PD gains to achieve
stable walking where feet support body weight (not harness).

The goal is to find gains where:
- mean(ΣF_n)/mg >= 0.85 (feet carry 85%+ of weight)
- p90(|F_stab|)/mg <= 0.15 (harness <= 15%)
- unloaded_frames <= 10%

Usage:
    cd ~/projects/wildrobot
    uv run python scripts/tune_pd_gains.py --motion assets/motions/walking_medium01.pkl
    uv run python scripts/tune_pd_gains.py --sweep
"""

import argparse
import pickle
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import mujoco
import numpy as np
from rich import print
from rich.console import Console
from rich.table import Table

console = Console()


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class PDGainConfig:
    """PD gain configuration to test."""

    name: str
    kp: float  # Position gain
    kv: float  # Velocity gain (damping)
    forcerange: float  # Max force per actuator (N)

    def __str__(self):
        return f"{self.name}: kp={self.kp}, kv={self.kv}, force=±{self.forcerange}N"


@dataclass
class HarnessConfig:
    """Harness configuration."""

    # Height stabilization (Z-axis)
    height_kp: float = 60.0  # N/m
    height_kd: float = 10.0  # N·s/m
    force_cap: float = 4.0  # N (~12% of mg)
    
    # XY stabilization (prevent drift)
    xy_kp: float = 100.0  # N/m
    xy_kd: float = 20.0  # N·s/m
    xy_cap: float = 10.0  # N
    
    # Orientation stabilization (pitch/roll) - CRITICAL for balance!
    rot_kp: float = 20.0  # N·m/rad
    rot_kd: float = 2.0  # N·m·s/rad
    rot_cap: float = 5.0  # N·m per axis
    
    # Orientation gate (disable if too tilted)
    orient_gate: float = 0.436  # radians (25°)
    
    # Mode: "weak" (assist only) or "full" (full harness)
    mode: str = "weak"


# =============================================================================
# Gain Configurations to Test
# =============================================================================

# Current default (from XML)
CURRENT_GAINS = PDGainConfig(
    name="Current (XML default)",
    kp=21.1,
    kv=0.5,
    forcerange=4.0,
)

# Configurations to sweep
GAIN_CONFIGS = [
    # Current baseline
    PDGainConfig("Current", kp=21.1, kv=0.5, forcerange=4.0),
    # Increase force range (most important)
    PDGainConfig("Force 8N", kp=21.1, kv=0.5, forcerange=8.0),
    PDGainConfig("Force 12N", kp=21.1, kv=0.5, forcerange=12.0),
    PDGainConfig("Force 16N", kp=21.1, kv=0.5, forcerange=16.0),
    # Higher kp with higher force
    PDGainConfig("kp=40, Force 12N", kp=40.0, kv=0.5, forcerange=12.0),
    PDGainConfig("kp=60, Force 12N", kp=60.0, kv=0.5, forcerange=12.0),
    PDGainConfig("kp=80, Force 16N", kp=80.0, kv=1.0, forcerange=16.0),
    # Higher damping
    PDGainConfig("kp=40, kv=1.0, Force 12N", kp=40.0, kv=1.0, forcerange=12.0),
    PDGainConfig("kp=60, kv=1.5, Force 16N", kp=60.0, kv=1.5, forcerange=16.0),
    # Aggressive (for comparison)
    PDGainConfig("Aggressive", kp=100.0, kv=2.0, forcerange=20.0),
]


# =============================================================================
# Model Modification
# =============================================================================


def create_modified_model(
    scene_xml_path: Path,
    pd_config: PDGainConfig,
) -> str:
    """Create modified XML with new PD gains.

    Args:
        scene_xml_path: Path to scene XML (e.g., scene_flat_terrain.xml)
        pd_config: PD gain configuration

    Returns:
        Modified XML string (with robot XML inlined)
    """
    import re

    # The scene XML includes the robot via <include file="wildrobot.xml" />
    # We need to modify the robot XML, then inline it into the scene
    assets_dir = scene_xml_path.parent
    robot_xml_path = assets_dir / "wildrobot.xml"

    # Read robot XML and modify PD gains
    robot_xml = robot_xml_path.read_text()
    pattern = r'<position\s+kp="[\d.]+" kv="[\d.]+" forcerange="-[\d.]+ [\d.]+"\s*/>'
    new_line = f'<position kp="{pd_config.kp}" kv="{pd_config.kv}" forcerange="-{pd_config.forcerange} {pd_config.forcerange}" />'
    modified_robot_xml, count = re.subn(pattern, new_line, robot_xml)

    if count == 0:
        print(f"[yellow]Warning: Failed to replace PD gains in robot XML[/yellow]")

    # Read scene XML and replace <include> with modified robot XML content
    scene_xml = scene_xml_path.read_text()

    # Remove <mujoco> wrapper from robot XML for inlining
    # Extract content between <mujoco model="wildrobot"> and </mujoco>
    robot_content_match = re.search(
        r"<mujoco[^>]*>(.*)</mujoco>", modified_robot_xml, re.DOTALL
    )
    if robot_content_match:
        robot_content = robot_content_match.group(1)
    else:
        robot_content = modified_robot_xml

    # Replace <include file="wildrobot.xml" /> with robot content
    modified_scene = re.sub(
        r'<include\s+file="wildrobot\.xml"\s*/>', robot_content, scene_xml
    )

    return modified_scene


# =============================================================================
# Physics Rollout
# =============================================================================


def run_rollout_with_config(
    motion_path: Path,
    model_xml: str,
    assets_dir: Path,
    harness_config: HarnessConfig,
    num_actuated_joints: int = 8,
) -> Dict[str, Any]:
    """Run physics rollout with given configuration.

    Args:
        motion_path: Path to GMR motion file
        model_xml: Modified XML string (scene with inlined robot)
        assets_dir: Path to assets directory
        harness_config: Harness parameters
        num_actuated_joints: Number of actuated joints

    Returns:
        Dict with quality metrics
    """
    # Load motion
    with open(motion_path, "rb") as f:
        motion = pickle.load(f)

    ref_dof_pos = motion["dof_pos"]
    ref_root_pos = motion["root_pos"]
    num_frames = motion["num_frames"]

    # Load model with assets
    from playground_amp.envs.wildrobot_env import get_assets

    mj_model = mujoco.MjModel.from_xml_string(model_xml, assets=get_assets(assets_dir))
    mj_data = mujoco.MjData(mj_model)

    n = num_actuated_joints
    sim_dt = 0.002
    control_dt = 0.02
    mj_model.opt.timestep = sim_dt
    n_substeps = int(control_dt / sim_dt)

    # Get foot geom IDs (both toe AND heel for complete foot contact)
    left_toe_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "left_toe")
    right_toe_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "right_toe")
    left_heel_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "left_heel")
    right_heel_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "right_heel")
    floor_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    # All foot geoms (both toes and heels)
    left_foot_geoms = [left_toe_id, left_heel_id]
    right_foot_geoms = [right_toe_id, right_heel_id]
    all_foot_geoms = left_foot_geoms + right_foot_geoms

    # Initialize from home keyframe (defined in scene XML)
    # This provides a valid standing pose on the floor
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
    mj_data.qvel[:] = 0
    mujoco.mj_forward(mj_model, mj_data)

    # Reference values
    target_height = ref_root_pos[:, 2].mean()
    ref_xy_start = ref_root_pos[0, :2].copy()
    sim_xy_start = mj_data.qpos[0:2].copy()

    # Storage
    all_sum_fn = []
    all_stab_force = []
    all_heights = []
    all_left_contact = []
    all_right_contact = []

    robot_mass = 3.383
    mg = robot_mass * 9.81

    for i in range(num_frames):
        # Set control targets
        mj_data.ctrl[:n] = ref_dof_pos[i]

        # Compute orientation for gating
        w, x, y, z = mj_data.qpos[3:7]
        pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))

        # Apply weak height harness (only if upright)
        if abs(pitch) < harness_config.orient_gate and abs(roll) < harness_config.orient_gate:
            height_error = target_height - mj_data.qpos[2]
            f_stab_z = (
                harness_config.height_kp * height_error
                - harness_config.height_kd * mj_data.qvel[2]
            )
            f_stab_z = np.clip(
                f_stab_z, -harness_config.force_cap, harness_config.force_cap
            )
        else:
            f_stab_z = 0.0

        # Apply XY tracking
        ref_xy_delta = ref_root_pos[i, :2] - ref_xy_start
        target_xy = sim_xy_start + ref_xy_delta
        xy_error = target_xy - mj_data.qpos[0:2]
        stab_force_xy = (
            harness_config.xy_kp * xy_error - harness_config.xy_kd * mj_data.qvel[0:2]
        )
        stab_force_xy = np.clip(
            stab_force_xy, -harness_config.xy_cap, harness_config.xy_cap
        )

        # Apply orientation stabilization torques (CRITICAL for balance in full mode)
        if harness_config.mode == "full":
            tau_pitch = -harness_config.rot_kp * pitch - harness_config.rot_kd * mj_data.qvel[4]
            tau_roll = -harness_config.rot_kp * roll - harness_config.rot_kd * mj_data.qvel[3]
            tau_pitch = np.clip(tau_pitch, -harness_config.rot_cap, harness_config.rot_cap)
            tau_roll = np.clip(tau_roll, -harness_config.rot_cap, harness_config.rot_cap)
        else:
            tau_pitch = 0.0
            tau_roll = 0.0

        # Apply forces and torques
        mj_data.qfrc_applied[0:2] = stab_force_xy
        mj_data.qfrc_applied[2] = f_stab_z
        mj_data.qfrc_applied[3] = tau_roll   # Roll torque (around X)
        mj_data.qfrc_applied[4] = tau_pitch  # Pitch torque (around Y)

        all_stab_force.append(abs(f_stab_z))

        # Step physics
        for _ in range(n_substeps):
            mujoco.mj_step(mj_model, mj_data)

        # Check for simulation blowup
        if np.any(np.isnan(mj_data.qpos)) or np.any(np.abs(mj_data.qpos) > 100):
            return {
                "success": False,
                "reason": "Simulation blowup",
                "frames_completed": i,
            }

        # Extract CORRECT vertical support force from generalized constraint force
        # qfrc_constraint[2] is the total vertical force from all contacts in the Z direction
        # This is the TRUE load support, NOT the sum of individual efc_force values!
        vertical_support = mj_data.qfrc_constraint[2]
        sum_fn = abs(vertical_support)  # Take absolute value (force is upward against gravity)
        
        # Also extract individual foot contact detection for left/right tracking
        left_contact = False
        right_contact = False
        for contact_idx in range(mj_data.ncon):
            contact = mj_data.contact[contact_idx]
            geom1, geom2 = contact.geom1, contact.geom2

            if geom1 == floor_id or geom2 == floor_id:
                other_geom = geom2 if geom1 == floor_id else geom1
                if other_geom in left_foot_geoms:
                    left_contact = True
                elif other_geom in right_foot_geoms:
                    right_contact = True
        all_sum_fn.append(sum_fn)
        all_left_contact.append(1.0 if left_contact else 0.0)
        all_right_contact.append(1.0 if right_contact else 0.0)
        all_heights.append(mj_data.qpos[2])

    # Compute metrics
    sum_fn_arr = np.array(all_sum_fn)
    stab_force_arr = np.array(all_stab_force)
    heights_arr = np.array(all_heights)

    mean_load_support = sum_fn_arr.mean() / mg
    p90_stab_force = np.percentile(stab_force_arr, 90) / mg
    unloaded_threshold = 0.6 * mg
    unloaded_frame_rate = (sum_fn_arr < unloaded_threshold).mean()

    # Tier classification
    tier0_pass = (
        mean_load_support >= 0.85
        and p90_stab_force <= 0.15
        and unloaded_frame_rate <= 0.10
    )
    tier1_pass = (
        mean_load_support >= 0.60
        and p90_stab_force <= 0.40
        and unloaded_frame_rate <= 0.30
    )

    if tier0_pass:
        tier = 0
    elif tier1_pass:
        tier = 1
    else:
        tier = -1

    return {
        "success": True,
        "tier": tier,
        "mean_load_support": mean_load_support,
        "p90_stab_force": p90_stab_force,
        "unloaded_frame_rate": unloaded_frame_rate,
        "height_mean": heights_arr.mean(),
        "height_std": heights_arr.std(),
        "left_contact_rate": np.mean(all_left_contact),
        "right_contact_rate": np.mean(all_right_contact),
        "num_frames": num_frames,
    }


# =============================================================================
# Main
# =============================================================================


def run_gain_sweep(
    motion_path: Path,
    scene_xml_path: Path,
    assets_dir: Path,
    configs: List[PDGainConfig],
    harness_config: HarnessConfig,
) -> List[Tuple[PDGainConfig, Dict[str, Any]]]:
    """Run sweep over gain configurations.

    Args:
        motion_path: Path to motion file
        scene_xml_path: Path to scene XML (includes robot and floor)
        assets_dir: Path to assets directory
        configs: List of gain configurations
        harness_config: Harness parameters

    Returns:
        List of (config, results) tuples
    """
    results = []

    print(f"\n[bold cyan]PD Gain Tuning Sweep[/bold cyan]")
    print(f"  Motion: {motion_path.name}")
    print(f"  Harness: kp={harness_config.height_kp}, cap={harness_config.force_cap}N")
    print(f"  Testing {len(configs)} configurations...")

    for config in configs:
        print(f"\n  Testing: {config}")

        # Create modified XML (scene with modified robot inlined)
        modified_xml = create_modified_model(scene_xml_path, config)

        # Run rollout
        result = run_rollout_with_config(
            motion_path, modified_xml, assets_dir, harness_config
        )

        results.append((config, result))

        if result["success"]:
            tier_str = f"Tier {result['tier']}" if result["tier"] >= 0 else "Rejected"
            print(
                f"    Load: {result['mean_load_support']:.1%}, "
                f"Stab: {result['p90_stab_force']:.1%}, "
                f"Unloaded: {result['unloaded_frame_rate']:.1%} "
                f"→ {tier_str}"
            )
        else:
            print(f"    [red]FAILED: {result['reason']}[/red]")

    return results


def print_results_table(results: List[Tuple[PDGainConfig, Dict[str, Any]]]):
    """Print results summary table."""
    print(f"\n")
    table = Table(title="PD Gain Tuning Results")
    table.add_column("Configuration", style="cyan")
    table.add_column("Tier", justify="center")
    table.add_column("Load Support", justify="right")
    table.add_column("p90 Stab", justify="right")
    table.add_column("Unloaded", justify="right")
    table.add_column("Height", justify="right")

    for config, result in results:
        if not result["success"]:
            table.add_row(config.name, "[red]FAIL[/red]", "-", "-", "-", "-")
            continue

        tier = result["tier"]
        if tier == 0:
            tier_str = "[green]Tier 0[/green]"
        elif tier == 1:
            tier_str = "[yellow]Tier 1[/yellow]"
        else:
            tier_str = "[red]Rejected[/red]"

        load_val = result["mean_load_support"]
        stab_val = result["p90_stab_force"]
        unloaded_val = result["unloaded_frame_rate"]

        # Color code by threshold
        load_str = f"{load_val:.1%}"
        if load_val >= 0.85:
            load_str = f"[green]{load_str}[/green]"
        elif load_val >= 0.60:
            load_str = f"[yellow]{load_str}[/yellow]"
        else:
            load_str = f"[red]{load_str}[/red]"

        stab_str = f"{stab_val:.1%}"
        if stab_val <= 0.15:
            stab_str = f"[green]{stab_str}[/green]"
        elif stab_val <= 0.40:
            stab_str = f"[yellow]{stab_str}[/yellow]"
        else:
            stab_str = f"[red]{stab_str}[/red]"

        unloaded_str = f"{unloaded_val:.1%}"
        if unloaded_val <= 0.10:
            unloaded_str = f"[green]{unloaded_str}[/green]"
        elif unloaded_val <= 0.30:
            unloaded_str = f"[yellow]{unloaded_str}[/yellow]"
        else:
            unloaded_str = f"[red]{unloaded_str}[/red]"

        table.add_row(
            config.name,
            tier_str,
            load_str,
            stab_str,
            unloaded_str,
            f"{result['height_mean']:.3f}m",
        )

    console.print(table)

    # Summary
    tier0_configs = [c for c, r in results if r.get("tier") == 0]
    if tier0_configs:
        print(
            f"\n[bold green]✓ Found {len(tier0_configs)} Tier 0 configuration(s):[/bold green]"
        )
        for config in tier0_configs:
            print(f"  - {config}")
    else:
        tier1_configs = [c for c, r in results if r.get("tier") == 1]
        if tier1_configs:
            print(
                f"\n[bold yellow]⚠ No Tier 0 found, but {len(tier1_configs)} Tier 1 configuration(s):[/bold yellow]"
            )
            for config in tier1_configs:
                print(f"  - {config}")
        else:
            print(
                f"\n[bold red]✗ No configurations achieved Tier 0 or Tier 1[/bold red]"
            )


def main():
    parser = argparse.ArgumentParser(description="Tune PD gains for Tier 0 walking")
    parser.add_argument(
        "--motion",
        type=str,
        default="assets/motions/walking_medium01.pkl",
        help="Motion file to test",
    )
    parser.add_argument(
        "--scene-xml",
        type=str,
        default="assets/scene_flat_terrain.xml",
        help="Scene XML file (includes robot and floor)",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run full gain sweep",
    )
    parser.add_argument(
        "--kp",
        type=float,
        default=None,
        help="Test specific kp value",
    )
    parser.add_argument(
        "--kv",
        type=float,
        default=None,
        help="Test specific kv value",
    )
    parser.add_argument(
        "--force",
        type=float,
        default=None,
        help="Test specific force range",
    )

    args = parser.parse_args()

    motion_path = Path(args.motion)
    scene_xml_path = Path(args.scene_xml)
    assets_dir = scene_xml_path.parent

    harness_config = HarnessConfig()

    # Determine configs to test
    if args.sweep:
        configs = GAIN_CONFIGS
    elif args.kp is not None or args.kv is not None or args.force is not None:
        # Custom single config
        kp = args.kp if args.kp is not None else 21.1
        kv = args.kv if args.kv is not None else 0.5
        force = args.force if args.force is not None else 4.0
        configs = [
            PDGainConfig(f"Custom kp={kp}, kv={kv}, force={force}", kp, kv, force)
        ]
    else:
        # Default: test a few key configs
        configs = [
            CURRENT_GAINS,
            PDGainConfig("Force 12N", kp=21.1, kv=0.5, forcerange=12.0),
            PDGainConfig("kp=60, Force 16N", kp=60.0, kv=1.0, forcerange=16.0),
        ]

    # Run sweep
    results = run_gain_sweep(
        motion_path, scene_xml_path, assets_dir, configs, harness_config
    )

    # Print results
    print_results_table(results)

    # Exit code based on Tier 0 success
    tier0_found = any(r.get("tier") == 0 for _, r in results)
    sys.exit(0 if tier0_found else 1)


if __name__ == "__main__":
    main()
