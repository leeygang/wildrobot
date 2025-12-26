#!/usr/bin/env python3
"""Physics Reference Feature Parity Test.

Verifies that physics-generated reference data has the same feature structure
and compatible distributions with online policy feature extraction.

This script:
1. Loads physics reference dataset
2. Simulates policy rollout and extracts online features
3. Compares feature dimensions, ranges, and distributions
4. Reports any mismatches that could cause discriminator issues

Key parity checks:
- Feature dimension: Must match exactly (27 dims for 8-joint robot)
- Value ranges: Should be similar (not orders of magnitude different)
- Contact patterns: Should have similar contact rates
- Velocity patterns: Should be in similar ranges

Usage:
    cd ~/projects/wildrobot
    uv run python scripts/verify_physics_ref_parity.py

    # With custom paths
    uv run python scripts/verify_physics_ref_parity.py \
        --physics-data playground_amp/data/physics_ref/walking_physics_merged.pkl \
        --robot-config assets/robot_config.yaml \
        --model assets/scene_flat_terrain.xml
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from rich import print
from rich.console import Console
from rich.table import Table

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

console = Console()


def load_physics_reference(path: str) -> Dict[str, Any]:
    """Load physics reference dataset."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def analyze_feature_statistics(
    features: np.ndarray, name: str
) -> Dict[str, np.ndarray]:
    """Compute per-dimension statistics for features."""
    return {
        "name": name,
        "shape": features.shape,
        "mean": np.mean(features, axis=0),
        "std": np.std(features, axis=0),
        "min": np.min(features, axis=0),
        "max": np.max(features, axis=0),
        "median": np.median(features, axis=0),
    }


def simulate_policy_features(
    mj_model: Any,
    robot_config: Any,
    amp_config: Any,
    num_frames: int = 500,
    dt: float = 0.02,
) -> np.ndarray:
    """Simulate policy rollout and extract features.

    This simulates what the online feature extractor sees during training:
    - Joint positions from qpos
    - Joint velocities via finite differences
    - Root velocity in heading-local frame
    - Root height
    - Estimated foot contacts from joint angles
    """
    import mujoco
    from scipy.spatial.transform import Rotation as R

    # Create data object
    mj_data = mujoco.MjData(mj_model)

    # Initialize from keyframe if available
    if mj_model.nkey > 0:
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)

    # Number of actuated joints (from robot config)
    num_joints = robot_config.action_dim
    feature_dim = num_joints * 2 + 11  # 2*N + 11

    features_list = []
    prev_joint_pos = None

    for i in range(num_frames):
        # Apply small random action to simulate policy
        action = np.random.randn(num_joints) * 0.1
        mj_data.ctrl[:num_joints] = mj_data.qpos[7 : 7 + num_joints] + action

        # Step physics
        for _ in range(int(dt / mj_model.opt.timestep)):
            mujoco.mj_step(mj_model, mj_data)

        # Extract joint positions (actuated only)
        joint_pos = mj_data.qpos[7 : 7 + num_joints].copy()

        # Joint velocities via finite differences
        if prev_joint_pos is not None:
            joint_vel = (joint_pos - prev_joint_pos) / dt
        else:
            joint_vel = np.zeros(num_joints)
        prev_joint_pos = joint_pos.copy()

        # Root pose
        root_pos = mj_data.qpos[0:3].copy()
        root_quat_wxyz = mj_data.qpos[3:7].copy()  # MuJoCo uses wxyz
        root_quat_xyzw = np.array(
            [root_quat_wxyz[1], root_quat_wxyz[2], root_quat_wxyz[3], root_quat_wxyz[0]]
        )

        # Root velocities (world frame)
        root_lin_vel_world = mj_data.qvel[0:3].copy()
        root_ang_vel_world = mj_data.qvel[3:6].copy()

        # Convert to heading-local frame
        rot = R.from_quat(root_quat_xyzw)
        euler = rot.as_euler("xyz")
        yaw = euler[2]

        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)

        # R_z(-yaw) rotation
        root_lin_vel_local = np.array(
            [
                cos_yaw * root_lin_vel_world[0] + sin_yaw * root_lin_vel_world[1],
                -sin_yaw * root_lin_vel_world[0] + cos_yaw * root_lin_vel_world[1],
                root_lin_vel_world[2],
            ]
        )
        root_ang_vel_local = np.array(
            [
                cos_yaw * root_ang_vel_world[0] + sin_yaw * root_ang_vel_world[1],
                -sin_yaw * root_ang_vel_world[0] + cos_yaw * root_ang_vel_world[1],
                root_ang_vel_world[2],
            ]
        )

        # Root height
        root_height = root_pos[2]

        # Estimate foot contacts from joint angles (matches policy extractor)
        # Using hip-pitch heuristic (same as extract_amp_features)
        # Joint order from robot_config: left_hip_pitch=0, left_hip_roll=1, ...
        joint_names = robot_config.actuator_joints

        try:
            left_hip_idx = joint_names.index("left_hip_pitch")
            right_hip_idx = joint_names.index("right_hip_pitch")
            left_knee_idx = joint_names.index("left_knee_pitch")
            right_knee_idx = joint_names.index("right_knee_pitch")
        except ValueError:
            # Fallback indices if names don't match exactly
            left_hip_idx = 0
            right_hip_idx = 4
            left_knee_idx = 2
            right_knee_idx = 6

        threshold_angle = 0.1
        knee_scale = 0.5
        min_confidence = 0.3

        left_hip_pitch = joint_pos[left_hip_idx]
        right_hip_pitch = joint_pos[right_hip_idx]
        left_knee = joint_pos[left_knee_idx]
        right_knee = joint_pos[right_knee_idx]

        # Left foot contact
        left_hip_contact = 1.0 if left_hip_pitch < -threshold_angle else 0.0
        left_confidence = np.clip(
            1.0 - abs(left_knee) / knee_scale, min_confidence, 1.0
        )
        left_contact = left_hip_contact * left_confidence

        # Right foot contact
        right_hip_contact = 1.0 if right_hip_pitch < -threshold_angle else 0.0
        right_confidence = np.clip(
            1.0 - abs(right_knee) / knee_scale, min_confidence, 1.0
        )
        right_contact = right_hip_contact * right_confidence

        # 4-point contacts: [left_toe, left_heel, right_toe, right_heel]
        foot_contacts = np.array(
            [left_contact, left_contact, right_contact, right_contact]
        )

        # Build feature vector
        # [joint_pos(N), joint_vel(N), root_lin_vel(3), root_ang_vel(3), root_height(1), contacts(4)]
        feature = np.concatenate(
            [
                joint_pos,  # N dims
                joint_vel,  # N dims
                root_lin_vel_local,  # 3 dims
                root_ang_vel_local,  # 3 dims
                [root_height],  # 1 dim
                foot_contacts,  # 4 dims
            ]
        )

        features_list.append(feature)

    return np.array(features_list, dtype=np.float32)


def compare_statistics(
    ref_stats: Dict[str, Any],
    policy_stats: Dict[str, Any],
    feature_names: list,
) -> Tuple[bool, list]:
    """Compare feature statistics and report mismatches."""
    issues = []

    # Check dimensions
    if ref_stats["shape"][1] != policy_stats["shape"][1]:
        issues.append(
            f"CRITICAL: Feature dimension mismatch! "
            f"Reference={ref_stats['shape'][1]}, Policy={policy_stats['shape'][1]}"
        )
        return False, issues

    feature_dim = ref_stats["shape"][1]

    # Per-feature comparison
    for i in range(feature_dim):
        fname = feature_names[i] if i < len(feature_names) else f"feature_{i}"

        ref_mean = ref_stats["mean"][i]
        policy_mean = policy_stats["mean"][i]
        ref_std = ref_stats["std"][i]
        policy_std = policy_stats["std"][i]
        ref_range = ref_stats["max"][i] - ref_stats["min"][i]
        policy_range = policy_stats["max"][i] - policy_stats["min"][i]

        # Check for large mean differences (relative to std)
        mean_diff = abs(ref_mean - policy_mean)
        combined_std = max(ref_std, policy_std, 0.01)

        if mean_diff > 3 * combined_std:
            issues.append(
                f"WARNING: {fname} mean differs significantly: "
                f"ref={ref_mean:.3f}, policy={policy_mean:.3f} (diff={mean_diff:.3f})"
            )

        # Check for very different ranges
        if ref_range > 0.01 and policy_range > 0.01:
            range_ratio = max(ref_range, policy_range) / min(ref_range, policy_range)
            if range_ratio > 5:
                issues.append(
                    f"WARNING: {fname} range differs significantly: "
                    f"ref=[{ref_stats['min'][i]:.3f}, {ref_stats['max'][i]:.3f}], "
                    f"policy=[{policy_stats['min'][i]:.3f}, {policy_stats['max'][i]:.3f}]"
                )

        # Check for constant features (zero std)
        if ref_std < 1e-6 and policy_std > 0.01:
            issues.append(
                f"WARNING: {fname} is constant in reference but varies in policy"
            )
        elif policy_std < 1e-6 and ref_std > 0.01:
            issues.append(
                f"WARNING: {fname} is constant in policy but varies in reference"
            )

    passed = len([i for i in issues if "CRITICAL" in i]) == 0
    return passed, issues


def print_comparison_table(
    ref_stats: Dict[str, Any],
    policy_stats: Dict[str, Any],
    feature_names: list,
):
    """Print detailed comparison table."""
    table = Table(title="Feature Parity Comparison")

    table.add_column("Feature", style="cyan")
    table.add_column("Ref Mean±Std", style="green")
    table.add_column("Policy Mean±Std", style="yellow")
    table.add_column("Ref Range", style="green")
    table.add_column("Policy Range", style="yellow")
    table.add_column("Status", style="bold")

    feature_dim = ref_stats["shape"][1]

    for i in range(feature_dim):
        fname = feature_names[i] if i < len(feature_names) else f"feature_{i}"

        ref_mean = ref_stats["mean"][i]
        policy_mean = policy_stats["mean"][i]
        ref_std = ref_stats["std"][i]
        policy_std = policy_stats["std"][i]

        ref_mean_std = f"{ref_mean:>7.3f}±{ref_std:.3f}"
        policy_mean_std = f"{policy_mean:>7.3f}±{policy_std:.3f}"

        ref_range = f"[{ref_stats['min'][i]:>6.2f}, {ref_stats['max'][i]:>6.2f}]"
        policy_range = (
            f"[{policy_stats['min'][i]:>6.2f}, {policy_stats['max'][i]:>6.2f}]"
        )

        # Determine status
        mean_diff = abs(ref_mean - policy_mean)
        combined_std = max(ref_std, policy_std, 0.01)

        if mean_diff > 3 * combined_std:
            status = "⚠️  DIFF"
        elif ref_std < 1e-6 or policy_std < 1e-6:
            status = "⚠️  CONST"
        else:
            status = "✓"

        table.add_row(
            fname, ref_mean_std, policy_mean_std, ref_range, policy_range, status
        )

    console.print(table)


def analyze_contacts(
    ref_features: np.ndarray, policy_features: np.ndarray, num_joints: int
):
    """Analyze foot contact patterns."""
    # Contact indices: last 4 features
    contact_start = num_joints * 2 + 7  # 2*N + 3 + 3 + 1 = 2*N + 7

    ref_contacts = ref_features[:, contact_start : contact_start + 4]
    policy_contacts = policy_features[:, contact_start : contact_start + 4]

    # Contact rates (fraction of frames with contact > 0.5)
    ref_contact_rate = np.mean(ref_contacts > 0.5, axis=0)
    policy_contact_rate = np.mean(policy_contacts > 0.5, axis=0)

    # Double stance rate
    ref_left = np.any(ref_contacts[:, :2] > 0.5, axis=1)
    ref_right = np.any(ref_contacts[:, 2:] > 0.5, axis=1)
    ref_double = np.mean(ref_left & ref_right)

    policy_left = np.any(policy_contacts[:, :2] > 0.5, axis=1)
    policy_right = np.any(policy_contacts[:, 2:] > 0.5, axis=1)
    policy_double = np.mean(policy_left & policy_right)

    print("\n[bold blue]Foot Contact Analysis[/bold blue]")
    print(f"  Contact rates (L_toe, L_heel, R_toe, R_heel):")
    print(f"    Reference: {ref_contact_rate}")
    print(f"    Policy:    {policy_contact_rate}")
    print(f"  Double stance rate:")
    print(f"    Reference: {ref_double:.1%}")
    print(f"    Policy:    {policy_double:.1%}")


def main():
    parser = argparse.ArgumentParser(
        description="Physics Reference Feature Parity Test"
    )
    parser.add_argument(
        "--physics-data",
        type=str,
        default="playground_amp/data/physics_ref/walking_physics_merged.pkl",
        help="Path to physics reference dataset",
    )
    parser.add_argument(
        "--robot-config",
        type=str,
        default="assets/robot_config.yaml",
        help="Path to robot_config.yaml",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="assets/scene_flat_terrain.xml",
        help="Path to MuJoCo scene XML",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=500,
        help="Number of frames to simulate for policy features",
    )

    args = parser.parse_args()

    # Load configurations
    print("[bold blue]Loading configurations...[/bold blue]")

    from playground_amp.amp.policy_features import create_config_from_robot
    from playground_amp.configs.training_config import get_robot_config, load_robot_config

    load_robot_config(args.robot_config)
    robot_config = get_robot_config()
    amp_config = create_config_from_robot(robot_config)

    num_joints = robot_config.action_dim
    print(f"  Robot joints: {num_joints}")
    print(f"  Joint names: {robot_config.actuator_joints}")
    print(f"  Feature dimension: {amp_config.feature_dim}")

    # Load MuJoCo model
    import mujoco

    model_path = Path(args.model)
    assets_dir = model_path.parent

    # Load assets
    assets = {}
    for xml_file in assets_dir.glob("*.xml"):
        assets[xml_file.name] = xml_file.read_bytes()
    meshes_dir = assets_dir / "assets"
    if meshes_dir.exists():
        for mesh_file in meshes_dir.glob("*.stl"):
            assets[mesh_file.name] = mesh_file.read_bytes()

    mj_model = mujoco.MjModel.from_xml_string(model_path.read_text(), assets)
    print(f"  MuJoCo model loaded: {mj_model.nq} DOF")

    # Load physics reference data
    print(f"\n[bold blue]Loading physics reference data...[/bold blue]")
    print(f"  Path: {args.physics_data}")

    ref_data = load_physics_reference(args.physics_data)
    ref_features = ref_data["features"]

    print(f"  Shape: {ref_features.shape}")
    print(f"  Duration: {ref_data.get('duration_sec', 'N/A')}s")
    print(f"  FPS: {ref_data.get('fps', 'N/A')}")

    # Generate simulated policy features
    print(
        f"\n[bold blue]Simulating policy rollout ({args.num_frames} frames)...[/bold blue]"
    )

    policy_features = simulate_policy_features(
        mj_model, robot_config, amp_config, num_frames=args.num_frames, dt=0.02
    )
    print(f"  Policy features shape: {policy_features.shape}")

    # Build feature names
    joint_names = robot_config.actuator_joints
    feature_names = (
        [f"pos_{j}" for j in joint_names]
        + [f"vel_{j}" for j in joint_names]
        + ["root_lin_vel_x", "root_lin_vel_y", "root_lin_vel_z"]
        + ["root_ang_vel_x", "root_ang_vel_y", "root_ang_vel_z"]
        + ["root_height"]
        + ["contact_L_toe", "contact_L_heel", "contact_R_toe", "contact_R_heel"]
    )

    # Compute statistics
    print(f"\n[bold blue]Computing feature statistics...[/bold blue]")

    ref_stats = analyze_feature_statistics(ref_features, "Reference")
    policy_stats = analyze_feature_statistics(policy_features, "Policy")

    # Print comparison table
    print_comparison_table(ref_stats, policy_stats, feature_names)

    # Compare and report issues
    print(f"\n[bold blue]Parity Check Results[/bold blue]")

    passed, issues = compare_statistics(ref_stats, policy_stats, feature_names)

    if passed and not issues:
        print("[bold green]✓ All parity checks passed![/bold green]")
    elif passed:
        print("[bold yellow]⚠ Parity checks passed with warnings:[/bold yellow]")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("[bold red]✗ Parity checks failed:[/bold red]")
        for issue in issues:
            print(f"  {issue}")

    # Analyze contacts
    analyze_contacts(ref_features, policy_features, num_joints)

    # Summary
    print(f"\n{'='*60}")
    print("[bold]Summary[/bold]")
    print(f"  Reference: {ref_features.shape[0]} frames, {ref_features.shape[1]} dims")
    print(
        f"  Policy:    {policy_features.shape[0]} frames, {policy_features.shape[1]} dims"
    )
    print(
        f"  Dimension match: {'✓' if ref_features.shape[1] == policy_features.shape[1] else '✗'}"
    )
    print(f"  Issues found: {len(issues)}")

    if passed:
        print(
            f"\n[bold green]✓ Physics reference data is compatible with policy features![/bold green]"
        )
        print(
            f"  Ready for training with: playground_amp/data/physics_ref/walking_physics_merged.pkl"
        )
    else:
        print(
            f"\n[bold red]✗ Feature parity issues detected - fix before training![/bold red]"
        )

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
