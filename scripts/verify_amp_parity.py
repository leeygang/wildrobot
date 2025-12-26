#!/usr/bin/env python3
"""Verify AMP feature parity between reference data and sim replay.

v0.10.0: Industry Golden Rule - TRUE qvel support for physics reference data.
v0.7.0: Verification script for heading-local frame implementation.

This script:
1. Loads reference AMP features from a converted motion clip
2. Replays the same motion in MuJoCo sim (open-loop PD tracking)
3. Extracts policy AMP features during replay
4. Compares per-dimension mean absolute error

v0.10.0: Physics Reference Support
For physics reference data (has dof_vel and foot_contacts stored), this script
automatically uses TRUE qvel instead of finite-diff velocities. This follows
the industry golden rule: use TRUE qvel from MuJoCo everywhere.

If heading-local frame parity is correct:
- Linear velocity dimensions should match closely (MAE < 0.1 m/s)
- Angular velocity dimensions should match closely (MAE < 0.1 rad/s)
- The discriminator should immediately move off 0.50 in early training

Usage:
    uv run python scripts/verify_amp_parity.py \
        --reference data/amp/walking_reference.pkl \
        --model assets/scene_flat_terrain.xml \
        --config configs/ppo_amass_training.yaml

Example output:
    Feature Parity Report:
    ----------------------
    Joint pos (0-7):     MAE = 0.012 rad
    Joint vel (8-15):    MAE = 0.045 rad/s
    Root lin vel (16-18): MAE = 0.078 m/s    <-- Should be < 0.1 after fix
    Root ang vel (19-21): MAE = 0.034 rad/s  <-- Should be < 0.1 after fix
    Root height (22):    MAE = 0.003 m
    Foot contacts (23-26): MAE = 0.15
"""

import argparse
import pickle

# Add parent path for imports
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx
from rich import print
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

from playground_amp.amp.policy_features import (
    FeatureConfig,
    create_config_from_robot,
    extract_amp_features,
)
from playground_amp.configs.training_config import get_robot_config, load_robot_config


def load_reference_data(path: str) -> Dict[str, Any]:
    """Load reference motion data.

    Args:
        path: Path to reference pickle file

    Returns:
        Reference data dict with 'features', 'dof_pos', 'fps', etc.
    """
    with open(path, "rb") as f:
        data = pickle.load(f)

    print(f"[cyan]Loaded reference:[/cyan] {path}")
    print(f"  Frames: {data['num_frames']}")
    print(f"  FPS: {data['fps']}")
    print(f"  Feature dim: {data['feature_dim']}")
    print(f"  Duration: {data['duration_sec']:.2f}s")

    # v0.10.0: Detect physics reference data
    is_physics_ref = "dof_vel" in data and "foot_contacts" in data
    if is_physics_ref:
        print(f"  [yellow]Physics reference detected: has TRUE qvel stored[/yellow]")
    else:
        print(f"  [dim]GMR reference: will use finite-diff velocities[/dim]")

    return data


def setup_mujoco_sim(model_path: str) -> Tuple[mujoco.MjModel, mujoco.MjData]:
    """Load MuJoCo model for simulation.

    Args:
        model_path: Path to scene XML file

    Returns:
        (mj_model, mj_data) tuple
    """
    from playground_amp.envs.wildrobot_env import get_assets

    model_path = Path(model_path)
    assets_dir = model_path.parent

    mj_model = mujoco.MjModel.from_xml_string(
        model_path.read_text(), assets=get_assets(assets_dir)
    )
    mj_data = mujoco.MjData(mj_model)

    print(f"[cyan]Loaded model:[/cyan] {model_path}")
    print(f"  nq: {mj_model.nq}, nv: {mj_model.nv}, nu: {mj_model.nu}")

    return mj_model, mj_data


def extract_sim_features(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    config: FeatureConfig,
    prev_joint_pos: np.ndarray,
    dt: float,
    use_true_qvel: bool = False,
    true_dof_vel: np.ndarray = None,
    true_foot_contacts: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract AMP features from current sim state.

    This mimics the policy's feature extraction during training.

    v0.10.0: TRUE qvel support for physics reference data.
    When use_true_qvel=True, uses the stored TRUE velocities and contacts
    instead of computing finite-diff or FK-estimated values.

    Args:
        mj_model: MuJoCo model
        mj_data: Current simulation data
        config: AMP feature config
        prev_joint_pos: Previous joint positions for finite diff
        dt: Time step
        use_true_qvel: If True, use TRUE qvel from reference (v0.10.0)
        true_dof_vel: TRUE joint velocities from reference (v0.10.0)
        true_foot_contacts: TRUE foot contacts from reference (v0.10.0)

    Returns:
        (features, current_joint_pos) tuple
    """
    robot_config = get_robot_config()

    # Get current state
    # Floating base: qpos[0:7] = [x, y, z, qw, qx, qy, qz]
    root_pos = mj_data.qpos[0:3]
    root_quat = mj_data.qpos[3:7]  # wxyz in MuJoCo

    # Joint positions (actuated joints start at index 7 for floating base)
    joint_pos = mj_data.qpos[7 : 7 + config.num_actuated_joints].copy()

    # Root height (waist height)
    root_height = root_pos[2]

    # Compute heading sin/cos from root quaternion
    w, x, y, z = root_quat
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    norm = np.sqrt(siny_cosp**2 + cosy_cosp**2 + 1e-8)
    sin_yaw = siny_cosp / norm
    cos_yaw = cosy_cosp / norm

    # Build observation vector (35-dim v0.7.0 layout)
    # Note: heading sin/cos not in obs - velocities are pre-converted to heading-local
    # Get gravity vector in body frame (from sensor if available, or compute)
    # For simplicity, we'll compute from quaternion
    gravity_world = np.array([0, 0, -1])
    # Rotate to body frame: R_body_from_world^T @ g_world
    # Using quaternion rotation formula: q^{-1} * v * q
    # Simplified: just project gravity onto body axes
    gravity_body = quaternion_rotate_inverse(root_quat, gravity_world)

    # Get velocities from sensors or qvel
    # Linear velocity: from qvel (world frame), then convert to heading-local
    lin_vel_world = mj_data.qvel[0:3]
    lin_vel_heading = world_to_heading_local_np(lin_vel_world, cos_yaw, sin_yaw)

    # Angular velocity: from qvel (world frame), then convert to heading-local
    ang_vel_world = mj_data.qvel[3:6]
    ang_vel_heading = world_to_heading_local_np(ang_vel_world, cos_yaw, sin_yaw)

    # v0.10.0: Use TRUE qvel if available, otherwise use MuJoCo qvel
    if use_true_qvel and true_dof_vel is not None:
        joint_vel = true_dof_vel.copy()
    else:
        joint_vel = mj_data.qvel[6 : 6 + config.num_actuated_joints].copy()

    # Build 35-dim observation (v0.7.0 layout - no heading sin/cos in obs)
    obs = np.concatenate(
        [
            gravity_body,  # 0-2: gravity in body frame (3)
            ang_vel_heading,  # 3-5: angular velocity (heading-local) (3)
            lin_vel_heading,  # 6-8: linear velocity (heading-local) (3)
            joint_pos,  # 9-16: joint positions (8)
            joint_vel,  # 17-24: joint velocities (8)
            np.zeros(8),  # 25-32: previous action (not used) (8)
            np.array([0.0]),  # 33: velocity command (1)
            np.array([0.0]),  # 34: padding (1)
        ]
    ).astype(np.float32)

    # Convert to JAX for feature extraction
    obs_jax = jnp.array(obs)
    prev_joint_pos_jax = (
        jnp.array(prev_joint_pos) if prev_joint_pos is not None else None
    )

    # v0.10.0: Extract AMP features with TRUE qvel support
    # For physics reference, use TRUE contacts and TRUE qvel from observation
    if use_true_qvel and true_foot_contacts is not None:
        features = extract_amp_features(
            obs=obs_jax,
            config=config,
            root_height=jnp.array(root_height),
            prev_joint_pos=prev_joint_pos_jax,
            dt=dt,
            foot_contacts_override=jnp.array(true_foot_contacts),
            use_obs_joint_vel=True,  # Use TRUE qvel from observation
        )
    else:
        # GMR reference: estimate contacts and use finite-diff internally
        features = extract_amp_features(
            obs=obs_jax,
            config=config,
            root_height=jnp.array(root_height),
            prev_joint_pos=prev_joint_pos_jax,
            dt=dt,
        )

    return np.array(features), joint_pos


def quaternion_rotate_inverse(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate vector by inverse of quaternion (wxyz format)."""
    w, x, y, z = quat
    # Conjugate: negate xyz
    qc = np.array([w, -x, -y, -z])
    return quaternion_rotate(qc, vec)


def quaternion_rotate(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate vector by quaternion (wxyz format)."""
    w, x, y, z = quat
    vx, vy, vz = vec

    # Quaternion multiplication: q * v * q^{-1}
    # where v = (0, vx, vy, vz) as quaternion
    # Optimized formula for pure quaternion rotation
    t = 2 * np.array(
        [
            y * vz - z * vy,
            z * vx - x * vz,
            x * vy - y * vx,
        ]
    )

    return vec + w * t + np.cross([x, y, z], t)


def world_to_heading_local_np(
    vec: np.ndarray,
    cos_yaw: float,
    sin_yaw: float,
) -> np.ndarray:
    """Convert world-frame vector to heading-local frame (numpy version)."""
    vx, vy, vz = vec
    vx_local = cos_yaw * vx + sin_yaw * vy
    vy_local = -sin_yaw * vx + cos_yaw * vy
    return np.array([vx_local, vy_local, vz])


def estimate_foot_contacts_np(
    joint_pos: np.ndarray,
    config: FeatureConfig,
) -> np.ndarray:
    """Estimate foot contacts from joint positions (numpy version)."""
    left_hip = joint_pos[config.left_hip_pitch_idx]
    left_knee = joint_pos[config.left_knee_pitch_idx]
    right_hip = joint_pos[config.right_hip_pitch_idx]
    right_knee = joint_pos[config.right_knee_pitch_idx]

    threshold = 0.1

    left_contact = float(left_hip < threshold) * np.clip(
        1.0 - abs(left_knee) / 0.5, 0.3, 1.0
    )
    right_contact = float(right_hip < threshold) * np.clip(
        1.0 - abs(right_knee) / 0.5, 0.3, 1.0
    )

    return np.array([left_contact, left_contact, right_contact, right_contact])


def replay_motion_and_extract_features(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    ref_data: Dict[str, Any],
    config: FeatureConfig,
) -> np.ndarray:
    """Replay reference motion in sim and extract features.

    v0.10.0: Physics Reference Support
    Automatically detects physics reference data and uses TRUE qvel.

    Args:
        mj_model: MuJoCo model
        mj_data: MuJoCo data
        ref_data: Reference motion data
        config: AMP feature config

    Returns:
        Extracted features shape (N, feature_dim)
    """
    ref_dof_pos = ref_data["dof_pos"]
    ref_root_pos = ref_data["root_pos"]
    ref_root_rot = ref_data["root_rot"]  # xyzw in reference
    num_frames = ref_data["num_frames"]
    dt = ref_data["dt"]

    # v0.10.0: Detect physics reference data (has TRUE qvel stored)
    is_physics_ref = "dof_vel" in ref_data and "foot_contacts" in ref_data
    if is_physics_ref:
        print(
            "  [yellow]Physics reference detected: using TRUE qvel (industry golden rule)[/yellow]"
        )
        ref_dof_vel = ref_data["dof_vel"]
        ref_foot_contacts = ref_data["foot_contacts"]
        ref_features = ref_data["features"]
        n = config.num_actuated_joints
    else:
        print("  [dim]GMR reference: using finite-diff velocities[/dim]")
        ref_dof_vel = None
        ref_foot_contacts = None
        ref_features = None

    # Set simulation timestep
    mj_model.opt.timestep = dt

    all_features = []
    prev_joint_pos = None

    print(f"\n[cyan]Replaying {num_frames} frames in simulation...[/cyan]")

    for i in range(num_frames):
        # Set root pose
        mj_data.qpos[0:3] = ref_root_pos[i]

        # Convert xyzw to wxyz for MuJoCo
        ref_quat_xyzw = ref_root_rot[i]
        mj_data.qpos[3:7] = [
            ref_quat_xyzw[3],
            ref_quat_xyzw[0],
            ref_quat_xyzw[1],
            ref_quat_xyzw[2],
        ]

        # Set joint positions
        mj_data.qpos[7 : 7 + config.num_actuated_joints] = ref_dof_pos[i]

        # Forward kinematics (updates sensor values and derived quantities)
        mujoco.mj_forward(mj_model, mj_data)

        # Get current joint positions BEFORE computing velocities
        current_joint_pos = mj_data.qpos[7 : 7 + config.num_actuated_joints].copy()

        # For first frame, use current as prev (velocity will be zero)
        if prev_joint_pos is None:
            prev_joint_pos = current_joint_pos.copy()

        # v0.10.0: Use TRUE qvel for physics reference, finite-diff for GMR
        if is_physics_ref:
            # Extract heading-local velocities from reference features
            ref_lin_vel_heading = ref_features[i, 2 * n : 2 * n + 3]
            ref_ang_vel_heading = ref_features[i, 2 * n + 3 : 2 * n + 6]

            # Compute yaw for world-frame conversion
            w, x, y, z = mj_data.qpos[3:7]
            siny_cosp = 2.0 * (w * z + x * y)
            cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
            norm = np.sqrt(siny_cosp**2 + cosy_cosp**2 + 1e-8)
            sin_yaw, cos_yaw = siny_cosp / norm, cosy_cosp / norm

            # Convert heading-local to world frame for qvel injection
            lin_vel_world = np.array(
                [
                    cos_yaw * ref_lin_vel_heading[0] - sin_yaw * ref_lin_vel_heading[1],
                    sin_yaw * ref_lin_vel_heading[0] + cos_yaw * ref_lin_vel_heading[1],
                    ref_lin_vel_heading[2],
                ]
            )
            ang_vel_world = np.array(
                [
                    cos_yaw * ref_ang_vel_heading[0] - sin_yaw * ref_ang_vel_heading[1],
                    sin_yaw * ref_ang_vel_heading[0] + cos_yaw * ref_ang_vel_heading[1],
                    ref_ang_vel_heading[2],
                ]
            )

            # Inject TRUE qvel into MuJoCo data
            mj_data.qvel[0:3] = lin_vel_world
            mj_data.qvel[3:6] = ang_vel_world
            mj_data.qvel[6 : 6 + config.num_actuated_joints] = ref_dof_vel[i]

            # Extract features with TRUE qvel
            features, _ = extract_sim_features(
                mj_model,
                mj_data,
                config,
                prev_joint_pos,
                dt,
                use_true_qvel=True,
                true_dof_vel=ref_dof_vel[i],
                true_foot_contacts=ref_foot_contacts[i],
            )
        else:
            # GMR reference: compute velocities using finite differences
            if i > 0:
                mj_data.qvel[0:3] = (ref_root_pos[i] - ref_root_pos[i - 1]) / dt

                # Angular velocity from quaternion using axis-angle method
                # This matches the reference data computation (SciPy Rotation)
                from scipy.spatial.transform import Rotation as R

                q0_xyzw = ref_root_rot[i - 1]
                q1_xyzw = ref_root_rot[i]

                r_prev = R.from_quat(q0_xyzw)
                r_curr = R.from_quat(q1_xyzw)

                # Relative rotation: R_delta = R_curr * R_prev.inv()
                r_delta = r_curr * r_prev.inv()
                rotvec = r_delta.as_rotvec()
                mj_data.qvel[3:6] = rotvec / dt

                # Joint velocities
                mj_data.qvel[6 : 6 + config.num_actuated_joints] = (
                    ref_dof_pos[i] - ref_dof_pos[i - 1]
                ) / dt

            # Extract features with finite-diff
            features, _ = extract_sim_features(
                mj_model, mj_data, config, prev_joint_pos, dt
            )

        all_features.append(features)
        prev_joint_pos = current_joint_pos

        if i % 100 == 0:
            print(f"  Frame {i}/{num_frames}")

    return np.stack(all_features, axis=0)


def compute_parity_report(
    ref_features: np.ndarray,
    sim_features: np.ndarray,
    config: FeatureConfig,
) -> Dict[str, float]:
    """Compute per-component MAE between reference and sim features.

    Args:
        ref_features: Reference features (N, feature_dim)
        sim_features: Simulated features (N, feature_dim)
        config: AMP feature config

    Returns:
        Dict of component names to MAE values
    """
    num_joints = config.num_actuated_joints

    # Define feature ranges
    ranges = {
        f"Joint pos (0-{num_joints-1})": (0, num_joints),
        f"Joint vel ({num_joints}-{2*num_joints-1})": (num_joints, 2 * num_joints),
        f"Root lin vel ({2*num_joints}-{2*num_joints+2})": (
            2 * num_joints,
            2 * num_joints + 3,
        ),
        f"Root ang vel ({2*num_joints+3}-{2*num_joints+5})": (
            2 * num_joints + 3,
            2 * num_joints + 6,
        ),
        f"Root height ({2*num_joints+6})": (2 * num_joints + 6, 2 * num_joints + 7),
        f"Foot contacts ({2*num_joints+7}-{2*num_joints+10})": (
            2 * num_joints + 7,
            2 * num_joints + 11,
        ),
    }

    results = {}
    for name, (start, end) in ranges.items():
        ref_comp = ref_features[:, start:end]
        sim_comp = sim_features[:, start:end]
        mae = np.mean(np.abs(ref_comp - sim_comp))
        results[name] = mae

    return results


def print_parity_report(results: Dict[str, float]) -> None:
    """Print formatted parity report."""
    console = Console()

    table = Table(title="AMP Feature Parity Report (v0.7.0 Heading-Local)")
    table.add_column("Component", style="cyan")
    table.add_column("MAE", justify="right")
    table.add_column("Status", justify="center")

    thresholds = {
        "Joint pos": 0.05,
        "Joint vel": 0.2,
        "Root lin vel": 0.1,  # Key metric for heading-local fix
        "Root ang vel": 0.1,  # Key metric for heading-local fix
        "Root height": 0.01,
        "Foot contacts": 0.3,
    }

    for component, mae in results.items():
        # Find matching threshold
        threshold = 0.1  # default
        for key, th in thresholds.items():
            if key in component:
                threshold = th
                break

        status = "[green]✓ PASS[/green]" if mae < threshold else "[red]✗ FAIL[/red]"
        table.add_row(component, f"{mae:.4f}", status)

    console.print(table)

    # Summary
    lin_vel_key = [k for k in results.keys() if "lin vel" in k][0]
    ang_vel_key = [k for k in results.keys() if "ang vel" in k][0]

    lin_vel_mae = results[lin_vel_key]
    ang_vel_mae = results[ang_vel_key]

    print(f"\n[bold]Key Metrics (Heading-Local Frame):[/bold]")
    if lin_vel_mae < 0.1 and ang_vel_mae < 0.1:
        print("[green]✓ Frame parity VERIFIED[/green]")
        print("  The discriminator should now distinguish real from fake motion.")
    else:
        print("[red]✗ Frame parity FAILED[/red]")
        print("  Check that both policy and reference use heading-local frame.")


def main():
    parser = argparse.ArgumentParser(description="Verify AMP feature parity")
    parser.add_argument(
        "--reference",
        type=str,
        required=True,
        help="Path to reference AMP pickle file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="assets/scene_flat_terrain.xml",
        help="Path to MuJoCo scene XML",
    )
    parser.add_argument(
        "--robot-config",
        type=str,
        default="assets/robot_config.yaml",
        help="Path to robot_config.yaml",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=500,
        help="Maximum frames to process (for speed)",
    )

    args = parser.parse_args()

    # Load robot config
    print(f"[bold blue]Loading robot config:[/bold blue] {args.robot_config}")
    load_robot_config(args.robot_config)
    robot_config = get_robot_config()
    amp_config = create_config_from_robot(robot_config)

    # Load reference data
    print(f"\n[bold blue]Loading reference data:[/bold blue]")
    ref_data = load_reference_data(args.reference)

    # Limit frames if needed
    if ref_data["num_frames"] > args.max_frames:
        print(f"[yellow]Limiting to {args.max_frames} frames[/yellow]")
        ref_data["num_frames"] = args.max_frames
        ref_data["features"] = ref_data["features"][: args.max_frames]
        ref_data["dof_pos"] = ref_data["dof_pos"][: args.max_frames]
        ref_data["root_pos"] = ref_data["root_pos"][: args.max_frames]
        ref_data["root_rot"] = ref_data["root_rot"][: args.max_frames]

    # Load MuJoCo model
    print(f"\n[bold blue]Loading MuJoCo model:[/bold blue]")
    mj_model, mj_data = setup_mujoco_sim(args.model)

    # Replay motion and extract features
    print(f"\n[bold blue]Extracting sim features:[/bold blue]")
    sim_features = replay_motion_and_extract_features(
        mj_model, mj_data, ref_data, amp_config
    )

    # Compare features
    print(f"\n[bold blue]Computing parity:[/bold blue]")
    ref_features = ref_data["features"]

    # Ensure same length (sim may have one fewer due to finite diff)
    min_len = min(len(ref_features), len(sim_features))
    ref_features = ref_features[:min_len]
    sim_features = sim_features[:min_len]

    results = compute_parity_report(ref_features, sim_features, amp_config)

    print("\n")
    print_parity_report(results)

    # Save detailed comparison for analysis
    output_path = Path(args.reference).with_suffix(".parity.npz")
    np.savez(
        output_path,
        ref_features=ref_features,
        sim_features=sim_features,
        mae_per_component=results,
    )
    print(f"\n[green]Detailed comparison saved to:[/green] {output_path}")


if __name__ == "__main__":
    main()
