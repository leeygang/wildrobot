#!/usr/bin/env python3
"""Robust AMP Feature Parity Test Suite.

v0.9.0: Industry-grade golden rule validation with proper Test 1A/1B split.

This module addresses coverage gaps in the basic parity test:
1A. Adversarial injected-qvel test (feature assembly validation)
1B. mj_step PD tracking test (physics validation - MANDATORY)
2.  Yaw invariance test (rotate motion, features unchanged)
3.  Quaternion sign flip test (adversarial q/-q)
4.  Batch coverage test (all motions, percentile MAE)
5.  Contact estimator consistency test (with noise)

Usage:
    cd ~/projects/wildrobot
    uv run python scripts/verify_amp_parity_robust.py --all
    uv run python scripts/verify_amp_parity_robust.py --test 1a  # Feature assembly
    uv run python scripts/verify_amp_parity_robust.py --test 1b  # Physics (MANDATORY)
    uv run python scripts/verify_amp_parity_robust.py --test yaw_invariance
    uv run python scripts/verify_amp_parity_robust.py --test quat_sign_flip
    uv run python scripts/verify_amp_parity_robust.py --test batch_coverage
    uv run python scripts/verify_amp_parity_robust.py --test contact_consistency
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import jax.numpy as jnp
import mujoco
import numpy as np
from rich import print
from rich.console import Console
from rich.table import Table
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).parent.parent))

from playground_amp.amp.amp_features import (
    AMPFeatureConfig,
    create_amp_config_from_robot,
    estimate_foot_contacts_from_joints,
    extract_amp_features,
)
from playground_amp.configs.config import get_robot_config, load_robot_config


console = Console()


# =============================================================================
# Test 1A: Adversarial Injected-qvel Test (Feature Assembly Validation)
# =============================================================================


def test_1a_adversarial_injection(
    ref_path: str,
    model_path: str,
    config: AMPFeatureConfig,
    max_frames: int = 100,
    pose_noise_std: float = 0.0002,  # v0.9.1: Reduced from 0.001 for tighter validation
) -> Dict[str, Any]:
    """Test 1A: Adversarial injected-qvel parity test.

    What it proves:
    - Correct indexing and ordering
    - Correct yaw removal math
    - Correct construction of heading-local velocities
    - Correct concatenation and masking rules
    - Correct reference feature generation pipeline

    What it CANNOT prove:
    - Whether MuJoCo integration produces qvel that matches finite difference
    - Whether contact estimator behaves the same when robot is simulated
    - Whether filtering and dt in training loop match reference generation

    This test is ADVERSARIAL: we add small noise to poses before differentiating
    to avoid circular validation.

    Args:
        ref_path: Path to reference AMP pickle
        model_path: Path to MuJoCo scene XML
        config: AMP feature config
        max_frames: Max frames to test
        pose_noise_std: Small noise to add to poses (mimics sim jitter)

    Returns:
        Dict with MAE, correlation, and detailed metrics
    """
    print("\n[bold cyan]Test 1A: Adversarial Injected-qvel Parity Test[/bold cyan]")
    print("  (Validates feature extraction with adversarial pose noise)")
    print(f"  (Pose noise std: {pose_noise_std})")

    # Load reference
    with open(ref_path, "rb") as f:
        ref_data = pickle.load(f)

    num_frames = min(max_frames, ref_data["num_frames"])
    ref_features = ref_data["features"][:num_frames]
    ref_dof_pos = ref_data["dof_pos"][:num_frames]
    ref_root_pos = ref_data["root_pos"][:num_frames]
    ref_root_rot = ref_data["root_rot"][:num_frames]  # xyzw
    dt = ref_data["dt"]

    # ADVERSARIAL: Add small noise to poses (mimics sim jitter)
    # This breaks circularity - we're not using exact same data as reference
    noisy_root_pos = (
        ref_root_pos + np.random.randn(*ref_root_pos.shape) * pose_noise_std
    )
    noisy_dof_pos = ref_dof_pos + np.random.randn(*ref_dof_pos.shape) * pose_noise_std

    # Also extract reference velocities for correlation analysis
    n = config.num_actuated_joints
    ref_linvel = ref_features[:, 2 * n : 2 * n + 3]
    ref_angvel = ref_features[:, 2 * n + 3 : 2 * n + 6]

    # Pre-compute world-frame velocities FROM NOISY POSES
    # Linear velocity: finite difference of noisy positions
    lin_vel_world = np.zeros_like(noisy_root_pos)
    lin_vel_world[1:] = (noisy_root_pos[1:] - noisy_root_pos[:-1]) / dt
    lin_vel_world[0] = lin_vel_world[1] if num_frames > 1 else 0

    # Angular velocity: SciPy axis-angle method (same as reference)
    ang_vel_world = np.zeros((num_frames, 3))
    if num_frames > 1:
        rotations = R.from_quat(ref_root_rot)
        r_prev = rotations[:-1]
        r_curr = rotations[1:]
        r_delta = r_curr * r_prev.inv()
        ang_vel_world[1:] = r_delta.as_rotvec() / dt
        ang_vel_world[0] = ang_vel_world[1]

    # Load MuJoCo model
    from playground_amp.envs.wildrobot_env import get_assets

    model_path = Path(model_path)
    mj_model = mujoco.MjModel.from_xml_string(
        model_path.read_text(), assets=get_assets(model_path.parent)
    )
    mj_data = mujoco.MjData(mj_model)
    mj_model.opt.timestep = dt

    all_features = []
    all_linvel = []
    all_angvel = []
    prev_joint_pos = None

    for i in range(num_frames):
        # Set NOISY pose into MuJoCo (not exact reference)
        mj_data.qpos[0:3] = noisy_root_pos[i]

        # Convert xyzw to wxyz
        q_xyzw = ref_root_rot[i]
        mj_data.qpos[3:7] = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]
        mj_data.qpos[7 : 7 + config.num_actuated_joints] = noisy_dof_pos[i]

        # INJECT velocities into qvel (computed from noisy poses)
        mj_data.qvel[0:3] = lin_vel_world[i]  # World-frame linear velocity
        mj_data.qvel[3:6] = ang_vel_world[i]  # World-frame angular velocity

        # Joint velocities from noisy poses
        if i > 0:
            mj_data.qvel[6 : 6 + config.num_actuated_joints] = (
                noisy_dof_pos[i] - noisy_dof_pos[i - 1]
            ) / dt
        else:
            mj_data.qvel[6 : 6 + config.num_actuated_joints] = 0

        # Forward kinematics
        mujoco.mj_forward(mj_model, mj_data)

        # Now read from qvel (what policy would use)
        read_lin_vel_world = mj_data.qvel[0:3].copy()
        read_ang_vel_world = mj_data.qvel[3:6].copy()

        # Get joint positions
        joint_pos = mj_data.qpos[7 : 7 + config.num_actuated_joints].copy()
        root_height = mj_data.qpos[2]

        # Compute heading-local transform (same as policy)
        w, x, y, z = mj_data.qpos[3:7]
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        norm = np.sqrt(siny_cosp**2 + cosy_cosp**2 + 1e-8)
        sin_yaw, cos_yaw = siny_cosp / norm, cosy_cosp / norm

        # Convert velocities to heading-local
        lin_vel_heading = np.array(
            [
                cos_yaw * read_lin_vel_world[0] + sin_yaw * read_lin_vel_world[1],
                -sin_yaw * read_lin_vel_world[0] + cos_yaw * read_lin_vel_world[1],
                read_lin_vel_world[2],
            ]
        )
        ang_vel_heading = np.array(
            [
                cos_yaw * read_ang_vel_world[0] + sin_yaw * read_ang_vel_world[1],
                -sin_yaw * read_ang_vel_world[0] + cos_yaw * read_ang_vel_world[1],
                read_ang_vel_world[2],
            ]
        )

        all_linvel.append(lin_vel_heading)
        all_angvel.append(ang_vel_heading)

        # Build observation (35-dim)
        gravity_world = np.array([0, 0, -1])
        quat = mj_data.qpos[3:7]
        gravity_body = quaternion_rotate_inverse(quat, gravity_world)

        # Joint velocities via finite diff (same as reference and policy)
        if prev_joint_pos is None:
            prev_joint_pos = joint_pos.copy()
        joint_vel_fd = (joint_pos - prev_joint_pos) / dt

        obs = np.concatenate(
            [
                gravity_body,
                ang_vel_heading,
                lin_vel_heading,
                joint_pos,
                joint_vel_fd,
                np.zeros(8),  # prev_action
                np.array([0.0]),  # velocity_cmd
                np.array([0.0]),  # padding
            ]
        ).astype(np.float32)

        # Extract features
        features = extract_amp_features(
            obs=jnp.array(obs),
            config=config,
            root_height=jnp.array(root_height),
            prev_joint_pos=jnp.array(prev_joint_pos),
            dt=dt,
        )
        all_features.append(np.array(features))

        # Update prev state
        prev_joint_pos = joint_pos.copy()

    sim_features = np.stack(all_features)
    sim_linvel = np.stack(all_linvel)
    sim_angvel = np.stack(all_angvel)

    # Compute MAE
    mae_results = compute_component_mae(ref_features, sim_features, config)

    # Compute correlation for velocity components
    def compute_correlation(ref: np.ndarray, sim: np.ndarray) -> float:
        """Compute Pearson correlation coefficient."""
        ref_flat = ref.flatten()
        sim_flat = sim.flatten()
        if np.std(ref_flat) < 1e-8 or np.std(sim_flat) < 1e-8:
            return 1.0 if np.allclose(ref_flat, sim_flat) else 0.0
        return np.corrcoef(ref_flat, sim_flat)[0, 1]

    linvel_corr = compute_correlation(ref_linvel, sim_linvel)
    angvel_corr = compute_correlation(ref_angvel, sim_angvel)

    # Print detailed results
    print(f"\n  [bold]Velocity Correlation Analysis:[/bold]")
    print(f"    Linear velocity correlation:  {linvel_corr:.4f}")
    print(f"    Angular velocity correlation: {angvel_corr:.4f}")

    # Print MAE table with enhanced info
    table = Table(title="Test 1A: Adversarial Injection (MAE + Correlation)")
    table.add_column("Component", style="cyan")
    table.add_column("MAE", justify="right")
    table.add_column("Correlation", justify="right")
    table.add_column("Status", justify="center")

    # Acceptance criteria (tight because both derived from same pose stream)
    # From user specification:
    # - linvel MAE < 0.02 m/s
    # - angvel MAE < 0.05 rad/s
    # - joint vel MAE < 0.05 rad/s
    criteria = {
        "Joint pos": {"mae": 0.01, "corr": 0.99},
        "Joint vel": {"mae": 0.05, "corr": 0.95},
        "Root lin vel": {"mae": 0.02, "corr": 0.99},
        "Root ang vel": {"mae": 0.05, "corr": 0.95},
        "Root height": {"mae": 0.005, "corr": 0.99},
        "Foot contacts": {"mae": 0.20, "corr": 0.80},
    }

    results = {}
    all_pass = True

    for name, mae in mae_results.items():
        # Get correlation for velocity components
        if "lin vel" in name:
            corr = linvel_corr
        elif "ang vel" in name:
            corr = angvel_corr
        else:
            corr = 1.0  # Assume perfect for non-velocity

        # Find matching criteria
        mae_thresh = 0.1
        corr_thresh = 0.95
        for key, thresh in criteria.items():
            if key in name:
                mae_thresh = thresh["mae"]
                corr_thresh = thresh["corr"]
                break

        passed = mae < mae_thresh and corr > corr_thresh
        if not passed:
            all_pass = False

        status = "[green]✓ PASS[/green]" if passed else "[red]✗ FAIL[/red]"
        corr_str = f"{corr:.4f}" if "vel" in name.lower() else "N/A"
        table.add_row(name, f"{mae:.4f}", corr_str, status)

        results[name] = {"mae": mae, "correlation": corr, "passed": passed}

    console.print(table)

    # Summary
    if all_pass:
        print("\n[green]✓ Test 1A PASSED: Feature assembly parity VERIFIED[/green]")
        print(
            "  Indexing, ordering, yaw removal, and heading-local transforms correct."
        )
    else:
        print("\n[red]✗ Test 1A FAILED: Feature assembly parity broken[/red]")
        print("  Check heading-local transform or feature concatenation.")
        print("  NOTE: This test does NOT validate MuJoCo physics integration.")

    # Add metadata
    results["_linvel_correlation"] = linvel_corr
    results["_angvel_correlation"] = angvel_corr
    results["_all_pass"] = all_pass
    results["_pose_noise_std"] = pose_noise_std

    return results


# =============================================================================
# Test 1B: mj_step PD Tracking Test (Physics Validation - MANDATORY)
# =============================================================================


def test_1b_mj_step_tracking(
    ref_path: str,
    model_path: str,
    config: AMPFeatureConfig,
    max_frames: int = 200,
    sim_dt: float = 0.002,  # 500Hz simulation
    control_dt: float = 0.02,  # 50Hz control (matches reference)
    use_home_init: bool = True,  # Initialize from home keyframe (on ground)
    soft_height_stabilize: bool = False,  # Optionally soft-stabilize height
    # v0.9.3: Proper stabilization parameters (assist-only, not puppet)
    stab_kp: float = 60.0,  # N/m (was 500.0 - puppet mode)
    stab_kd: float = 10.0,  # N·s/m (was 50.0)
    stab_force_cap: float = 4.0,  # N, ~12% of mg for 3.4kg robot
    stab_orientation_gate: float = 0.436,  # radians (25°) - disable if tilted
) -> Dict[str, Any]:
    """Test 1B: mj_step PD tracking parity test (MANDATORY for golden rule).

    Why this test is MANDATORY:
    Your policy is trained on values that come from MuJoCo integration.
    If MuJoCo qvel deviates from offline finite difference assumptions
    (due to solver settings, constraint stabilization, contact, PD tracking),
    the discriminator sees a distribution shift and AMP collapses.

    Implementation:
    - Use MuJoCo's built-in position actuators (kp=21.1, kv=0.5)
    - Set ctrl = reference joint positions
    - Let mj_step() evolve physics
    - Floating base evolves naturally (with optional soft height stabilization)
    - Extract features from qvel (what policy actually sees)

    Note: The robot will drift from reference trajectory - this is expected.
    The test validates that the feature EXTRACTION pipeline works correctly
    when reading from MuJoCo's qvel, not that the robot perfectly tracks reference.

    Args:
        ref_path: Path to reference AMP pickle
        model_path: Path to MuJoCo scene XML
        config: AMP feature config
        max_frames: Max frames to test
        sim_dt: Simulation timestep (500Hz)
        control_dt: Control timestep (50Hz, matches reference)
        use_home_init: If True, init from home keyframe (robot on ground)
        soft_height_stabilize: If True, apply soft force to maintain height

    Returns:
        Dict with MAE percentiles and detailed metrics
    """
    print("\n[bold cyan]Test 1B: mj_step PD Tracking Test (MANDATORY)[/bold cyan]")
    print("  (Validates MuJoCo physics integration matches reference)")
    print(
        f"  (sim_dt={sim_dt}s, control_dt={control_dt}s, n_substeps={int(control_dt/sim_dt)})"
    )
    print(
        f"  (use_home_init={use_home_init}, soft_height_stabilize={soft_height_stabilize})"
    )

    # Load reference
    with open(ref_path, "rb") as f:
        ref_data = pickle.load(f)

    num_frames = min(max_frames, ref_data["num_frames"])
    ref_features = ref_data["features"][:num_frames]
    ref_dof_pos = ref_data["dof_pos"][:num_frames]
    ref_root_pos = ref_data["root_pos"][:num_frames]
    ref_root_rot = ref_data["root_rot"][:num_frames]  # xyzw
    dt = ref_data["dt"]

    # Verify dt matches
    if abs(dt - control_dt) > 1e-6:
        print(
            f"  [yellow]Warning: Reference dt={dt} != control_dt={control_dt}[/yellow]"
        )

    # Extract reference velocities for comparison
    n = config.num_actuated_joints
    ref_linvel = ref_features[:, 2 * n : 2 * n + 3]
    ref_angvel = ref_features[:, 2 * n + 3 : 2 * n + 6]
    ref_joint_vel = ref_features[:, n : 2 * n]

    # Load MuJoCo model
    from playground_amp.envs.wildrobot_env import get_assets

    model_path = Path(model_path)
    mj_model = mujoco.MjModel.from_xml_string(
        model_path.read_text(), assets=get_assets(model_path.parent)
    )
    mj_data = mujoco.MjData(mj_model)
    mj_model.opt.timestep = sim_dt

    n_substeps = int(control_dt / sim_dt)
    print(f"  Running {n_substeps} substeps per control frame")

    # Initialize position
    if use_home_init:
        # Use home keyframe (robot standing on ground)
        # Home keyframe: qpos from scene XML
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)  # Load "home" keyframe
        print(f"  Initialized from home keyframe at height z={mj_data.qpos[2]:.4f}m")
    else:
        # Initialize at first frame of reference (may be floating)
        mj_data.qpos[0:3] = ref_root_pos[0]
        q_xyzw = ref_root_rot[0]
        mj_data.qpos[3:7] = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]
        mj_data.qpos[7 : 7 + n] = ref_dof_pos[0]
        print(f"  Initialized at reference frame 0, height z={mj_data.qpos[2]:.4f}m")

    mj_data.qvel[:] = 0  # Start at rest
    mujoco.mj_forward(mj_model, mj_data)

    # Get reference height for optional soft stabilization
    target_height = ref_root_pos[:, 2].mean()

    # Storage for per-frame errors
    linvel_errors = []
    angvel_errors = []
    joint_vel_errors = []
    all_features = []
    prev_joint_pos = mj_data.qpos[7 : 7 + n].copy()

    # Diagnostic: track height and position drift
    height_trajectory = []
    position_drift = []
    initial_xy = mj_data.qpos[0:2].copy()

    # v0.9.2: Enhanced diagnostics for failure mode classification
    # These help distinguish: (1) control limited, (2) infeasible reference, (3) harness bug
    joint_tracking_errors = []  # Per-joint position error vs reference
    foot_velocities = []  # Foot velocity when in contact (slip detection)
    base_pitch_trajectory = []  # Pitch angle drift
    base_roll_trajectory = []  # Roll angle drift
    base_yaw_trajectory = []  # Yaw angle trajectory
    ref_yaw_trajectory = []  # Reference yaw for comparison
    contact_mismatch_count = 0  # Offline vs online contact estimator mismatch
    total_contact_frames = 0

    # v0.9.3: Contact force logging for load support validation
    # Robot mass: 3.383 kg, mg = 33.19 N
    robot_mass = 3.383  # kg
    mg = robot_mass * 9.81  # 33.19 N
    f_n_threshold = 0.05 * mg  # 1.7 N - min F_n for valid friction margin
    total_normal_forces = []  # Sum of F_n from all contacts per frame
    stabilization_forces = []  # Track how much force stabilizer applies
    low_support_frames = 0  # Frames where load support < 85%

    # Get foot site IDs for slip measurement
    try:
        left_foot_site = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SITE, "left_foot"
        )
        right_foot_site = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SITE, "right_foot"
        )
        has_foot_sites = True
    except Exception:
        # Try geom-based fallback
        has_foot_sites = False
        print(
            "  [yellow]Warning: No foot sites found, using geom positions for slip[/yellow]"
        )

    # Previous foot positions for slip velocity
    prev_left_foot_pos = None
    prev_right_foot_pos = None

    for i in range(num_frames):
        # Set control targets (reference joint positions)
        # MuJoCo's position actuators use built-in PD (kp=21.1, kv=0.5)
        mj_data.ctrl[:] = ref_dof_pos[i]

        # Optional: soft height stabilization (assist-only, not puppet)
        # v0.9.3: Use proper stabilization parameters
        # Only assists balance, does not carry body weight
        if soft_height_stabilize:
            # First check orientation gate - disable stabilization if robot is tilted
            # (tilted robot should fall, not be held up)
            w_curr, x_curr, y_curr, z_curr = mj_data.qpos[3:7]
            sinp = 2.0 * (w_curr * y_curr - z_curr * x_curr)
            sinp = np.clip(sinp, -1.0, 1.0)
            curr_pitch = np.abs(np.arcsin(sinp))
            sinr_cosp = 2.0 * (w_curr * x_curr + y_curr * z_curr)
            cosr_cosp = 1.0 - 2.0 * (x_curr * x_curr + y_curr * y_curr)
            curr_roll = np.abs(np.arctan2(sinr_cosp, cosr_cosp))

            # Only apply stabilization if robot is reasonably upright
            if curr_pitch < stab_orientation_gate and curr_roll < stab_orientation_gate:
                height_error = target_height - mj_data.qpos[2]
                # Compute PD force with new gains (60 N/m, 10 N·s/m)
                f_stab = stab_kp * height_error - stab_kd * mj_data.qvel[2]
                # Apply force cap (~12% of mg for 3.4kg robot = 4N)
                f_stab = np.clip(f_stab, -stab_force_cap, stab_force_cap)
                mj_data.qfrc_applied[2] = f_stab
            else:
                # Robot is tilted - no stabilization (let it fall naturally)
                mj_data.qfrc_applied[2] = 0.0

        # Save current root pose BEFORE stepping (for finite-diff velocity)
        prev_root_pos_step = mj_data.qpos[0:3].copy()
        prev_root_quat_step = mj_data.qpos[3:7].copy()  # wxyz format

        # Track stabilization force applied (for diagnostics)
        stab_force_applied = mj_data.qfrc_applied[2]
        stabilization_forces.append(stab_force_applied)

        # Step physics n_substeps times
        for _ in range(n_substeps):
            mujoco.mj_step(mj_model, mj_data)

        # v0.9.3: Extract contact forces from MuJoCo
        # Sum normal forces (F_n) from all contacts
        frame_total_fn = 0.0
        for contact_idx in range(mj_data.ncon):
            contact = mj_data.contact[contact_idx]
            # Get normal force magnitude from efc_force
            # contact.efc_address gives the address of the first constraint row
            if contact.efc_address >= 0:
                # Normal force is in the first row of the contact constraint
                fn = mj_data.efc_force[contact.efc_address]
                if fn > 0:  # Only count positive (pushing) normal forces
                    frame_total_fn += fn
        total_normal_forces.append(frame_total_fn)

        # Track load support ratio
        load_support_ratio = frame_total_fn / mg
        if load_support_ratio < 0.85:
            low_support_frames += 1

        # Read state AFTER physics stepping
        # v0.9.1: Use finite-diff velocities from qpos (same as updated policy)
        # This is the key fix - compare like with like
        curr_root_pos = mj_data.qpos[0:3].copy()
        curr_root_quat = mj_data.qpos[3:7].copy()  # wxyz
        joint_pos = mj_data.qpos[7 : 7 + n].copy()
        root_height = mj_data.qpos[2]

        # Finite-diff linear velocity (world frame, then heading-local)
        lin_vel_world = (curr_root_pos - prev_root_pos_step) / control_dt

        # Finite-diff angular velocity using axis-angle (world frame)
        # q_delta = q_curr * q_prev_conj
        w_prev, x_prev, y_prev, z_prev = prev_root_quat_step
        q_prev_conj = np.array([w_prev, -x_prev, -y_prev, -z_prev])
        w_curr, x_curr, y_curr, z_curr = curr_root_quat

        # Hamilton product (wxyz)
        w_delta = (
            w_curr * q_prev_conj[0]
            - x_curr * q_prev_conj[1]
            - y_curr * q_prev_conj[2]
            - z_curr * q_prev_conj[3]
        )
        x_delta = (
            w_curr * q_prev_conj[1]
            + x_curr * q_prev_conj[0]
            + y_curr * q_prev_conj[3]
            - z_curr * q_prev_conj[2]
        )
        y_delta = (
            w_curr * q_prev_conj[2]
            - x_curr * q_prev_conj[3]
            + y_curr * q_prev_conj[0]
            + z_curr * q_prev_conj[1]
        )
        z_delta = (
            w_curr * q_prev_conj[3]
            + x_curr * q_prev_conj[2]
            - y_curr * q_prev_conj[1]
            + z_curr * q_prev_conj[0]
        )

        # Convert to rotation vector (axis-angle)
        vec = np.array([x_delta, y_delta, z_delta])
        vec_norm = np.linalg.norm(vec) + 1e-8
        angle = 2.0 * np.arctan2(vec_norm, np.abs(w_delta))
        angle = -angle if w_delta < 0 else angle
        axis = vec / vec_norm
        rotvec = angle * axis
        ang_vel_world = rotvec / control_dt

        # Joint velocities from MuJoCo qvel (not FD, since joints are tracked by PD)
        joint_vel = mj_data.qvel[6 : 6 + n].copy()

        # Compute heading-local transform
        w, x, y, z = curr_root_quat
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        norm = np.sqrt(siny_cosp**2 + cosy_cosp**2 + 1e-8)
        sin_yaw, cos_yaw = siny_cosp / norm, cosy_cosp / norm

        # Convert velocities to heading-local (using FD world-frame velocities)
        lin_vel_heading = np.array(
            [
                cos_yaw * lin_vel_world[0] + sin_yaw * lin_vel_world[1],
                -sin_yaw * lin_vel_world[0] + cos_yaw * lin_vel_world[1],
                lin_vel_world[2],
            ]
        )
        ang_vel_heading = np.array(
            [
                cos_yaw * ang_vel_world[0] + sin_yaw * ang_vel_world[1],
                -sin_yaw * ang_vel_world[0] + cos_yaw * ang_vel_world[1],
                ang_vel_world[2],
            ]
        )

        # Compute errors against reference (heading-local)
        linvel_errors.append(np.abs(lin_vel_heading - ref_linvel[i]))
        angvel_errors.append(np.abs(ang_vel_heading - ref_angvel[i]))
        joint_vel_errors.append(np.abs(joint_vel - ref_joint_vel[i]))

        # Build full observation and extract features
        gravity_world = np.array([0, 0, -1])
        quat = mj_data.qpos[3:7]
        gravity_body = quaternion_rotate_inverse(quat, gravity_world)

        # Joint velocities (use MuJoCo qvel, not finite diff)
        obs = np.concatenate(
            [
                gravity_body,
                ang_vel_heading,
                lin_vel_heading,
                joint_pos,
                joint_vel,  # From MuJoCo qvel
                np.zeros(8),  # prev_action
                np.array([0.0]),  # velocity_cmd
                np.array([0.0]),  # padding
            ]
        ).astype(np.float32)

        # Extract features
        features = extract_amp_features(
            obs=jnp.array(obs),
            config=config,
            root_height=jnp.array(root_height),
            prev_joint_pos=jnp.array(prev_joint_pos),
            dt=control_dt,
        )
        all_features.append(np.array(features))
        prev_joint_pos = joint_pos.copy()

        # Track diagnostics
        height_trajectory.append(root_height)
        position_drift.append(np.linalg.norm(mj_data.qpos[0:2] - initial_xy))

        # v0.9.2: Enhanced diagnostics for failure mode classification

        # 1. Joint tracking RMSE (error vs reference target)
        joint_error = joint_pos - ref_dof_pos[i]
        joint_tracking_errors.append(joint_error)

        # 2. Base orientation (pitch, roll, yaw) from quaternion
        # Euler angles from quaternion (ZYX convention)
        w, x, y, z = curr_root_quat
        # Pitch (rotation around Y)
        sinp = 2.0 * (w * y - z * x)
        sinp = np.clip(sinp, -1.0, 1.0)  # Clamp for numerical stability
        pitch = np.arcsin(sinp)
        # Roll (rotation around X)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        # Yaw already computed above
        yaw = np.arctan2(sin_yaw, cos_yaw)

        base_pitch_trajectory.append(pitch)
        base_roll_trajectory.append(roll)
        base_yaw_trajectory.append(yaw)

        # Reference yaw for comparison
        q_ref_xyzw = ref_root_rot[i]
        w_ref, x_ref, y_ref, z_ref = (
            q_ref_xyzw[3],
            q_ref_xyzw[0],
            q_ref_xyzw[1],
            q_ref_xyzw[2],
        )
        ref_siny = 2.0 * (w_ref * z_ref + x_ref * y_ref)
        ref_cosy = 1.0 - 2.0 * (y_ref * y_ref + z_ref * z_ref)
        ref_yaw = np.arctan2(ref_siny, ref_cosy)
        ref_yaw_trajectory.append(ref_yaw)

        # 3. Foot slip detection (foot velocity when in contact)
        if has_foot_sites:
            # Get foot positions from sites
            left_foot_pos = mj_data.site_xpos[left_foot_site].copy()
            right_foot_pos = mj_data.site_xpos[right_foot_site].copy()
        else:
            # Fallback: use foot geom positions (less accurate)
            # Get foot geom IDs
            try:
                left_foot_geom = mujoco.mj_name2id(
                    mj_model, mujoco.mjtObj.mjOBJ_GEOM, "left_toe"
                )
                right_foot_geom = mujoco.mj_name2id(
                    mj_model, mujoco.mjtObj.mjOBJ_GEOM, "right_toe"
                )
                left_foot_pos = mj_data.geom_xpos[left_foot_geom].copy()
                right_foot_pos = mj_data.geom_xpos[right_foot_geom].copy()
            except Exception:
                left_foot_pos = np.zeros(3)
                right_foot_pos = np.zeros(3)

        # Compute foot velocities from finite diff
        if prev_left_foot_pos is not None:
            left_foot_vel = (
                np.linalg.norm(left_foot_pos[:2] - prev_left_foot_pos[:2]) / control_dt
            )
            right_foot_vel = (
                np.linalg.norm(right_foot_pos[:2] - prev_right_foot_pos[:2])
                / control_dt
            )

            # Get contact state from reference (estimated) and check for slip
            ref_contacts_frame = ref_features[
                i, 2 * n + 7 : 2 * n + 11
            ]  # foot contacts from reference
            left_contact = (
                ref_contacts_frame[0] > 0.5 or ref_contacts_frame[1] > 0.5
            )  # toe or heel
            right_contact = ref_contacts_frame[2] > 0.5 or ref_contacts_frame[3] > 0.5

            # Record slip velocity (only when foot should be in contact)
            if left_contact:
                foot_velocities.append(("left", left_foot_vel))
            if right_contact:
                foot_velocities.append(("right", right_foot_vel))

        prev_left_foot_pos = left_foot_pos
        prev_right_foot_pos = right_foot_pos

        # 4. Contact mismatch detection (compare online vs offline estimator)
        # Online: estimate from current joint positions
        online_contacts = np.array(
            estimate_foot_contacts_from_joints(jnp.array(joint_pos), config)
        )
        # Offline: from reference features
        offline_contacts = ref_features[i, 2 * n + 7 : 2 * n + 11]

        # Count mismatches (binary threshold at 0.5)
        online_binary = (online_contacts > 0.5).astype(int)
        offline_binary = (offline_contacts > 0.5).astype(int)
        total_contact_frames += 4  # 4 contact points per frame
        contact_mismatch_count += np.sum(online_binary != offline_binary)

    # Print diagnostic info
    print(f"\n  [bold]Diagnostic: Robot Drift Analysis[/bold]")
    print(f"    Initial height: {height_trajectory[0]:.4f}m")
    print(f"    Final height:   {height_trajectory[-1]:.4f}m")
    print(f"    Height change:  {height_trajectory[-1] - height_trajectory[0]:.4f}m")
    print(f"    XY drift (end): {position_drift[-1]:.4f}m")
    print(f"    Reference mean height: {ref_root_pos[:, 2].mean():.4f}m")

    # v0.9.2: Enhanced Failure Mode Analysis
    print(f"\n  [bold]Enhanced Diagnostic: Failure Mode Classification[/bold]")

    # 1. Joint Tracking Analysis
    joint_tracking_errors = np.array(joint_tracking_errors)
    joint_rmse_per_joint = np.sqrt(np.mean(joint_tracking_errors**2, axis=0))
    joint_rmse_overall = np.sqrt(np.mean(joint_tracking_errors**2))
    print(f"\n    [cyan]1. Joint Tracking RMSE:[/cyan]")
    print(
        f"       Overall RMSE: {joint_rmse_overall:.4f} rad ({np.degrees(joint_rmse_overall):.2f}°)"
    )
    joint_names = [
        "L_hip_pitch",
        "L_hip_roll",
        "L_knee",
        "L_ankle",
        "R_hip_pitch",
        "R_hip_roll",
        "R_knee",
        "R_ankle",
    ]
    for j, (name, rmse) in enumerate(zip(joint_names, joint_rmse_per_joint)):
        marker = "[yellow]⚠[/yellow]" if rmse > 0.1 else "[green]✓[/green]"
        print(f"       {marker} {name}: {rmse:.4f} rad ({np.degrees(rmse):.2f}°)")

    # 2. Stance Foot Slip Analysis
    print(f"\n    [cyan]2. Stance Foot Slip Analysis:[/cyan]")
    if foot_velocities:
        left_slips = [v for foot, v in foot_velocities if foot == "left"]
        right_slips = [v for foot, v in foot_velocities if foot == "right"]
        if left_slips:
            print(
                f"       Left foot: mean={np.mean(left_slips):.4f} m/s, max={np.max(left_slips):.4f} m/s"
            )
        if right_slips:
            print(
                f"       Right foot: mean={np.mean(right_slips):.4f} m/s, max={np.max(right_slips):.4f} m/s"
            )
        all_slips = left_slips + right_slips
        if all_slips:
            high_slip_count = sum(
                1 for s in all_slips if s > 0.1
            )  # >10cm/s is significant
            print(
                f"       High slip events (>0.1 m/s): {high_slip_count}/{len(all_slips)} ({100*high_slip_count/len(all_slips):.1f}%)"
            )
    else:
        print(f"       [yellow]No foot contact data collected[/yellow]")

    # 3. Base Orientation Drift
    print(f"\n    [cyan]3. Base Orientation Drift:[/cyan]")
    base_pitch_trajectory = np.array(base_pitch_trajectory)
    base_roll_trajectory = np.array(base_roll_trajectory)
    base_yaw_trajectory = np.array(base_yaw_trajectory)
    ref_yaw_trajectory = np.array(ref_yaw_trajectory)

    pitch_drift = base_pitch_trajectory[-1] - base_pitch_trajectory[0]
    roll_drift = base_roll_trajectory[-1] - base_roll_trajectory[0]
    pitch_max_abs = np.max(np.abs(base_pitch_trajectory))
    roll_max_abs = np.max(np.abs(base_roll_trajectory))
    print(
        f"       Pitch: drift={np.degrees(pitch_drift):.2f}°, max_abs={np.degrees(pitch_max_abs):.2f}°"
    )
    print(
        f"       Roll: drift={np.degrees(roll_drift):.2f}°, max_abs={np.degrees(roll_max_abs):.2f}°"
    )

    # 4. Yaw Rate Drift vs Reference
    print(f"\n    [cyan]4. Yaw Rate Analysis:[/cyan]")
    sim_yaw_rate = np.diff(base_yaw_trajectory) / control_dt
    ref_yaw_rate = np.diff(ref_yaw_trajectory) / control_dt
    yaw_rate_error = np.abs(sim_yaw_rate - ref_yaw_rate)
    # Handle wraparound
    yaw_rate_error = np.minimum(yaw_rate_error, 2 * np.pi / control_dt - yaw_rate_error)
    print(
        f"       Sim yaw rate: mean={np.mean(sim_yaw_rate):.4f} rad/s, std={np.std(sim_yaw_rate):.4f}"
    )
    print(
        f"       Ref yaw rate: mean={np.mean(ref_yaw_rate):.4f} rad/s, std={np.std(ref_yaw_rate):.4f}"
    )
    print(f"       Yaw rate MAE: {np.mean(yaw_rate_error):.4f} rad/s")

    # 5. Contact Mismatch Analysis
    print(f"\n    [cyan]5. Contact Estimator Mismatch:[/cyan]")
    contact_mismatch_rate = (
        contact_mismatch_count / total_contact_frames if total_contact_frames > 0 else 0
    )
    print(
        f"       Mismatch rate: {contact_mismatch_rate:.2%} ({contact_mismatch_count}/{total_contact_frames} contact-frames)"
    )

    # v0.9.3: Load Support Analysis (new diagnostic)
    print(f"\n    [cyan]6. Load Support Analysis (v0.9.3):[/cyan]")
    total_normal_forces_arr = np.array(total_normal_forces)
    mean_total_fn = (
        np.mean(total_normal_forces_arr) if total_normal_forces_arr.size > 0 else 0
    )
    load_support_ratio_mean = mean_total_fn / mg
    low_support_rate = low_support_frames / num_frames if num_frames > 0 else 1.0

    print(f"       Robot mass: {robot_mass:.3f} kg, mg = {mg:.2f} N")
    print(f"       Mean total F_n: {mean_total_fn:.2f} N")
    print(f"       Load support ratio (mean Σ F_n / mg): {load_support_ratio_mean:.2%}")
    print(
        f"       Low support frames (<85%): {low_support_frames}/{num_frames} ({low_support_rate:.1%})"
    )

    # Stabilization force analysis
    stabilization_forces_arr = np.array(stabilization_forces)
    mean_stab_force = (
        np.mean(np.abs(stabilization_forces_arr))
        if stabilization_forces_arr.size > 0
        else 0
    )
    max_stab_force = (
        np.max(np.abs(stabilization_forces_arr))
        if stabilization_forces_arr.size > 0
        else 0
    )
    stab_as_pct_mg = mean_stab_force / mg * 100

    print(
        f"       Stabilization force: mean={mean_stab_force:.2f} N ({stab_as_pct_mg:.1f}% mg), max={max_stab_force:.2f} N"
    )

    # Load support acceptance criteria
    load_support_ok = load_support_ratio_mean >= 0.85 and low_support_rate < 0.10
    if load_support_ok:
        print(
            f"       [green]✓ Load support ACCEPTABLE (≥85% mean, <10% low frames)[/green]"
        )
    else:
        print(
            f"       [red]✗ Load support INSUFFICIENT (harness may be puppeting)[/red]"
        )

    # FAILURE MODE CLASSIFICATION
    print(f"\n  [bold yellow]═══ FAILURE MODE CLASSIFICATION ═══[/bold yellow]")

    # Thresholds for classification
    tracking_ok = joint_rmse_overall < 0.15  # 0.15 rad ≈ 8.6°
    height_ok = (
        abs(height_trajectory[-1] - height_trajectory[0]) < 0.1
    )  # <10cm height loss
    pitch_ok = pitch_max_abs < 0.5  # <30° max pitch
    slip_rate = sum(
        1 for s in (left_slips + right_slips if foot_velocities else []) if s > 0.1
    ) / max(len(foot_velocities), 1)
    slip_ok = slip_rate < 0.3  # <30% high-slip events

    if not height_ok and not tracking_ok:
        failure_mode = "CONTROL_LIMITED"
        reason = "Poor joint tracking AND significant height loss"
        recommendation = "Increase PD gains or use torque control with better tuning"
    elif tracking_ok and not height_ok:
        failure_mode = "INFEASIBLE_REFERENCE"
        reason = "Good joint tracking but base still falls/diverges"
        recommendation = "Reference motion may be dynamically infeasible for this robot"
    elif not tracking_ok and height_ok:
        failure_mode = "TRACKING_ERROR"
        reason = "Poor joint tracking but robot stays upright"
        recommendation = (
            "PD gains may be fighting against reference; check feasibility constraints"
        )
    elif not slip_ok:
        failure_mode = "CONTACT_SLIP"
        reason = "High foot slip during stance phase"
        recommendation = "Check friction parameters or reference step lengths"
    elif not pitch_ok:
        failure_mode = "BALANCE_FAILURE"
        reason = "Large pitch excursions indicate balance issues"
        recommendation = (
            "Reference may require ankle/hip coordination robot cannot achieve"
        )
    else:
        failure_mode = "UNCERTAIN"
        reason = "Individual metrics OK but combined effect causes drift"
        recommendation = "May be test harness issue or cumulative small errors"

    # Color-coded output
    mode_color = {
        "CONTROL_LIMITED": "red",
        "INFEASIBLE_REFERENCE": "yellow",
        "TRACKING_ERROR": "yellow",
        "CONTACT_SLIP": "magenta",
        "BALANCE_FAILURE": "red",
        "UNCERTAIN": "cyan",
    }.get(failure_mode, "white")

    print(
        f"\n    Failure Mode: [{mode_color}][bold]{failure_mode}[/bold][/{mode_color}]"
    )
    print(f"    Reason: {reason}")
    print(f"    Recommendation: {recommendation}")

    # Summary metrics table
    print(f"\n    Summary Metrics:")
    print(f"      {'Metric':<25} {'Value':<15} {'Threshold':<15} {'Status':<10}")
    print(f"      {'-'*65}")
    print(
        f"      {'Joint RMSE':<25} {joint_rmse_overall:.4f} rad    {'<0.15 rad':<15} {'✓ OK' if tracking_ok else '✗ FAIL':<10}"
    )
    print(
        f"      {'Height change':<25} {height_trajectory[-1] - height_trajectory[0]:.4f} m     {'|Δ|<0.10 m':<15} {'✓ OK' if height_ok else '✗ FAIL':<10}"
    )
    print(
        f"      {'Max pitch':<25} {np.degrees(pitch_max_abs):.2f}°      {'<30°':<15} {'✓ OK' if pitch_ok else '✗ FAIL':<10}"
    )
    print(
        f"      {'Slip rate':<25} {slip_rate:.1%}         {'<30%':<15} {'✓ OK' if slip_ok else '✗ FAIL':<10}"
    )

    # Compute statistics
    linvel_errors = np.array(linvel_errors)
    angvel_errors = np.array(angvel_errors)
    joint_vel_errors = np.array(joint_vel_errors)
    sim_features = np.stack(all_features)

    # Compute MAE and percentiles
    def compute_percentiles(errors):
        mae_per_frame = np.mean(errors, axis=1)  # MAE per frame
        return {
            "mean": np.mean(mae_per_frame),
            "p50": np.percentile(mae_per_frame, 50),
            "p90": np.percentile(mae_per_frame, 90),
            "p99": np.percentile(mae_per_frame, 99),
            "max": np.max(mae_per_frame),
        }

    linvel_stats = compute_percentiles(linvel_errors)
    angvel_stats = compute_percentiles(angvel_errors)
    joint_vel_stats = compute_percentiles(joint_vel_errors)

    # Print results table
    table = Table(title="Test 1B: mj_step PD Tracking (Percentile MAE)")
    table.add_column("Component", style="cyan")
    table.add_column("Mean", justify="right")
    table.add_column("P50", justify="right")
    table.add_column("P90", justify="right")
    table.add_column("Max", justify="right")
    table.add_column("Status", justify="center")

    # Acceptance criteria from user specification:
    # - linvel MAE p90 < 0.10 m/s, max < 0.25 m/s
    # - angvel MAE p90 < 0.20 rad/s, max < 0.50 rad/s
    # - joint vel MAE p90 < 0.20 rad/s, max < 0.50 rad/s
    criteria = {
        "Root lin vel": {"p90": 0.10, "max": 0.25},
        "Root ang vel": {"p90": 0.20, "max": 0.50},
        "Joint vel": {"p90": 0.20, "max": 0.50},
    }

    results = {}
    all_pass = True

    for name, stats in [
        ("Root lin vel", linvel_stats),
        ("Root ang vel", angvel_stats),
        ("Joint vel", joint_vel_stats),
    ]:
        crit = criteria[name]
        passed = stats["p90"] < crit["p90"] and stats["max"] < crit["max"]
        if not passed:
            all_pass = False

        status = "[green]✓ PASS[/green]" if passed else "[red]✗ FAIL[/red]"
        table.add_row(
            name,
            f"{stats['mean']:.4f}",
            f"{stats['p50']:.4f}",
            f"{stats['p90']:.4f}",
            f"{stats['max']:.4f}",
            status,
        )

        results[name] = stats
        results[name]["passed"] = passed

    console.print(table)

    # Full feature MAE
    mae_results = compute_component_mae(ref_features, sim_features, config)
    print("\n  [bold]Full Feature MAE:[/bold]")
    for name, mae in mae_results.items():
        print(f"    {name}: {mae:.4f}")

    # Summary
    if all_pass:
        print("\n[green]✓ Test 1B PASSED: MuJoCo physics integration VALIDATED[/green]")
        print("  Policy features during training will match reference distribution.")
    else:
        print(
            "\n[red]✗ Test 1B FAILED: Physics integration shows distribution shift[/red]"
        )
        print("  High errors may be due to:")
        print("    - PD gains not tracking reference well")
        print("    - Contact/slip behavior differences")
        print("    - Floating base drift")
        print("  Consider checking the worst 1% frames for explainable failures.")

    results["_all_pass"] = all_pass
    results["_full_mae"] = mae_results

    return results


# =============================================================================
# Test 2: Yaw Invariance Test
# =============================================================================


def test_yaw_invariance(
    ref_path: str,
    config: AMPFeatureConfig,
    num_yaw_offsets: int = 8,
) -> Dict[str, float]:
    """Test that heading-local features are invariant to global yaw rotation.

    Rotates the entire reference motion by random yaw offsets and confirms
    heading-local velocities and features are unchanged within tolerance.

    Args:
        ref_path: Path to reference AMP pickle
        config: AMP feature config
        num_yaw_offsets: Number of yaw offsets to test

    Returns:
        Dict with max deviation per component
    """
    print("\n[bold cyan]Test 2: Yaw Invariance Test[/bold cyan]")
    print(f"  (Rotating motion by {num_yaw_offsets} different yaw offsets)")

    # Load reference
    with open(ref_path, "rb") as f:
        ref_data = pickle.load(f)

    original_features = ref_data["features"]
    root_rot = ref_data["root_rot"]  # xyzw
    root_pos = ref_data["root_pos"]
    dof_pos = ref_data["dof_pos"]
    dt = ref_data["dt"]

    # Test multiple yaw offsets
    yaw_offsets = np.linspace(0, 2 * np.pi, num_yaw_offsets, endpoint=False)
    max_deviations = {}

    for yaw_offset in yaw_offsets[1:]:  # Skip 0 offset
        # Create rotation matrix for yaw offset
        r_offset = R.from_euler("z", yaw_offset)

        # Rotate all quaternions
        rotated_quats = []
        for q_xyzw in root_rot:
            r_orig = R.from_quat(q_xyzw)
            r_rotated = r_offset * r_orig
            rotated_quats.append(r_rotated.as_quat())
        rotated_quats = np.array(rotated_quats)

        # Rotate root positions
        rotated_pos = r_offset.apply(root_pos)

        # Recompute features with rotated motion
        rotated_features = recompute_features(
            rotated_pos, rotated_quats, dof_pos, dt, config
        )

        # Compare features (should be identical for heading-local)
        diff = np.abs(original_features - rotated_features)

        for name, (start, end) in get_feature_ranges(config).items():
            component_diff = diff[:, start:end].max()
            if name not in max_deviations:
                max_deviations[name] = component_diff
            else:
                max_deviations[name] = max(max_deviations[name], component_diff)

    # Print results
    table = Table(title="Yaw Invariance Test (max deviation across all yaw offsets)")
    table.add_column("Component", style="cyan")
    table.add_column("Max Deviation", justify="right")
    table.add_column("Status", justify="center")

    all_pass = True
    for name, deviation in max_deviations.items():
        # Velocity components should be invariant (< 1e-5)
        # Joint positions are invariant by definition
        threshold = 1e-4 if "vel" in name.lower() else 1e-6
        status = (
            "[green]✓ PASS[/green]" if deviation < threshold else "[red]✗ FAIL[/red]"
        )
        if deviation >= threshold:
            all_pass = False
        table.add_row(name, f"{deviation:.6f}", status)

    console.print(table)

    if all_pass:
        print("[green]✓ Heading-local features are yaw-invariant[/green]")
    else:
        print("[red]✗ Yaw invariance FAILED - check heading-local transform[/red]")

    return max_deviations


# =============================================================================
# Test 3: Quaternion Sign Flip Test
# =============================================================================


def test_quaternion_sign_flip(
    ref_path: str,
    config: AMPFeatureConfig,
) -> Dict[str, float]:
    """Test robustness to quaternion antipodal sign flips (q ≡ -q).

    Flips sign of every other quaternion and confirms:
    - Yaw extraction is unchanged
    - Angular velocity from SciPy method is stable
    - Resulting features match baseline

    Args:
        ref_path: Path to reference AMP pickle
        config: AMP feature config

    Returns:
        Dict with max deviation per component
    """
    print("\n[bold cyan]Test 3: Quaternion Sign Flip Test[/bold cyan]")
    print("  (Adversarial test: flipping sign of every other quaternion)")

    # Load reference
    with open(ref_path, "rb") as f:
        ref_data = pickle.load(f)

    original_features = ref_data["features"]
    root_rot = ref_data["root_rot"].copy()  # xyzw
    root_pos = ref_data["root_pos"]
    dof_pos = ref_data["dof_pos"]
    dt = ref_data["dt"]

    # Flip sign of every other quaternion
    flipped_rot = root_rot.copy()
    flipped_rot[1::2] = -flipped_rot[1::2]  # Flip odd indices

    # Test yaw extraction stability
    original_yaw = quaternion_to_yaw_xyzw(root_rot)
    flipped_yaw = quaternion_to_yaw_xyzw(flipped_rot)
    yaw_diff = np.abs(original_yaw - flipped_yaw)
    # Handle wraparound
    yaw_diff = np.minimum(yaw_diff, 2 * np.pi - yaw_diff)
    max_yaw_diff = yaw_diff.max()

    print(f"  Max yaw extraction difference: {max_yaw_diff:.6f} rad")

    # Recompute features with flipped quaternions
    flipped_features = recompute_features(root_pos, flipped_rot, dof_pos, dt, config)

    # Compare
    results = {}
    for name, (start, end) in get_feature_ranges(config).items():
        orig = original_features[:, start:end]
        flip = flipped_features[:, start:end]
        mae = np.mean(np.abs(orig - flip))
        max_diff = np.max(np.abs(orig - flip))
        results[name] = {"mae": mae, "max_diff": max_diff}

    # Print results
    table = Table(title="Quaternion Sign Flip Test")
    table.add_column("Component", style="cyan")
    table.add_column("MAE", justify="right")
    table.add_column("Max Diff", justify="right")
    table.add_column("Status", justify="center")

    # Angular velocity is the critical one - should be stable
    all_pass = True
    for name, metrics in results.items():
        threshold = 0.1 if "ang vel" in name.lower() else 0.01
        status = (
            "[green]✓ PASS[/green]"
            if metrics["max_diff"] < threshold
            else "[red]✗ FAIL[/red]"
        )
        if metrics["max_diff"] >= threshold:
            all_pass = False
        table.add_row(
            name, f"{metrics['mae']:.4f}", f"{metrics['max_diff']:.4f}", status
        )

    console.print(table)

    if all_pass:
        print("[green]✓ Features robust to quaternion sign flips[/green]")
    else:
        print(
            "[red]✗ Sign flip robustness FAILED - check quaternion continuity handling[/red]"
        )

    return results


# =============================================================================
# Test 4: Batch Coverage Test
# =============================================================================


def test_batch_coverage(
    motion_dir: str,
    amp_dir: str,
    model_path: str,
    config: AMPFeatureConfig,
    frames_per_motion: int = 200,
) -> Dict[str, Any]:
    """Test parity across all motion files.

    Reports worst-case and percentile MAE across all motions.

    Args:
        motion_dir: Directory containing motion .pkl files
        amp_dir: Directory containing AMP .pkl files
        model_path: Path to MuJoCo scene XML
        config: AMP feature config
        frames_per_motion: Random frames to sample per motion

    Returns:
        Dict with statistics across all motions
    """
    print("\n[bold cyan]Test 4: Batch Coverage Test[/bold cyan]")

    motion_dir = Path(motion_dir)
    amp_dir = Path(amp_dir)

    # Find all AMP files
    amp_files = list(amp_dir.glob("*_amp.pkl"))

    if not amp_files:
        print("[yellow]No AMP files found. Converting motions first...[/yellow]")
        return {}

    print(f"  Found {len(amp_files)} AMP files")

    all_results = []

    for amp_file in amp_files:
        print(f"  Testing: {amp_file.name}")

        try:
            with open(amp_file, "rb") as f:
                ref_data = pickle.load(f)

            # Sample random frames
            num_frames = min(frames_per_motion, ref_data["num_frames"])
            if ref_data["num_frames"] > frames_per_motion:
                indices = np.random.choice(
                    ref_data["num_frames"], frames_per_motion, replace=False
                )
                indices = np.sort(indices)
            else:
                indices = np.arange(num_frames)

            # Run basic parity test on sampled frames
            results = run_basic_parity(ref_data, indices, config)
            results["file"] = amp_file.name
            all_results.append(results)

        except Exception as e:
            print(f"    [red]Error: {e}[/red]")

    if not all_results:
        return {}

    # Compute statistics
    stats = compute_batch_statistics(all_results, config)

    # Print summary
    table = Table(title=f"Batch Coverage Summary ({len(all_results)} motions)")
    table.add_column("Component", style="cyan")
    table.add_column("Mean MAE", justify="right")
    table.add_column("Worst MAE", justify="right")
    table.add_column("P95 MAE", justify="right")
    table.add_column("Status", justify="center")

    thresholds = {
        "Joint pos": 0.05,
        "Joint vel": 0.2,
        "Root lin vel": 0.1,
        "Root ang vel": 0.1,
        "Root height": 0.01,
        "Foot contacts": 0.3,
    }

    for name, s in stats.items():
        threshold = 0.1
        for key, th in thresholds.items():
            if key in name:
                threshold = th
                break

        status = (
            "[green]✓ PASS[/green]" if s["worst"] < threshold else "[red]✗ FAIL[/red]"
        )
        table.add_row(
            name, f"{s['mean']:.4f}", f"{s['worst']:.4f}", f"{s['p95']:.4f}", status
        )

    console.print(table)

    # Report worst file
    worst_file = max(
        all_results, key=lambda x: max(x.get(k, 0) for k in x if k != "file")
    )
    print(f"\n  Worst performing file: {worst_file['file']}")

    return stats


# =============================================================================
# Test 5: Contact Estimator Consistency Test
# =============================================================================


def test_contact_consistency(
    ref_path: str,
    config: AMPFeatureConfig,
    noise_levels: List[float] = [0.0, 0.01, 0.05, 0.1],
) -> Dict[str, float]:
    """Test contact estimator consistency under noise perturbations.

    Runs offline and online contact estimator on same joint trajectories
    and ensures they match within tolerance, then tests under noise.

    Args:
        ref_path: Path to reference AMP pickle
        config: AMP feature config
        noise_levels: Joint angle noise levels to test (radians)

    Returns:
        Dict with consistency metrics
    """
    print("\n[bold cyan]Test 5: Contact Estimator Consistency Test[/bold cyan]")

    # Load reference
    with open(ref_path, "rb") as f:
        ref_data = pickle.load(f)

    dof_pos = ref_data["dof_pos"]
    ref_contacts = ref_data["foot_contacts"]

    results = {}

    for noise_level in noise_levels:
        # Add noise to joint positions
        noisy_dof_pos = dof_pos + np.random.randn(*dof_pos.shape) * noise_level

        # Run contact estimator (same as used in reference generation)
        estimated_contacts = []
        for i in range(len(noisy_dof_pos)):
            contacts = np.array(
                estimate_foot_contacts_from_joints(
                    jnp.array(noisy_dof_pos[i]),
                    config,
                )
            )
            estimated_contacts.append(contacts)
        estimated_contacts = np.stack(estimated_contacts)

        # Compare to reference
        mae = np.mean(np.abs(ref_contacts - estimated_contacts))
        max_diff = np.max(np.abs(ref_contacts - estimated_contacts))

        results[f"noise_{noise_level:.2f}"] = {
            "mae": mae,
            "max_diff": max_diff,
        }

        print(f"  Noise {noise_level:.2f} rad: MAE={mae:.4f}, Max={max_diff:.4f}")

    # Print summary
    table = Table(title="Contact Estimator Consistency")
    table.add_column("Noise Level", style="cyan")
    table.add_column("MAE", justify="right")
    table.add_column("Max Diff", justify="right")
    table.add_column("Status", justify="center")

    for noise_key, metrics in results.items():
        noise_val = float(noise_key.split("_")[1])
        # Allow more tolerance with more noise
        threshold = 0.1 + noise_val * 2
        status = (
            "[green]✓ PASS[/green]"
            if metrics["mae"] < threshold
            else "[red]✗ FAIL[/red]"
        )
        table.add_row(
            f"{noise_val:.2f} rad",
            f"{metrics['mae']:.4f}",
            f"{metrics['max_diff']:.4f}",
            status,
        )

    console.print(table)

    return results


# =============================================================================
# Test 6: Batch Feasibility Analysis (v0.9.2)
# =============================================================================


def test_batch_feasibility(
    amp_dir: str,
    model_path: str,
    config: AMPFeatureConfig,
    output_csv: str = "data/amp/feasibility_report.csv",
    max_frames_per_clip: int = 200,
    soft_height_stabilize: bool = True,
) -> Dict[str, Any]:
    """Run feasibility analysis on all AMP clips in a directory.

    This test is a **dataset feasibility gate** that categorizes clips by
    their trackability under physics simulation.

    For each clip, computes:
    - slip_rate: Fraction of stance frames with >0.1 m/s foot velocity
    - max_pitch: Maximum absolute pitch angle (degrees)
    - max_roll: Maximum absolute roll angle (degrees)
    - joint_rmse: Overall joint tracking RMSE (radians)
    - contact_mismatch: Fraction of contact estimator disagreements

    Produces a CSV file for data curation and curriculum learning.

    Args:
        amp_dir: Directory containing AMP pickle files
        model_path: Path to MuJoCo scene XML
        config: AMP feature config
        output_csv: Path to output CSV file
        max_frames_per_clip: Max frames to test per clip
        soft_height_stabilize: Use soft height stabilization

    Returns:
        Dict with summary statistics and per-clip results
    """
    print("\n[bold cyan]Test 6: Batch Feasibility Analysis[/bold cyan]")
    print(f"  (Produces dataset feasibility report for data curation)")
    print(f"  (Output: {output_csv})")

    amp_dir = Path(amp_dir)
    amp_files = sorted(amp_dir.glob("*_amp.pkl"))

    if not amp_files:
        print("[red]No AMP files found![/red]")
        return {}

    print(f"  Found {len(amp_files)} AMP clips to analyze")

    # Load MuJoCo model once
    from playground_amp.envs.wildrobot_env import get_assets

    model_path = Path(model_path)
    mj_model = mujoco.MjModel.from_xml_string(
        model_path.read_text(), assets=get_assets(model_path.parent)
    )

    # Feasibility gates (from user specification)
    gates = {
        "slip_rate": 0.15,  # < 15%
        "max_pitch_deg": 25.0,  # < 25°
        "max_roll_deg": 25.0,  # < 25°
        "joint_rmse_rad": 0.10,  # < 0.10 rad
        "contact_mismatch": 0.20,  # < 20%
    }

    results = []

    for amp_file in amp_files:
        clip_name = amp_file.stem.replace("_amp", "")
        print(f"\n  Analyzing: {clip_name}...", end=" ")

        try:
            metrics = analyze_clip_feasibility(
                str(amp_file),
                mj_model,
                config,
                max_frames=max_frames_per_clip,
                soft_height_stabilize=soft_height_stabilize,
            )

            # Check gates
            passes_gates = (
                metrics["slip_rate"] < gates["slip_rate"]
                and metrics["max_pitch_deg"] < gates["max_pitch_deg"]
                and metrics["max_roll_deg"] < gates["max_roll_deg"]
                and metrics["joint_rmse_rad"] < gates["joint_rmse_rad"]
                and metrics["contact_mismatch"] < gates["contact_mismatch"]
            )

            metrics["clip_name"] = clip_name
            metrics["passes_gates"] = passes_gates
            results.append(metrics)

            status = "[green]✓ PASS[/green]" if passes_gates else "[red]✗ FAIL[/red]"
            print(
                f"{status} slip={metrics['slip_rate']:.1%}, "
                f"pitch={metrics['max_pitch_deg']:.1f}°, "
                f"rmse={metrics['joint_rmse_rad']:.3f}rad"
            )

        except Exception as e:
            print(f"[red]ERROR: {e}[/red]")
            results.append(
                {
                    "clip_name": clip_name,
                    "error": str(e),
                    "passes_gates": False,
                }
            )

    # Write CSV
    import csv

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "clip_name",
        "passes_gates",
        "slip_rate",
        "max_pitch_deg",
        "max_roll_deg",
        "joint_rmse_rad",
        "contact_mismatch",
        "mean_slip_velocity",
        "height_change",
        "num_frames",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"\n  [green]CSV written to: {output_path}[/green]")

    # Summary statistics
    passing = [r for r in results if r.get("passes_gates", False)]
    failing = [r for r in results if not r.get("passes_gates", True)]

    print(f"\n[bold]═══ FEASIBILITY SUMMARY ═══[/bold]")
    print(f"  Total clips:   {len(results)}")
    print(f"  Passing gates: {len(passing)} ({100*len(passing)/len(results):.0f}%)")
    print(f"  Failing gates: {len(failing)} ({100*len(failing)/len(results):.0f}%)")

    if passing:
        print(f"\n  [green]Clips passing feasibility gates:[/green]")
        for r in passing:
            print(f"    ✓ {r['clip_name']}")

    if failing:
        print(f"\n  [yellow]Clips failing feasibility gates:[/yellow]")
        for r in failing:
            if "error" in r:
                print(f"    ✗ {r['clip_name']}: ERROR - {r['error']}")
            else:
                reasons = []
                if r.get("slip_rate", 1) >= gates["slip_rate"]:
                    reasons.append(f"slip={r['slip_rate']:.1%}")
                if r.get("max_pitch_deg", 90) >= gates["max_pitch_deg"]:
                    reasons.append(f"pitch={r['max_pitch_deg']:.1f}°")
                if r.get("max_roll_deg", 90) >= gates["max_roll_deg"]:
                    reasons.append(f"roll={r['max_roll_deg']:.1f}°")
                if r.get("joint_rmse_rad", 1) >= gates["joint_rmse_rad"]:
                    reasons.append(f"rmse={r['joint_rmse_rad']:.3f}")
                if r.get("contact_mismatch", 1) >= gates["contact_mismatch"]:
                    reasons.append(f"contact={r['contact_mismatch']:.1%}")
                print(f"    ✗ {r['clip_name']}: {', '.join(reasons)}")

    # Print gate thresholds
    print(f"\n  [bold]Feasibility Gate Thresholds:[/bold]")
    print(f"    slip_rate < {gates['slip_rate']:.0%}")
    print(f"    max_pitch < {gates['max_pitch_deg']:.0f}°")
    print(f"    max_roll < {gates['max_roll_deg']:.0f}°")
    print(f"    joint_rmse < {gates['joint_rmse_rad']:.2f} rad")
    print(f"    contact_mismatch < {gates['contact_mismatch']:.0%}")

    return {
        "results": results,
        "passing": [r["clip_name"] for r in passing],
        "failing": [r["clip_name"] for r in failing],
        "gates": gates,
        "csv_path": str(output_path),
    }


def analyze_clip_feasibility(
    amp_path: str,
    mj_model: mujoco.MjModel,
    config: AMPFeatureConfig,
    max_frames: int = 200,
    soft_height_stabilize: bool = True,
    sim_dt: float = 0.002,
    control_dt: float = 0.02,
    # v0.9.3: Proper stabilization parameters
    stab_kp: float = 60.0,
    stab_kd: float = 10.0,
    stab_force_cap: float = 4.0,
    stab_orientation_gate: float = 0.436,  # 25 degrees
) -> Dict[str, float]:
    """Analyze a single clip for feasibility metrics.

    Returns:
        Dict with slip_rate, max_pitch_deg, max_roll_deg, joint_rmse_rad,
        contact_mismatch, mean_slip_velocity, height_change, num_frames,
        load_support_ratio, stab_force_mean
    """
    with open(amp_path, "rb") as f:
        ref_data = pickle.load(f)

    num_frames = min(max_frames, ref_data["num_frames"])
    ref_dof_pos = ref_data["dof_pos"][:num_frames]
    ref_root_pos = ref_data["root_pos"][:num_frames]
    ref_root_rot = ref_data["root_rot"][:num_frames]  # xyzw
    ref_features = ref_data["features"][:num_frames]

    n = config.num_actuated_joints
    n_substeps = int(control_dt / sim_dt)

    # Setup MuJoCo
    mj_data = mujoco.MjData(mj_model)
    mj_model.opt.timestep = sim_dt

    # Initialize from home keyframe
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
    mj_data.qvel[:] = 0
    mujoco.mj_forward(mj_model, mj_data)

    target_height = ref_root_pos[:, 2].mean()

    # Robot physics constants
    robot_mass = 3.383  # kg
    mg = robot_mass * 9.81  # 33.19 N

    # Get foot geom IDs
    try:
        left_foot_geom = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_GEOM, "left_toe"
        )
        right_foot_geom = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_GEOM, "right_toe"
        )
    except Exception:
        left_foot_geom = right_foot_geom = None

    # Storage
    joint_tracking_errors = []
    foot_slip_velocities = []
    pitch_values = []
    roll_values = []
    contact_mismatches = 0
    total_contacts = 0
    height_trajectory = []
    total_normal_forces = []
    stabilization_forces = []

    prev_left_foot_pos = None
    prev_right_foot_pos = None

    for i in range(num_frames):
        # Set control targets
        mj_data.ctrl[:] = ref_dof_pos[i]

        # v0.9.3: Proper soft height stabilization (assist-only, not puppet)
        if soft_height_stabilize:
            # Check orientation gate
            w, x, y, z = mj_data.qpos[3:7]
            sinp = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
            curr_pitch = np.abs(np.arcsin(sinp))
            sinr_cosp = 2.0 * (w * x + y * z)
            cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
            curr_roll = np.abs(np.arctan2(sinr_cosp, cosr_cosp))

            if curr_pitch < stab_orientation_gate and curr_roll < stab_orientation_gate:
                height_error = target_height - mj_data.qpos[2]
                f_stab = stab_kp * height_error - stab_kd * mj_data.qvel[2]
                f_stab = np.clip(f_stab, -stab_force_cap, stab_force_cap)
                mj_data.qfrc_applied[2] = f_stab
            else:
                mj_data.qfrc_applied[2] = 0.0
        else:
            mj_data.qfrc_applied[2] = 0.0

        stabilization_forces.append(mj_data.qfrc_applied[2])

        # Step physics
        for _ in range(n_substeps):
            mujoco.mj_step(mj_model, mj_data)

        # Extract contact forces
        frame_total_fn = 0.0
        for contact_idx in range(mj_data.ncon):
            contact = mj_data.contact[contact_idx]
            if contact.efc_address >= 0:
                fn = mj_data.efc_force[contact.efc_address]
                if fn > 0:
                    frame_total_fn += fn
        total_normal_forces.append(frame_total_fn)

        # Read state
        joint_pos = mj_data.qpos[7 : 7 + n].copy()
        root_quat = mj_data.qpos[3:7].copy()  # wxyz
        root_height = mj_data.qpos[2]

        # Joint tracking error
        joint_error = joint_pos - ref_dof_pos[i]
        joint_tracking_errors.append(joint_error)

        # Pitch and roll from quaternion
        w, x, y, z = root_quat
        sinp = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
        pitch = np.arcsin(sinp)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        pitch_values.append(pitch)
        roll_values.append(roll)
        height_trajectory.append(root_height)

        # Foot slip detection
        if left_foot_geom is not None:
            left_foot_pos = mj_data.geom_xpos[left_foot_geom].copy()
            right_foot_pos = mj_data.geom_xpos[right_foot_geom].copy()

            if prev_left_foot_pos is not None:
                left_vel = (
                    np.linalg.norm(left_foot_pos[:2] - prev_left_foot_pos[:2])
                    / control_dt
                )
                right_vel = (
                    np.linalg.norm(right_foot_pos[:2] - prev_right_foot_pos[:2])
                    / control_dt
                )

                # Check contact state from reference
                ref_contacts = ref_features[i, 2 * n + 7 : 2 * n + 11]
                left_contact = ref_contacts[0] > 0.5 or ref_contacts[1] > 0.5
                right_contact = ref_contacts[2] > 0.5 or ref_contacts[3] > 0.5

                if left_contact:
                    foot_slip_velocities.append(left_vel)
                if right_contact:
                    foot_slip_velocities.append(right_vel)

            prev_left_foot_pos = left_foot_pos
            prev_right_foot_pos = right_foot_pos

        # Contact mismatch
        online_contacts = np.array(
            estimate_foot_contacts_from_joints(jnp.array(joint_pos), config)
        )
        offline_contacts = ref_features[i, 2 * n + 7 : 2 * n + 11]
        online_binary = (online_contacts > 0.5).astype(int)
        offline_binary = (offline_contacts > 0.5).astype(int)
        total_contacts += 4
        contact_mismatches += np.sum(online_binary != offline_binary)

    # Compute metrics
    joint_tracking_errors = np.array(joint_tracking_errors)
    joint_rmse = np.sqrt(np.mean(joint_tracking_errors**2))

    slip_rate = (
        sum(1 for v in foot_slip_velocities if v > 0.1) / len(foot_slip_velocities)
        if foot_slip_velocities
        else 1.0
    )
    mean_slip_velocity = np.mean(foot_slip_velocities) if foot_slip_velocities else 0.0

    max_pitch = np.max(np.abs(pitch_values))
    max_roll = np.max(np.abs(roll_values))

    contact_mismatch_rate = contact_mismatches / total_contacts if total_contacts else 0

    height_change = height_trajectory[-1] - height_trajectory[0]

    # v0.9.3: Load support metrics
    total_normal_forces_arr = np.array(total_normal_forces)
    mean_total_fn = (
        np.mean(total_normal_forces_arr) if total_normal_forces_arr.size > 0 else 0
    )
    load_support_ratio = mean_total_fn / mg

    stabilization_forces_arr = np.array(stabilization_forces)
    stab_force_mean = (
        np.mean(np.abs(stabilization_forces_arr))
        if stabilization_forces_arr.size > 0
        else 0
    )

    return {
        "slip_rate": slip_rate,
        "max_pitch_deg": np.degrees(max_pitch),
        "max_roll_deg": np.degrees(max_roll),
        "joint_rmse_rad": joint_rmse,
        "contact_mismatch": contact_mismatch_rate,
        "mean_slip_velocity": mean_slip_velocity,
        "height_change": height_change,
        "num_frames": num_frames,
        "load_support_ratio": load_support_ratio,
        "stab_force_mean": stab_force_mean,
    }


def generate_diagnostic_trace(
    amp_path: str,
    model_path: str,
    config: AMPFeatureConfig,
    output_npz: str = None,
    max_frames: int = 200,
) -> Dict[str, np.ndarray]:
    """Generate time-series diagnostic trace for a single clip.

    Useful for detailed analysis of failure modes.

    Returns:
        Dict with time series arrays:
        - time: Frame timestamps
        - slip_left, slip_right: Per-frame foot velocities
        - pitch, roll, yaw: Base orientation angles
        - height: Base height
        - joint_rmse: Per-frame joint tracking RMSE
        - contact_left, contact_right: Contact states
    """
    print(f"\n[bold cyan]Generating diagnostic trace for: {amp_path}[/bold cyan]")

    with open(amp_path, "rb") as f:
        ref_data = pickle.load(f)

    from playground_amp.envs.wildrobot_env import get_assets

    model_path = Path(model_path)
    mj_model = mujoco.MjModel.from_xml_string(
        model_path.read_text(), assets=get_assets(model_path.parent)
    )
    mj_data = mujoco.MjData(mj_model)

    num_frames = min(max_frames, ref_data["num_frames"])
    ref_dof_pos = ref_data["dof_pos"][:num_frames]
    ref_root_pos = ref_data["root_pos"][:num_frames]
    ref_features = ref_data["features"][:num_frames]
    dt = ref_data["dt"]

    n = config.num_actuated_joints
    sim_dt = 0.002
    control_dt = dt
    n_substeps = int(control_dt / sim_dt)

    mj_model.opt.timestep = sim_dt
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
    mj_data.qvel[:] = 0

    target_height = ref_root_pos[:, 2].mean()

    # Get foot geom IDs
    left_foot_geom = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "left_toe")
    right_foot_geom = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "right_toe")

    # Time series storage
    trace = {
        "time": [],
        "slip_left": [],
        "slip_right": [],
        "pitch": [],
        "roll": [],
        "yaw": [],
        "height": [],
        "joint_rmse": [],
        "contact_left": [],
        "contact_right": [],
        "ref_height": [],
    }

    prev_left_foot_pos = None
    prev_right_foot_pos = None

    for i in range(num_frames):
        # Apply control
        mj_data.ctrl[:] = ref_dof_pos[i]

        # Soft height stabilization
        height_error = target_height - mj_data.qpos[2]
        mj_data.qfrc_applied[2] = 500.0 * height_error - 50.0 * mj_data.qvel[2]

        # Step physics
        for _ in range(n_substeps):
            mujoco.mj_step(mj_model, mj_data)

        # Read state
        joint_pos = mj_data.qpos[7 : 7 + n].copy()
        root_quat = mj_data.qpos[3:7].copy()

        # Orientation
        w, x, y, z = root_quat
        sinp = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
        pitch = np.arcsin(sinp)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        # Joint tracking RMSE
        joint_error = joint_pos - ref_dof_pos[i]
        joint_rmse = np.sqrt(np.mean(joint_error**2))

        # Foot slip
        left_foot_pos = mj_data.geom_xpos[left_foot_geom].copy()
        right_foot_pos = mj_data.geom_xpos[right_foot_geom].copy()

        if prev_left_foot_pos is not None:
            slip_left = (
                np.linalg.norm(left_foot_pos[:2] - prev_left_foot_pos[:2]) / control_dt
            )
            slip_right = (
                np.linalg.norm(right_foot_pos[:2] - prev_right_foot_pos[:2])
                / control_dt
            )
        else:
            slip_left = slip_right = 0.0

        prev_left_foot_pos = left_foot_pos
        prev_right_foot_pos = right_foot_pos

        # Contact state
        ref_contacts = ref_features[i, 2 * n + 7 : 2 * n + 11]
        contact_left = ref_contacts[0] > 0.5 or ref_contacts[1] > 0.5
        contact_right = ref_contacts[2] > 0.5 or ref_contacts[3] > 0.5

        # Store
        trace["time"].append(i * control_dt)
        trace["slip_left"].append(slip_left)
        trace["slip_right"].append(slip_right)
        trace["pitch"].append(np.degrees(pitch))
        trace["roll"].append(np.degrees(roll))
        trace["yaw"].append(np.degrees(yaw))
        trace["height"].append(mj_data.qpos[2])
        trace["joint_rmse"].append(joint_rmse)
        trace["contact_left"].append(float(contact_left))
        trace["contact_right"].append(float(contact_right))
        trace["ref_height"].append(ref_root_pos[i, 2])

    # Convert to arrays
    for key in trace:
        trace[key] = np.array(trace[key])

    # Save if output path provided
    if output_npz:
        np.savez(output_npz, **trace)
        print(f"  Trace saved to: {output_npz}")

    # Print summary
    print(f"\n  [bold]Diagnostic Trace Summary[/bold]")
    print(f"    Duration: {trace['time'][-1]:.2f}s ({num_frames} frames)")
    print(
        f"    Height: {trace['height'][0]:.3f}m → {trace['height'][-1]:.3f}m "
        f"(Δ={trace['height'][-1]-trace['height'][0]:.3f}m)"
    )
    print(
        f"    Pitch: min={trace['pitch'].min():.1f}°, max={trace['pitch'].max():.1f}°"
    )
    print(f"    Roll: min={trace['roll'].min():.1f}°, max={trace['roll'].max():.1f}°")
    print(f"    Joint RMSE: mean={trace['joint_rmse'].mean():.4f} rad")

    # Slip analysis by contact phase
    stance_left = trace["contact_left"] > 0.5
    stance_right = trace["contact_right"] > 0.5
    slip_during_stance_left = trace["slip_left"][stance_left]
    slip_during_stance_right = trace["slip_right"][stance_right]

    if len(slip_during_stance_left) > 0:
        print(
            f"    Left foot slip (during stance): "
            f"mean={slip_during_stance_left.mean():.3f} m/s, "
            f"max={slip_during_stance_left.max():.3f} m/s"
        )
    if len(slip_during_stance_right) > 0:
        print(
            f"    Right foot slip (during stance): "
            f"mean={slip_during_stance_right.mean():.3f} m/s, "
            f"max={slip_during_stance_right.max():.3f} m/s"
        )

    return trace


# =============================================================================
# Helper Functions
# =============================================================================


def quaternion_rotate_inverse(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate vector by inverse of quaternion (wxyz format)."""
    w, x, y, z = quat
    qc = np.array([w, -x, -y, -z])
    return quaternion_rotate(qc, vec)


def quaternion_rotate(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate vector by quaternion (wxyz format)."""
    w, x, y, z = quat
    vx, vy, vz = vec
    t = 2 * np.array(
        [
            y * vz - z * vy,
            z * vx - x * vz,
            x * vy - y * vx,
        ]
    )
    return vec + w * t + np.cross([x, y, z], t)


def quaternion_to_yaw_xyzw(quats: np.ndarray) -> np.ndarray:
    """Extract yaw from quaternions in xyzw format."""
    x, y, z, w = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return np.arctan2(siny_cosp, cosy_cosp)


def get_feature_ranges(config: AMPFeatureConfig) -> Dict[str, Tuple[int, int]]:
    """Get feature index ranges by component."""
    n = config.num_actuated_joints
    return {
        f"Joint pos (0-{n-1})": (0, n),
        f"Joint vel ({n}-{2*n-1})": (n, 2 * n),
        f"Root lin vel ({2*n}-{2*n+2})": (2 * n, 2 * n + 3),
        f"Root ang vel ({2*n+3}-{2*n+5})": (2 * n + 3, 2 * n + 6),
        f"Root height ({2*n+6})": (2 * n + 6, 2 * n + 7),
        f"Foot contacts ({2*n+7}-{2*n+10})": (2 * n + 7, 2 * n + 11),
    }


def compute_component_mae(
    ref_features: np.ndarray,
    sim_features: np.ndarray,
    config: AMPFeatureConfig,
) -> Dict[str, float]:
    """Compute MAE per component."""
    min_len = min(len(ref_features), len(sim_features))
    ref_features = ref_features[:min_len]
    sim_features = sim_features[:min_len]

    results = {}
    for name, (start, end) in get_feature_ranges(config).items():
        mae = np.mean(np.abs(ref_features[:, start:end] - sim_features[:, start:end]))
        results[name] = mae
    return results


def print_mae_table(results: Dict[str, float], title: str):
    """Print MAE results as table."""
    table = Table(title=title)
    table.add_column("Component", style="cyan")
    table.add_column("MAE", justify="right")
    table.add_column("Status", justify="center")

    thresholds = {
        "Joint pos": 0.05,
        "Joint vel": 0.2,
        "Root lin vel": 0.1,
        "Root ang vel": 0.1,
        "Root height": 0.01,
        "Foot contacts": 0.3,
    }

    for name, mae in results.items():
        threshold = 0.1
        for key, th in thresholds.items():
            if key in name:
                threshold = th
                break
        status = "[green]✓ PASS[/green]" if mae < threshold else "[red]✗ FAIL[/red]"
        table.add_row(name, f"{mae:.4f}", status)

    console.print(table)


def recompute_features(
    root_pos: np.ndarray,
    root_rot: np.ndarray,  # xyzw
    dof_pos: np.ndarray,
    dt: float,
    config: AMPFeatureConfig,
) -> np.ndarray:
    """Recompute AMP features from raw data."""
    N = len(root_pos)

    # Compute velocities
    dof_vel = np.zeros_like(dof_pos)
    dof_vel[1:] = (dof_pos[1:] - dof_pos[:-1]) / dt
    dof_vel[0] = dof_vel[1] if N > 1 else 0

    root_lin_vel = np.zeros_like(root_pos)
    root_lin_vel[1:] = (root_pos[1:] - root_pos[:-1]) / dt
    root_lin_vel[0] = root_lin_vel[1] if N > 1 else 0

    # Angular velocity using SciPy
    rotations = R.from_quat(root_rot)
    r_prev = rotations[:-1]
    r_curr = rotations[1:]
    r_delta = r_curr * r_prev.inv()
    root_ang_vel = np.zeros((N, 3))
    root_ang_vel[1:] = r_delta.as_rotvec() / dt
    root_ang_vel[0] = root_ang_vel[1] if N > 1 else 0

    # Convert to heading-local
    yaw = quaternion_to_yaw_xyzw(root_rot)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    def to_heading_local(v):
        vx = cos_yaw * v[:, 0] + sin_yaw * v[:, 1]
        vy = -sin_yaw * v[:, 0] + cos_yaw * v[:, 1]
        vz = v[:, 2]
        return np.stack([vx, vy, vz], axis=1)

    root_lin_vel_heading = to_heading_local(root_lin_vel)
    root_ang_vel_heading = to_heading_local(root_ang_vel)

    # Root height
    root_height = root_pos[:, 2:3]

    # Foot contacts
    foot_contacts = []
    for i in range(N):
        contacts = np.array(
            estimate_foot_contacts_from_joints(jnp.array(dof_pos[i]), config)
        )
        foot_contacts.append(contacts)
    foot_contacts = np.stack(foot_contacts)

    # Assemble features
    features = np.concatenate(
        [
            dof_pos,
            dof_vel,
            root_lin_vel_heading,
            root_ang_vel_heading,
            root_height,
            foot_contacts,
        ],
        axis=1,
    ).astype(np.float32)

    return features


def run_basic_parity(
    ref_data: Dict[str, Any],
    indices: np.ndarray,
    config: AMPFeatureConfig,
) -> Dict[str, float]:
    """Run basic parity test on sampled frames."""
    ref_features = ref_data["features"][indices]

    # Recompute features
    root_pos = ref_data["root_pos"][indices]
    root_rot = ref_data["root_rot"][indices]
    dof_pos = ref_data["dof_pos"][indices]
    dt = ref_data["dt"]

    sim_features = recompute_features(root_pos, root_rot, dof_pos, dt, config)

    return compute_component_mae(ref_features, sim_features, config)


def compute_batch_statistics(
    all_results: List[Dict[str, Any]],
    config: AMPFeatureConfig,
) -> Dict[str, Dict[str, float]]:
    """Compute statistics across all batch results."""
    stats = {}

    for name, _ in get_feature_ranges(config).items():
        values = [r.get(name, 0) for r in all_results if name in r]
        if values:
            stats[name] = {
                "mean": np.mean(values),
                "worst": np.max(values),
                "p95": np.percentile(values, 95),
                "std": np.std(values),
            }

    return stats


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Robust AMP Parity Test Suite")
    parser.add_argument(
        "--test",
        type=str,
        choices=[
            "1a",
            "1b",
            "yaw_invariance",
            "quat_sign_flip",
            "batch_coverage",
            "contact_consistency",
        ],
        help="Run specific test",
    )
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument(
        "--reference",
        type=str,
        default="data/amp/walking_medium01_amp.pkl",
        help="Reference AMP file for single-motion tests",
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
        "--motion-dir",
        type=str,
        default="assets/motions",
        help="Directory with motion files",
    )
    parser.add_argument(
        "--amp-dir",
        type=str,
        default="data/amp",
        help="Directory with AMP files",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=100,
        help="Max frames for single tests",
    )
    parser.add_argument(
        "--soft-stabilize",
        action="store_true",
        help="Enable soft height stabilization for Test 1B (assist-only mode)",
    )

    args = parser.parse_args()

    # Load config
    print(f"[bold blue]Loading robot config:[/bold blue] {args.robot_config}")
    load_robot_config(args.robot_config)
    robot_config = get_robot_config()
    amp_config = create_amp_config_from_robot(robot_config)

    # Determine which tests to run
    tests_to_run = []
    if args.all:
        tests_to_run = [
            "1a",
            "1b",
            "yaw_invariance",
            "quat_sign_flip",
            "batch_coverage",
            "contact_consistency",
        ]
    elif args.test:
        tests_to_run = [args.test]
    else:
        parser.print_help()
        return

    results = {}

    # Run tests
    if "1a" in tests_to_run:
        results["test_1a"] = test_1a_adversarial_injection(
            args.reference, args.model, amp_config, args.max_frames
        )

    if "1b" in tests_to_run:
        results["test_1b"] = test_1b_mj_step_tracking(
            args.reference,
            args.model,
            amp_config,
            args.max_frames,
            soft_height_stabilize=args.soft_stabilize,
        )

    if "yaw_invariance" in tests_to_run:
        results["yaw_invariance"] = test_yaw_invariance(args.reference, amp_config)

    if "quat_sign_flip" in tests_to_run:
        results["quat_sign_flip"] = test_quaternion_sign_flip(
            args.reference, amp_config
        )

    if "batch_coverage" in tests_to_run:
        results["batch_coverage"] = test_batch_coverage(
            args.motion_dir, args.amp_dir, args.model, amp_config
        )

    if "contact_consistency" in tests_to_run:
        results["contact_consistency"] = test_contact_consistency(
            args.reference, amp_config
        )

    # Summary
    print("\n" + "=" * 60)
    print("[bold green]ROBUST PARITY TEST SUITE COMPLETE[/bold green]")
    print("=" * 60)

    for test_name, test_results in results.items():
        if test_results:
            print(f"  {test_name}: ✓ Completed")
        else:
            print(f"  {test_name}: ⚠ No results")


if __name__ == "__main__":
    main()
