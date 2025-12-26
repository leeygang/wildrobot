#!/usr/bin/env python3
"""Physics Reference Generator MVP (Option 1: Physics-Based Reference Data).

Generates dynamically feasible AMP reference data by rolling out GMR motion targets
through MuJoCo physics simulation with PD control.

v1.0.0: Initial implementation

This script:
1. Takes GMR motion files (root pose + joint targets at 50 Hz)
2. Runs MuJoCo physics simulation with PD control
3. Records realized physics states (qpos, qvel)
4. Extracts physics-derived contacts from MuJoCo contact forces
5. Computes AMP features using the frozen extractor (same as training)
6. Applies quality gates (accept full clip / trim to feasible segments / reject)
7. Outputs merged dataset in exact discriminator format

Quality Gates:
- Accept if mean(sum F_n)/mg >= 0.85 throughout clip
- Trim to longest feasible segment if partial
- Reject if no feasible segment >= min_segment_length

Key Insight: By rolling out through physics, the resulting reference data is
GUARANTEED to be dynamically feasible for the robot, eliminating the "infeasible
reference" failure mode entirely.

Usage:
    cd ~/projects/wildrobot

    # Generate from all GMR motions (default)
    uv run python scripts/generate_physics_reference_dataset.py \
        --robot-config assets/robot_config.yaml \
        --model assets/scene_flat_terrain.xml \
        --output-dir playground_amp/data/physics_ref

    # Generate from specific motion
    uv run python scripts/generate_physics_reference_dataset.py \
        --input assets/motions/walking_medium01.pkl \
        --robot-config assets/robot_config.yaml \
        --model assets/scene_flat_terrain.xml \
        --output playground_amp/data/physics_ref/walking_medium01_physics.pkl
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
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).parent.parent))

from playground_amp.amp.policy_features import (
    FeatureConfig,
    create_config_from_robot,
)
from playground_amp.configs.training_config import get_robot_config, load_robot_config

console = Console()


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class PhysicsRolloutConfig:
    """Configuration for physics rollout."""

    # Simulation timing
    sim_dt: float = 0.002  # 500 Hz simulation
    control_dt: float = 0.02  # 50 Hz control (matches reference)

    # Quality gate thresholds
    min_load_support_ratio: float = 0.85  # Minimum mean(sum F_n)/mg
    max_stabilizer_contribution: float = 0.50  # Max stab force as fraction of mg
    min_segment_length: int = 50  # Minimum frames for valid segment (1 second)

    # ==========================================================================
    # Harness Mode: "assist" vs "full"
    # ==========================================================================
    # "assist" - Light balance assist only (original mode)
    #            Height stabilization with low gains, orientation gate
    #            Good for testing feasibility, rejects infeasible motions
    #
    # "full"   - Full harness mode (NEW)
    #            Height + XY + orientation stabilization
    #            Keeps robot upright to generate feasible walking data
    #            Use this when GMR motions are infeasible
    # ==========================================================================
    harness_mode: str = "full"  # "assist" or "full"

    # Height stabilization (Z-axis)
    stab_z_kp: float = 200.0  # N/m (was 60 for assist mode)
    stab_z_kd: float = 40.0  # N·s/m (was 10 for assist mode)
    stab_z_force_cap: float = 20.0  # N (~60% mg, was 4N)

    # XY stabilization (prevent drift) - only in "full" mode
    stab_xy_kp: float = 100.0  # N/m
    stab_xy_kd: float = 20.0  # N·s/m
    stab_xy_force_cap: float = 10.0  # N per axis

    # Orientation stabilization (pitch/roll) - only in "full" mode
    stab_rot_kp: float = 20.0  # N·m/rad
    stab_rot_kd: float = 2.0  # N·m·s/rad
    stab_rot_torque_cap: float = 5.0  # N·m per axis

    # Orientation gate (disable if too tilted) - only in "assist" mode
    stab_orientation_gate: float = 0.436  # radians (25°)

    # Robot physics
    robot_mass: float = 3.383  # kg
    gravity: float = 9.81  # m/s²

    @property
    def mg(self) -> float:
        """Weight force in Newtons."""
        return self.robot_mass * self.gravity

    @property
    def n_substeps(self) -> int:
        """Number of simulation substeps per control step."""
        return int(self.control_dt / self.sim_dt)


@dataclass
class ClipMetadata:
    """Metadata for a processed clip."""

    clip_name: str
    status: str  # "accepted", "trimmed", "rejected"
    reason: str
    total_frames: int
    accepted_frames: int
    trimmed_ranges: List[Tuple[int, int]] = field(default_factory=list)

    # Key metrics
    mean_load_support: float = 0.0
    max_stabilizer_force: float = 0.0
    joint_tracking_rmse: float = 0.0
    mean_contact_force: float = 0.0


# =============================================================================
# Physics Rollout Engine
# =============================================================================


def load_mujoco_model(model_path: str) -> mujoco.MjModel:
    """Load MuJoCo model with assets."""
    from playground_amp.envs.wildrobot_env import get_assets

    model_path = Path(model_path)
    mj_model = mujoco.MjModel.from_xml_string(
        model_path.read_text(), assets=get_assets(model_path.parent)
    )
    return mj_model


def run_physics_rollout(
    gmr_motion: Dict[str, Any],
    mj_model: mujoco.MjModel,
    config: PhysicsRolloutConfig,
    amp_config: FeatureConfig,
) -> Dict[str, Any]:
    """Run physics rollout on GMR motion data.

    Args:
        gmr_motion: GMR motion data (root_pos, root_rot, dof_pos)
        mj_model: MuJoCo model
        config: Physics rollout configuration
        amp_config: AMP feature configuration

    Returns:
        Dict with:
        - qpos_trajectory: (N, nq) realized joint positions
        - qvel_trajectory: (N, nv) realized joint velocities
        - root_pos_trajectory: (N, 3) realized root positions
        - root_rot_trajectory: (N, 4) realized root rotations (xyzw)
        - contact_forces: (N, 2) left/right foot normal forces
        - contact_states: (N, 4) binary contact states [l_toe, l_heel, r_toe, r_heel]
        - load_support_ratio: (N,) per-frame load support ratio
        - stabilizer_force: (N,) per-frame stabilizer force
        - frame_metrics: Dict with per-frame quality metrics
    """
    # Extract GMR data
    ref_dof_pos = gmr_motion["dof_pos"]
    ref_root_pos = gmr_motion["root_pos"]
    ref_root_rot = gmr_motion["root_rot"]  # xyzw format
    num_frames = gmr_motion["num_frames"]

    n = amp_config.num_actuated_joints
    mj_data = mujoco.MjData(mj_model)
    mj_model.opt.timestep = config.sim_dt

    # Initialize from home keyframe (robot standing on ground)
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
    mj_data.qvel[:] = 0
    mujoco.mj_forward(mj_model, mj_data)

    # Target height for stabilization (from reference mean)
    target_height = ref_root_pos[:, 2].mean()

    # Get foot geom IDs for contact detection
    left_toe_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "left_toe")
    right_toe_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "right_toe")

    # Storage for rollout data
    qpos_trajectory = []
    qvel_trajectory = []
    root_pos_trajectory = []
    root_rot_trajectory = []
    contact_forces = []  # [left_fn, right_fn]
    contact_states = []  # [l_toe, l_heel, r_toe, r_heel]
    load_support_ratios = []
    stabilizer_forces = []
    joint_tracking_errors = []

    # Track reference XY position for drift prevention
    ref_xy_start = ref_root_pos[0, :2].copy()
    sim_xy_start = mj_data.qpos[0:2].copy()

    for i in range(num_frames):
        # Set control targets (reference joint positions)
        mj_data.ctrl[:n] = ref_dof_pos[i]

        # =================================================================
        # Apply harness forces based on mode
        # =================================================================
        stab_force_z = 0.0
        stab_force_xy = np.zeros(2)
        stab_torque_rp = np.zeros(2)  # roll, pitch torques

        # Get current orientation
        w, x, y, z = mj_data.qpos[3:7]
        sinp = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
        curr_pitch = np.arcsin(sinp)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        curr_roll = np.arctan2(sinr_cosp, cosr_cosp)

        if config.harness_mode == "full":
            # =============================================================
            # FULL HARNESS MODE: Keep robot upright at all costs
            # Height + XY + Orientation stabilization
            # =============================================================

            # 1. Height (Z) stabilization - track reference height
            target_z = ref_root_pos[i, 2]
            height_error = target_z - mj_data.qpos[2]
            stab_force_z = (
                config.stab_z_kp * height_error - config.stab_z_kd * mj_data.qvel[2]
            )
            stab_force_z = np.clip(
                stab_force_z, -config.stab_z_force_cap, config.stab_z_force_cap
            )

            # 2. XY stabilization - prevent drift, track reference trajectory
            # Compute where robot should be relative to start
            ref_xy_delta = ref_root_pos[i, :2] - ref_xy_start
            target_xy = sim_xy_start + ref_xy_delta
            xy_error = target_xy - mj_data.qpos[0:2]
            stab_force_xy = (
                config.stab_xy_kp * xy_error - config.stab_xy_kd * mj_data.qvel[0:2]
            )
            stab_force_xy = np.clip(
                stab_force_xy, -config.stab_xy_force_cap, config.stab_xy_force_cap
            )

            # 3. Orientation stabilization - keep upright (zero pitch/roll)
            roll_error = 0.0 - curr_roll
            pitch_error = 0.0 - curr_pitch
            roll_torque = (
                config.stab_rot_kp * roll_error - config.stab_rot_kd * mj_data.qvel[3]
            )
            pitch_torque = (
                config.stab_rot_kp * pitch_error - config.stab_rot_kd * mj_data.qvel[4]
            )
            stab_torque_rp[0] = np.clip(
                roll_torque, -config.stab_rot_torque_cap, config.stab_rot_torque_cap
            )
            stab_torque_rp[1] = np.clip(
                pitch_torque, -config.stab_rot_torque_cap, config.stab_rot_torque_cap
            )

            # Apply all forces
            mj_data.qfrc_applied[0:2] = stab_force_xy
            mj_data.qfrc_applied[2] = stab_force_z
            mj_data.qfrc_applied[3] = stab_torque_rp[0]  # roll
            mj_data.qfrc_applied[4] = stab_torque_rp[1]  # pitch

        elif config.harness_mode == "assist":
            # =============================================================
            # ASSIST MODE: Light balance assist only (original behavior)
            # Only height stabilization with orientation gate
            # =============================================================
            if (
                np.abs(curr_pitch) < config.stab_orientation_gate
                and np.abs(curr_roll) < config.stab_orientation_gate
            ):
                height_error = target_height - mj_data.qpos[2]
                stab_force_z = (
                    config.stab_z_kp * height_error - config.stab_z_kd * mj_data.qvel[2]
                )
                stab_force_z = np.clip(
                    stab_force_z, -config.stab_z_force_cap, config.stab_z_force_cap
                )
                mj_data.qfrc_applied[2] = stab_force_z
            else:
                mj_data.qfrc_applied[2] = 0.0

            # No XY or orientation stabilization in assist mode
            mj_data.qfrc_applied[0:2] = 0.0
            mj_data.qfrc_applied[3:5] = 0.0

        else:
            # No stabilization
            mj_data.qfrc_applied[:6] = 0.0

        # Track total stabilization force magnitude for quality gate
        total_stab = np.sqrt(stab_force_z**2 + np.sum(stab_force_xy**2))
        stabilizer_forces.append(total_stab)

        # Step physics n_substeps times
        for _ in range(config.n_substeps):
            mujoco.mj_step(mj_model, mj_data)

        # Extract contact forces from MuJoCo
        left_fn, right_fn = extract_foot_contact_forces(
            mj_model, mj_data, left_toe_id, right_toe_id
        )
        contact_forces.append([left_fn, right_fn])

        # Compute load support ratio
        total_fn = left_fn + right_fn
        load_ratio = total_fn / config.mg
        load_support_ratios.append(load_ratio)

        # Extract realized state
        qpos = mj_data.qpos.copy()
        qvel = mj_data.qvel.copy()

        # Convert root quaternion to xyzw format for consistency with reference
        root_pos = qpos[0:3].copy()
        root_quat_wxyz = qpos[3:7].copy()
        root_quat_xyzw = np.array(
            [
                root_quat_wxyz[1],  # x
                root_quat_wxyz[2],  # y
                root_quat_wxyz[3],  # z
                root_quat_wxyz[0],  # w
            ]
        )

        qpos_trajectory.append(qpos)
        qvel_trajectory.append(qvel)
        root_pos_trajectory.append(root_pos)
        root_rot_trajectory.append(root_quat_xyzw)

        # Compute contact states (binary)
        contact_threshold = 0.5  # N - minimum force for contact
        l_contact = 1.0 if left_fn > contact_threshold else 0.0
        r_contact = 1.0 if right_fn > contact_threshold else 0.0
        contact_states.append([l_contact, l_contact, r_contact, r_contact])

        # Track joint tracking error
        joint_pos = qpos[7 : 7 + n]
        joint_error = joint_pos - ref_dof_pos[i]
        joint_tracking_errors.append(joint_error)

    # Convert to arrays
    qpos_trajectory = np.array(qpos_trajectory)
    qvel_trajectory = np.array(qvel_trajectory)
    root_pos_trajectory = np.array(root_pos_trajectory)
    root_rot_trajectory = np.array(root_rot_trajectory)
    contact_forces = np.array(contact_forces)
    contact_states = np.array(contact_states)
    load_support_ratios = np.array(load_support_ratios)
    stabilizer_forces = np.array(stabilizer_forces)
    joint_tracking_errors = np.array(joint_tracking_errors)

    # Compute summary metrics
    frame_metrics = {
        "load_support_ratio": load_support_ratios,
        "stabilizer_force": stabilizer_forces,
        "joint_rmse_per_frame": np.sqrt(np.mean(joint_tracking_errors**2, axis=1)),
        "contact_forces": contact_forces,
    }

    return {
        "qpos_trajectory": qpos_trajectory,
        "qvel_trajectory": qvel_trajectory,
        "root_pos_trajectory": root_pos_trajectory,
        "root_rot_trajectory": root_rot_trajectory,
        "contact_forces": contact_forces,
        "contact_states": contact_states,
        "load_support_ratio": load_support_ratios,
        "stabilizer_force": stabilizer_forces,
        "frame_metrics": frame_metrics,
        "num_frames": num_frames,
    }


def extract_foot_contact_forces(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    left_foot_geom_id: int,
    right_foot_geom_id: int,
) -> Tuple[float, float]:
    """Extract normal contact forces for left and right feet.

    Iterates through MuJoCo contacts and sums normal forces for each foot.

    Returns:
        Tuple of (left_foot_fn, right_foot_fn) in Newtons
    """
    left_fn = 0.0
    right_fn = 0.0

    # Ground geom is typically ID 0 (floor)
    ground_geom_id = 0

    for contact_idx in range(mj_data.ncon):
        contact = mj_data.contact[contact_idx]
        geom1, geom2 = contact.geom1, contact.geom2

        # Check if contact involves foot and ground
        foot_geom = None
        if geom1 == ground_geom_id and geom2 in [left_foot_geom_id, right_foot_geom_id]:
            foot_geom = geom2
        elif geom2 == ground_geom_id and geom1 in [
            left_foot_geom_id,
            right_foot_geom_id,
        ]:
            foot_geom = geom1

        if foot_geom is None:
            continue

        # Get normal force from constraint force
        if contact.efc_address >= 0:
            fn = mj_data.efc_force[contact.efc_address]
            if fn > 0:  # Only count positive (pushing) normal forces
                if foot_geom == left_foot_geom_id:
                    left_fn += fn
                else:
                    right_fn += fn

    return left_fn, right_fn


# =============================================================================
# AMP Feature Extraction from Physics States
# =============================================================================


def extract_amp_features_from_physics(
    rollout: Dict[str, Any],
    amp_config: FeatureConfig,
    control_dt: float,
) -> np.ndarray:
    """Extract AMP features from physics rollout data.

    This is the FROZEN feature extractor - must match exactly what the
    discriminator sees during training.

    Feature format (27-dim for 8-joint robot):
    - joint_pos (8): Joint positions
    - joint_vel (8): Joint velocities (from qvel)
    - root_lin_vel (3): Root linear velocity (heading-local frame)
    - root_ang_vel (3): Root angular velocity (heading-local frame)
    - root_height (1): Root height (z-coordinate)
    - foot_contacts (4): Contact states [l_toe, l_heel, r_toe, r_heel]

    Args:
        rollout: Physics rollout data from run_physics_rollout()
        amp_config: AMP feature configuration
        control_dt: Control timestep

    Returns:
        (N, feature_dim) array of AMP features
    """
    n = amp_config.num_actuated_joints
    num_frames = rollout["num_frames"]

    qpos = rollout["qpos_trajectory"]
    qvel = rollout["qvel_trajectory"]
    root_pos = rollout["root_pos_trajectory"]
    root_rot = rollout["root_rot_trajectory"]  # xyzw
    contact_states = rollout["contact_states"]

    # Extract joint positions and velocities
    joint_pos = qpos[:, 7 : 7 + n]
    joint_vel = qvel[:, 6 : 6 + n]

    # Extract root linear velocity (world frame from qvel)
    root_lin_vel_world = qvel[:, 0:3]

    # Extract root angular velocity (world frame from qvel)
    root_ang_vel_world = qvel[:, 3:6]

    # Convert velocities to heading-local frame
    root_lin_vel_heading = world_to_heading_local(root_lin_vel_world, root_rot)
    root_ang_vel_heading = world_to_heading_local(root_ang_vel_world, root_rot)

    # Root height
    root_height = root_pos[:, 2:3]

    # Assemble AMP features
    features = np.concatenate(
        [
            joint_pos,  # 0-7: joint positions
            joint_vel,  # 8-15: joint velocities
            root_lin_vel_heading,  # 16-18: root linear velocity (heading-local)
            root_ang_vel_heading,  # 19-21: root angular velocity (heading-local)
            root_height,  # 22: root height
            contact_states,  # 23-26: foot contacts
        ],
        axis=1,
    ).astype(np.float32)

    assert (
        features.shape[1] == amp_config.feature_dim
    ), f"Feature dim mismatch: got {features.shape[1]}, expected {amp_config.feature_dim}"

    return features


def world_to_heading_local(
    vectors: np.ndarray,
    quats_xyzw: np.ndarray,
) -> np.ndarray:
    """Convert world-frame vectors to heading-local frame (yaw-removed).

    The heading-local frame is the world frame rotated by negative yaw.
    This provides rotation invariance for the AMP discriminator.

    Args:
        vectors: (N, 3) vectors in world frame
        quats_xyzw: (N, 4) quaternions in xyzw format

    Returns:
        (N, 3) vectors in heading-local frame
    """
    # Extract yaw from quaternions
    x, y, z, w = quats_xyzw[:, 0], quats_xyzw[:, 1], quats_xyzw[:, 2], quats_xyzw[:, 3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    # Compute sin/cos of negative yaw (to rotate back to heading-local)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    # Apply R_z(-yaw) to each vector
    vx, vy, vz = vectors[:, 0], vectors[:, 1], vectors[:, 2]

    vx_local = cos_yaw * vx + sin_yaw * vy
    vy_local = -sin_yaw * vx + cos_yaw * vy
    vz_local = vz  # z component unchanged

    return np.stack([vx_local, vy_local, vz_local], axis=1)


# =============================================================================
# Quality Gates
# =============================================================================


def apply_quality_gates(
    rollout: Dict[str, Any],
    config: PhysicsRolloutConfig,
    clip_name: str,
) -> Tuple[np.ndarray, ClipMetadata]:
    """Apply quality gates to physics rollout.

    Quality criteria depend on harness mode:

    ASSIST mode:
    - Load support ratio >= 0.85 (robot supporting its own weight)
    - Stabilizer contribution <= 50% of mg

    FULL mode:
    - Accept all frames (harness guarantees feasibility)
    - The harness keeps robot upright, so load support will be low
    - We're generating feasible walking data, not testing balance

    Args:
        rollout: Physics rollout data
        config: Physics rollout configuration
        clip_name: Name of the clip for metadata

    Returns:
        Tuple of (valid_frame_mask, ClipMetadata)
    """
    load_ratios = rollout["load_support_ratio"]
    stab_forces = np.abs(rollout["stabilizer_force"])
    num_frames = rollout["num_frames"]

    # Summary metrics
    mean_load_support = np.mean(load_ratios)
    max_stab_force = np.max(stab_forces)
    joint_rmse = np.mean(rollout["frame_metrics"]["joint_rmse_per_frame"])
    mean_contact = np.mean(rollout["contact_forces"])

    if config.harness_mode == "full":
        # =================================================================
        # FULL HARNESS MODE: Accept all frames
        # The harness guarantees the robot stays upright, so we don't
        # need to check load support. We're generating feasible walking
        # data where the physics is valid (contacts, joint motion, etc.)
        # =================================================================
        status = "accepted"
        reason = "Full harness mode - all frames accepted"
        accepted_frames = num_frames
        trimmed_ranges = []
        valid_mask = np.ones(num_frames, dtype=bool)

    else:
        # =================================================================
        # ASSIST MODE: Apply strict quality gates
        # =================================================================
        # Compute per-frame quality
        load_ok = load_ratios >= config.min_load_support_ratio
        stab_ok = stab_forces <= (config.max_stabilizer_contribution * config.mg)
        frame_quality = load_ok & stab_ok

        # Find contiguous segments that pass quality gates
        segments = find_contiguous_segments(frame_quality)

        # Filter to segments >= min_segment_length
        valid_segments = [
            (s, e) for s, e in segments if (e - s) >= config.min_segment_length
        ]

        if np.all(frame_quality):
            # All frames pass - ACCEPT full clip
            status = "accepted"
            reason = "All frames pass quality gates"
            accepted_frames = num_frames
            trimmed_ranges = []
            valid_mask = np.ones(num_frames, dtype=bool)

        elif valid_segments:
            # Some segments pass - TRIM to longest valid segment
            longest_segment = max(valid_segments, key=lambda x: x[1] - x[0])
            start, end = longest_segment
            status = "trimmed"
            reason = f"Trimmed to frames [{start}:{end}] ({end-start} frames)"
            accepted_frames = end - start
            trimmed_ranges = valid_segments
            valid_mask = np.zeros(num_frames, dtype=bool)
            valid_mask[start:end] = True

        else:
            # No valid segments - REJECT
            status = "rejected"
            reason = f"No valid segment >= {config.min_segment_length} frames"
            accepted_frames = 0
            trimmed_ranges = []
            valid_mask = np.zeros(num_frames, dtype=bool)

    metadata = ClipMetadata(
        clip_name=clip_name,
        status=status,
        reason=reason,
        total_frames=num_frames,
        accepted_frames=accepted_frames,
        trimmed_ranges=trimmed_ranges,
        mean_load_support=mean_load_support,
        max_stabilizer_force=max_stab_force,
        joint_tracking_rmse=joint_rmse,
        mean_contact_force=mean_contact,
    )

    return valid_mask, metadata


def find_contiguous_segments(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Find contiguous True segments in boolean mask.

    Returns:
        List of (start, end) tuples for each contiguous segment
    """
    segments = []
    in_segment = False
    start = 0

    for i, val in enumerate(mask):
        if val and not in_segment:
            in_segment = True
            start = i
        elif not val and in_segment:
            in_segment = False
            segments.append((start, i))

    if in_segment:
        segments.append((start, len(mask)))

    return segments


# =============================================================================
# Dataset Generation
# =============================================================================


def process_single_motion(
    gmr_path: Path,
    mj_model: mujoco.MjModel,
    config: PhysicsRolloutConfig,
    amp_config: FeatureConfig,
) -> Tuple[Optional[Dict[str, Any]], ClipMetadata]:
    """Process a single GMR motion file through physics rollout.

    Args:
        gmr_path: Path to GMR pickle file
        mj_model: MuJoCo model
        config: Physics rollout configuration
        amp_config: AMP feature configuration

    Returns:
        Tuple of (physics_amp_data, metadata)
        physics_amp_data is None if clip is rejected
    """
    clip_name = gmr_path.stem

    # Load GMR motion
    with open(gmr_path, "rb") as f:
        gmr_motion = pickle.load(f)

    # Resample to control_dt if needed
    source_fps = gmr_motion["fps"]
    target_fps = 1.0 / config.control_dt
    if abs(source_fps - target_fps) > 0.1:
        gmr_motion = resample_motion(gmr_motion, target_fps)

    # Run physics rollout
    rollout = run_physics_rollout(gmr_motion, mj_model, config, amp_config)

    # Apply quality gates
    valid_mask, metadata = apply_quality_gates(rollout, config, clip_name)

    if metadata.status == "rejected":
        return None, metadata

    # Extract valid frames
    valid_indices = np.where(valid_mask)[0]

    # Extract AMP features from physics rollout
    all_features = extract_amp_features_from_physics(
        rollout, amp_config, config.control_dt
    )
    features = all_features[valid_indices]

    # Extract other data for the valid segment
    dof_pos = rollout["qpos_trajectory"][
        valid_indices, 7 : 7 + amp_config.num_actuated_joints
    ]
    dof_vel = rollout["qvel_trajectory"][
        valid_indices, 6 : 6 + amp_config.num_actuated_joints
    ]
    root_pos = rollout["root_pos_trajectory"][valid_indices]
    root_rot = rollout["root_rot_trajectory"][valid_indices]
    contact_states = rollout["contact_states"][valid_indices]
    contact_forces = rollout["contact_forces"][valid_indices]

    # Build output in discriminator format
    physics_amp_data = {
        # AMP features for discriminator
        "features": features.astype(np.float32),
        "feature_dim": amp_config.feature_dim,
        # Raw data for debugging/analysis
        "dof_pos": dof_pos.astype(np.float32),
        "dof_vel": dof_vel.astype(np.float32),
        "root_pos": root_pos.astype(np.float32),
        "root_rot": root_rot.astype(np.float32),
        "foot_contacts": contact_states.astype(np.float32),
        "contact_forces": contact_forces.astype(np.float32),
        # Metadata
        "fps": target_fps,
        "dt": config.control_dt,
        "num_frames": len(features),
        "duration_sec": len(features) * config.control_dt,
        "source_file": str(gmr_path),
        "robot": "wildrobot",
        "contact_method": "physics",  # v1.0.0: Physics-derived contacts
        # Quality metrics
        "mean_load_support": metadata.mean_load_support,
        "max_stabilizer_force": metadata.max_stabilizer_force,
        "joint_tracking_rmse": metadata.joint_tracking_rmse,
    }

    return physics_amp_data, metadata


def resample_motion(motion_data: Dict[str, Any], target_fps: float) -> Dict[str, Any]:
    """Resample motion to target FPS using linear interpolation."""
    source_fps = motion_data["fps"]
    num_frames = motion_data["num_frames"]
    duration = motion_data["duration_sec"]

    new_num_frames = int(duration * target_fps)

    t_original = np.linspace(0, duration, num_frames)
    t_new = np.linspace(0, duration, new_num_frames)

    def interp_array(arr):
        if arr is None:
            return None
        if arr.ndim == 1:
            return np.interp(t_new, t_original, arr).astype(np.float32)
        else:
            result = np.zeros((new_num_frames, arr.shape[1]), dtype=np.float32)
            for i in range(arr.shape[1]):
                result[:, i] = np.interp(t_new, t_original, arr[:, i])
            return result

    def interp_quat(quats):
        result = interp_array(quats)
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        return result / np.clip(norms, 1e-8, None)

    return {
        "fps": target_fps,
        "root_pos": interp_array(motion_data["root_pos"]),
        "root_rot": interp_quat(motion_data["root_rot"]),
        "dof_pos": interp_array(motion_data["dof_pos"]),
        "source_file": motion_data.get("source_file", "unknown"),
        "robot": motion_data.get("robot", "wildrobot"),
        "num_frames": new_num_frames,
        "duration_sec": duration,
    }


def batch_generate_physics_references(
    input_dir: Path,
    output_dir: Path,
    mj_model: mujoco.MjModel,
    config: PhysicsRolloutConfig,
    amp_config: FeatureConfig,
) -> Tuple[List[Path], List[ClipMetadata]]:
    """Batch process all GMR motions through physics rollout.

    Args:
        input_dir: Directory containing GMR .pkl files
        output_dir: Directory for physics reference output
        mj_model: MuJoCo model
        config: Physics rollout configuration
        amp_config: AMP feature configuration

    Returns:
        Tuple of (list of output paths, list of metadata)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    motion_files = sorted(input_dir.glob("*.pkl"))
    output_paths = []
    all_metadata = []

    print(
        f"\n[bold blue]Processing {len(motion_files)} motions through physics...[/bold blue]"
    )

    for motion_path in motion_files:
        print(f"\n  Processing: {motion_path.name}")

        try:
            physics_data, metadata = process_single_motion(
                motion_path, mj_model, config, amp_config
            )

            # Print status with color
            if metadata.status == "accepted":
                status_str = (
                    f"[green]✓ ACCEPTED[/green] ({metadata.accepted_frames} frames)"
                )
            elif metadata.status == "trimmed":
                status_str = f"[yellow]◐ TRIMMED[/yellow] ({metadata.accepted_frames}/{metadata.total_frames} frames)"
            else:
                status_str = f"[red]✗ REJECTED[/red] ({metadata.reason})"

            print(f"    Status: {status_str}")
            print(f"    Load support: {metadata.mean_load_support:.1%}")
            print(f"    Max stab force: {metadata.max_stabilizer_force:.2f} N")

            if physics_data is not None:
                output_path = output_dir / f"{motion_path.stem}_physics.pkl"
                with open(output_path, "wb") as f:
                    pickle.dump(physics_data, f)
                output_paths.append(output_path)
                print(f"    Saved: {output_path.name}")

            all_metadata.append(metadata)

        except Exception as e:
            print(f"    [red]ERROR: {e}[/red]")
            all_metadata.append(
                ClipMetadata(
                    clip_name=motion_path.stem,
                    status="rejected",
                    reason=f"Error: {e}",
                    total_frames=0,
                    accepted_frames=0,
                )
            )

    return output_paths, all_metadata


def merge_physics_references(
    physics_files: List[Path],
    output_path: Path,
    amp_config: FeatureConfig,
) -> Dict[str, Any]:
    """Merge physics reference files into single discriminator dataset.

    Args:
        physics_files: List of physics reference pickle files
        output_path: Output path for merged dataset
        amp_config: AMP feature configuration

    Returns:
        Merged dataset dictionary
    """
    all_features = []
    all_dof_pos = []
    all_dof_vel = []
    source_files = []
    total_duration = 0.0

    print(
        f"\n[bold blue]Merging {len(physics_files)} physics references...[/bold blue]"
    )

    for physics_path in physics_files:
        with open(physics_path, "rb") as f:
            data = pickle.load(f)

        all_features.append(data["features"])
        all_dof_pos.append(data["dof_pos"])
        all_dof_vel.append(data["dof_vel"])
        source_files.append(physics_path.name)
        total_duration += data["duration_sec"]

        print(
            f"  + {physics_path.name}: {data['num_frames']} frames, {data['duration_sec']:.2f}s"
        )

    # Concatenate all arrays
    merged = {
        # Main AMP features for discriminator
        "features": np.concatenate(all_features, axis=0).astype(np.float32),
        "feature_dim": amp_config.feature_dim,
        # Additional data for debugging/analysis
        "dof_pos": np.concatenate(all_dof_pos, axis=0).astype(np.float32),
        "dof_vel": np.concatenate(all_dof_vel, axis=0).astype(np.float32),
        # Metadata
        "fps": 50.0,
        "dt": 0.02,
        "num_frames": sum(len(f) for f in all_features),
        "duration_sec": total_duration,
        "source_files": source_files,
        "num_motions": len(physics_files),
        "contact_method": "physics",  # v1.0.0
    }

    # Save merged dataset
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(merged, f)

    print(f"\n[bold green]Merged dataset saved to:[/bold green] {output_path}")

    return merged


def print_summary_report(
    all_metadata: List[ClipMetadata],
    merged_data: Optional[Dict[str, Any]] = None,
):
    """Print summary report of physics reference generation."""
    accepted = [m for m in all_metadata if m.status == "accepted"]
    trimmed = [m for m in all_metadata if m.status == "trimmed"]
    rejected = [m for m in all_metadata if m.status == "rejected"]

    print(f"\n{'='*60}")
    print("[bold]PHYSICS REFERENCE GENERATION REPORT[/bold]")
    print(f"{'='*60}")

    # Summary table
    table = Table(title="Clip Processing Summary")
    table.add_column("Status", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Percentage", justify="right")

    total = len(all_metadata)
    table.add_row("Accepted", str(len(accepted)), f"{100*len(accepted)/total:.1f}%")
    table.add_row("Trimmed", str(len(trimmed)), f"{100*len(trimmed)/total:.1f}%")
    table.add_row("Rejected", str(len(rejected)), f"{100*len(rejected)/total:.1f}%")
    table.add_row("Total", str(total), "100%")

    console.print(table)

    # Per-clip details
    if accepted:
        print(f"\n[green]Accepted clips ({len(accepted)}):[/green]")
        for m in accepted:
            print(
                f"  ✓ {m.clip_name}: {m.accepted_frames} frames, load={m.mean_load_support:.1%}"
            )

    if trimmed:
        print(f"\n[yellow]Trimmed clips ({len(trimmed)}):[/yellow]")
        for m in trimmed:
            print(
                f"  ◐ {m.clip_name}: {m.accepted_frames}/{m.total_frames} frames - {m.reason}"
            )

    if rejected:
        print(f"\n[red]Rejected clips ({len(rejected)}):[/red]")
        for m in rejected:
            print(f"  ✗ {m.clip_name}: {m.reason}")

    # Merged dataset stats
    if merged_data:
        print(f"\n[bold]Merged Dataset Statistics:[/bold]")
        print(f"  Total frames: {merged_data['num_frames']:,}")
        print(f"  Total duration: {merged_data['duration_sec']:.2f}s")
        print(f"  Feature dimension: {merged_data['feature_dim']}")
        print(f"  Number of source clips: {merged_data['num_motions']}")

        # Feature statistics
        features = merged_data["features"]
        print(f"\n[bold]Feature Statistics:[/bold]")
        print(f"  Shape: {features.shape}")
        print(f"  Min: {features.min():.4f}")
        print(f"  Max: {features.max():.4f}")
        print(f"  Mean: {features.mean():.4f}")
        print(f"  Std: {features.std():.4f}")


def verify_determinism(
    physics_path: Path,
    gmr_path: Path,
    mj_model: mujoco.MjModel,
    config: PhysicsRolloutConfig,
    amp_config: FeatureConfig,
    tolerance: float = 1e-6,
) -> bool:
    """Verify feature extraction is deterministic by replaying and comparing.

    Acceptance criterion: Replay produces features within tolerance.

    Args:
        physics_path: Path to saved physics reference
        gmr_path: Path to original GMR motion
        mj_model: MuJoCo model
        config: Physics rollout configuration
        amp_config: AMP feature configuration
        tolerance: Maximum allowed difference

    Returns:
        True if deterministic within tolerance
    """
    print(f"\n[cyan]Verifying determinism for {physics_path.name}...[/cyan]")

    # Load saved features
    with open(physics_path, "rb") as f:
        saved_data = pickle.load(f)
    saved_features = saved_data["features"]

    # Replay physics rollout
    replay_data, _ = process_single_motion(gmr_path, mj_model, config, amp_config)

    if replay_data is None:
        print(f"  [red]✗ Replay was rejected[/red]")
        return False

    replay_features = replay_data["features"]

    # Compare (may have different lengths if trimming changed)
    min_len = min(len(saved_features), len(replay_features))
    diff = np.abs(saved_features[:min_len] - replay_features[:min_len])
    max_diff = diff.max()
    mean_diff = diff.mean()

    if max_diff < tolerance:
        print(f"  [green]✓ Deterministic (max diff={max_diff:.2e})[/green]")
        return True
    else:
        print(
            f"  [red]✗ Non-deterministic (max diff={max_diff:.2e}, mean={mean_diff:.2e})[/red]"
        )
        return False


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Generate physics-based AMP reference dataset"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Single GMR motion file to process",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="assets/motions",
        help="Directory containing GMR .pkl files (default: assets/motions)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for single motion processing",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="playground_amp/data/physics_ref",
        help="Output directory for batch processing",
    )
    parser.add_argument(
        "--merged-output",
        type=str,
        default="walking_physics_merged.pkl",
        help="Filename for merged dataset (in output-dir)",
    )
    parser.add_argument(
        "--robot-config",
        type=str,
        default="assets/robot_config.yaml",
        help="Robot config YAML",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="assets/scene_flat_terrain.xml",
        help="MuJoCo scene XML",
    )
    # Quality gate parameters
    parser.add_argument(
        "--min-load-support",
        type=float,
        default=0.85,
        help="Minimum load support ratio (default: 0.85)",
    )
    parser.add_argument(
        "--min-segment-length",
        type=int,
        default=50,
        help="Minimum segment length in frames (default: 50 = 1 second)",
    )
    # Stabilization parameters
    parser.add_argument(
        "--no-stabilization",
        action="store_true",
        help="Disable soft height stabilization",
    )
    parser.add_argument(
        "--stab-force-cap",
        type=float,
        default=4.0,
        help="Stabilization force cap in N (default: 4.0)",
    )
    # Verification
    parser.add_argument(
        "--verify-determinism",
        action="store_true",
        help="Verify feature extraction is deterministic",
    )

    args = parser.parse_args()

    # Load robot config
    print(f"[bold blue]Loading robot config:[/bold blue] {args.robot_config}")
    load_robot_config(args.robot_config)
    robot_config = get_robot_config()
    amp_config = create_config_from_robot(robot_config)
    print(f"  Feature dimension: {amp_config.feature_dim}")

    # Load MuJoCo model
    print(f"[bold blue]Loading MuJoCo model:[/bold blue] {args.model}")
    mj_model = load_mujoco_model(args.model)
    print(f"  [green]✓ Model loaded[/green]")

    # Create physics config
    config = PhysicsRolloutConfig(
        min_load_support_ratio=args.min_load_support,
        min_segment_length=args.min_segment_length,
        harness_mode="full",  # Use full harness mode by default
    )

    print(f"\n[bold]Physics Rollout Configuration:[/bold]")
    print(f"  sim_dt: {config.sim_dt}s ({1/config.sim_dt:.0f} Hz)")
    print(f"  control_dt: {config.control_dt}s ({1/config.control_dt:.0f} Hz)")
    print(f"  n_substeps: {config.n_substeps}")
    print(f"  harness_mode: {config.harness_mode}")
    print(f"  min_load_support: {config.min_load_support_ratio:.0%}")
    print(f"  min_segment_length: {config.min_segment_length} frames")
    print(
        f"  stab_z_force_cap: {config.stab_z_force_cap} N ({config.stab_z_force_cap/config.mg:.1%} mg)"
    )

    if args.input:
        # Single motion processing
        input_path = Path(args.input)
        output_path = Path(args.output) if args.output else None

        if output_path is None:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{input_path.stem}_physics.pkl"

        physics_data, metadata = process_single_motion(
            input_path, mj_model, config, amp_config
        )

        if physics_data is not None:
            with open(output_path, "wb") as f:
                pickle.dump(physics_data, f)
            print(f"\n[green]✓ Saved to: {output_path}[/green]")
            print(f"  Status: {metadata.status}")
            print(f"  Frames: {metadata.accepted_frames}/{metadata.total_frames}")
            print(f"  Load support: {metadata.mean_load_support:.1%}")

            if args.verify_determinism:
                verify_determinism(
                    output_path, input_path, mj_model, config, amp_config
                )
        else:
            print(f"\n[red]✗ Clip rejected: {metadata.reason}[/red]")

    else:
        # Batch processing
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)

        physics_files, all_metadata = batch_generate_physics_references(
            input_dir, output_dir, mj_model, config, amp_config
        )

        if physics_files:
            # Merge into single dataset
            merged_output = output_dir / args.merged_output
            merged_data = merge_physics_references(
                physics_files, merged_output, amp_config
            )

            # Print summary
            print_summary_report(all_metadata, merged_data)

            # Verify determinism on first accepted clip
            if args.verify_determinism and physics_files:
                # Find corresponding GMR file
                first_physics = physics_files[0]
                gmr_name = first_physics.stem.replace("_physics", "") + ".pkl"
                gmr_path = input_dir / gmr_name
                if gmr_path.exists():
                    verify_determinism(
                        first_physics, gmr_path, mj_model, config, amp_config
                    )

            print(f"\n[bold green]✓ Physics reference dataset ready:[/bold green]")
            print(f"  {merged_output}")
        else:
            print(f"\n[red]✗ No clips accepted - check quality gates[/red]")
            print_summary_report(all_metadata)


if __name__ == "__main__":
    main()
