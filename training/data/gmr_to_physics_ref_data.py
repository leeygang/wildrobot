#!/usr/bin/env python3
"""Tier 0+ Segment Generator - Physics Reference with Strict Harness Caps and Segment Curation.

v0.1.0: Implements Path A from the AMP training plan.

This script generates physics-based reference data with:
1. Capped harness (assist mode with strict limits)
2. Frame-level quality metrics
3. Automatic segmentation to extract ALL Tier 0+ segments
4. Segment-level metadata for tracking

Tier 0+ Quality Gates (for m=3.383kg, mg≈33.2N):
- mean(ΣF_n) / mg >= 0.90  (contact forces carry 90%+ of weight)
- p95(|F_stab|) / mg <= 0.10 (harness contributes ≤10% at p95)
- unloaded_frames <= 5% (frames with ΣF_n < 0.6*mg are rare)
- p95(|pitch|) <= 20° and p95(|roll|) <= 20°

Key difference from original generator:
- Extracts ALL valid segments, not just the longest
- Uses stricter harness caps (10-15% mg)
- Includes pitch/roll stability checks
- Outputs detailed segment-level metadata

Usage:
    cd ~/projects/wildrobot

    # Generate Tier 0+ segments from all motions
    uv run python training/data/gmr_to_physics_ref_data.py

    # With verbose segment info
    uv run python training/data/gmr_to_physics_ref_data.py --verbose

    # Custom harness cap (stricter)
    uv run python training/data/gmr_to_physics_ref_data.py --harness-cap 0.10
"""

import argparse
import json
import pickle
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mujoco
import numpy as np
from rich import print
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training.amp.ref_features import extract_ref_features
from training.configs.feature_config import FeatureConfig, create_config_from_robot
from training.configs.training_config import get_robot_config, load_robot_config

console = Console()


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class GMR2PRDConfig:
    """Configuration for GMR to Physics Reference Data conversion."""

    # Simulation timing
    sim_dt: float = 0.002  # 500 Hz simulation
    control_dt: float = 0.02  # 50 Hz control (matches reference)

    # Robot physics
    robot_mass: float = 3.383  # kg
    gravity: float = 9.81  # m/s²

    # ==========================================================================
    # Harness Configuration - CAPPED ASSIST MODE
    # ==========================================================================
    # Key insight: harness should be an assist, not support
    # We cap at 10-15% mg to ensure feet carry the load
    harness_cap_fraction: float = 0.15  # Cap at 15% of mg (configurable via CLI)

    # Height stabilization (Z-axis) - gentle assist
    stab_z_kp: float = 60.0  # N/m
    stab_z_kd: float = 10.0  # N·s/m

    # XY stabilization - prevent drift, keep on trajectory
    stab_xy_kp: float = 50.0  # N/m
    stab_xy_kd: float = 10.0  # N·s/m
    stab_xy_cap_fraction: float = 0.10  # 10% mg per axis

    # Orientation gate (disable harness if tilted too much)
    stab_orientation_gate: float = 0.436  # radians (25°)

    # ==========================================================================
    # Tier 0+ Quality Thresholds (STRICT)
    # ==========================================================================
    min_load_support: float = 0.90  # mean(ΣF_n)/mg >= 90%
    max_harness_p95: float = 0.10  # p95(|F_stab|)/mg <= 10%
    max_unloaded_rate: float = 0.05  # unloaded frames <= 5%
    max_pitch_p95: float = 0.349  # p95(|pitch|) <= 20°
    max_roll_p95: float = 0.349  # p95(|roll|) <= 20°
    unloaded_threshold: float = 0.6  # Frame is "unloaded" if ΣF_n < 0.6*mg

    # Segment length requirements
    min_segment_frames: int = 25  # Minimum 0.5 seconds at 50Hz
    min_segment_seconds: float = 0.5

    @property
    def mg(self) -> float:
        """Weight force in Newtons."""
        return self.robot_mass * self.gravity

    @property
    def stab_z_cap(self) -> float:
        """Z-axis harness force cap in Newtons."""
        return self.harness_cap_fraction * self.mg

    @property
    def stab_xy_cap(self) -> float:
        """XY-axis harness force cap in Newtons."""
        return self.stab_xy_cap_fraction * self.mg

    @property
    def n_substeps(self) -> int:
        """Number of simulation substeps per control step."""
        return int(self.control_dt / self.sim_dt)


@dataclass
class SegmentMetrics:
    """Metrics for a single segment."""

    # Identity
    clip_name: str
    segment_idx: int
    start_frame: int
    end_frame: int

    # Duration
    num_frames: int
    duration_sec: float

    # Load support metrics
    mean_load_support: float  # mean(ΣF_n) / mg
    min_load_support: float  # min(ΣF_n) / mg
    p5_load_support: float  # 5th percentile load support

    # Harness reliance metrics
    mean_harness: float  # mean(|F_stab|) / mg
    p95_harness: float  # 95th percentile harness
    max_harness: float  # max(|F_stab|) / mg

    # Unloaded frame analysis
    unloaded_frame_rate: float  # fraction of frames with low contact force
    unloaded_frame_count: int

    # Orientation stability
    mean_pitch: float  # mean(|pitch|) in degrees
    mean_roll: float  # mean(|roll|) in degrees
    p95_pitch: float  # 95th percentile pitch in degrees
    p95_roll: float  # 95th percentile roll in degrees
    max_pitch: float  # max(|pitch|) in degrees
    max_roll: float  # max(|roll|) in degrees

    # Contact metrics
    left_contact_rate: float
    right_contact_rate: float
    alternating_contact_rate: float  # frames with exactly one foot down
    both_contact_rate: float  # frames with both feet down
    no_contact_rate: float  # frames with no feet down

    # Slip metrics (if available)
    mean_slip_rate: float = 0.0

    # Tier classification
    is_tier0plus: bool = False
    tier0plus_failures: List[str] = field(default_factory=list)


@dataclass
class ClipSummary:
    """Summary for a processed clip."""

    clip_name: str
    total_frames: int
    tier0plus_segments: int
    tier0plus_frames: int
    tier0plus_seconds: float
    rejected_frames: int
    segments: List[SegmentMetrics] = field(default_factory=list)


# =============================================================================
# Physics Rollout with Frame-Level Metrics
# =============================================================================


def load_mujoco_model(model_path: str) -> mujoco.MjModel:
    """Load MuJoCo model with assets."""
    from training.envs.wildrobot_env import get_assets

    model_path = Path(model_path)
    mj_model = mujoco.MjModel.from_xml_string(
        model_path.read_text(), assets=get_assets(model_path.parent)
    )
    return mj_model


def run_physics_rollout_with_frame_metrics(
    gmr_motion: Dict[str, Any],
    mj_model: mujoco.MjModel,
    config: GMR2PRDConfig,
    amp_config: FeatureConfig,
    start_frame: int = 0,
    max_frames: Optional[int] = None,
) -> Dict[str, Any]:
    """Run physics rollout and collect frame-level quality metrics.

    Uses capped harness (assist mode) to allow feet to naturally support weight.

    Multi-start support: Can start physics from any GMR frame, not just frame 0.

    Args:
        gmr_motion: GMR motion data (root_pos, root_rot, dof_pos)
        mj_model: MuJoCo model
        config: Tier 0+ configuration
        amp_config: AMP feature configuration
        start_frame: GMR frame index to start physics from (default: 0)
        max_frames: Maximum frames to simulate (default: all remaining)

    Returns:
        Dict with trajectories and frame-level metrics
    """
    # Extract GMR data
    ref_dof_pos = gmr_motion["dof_pos"]
    ref_root_pos = gmr_motion["root_pos"]
    ref_root_rot = gmr_motion.get("root_rot")  # Quaternion (N, 4) wxyz
    total_frames = gmr_motion["num_frames"]

    # Calculate frames to simulate
    end_frame = total_frames
    if max_frames is not None:
        end_frame = min(start_frame + max_frames, total_frames)
    num_frames = end_frame - start_frame

    n = amp_config.num_actuated_joints
    mj_data = mujoco.MjData(mj_model)
    mj_model.opt.timestep = config.sim_dt

    # Initialize physics state from GMR pose at start_frame
    if start_frame == 0:
        # Original behavior: use home keyframe
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
        mj_data.qvel[:] = 0
    else:
        # Multi-start: Initialize from GMR pose at start_frame
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)  # Reset to get default structure
        mj_data.qvel[:] = 0

        # Set root position from GMR
        mj_data.qpos[0:3] = ref_root_pos[start_frame]

        # Set root orientation from GMR if available
        if ref_root_rot is not None:
            mj_data.qpos[3:7] = ref_root_rot[start_frame]  # wxyz format

        # Set joint positions from GMR
        mj_data.qpos[7:7+n] = ref_dof_pos[start_frame]

    mujoco.mj_forward(mj_model, mj_data)

    # Target height for stabilization (from reference mean)
    target_height = ref_root_pos[:, 2].mean()

    # Get foot geom IDs (including heel geoms) - raise if not found
    left_toe_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "left_toe")
    if left_toe_id == -1:
        raise ValueError("Geom 'left_toe' not found in model")
    left_heel_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "left_heel")
    if left_heel_id == -1:
        raise ValueError("Geom 'left_heel' not found in model")
    right_toe_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "right_toe")
    if right_toe_id == -1:
        raise ValueError("Geom 'right_toe' not found in model")
    right_heel_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "right_heel")
    if right_heel_id == -1:
        raise ValueError("Geom 'right_heel' not found in model")
    foot_geom_ids = [left_toe_id, left_heel_id, right_toe_id, right_heel_id]
    ground_id = 0

    # Track reference XY position for drift prevention
    ref_xy_start = ref_root_pos[0, :2].copy()
    sim_xy_start = mj_data.qpos[0:2].copy()

    # Storage for trajectories and frame metrics
    qpos_trajectory = []
    qvel_trajectory = []
    root_pos_trajectory = []
    root_rot_trajectory = []
    contact_states = []

    # Frame-level metrics for quality gating
    frame_sum_fn = []  # Total normal force per frame
    frame_stab_force = []  # Total harness force per frame
    frame_pitch = []  # Pitch angle per frame
    frame_roll = []  # Roll angle per frame
    frame_left_contact = []  # Left foot contact per frame
    frame_right_contact = []  # Right foot contact per frame

    mg = config.mg

    for i in range(num_frames):
        # Set control targets (reference joint positions)
        mj_data.ctrl[:n] = ref_dof_pos[i]

        # Compute current orientation
        w, x, y, z = mj_data.qpos[3:7]
        sinp = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
        curr_pitch = np.arcsin(sinp)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        curr_roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Record orientation
        frame_pitch.append(abs(curr_pitch))
        frame_roll.append(abs(curr_roll))

        # =================================================================
        # CAPPED ASSIST HARNESS
        # Only active when upright, with strict force caps
        # =================================================================
        stab_force_z = 0.0
        stab_force_xy = np.zeros(2)

        # Only apply harness if upright (orientation gate)
        if (
            abs(curr_pitch) < config.stab_orientation_gate
            and abs(curr_roll) < config.stab_orientation_gate
        ):
            # Height (Z) stabilization - gentle assist with hard cap
            height_error = target_height - mj_data.qpos[2]
            stab_force_z = (
                config.stab_z_kp * height_error - config.stab_z_kd * mj_data.qvel[2]
            )
            stab_force_z = np.clip(stab_force_z, -config.stab_z_cap, config.stab_z_cap)

            # XY stabilization - keep on trajectory with hard cap
            ref_xy_delta = ref_root_pos[i, :2] - ref_xy_start
            target_xy = sim_xy_start + ref_xy_delta
            xy_error = target_xy - mj_data.qpos[0:2]
            stab_force_xy = (
                config.stab_xy_kp * xy_error - config.stab_xy_kd * mj_data.qvel[0:2]
            )
            stab_force_xy = np.clip(
                stab_force_xy, -config.stab_xy_cap, config.stab_xy_cap
            )

        # Apply forces
        mj_data.qfrc_applied[0:2] = stab_force_xy
        mj_data.qfrc_applied[2] = stab_force_z
        mj_data.qfrc_applied[3:6] = 0.0  # No torque assist

        # Record total harness force magnitude
        total_stab = np.sqrt(stab_force_z**2 + np.sum(stab_force_xy**2))
        frame_stab_force.append(total_stab)

        # Step physics
        for _ in range(config.n_substeps):
            mujoco.mj_step(mj_model, mj_data)

        # Extract contact normal forces using mj_contactForce (correct API)
        left_fn, right_fn = 0.0, 0.0
        wrench = np.zeros(6)  # Reusable buffer for contact wrench
        for contact_idx in range(mj_data.ncon):
            contact = mj_data.contact[contact_idx]
            geom1, geom2 = contact.geom1, contact.geom2

            foot_geom = None
            if geom1 == ground_id and geom2 in foot_geom_ids:
                foot_geom = geom2
            elif geom2 == ground_id and geom1 in foot_geom_ids:
                foot_geom = geom1

            if foot_geom is not None:
                # Use mj_contactForce to get proper 6D contact wrench
                mujoco.mj_contactForce(mj_model, mj_data, contact_idx, wrench)
                fn = wrench[0]  # Normal force is first element
                if fn > 0:
                    if foot_geom in [left_toe_id, left_heel_id]:
                        left_fn += fn
                    else:
                        right_fn += fn

        # Record contact forces
        sum_fn = left_fn + right_fn
        frame_sum_fn.append(sum_fn)

        contact_threshold = 0.5  # N
        l_contact = left_fn > contact_threshold
        r_contact = right_fn > contact_threshold
        frame_left_contact.append(1.0 if l_contact else 0.0)
        frame_right_contact.append(1.0 if r_contact else 0.0)

        # Extract realized state
        qpos = mj_data.qpos.copy()
        qvel = mj_data.qvel.copy()

        # Convert root quaternion to xyzw format
        root_pos = qpos[0:3].copy()
        root_quat_wxyz = qpos[3:7].copy()
        root_quat_xyzw = np.array(
            [root_quat_wxyz[1], root_quat_wxyz[2], root_quat_wxyz[3], root_quat_wxyz[0]]
        )

        qpos_trajectory.append(qpos)
        qvel_trajectory.append(qvel)
        root_pos_trajectory.append(root_pos)
        root_rot_trajectory.append(root_quat_xyzw)

        # Contact states (simplified: toe only, duplicated for heel)
        contact_states.append(
            [
                1.0 if l_contact else 0.0,
                1.0 if l_contact else 0.0,
                1.0 if r_contact else 0.0,
                1.0 if r_contact else 0.0,
            ]
        )

    # Convert to arrays
    return {
        "qpos_trajectory": np.array(qpos_trajectory),
        "qvel_trajectory": np.array(qvel_trajectory),
        "root_pos_trajectory": np.array(root_pos_trajectory),
        "root_rot_trajectory": np.array(root_rot_trajectory),
        "contact_states": np.array(contact_states),
        "num_frames": num_frames,
        # Frame-level metrics
        "frame_sum_fn": np.array(frame_sum_fn),
        "frame_stab_force": np.array(frame_stab_force),
        "frame_pitch": np.array(frame_pitch),
        "frame_roll": np.array(frame_roll),
        "frame_left_contact": np.array(frame_left_contact),
        "frame_right_contact": np.array(frame_right_contact),
    }


# =============================================================================
# Segment Identification and Quality Gating
# =============================================================================


def compute_frame_tier0plus_mask(
    rollout: Dict[str, Any],
    config: GMR2PRDConfig,
) -> np.ndarray:
    """Compute per-frame mask for Tier 0+ eligibility.

    A frame is Tier 0+ eligible if:
    - load_support >= 0.6 (not unloaded)
    - harness <= cap (not over-relying on harness)
    - pitch and roll < 30° (not falling over)

    Note: Full Tier 0+ classification requires segment-level statistics.

    Args:
        rollout: Physics rollout data
        config: Tier 0+ configuration

    Returns:
        Boolean mask (N,) where True = frame is Tier 0+ eligible
    """
    mg = config.mg

    # Load support check (not unloaded)
    load_ratio = rollout["frame_sum_fn"] / mg
    load_ok = load_ratio >= config.unloaded_threshold

    # Harness check (within cap)
    harness_ratio = rollout["frame_stab_force"] / mg
    harness_ok = harness_ratio <= config.harness_cap_fraction * 1.5  # Allow 1.5x for transients

    # Orientation check (not falling)
    pitch_ok = rollout["frame_pitch"] < 0.524  # 30°
    roll_ok = rollout["frame_roll"] < 0.524  # 30°

    return load_ok & harness_ok & pitch_ok & roll_ok


def find_contiguous_segments(
    mask: np.ndarray,
    min_length: int,
) -> List[Tuple[int, int]]:
    """Find contiguous True segments in boolean mask.

    Args:
        mask: Boolean array
        min_length: Minimum segment length

    Returns:
        List of (start, end) tuples for each valid segment
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
            if i - start >= min_length:
                segments.append((start, i))

    if in_segment and len(mask) - start >= min_length:
        segments.append((start, len(mask)))

    return segments


def compute_segment_metrics(
    rollout: Dict[str, Any],
    start: int,
    end: int,
    clip_name: str,
    segment_idx: int,
    config: GMR2PRDConfig,
) -> SegmentMetrics:
    """Compute detailed metrics for a segment.

    Args:
        rollout: Full clip rollout data
        start: Start frame index
        end: End frame index
        clip_name: Source clip name
        segment_idx: Segment index within clip
        config: Tier 0+ configuration

    Returns:
        SegmentMetrics with quality classification
    """
    mg = config.mg
    num_frames = end - start
    duration_sec = num_frames * config.control_dt

    # Extract segment data
    sum_fn = rollout["frame_sum_fn"][start:end]
    stab_force = rollout["frame_stab_force"][start:end]
    pitch = rollout["frame_pitch"][start:end]
    roll = rollout["frame_roll"][start:end]
    left_contact = rollout["frame_left_contact"][start:end]
    right_contact = rollout["frame_right_contact"][start:end]

    # Load support metrics
    load_ratio = sum_fn / mg
    mean_load_support = float(load_ratio.mean())
    min_load_support = float(load_ratio.min())
    p5_load_support = float(np.percentile(load_ratio, 5))

    # Harness reliance metrics
    harness_ratio = stab_force / mg
    mean_harness = float(harness_ratio.mean())
    p95_harness = float(np.percentile(harness_ratio, 95))
    max_harness = float(harness_ratio.max())

    # Unloaded frame analysis
    unloaded_mask = load_ratio < config.unloaded_threshold
    unloaded_frame_count = int(unloaded_mask.sum())
    unloaded_frame_rate = float(unloaded_frame_count / num_frames)

    # Orientation stability (convert to degrees for readability)
    mean_pitch = float(np.degrees(pitch.mean()))
    mean_roll = float(np.degrees(roll.mean()))
    p95_pitch = float(np.degrees(np.percentile(pitch, 95)))
    p95_roll = float(np.degrees(np.percentile(roll, 95)))
    max_pitch = float(np.degrees(pitch.max()))
    max_roll = float(np.degrees(roll.max()))

    # Contact metrics
    left_contact_rate = float(left_contact.mean())
    right_contact_rate = float(right_contact.mean())
    both_mask = (left_contact > 0.5) & (right_contact > 0.5)
    none_mask = (left_contact < 0.5) & (right_contact < 0.5)
    one_mask = ~both_mask & ~none_mask
    both_contact_rate = float(both_mask.mean())
    no_contact_rate = float(none_mask.mean())
    alternating_contact_rate = float(one_mask.mean())

    # Tier 0+ classification
    tier0plus_failures = []

    if mean_load_support < config.min_load_support:
        tier0plus_failures.append(
            f"load_support={mean_load_support:.1%}<{config.min_load_support:.0%}"
        )

    if p95_harness > config.max_harness_p95:
        tier0plus_failures.append(
            f"p95_harness={p95_harness:.1%}>{config.max_harness_p95:.0%}"
        )

    if unloaded_frame_rate > config.max_unloaded_rate:
        tier0plus_failures.append(
            f"unloaded={unloaded_frame_rate:.1%}>{config.max_unloaded_rate:.0%}"
        )

    if p95_pitch > np.degrees(config.max_pitch_p95):
        tier0plus_failures.append(
            f"p95_pitch={p95_pitch:.1f}°>{np.degrees(config.max_pitch_p95):.0f}°"
        )

    if p95_roll > np.degrees(config.max_roll_p95):
        tier0plus_failures.append(
            f"p95_roll={p95_roll:.1f}°>{np.degrees(config.max_roll_p95):.0f}°"
        )

    is_tier0plus = len(tier0plus_failures) == 0

    return SegmentMetrics(
        clip_name=clip_name,
        segment_idx=segment_idx,
        start_frame=start,
        end_frame=end,
        num_frames=num_frames,
        duration_sec=duration_sec,
        mean_load_support=mean_load_support,
        min_load_support=min_load_support,
        p5_load_support=p5_load_support,
        mean_harness=mean_harness,
        p95_harness=p95_harness,
        max_harness=max_harness,
        unloaded_frame_rate=unloaded_frame_rate,
        unloaded_frame_count=unloaded_frame_count,
        mean_pitch=mean_pitch,
        mean_roll=mean_roll,
        p95_pitch=p95_pitch,
        p95_roll=p95_roll,
        max_pitch=max_pitch,
        max_roll=max_roll,
        left_contact_rate=left_contact_rate,
        right_contact_rate=right_contact_rate,
        alternating_contact_rate=alternating_contact_rate,
        both_contact_rate=both_contact_rate,
        no_contact_rate=no_contact_rate,
        is_tier0plus=is_tier0plus,
        tier0plus_failures=tier0plus_failures,
    )


def find_tier0plus_segments(
    rollout: Dict[str, Any],
    clip_name: str,
    config: GMR2PRDConfig,
) -> Tuple[List[SegmentMetrics], List[Tuple[int, int]]]:
    """Find all Tier 0+ segments in a rollout.

    Strategy:
    1. Compute frame-level eligibility mask
    2. Find contiguous eligible segments
    3. Compute segment-level metrics
    4. Filter to Tier 0+ segments

    Args:
        rollout: Physics rollout data
        clip_name: Source clip name
        config: Tier 0+ configuration

    Returns:
        Tuple of (list of segment metrics, list of (start, end) for Tier 0+ segments)
    """
    # Get frame-level eligibility
    frame_mask = compute_frame_tier0plus_mask(rollout, config)

    # Find contiguous segments
    candidate_segments = find_contiguous_segments(frame_mask, config.min_segment_frames)

    all_segments = []
    tier0plus_ranges = []

    for idx, (start, end) in enumerate(candidate_segments):
        metrics = compute_segment_metrics(
            rollout, start, end, clip_name, idx, config
        )
        all_segments.append(metrics)

        if metrics.is_tier0plus:
            tier0plus_ranges.append((start, end))

    return all_segments, tier0plus_ranges


# =============================================================================
# AMP Feature Extraction
# =============================================================================


def world_to_heading_local(
    vectors: np.ndarray,
    quats_xyzw: np.ndarray,
) -> np.ndarray:
    """Convert world-frame vectors to heading-local frame (yaw-removed)."""
    x, y, z, w = quats_xyzw[:, 0], quats_xyzw[:, 1], quats_xyzw[:, 2], quats_xyzw[:, 3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    vx, vy, vz = vectors[:, 0], vectors[:, 1], vectors[:, 2]
    vx_local = cos_yaw * vx + sin_yaw * vy
    vy_local = -sin_yaw * vx + cos_yaw * vy
    vz_local = vz

    return np.stack([vx_local, vy_local, vz_local], axis=1)


def extract_amp_features(
    rollout: Dict[str, Any],
    amp_config: FeatureConfig,
) -> np.ndarray:
    """Extract AMP features from physics rollout data.

    Uses extract_ref_features from ref_features.py for consistent feature ordering.
    """
    n = amp_config.num_actuated_joints

    qpos = rollout["qpos_trajectory"]
    qvel = rollout["qvel_trajectory"]
    root_pos = rollout["root_pos_trajectory"]
    root_rot = rollout["root_rot_trajectory"]
    contact_states = rollout["contact_states"]

    # Joint positions and velocities
    joint_pos = qpos[:, 7 : 7 + n]
    joint_vel = qvel[:, 6 : 6 + n]

    # Root velocities (world frame from qvel)
    root_lin_vel_world = qvel[:, 0:3]
    root_ang_vel_world = qvel[:, 3:6]

    # Convert to heading-local frame
    root_lin_vel_heading = world_to_heading_local(root_lin_vel_world, root_rot)
    root_ang_vel_heading = world_to_heading_local(root_ang_vel_world, root_rot)

    # Root height
    root_height = root_pos[:, 2]

    # Use extract_ref_features for consistent feature ordering
    return extract_ref_features(
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        root_linvel=root_lin_vel_heading,
        root_angvel=root_ang_vel_heading,
        root_height=root_height,
        foot_contacts=contact_states,
        config=amp_config,
    )


# =============================================================================
# Main Processing Pipeline
# =============================================================================


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


def process_clip(
    gmr_path: Path,
    mj_model: mujoco.MjModel,
    config: GMR2PRDConfig,
    amp_config: FeatureConfig,
    verbose: bool = False,
    multi_start_interval: int = 0,  # 0 = disabled, >0 = start interval in frames
    max_window_frames: Optional[int] = None,  # Maximum frames per window
) -> Tuple[Optional[Dict[str, Any]], ClipSummary]:
    """Process a single GMR motion clip through physics rollout and segmentation.

    Multi-start mode: When multi_start_interval > 0, runs physics from multiple
    starting frames (0, interval, 2*interval, ...) to capture segments that would
    otherwise be lost due to tracking divergence.

    Args:
        gmr_path: Path to GMR pickle file
        mj_model: MuJoCo model
        config: Tier 0+ configuration
        amp_config: AMP feature configuration
        verbose: Print detailed segment info
        multi_start_interval: Frame interval between starting points (0 = disabled)
        max_window_frames: Maximum frames per window (None = unlimited)

    Returns:
        Tuple of (tier0plus_data, clip_summary)
        tier0plus_data contains features for all Tier 0+ segments
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

    total_gmr_frames = gmr_motion["num_frames"]

    # Determine starting frames
    if multi_start_interval > 0:
        # Multi-start mode: start from multiple frames
        start_frames = list(range(0, total_gmr_frames, multi_start_interval))
        if verbose:
            print(f"\n  {clip_name}: Multi-start mode ({len(start_frames)} windows)")
    else:
        # Single rollout from frame 0
        start_frames = [0]

    # Collect all segments from all rollouts
    all_segments_collected = []
    all_tier0plus_ranges = []  # List of (global_start, global_end, rollout_data, local_start, local_end)

    for window_idx, start_frame in enumerate(start_frames):
        # Run physics rollout from this starting frame
        rollout = run_physics_rollout_with_frame_metrics(
            gmr_motion, mj_model, config, amp_config,
            start_frame=start_frame,
            max_frames=max_window_frames,
        )

        # Find Tier 0+ segments in this window
        window_segments, window_tier0plus = find_tier0plus_segments(
            rollout, clip_name, config
        )

        # Convert local frame indices to global GMR frame indices
        for seg in window_segments:
            # Adjust frame indices to global coordinates
            seg.start_frame += start_frame
            seg.end_frame += start_frame
            seg.segment_idx = len(all_segments_collected)
            all_segments_collected.append(seg)

        for local_start, local_end in window_tier0plus:
            global_start = start_frame + local_start
            global_end = start_frame + local_end
            # Store with rollout data for feature extraction
            all_tier0plus_ranges.append((global_start, global_end, rollout, local_start, local_end))

        if verbose and multi_start_interval > 0:
            window_t0plus = len(window_tier0plus)
            window_frames = rollout["num_frames"]
            t0plus_frames = sum(end - start for start, end in window_tier0plus)
            print(f"    Window {window_idx} (frame {start_frame}): {window_frames} frames, {window_t0plus} T0+ segments ({t0plus_frames} frames)")

    # Remove overlapping segments (keep the one with better metrics)
    unique_ranges = _deduplicate_segments(all_tier0plus_ranges, config)

    # Create clip summary
    tier0plus_frames = sum(gend - gstart for gstart, gend, _, _, _ in unique_ranges)
    tier0plus_seconds = tier0plus_frames * config.control_dt

    # Filter all_segments_collected to only include Tier 0+ segments
    tier0plus_segments_only = [s for s in all_segments_collected if s.is_tier0plus]

    summary = ClipSummary(
        clip_name=clip_name,
        total_frames=total_gmr_frames,
        tier0plus_segments=len(unique_ranges),
        tier0plus_frames=tier0plus_frames,
        tier0plus_seconds=tier0plus_seconds,
        rejected_frames=total_gmr_frames - tier0plus_frames,
        segments=tier0plus_segments_only,  # Only include Tier 0+ segments
    )

    if verbose:
        print(f"\n  {clip_name} Summary:")
        print(f"    Total GMR frames: {total_gmr_frames}")
        print(f"    Windows processed: {len(start_frames)}")
        print(f"    Tier 0+ segments: {len(unique_ranges)}")
        print(f"    Tier 0+ frames: {tier0plus_frames} ({tier0plus_seconds:.2f}s)")

        for seg in tier0plus_segments_only:
            print(
                f"      Segment {seg.segment_idx}: [{seg.start_frame}:{seg.end_frame}] "
                f"({seg.duration_sec:.2f}s) load={seg.mean_load_support:.0%} harness_p95={seg.p95_harness:.0%}"
            )

    if not unique_ranges:
        return None, summary

    # Extract AMP features for Tier 0+ segments
    tier0plus_features = []
    tier0plus_metadata = []

    for global_start, global_end, rollout, local_start, local_end in unique_ranges:
        # Extract features from the rollout that contains this segment
        all_features = extract_amp_features(rollout, amp_config)
        tier0plus_features.append(all_features[local_start:local_end])
        tier0plus_metadata.append({
            "clip_name": clip_name,
            "start_frame": global_start,
            "end_frame": global_end,
            "num_frames": global_end - global_start,
        })

    tier0plus_data = {
        "features": np.concatenate(tier0plus_features, axis=0).astype(np.float32),
        "segment_metadata": tier0plus_metadata,
        "clip_name": clip_name,
        "num_segments": len(unique_ranges),
        "total_frames": sum(gend - gstart for gstart, gend, _, _, _ in unique_ranges),
        "duration_sec": sum(gend - gstart for gstart, gend, _, _, _ in unique_ranges) * config.control_dt,
    }

    return tier0plus_data, summary


def _deduplicate_segments(
    segments: List[Tuple[int, int, Dict, int, int]],
    config: GMR2PRDConfig,
) -> List[Tuple[int, int, Dict, int, int]]:
    """Remove overlapping segments, keeping the one with more frames.

    When using multi-start, segments from different windows may overlap.
    This function removes duplicates by keeping the longer segment.

    Args:
        segments: List of (global_start, global_end, rollout, local_start, local_end)
        config: Tier 0+ configuration

    Returns:
        Deduplicated list of segments
    """
    if len(segments) <= 1:
        return segments

    # Sort by global start frame
    sorted_segs = sorted(segments, key=lambda x: x[0])

    result = []
    current = sorted_segs[0]

    for next_seg in sorted_segs[1:]:
        curr_start, curr_end = current[0], current[1]
        next_start, next_end = next_seg[0], next_seg[1]

        # Check for overlap
        if next_start < curr_end:
            # Overlap detected - keep the longer one
            curr_len = curr_end - curr_start
            next_len = next_end - next_start

            if next_len > curr_len:
                current = next_seg
            # else keep current (it's longer or equal)
        else:
            # No overlap - add current to result and move to next
            result.append(current)
            current = next_seg

    # Add the last segment
    result.append(current)

    return result


def batch_generate_tier0plus(
    input_dir: Path,
    output_dir: Path,
    mj_model: mujoco.MjModel,
    config: GMR2PRDConfig,
    amp_config: FeatureConfig,
    verbose: bool = False,
    multi_start_interval: int = 0,
    max_window_frames: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[ClipSummary]]:
    """Batch process all GMR motions and extract Tier 0+ segments.

    Args:
        input_dir: Directory containing GMR .pkl files
        output_dir: Directory for output
        mj_model: MuJoCo model
        config: Tier 0+ configuration
        amp_config: AMP feature configuration
        verbose: Print detailed output
        multi_start_interval: Frame interval between physics starting points (0 = disabled)
        max_window_frames: Maximum frames per physics window (None = unlimited)

    Returns:
        Tuple of (list of tier0plus data dicts, list of clip summaries)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    motion_files = sorted(input_dir.glob("*.pkl"))
    all_tier0plus_data = []
    all_summaries = []

    print(
        f"\n[bold blue]Processing {len(motion_files)} motions for Tier 0+ segments...[/bold blue]"
    )
    print(f"  Harness cap: {config.harness_cap_fraction:.0%} mg")
    if multi_start_interval > 0:
        print(f"  Multi-start mode: interval={multi_start_interval} frames ({multi_start_interval * config.control_dt:.1f}s)")
        if max_window_frames:
            print(f"    Max window: {max_window_frames} frames ({max_window_frames * config.control_dt:.1f}s)")
    print(f"  Tier 0+ thresholds:")
    print(f"    mean(ΣFn)/mg >= {config.min_load_support:.0%}")
    print(f"    p95(|F_stab|)/mg <= {config.max_harness_p95:.0%}")
    print(f"    unloaded_frames <= {config.max_unloaded_rate:.0%}")
    print(f"    p95(|pitch|) <= {np.degrees(config.max_pitch_p95):.0f}°")
    print(f"    p95(|roll|) <= {np.degrees(config.max_roll_p95):.0f}°")

    for motion_path in motion_files:
        try:
            tier0plus_data, summary = process_clip(
                motion_path, mj_model, config, amp_config,
                verbose=verbose,
                multi_start_interval=multi_start_interval,
                max_window_frames=max_window_frames,
            )

            all_summaries.append(summary)

            if tier0plus_data is not None:
                all_tier0plus_data.append(tier0plus_data)

                # Save individual clip data
                clip_output = output_dir / f"{motion_path.stem}_tier0plus.pkl"
                with open(clip_output, "wb") as f:
                    pickle.dump(tier0plus_data, f)

        except Exception as e:
            print(f"  [red]ERROR processing {motion_path.name}: {e}[/red]")
            import traceback
            traceback.print_exc()

    return all_tier0plus_data, all_summaries


def merge_tier0plus_dataset(
    tier0plus_data_list: List[Dict[str, Any]],
    output_path: Path,
    amp_config: FeatureConfig,
) -> Dict[str, Any]:
    """Merge all Tier 0+ segments into a single training dataset.

    Args:
        tier0plus_data_list: List of tier0plus data dicts from process_clip
        output_path: Output path for merged dataset
        amp_config: AMP feature configuration

    Returns:
        Merged dataset dictionary
    """
    if not tier0plus_data_list:
        return None

    all_features = []
    all_metadata = []
    source_clips = []
    total_duration = 0.0

    for data in tier0plus_data_list:
        all_features.append(data["features"])
        all_metadata.extend(data["segment_metadata"])
        source_clips.append(data["clip_name"])
        total_duration += data["duration_sec"]

    merged = {
        "features": np.concatenate(all_features, axis=0).astype(np.float32),
        "feature_dim": amp_config.feature_dim,
        "segment_metadata": all_metadata,
        "source_clips": source_clips,
        "num_clips": len(tier0plus_data_list),
        "num_segments": len(all_metadata),
        "num_frames": sum(len(f) for f in all_features),
        "duration_sec": total_duration,
        "fps": 50.0,
        "dt": 0.02,
        "tier": "0+",
        "contact_method": "physics",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(merged, f)

    return merged


def print_summary_report(
    summaries: List[ClipSummary],
    merged_data: Optional[Dict[str, Any]] = None,
    config: GMR2PRDConfig = None,
):
    """Print summary report of Tier 0+ segment generation."""
    print(f"\n{'='*70}")
    print("[bold]TIER 0+ SEGMENT GENERATION REPORT[/bold]")
    print(f"{'='*70}")

    # Per-clip table
    table = Table(title="Per-Clip Summary")
    table.add_column("Clip", style="cyan")
    table.add_column("Total\nFrames", justify="right")
    table.add_column("T0+\nSegs", justify="right")
    table.add_column("T0+\nFrames", justify="right")
    table.add_column("T0+\nSeconds", justify="right")
    table.add_column("T0+\nRate", justify="right")
    table.add_column("Status")

    total_frames = 0
    total_t0plus_frames = 0
    total_t0plus_seconds = 0.0

    for summary in summaries:
        total_frames += summary.total_frames
        total_t0plus_frames += summary.tier0plus_frames
        total_t0plus_seconds += summary.tier0plus_seconds

        t0plus_rate = summary.tier0plus_frames / summary.total_frames if summary.total_frames > 0 else 0

        if summary.tier0plus_segments > 0:
            status = f"[green]✓ {summary.tier0plus_segments} segments[/green]"
        else:
            status = "[red]✗ No T0+ segments[/red]"

        table.add_row(
            summary.clip_name,
            str(summary.total_frames),
            str(summary.tier0plus_segments),
            str(summary.tier0plus_frames),
            f"{summary.tier0plus_seconds:.2f}",
            f"{t0plus_rate:.1%}",
            status,
        )

    # Totals row
    total_rate = total_t0plus_frames / total_frames if total_frames > 0 else 0
    table.add_row(
        "[bold]TOTAL[/bold]",
        f"[bold]{total_frames}[/bold]",
        f"[bold]{sum(s.tier0plus_segments for s in summaries)}[/bold]",
        f"[bold]{total_t0plus_frames}[/bold]",
        f"[bold]{total_t0plus_seconds:.2f}[/bold]",
        f"[bold]{total_rate:.1%}[/bold]",
        "",
    )

    console.print(table)

    # Key decision metrics
    print(f"\n[bold]KEY DECISION METRICS:[/bold]")
    print(f"  Total Tier 0+ duration: [bold cyan]{total_t0plus_seconds:.1f} seconds[/bold cyan]")

    if total_t0plus_seconds >= 60:
        print(f"  [bold green]✓ SUFFICIENT for AMP training (>= 60s)[/bold green]")
        print(f"    Proceed with micro-train smoke test")
    elif total_t0plus_seconds >= 30:
        print(f"  [bold yellow]⚠ BORDERLINE for AMP training (30-60s)[/bold yellow]")
        print(f"    Can attempt training, but may overfit")
    else:
        print(f"  [bold red]✗ INSUFFICIENT for AMP training (< 30s)[/bold red]")
        print(f"    Consider: baseline RL walking first, then add AMP")

    # Harness reliance summary
    print(f"\n[bold]HARNESS RELIANCE DISTRIBUTION:[/bold]")
    all_p95_harness = []
    all_mean_harness = []
    for summary in summaries:
        for seg in summary.segments:
            if seg.is_tier0plus:
                all_p95_harness.append(seg.p95_harness)
                all_mean_harness.append(seg.mean_harness)

    if all_p95_harness:
        print(f"  Mean of p95(|F_stab|)/mg across T0+ segments: {np.mean(all_p95_harness):.1%}")
        print(f"  Max of p95(|F_stab|)/mg across T0+ segments: {np.max(all_p95_harness):.1%}")
        print(f"  Mean harness reliance: {np.mean(all_mean_harness):.1%}")
    else:
        print(f"  [dim]No Tier 0+ segments to analyze[/dim]")

    # Contact pattern summary
    print(f"\n[bold]CONTACT PATTERN SUMMARY:[/bold]")
    alt_rates = []
    for summary in summaries:
        for seg in summary.segments:
            if seg.is_tier0plus:
                alt_rates.append(seg.alternating_contact_rate)

    if alt_rates:
        print(f"  Mean alternating contact rate: {np.mean(alt_rates):.1%}")
        print(f"  (Higher = more walking-like gait)")
    else:
        print(f"  [dim]No Tier 0+ segments to analyze[/dim]")

    # Merged dataset info
    if merged_data:
        print(f"\n[bold]MERGED DATASET:[/bold]")
        print(f"  Total frames: {merged_data['num_frames']:,}")
        print(f"  Total duration: {merged_data['duration_sec']:.2f}s")
        print(f"  Number of segments: {merged_data['num_segments']}")
        print(f"  Source clips: {merged_data['num_clips']}")
        print(f"  Feature dimension: {merged_data['feature_dim']}")


def save_segment_metrics_json(
    summaries: List[ClipSummary],
    output_path: Path,
):
    """Save detailed segment metrics to JSON for analysis."""
    data = {
        "clips": [],
        "summary": {
            "total_clips": len(summaries),
            "total_frames": sum(s.total_frames for s in summaries),
            "tier0plus_segments": sum(s.tier0plus_segments for s in summaries),
            "tier0plus_frames": sum(s.tier0plus_frames for s in summaries),
            "tier0plus_seconds": sum(s.tier0plus_seconds for s in summaries),
        }
    }

    for summary in summaries:
        clip_data = {
            "clip_name": summary.clip_name,
            "total_frames": summary.total_frames,
            "tier0plus_segments": summary.tier0plus_segments,
            "tier0plus_frames": summary.tier0plus_frames,
            "tier0plus_seconds": summary.tier0plus_seconds,
            "segments": []
        }

        for seg in summary.segments:
            seg_dict = asdict(seg)
            clip_data["segments"].append(seg_dict)

        data["clips"].append(clip_data)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n[dim]Segment metrics saved to: {output_path}[/dim]")


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Generate Physics Based segments from GMR motions"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="training/data/gmr",
        help="Directory containing GMR .pkl files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="training/data/pbrd",
        help="Output directory for Tier 0+ physics-based reference data",
    )
    parser.add_argument(
        "--merged-output",
        type=str,
        default="physics_validated_merged.pkl",
        help="Filename for merged dataset",
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
    # Tier 0+ thresholds
    parser.add_argument(
        "--harness-cap",
        type=float,
        default=0.15,
        help="Harness force cap as fraction of mg (default: 0.15)",
    )
    parser.add_argument(
        "--min-load-support",
        type=float,
        default=0.90,
        help="Minimum mean load support ratio (default: 0.90)",
    )
    parser.add_argument(
        "--max-harness-p95",
        type=float,
        default=0.10,
        help="Maximum p95 harness reliance (default: 0.10)",
    )
    parser.add_argument(
        "--max-unloaded-rate",
        type=float,
        default=0.05,
        help="Maximum unloaded frame rate (default: 0.05)",
    )
    parser.add_argument(
        "--max-pitch-p95",
        type=float,
        default=20.0,
        help="Maximum p95 pitch angle in degrees (default: 20.0)",
    )
    parser.add_argument(
        "--max-roll-p95",
        type=float,
        default=20.0,
        help="Maximum p95 roll angle in degrees (default: 20.0)",
    )
    parser.add_argument(
        "--min-segment-seconds",
        type=float,
        default=0.5,
        help="Minimum segment duration in seconds (default: 0.5)",
    )
    # Multi-start options
    parser.add_argument(
        "--multi-start-interval",
        type=int,
        default=0,
        help="Frame interval between physics starting points (0 = disabled, 50 = 1 second intervals)",
    )
    parser.add_argument(
        "--max-window-frames",
        type=int,
        default=None,
        help="Maximum frames per physics window (default: unlimited)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed segment info",
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

    # Create configuration
    config = GMR2PRDConfig(
        harness_cap_fraction=args.harness_cap,
        min_load_support=args.min_load_support,
        max_harness_p95=args.max_harness_p95,
        max_unloaded_rate=args.max_unloaded_rate,
        max_pitch_p95=np.radians(args.max_pitch_p95),
        max_roll_p95=np.radians(args.max_roll_p95),
        min_segment_seconds=args.min_segment_seconds,
        min_segment_frames=int(args.min_segment_seconds / 0.02),
    )

    # Process all clips
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    tier0plus_data_list, summaries = batch_generate_tier0plus(
        input_dir, output_dir, mj_model, config, amp_config,
        verbose=args.verbose,
        multi_start_interval=args.multi_start_interval,
        max_window_frames=args.max_window_frames,
    )

    # Merge into single dataset
    merged_data = None
    if tier0plus_data_list:
        merged_output = output_dir / args.merged_output
        merged_data = merge_tier0plus_dataset(
            tier0plus_data_list, merged_output, amp_config
        )
        print(f"\n[bold green]✓ Merged dataset saved:[/bold green] {merged_output}")

    # Save segment metrics JSON
    metrics_output = output_dir / "segment_metrics.json"
    save_segment_metrics_json(summaries, metrics_output)

    # Print summary report
    print_summary_report(summaries, merged_data, config)

    # Exit code based on sufficiency
    total_t0plus_seconds = sum(s.tier0plus_seconds for s in summaries)
    if total_t0plus_seconds >= 30:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
