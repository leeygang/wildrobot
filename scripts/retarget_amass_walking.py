#!/usr/bin/env python3
"""Batch retarget AMASS walking motions to WildRobot skeleton.

Usage:
    python scripts/retarget_amass_walking.py --amass-dir /path/to/amass --output data/walking_motions.pkl
"""

from __future__ import annotations

import argparse
import pickle
from glob import glob
from pathlib import Path
from typing import List, Tuple

import numpy as np


# WildRobot joint order (matching actuator order in XML)
WILDROBOT_JOINTS = [
    "right_hip_pitch",
    "right_hip_roll",
    "right_knee_pitch",
    "right_ankle_pitch",
    "left_hip_pitch",
    "left_hip_roll",
    "left_knee_pitch",
    "left_ankle_pitch",
    "waist_yaw",
]

# Joint limits from the XML (radians)
JOINT_LIMITS = {
    "right_hip_pitch": (-1.571, 0.087),
    "right_hip_roll": (-0.175, 1.571),
    "right_knee_pitch": (0.0, 1.396),
    "right_ankle_pitch": (-0.785, 0.785),
    "left_hip_pitch": (-0.188, 1.470),
    "left_hip_roll": (-1.634, 0.111),
    "left_knee_pitch": (-0.208, 1.188),
    "left_ankle_pitch": (-0.785, 0.785),
    "waist_yaw": (-0.524, 0.524),
}

# SMPL joint indices (standard SMPL with 24 joints, or SMPL-H with 52)
# SMPL: 0=pelvis, 1=left_hip, 2=right_hip, 3=spine1, 4=left_knee, 5=right_knee,
#       6=spine2, 7=left_ankle, 8=right_ankle, 9=spine3, 10=left_foot, 11=right_foot, ...
SMPL_JOINTS = {
    "pelvis": 0,
    "left_hip": 1,
    "right_hip": 2,
    "spine1": 3,
    "left_knee": 4,
    "right_knee": 5,
    "spine2": 6,
    "left_ankle": 7,
    "right_ankle": 8,
    "spine3": 9,
}


def axis_angle_to_euler(axis_angle: np.ndarray) -> Tuple[float, float, float]:
    """Convert axis-angle to Euler angles (roll, pitch, yaw).

    For small angles, axis-angle ≈ rotation vector, and we can extract
    approximate Euler angles directly.
    """
    # axis_angle is a 3D vector where magnitude = angle, direction = axis
    angle = np.linalg.norm(axis_angle)
    if angle < 1e-6:
        return 0.0, 0.0, 0.0

    # For locomotion, we primarily care about sagittal plane (pitch)
    # The axis-angle components roughly correspond to rotations
    # around x (roll), y (pitch), z (yaw) for small angles
    roll = axis_angle[0]  # rotation around x
    pitch = axis_angle[1]  # rotation around y (sagittal)
    yaw = axis_angle[2]  # rotation around z

    return roll, pitch, yaw


def retarget_single_sequence(
    npz_path: str,
    target_fps: float = 50.0,
    scale_factor: float = 0.7,  # SMPL is ~1.7m, WildRobot is smaller
) -> dict:
    """Retarget a single AMASS sequence to WildRobot.

    Args:
        npz_path: Path to AMASS .npz file
        target_fps: Target frame rate (WildRobot control frequency)
        scale_factor: Scale factor for joint angles (SMPL human vs small robot)

    Returns:
        Dictionary with retargeted motion data
    """
    data = np.load(npz_path)

    poses = data["poses"]  # (N, 156) for SMPL-H or (N, 72) for SMPL
    trans = data["trans"]  # (N, 3)
    source_fps = float(data["mocap_framerate"])

    num_frames_source = poses.shape[0]
    num_joints = poses.shape[1] // 3

    # Reshape to (N, num_joints, 3) - each joint is 3D axis-angle
    poses = poses.reshape(num_frames_source, num_joints, 3)

    # Resample to target FPS
    resample_factor = source_fps / target_fps
    num_frames_target = max(1, int(num_frames_source / resample_factor))
    indices = np.linspace(0, num_frames_source - 1, num_frames_target).astype(int)

    # Extract relevant joints
    joint_positions = np.zeros((num_frames_target, 9), dtype=np.float32)

    for i, idx in enumerate(indices):
        # Right leg
        right_hip_aa = poses[idx, SMPL_JOINTS["right_hip"]]
        right_knee_aa = poses[idx, SMPL_JOINTS["right_knee"]]
        right_ankle_aa = poses[idx, SMPL_JOINTS["right_ankle"]]

        # Left leg
        left_hip_aa = poses[idx, SMPL_JOINTS["left_hip"]]
        left_knee_aa = poses[idx, SMPL_JOINTS["left_knee"]]
        left_ankle_aa = poses[idx, SMPL_JOINTS["left_ankle"]]

        # Spine (for waist yaw)
        spine_aa = poses[idx, SMPL_JOINTS["spine1"]]

        # Convert to Euler-like angles and scale
        # Hip pitch is primarily sagittal plane rotation (around lateral axis)
        # In SMPL, this is roughly the x-component for sagittal motion

        # Right leg mapping
        joint_positions[i, 0] = (
            -right_hip_aa[0] * scale_factor
        )  # hip_pitch (negate for convention)
        joint_positions[i, 1] = (
            right_hip_aa[2] * scale_factor * 0.5
        )  # hip_roll (reduced)
        joint_positions[i, 2] = max(
            0, right_knee_aa[0] * scale_factor
        )  # knee_pitch (always positive)
        joint_positions[i, 3] = right_ankle_aa[0] * scale_factor  # ankle_pitch

        # Left leg mapping
        joint_positions[i, 4] = -left_hip_aa[0] * scale_factor  # hip_pitch
        joint_positions[i, 5] = (
            -left_hip_aa[2] * scale_factor * 0.5
        )  # hip_roll (negate for symmetry)
        joint_positions[i, 6] = max(0, left_knee_aa[0] * scale_factor)  # knee_pitch
        joint_positions[i, 7] = left_ankle_aa[0] * scale_factor  # ankle_pitch

        # Waist yaw from spine
        joint_positions[i, 8] = spine_aa[1] * scale_factor * 0.5  # yaw (reduced)

    # Clamp to joint limits
    for j, joint_name in enumerate(WILDROBOT_JOINTS):
        limits = JOINT_LIMITS[joint_name]
        joint_positions[:, j] = np.clip(joint_positions[:, j], limits[0], limits[1])

    # Smooth the trajectory with a simple moving average
    kernel_size = 3
    if num_frames_target > kernel_size:
        kernel = np.ones(kernel_size) / kernel_size
        for j in range(9):
            joint_positions[:, j] = np.convolve(
                joint_positions[:, j], kernel, mode="same"
            )

    # Compute velocities via finite differences
    dt = 1.0 / target_fps
    joint_velocities = np.zeros_like(joint_positions)
    joint_velocities[1:] = (joint_positions[1:] - joint_positions[:-1]) / dt
    joint_velocities[0] = (
        joint_velocities[1] if num_frames_target > 1 else joint_velocities[0]
    )

    # Root position (scaled and offset for WildRobot height)
    root_position = trans[indices].astype(np.float32)
    root_position[:, 2] = 0.5  # Fixed base height for WildRobot
    root_position[:, :2] *= 0.5  # Scale horizontal motion

    # Root velocity
    root_velocity = np.zeros((num_frames_target, 3), dtype=np.float32)
    root_velocity[1:] = (root_position[1:] - root_position[:-1]) / dt
    root_velocity[0] = root_velocity[1] if num_frames_target > 1 else root_velocity[0]

    # Root orientation (identity for now - walking is mostly upright)
    root_orientation = np.zeros((num_frames_target, 4), dtype=np.float32)
    root_orientation[:, 0] = 1.0  # w=1 for identity quaternion (wxyz)

    root_angular_velocity = np.zeros((num_frames_target, 3), dtype=np.float32)

    return {
        "joint_positions": joint_positions,
        "joint_velocities": joint_velocities,
        "root_position": root_position,
        "root_orientation": root_orientation,
        "root_velocity": root_velocity,
        "root_angular_velocity": root_angular_velocity,
        "dt": dt,
        "fps": target_fps,
        "source_file": str(npz_path),
        "num_frames": num_frames_target,
    }


def find_walking_files(amass_dir: str) -> List[str]:
    """Find all walking motion files in AMASS directory."""
    patterns = [
        "**/Walk*Forward*.npz",
        "**/Walk*Straight*.npz",
        "**/walk*forward*.npz",
        "**/walk*straight*.npz",
    ]

    files = []
    for pattern in patterns:
        files.extend(glob(str(Path(amass_dir) / pattern), recursive=True))

    # Deduplicate
    files = list(set(files))
    return sorted(files)


def motion_to_amp_features(motion: dict, feature_dim: int = 29) -> np.ndarray:
    """Convert retargeted motion to AMP feature format.

    AMP features for WildRobot (29-dim):
    - joint_positions: 9
    - joint_velocities: 9
    - root_linear_velocity: 3
    - root_angular_velocity: 3
    - root_height: 1
    - foot_contacts: 4 (estimated from phase)

    Args:
        motion: Retargeted motion dictionary
        feature_dim: Expected feature dimension

    Returns:
        Array of shape (num_frames, feature_dim)
    """
    num_frames = motion["num_frames"]
    features = np.zeros((num_frames, feature_dim), dtype=np.float32)

    # Joint positions (0-8)
    features[:, 0:9] = motion["joint_positions"]

    # Joint velocities (9-17)
    features[:, 9:18] = motion["joint_velocities"]

    # Root linear velocity (18-20)
    features[:, 18:21] = motion["root_velocity"]

    # Root angular velocity (21-23)
    features[:, 21:24] = motion["root_angular_velocity"]

    # Root height (24)
    features[:, 24] = motion["root_position"][:, 2]

    # Foot contacts (25-28) - estimate from gait phase
    # During walking, feet alternate contact
    # Use hip pitch as proxy for gait phase
    left_hip_pitch = motion["joint_positions"][:, 4]
    right_hip_pitch = motion["joint_positions"][:, 0]

    # Left foot contact when left hip is extended (negative pitch)
    features[:, 25] = (left_hip_pitch < 0).astype(np.float32)  # left_front
    features[:, 26] = (left_hip_pitch < 0).astype(np.float32)  # left_back
    # Right foot contact when right hip is extended
    features[:, 27] = (right_hip_pitch < 0).astype(np.float32)  # right_front
    features[:, 28] = (right_hip_pitch < 0).astype(np.float32)  # right_back

    return features


def main():
    parser = argparse.ArgumentParser(description="Batch retarget AMASS walking data")
    parser.add_argument(
        "--amass-dir",
        type=str,
        default="/Users/ygli/projects/amass",
        help="Path to AMASS dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/walking_motions.pkl",
        help="Output pickle file",
    )
    parser.add_argument(
        "--target-fps", type=float, default=50.0, help="Target frame rate"
    )
    parser.add_argument(
        "--max-files", type=int, default=50, help="Maximum number of files to process"
    )
    parser.add_argument(
        "--scale", type=float, default=0.7, help="Scale factor for joint angles"
    )

    args = parser.parse_args()

    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Find walking files
    print(f"Searching for walking motions in {args.amass_dir}...")
    files = find_walking_files(args.amass_dir)
    print(f"Found {len(files)} walking motion files")

    if args.max_files and len(files) > args.max_files:
        print(f"Limiting to {args.max_files} files")
        files = files[: args.max_files]

    # Process each file
    all_motions = []
    all_features = []
    total_frames = 0

    for i, filepath in enumerate(files):
        try:
            print(f"[{i+1}/{len(files)}] Processing {Path(filepath).name}...")
            motion = retarget_single_sequence(
                filepath,
                target_fps=args.target_fps,
                scale_factor=args.scale,
            )
            features = motion_to_amp_features(motion)

            all_motions.append(motion)
            all_features.append(features)
            total_frames += motion["num_frames"]

            print(f"  → {motion['num_frames']} frames")
        except Exception as e:
            print(f"  → FAILED: {e}")
            continue

    print(f"\nProcessed {len(all_motions)} sequences, {total_frames} total frames")

    # Concatenate all features
    if all_features:
        combined_features = np.concatenate(all_features, axis=0)
        print(f"Combined features shape: {combined_features.shape}")
    else:
        combined_features = np.zeros((0, 29), dtype=np.float32)

    # Save
    output_data = {
        "motions": all_motions,
        "features": combined_features,
        "feature_dim": 29,
        "joint_names": WILDROBOT_JOINTS,
        "num_sequences": len(all_motions),
        "total_frames": total_frames,
        "target_fps": args.target_fps,
    }

    with open(args.output, "wb") as f:
        pickle.dump(output_data, f)

    print(f"\nSaved to {args.output}")
    print(f"  Sequences: {len(all_motions)}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {total_frames / args.target_fps:.1f}s")

    # Also save just the features for quick loading
    features_path = args.output.replace(".pkl", "_features.npy")
    np.save(features_path, combined_features)
    print(f"  Features saved to: {features_path}")


if __name__ == "__main__":
    main()
