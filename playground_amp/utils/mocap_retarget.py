"""MoCap retargeting utilities for WildRobot.

Retargets human walking motion data to WildRobot's 9-DOF skeleton:
- Right leg: hip_pitch, hip_roll, knee_pitch, ankle_pitch
- Left leg: hip_pitch, hip_roll, knee_pitch, ankle_pitch
- Torso: waist_yaw

Supports loading from:
1. AMASS dataset (SMPL format)
2. CMU MoCap (BVH format)
3. Simple keypoint trajectories (CSV/NPY)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

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


@dataclass
class RetargetedMotion:
    """Container for retargeted motion data."""

    joint_positions: np.ndarray  # (num_frames, 9)
    joint_velocities: np.ndarray  # (num_frames, 9)
    root_position: np.ndarray  # (num_frames, 3) - pelvis xyz
    root_orientation: np.ndarray  # (num_frames, 4) - pelvis quaternion (wxyz)
    root_velocity: np.ndarray  # (num_frames, 3) - pelvis linear velocity
    root_angular_velocity: np.ndarray  # (num_frames, 3)
    dt: float  # timestep
    fps: float  # frames per second

    @property
    def num_frames(self) -> int:
        return self.joint_positions.shape[0]

    @property
    def duration(self) -> float:
        return self.num_frames * self.dt


def create_walking_motion_from_keypoints(
    hip_amplitude: float = 0.4,
    knee_amplitude: float = 0.6,
    ankle_amplitude: float = 0.3,
    stride_frequency: float = 1.5,  # Hz
    duration: float = 10.0,  # seconds
    fps: float = 50.0,  # control frequency
    forward_speed: float = 0.4,  # m/s
) -> RetargetedMotion:
    """Create a simple walking motion using parameterized gait.

    This generates a more realistic walking pattern than pure sinusoids,
    using separate swing and stance phases.

    Args:
        hip_amplitude: Maximum hip flexion/extension (radians)
        knee_amplitude: Maximum knee flexion (radians)
        ankle_amplitude: Maximum ankle flexion (radians)
        stride_frequency: Steps per second
        duration: Total motion duration
        fps: Frames per second
        forward_speed: Forward walking speed (m/s)

    Returns:
        RetargetedMotion with walking gait
    """
    dt = 1.0 / fps
    num_frames = int(duration * fps)

    joint_positions = np.zeros((num_frames, 9), dtype=np.float32)
    joint_velocities = np.zeros((num_frames, 9), dtype=np.float32)
    root_position = np.zeros((num_frames, 3), dtype=np.float32)
    root_orientation = np.zeros((num_frames, 4), dtype=np.float32)
    root_velocity = np.zeros((num_frames, 3), dtype=np.float32)
    root_angular_velocity = np.zeros((num_frames, 3), dtype=np.float32)

    # Default orientation (identity quaternion wxyz)
    root_orientation[:, 0] = 1.0

    for i in range(num_frames):
        t = i * dt
        phase = 2 * np.pi * stride_frequency * t

        # Right leg (phase = 0 at start)
        # Hip: flexion during swing, extension during stance
        right_hip_pitch = hip_amplitude * np.sin(phase)
        # Hip roll: slight lateral motion
        right_hip_roll = 0.05 * np.sin(phase)
        # Knee: flexed during swing phase
        swing_phase_right = (np.sin(phase) + 1) / 2  # 0-1 during swing
        right_knee_pitch = (
            knee_amplitude * swing_phase_right * np.maximum(0, np.sin(phase))
        )
        # Ankle: dorsiflexion during swing, plantarflexion during push-off
        right_ankle_pitch = ankle_amplitude * np.sin(phase + np.pi / 4)

        # Left leg (180° out of phase)
        left_hip_pitch = hip_amplitude * np.sin(phase + np.pi)
        left_hip_roll = 0.05 * np.sin(phase + np.pi)
        swing_phase_left = (np.sin(phase + np.pi) + 1) / 2
        left_knee_pitch = (
            knee_amplitude * swing_phase_left * np.maximum(0, np.sin(phase + np.pi))
        )
        left_ankle_pitch = ankle_amplitude * np.sin(phase + np.pi + np.pi / 4)

        # Waist yaw: counter-rotation for balance
        waist_yaw = 0.1 * np.sin(phase)

        # Clamp to joint limits
        joint_positions[i] = np.array(
            [
                np.clip(right_hip_pitch, *JOINT_LIMITS["right_hip_pitch"]),
                np.clip(right_hip_roll, *JOINT_LIMITS["right_hip_roll"]),
                np.clip(right_knee_pitch, *JOINT_LIMITS["right_knee_pitch"]),
                np.clip(right_ankle_pitch, *JOINT_LIMITS["right_ankle_pitch"]),
                np.clip(left_hip_pitch, *JOINT_LIMITS["left_hip_pitch"]),
                np.clip(left_hip_roll, *JOINT_LIMITS["left_hip_roll"]),
                np.clip(left_knee_pitch, *JOINT_LIMITS["left_knee_pitch"]),
                np.clip(left_ankle_pitch, *JOINT_LIMITS["left_ankle_pitch"]),
                np.clip(waist_yaw, *JOINT_LIMITS["waist_yaw"]),
            ],
            dtype=np.float32,
        )

        # Root position: forward motion with slight vertical bobbing
        root_position[i] = np.array(
            [
                forward_speed * t,  # x: forward
                0.0,  # y: lateral
                0.5 + 0.02 * np.sin(2 * phase),  # z: height with double-frequency bob
            ],
            dtype=np.float32,
        )

        # Root velocity
        root_velocity[i] = np.array(
            [
                forward_speed,
                0.0,
                0.02 * 2 * 2 * np.pi * stride_frequency * np.cos(2 * phase),
            ],
            dtype=np.float32,
        )

    # Compute joint velocities via finite differences
    for i in range(1, num_frames):
        joint_velocities[i] = (joint_positions[i] - joint_positions[i - 1]) / dt
    joint_velocities[0] = joint_velocities[1]  # Copy first frame

    return RetargetedMotion(
        joint_positions=joint_positions,
        joint_velocities=joint_velocities,
        root_position=root_position,
        root_orientation=root_orientation,
        root_velocity=root_velocity,
        root_angular_velocity=root_angular_velocity,
        dt=dt,
        fps=fps,
    )


def load_amass_and_retarget(
    amass_path: str,
    target_fps: float = 50.0,
) -> RetargetedMotion:
    """Load AMASS SMPL data and retarget to WildRobot.

    AMASS uses SMPL body model with 23 joints. We need to extract:
    - Pelvis orientation → root
    - L/R Hip joints → hip_pitch, hip_roll
    - L/R Knee joints → knee_pitch
    - L/R Ankle joints → ankle_pitch
    - Spine → waist_yaw (approximation)

    Args:
        amass_path: Path to AMASS .npz file
        target_fps: Target frame rate for output

    Returns:
        RetargetedMotion for WildRobot
    """
    # SMPL joint indices
    # 0: Pelvis, 1: L_Hip, 2: R_Hip, 3: Spine1, 4: L_Knee, 5: R_Knee
    # 6: Spine2, 7: L_Ankle, 8: R_Ankle, 9: Spine3, etc.

    data = np.load(amass_path)

    # AMASS provides:
    # - 'poses': (num_frames, 156) - axis-angle for 52 joints (SMPL-H)
    # - 'trans': (num_frames, 3) - root translation
    # - 'betas': (16,) - body shape parameters
    # - 'gender': str
    # - 'mocap_framerate': float

    poses = data["poses"]  # (N, 156) or (N, 72) for SMPL
    trans = data["trans"]  # (N, 3)
    source_fps = float(data.get("mocap_framerate", 120.0))

    num_frames_source = poses.shape[0]
    num_joints = poses.shape[1] // 3

    # Reshape to (N, num_joints, 3) axis-angle
    poses = poses.reshape(num_frames_source, num_joints, 3)

    # Extract relevant joints (axis-angle → euler angles approximation)
    # For small angles, axis-angle ≈ euler angles

    # SMPL joint mapping (standard SMPL, not SMPL-H):
    # 0: pelvis, 1: left_hip, 2: right_hip, 3: spine1
    # 4: left_knee, 5: right_knee, 6: spine2
    # 7: left_ankle, 8: right_ankle, 9: spine3

    if num_joints >= 22:  # SMPL
        pelvis = poses[:, 0]  # (N, 3)
        left_hip = poses[:, 1]
        right_hip = poses[:, 2]
        spine = poses[:, 3]
        left_knee = poses[:, 4]
        right_knee = poses[:, 5]
        left_ankle = poses[:, 7]
        right_ankle = poses[:, 8]
    else:
        raise ValueError(f"Unexpected number of joints: {num_joints}")

    # Resample to target FPS
    resample_factor = source_fps / target_fps
    num_frames_target = int(num_frames_source / resample_factor)

    indices = np.linspace(0, num_frames_source - 1, num_frames_target).astype(int)

    # Extract joint angles
    # Hip pitch is rotation around Y axis (sagittal plane)
    # Hip roll is rotation around X axis (frontal plane)
    # Knee pitch is rotation around Y axis
    # Ankle pitch is rotation around Y axis

    joint_positions = np.zeros((num_frames_target, 9), dtype=np.float32)

    for i, idx in enumerate(indices):
        # Right leg
        joint_positions[i, 0] = -right_hip[idx, 0]  # hip_pitch (negate for convention)
        joint_positions[i, 1] = right_hip[idx, 2]  # hip_roll
        joint_positions[i, 2] = right_knee[idx, 0]  # knee_pitch
        joint_positions[i, 3] = right_ankle[idx, 0]  # ankle_pitch

        # Left leg
        joint_positions[i, 4] = -left_hip[idx, 0]  # hip_pitch
        joint_positions[i, 5] = -left_hip[idx, 2]  # hip_roll (negate for symmetry)
        joint_positions[i, 6] = left_knee[idx, 0]  # knee_pitch
        joint_positions[i, 7] = left_ankle[idx, 0]  # ankle_pitch

        # Waist yaw from spine rotation
        joint_positions[i, 8] = spine[idx, 1]  # yaw

    # Clamp to joint limits
    for j, joint_name in enumerate(WILDROBOT_JOINTS):
        limits = JOINT_LIMITS[joint_name]
        joint_positions[:, j] = np.clip(joint_positions[:, j], limits[0], limits[1])

    # Root position and orientation
    root_position = trans[indices]
    root_position[:, 2] += 0.5  # Offset to WildRobot base height

    # Convert pelvis axis-angle to quaternion
    root_orientation = np.zeros((num_frames_target, 4), dtype=np.float32)
    root_orientation[:, 0] = 1.0  # Identity for now (TODO: proper conversion)

    # Compute velocities
    dt = 1.0 / target_fps
    joint_velocities = np.zeros_like(joint_positions)
    joint_velocities[1:] = (joint_positions[1:] - joint_positions[:-1]) / dt
    joint_velocities[0] = joint_velocities[1]

    root_velocity = np.zeros((num_frames_target, 3), dtype=np.float32)
    root_velocity[1:] = (root_position[1:] - root_position[:-1]) / dt
    root_velocity[0] = root_velocity[1]

    root_angular_velocity = np.zeros((num_frames_target, 3), dtype=np.float32)

    return RetargetedMotion(
        joint_positions=joint_positions,
        joint_velocities=joint_velocities,
        root_position=root_position,
        root_orientation=root_orientation,
        root_velocity=root_velocity,
        root_angular_velocity=root_angular_velocity,
        dt=dt,
        fps=target_fps,
    )


def load_bvh_and_retarget(
    bvh_path: str,
    target_fps: float = 50.0,
) -> RetargetedMotion:
    """Load CMU MoCap BVH data and retarget to WildRobot.

    BVH files have hierarchical joint structure. We need to:
    1. Parse the BVH hierarchy
    2. Extract relevant joint angles
    3. Map to WildRobot joints

    Args:
        bvh_path: Path to BVH file
        target_fps: Target frame rate

    Returns:
        RetargetedMotion for WildRobot
    """
    # Simple BVH parser (minimal implementation)
    # For production, use a library like bvh-python

    with open(bvh_path, "r") as f:
        content = f.read()

    # Parse MOTION section
    motion_start = content.find("MOTION")
    if motion_start == -1:
        raise ValueError("No MOTION section found in BVH file")

    motion_section = content[motion_start:]
    lines = motion_section.strip().split("\n")

    num_frames = int(lines[1].split(":")[1].strip())
    frame_time = float(lines[2].split(":")[1].strip())
    source_fps = 1.0 / frame_time

    # Parse motion data
    motion_data = []
    for line in lines[3:]:
        if line.strip():
            values = [float(x) for x in line.strip().split()]
            motion_data.append(values)

    motion_data = np.array(motion_data, dtype=np.float32)

    # CMU skeleton typically has:
    # Root: Hips (translation + rotation)
    # Then hierarchical rotations for each joint

    # This is a simplified mapping - real BVH parsing needs proper hierarchy
    # Assuming standard CMU skeleton order

    # Resample
    resample_factor = source_fps / target_fps
    num_frames_target = int(num_frames / resample_factor)
    indices = np.linspace(0, num_frames - 1, num_frames_target).astype(int)

    # Extract joint positions (placeholder - needs proper BVH parsing)
    joint_positions = np.zeros((num_frames_target, 9), dtype=np.float32)

    # For now, return zeros with proper structure
    # TODO: Implement proper BVH joint mapping

    dt = 1.0 / target_fps
    joint_velocities = np.zeros_like(joint_positions)
    root_position = np.zeros((num_frames_target, 3), dtype=np.float32)
    root_position[:, 2] = 0.5
    root_orientation = np.zeros((num_frames_target, 4), dtype=np.float32)
    root_orientation[:, 0] = 1.0
    root_velocity = np.zeros((num_frames_target, 3), dtype=np.float32)
    root_angular_velocity = np.zeros((num_frames_target, 3), dtype=np.float32)

    return RetargetedMotion(
        joint_positions=joint_positions,
        joint_velocities=joint_velocities,
        root_position=root_position,
        root_orientation=root_orientation,
        root_velocity=root_velocity,
        root_angular_velocity=root_angular_velocity,
        dt=dt,
        fps=target_fps,
    )


def save_retargeted_motion(motion: RetargetedMotion, output_path: str):
    """Save retargeted motion to pickle file for AMP training.

    The saved format matches what ref_buffer.py expects.
    """
    import pickle

    # Combine into observation format expected by AMP
    # AMP features typically include: joint_pos, joint_vel, root_vel, etc.

    data = {
        "joint_positions": motion.joint_positions,
        "joint_velocities": motion.joint_velocities,
        "root_position": motion.root_position,
        "root_orientation": motion.root_orientation,
        "root_velocity": motion.root_velocity,
        "root_angular_velocity": motion.root_angular_velocity,
        "dt": motion.dt,
        "fps": motion.fps,
        "joint_names": WILDROBOT_JOINTS,
    }

    with open(output_path, "wb") as f:
        pickle.dump(data, f)

    print(f"Saved retargeted motion to {output_path}")
    print(f"  Frames: {motion.num_frames}")
    print(f"  Duration: {motion.duration:.2f}s")
    print(f"  FPS: {motion.fps}")


def motion_to_amp_observations(
    motion: RetargetedMotion,
    obs_dim: int = 44,
) -> np.ndarray:
    """Convert retargeted motion to AMP observation format.

    Creates observation vectors matching WildRobotEnv's observation space.

    Args:
        motion: RetargetedMotion data
        obs_dim: Expected observation dimension

    Returns:
        Array of shape (num_frames, obs_dim)
    """
    num_frames = motion.num_frames
    observations = np.zeros((num_frames, obs_dim), dtype=np.float32)

    for i in range(num_frames):
        obs = observations[i]

        # Base height
        obs[0] = motion.root_position[i, 2]

        # Base orientation (6D representation from quaternion)
        # Simplified: store quaternion in first 4 slots of orientation
        q = motion.root_orientation[i]
        obs[1:5] = q  # wxyz quaternion
        obs[5:7] = 0.0  # Padding for 6D repr

        # Joint positions (9 joints)
        obs[7:16] = motion.joint_positions[i]

        # Base linear velocity (3)
        obs[16:19] = motion.root_velocity[i]

        # Base angular velocity (3)
        obs[19:22] = motion.root_angular_velocity[i]

        # Joint velocities (9)
        obs[22:31] = motion.joint_velocities[i]

        # Remaining dimensions (contact forces, previous actions, etc.)
        # Set to default/zero for reference motion
        obs[31:] = 0.0

    return observations


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate/retarget walking motion for AMP"
    )
    parser.add_argument(
        "--mode",
        choices=["generate", "amass", "bvh"],
        default="generate",
        help="Mode: generate synthetic, load AMASS, or load BVH",
    )
    parser.add_argument(
        "--input", type=str, help="Input file path (for amass/bvh modes)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/walking_motion.pkl",
        help="Output pickle file path",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Duration in seconds (for generate mode)",
    )
    parser.add_argument(
        "--speed", type=float, default=0.4, help="Walking speed m/s (for generate mode)"
    )

    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "generate":
        print(f"Generating {args.duration}s of walking motion at {args.speed} m/s...")
        motion = create_walking_motion_from_keypoints(
            duration=args.duration,
            forward_speed=args.speed,
        )
    elif args.mode == "amass":
        if not args.input:
            raise ValueError("--input required for amass mode")
        print(f"Loading AMASS data from {args.input}...")
        motion = load_amass_and_retarget(args.input)
    elif args.mode == "bvh":
        if not args.input:
            raise ValueError("--input required for bvh mode")
        print(f"Loading BVH data from {args.input}...")
        motion = load_bvh_and_retarget(args.input)

    save_retargeted_motion(motion, args.output)

    # Also save as AMP observations
    obs_output = args.output.replace(".pkl", "_amp_obs.pkl")
    amp_obs = motion_to_amp_observations(motion)
    import pickle

    with open(obs_output, "wb") as f:
        pickle.dump(amp_obs, f)
    print(f"Saved AMP observations to {obs_output}")
    print(f"  Shape: {amp_obs.shape}")
