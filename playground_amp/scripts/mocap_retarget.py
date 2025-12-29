"""MoCap utilities for WildRobot AMP training.

For motion retargeting from SMPL/SMPLX to WildRobot, use GMR (General Motion Retargeting):
    /Users/ygli/projects/GMR/general_motion_retargeting/ik_configs/smplx_to_wildrobot.json

This module provides:
1. Synthetic walking motion generation (parameterized gait)
2. Loading pre-retargeted motion data (from GMR output)
3. Converting motion to AMP observation format

Robot configuration is loaded from robot_config.yaml via get_robot_config().
The config must be loaded by the caller before using these utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from playground_amp.configs.training_config import get_robot_config


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
    hip_amplitude: float,
    knee_amplitude: float,
    ankle_amplitude: float,
    hip_roll_amplitude: float,
    waist_yaw_amplitude: float,
    stride_frequency: float,
    duration: float,
    fps: float,
    forward_speed: float,
    standing_height: float,
    height_bob_amplitude: float,
) -> RetargetedMotion:
    """Create a simple walking motion using parameterized gait.

    This generates a more realistic walking pattern than pure sinusoids,
    using separate swing and stance phases.

    Requires: Robot config must be loaded via load_robot_config() before calling.

    Args:
        hip_amplitude: Maximum hip flexion/extension (radians)
        knee_amplitude: Maximum knee flexion (radians)
        ankle_amplitude: Maximum ankle flexion (radians)
        hip_roll_amplitude: Maximum hip roll (radians)
        waist_yaw_amplitude: Maximum waist yaw (radians)
        stride_frequency: Steps per second (Hz)
        duration: Total motion duration (seconds)
        fps: Frames per second (control frequency)
        forward_speed: Forward walking speed (m/s)
        standing_height: Base standing height (meters)
        height_bob_amplitude: Vertical bobbing amplitude (meters)

    Returns:
        RetargetedMotion with walking gait
    """
    # Get joint limits from cached config
    robot_config = get_robot_config()
    joint_limits = robot_config.joint_limits
    joint_names = robot_config.actuator_joints
    num_joints = robot_config.action_dim

    # Build index mapping from joint names
    def get_joint_idx(name: str) -> int:
        try:
            return joint_names.index(name)
        except ValueError:
            raise ValueError(f"Joint '{name}' not found in config. Available: {joint_names}")

    # Get indices for each joint from config
    idx_waist_yaw = get_joint_idx("waist_yaw")
    idx_left_hip_pitch = get_joint_idx("left_hip_pitch")
    idx_left_hip_roll = get_joint_idx("left_hip_roll")
    idx_left_knee_pitch = get_joint_idx("left_knee_pitch")
    idx_left_ankle_pitch = get_joint_idx("left_ankle_pitch")
    idx_right_hip_pitch = get_joint_idx("right_hip_pitch")
    idx_right_hip_roll = get_joint_idx("right_hip_roll")
    idx_right_knee_pitch = get_joint_idx("right_knee_pitch")
    idx_right_ankle_pitch = get_joint_idx("right_ankle_pitch")

    dt = 1.0 / fps
    num_frames = int(duration * fps)

    joint_positions = np.zeros((num_frames, num_joints), dtype=np.float32)
    joint_velocities = np.zeros((num_frames, num_joints), dtype=np.float32)
    root_position = np.zeros((num_frames, 3), dtype=np.float32)
    root_orientation = np.zeros((num_frames, 4), dtype=np.float32)
    root_velocity = np.zeros((num_frames, 3), dtype=np.float32)
    root_angular_velocity = np.zeros((num_frames, 3), dtype=np.float32)

    # Default orientation (identity quaternion wxyz)
    root_orientation[:, 0] = 1.0

    # Clamp helper function
    def clamp(val: float, joint: str) -> float:
        limits = joint_limits.get(joint, (-np.pi, np.pi))
        return np.clip(val, limits[0], limits[1])

    for i in range(num_frames):
        t = i * dt
        phase = 2 * np.pi * stride_frequency * t

        # Right leg (phase = 0 at start)
        # Hip: flexion during swing, extension during stance
        right_hip_pitch = hip_amplitude * np.sin(phase)
        # Hip roll: slight lateral motion
        right_hip_roll = hip_roll_amplitude * np.sin(phase)
        # Knee: flexed during swing phase
        swing_phase_right = (np.sin(phase) + 1) / 2  # 0-1 during swing
        right_knee_pitch = (
            knee_amplitude * swing_phase_right * np.maximum(0, np.sin(phase))
        )
        # Ankle: dorsiflexion during swing, plantarflexion during push-off
        right_ankle_pitch = ankle_amplitude * np.sin(phase + np.pi / 4)

        # Left leg (180° out of phase)
        left_hip_pitch = hip_amplitude * np.sin(phase + np.pi)
        left_hip_roll = hip_roll_amplitude * np.sin(phase + np.pi)
        swing_phase_left = (np.sin(phase + np.pi) + 1) / 2
        left_knee_pitch = (
            knee_amplitude * swing_phase_left * np.maximum(0, np.sin(phase + np.pi))
        )
        left_ankle_pitch = ankle_amplitude * np.sin(phase + np.pi + np.pi / 4)

        # Waist yaw: counter-rotation for balance
        waist_yaw = waist_yaw_amplitude * np.sin(phase)

        # Assign to joint positions using config-based indices
        joint_positions[i, idx_right_hip_pitch] = clamp(right_hip_pitch, "right_hip_pitch")
        joint_positions[i, idx_right_hip_roll] = clamp(right_hip_roll, "right_hip_roll")
        joint_positions[i, idx_right_knee_pitch] = clamp(right_knee_pitch, "right_knee_pitch")
        joint_positions[i, idx_right_ankle_pitch] = clamp(right_ankle_pitch, "right_ankle_pitch")
        joint_positions[i, idx_left_hip_pitch] = clamp(left_hip_pitch, "left_hip_pitch")
        joint_positions[i, idx_left_hip_roll] = clamp(left_hip_roll, "left_hip_roll")
        joint_positions[i, idx_left_knee_pitch] = clamp(left_knee_pitch, "left_knee_pitch")
        joint_positions[i, idx_left_ankle_pitch] = clamp(left_ankle_pitch, "left_ankle_pitch")
        joint_positions[i, idx_waist_yaw] = clamp(waist_yaw, "waist_yaw")

        # Root position: forward motion with slight vertical bobbing
        root_position[i] = np.array(
            [
                forward_speed * t,  # x: forward
                0.0,  # y: lateral
                standing_height + height_bob_amplitude * np.sin(2 * phase),  # z: height with double-frequency bob
            ],
            dtype=np.float32,
        )

        # Root velocity
        root_velocity[i] = np.array(
            [
                forward_speed,
                0.0,
                height_bob_amplitude * 2 * 2 * np.pi * stride_frequency * np.cos(2 * phase),
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


def load_bvh_and_retarget(
    bvh_path: str,
    target_fps: float,
    standing_height: float,
) -> RetargetedMotion:
    """Load CMU MoCap BVH data and retarget to WildRobot.

    BVH files have hierarchical joint structure. We need to:
    1. Parse the BVH hierarchy
    2. Extract relevant joint angles
    3. Map to WildRobot joints

    Requires: Robot config must be loaded via load_robot_config() before calling.

    Args:
        bvh_path: Path to BVH file
        target_fps: Target frame rate
        standing_height: Robot standing height in meters

    Returns:
        RetargetedMotion for WildRobot
    """
    # Get config for joint count
    robot_config = get_robot_config()
    num_joints = robot_config.action_dim

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

    # Extract joint positions (placeholder - needs proper BVH parsing)
    joint_positions = np.zeros((num_frames_target, num_joints), dtype=np.float32)

    # For now, return zeros with proper structure
    # TODO: Implement proper BVH joint mapping

    dt = 1.0 / target_fps
    joint_velocities = np.zeros_like(joint_positions)
    root_position = np.zeros((num_frames_target, 3), dtype=np.float32)
    root_position[:, 2] = standing_height
    root_orientation = np.zeros((num_frames_target, 4), dtype=np.float32)
    root_orientation[:, 0] = 1.0  # Identity quaternion (w=1, x=y=z=0)
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


def save_retargeted_motion(
    motion: RetargetedMotion,
    output_path: str,
):
    """Save retargeted motion to pickle file for AMP training.

    The saved format matches what ref_buffer.py expects.

    Requires: Robot config must be loaded via load_robot_config() before calling.

    Args:
        motion: RetargetedMotion data to save
        output_path: Output pickle file path
    """
    import pickle

    # Get joint names from cached config
    robot_config = get_robot_config()
    joint_names = robot_config.actuator_joints

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
        "joint_names": joint_names,
    }

    with open(output_path, "wb") as f:
        pickle.dump(data, f)

    print(f"Saved retargeted motion to {output_path}")
    print(f"  Frames: {motion.num_frames}")
    print(f"  Duration: {motion.duration:.2f}s")
    print(f"  FPS: {motion.fps}")


def motion_to_amp_observations(
    motion: RetargetedMotion,
    obs_dim: int,
) -> np.ndarray:
    """Convert retargeted motion to AMP observation format.

    Creates observation vectors matching WildRobotEnv's observation space.
    Uses observation indices from robot_config.yaml.

    Requires: Robot config must be loaded via load_robot_config() before calling.

    Args:
        motion: RetargetedMotion data
        obs_dim: Expected observation dimension

    Returns:
        Array of shape (num_frames, obs_dim)
    """
    # Get observation indices from config
    robot_config = get_robot_config()
    obs_indices = robot_config.observation_indices

    # Get slice ranges from config
    def get_slice(component: str) -> slice:
        indices = obs_indices.get(component, {})
        return slice(indices.get("start", 0), indices.get("end", 0))

    slice_gravity = get_slice("gravity_vector")
    slice_base_angvel = get_slice("base_angular_velocity")
    slice_base_linvel = get_slice("base_linear_velocity")
    slice_joint_pos = get_slice("joint_positions")
    slice_joint_vel = get_slice("joint_velocities")

    num_frames = motion.num_frames
    observations = np.zeros((num_frames, obs_dim), dtype=np.float32)

    for i in range(num_frames):
        obs = observations[i]

        # Gravity vector (world up vector rotated into body frame)
        # World up vector is [0, 0, 1], rotate by inverse of body orientation
        quat = motion.root_orientation[i]  # wxyz format
        obs[slice_gravity] = _rotate_vector_by_quat_inverse(
            np.array([0.0, 0.0, 1.0]), quat
        )

        # Base angular velocity
        obs[slice_base_angvel] = motion.root_angular_velocity[i]

        # Base linear velocity
        obs[slice_base_linvel] = motion.root_velocity[i]

        # Joint positions
        obs[slice_joint_pos] = motion.joint_positions[i]

        # Joint velocities
        obs[slice_joint_vel] = motion.joint_velocities[i]

        # Remaining dimensions (previous_action, velocity_command, padding)
        # Set to default/zero for reference motion

    return observations


def _rotate_vector_by_quat_inverse(vec: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """Rotate a vector by the inverse of a quaternion.

    Args:
        vec: 3D vector to rotate
        quat: Quaternion in wxyz format [w, x, y, z]

    Returns:
        Rotated vector
    """
    # Extract quaternion components (wxyz format)
    w, x, y, z = quat

    # For inverse rotation, negate the vector part (conjugate)
    # q_inv = [w, -x, -y, -z] for unit quaternion

    # Rotation matrix from quaternion (transposed for inverse)
    # This is equivalent to rotating by q_conjugate
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y + w * z)
    r02 = 2 * (x * z - w * y)
    r10 = 2 * (x * y - w * z)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z + w * x)
    r20 = 2 * (x * z + w * y)
    r21 = 2 * (y * z - w * x)
    r22 = 1 - 2 * (x * x + y * y)

    # Apply transposed rotation (inverse)
    return np.array([
        r00 * vec[0] + r10 * vec[1] + r20 * vec[2],
        r01 * vec[0] + r11 * vec[1] + r21 * vec[2],
        r02 * vec[0] + r12 * vec[1] + r22 * vec[2],
    ], dtype=np.float32)


if __name__ == "__main__":
    import argparse
    from playground_amp.configs.training_config import load_robot_config

    parser = argparse.ArgumentParser(
        description="Generate/retarget walking motion for AMP"
    )
    parser.add_argument(
        "--mode",
        choices=["generate", "bvh"],
        required=True,
        help="Mode: generate synthetic or load BVH. For AMASS/SMPL data, use GMR.",
    )
    parser.add_argument(
        "--input", type=str, help="Input file path (for amass/bvh modes)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output pickle file path",
    )
    parser.add_argument(
        "--robot-config",
        type=str,
        required=True,
        help="Path to robot_config.yaml",
    )
    # Common parameters
    parser.add_argument(
        "--fps",
        type=float,
        required=True,
        help="Frames per second (target fps)",
    )
    parser.add_argument(
        "--standing-height",
        type=float,
        required=True,
        help="Robot standing height in meters",
    )
    # Gait parameters (required for generate mode)
    parser.add_argument(
        "--hip-amplitude",
        type=float,
        help="Max hip flexion/extension in radians (required for generate mode)",
    )
    parser.add_argument(
        "--knee-amplitude",
        type=float,
        help="Max knee flexion in radians (required for generate mode)",
    )
    parser.add_argument(
        "--ankle-amplitude",
        type=float,
        help="Max ankle flexion in radians (required for generate mode)",
    )
    parser.add_argument(
        "--hip-roll-amplitude",
        type=float,
        help="Max hip roll in radians (required for generate mode)",
    )
    parser.add_argument(
        "--waist-yaw-amplitude",
        type=float,
        help="Max waist yaw in radians (required for generate mode)",
    )
    parser.add_argument(
        "--stride-frequency",
        type=float,
        help="Steps per second in Hz (required for generate mode)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        help="Duration in seconds (required for generate mode)",
    )
    parser.add_argument(
        "--forward-speed",
        type=float,
        help="Walking speed in m/s (required for generate mode)",
    )
    parser.add_argument(
        "--height-bob-amplitude",
        type=float,
        help="Vertical bobbing amplitude in meters (required for generate mode)",
    )

    args = parser.parse_args()

    # Load robot config first (required by all retargeting functions)
    load_robot_config(args.robot_config)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "generate":
        # Validate required parameters for generate mode
        required_params = [
            ("hip_amplitude", args.hip_amplitude),
            ("knee_amplitude", args.knee_amplitude),
            ("ankle_amplitude", args.ankle_amplitude),
            ("hip_roll_amplitude", args.hip_roll_amplitude),
            ("waist_yaw_amplitude", args.waist_yaw_amplitude),
            ("stride_frequency", args.stride_frequency),
            ("duration", args.duration),
            ("forward_speed", args.forward_speed),
            ("height_bob_amplitude", args.height_bob_amplitude),
        ]
        missing = [name for name, val in required_params if val is None]
        if missing:
            parser.error(
                f"Generate mode requires: --{', --'.join(name.replace('_', '-') for name in missing)}"
            )

        print(f"Generating {args.duration}s of walking motion at {args.forward_speed} m/s...")
        motion = create_walking_motion_from_keypoints(
            hip_amplitude=args.hip_amplitude,
            knee_amplitude=args.knee_amplitude,
            ankle_amplitude=args.ankle_amplitude,
            hip_roll_amplitude=args.hip_roll_amplitude,
            waist_yaw_amplitude=args.waist_yaw_amplitude,
            stride_frequency=args.stride_frequency,
            duration=args.duration,
            fps=args.fps,
            forward_speed=args.forward_speed,
            standing_height=args.standing_height,
            height_bob_amplitude=args.height_bob_amplitude,
        )
    elif args.mode == "bvh":
        if not args.input:
            parser.error("--input required for bvh mode")
        print(f"Loading BVH data from {args.input}...")
        motion = load_bvh_and_retarget(
            args.input,
            target_fps=args.fps,
            standing_height=args.standing_height,
        )

    save_retargeted_motion(motion, args.output)

    # Also save as AMP observations
    obs_output = args.output.replace(".pkl", "_amp_obs.pkl")
    # Get obs_dim from robot config
    robot_config = get_robot_config()
    amp_obs = motion_to_amp_observations(motion, obs_dim=robot_config.observation_dim)
    import pickle

    with open(obs_output, "wb") as f:
        pickle.dump(amp_obs, f)
    print(f"Saved AMP observations to {obs_output}")
    print(f"  Shape: {amp_obs.shape}")
