"""Minimal WildRobot-specific retarget/export helpers for teacher bootstrap."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from training.reference_motion.loader import ReferenceMotionClip
from training.reference_motion.phase import wrap_phase


@dataclass(frozen=True)
class JointLimits:
    """Per-actuator joint limits in WildRobot actuator order."""

    lower: np.ndarray
    upper: np.ndarray


def _finite_difference(values: np.ndarray, dt: float) -> np.ndarray:
    """First-order finite difference with wrap-safe edge handling."""
    if values.shape[0] < 2:
        return np.zeros_like(values)
    grad = np.zeros_like(values, dtype=np.float32)
    grad[1:-1] = (values[2:] - values[:-2]) / (2.0 * dt)
    grad[0] = (values[1] - values[0]) / dt
    grad[-1] = (values[-1] - values[-2]) / dt
    return grad.astype(np.float32)


def retarget_wildrobot_clip(
    clip: ReferenceMotionClip,
    *,
    joint_limits: JointLimits,
    pelvis_height_range: tuple[float, float] = (0.38, 0.52),
    stance_width_range: tuple[float, float] = (0.14, 0.26),
) -> ReferenceMotionClip:
    """Apply minimal WildRobot feasibility constraints to a clip."""
    joint_pos = np.clip(clip.joint_pos, joint_limits.lower, joint_limits.upper).astype(np.float32)
    joint_vel = _finite_difference(joint_pos, clip.dt)

    root_pos = np.asarray(clip.root_pos, dtype=np.float32).copy()
    root_pos[:, 2] = np.clip(root_pos[:, 2], pelvis_height_range[0], pelvis_height_range[1])
    root_lin_vel = _finite_difference(root_pos, clip.dt)

    left_foot_pos = np.asarray(clip.left_foot_pos, dtype=np.float32).copy()
    right_foot_pos = np.asarray(clip.right_foot_pos, dtype=np.float32).copy()
    center_y = 0.5 * (left_foot_pos[:, 1] + right_foot_pos[:, 1])
    width = np.clip(left_foot_pos[:, 1] - right_foot_pos[:, 1], stance_width_range[0], stance_width_range[1])
    left_foot_pos[:, 1] = center_y + 0.5 * width
    right_foot_pos[:, 1] = center_y - 0.5 * width

    metadata = dict(clip.metadata)
    metadata["retarget_version"] = "wildrobot_v0.16.0"
    metadata["pelvis_height_m"] = float(np.mean(root_pos[:, 2]))
    metadata["stance_width_m"] = float(np.mean(left_foot_pos[:, 1] - right_foot_pos[:, 1]))

    return ReferenceMotionClip(
        name=clip.name,
        dt=clip.dt,
        phase=wrap_phase(clip.phase),
        root_pos=root_pos,
        root_quat=np.asarray(clip.root_quat, dtype=np.float32),
        root_lin_vel=root_lin_vel,
        root_ang_vel=np.asarray(clip.root_ang_vel, dtype=np.float32),
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        left_foot_pos=left_foot_pos,
        right_foot_pos=right_foot_pos,
        left_foot_contact=np.asarray(clip.left_foot_contact, dtype=np.float32),
        right_foot_contact=np.asarray(clip.right_foot_contact, dtype=np.float32),
        metadata=metadata,
    )


def generate_bootstrap_nominal_walk_clip(
    *,
    actuator_names: Sequence[str],
    joint_limits: JointLimits,
    frame_count: int = 72,
    dt: float = 0.02,
    nominal_forward_velocity_mps: float = 0.10,
    name: str = "wildrobot_walk_forward_v001",
) -> ReferenceMotionClip:
    """Generate a deterministic robot-native nominal walk clip (bootstrap placeholder)."""
    if frame_count < 4:
        raise ValueError(f"frame_count must be >= 4, got {frame_count}")
    if len(actuator_names) != int(joint_limits.lower.shape[0]):
        raise ValueError("actuator_names and joint_limits size mismatch")

    t = np.arange(frame_count, dtype=np.float32)
    phase = (t / float(frame_count)).astype(np.float32)
    theta = 2.0 * np.pi * phase
    x = nominal_forward_velocity_mps * dt * t

    root_pos = np.stack(
        [
            x,
            np.zeros_like(x),
            0.45 + 0.006 * np.sin(2.0 * theta),
        ],
        axis=-1,
    ).astype(np.float32)
    root_quat = np.tile(np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (frame_count, 1))
    root_lin_vel = np.stack(
        [
            np.full_like(x, nominal_forward_velocity_mps),
            np.zeros_like(x),
            _finite_difference(root_pos[:, 2:3], dt).reshape(-1),
        ],
        axis=-1,
    ).astype(np.float32)
    root_ang_vel = np.zeros((frame_count, 3), dtype=np.float32)

    a = len(actuator_names)
    joint_pos = np.zeros((frame_count, a), dtype=np.float32)
    index = {name: i for i, name in enumerate(actuator_names)}
    swing = np.sin(theta)
    swing_opposite = np.sin(theta + np.pi)

    def _set_if_exists(name: str, value: np.ndarray) -> None:
        if name in index:
            joint_pos[:, index[name]] = value.astype(np.float32)

    _set_if_exists("left_hip_pitch", 0.22 + 0.20 * swing)
    _set_if_exists("right_hip_pitch", -0.22 + 0.20 * swing_opposite)
    _set_if_exists("left_knee_pitch", 0.32 + 0.28 * np.clip(swing, 0.0, None))
    _set_if_exists("right_knee_pitch", 0.32 + 0.28 * np.clip(swing_opposite, 0.0, None))
    _set_if_exists("left_ankle_pitch", -0.08 - 0.10 * np.clip(swing, 0.0, None))
    _set_if_exists("right_ankle_pitch", -0.08 - 0.10 * np.clip(swing_opposite, 0.0, None))
    _set_if_exists("left_hip_roll", -0.02 + 0.04 * np.sin(theta + np.pi / 2.0))
    _set_if_exists("right_hip_roll", 0.02 + 0.04 * np.sin(theta - np.pi / 2.0))
    _set_if_exists("waist_yaw", 0.03 * np.sin(theta))

    joint_pos = np.clip(joint_pos, joint_limits.lower, joint_limits.upper).astype(np.float32)
    joint_vel = _finite_difference(joint_pos, dt)

    left_foot_pos = np.stack(
        [
            x + 0.055 * np.sin(theta),
            np.full_like(x, 0.095),
            0.028 * np.clip(np.sin(theta), 0.0, None),
        ],
        axis=-1,
    ).astype(np.float32)
    right_foot_pos = np.stack(
        [
            x + 0.055 * np.sin(theta + np.pi),
            np.full_like(x, -0.095),
            0.028 * np.clip(np.sin(theta + np.pi), 0.0, None),
        ],
        axis=-1,
    ).astype(np.float32)
    left_contact = (left_foot_pos[:, 2] < 0.006).astype(np.float32)
    right_contact = (right_foot_pos[:, 2] < 0.006).astype(np.float32)

    metadata = {
        "name": name,
        "source_name": "bootstrap_nominal_walk",
        "source_format": "wildrobot_procedural_v0",
        "frame_count": int(frame_count),
        "dt": float(dt),
        "actuator_names": list(actuator_names),
        "loop_mode": "wrap",
        "phase_mode": "normalized_cycle",
        "retarget_version": "wildrobot_v0.16.0",
        "notes": "Deterministic bootstrap placeholder clip generated locally (no external mocap).",
        "nominal_forward_velocity_mps": float(nominal_forward_velocity_mps),
    }

    clip = ReferenceMotionClip(
        name=name,
        dt=float(dt),
        phase=phase.astype(np.float32),
        root_pos=root_pos,
        root_quat=root_quat,
        root_lin_vel=root_lin_vel,
        root_ang_vel=root_ang_vel,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        left_foot_pos=left_foot_pos,
        right_foot_pos=right_foot_pos,
        left_foot_contact=left_contact,
        right_foot_contact=right_contact,
        metadata=metadata,
    )
    return retarget_wildrobot_clip(clip, joint_limits=joint_limits)

