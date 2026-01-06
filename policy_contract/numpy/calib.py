from __future__ import annotations

import numpy as np

from policy_contract.spec import PolicySpec


def action_to_ctrl(*, spec: PolicySpec, action: np.ndarray) -> np.ndarray:
    if spec.action.mapping_id != "pos_target_rad_v1":
        raise ValueError(f"Unsupported mapping_id: {spec.action.mapping_id}")

    centers, spans, mirror_signs = _get_joint_params(spec)
    action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
    corrected = action * mirror_signs
    ctrl = corrected * spans + centers
    return ctrl.astype(np.float32)


def ctrl_to_policy_action(*, spec: PolicySpec, ctrl_rad: np.ndarray) -> np.ndarray:
    if spec.action.mapping_id != "pos_target_rad_v1":
        raise ValueError(f"Unsupported mapping_id: {spec.action.mapping_id}")

    centers, spans, mirror_signs = _get_joint_params(spec)
    ctrl = np.asarray(ctrl_rad, dtype=np.float32)
    corrected = (ctrl - centers) / (spans + 1e-6)
    action = corrected * mirror_signs
    action = np.clip(action, -1.0, 1.0)
    return action.astype(np.float32)


def normalize_joint_pos(*, spec: PolicySpec, joint_pos_rad: np.ndarray) -> np.ndarray:
    """Normalize joint positions to [-1, 1] for logging/replay/debug tools."""
    centers, spans, _ = _get_joint_params(spec)
    normalized = (np.asarray(joint_pos_rad, dtype=np.float32) - centers) / (spans + 1e-6)
    return normalized.astype(np.float32)


def normalize_joint_vel(*, spec: PolicySpec, joint_vel_rad_s: np.ndarray) -> np.ndarray:
    """Normalize joint velocities to [-1, 1] for logging/replay/debug tools."""
    limits = _get_velocity_limits(spec)
    normalized = np.asarray(joint_vel_rad_s, dtype=np.float32) / limits
    normalized = np.clip(normalized, -1.0, 1.0)
    return normalized.astype(np.float32)


def _get_joint_params(spec: PolicySpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers = []
    spans = []
    mirror = []
    for name in spec.robot.actuator_names:
        joint = spec.robot.joints[name]
        center = (joint.range_min_rad + joint.range_max_rad) / 2.0
        span = (joint.range_max_rad - joint.range_min_rad) / 2.0
        centers.append(center)
        spans.append(span)
        mirror.append(joint.mirror_sign)
    return (
        np.asarray(centers, dtype=np.float32),
        np.asarray(spans, dtype=np.float32),
        np.asarray(mirror, dtype=np.float32),
    )


def _get_velocity_limits(spec: PolicySpec) -> np.ndarray:
    limits = []
    for name in spec.robot.actuator_names:
        joint = spec.robot.joints[name]
        limits.append(joint.max_velocity_rad_s)
    return np.asarray(limits, dtype=np.float32)
