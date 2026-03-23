from __future__ import annotations

import numpy as np

from policy_contract.spec import PolicySpec


def action_to_joint_target_rad(*, spec: PolicySpec, action: np.ndarray) -> np.ndarray:
    if spec.action.mapping_id == "pos_target_rad_v1":
        centers, spans, policy_action_signs = _get_joint_params_legacy(spec)
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        corrected = action * policy_action_signs
        joint_target_rad = corrected * spans + centers
        return joint_target_rad.astype(np.float32)
    elif spec.action.mapping_id == "pos_target_home_v1":
        homes, spans, mins, maxs, policy_action_signs = _get_joint_params_home(spec)
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        corrected = action * policy_action_signs
        joint_target_rad = np.clip(homes + corrected * spans, mins, maxs)
        return joint_target_rad.astype(np.float32)
    else:
        raise ValueError(f"Unsupported mapping_id: {spec.action.mapping_id}")


def joint_target_rad_to_action(*, spec: PolicySpec, joint_target_rad: np.ndarray) -> np.ndarray:
    if spec.action.mapping_id == "pos_target_rad_v1":
        centers, spans, policy_action_signs = _get_joint_params_legacy(spec)
        joint_target = np.asarray(joint_target_rad, dtype=np.float32)
        corrected = (joint_target - centers) / (spans + 1e-6)
        action = corrected * policy_action_signs
        action = np.clip(action, -1.0, 1.0)
        return action.astype(np.float32)
    elif spec.action.mapping_id == "pos_target_home_v1":
        homes, spans, _, _, policy_action_signs = _get_joint_params_home(spec)
        joint_target = np.asarray(joint_target_rad, dtype=np.float32)
        corrected = (joint_target - homes) / (spans + 1e-6)
        action = corrected * policy_action_signs
        action = np.clip(action, -1.0, 1.0)
        return action.astype(np.float32)
    else:
        raise ValueError(f"Unsupported mapping_id: {spec.action.mapping_id}")


def normalize_joint_pos(*, spec: PolicySpec, joint_pos_rad: np.ndarray) -> np.ndarray:
    """Normalize joint positions for observation building."""
    if spec.action.mapping_id == "pos_target_home_v1":
        homes, spans, _, _, _ = _get_joint_params_home(spec)
        normalized = (np.asarray(joint_pos_rad, dtype=np.float32) - homes) / (spans + 1e-6)
        return normalized.astype(np.float32)
    else:
        centers, spans, _ = _get_joint_params_legacy(spec)
        normalized = (np.asarray(joint_pos_rad, dtype=np.float32) - centers) / (spans + 1e-6)
        return normalized.astype(np.float32)


def normalize_joint_vel(*, spec: PolicySpec, joint_vel_rad_s: np.ndarray) -> np.ndarray:
    """Normalize joint velocities to [-1, 1] for logging/replay/debug tools."""
    limits = _get_velocity_limits(spec)
    normalized = np.asarray(joint_vel_rad_s, dtype=np.float32) / limits
    normalized = np.clip(normalized, -1.0, 1.0)
    return normalized.astype(np.float32)


def _get_joint_params_legacy(spec: PolicySpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Legacy: centers at midpoint of joint range, spans = half-range."""
    centers = []
    spans = []
    policy_action_signs = []
    for name in spec.robot.actuator_names:
        joint = spec.robot.joints[name]
        center = (joint.range_min_rad + joint.range_max_rad) / 2.0
        span = (joint.range_max_rad - joint.range_min_rad) / 2.0
        centers.append(center)
        spans.append(span)
        policy_action_signs.append(joint.policy_action_sign)
    return (
        np.asarray(centers, dtype=np.float32),
        np.asarray(spans, dtype=np.float32),
        np.asarray(policy_action_signs, dtype=np.float32),
    )


def _get_joint_params_home(
    spec: PolicySpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Home-centered: center at home_ctrl_rad, span = max distance to either limit."""
    homes = []
    spans = []
    mins = []
    maxs = []
    policy_action_signs = []
    for i, name in enumerate(spec.robot.actuator_names):
        joint = spec.robot.joints[name]
        home = 0.0
        if spec.robot.home_ctrl_rad is not None and i < len(spec.robot.home_ctrl_rad):
            home = spec.robot.home_ctrl_rad[i]
        span = max(abs(joint.range_min_rad - home), abs(joint.range_max_rad - home))
        homes.append(home)
        spans.append(span)
        mins.append(joint.range_min_rad)
        maxs.append(joint.range_max_rad)
        policy_action_signs.append(joint.policy_action_sign)
    return (
        np.asarray(homes, dtype=np.float32),
        np.asarray(spans, dtype=np.float32),
        np.asarray(mins, dtype=np.float32),
        np.asarray(maxs, dtype=np.float32),
        np.asarray(policy_action_signs, dtype=np.float32),
    )


def _get_velocity_limits(spec: PolicySpec) -> np.ndarray:
    limits = []
    for name in spec.robot.actuator_names:
        joint = spec.robot.joints[name]
        limits.append(joint.max_velocity_rad_s)
    return np.asarray(limits, dtype=np.float32)
