from __future__ import annotations

import jax.numpy as jnp

from policy_contract.spec import PolicySpec


def action_to_ctrl(*, spec: PolicySpec, action: jnp.ndarray) -> jnp.ndarray:
    if spec.action.mapping_id == "pos_target_rad_v1":
        centers, spans, policy_action_signs = _get_joint_params_legacy(spec)
        action = jnp.clip(action, -1.0, 1.0)
        corrected = action * policy_action_signs
        ctrl = corrected * spans + centers
        return ctrl.astype(jnp.float32)
    elif spec.action.mapping_id == "pos_target_home_v1":
        homes, spans, mins, maxs, policy_action_signs = _get_joint_params_home(spec)
        action = jnp.clip(action, -1.0, 1.0)
        corrected = action * policy_action_signs
        ctrl = jnp.clip(homes + corrected * spans, mins, maxs)
        return ctrl.astype(jnp.float32)
    else:
        raise ValueError(f"Unsupported mapping_id: {spec.action.mapping_id}")


def ctrl_to_policy_action(*, spec: PolicySpec, ctrl_rad: jnp.ndarray) -> jnp.ndarray:
    if spec.action.mapping_id == "pos_target_rad_v1":
        centers, spans, policy_action_signs = _get_joint_params_legacy(spec)
        ctrl = jnp.asarray(ctrl_rad, dtype=jnp.float32)
        corrected = (ctrl - centers) / (spans + 1e-6)
        action = corrected * policy_action_signs
        action = jnp.clip(action, -1.0, 1.0)
        return action.astype(jnp.float32)
    elif spec.action.mapping_id == "pos_target_home_v1":
        homes, spans, _, _, policy_action_signs = _get_joint_params_home(spec)
        ctrl = jnp.asarray(ctrl_rad, dtype=jnp.float32)
        corrected = (ctrl - homes) / (spans + 1e-6)
        action = corrected * policy_action_signs
        action = jnp.clip(action, -1.0, 1.0)
        return action.astype(jnp.float32)
    else:
        raise ValueError(f"Unsupported mapping_id: {spec.action.mapping_id}")


def normalize_joint_pos(*, spec: PolicySpec, joint_pos_rad: jnp.ndarray) -> jnp.ndarray:
    """Normalize joint positions for observation building."""
    if spec.action.mapping_id == "pos_target_home_v1":
        homes, spans, _, _, _ = _get_joint_params_home(spec)
        normalized = (joint_pos_rad - homes) / (spans + 1e-6)
        return normalized.astype(jnp.float32)
    else:
        centers, spans, _ = _get_joint_params_legacy(spec)
        normalized = (joint_pos_rad - centers) / (spans + 1e-6)
        return normalized.astype(jnp.float32)


def normalize_joint_vel(*, spec: PolicySpec, joint_vel_rad_s: jnp.ndarray) -> jnp.ndarray:
    """Normalize joint velocities to [-1, 1] for logging/replay/debug tools."""
    limits = _get_velocity_limits(spec)
    normalized = joint_vel_rad_s / limits
    normalized = jnp.clip(normalized, -1.0, 1.0)
    return normalized.astype(jnp.float32)


def _get_joint_params_legacy(spec: PolicySpec) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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
        jnp.asarray(centers, dtype=jnp.float32),
        jnp.asarray(spans, dtype=jnp.float32),
        jnp.asarray(policy_action_signs, dtype=jnp.float32),
    )


def _get_joint_params_home(
    spec: PolicySpec,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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
        jnp.asarray(homes, dtype=jnp.float32),
        jnp.asarray(spans, dtype=jnp.float32),
        jnp.asarray(mins, dtype=jnp.float32),
        jnp.asarray(maxs, dtype=jnp.float32),
        jnp.asarray(policy_action_signs, dtype=jnp.float32),
    )


def _get_velocity_limits(spec: PolicySpec) -> jnp.ndarray:
    limits = []
    for name in spec.robot.actuator_names:
        joint = spec.robot.joints[name]
        limits.append(joint.max_velocity_rad_s)
    return jnp.asarray(limits, dtype=jnp.float32)
