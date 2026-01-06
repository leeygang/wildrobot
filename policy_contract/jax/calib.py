from __future__ import annotations

import jax.numpy as jnp

from policy_contract.spec import PolicySpec


def action_to_ctrl(*, spec: PolicySpec, action: jnp.ndarray) -> jnp.ndarray:
    if spec.action.mapping_id != "pos_target_rad_v1":
        raise ValueError(f"Unsupported mapping_id: {spec.action.mapping_id}")

    centers, spans, mirror_signs = _get_joint_params(spec)
    action = jnp.clip(action, -1.0, 1.0)
    corrected = action * mirror_signs
    ctrl = corrected * spans + centers
    return ctrl.astype(jnp.float32)


def ctrl_to_policy_action(*, spec: PolicySpec, ctrl_rad: jnp.ndarray) -> jnp.ndarray:
    if spec.action.mapping_id != "pos_target_rad_v1":
        raise ValueError(f"Unsupported mapping_id: {spec.action.mapping_id}")

    centers, spans, mirror_signs = _get_joint_params(spec)
    ctrl = jnp.asarray(ctrl_rad, dtype=jnp.float32)
    corrected = (ctrl - centers) / (spans + 1e-6)
    action = corrected * mirror_signs
    action = jnp.clip(action, -1.0, 1.0)
    return action.astype(jnp.float32)


def normalize_joint_pos(*, spec: PolicySpec, joint_pos_rad: jnp.ndarray) -> jnp.ndarray:
    """Normalize joint positions to [-1, 1] for logging/replay/debug tools."""
    centers, spans, _ = _get_joint_params(spec)
    normalized = (joint_pos_rad - centers) / (spans + 1e-6)
    return normalized.astype(jnp.float32)


def normalize_joint_vel(*, spec: PolicySpec, joint_vel_rad_s: jnp.ndarray) -> jnp.ndarray:
    """Normalize joint velocities to [-1, 1] for logging/replay/debug tools."""
    limits = _get_velocity_limits(spec)
    normalized = joint_vel_rad_s / limits
    normalized = jnp.clip(normalized, -1.0, 1.0)
    return normalized.astype(jnp.float32)


def _get_joint_params(spec: PolicySpec) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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
        jnp.asarray(centers, dtype=jnp.float32),
        jnp.asarray(spans, dtype=jnp.float32),
        jnp.asarray(mirror, dtype=jnp.float32),
    )


def _get_velocity_limits(spec: PolicySpec) -> jnp.ndarray:
    limits = []
    for name in spec.robot.actuator_names:
        joint = spec.robot.joints[name]
        limits.append(joint.max_velocity_rad_s)
    return jnp.asarray(limits, dtype=jnp.float32)
