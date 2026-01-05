from __future__ import annotations

import jax.numpy as jnp

from policy_contract.calib import JaxCalibOps
from policy_contract.jax.frames import angvel_heading_local, gravity_local_from_quat
from policy_contract.jax.signals import Signals
from policy_contract.jax.state import PolicyState
from policy_contract.spec import PolicySpec


def build_observation_from_components(
    *,
    spec: PolicySpec,
    gravity_local: jnp.ndarray,
    angvel_heading_local: jnp.ndarray,
    linvel_heading_local: jnp.ndarray,
    joint_pos_normalized: jnp.ndarray,
    joint_vel_normalized: jnp.ndarray,
    foot_switches: jnp.ndarray,
    prev_action: jnp.ndarray,
    velocity_cmd: jnp.ndarray,
) -> jnp.ndarray:
    if spec.observation.layout_id != "wr_obs_v1":
        raise ValueError(f"Unsupported layout_id: {spec.observation.layout_id}")

    if spec.observation.linvel_mode == "zero":
        linvel_heading_local = jnp.zeros_like(linvel_heading_local)
    elif spec.observation.linvel_mode in ("sim", "dropout"):
        # Dropout is applied upstream via the env's linvel mask.
        pass
    else:
        raise ValueError(f"Unsupported linvel_mode: {spec.observation.linvel_mode}")

    obs = jnp.concatenate(
        [
            gravity_local.reshape(3),
            angvel_heading_local.reshape(3),
            linvel_heading_local.reshape(3),
            joint_pos_normalized.reshape(-1),
            joint_vel_normalized.reshape(-1),
            foot_switches.reshape(4),
            prev_action.reshape(-1),
            jnp.atleast_1d(velocity_cmd).reshape(1),
            jnp.zeros((1,), dtype=jnp.float32),
        ]
    )
    return obs.astype(jnp.float32)


def build_observation(
    *,
    spec: PolicySpec,
    state: PolicyState,
    signals: Signals,
    velocity_cmd: jnp.ndarray,
) -> jnp.ndarray:
    gravity = gravity_local_from_quat(signals.quat_xyzw)
    angvel = angvel_heading_local(signals.gyro_rad_s, signals.quat_xyzw)
    linvel = jnp.zeros((3,), dtype=jnp.float32)

    joint_pos_norm = JaxCalibOps.normalize_joint_pos(spec=spec, joint_pos_rad=signals.joint_pos_rad)
    joint_vel_norm = JaxCalibOps.normalize_joint_vel(spec=spec, joint_vel_rad_s=signals.joint_vel_rad_s)

    return build_observation_from_components(
        spec=spec,
        gravity_local=gravity,
        angvel_heading_local=angvel,
        linvel_heading_local=linvel,
        joint_pos_normalized=joint_pos_norm,
        joint_vel_normalized=joint_vel_norm,
        foot_switches=signals.foot_switches,
        prev_action=state.prev_action,
        velocity_cmd=velocity_cmd,
    )
