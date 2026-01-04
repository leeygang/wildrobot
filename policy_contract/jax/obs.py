from __future__ import annotations

import jax.numpy as jnp

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
