"""Scaffold for a full pure-JAX simulator port.

This module provides a more complete starting point than `jax_sim_port.py`:
- `JaxData` pytree matching common simulator fields
- JAX PD controller and `step_fn` implementing a single substep
- jitted and vmapped helpers and a smoke test harness

Replace the simple integrator here with richer physics over time.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

try:
    import jax
    import jax.numpy as jnp
except Exception:
    jax = None
    jnp = None


@dataclass
class JaxData:
    qpos: Any  # shape (batch, nq)
    qvel: Any  # shape (batch, nv)
    ctrl: Any  # shape (batch, nv)
    # optional placeholders for derived quantities
    xpos: Any = None
    xquat: Any = None


def _flatten(data: JaxData):
    children = (data.qpos, data.qvel, data.ctrl, data.xpos, data.xquat)
    aux = None
    return children, aux


def _unflatten(aux, children):
    qpos, qvel, ctrl, xpos, xquat = children
    return JaxData(qpos=qpos, qvel=qvel, ctrl=ctrl, xpos=xpos, xquat=xquat)


if jax is not None:
    jax.tree_util.register_pytree_node(JaxData, _flatten, _unflatten)


def make_jax_data(nq: int, nv: int, batch: int = 1):
    qpos = jnp.zeros((batch, nq), dtype=jnp.float32)
    qvel = jnp.zeros((batch, nv), dtype=jnp.float32)
    ctrl = jnp.zeros((batch, nv), dtype=jnp.float32)
    xpos = jnp.zeros((batch, 3), dtype=jnp.float32)
    xquat = jnp.zeros((batch, 4), dtype=jnp.float32)
    return JaxData(qpos=qpos, qvel=qvel, ctrl=ctrl, xpos=xpos, xquat=xquat)


def jax_pd_control(target_qpos: jnp.ndarray, current_qpos: jnp.ndarray, current_qvel: jnp.ndarray, kp: float = 50.0, kd: float = 1.0, clip: float = 4.0):
    """Compute PD torques in JAX for batch inputs.

    Args:
        target_qpos: (batch, nv)
        current_qpos: (batch, nv)
        current_qvel: (batch, nv)
    Returns:
        torque: (batch, nv)
    """
    pos_err = target_qpos - current_qpos
    vel_err = -current_qvel
    torque = kp * pos_err + kd * vel_err
    torque = jnp.clip(torque, -clip, clip)
    return torque


def step_substep(data: JaxData, torque: jnp.ndarray, dt: float = 0.02, mass_scale: jnp.ndarray = None):
    """Apply torques and a simple dynamics update (batch-enabled).

    This is intentionally simple: accel = torque / mass_scale, then integrate.
    Replace with richer dynamics (contacts, constraints) in later phases.
    """
    # ensure shapes: (batch, nv)
    if torque.ndim == 1:
        torque = torque[None, :]
    if mass_scale is None:
        mass_scale = 1.0

    accel = torque / (mass_scale + 1e-6)
    qvel = data.qvel + accel * dt
    # pad qvel to match qpos dimensionality when qpos includes base dofs
    nq = data.qpos.shape[-1]
    nv = qvel.shape[-1]
    if nq == nv:
        qpos = data.qpos + qvel * dt
    else:
        # assume joint velocities map to the first `nv` qpos entries or to the
        # tail depending on caller; here we conservatively add into the first
        # `nv` DOFs and keep remaining qpos entries unchanged.
        pad = nq - nv
        qvel_padded = jnp.pad(qvel, ((0, 0), (0, pad))) if qvel.ndim == 2 else jnp.pad(qvel, (0, pad))
        qpos = data.qpos + qvel_padded * dt
    # update placeholders: simplistic base pose update
    xpos = data.xpos + jnp.zeros_like(data.xpos)
    xquat = data.xquat + jnp.zeros_like(data.xquat)
    return JaxData(qpos=qpos, qvel=qvel, ctrl=torque, xpos=xpos, xquat=xquat)


def step_fn(data: JaxData, target_qpos: jnp.ndarray, dt: float = 0.02, kp: float = 50.0, kd: float = 1.0):
    """Given a `target_qpos`, compute torques and step one substep.

    Returns new JaxData.
    """
    # use qpos/qvel slices matching action dim
    nv = data.qvel.shape[-1]
    # ensure target shape (batch, nv)
    if target_qpos.ndim == 1:
        target_qpos = target_qpos[None, :nv]
    else:
        target_qpos = target_qpos[..., :nv]

    current_qpos = data.qpos[..., :nv]
    current_qvel = data.qvel
    torque = jax_pd_control(target_qpos, current_qpos, current_qvel, kp=kp, kd=kd)
    return step_substep(data, torque, dt=dt)


# Integrate observations, reward, done using jax_env_fns when available
try:
    from playground_amp.envs.jax_env_fns import build_obs_from_state_j, compute_reward_from_state_j, is_done_from_state_j
except Exception:
    build_obs_from_state_j = None
    compute_reward_from_state_j = None
    is_done_from_state_j = None


def step_and_observe(data: JaxData, target_qpos: jnp.ndarray, dt: float = 0.02, kp: float = 50.0, kd: float = 1.0, obs_noise_std: float = 0.0, key: Optional[jax.random.KeyArray] = None):
    """Step the `data` and return (new_data, obs, reward, done).

    `obs` is constructed from the new qpos/qvel using `build_obs_from_state_j`.
    """
    newd = step_fn(data, target_qpos, dt=dt, kp=kp, kd=kd)
    # build obs/reward/done using JAX helpers when available
    if build_obs_from_state_j is None:
        return newd, None, None, None

    # derive minimal derived dict placeholders from newd (xpos/xquat/com/cfrc not available yet)
    derived = {"xpos": newd.xpos[0] if newd.xpos is not None else None, "xquat": newd.xquat[0] if newd.xquat is not None else None}
    # handle batch case: currently support batch=1 or vectorized via vmap externally
    qpos = newd.qpos[0] if newd.qpos.ndim == 2 else newd.qpos
    qvel = newd.qvel[0] if newd.qvel.ndim == 2 else newd.qvel
    obs = build_obs_from_state_j(qpos, qvel, newd.ctrl[0] if newd.ctrl.ndim == 2 else newd.ctrl, derived=derived, obs_noise_std=obs_noise_std, key=key)
    reward = compute_reward_from_state_j(qvel, newd.ctrl[0] if newd.ctrl.ndim == 2 else newd.ctrl)
    done = is_done_from_state_j(qpos, qvel, obs, derived=derived, max_episode_steps=500, step_count=0)
    return newd, obs, reward, done


if jax is not None and build_obs_from_state_j is not None:
    jitted_step_and_observe = jax.jit(step_and_observe)
else:
    jitted_step_and_observe = None


# JIT and vmap helpers
if jax is not None:
    jitted_step = jax.jit(step_fn)
    vmapped_step = jax.vmap(step_fn, in_axes=(0, 0), out_axes=0)
    # vmap the full step_and_observe to operate on batches of JaxData
    try:
        vmapped_step_and_observe = jax.vmap(step_and_observe, in_axes=(0, 0), out_axes=(0, 0, 0, 0))
        jitted_vmapped_step_and_observe = jax.jit(vmapped_step_and_observe)
    except Exception:
        vmapped_step_and_observe = None
        jitted_vmapped_step_and_observe = None


if __name__ == "__main__":
    if jax is None:
        print("JAX not available; skipping jax_full_port smoke test")
    else:
        nq = 12
        nv = 6
        batch = 4
        data = make_jax_data(nq=nq, nv=nv, batch=batch)
        # random target qpos for the action dim
        key = jax.random.PRNGKey(0)
        targ = jax.random.normal(key, (batch, nv)) * 0.1
        newd = jitted_step(data, targ)
        print("newd.qpos.shape:", newd.qpos.shape)
        # run a vmapped step with per-batch targets
        targ2 = jax.random.normal(key, (batch, nv)) * 0.2
        out = jax.vmap(step_fn, in_axes=(0, 0))(data, targ2)
        print("vmapped out qpos shape:", out.qpos.shape)
        if jitted_vmapped_step_and_observe is not None:
            # smoke test batch step_and_observe
            nds, obss, rews, dones = jitted_vmapped_step_and_observe(data, targ2, dt=0.02, kp=50.0, kd=1.0, obs_noise_std=0.0, key=None)
            print("batch obs shape:", obss.shape)
