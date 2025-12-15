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


def jax_pd_control(
    target_qpos: jnp.ndarray,
    current_qpos: jnp.ndarray,
    current_qvel: jnp.ndarray,
    kp: float = 50.0,
    kd: float = 1.0,
    clip: float = 4.0,
):
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


def _quat_mul(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    # a,b: (...,4) in w,x,y,z order
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    rw = aw * bw - ax * bx - ay * by - az * bz
    rx = aw * bx + ax * bw + ay * bz - az * by
    ry = aw * by - ax * bz + ay * bw + az * bx
    rz = aw * bz + ax * by - ay * bx + az * bw
    return jnp.stack([rw, rx, ry, rz], axis=-1)


def step_substep(
    data: JaxData, torque: jnp.ndarray, dt: float = 0.02, mass_scale: jnp.ndarray = None
):
    """Apply torques and a simple dynamics update (batch-enabled).

    This is intentionally simple: accel = torque / mass_scale, then integrate.
    Replace with richer dynamics (contacts, constraints) in later phases.
    """
    # remember original qpos dimensionality to preserve output shapes
    orig_qpos_ndim = getattr(data.qpos, "ndim", None)

    # normalize inputs to 2D batch shapes when possible
    # data.qpos/qvel/ctrl may be (n,) or (batch,n) or have extra singleton dims from vmapping
    def _ensure_2d(x):
        if x is None:
            return None
        if getattr(x, "ndim", 0) == 1:
            return x[None, :]
        # collapse trailing dims into a single feature dimension
        return jnp.reshape(x, (x.shape[0], -1))

    qpos_in = _ensure_2d(data.qpos)
    qvel_in = _ensure_2d(data.qvel)
    ctrl_in = _ensure_2d(data.ctrl)

    # ensure torque shape: (batch, act_nv)
    if torque.ndim == 1:
        torque = torque[None, :]
    if mass_scale is None:
        mass_scale = 1.0

    # accel for action-dimension
    accel = torque / (mass_scale + 1e-6)

    # ensure accel is (batch, act_nv)
    if accel.ndim == 1:
        accel = accel[None, :]

    # pad accel to match the full qvel dimensionality if needed
    nv_full = qvel_in.shape[-1]
    nv_acc = accel.shape[-1]
    if nv_acc < nv_full:
        pad_width = nv_full - nv_acc
        accel = jnp.pad(accel, ((0, 0), (0, pad_width)))

    qvel = qvel_in + accel * dt

    # Integrate floating-base (if present) explicitly: qpos layout assumed
    # to have base pos (3) + base quat (4) at the start when nq >= 7.
    nq = data.qpos.shape[-1]
    nv = qvel.shape[-1]

    # start with current qpos
    qpos = data.qpos

    # base linear integration with gravity
    if nq >= 7 and nv >= 6:
        # gravity vector (m/s^2) applied to base linear DOFs
        g = jnp.array([0.0, 0.0, -9.81], dtype=jnp.float32)
        # ensure qvel parts are 2D: (batch, ...)
        base_lin_vel = jnp.reshape(qvel_in[..., :3], (qvel_in.shape[0], -1))
        base_ang = jnp.reshape(qvel_in[..., 3:6], (qvel_in.shape[0], -1))
        # integrate linear velocity with gravity
        base_lin_vel_new = base_lin_vel + g * dt
        base_pos = (
            jnp.reshape(qpos_in[..., :3], (qpos_in.shape[0], -1))
            + base_lin_vel * dt
            + 0.5 * g * (dt**2)
        )
        # quaternion update via q_dot = 0.5 * q * omega_quat (omega from angular vel)
        q = jnp.reshape(qpos_in[..., 3:7], (qpos_in.shape[0], -1))
        omega_quat = jnp.concatenate(
            [jnp.zeros_like(base_ang[..., :1]), base_ang], axis=-1
        )
        q_dot = 0.5 * _quat_mul(q, omega_quat)
        q_new = q + q_dot * dt
        # normalize with safe fallback to identity quaternion
        norm = jnp.linalg.norm(q_new, axis=-1, keepdims=True)
        eps = 1e-8
        default_q = jnp.broadcast_to(
            jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32), q_new.shape
        )
        cond = norm > eps
        q_new_normed = jnp.where(cond, q_new / norm, default_q)
        q_new = q_new_normed
        # assemble qpos: replace first 7 entries
        tail_qpos = jnp.reshape(qpos_in[..., 7:], (qpos_in.shape[0], -1))
        qpos = jnp.concatenate([base_pos, q_new, tail_qpos], axis=-1)
        # also update qvel base entries to include gravity effect
        tail = jnp.reshape(qvel_in[..., 6:], (qvel_in.shape[0], -1))
        qvel = jnp.concatenate([base_lin_vel_new, base_ang, tail], axis=-1)
    else:
        # fallback: simple joint integrator for non-floating base
        if nq == nv:
            qpos = data.qpos + qvel * dt
        else:
            pad = nq - nv
            qvel_padded = (
                jnp.pad(qvel, ((0, 0), (0, pad)))
                if qvel.ndim == 2
                else jnp.pad(qvel, (0, pad))
            )
            qpos = data.qpos + qvel_padded * dt

    # update placeholders
    xpos = qpos[..., :3] if qpos.shape[-1] >= 3 else data.xpos
    xquat = qpos[..., 3:7] if qpos.shape[-1] >= 7 else data.xquat

    # If original inputs were single-example 1D arrays, return 1D outputs to
    # avoid creating an extra singleton leading dimension when vmapped.
    if orig_qpos_ndim == 1:
        qpos_out = jnp.reshape(qpos, (-1,))
        qvel_out = jnp.reshape(qvel, (-1,))
        ctrl_out = (
            jnp.reshape(torque, (-1,)) if getattr(torque, "ndim", 0) >= 1 else torque
        )
        xpos_out = jnp.reshape(xpos, (-1,)) if getattr(xpos, "ndim", 0) >= 2 else xpos
        xquat_out = (
            jnp.reshape(xquat, (-1,)) if getattr(xquat, "ndim", 0) >= 2 else xquat
        )
        return JaxData(
            qpos=qpos_out, qvel=qvel_out, ctrl=ctrl_out, xpos=xpos_out, xquat=xquat_out
        )

    return JaxData(qpos=qpos, qvel=qvel, ctrl=torque, xpos=xpos, xquat=xquat)


def step_fn(
    data: JaxData,
    target_qpos: jnp.ndarray,
    dt: float = 0.02,
    kp: float = 50.0,
    kd: float = 1.0,
):
    """Given a `target_qpos`, compute torques and step one substep.

    Returns new JaxData.
    """
    # determine action dimension from target_qpos and slice state accordingly
    if target_qpos.ndim == 1:
        act_nv = target_qpos.shape[-1]
        target_qpos = target_qpos[None, :act_nv]
    else:
        act_nv = target_qpos.shape[-1]
        target_qpos = target_qpos[..., :act_nv]

    current_qpos = data.qpos[..., :act_nv]
    current_qvel = data.qvel[..., :act_nv]
    torque = jax_pd_control(target_qpos, current_qpos, current_qvel, kp=kp, kd=kd)
    return step_substep(data, torque, dt=dt)


# Integrate observations, reward, done using jax_env_fns when available
try:
    from playground_amp.envs.jax_env_fns import (
        build_obs_from_state_j,
        compute_reward_from_state_j,
        is_done_from_state_j,
    )
except Exception:
    build_obs_from_state_j = None
    compute_reward_from_state_j = None
    is_done_from_state_j = None


def step_and_observe(
    data: JaxData,
    target_qpos: jnp.ndarray,
    dt: float = 0.02,
    kp: float = 50.0,
    kd: float = 1.0,
    obs_noise_std: float = 0.0,
    key: Optional[jax.random.KeyArray] = None,
):
    """Step the `data` and return (new_data, obs, reward, done).

    `obs` is constructed from the new qpos/qvel using `build_obs_from_state_j`.
    """
    newd = step_fn(data, target_qpos, dt=dt, kp=kp, kd=kd)
    # build obs/reward/done using JAX helpers when available
    if build_obs_from_state_j is None:
        return newd, None, None, None

    # derive minimal derived dict placeholders from newd (xpos/xquat/com/cfrc not available yet)
    derived = {
        "xpos": newd.xpos[0] if newd.xpos is not None else None,
        "xquat": newd.xquat[0] if newd.xquat is not None else None,
    }
    # handle batch case: currently support batch=1 or vectorized via vmap externally
    qpos = newd.qpos[0] if newd.qpos.ndim == 2 else newd.qpos
    qvel = newd.qvel[0] if newd.qvel.ndim == 2 else newd.qvel
    obs = build_obs_from_state_j(
        qpos,
        qvel,
        newd.ctrl[0] if newd.ctrl.ndim == 2 else newd.ctrl,
        derived=derived,
        obs_noise_std=obs_noise_std,
        key=key,
    )
    reward = compute_reward_from_state_j(
        qvel, newd.ctrl[0] if newd.ctrl.ndim == 2 else newd.ctrl
    )
    done = is_done_from_state_j(
        qpos, qvel, obs, derived=derived, max_episode_steps=500, step_count=0
    )
    # Force a fresh evaluation of done using the jitted helper to avoid tracing mismatches.
    try:
        done = is_done_from_state_j(
            qpos, qvel, obs, derived=derived, max_episode_steps=500, step_count=0
        )
    except Exception:
        # keep prior value if re-evaluation fails
        pass
    return newd, obs, reward, done


if jax is not None and build_obs_from_state_j is not None:
    jitted_step_and_observe = jax.jit(step_and_observe)
else:
    jitted_step_and_observe = None


# JIT and vmap helpers
if jax is not None:
    jitted_step = jax.jit(step_fn)
    vmapped_step = jax.vmap(step_fn, in_axes=(0, 0), out_axes=0)

    # Provide a single-example step_and_observe variant that returns scalar (non-batched)
    def step_and_observe_single(
        data: JaxData,
        target_qpos: jnp.ndarray,
        dt: float = 0.02,
        kp: float = 50.0,
        kd: float = 1.0,
        obs_noise_std: float = 0.0,
        key: Optional[jax.random.KeyArray] = None,
    ):
        newd = step_fn(data, target_qpos, dt=dt, kp=kp, kd=kd)
        if build_obs_from_state_j is None:
            return newd, None, None, None

        # derive single-example arrays (handle both (n,) and (1,n) shapes)
        def _first_or_identity(x):
            if x is None:
                return None
            if getattr(x, "ndim", 0) == 1:
                return x
            return x[0]

        qpos = _first_or_identity(newd.qpos)
        qvel = _first_or_identity(newd.qvel)
        ctrl = _first_or_identity(newd.ctrl)
        xpos = (
            _first_or_identity(newd.xpos)
            if getattr(newd, "xpos", None) is not None
            else None
        )
        xquat = (
            _first_or_identity(newd.xquat)
            if getattr(newd, "xquat", None) is not None
            else None
        )

        derived = {"xpos": xpos, "xquat": xquat}
        obs = build_obs_from_state_j(
            qpos, qvel, ctrl, derived=derived, obs_noise_std=obs_noise_std, key=key
        )
        reward = compute_reward_from_state_j(qvel, ctrl)
        done = is_done_from_state_j(
            qpos, qvel, obs, derived=derived, max_episode_steps=500, step_count=0
        )
        # Squeeze leading singleton batch dims so vmapping stacks into shape (batch, n)
        try:
            if getattr(newd.qpos, "ndim", 0) == 2 and newd.qpos.shape[0] == 1:
                newd = JaxData(
                    qpos=newd.qpos[0],
                    qvel=newd.qvel[0],
                    ctrl=newd.ctrl[0],
                    xpos=(newd.xpos[0] if newd.xpos is not None else None),
                    xquat=(newd.xquat[0] if newd.xquat is not None else None),
                )
        except Exception:
            pass
        try:
            if obs is not None and getattr(obs, "ndim", 0) == 2 and obs.shape[0] == 1:
                obs = obs[0]
        except Exception:
            pass
        try:
            if getattr(reward, "ndim", 0) == 1 and reward.shape[0] == 1:
                reward = reward[0]
        except Exception:
            pass
        try:
            if getattr(done, "ndim", 0) == 0:
                # scalar boolean OK
                pass
            elif getattr(done, "ndim", 0) == 1 and done.shape[0] == 1:
                done = done[0]
        except Exception:
            pass
        return newd, obs, reward, done

    # vmap a single-example helper to obtain a true batched jitted function
    try:
        # Fix: in_axes should match actual function parameters
        # step_and_observe_single(data, target_qpos, dt=0.02, kp=50.0, kd=1.0, obs_noise_std=0.0, key=None)
        # Only first 2 args are batched (data, target_qpos), rest are shared scalars
        vmapped_step_and_observe = jax.vmap(
            step_and_observe_single,
            in_axes=(0, 0, None, None, None, None, 0),
            out_axes=(0, 0, 0, 0),
        )
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
            nds, obss, rews, dones = jitted_vmapped_step_and_observe(
                data, targ2, dt=0.02, kp=50.0, kd=1.0, obs_noise_std=0.0, key=None
            )
            print("batch obs shape:", obss.shape)
