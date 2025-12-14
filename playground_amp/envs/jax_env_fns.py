"""JAX versions of the pure env helper functions.

These mirror `pure_env_fns.py` but operate on JAX arrays and are jittable.
Noise is optional and can be provided by passing a `key` for deterministic tests.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

try:
    import jax
    import jax.numpy as jnp
except Exception:
    jax = None
    jnp = None


def normalize_quat_wxyz_j(q: jnp.ndarray) -> jnp.ndarray:
    q = jnp.asarray(q, dtype=jnp.float32)
    # safety: if not length 4, return zeros
    q = jnp.where(q.size == 4, q, jnp.zeros(4, dtype=jnp.float32))
    # heuristic branch needs to be jax-friendly
    cond = jnp.abs(q[3]) > (jnp.abs(q[0]) + 0.3)
    out = jnp.where(cond, jnp.array([q[3], q[0], q[1], q[2]], dtype=jnp.float32), q.astype(jnp.float32))
    return out


def quat_to_euler_wxyz_j(q: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    w, x, y, z = q[0], q[1], q[2], q[3]
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = jnp.arctan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = jnp.where(t2 > 1.0, 1.0, t2)
    t2 = jnp.where(t2 < -1.0, -1.0, t2)
    pitch = jnp.arcsin(t2)
    return roll, pitch


def build_obs_from_state_j(
    qpos: jnp.ndarray,
    qvel: jnp.ndarray,
    prev_action: jnp.ndarray,
    derived: Optional[Dict[str, Any]] = None,
    obs_noise_std: float = 0.0,
    key: Optional[jax.random.KeyArray] = None,
) -> jnp.ndarray:
    derived = derived or {}
    gravity = jnp.array([0.0, 0.0, -1.0], dtype=jnp.float32)

    qpos = jnp.asarray(qpos, dtype=jnp.float32)
    qvel = jnp.asarray(qvel, dtype=jnp.float32)
    nq = int(qpos.shape[-1])

    # joint positions: prefer qpos[7:7+11] when available, otherwise take first 11 or zeros
    if nq >= 7 + 11:
        joint_pos = qpos[7 : 7 + 11]
    else:
        if nq >= 11:
            joint_pos = qpos[:11]
        else:
            joint_pos = jnp.zeros(11, dtype=jnp.float32)

    nv = int(qvel.shape[-1])
    # match numpy implementation: only use qvel if at least 11 entries, otherwise zeros
    if nv >= 11:
        joint_vel = qvel[:11]
    else:
        joint_vel = jnp.zeros(11, dtype=jnp.float32)

    pa = jnp.asarray(prev_action, dtype=jnp.float32)
    pa = pa[:11] if int(pa.shape[0]) >= 11 else jnp.pad(pa, (0, 11 - int(pa.shape[0])))

    base_height = 0.0
    pitch = 0.0
    roll = 0.0
    com = jnp.zeros(3, dtype=jnp.float32)

    # derived handling (xpos/xquat/com_pos/cfrc_ext)
    try:
        xpos = derived.get("xpos", None)
        if xpos is not None:
            ap = jnp.asarray(xpos, dtype=jnp.float32)
            base_height = ap[2] if ap.size >= 3 else base_height
        else:
            base_height = qpos[2] if qpos.size >= 3 else base_height

        raw_q = None
        if "xquat" in derived and derived["xquat"] is not None:
            raw_q = jnp.asarray(derived["xquat"], dtype=jnp.float32)
            if raw_q.ndim == 2:
                raw_q = raw_q[0, :4]
        elif qpos.size >= 7:
            raw_q = qpos[3:7]

        if raw_q is not None:
            qn = normalize_quat_wxyz_j(raw_q)
            r, p = quat_to_euler_wxyz_j(qn)
            roll = r
            pitch = p

        if "com_pos" in derived and derived["com_pos"] is not None:
            com = jnp.asarray(derived["com_pos"], dtype=jnp.float32)
        if "cfrc_ext" in derived and derived["cfrc_ext"] is not None:
            c = jnp.asarray(derived["cfrc_ext"], dtype=jnp.float32)
            contact_sum = jnp.sum(jnp.abs(c))
            com = com.at[2].set(com[2] + 0.001 * contact_sum)
    except Exception:
        pass

    extras = jnp.array([base_height, pitch, roll, com[0], com[1], com[2]], dtype=jnp.float32)
    obs = jnp.concatenate([gravity, joint_pos, joint_vel, pa[:11], jnp.zeros(2, dtype=jnp.float32), extras], axis=0)

    if obs_noise_std > 0.0 and key is not None:
        noise = jax.random.normal(key, obs.shape, dtype=jnp.float32) * obs_noise_std
        obs = obs + noise

    return obs


def compute_reward_from_state_j(qvel: jnp.ndarray, torques: jnp.ndarray) -> jnp.ndarray:
    v = jnp.asarray(qvel, dtype=jnp.float32)
    forward_vel = jnp.where(v.size >= 1, v[0], 0.0)
    torque_penalty = 0.0002 * jnp.sum(jnp.square(jnp.asarray(torques, dtype=jnp.float32)))
    return -jnp.abs(forward_vel) - torque_penalty


def is_done_from_state_j(qpos: jnp.ndarray, qvel: jnp.ndarray, obs: jnp.ndarray, derived: Optional[Dict[str, Any]] = None, max_episode_steps: int = 500, step_count: int = 0) -> bool:
    derived = derived or {}
    if step_count >= max_episode_steps:
        return True
    try:
        base_height = jnp.where(qpos is not None and qpos.size >= 3, qpos[2], obs[-6])
        pitch = 0.0
        roll = 0.0
        if qpos is not None and qpos.size >= 7:
            raw_q = qpos[3:7]
            nq = normalize_quat_wxyz_j(raw_q)
            r, p = quat_to_euler_wxyz_j(nq)
            roll = r
            pitch = p
        else:
            pitch = obs[-5]
            roll = obs[-4]

        cond1 = base_height < 0.25
        cond2 = (jnp.abs(pitch) > (45.0 * jnp.pi / 180.0)) | (jnp.abs(roll) > (45.0 * jnp.pi / 180.0))
        if cond1 or cond2:
            return True

        if "cfrc_ext" in derived and derived["cfrc_ext"] is not None:
            c = jnp.asarray(derived["cfrc_ext"], dtype=jnp.float32)
            if jnp.sum(jnp.abs(c)) > 500.0:
                return True
        if jnp.any(jnp.abs(jnp.asarray(qvel, dtype=jnp.float32)) > 50.0):
            return True
    except Exception:
        pass
    return False
