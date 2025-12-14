"""Pure, side-effect-free observation/reward/done functions for portability.

These functions accept host arrays and small derived-field dicts and return
numpy/jax-compatible arrays. They are intentionally minimal and mirror the
logic in `WildRobotEnv` to ease a later port to a jitted, functional API.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np


def normalize_quat_wxyz(q: np.ndarray) -> np.ndarray:
    """Normalize quaternion ordering heuristically to (w,x,y,z).

    Accepts a length-4 array in either (w,x,y,z) or (x,y,z,w) and returns
    a normalized (w,x,y,z) float32 array.
    """
    q = np.asarray(q, dtype=np.float32)
    if q.size != 4:
        return np.zeros(4, dtype=np.float32)
    # heuristic: if last component is large compared to others, assume xyzw
    if abs(q[3]) > (abs(q[0]) + 0.3):
        return np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)
    return q.astype(np.float32)


def quat_to_euler_wxyz(q: np.ndarray) -> Tuple[float, float]:
    """Convert (w,x,y,z) quaternion to (roll, pitch) in radians."""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = float(np.arctan2(t0, t1))
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = float(np.arcsin(t2))
    return roll, pitch


def build_obs_from_state(
    qpos: np.ndarray,
    qvel: np.ndarray,
    prev_action: np.ndarray,
    derived: Optional[Dict[str, Any]] = None,
    obs_noise_std: float = 0.02,
) -> np.ndarray:
    """Construct observation vector from host state and derived fields.

    `derived` may contain optional keys: `xpos` (array-like), `xquat`,
    `com_pos`, `cfrc_ext`. This mirrors the layout used in `WildRobotEnv`.
    """
    derived = derived or {}
    gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)

    # joint positions/velocities heuristics
    qpos = np.asarray(qpos, dtype=np.float32)
    qvel = np.asarray(qvel, dtype=np.float32)
    if qpos.size >= 7 + 11:
        joint_pos = qpos[7 : 7 + 11].astype(np.float32)
    else:
        joint_pos = qpos[:11].astype(np.float32) if qpos.size >= 11 else np.zeros(11, dtype=np.float32)
    joint_vel = qvel[:11].astype(np.float32) if qvel.size >= 11 else np.zeros(11, dtype=np.float32)

    prev_action = np.asarray(prev_action, dtype=np.float32)
    if prev_action.size < 11:
        prev_action = np.pad(prev_action, (0, 11 - prev_action.size))

    # extras
    base_height = 0.0
    pitch = 0.0
    roll = 0.0
    com = np.zeros(3, dtype=np.float32)

    try:
        xpos = derived.get("xpos", None)
        if xpos is not None:
            ap = np.asarray(xpos, dtype=np.float32)
            base_height = float(ap[2]) if ap.size >= 3 else base_height
        elif qpos.size >= 3:
            base_height = float(qpos[2])

        raw_q = None
        if "xquat" in derived and derived["xquat"] is not None:
            raw_q = np.asarray(derived["xquat"], dtype=np.float32)
            if raw_q.ndim == 2:
                raw_q = raw_q[0, :4]
        elif qpos.size >= 7:
            raw_q = qpos[3:7]

        if raw_q is not None:
            q = normalize_quat_wxyz(raw_q)
            roll, pitch = quat_to_euler_wxyz(q)

        if "com_pos" in derived and derived["com_pos"] is not None:
            com = np.asarray(derived["com_pos"], dtype=np.float32)
        if "cfrc_ext" in derived and derived["cfrc_ext"] is not None:
            c = np.asarray(derived["cfrc_ext"], dtype=np.float32)
            contact_sum = float(np.sum(np.abs(c)))
            com[2] = com[2] + 0.001 * contact_sum
    except Exception:
        pass

    extras = np.array([base_height, pitch, roll, com[0], com[1], com[2]], dtype=np.float32)
    obs = np.concatenate([gravity, joint_pos, joint_vel, prev_action[:11], np.zeros(2, dtype=np.float32), extras], axis=0)
    # add noise
    noise = np.random.normal(scale=obs_noise_std, size=obs.shape).astype(np.float32)
    return obs + noise


def compute_reward_from_state(qvel: np.ndarray, torques: np.ndarray) -> float:
    """Simple reward mirroring `_compute_reward` placeholder."""
    forward_vel = 0.0
    try:
        v = np.asarray(qvel, dtype=np.float32)
        if v.size >= 1:
            forward_vel = float(v[0])
    except Exception:
        forward_vel = 0.0
    torque_penalty = 0.0002 * float(np.sum(np.square(np.asarray(torques, dtype=np.float32))))
    return float(-abs(forward_vel) - torque_penalty)


def is_done_from_state(qpos: np.ndarray, qvel: np.ndarray, obs: np.ndarray, derived: Optional[Dict[str, Any]] = None, max_episode_steps: int = 500, step_count: int = 0) -> bool:
    """Termination logic independent of mjx.Data side effects.

    Uses host qpos/qvel and obs as fallbacks. `derived` may include `cfrc_ext`.
    """
    derived = derived or {}
    if step_count >= max_episode_steps:
        return True
    try:
        if qpos is not None and np.asarray(qpos).size >= 3:
            base_height = float(np.asarray(qpos)[2])
        else:
            base_height = float(obs[-6])
        # orientation
        pitch = 0.0
        roll = 0.0
        if qpos is not None and np.asarray(qpos).size >= 7:
            raw_q = np.asarray(qpos)[3:7]
            nq = normalize_quat_wxyz(raw_q)
            roll, pitch = quat_to_euler_wxyz(nq)
        else:
            pitch = float(obs[-5])
            roll = float(obs[-4])

        if base_height < 0.25:
            return True
        if abs(pitch) > (45.0 * np.pi / 180.0) or abs(roll) > (45.0 * np.pi / 180.0):
            return True

        # contact forces
        if "cfrc_ext" in derived and derived["cfrc_ext"] is not None:
            c = np.asarray(derived["cfrc_ext"], dtype=np.float32)
            if np.sum(np.abs(c)) > 500.0:
                return True
        if np.any(np.abs(np.asarray(qvel, dtype=np.float32)) > 50.0):
            return True
    except Exception:
        pass
    return False
