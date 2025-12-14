"""Equivalence test: compare JAX step + helpers vs NumPy Euler fallback + helpers.

Runs a short rollout with random torques/targets and verifies qpos/qvel/obs/reward match
within tolerance between the NumPy reference and the JAX implementation.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from playground_amp.envs.jax_full_port import make_jax_data, step_substep
from playground_amp.envs.pure_env_fns import build_obs_from_state, compute_reward_from_state, is_done_from_state
from playground_amp.envs.jax_env_fns import build_obs_from_state_j, compute_reward_from_state_j, is_done_from_state_j

try:
    import jax
    import jax.numpy as jnp
except Exception:
    jax = None
    jnp = None


def numpy_euler_step(qpos, qvel, torque, dt=0.02, mass_scale=1.0):
    accel = np.array(torque, dtype=np.float32) / (mass_scale + 1e-6)
    qvel = qvel + accel * dt
    # pad qvel to match qpos dimensionality when qpos includes base dofs
    nq = qpos.shape[0]
    nv = qvel.shape[0]
    if nq == nv:
        qpos = qpos + qvel * dt
    else:
        qvel_padded = np.pad(qvel, (0, nq - nv))
        qpos = qpos + qvel_padded * dt
    return qpos, qvel


def run_equiv_test():
    nq = 12
    nv = 6
    steps = 8
    rng = np.random.RandomState(0)
    # REMOVED: tests moved to tests/envs/
    # This file was removed from active test suite to avoid duplicate imports.

