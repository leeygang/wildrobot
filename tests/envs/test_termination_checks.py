"""Standalone checks for termination logic (pure + jax helpers).

Run with: .venv/bin/python tests/envs/test_termination_checks.py
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

# Ensure repo root on sys.path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from playground_amp.envs.pure_env_fns import is_done_from_state
try:
    from playground_amp.envs.jax_env_fns import is_done_from_state_j
    import jax
    import jax.numpy as jnp
except Exception:
    is_done_from_state_j = None
    jnp = None


def test_low_base_height():
    qpos = np.zeros(12, dtype=np.float32)
    qpos[2] = 0.1
    qvel = np.zeros(6, dtype=np.float32)
    obs = np.zeros(44, dtype=np.float32)
    assert is_done_from_state(qpos, qvel, obs, derived=None, max_episode_steps=500, step_count=0)
    if is_done_from_state_j is not None:
        qpos_j = jnp.array(qpos)
        qvel_j = jnp.array(qvel)
        obs_j = jnp.array(obs)
        assert bool(is_done_from_state_j(qpos_j, qvel_j, obs_j, derived=None, max_episode_steps=500, step_count=0))


def test_large_pitch_roll_from_obs():
    # craft obs with large pitch/roll in extras (last 6 entries)
    obs = np.zeros(44, dtype=np.float32)
    # extras layout: base_height, pitch, roll, com_x, com_y, com_z
    obs[-5] = 100.0  # pitch large
    qpos = np.zeros(7, dtype=np.float32)
    qvel = np.zeros(6, dtype=np.float32)
    assert is_done_from_state(qpos, qvel, obs, derived=None, max_episode_steps=500, step_count=0)
    if is_done_from_state_j is not None:
        assert bool(is_done_from_state_j(jnp.array(qpos), jnp.array(qvel), jnp.array(obs), derived=None, max_episode_steps=500, step_count=0))


def test_contact_force_triggers_done():
    qpos = np.zeros(12, dtype=np.float32)
    qvel = np.zeros(6, dtype=np.float32)
    obs = np.zeros(44, dtype=np.float32)
    derived = {"cfrc_ext": np.ones(10, dtype=np.float32) * 100.0}
    assert is_done_from_state(qpos, qvel, obs, derived=derived, max_episode_steps=500, step_count=0)
    if is_done_from_state_j is not None:
        d_j = {"cfrc_ext": jnp.array(derived["cfrc_ext"]) }
        assert bool(is_done_from_state_j(jnp.array(qpos), jnp.array(qvel), jnp.array(obs), derived=d_j, max_episode_steps=500, step_count=0))


def test_step_count_termination():
    qpos = np.zeros(12, dtype=np.float32)
    qvel = np.zeros(6, dtype=np.float32)
    obs = np.zeros(44, dtype=np.float32)
    assert is_done_from_state(qpos, qvel, obs, derived=None, max_episode_steps=5, step_count=5)
    if is_done_from_state_j is not None:
        assert bool(is_done_from_state_j(jnp.array(qpos), jnp.array(qvel), jnp.array(obs), derived=None, max_episode_steps=5, step_count=5))


def run_all():
    test_low_base_height()
    print('test_low_base_height passed')
    test_large_pitch_roll_from_obs()
    print('test_large_pitch_roll_from_obs passed')
    test_contact_force_triggers_done()
    print('test_contact_force_triggers_done passed')
    test_step_count_termination()
    print('test_step_count_termination passed')


if __name__ == '__main__':
    run_all()
    print('All termination helper tests passed')
