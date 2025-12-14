"""Unit test comparing numpy vs JAX env helper outputs.

Run via: . .venv/bin/activate && python tests/envs/test_jax_env_fns.py
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

# Ensure repo root is on sys.path for direct script execution
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from playground_amp.envs.pure_env_fns import build_obs_from_state, compute_reward_from_state, is_done_from_state
from playground_amp.envs.jax_env_fns import build_obs_from_state_j, compute_reward_from_state_j, is_done_from_state_j

def approx(a, b, atol=1e-5):
    return np.allclose(np.array(a), np.array(b), atol=atol, rtol=1e-5)

def run_test():
    nq = 12
    nv = 6
    qpos = np.zeros(nq, dtype=np.float32)
    qvel = np.zeros(nv, dtype=np.float32)
    prev_action = np.zeros(11, dtype=np.float32)
    derived = {"xpos": np.array([0.0, 0.0, 0.5], dtype=np.float32), "xquat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)}

    # no noise for deterministic compare
    obs_np = build_obs_from_state(qpos, qvel, prev_action, derived=derived, obs_noise_std=0.0)
    obs_j = build_obs_from_state_j(qpos, qvel, prev_action, derived=derived, obs_noise_std=0.0)

    print('obs shapes:', obs_np.shape, obs_j.shape)
    assert approx(obs_np, obs_j), f'obs mismatch\nnp:{obs_np}\nj:{obs_j}'

    rew_np = compute_reward_from_state(qvel, np.zeros(nv))
    rew_j = float(compute_reward_from_state_j(qvel, np.zeros(nv)))
    assert approx(rew_np, rew_j), f'reward mismatch {rew_np} vs {rew_j}'

    done_np = is_done_from_state(qpos, qvel, obs_np, derived=derived, max_episode_steps=100, step_count=0)
    done_j = is_done_from_state_j(qpos, qvel, obs_j, derived=derived, max_episode_steps=100, step_count=0)
    assert done_np == done_j, f'done mismatch {done_np} vs {done_j}'

    print('All JAX vs NumPy env helper tests passed')

if __name__ == '__main__':
    run_test()
