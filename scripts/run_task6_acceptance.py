#!/usr/bin/env python3
"""Run full Task6 acceptance: long-horizon stress runs.

Usage:
  python scripts/run_task6_acceptance.py --episodes 50 --episode-steps 2000 --num-envs 16

The script runs episodes*episode_steps environment-steps using the JAX env and
writes a JSON summary to docs/phase3_task6_acceptance_run_<timestamp>.json.
"""
from __future__ import annotations
import argparse
import json
import time
from datetime import datetime
import numpy as np

from playground_amp.envs.wildrobot_env import WildRobotEnv, EnvConfig


def run(episodes: int, episode_steps: int, num_envs: int, seed: int, use_jax: bool, progress_every: int = 250):
    cfg = EnvConfig(num_envs=num_envs, seed=seed, use_jax=use_jax)
    cfg.max_episode_steps = episode_steps
    cfg.obs_noise_std = 0.0

    env = WildRobotEnv(cfg)
    obs = env.reset()

    total_env_steps = episodes * episode_steps
    env_step_calls = total_env_steps // num_envs

    start = time.time()
    terminations = 0
    nan_found = False
    step_times = []

    for i in range(int(env_step_calls)):
        t0 = time.time()
        acts = np.zeros((env.num_envs, env.ACT_DIM), dtype=np.float32)
        obs, rew, done, info = env.step(acts)
        dt = time.time() - t0
        step_times.append(dt)

        terminations += int(np.sum(done))
        if np.isnan(obs).any() or np.isinf(obs).any():
            nan_found = True
            print(f"NaN/Inf detected at iteration {i}, aborting")
            break
        if (i + 1) % progress_every == 0:
            elapsed = time.time() - start
            avg_step = sum(step_times) / len(step_times) if step_times else 0.0
            print(f"iter {i+1}/{env_step_calls}: elapsed {elapsed:.1f}s, avg_step={avg_step:.4f}s, terminations={terminations}")

    end = time.time()
    wall_seconds = end - start
    avg_step_time = float(sum(step_times) / len(step_times)) if step_times else None

    # Collect done reasons if available
    done_reasons = {}
    if hasattr(env, '_done_reasons'):
        done_reasons = dict(env._done_reasons)

    summary = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "episodes": episodes,
        "episode_steps": episode_steps,
        "num_envs": num_envs,
        "total_env_steps": total_env_steps,
        "env_step_calls": env_step_calls,
        "wall_seconds": wall_seconds,
        "avg_step_time": avg_step_time,
        "terminations": int(terminations),
        "nan_found": bool(nan_found),
        "done_reasons": done_reasons,
    }

    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--episode-steps", type=int, default=2000)
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--use-jax", action="store_true", default=True)
    p.add_argument("--progress-every", type=int, default=250)
    args = p.parse_args()

    print("Starting Task6 acceptance run with:", args)
    summary = run(args.episodes, args.episode_steps, args.num_envs, args.seed, args.use_jax, args.progress_every)

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_path = f"docs/phase3_task6_acceptance_run_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("Run complete. Summary:")
    print(json.dumps(summary, indent=2))
    print(f"Saved summary to {out_path}")


if __name__ == "__main__":
    main()
