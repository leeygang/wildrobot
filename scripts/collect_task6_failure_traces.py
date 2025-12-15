#!/usr/bin/env python3
"""Collect termination traces from the WildRobot env for debugging.

Runs the env until a target number of termination events are observed (default 16),
records per-termination: env_index, global_step, step_in_episode, termination_reason, obs snapshot.
Saves results to docs/phase3_task6_failure_traces_<timestamp>.json
"""
from __future__ import annotations
import argparse
import json
import time
from datetime import datetime
import numpy as np

from playground_amp.envs.wildrobot_env import WildRobotEnv, EnvConfig


def infer_reason_from_obs(obs):
    try:
        base_height = float(obs[-6])
        pitch = float(obs[-5])
        roll = float(obs[-4])
        # joint vels are obs[14:25] (11 joints)
        joint_vels = obs[14:25]
    except Exception:
        return "unknown"
    
    reasons = []
    if base_height < 0.25:
        reasons.append("low_base_height")
    if abs(pitch) > (45.0 * 3.14159 / 180.0):
        reasons.append("pitch_large")
    if abs(roll) > (45.0 * 3.14159 / 180.0):
        reasons.append("roll_large")
    # check if any joint vel exceeds 50 rad/s
    import numpy as np
    if np.any(np.abs(joint_vels) > 50.0):
        reasons.append("large_joint_vel")
    
    if len(reasons) == 0:
        return "other"
    return ",".join(reasons)


def run(target_terminations: int, num_envs: int, seed: int, max_iters: int):
    cfg = EnvConfig(num_envs=num_envs, seed=seed, use_jax=True)
    cfg.max_episode_steps = 2000
    cfg.obs_noise_std = 0.0

    env = WildRobotEnv(cfg)
    obs = env.reset()

    term_records = []
    total_steps = 0
    iters = 0

    while len(term_records) < target_terminations and iters < max_iters:
        acts = np.zeros((env.num_envs, env.ACT_DIM), dtype=np.float32)
        obs, rew, done, info = env.step(acts)
        iters += 1
        total_steps += env.num_envs
        for i in range(env.num_envs):
            if done[i]:
                rec = {
                    "env_index": int(i),
                    "global_step_count": int(env._step_count[i]),
                    "obs_snapshot": [float(x) for x in obs[i].tolist()],
                    "reason_inferred": infer_reason_from_obs(obs[i]),
                    "base_height": float(obs[i][-6]),
                    "pitch": float(obs[i][-5]),
                    "roll": float(obs[i][-4]),
                    "joint_vels_max": float(max(abs(obs[i][14:25]))),
                }
                term_records.append(rec)
                if len(term_records) >= target_terminations:
                    break
    summary = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "target_terminations": target_terminations,
        "collected": len(term_records),
        "total_env_steps": total_steps,
        "iterations": iters,
        "terms": term_records,
    }
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target-terminations", type=int, default=16)
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-iters", type=int, default=5000)
    args = p.parse_args()

    print("Collecting up to", args.target_terminations, "terminations (num_envs=", args.num_envs, ")")
    summary = run(args.target_terminations, args.num_envs, args.seed, args.max_iters)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_path = f"docs/phase3_task6_failure_traces_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print("Wrote", out_path)
    print(json.dumps(summary, indent=2))

    # Print concise termination timing summary for quick triage
    try:
        steps = [int(t.get('global_step_count', -1)) for t in summary.get('terms', [])]
        from collections import Counter
        hist = dict(Counter(steps))
        envs = [int(t.get('env_index', -1)) for t in summary.get('terms', [])]
        env_hist = dict(Counter(envs))
        print('global_step_counts:', steps)
        print('global_step_count_hist:', hist)
        print('env_index_counts:', env_hist)
    except Exception:
        pass

if __name__ == '__main__':
    main()
