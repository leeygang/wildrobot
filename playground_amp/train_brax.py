"""Brax v2 training entrypoint for WildRobot using MJX physics.

This script wires the existing WildRobot MJX environment into a Brax-style
training loop. It prefers using Brax's PPO trainer when available; otherwise
it exposes a clear path for adapting the training loop.

Usage:
  uv run python -m playground_amp.train_brax --config playground_amp/configs/wildrobot_phase3_training.yaml --verify

"""
from __future__ import annotations

import argparse
import os
import time
from typing import Any

import jax
import jax.numpy as jnp

from playground_amp.brax_wrapper import create_brax_env_from_stage_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='playground_amp/configs/wildrobot_phase3_training.yaml')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()

    stage_cfg_path = args.config

    # Load config
    import yaml
    with open(stage_cfg_path, 'r') as f:
        config = yaml.safe_load(f)

    # create brax-wrapped env
    env = create_brax_env_from_stage_config(stage_cfg_path)

    print('Created Brax wrapper:', env)
    print('obs_dim, act_dim, num_envs =', env.obs_dim, env.act_dim, env.num_envs)

    # Try to use brax.training.ppo if available
    try:
        from brax.training import ppo
        from brax import training

        print('Found brax.training.ppo — building brax training run')

        # Build PPO config from stage config
        ppo_config = {
            'total_steps': config.get('total_steps', 100_000),
            'update_every': config.get('update_every', 2048),
            'num_envs': env.num_envs,
            'learning_rate': config.get('learning_rate', 3e-4),
        }

        print('NOTE: brax training integration is project-specific.\n'
              'This script demonstrates wiring; adapt ppo.train invocation to your'
              ' preferred training harness (trainer, logging, checkpoints).')

        # Attempt to call brax's ppo.train if present. Real usage may require
        # constructing a brax env class that inherits brax.envs.base.Env and
        # supports vectorized jitting; the wrapper here is a thin adapter.
        try:
            # ppo.train expects a function creating environments or a string env name
            print('Invoking brax.training.ppo.train (best-effort).')
            # This is a placeholder call – adapt as needed for your brax version
            ppo.train(ppo_config)
        except Exception as e:
            print('Could not call brax.training.ppo.train directly:', e)
            print('Proceeding to a small smoke-run using the wrapper (no optimization).')
            smoke_run(env)

    except Exception:
        print('brax.training.ppo not available; running minimal smoke-run.')
        smoke_run(env)


def smoke_run(env: Any):
    """Run a tiny forward-only smoke test using the Brax wrapper env."""
    rng = jax.random.PRNGKey(int(time.time()))
    obs = env.reset(rng)
    print('reset obs shape:', obs.shape)
    steps = 16
    total_rew = 0.0
    for t in range(steps):
        # use zero action or policy mean placeholder
        acts = jnp.zeros((env.num_envs, env.act_dim))
        obs, rew, done, info = env.step(acts)
        total_rew += float(jnp.mean(rew))
    print('smoke-run complete, avg reward per step:', total_rew / max(1, steps))


if __name__ == '__main__':
    main()
