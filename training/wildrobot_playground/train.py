#!/usr/bin/env python3
# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Train WildRobot with PPO using mujoco_playground."""

import os
import sys
import datetime
import functools
import json
import time
from pathlib import Path

from absl import app
from absl import flags
from absl import logging
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from etils import epath
import jax
import jax.numpy as jp
import mediapy as media
from ml_collections import config_dict
import mujoco
import wandb

# Add the project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.wildrobot_playground.wildrobot import locomotion
from training.wildrobot_playground.wildrobot import config_utils

# Configuration file
_CONFIG = flags.DEFINE_string(
    "config",
    None,
    "Path to YAML config file (default: default.yaml)"
)

# Environment flags (override config)
_TERRAIN = flags.DEFINE_string(
    "terrain",
    None,
    "Terrain type: 'flat' or 'rough' (overrides config)"
)

# Training flags (override config)
_NUM_TIMESTEPS = flags.DEFINE_integer(
    "num_timesteps",
    None,
    "Number of timesteps to train (overrides config)"
)
_NUM_ENVS = flags.DEFINE_integer(
    "num_envs",
    None,
    "Number of parallel environments (overrides config)"
)
_SEED = flags.DEFINE_integer(
    "seed",
    None,
    "Random seed (overrides config)"
)
_EPISODE_LENGTH = flags.DEFINE_integer(
    "episode_length",
    None,
    "Episode length (overrides config)"
)

# PPO flags (override config)
_BATCH_SIZE = flags.DEFINE_integer(
    "batch_size",
    None,
    "Batch size (overrides config)"
)
_LEARNING_RATE = flags.DEFINE_float(
    "learning_rate",
    None,
    "Learning rate (overrides config)"
)
_ENTROPY_COST = flags.DEFINE_float(
    "entropy_cost",
    None,
    "Entropy cost (overrides config)"
)

# Checkpointing
_LOAD_CHECKPOINT = flags.DEFINE_string(
    "load_checkpoint",
    None,
    "Path to checkpoint to load"
)

# Rendering
_RENDER = flags.DEFINE_boolean(
    "render",
    None,
    "Render videos after training (overrides config)"
)
_NUM_VIDEOS = flags.DEFINE_integer(
    "num_videos",
    None,
    "Number of videos to render (overrides config)"
)

# W&B Logging (override config)
_USE_WANDB = flags.DEFINE_boolean(
    "use_wandb",
    None,
    "Use Weights & Biases for logging (overrides config)"
)
_WANDB_PROJECT = flags.DEFINE_string(
    "wandb_project",
    None,
    "W&B project name (overrides config)"
)

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"


def main(argv):
    del argv

    # Set up logging
    logging.set_verbosity(logging.INFO)

    # Load config file
    cfg = config_utils.load_config(_CONFIG.value)

    # Create override dictionary from command-line flags
    overrides = {}
    if _TERRAIN.value is not None:
        overrides['env.terrain'] = _TERRAIN.value
    if _NUM_TIMESTEPS.value is not None:
        overrides['training.num_timesteps'] = _NUM_TIMESTEPS.value
    if _NUM_ENVS.value is not None:
        overrides['ppo.num_envs'] = _NUM_ENVS.value
    if _SEED.value is not None:
        overrides['training.seed'] = _SEED.value
    if _EPISODE_LENGTH.value is not None:
        overrides['training.episode_length'] = _EPISODE_LENGTH.value
    if _BATCH_SIZE.value is not None:
        overrides['ppo.batch_size'] = _BATCH_SIZE.value
    if _LEARNING_RATE.value is not None:
        overrides['ppo.learning_rate'] = _LEARNING_RATE.value
    if _ENTROPY_COST.value is not None:
        overrides['ppo.entropy_cost'] = _ENTROPY_COST.value
    if _RENDER.value is not None:
        overrides['rendering.render_videos'] = _RENDER.value
    if _NUM_VIDEOS.value is not None:
        overrides['rendering.num_videos'] = _NUM_VIDEOS.value
    if _USE_WANDB.value is not None:
        overrides['logging.use_wandb'] = _USE_WANDB.value
    if _WANDB_PROJECT.value is not None:
        overrides['logging.wandb_project'] = _WANDB_PROJECT.value

    # Apply overrides
    cfg = config_utils.override_config(cfg, overrides)

    # Extract values from config
    terrain = cfg['env']['terrain']
    task_name = f"wildrobot_{terrain}"

    print(f"Training WildRobot Locomotion")
    print(f"  Terrain: {terrain}")
    print(f"  Velocity range: 0.0-1.0 m/s")
    print(f"  Config: {_CONFIG.value or 'default.yaml'}")

    # Create environment config
    env_config = config_dict.ConfigDict()
    env_config.ctrl_dt = cfg['env']['ctrl_dt']
    env_config.sim_dt = cfg['env']['sim_dt']

    # Create environment
    env = locomotion.WildRobotLocomotion(task=task_name, config=env_config)

    print(f"\nEnvironment: {env.__class__.__name__}")
    print(f"  Observation size: {env.observation_size}")
    print(f"  Action size: {env.action_size}")

    # PPO training parameters
    ppo_config = config_dict.ConfigDict()
    ppo_config.num_timesteps = cfg['training']['num_timesteps']
    ppo_config.num_evals = cfg['training']['num_evals']
    ppo_config.reward_scaling = cfg['ppo']['reward_scaling']
    ppo_config.episode_length = cfg['training']['episode_length']
    ppo_config.normalize_observations = cfg['ppo']['normalize_observations']
    ppo_config.action_repeat = cfg['ppo']['action_repeat']
    ppo_config.unroll_length = cfg['ppo']['unroll_length']
    ppo_config.num_minibatches = cfg['ppo']['num_minibatches']
    ppo_config.num_updates_per_batch = cfg['ppo']['num_updates_per_batch']
    ppo_config.discounting = cfg['ppo']['discounting']
    ppo_config.learning_rate = cfg['ppo']['learning_rate']
    ppo_config.entropy_cost = cfg['ppo']['entropy_cost']
    ppo_config.num_envs = cfg['ppo']['num_envs']
    ppo_config.batch_size = cfg['ppo']['batch_size']
    ppo_config.max_grad_norm = cfg['ppo']['max_grad_norm']
    ppo_config.clipping_epsilon = cfg['ppo']['clipping_epsilon']

    # Network parameters
    network_config = config_dict.ConfigDict()
    network_config.policy_hidden_layer_sizes = cfg['network']['policy_hidden_layers']
    network_config.value_hidden_layer_sizes = cfg['network']['value_hidden_layers']

    print(f"\nTraining Configuration:")
    print(f"  Timesteps: {ppo_config.num_timesteps:,}")
    print(f"  Environments: {ppo_config.num_envs}")
    print(f"  Batch size: {ppo_config.batch_size}")
    print(f"  Learning rate: {ppo_config.learning_rate}")
    print(f"  Episode length: {ppo_config.episode_length}")

    # Set up experiment directory
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    exp_name = f"wildrobot_locomotion_{terrain}_{timestamp}"
    logdir = epath.Path(PROJECT_ROOT) / "training" / "logs" / exp_name
    logdir.mkdir(parents=True, exist_ok=True)

    print(f"\nExperiment: {exp_name}")
    print(f"Logs: {logdir}")

    # Initialize W&B if enabled
    if cfg['logging']['use_wandb']:
        wandb_config = {
            'terrain': terrain,
            'num_timesteps': ppo_config.num_timesteps,
            'num_envs': ppo_config.num_envs,
            'batch_size': ppo_config.batch_size,
            'learning_rate': ppo_config.learning_rate,
            'episode_length': ppo_config.episode_length,
            'network_policy': network_config.policy_hidden_layer_sizes,
            'network_value': network_config.value_hidden_layer_sizes,
        }

        wandb.init(
            project=cfg['logging']['wandb_project'],
            entity=cfg['logging']['wandb_entity'],
            name=exp_name,
            config=wandb_config,
            dir=str(logdir),
        )
        print(f"W&B initialized: {wandb.run.url}")

    # Save config
    ckpt_dir = logdir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    # Save both the config file path and overrides
    with open(ckpt_dir / "config.json", "w") as f:
        json.dump({
            "config_file": _CONFIG.value or "default.yaml",
            "overrides": overrides,
            "final_config": cfg,
        }, f, indent=2)

    # Create network factory
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=network_config.policy_hidden_layer_sizes,
        value_hidden_layer_sizes=network_config.value_hidden_layer_sizes,
    )

    # Progress callback
    times = [time.monotonic()]

    def progress(num_steps, metrics):
        times.append(time.monotonic())
        elapsed = times[-1] - times[1] if len(times) > 1 else 0

        # Extract key metrics
        eval_reward = metrics.get('eval/episode_reward', 0.0)
        eval_length = metrics.get('eval/episode_length', 0.0)

        print(f"Step {num_steps:,}: reward={eval_reward:.3f}, length={eval_length:.1f}, elapsed={elapsed:.1f}s")

        # Log to W&B if enabled
        if cfg['logging']['use_wandb']:
            log_dict = {
                'train/step': num_steps,
                'train/elapsed_time': elapsed,
                'eval/episode_reward': eval_reward,
                'eval/episode_length': eval_length,
            }

            # Log additional training metrics if available
            for key, value in metrics.items():
                if key.startswith('training/') or key.startswith('losses/'):
                    log_dict[key] = value

            # Log learning rate and other optimizer metrics
            if 'learning_rate' in metrics:
                log_dict['train/learning_rate'] = metrics['learning_rate']

            wandb.log(log_dict, step=num_steps)

    # Wrap environment for training
    from mujoco_playground import wrapper
    wrapped_env = wrapper.wrap_for_brax_training(
        env,
        episode_length=ppo_config.episode_length,
        action_repeat=ppo_config.action_repeat,
    )

    # Train
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")

    train_fn = functools.partial(
        ppo.train,
        num_timesteps=ppo_config.num_timesteps,
        num_evals=ppo_config.num_evals,
        reward_scaling=ppo_config.reward_scaling,
        episode_length=ppo_config.episode_length,
        normalize_observations=ppo_config.normalize_observations,
        action_repeat=ppo_config.action_repeat,
        unroll_length=ppo_config.unroll_length,
        num_minibatches=ppo_config.num_minibatches,
        num_updates_per_batch=ppo_config.num_updates_per_batch,
        discounting=ppo_config.discounting,
        learning_rate=ppo_config.learning_rate,
        entropy_cost=ppo_config.entropy_cost,
        num_envs=ppo_config.num_envs,
        batch_size=ppo_config.batch_size,
        max_grad_norm=ppo_config.max_grad_norm,
        clipping_epsilon=ppo_config.clipping_epsilon,
        seed=cfg['training']['seed'],
        network_factory=network_factory,
        save_checkpoint_path=ckpt_dir,
        restore_checkpoint_path=_LOAD_CHECKPOINT.value,
    )

    make_inference_fn, params, _ = train_fn(
        environment=wrapped_env,
        progress_fn=progress,
    )

    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)

    if len(times) > 1:
        print(f"Time to JIT: {times[1] - times[0]:.2f}s")
        print(f"Time to train: {times[-1] - times[1]:.2f}s")

    # Save final policy
    print("\nSaving policy...")
    import pickle
    policy_path = logdir / "policy.pkl"
    with open(policy_path, "wb") as f:
        pickle.dump(params, f)
    print(f"Policy saved to: {policy_path}")

    # Render videos
    if cfg['rendering']['render_videos']:
        print("\nRendering evaluation videos...")
        inference_fn = make_inference_fn(params, deterministic=True)
        jit_inference_fn = jax.jit(inference_fn)

        # Run rollout
        def do_rollout(rng):
            state = env.reset(rng)
            states = []
            for _ in range(ppo_config.episode_length):
                act_rng, rng = jax.random.split(rng)
                action = jit_inference_fn(state.obs, act_rng)[0]
                state = env.step(state, action)
                states.append(state.data)
            return states

        # Generate videos
        num_videos = cfg['rendering']['num_videos']
        video_paths = []

        for i in range(num_videos):
            rng = jax.random.PRNGKey(cfg['training']['seed'] + i)
            traj = do_rollout(rng)

            # Render frames
            frames = env.render(
                traj[::2],
                height=cfg['rendering']['render_height'],
                width=cfg['rendering']['render_width']
            )
            video_path = logdir / f"rollout_{i}.mp4"
            media.write_video(str(video_path), frames, fps=25)
            print(f"Video saved: {video_path}")
            video_paths.append(video_path)

        # Log videos to W&B
        if cfg['logging']['use_wandb']:
            print("\nLogging videos to W&B...")
            for i, video_path in enumerate(video_paths):
                wandb.log({
                    f"videos/rollout_{i}": wandb.Video(str(video_path), fps=25, format="mp4")
                })

    # Finish W&B run
    if cfg['logging']['use_wandb']:
        wandb.finish()
        print("W&B run finished")

    print(f"\nAll done! Results in: {logdir}")


if __name__ == "__main__":
    app.run(main)
