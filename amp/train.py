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
"""Train WildRobot AMP with PPO for human-like walking."""

import datetime
import functools
import json
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jp
import mediapy as media
import mujoco
import wandb
import yaml

from absl import app, flags, logging
from brax.training.agents.ppo import networks as ppo_networks, train as ppo
from etils import epath
from ml_collections import config_dict

# Add amp directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from amp import walk

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Flags
_CONFIG = flags.DEFINE_string(
    "config", "phase1_contact.yaml", "Config file name (in amp/)"
)
_QUICK_VERIFY = flags.DEFINE_boolean(
    "quick_verify", False, "Run quick verification mode (~20 seconds)"
)
_TERRAIN = flags.DEFINE_string(
    "terrain", None, "Terrain type: 'flat' or 'rough' (overrides config)"
)
_NUM_TIMESTEPS = flags.DEFINE_integer(
    "num_timesteps", None, "Number of timesteps (overrides config)"
)
_NUM_ENVS = flags.DEFINE_integer(
    "num_envs", None, "Number of parallel environments (overrides config)"
)
_SEED = flags.DEFINE_integer("seed", None, "Random seed (overrides config)")
_LOAD_CHECKPOINT = flags.DEFINE_string(
    "load_checkpoint", None, "Path to checkpoint to load"
)
_USE_WANDB = flags.DEFINE_boolean(
    "use_wandb", None, "Use Weights & Biases (overrides config)"
)

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def apply_quick_verify_overrides(cfg: dict) -> dict:
    """Apply quick_verify overrides to config."""
    qv = cfg.get("quick_verify", {})

    # Override training params
    cfg["training"]["num_timesteps"] = qv.get("num_timesteps", 10000)
    cfg["training"]["num_evals"] = qv.get("num_evals", 2)
    cfg["training"]["episode_length"] = qv.get("episode_length", 100)

    # Override PPO params
    cfg["ppo"]["num_envs"] = qv.get("num_envs", 128)
    cfg["ppo"]["num_eval_envs"] = qv.get("num_eval_envs", 16)

    # Override logging/rendering
    cfg["logging"]["use_wandb"] = qv.get("use_wandb", False)
    cfg["rendering"]["render_videos"] = qv.get("render_videos", False)

    print("\n" + "="*80)
    print("🔍 QUICK VERIFY MODE ENABLED")
    print("="*80)
    print(f"  Timesteps: {cfg['training']['num_timesteps']:,} (~20 seconds)")
    print(f"  Evals: {cfg['training']['num_evals']}")
    print(f"  Episode length: {cfg['training']['episode_length']}")
    print(f"  Num envs: {cfg['ppo']['num_envs']}")
    print(f"  W&B logging: {cfg['logging']['use_wandb']}")
    print(f"  Video rendering: {cfg['rendering']['render_videos']}")
    print("="*80 + "\n")

    return cfg


def main(argv):
    del argv

    logging.set_verbosity(logging.INFO)

    # Load config
    config_path = Path(__file__).parent / _CONFIG.value
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = load_config(config_path)

    # Apply quick verify if requested
    if _QUICK_VERIFY.value or cfg.get("quick_verify", {}).get("enabled", False):
        cfg = apply_quick_verify_overrides(cfg)

    # Apply CLI overrides
    if _TERRAIN.value:
        cfg["env"]["terrain"] = _TERRAIN.value
    if _NUM_TIMESTEPS.value:
        cfg["training"]["num_timesteps"] = _NUM_TIMESTEPS.value
    if _NUM_ENVS.value:
        cfg["ppo"]["num_envs"] = _NUM_ENVS.value
    if _SEED.value:
        cfg["training"]["seed"] = _SEED.value
    if _USE_WANDB.value is not None:
        cfg["logging"]["use_wandb"] = _USE_WANDB.value

    # Extract config values
    terrain = cfg["env"]["terrain"]
    task_name = f"wildrobot_{terrain}"

    print(f"\n{'='*80}")
    print(f"WildRobot AMP Training")
    print(f"{'='*80}")
    print(f"  Config: {_CONFIG.value}")
    print(f"  Terrain: {terrain}")
    print(f"  Task: {task_name}")
    print(f"{'='*80}\n")

    # Create environment config
    env_config = config_dict.ConfigDict()
    env_config.ctrl_dt = cfg["env"]["ctrl_dt"]
    env_config.sim_dt = cfg["env"]["sim_dt"]
    env_config.velocity_command_mode = cfg["env"]["velocity_command_mode"]
    env_config.min_velocity = cfg["env"]["min_velocity"]
    env_config.max_velocity = cfg["env"]["max_velocity"]
    env_config.use_action_filter = cfg["env"]["use_action_filter"]
    env_config.action_filter_alpha = cfg["env"]["action_filter_alpha"]
    env_config.use_phase_signal = cfg["env"]["use_phase_signal"]
    env_config.phase_period = cfg["env"]["phase_period"]
    env_config.num_phase_clocks = cfg["env"]["num_phase_clocks"]
    env_config.min_height = cfg["env"]["min_height"]
    env_config.max_height = cfg["env"]["max_height"]

    # Pass reward weights to environment
    env_config.reward_weights = cfg["reward_weights"]

    # Create environment
    env = walk.WildRobotWalk(task=task_name, config=env_config)

    print(f"Environment: {env.__class__.__name__}")
    print(f"  Observation size: {env.observation_size}")
    print(f"  Action size: {env.action_size}")
    print(f"  Velocity range: {env_config.min_velocity}-{env_config.max_velocity} m/s")

    # Show active reward components
    print(f"\nActive Reward Components:")
    for name, weight in cfg["reward_weights"].items():
        if weight != 0.0 and name not in ["tracking_sigma"]:
            print(f"  {name}: {weight}")

    # PPO config
    ppo_config = config_dict.ConfigDict()
    ppo_config.num_timesteps = cfg["training"]["num_timesteps"]
    ppo_config.num_evals = cfg["training"]["num_evals"]
    ppo_config.reward_scaling = cfg["ppo"]["reward_scaling"]
    ppo_config.episode_length = cfg["training"]["episode_length"]
    ppo_config.normalize_observations = cfg["ppo"]["normalize_observations"]
    ppo_config.action_repeat = cfg["ppo"]["action_repeat"]
    ppo_config.unroll_length = cfg["ppo"]["unroll_length"]
    ppo_config.num_minibatches = cfg["ppo"]["num_minibatches"]
    ppo_config.num_updates_per_batch = cfg["ppo"]["num_updates_per_batch"]
    ppo_config.discounting = cfg["ppo"]["discounting"]
    ppo_config.learning_rate = cfg["ppo"]["learning_rate"]
    ppo_config.entropy_cost = cfg["ppo"]["entropy_cost"]
    ppo_config.num_envs = cfg["ppo"]["num_envs"]
    ppo_config.batch_size = cfg["ppo"]["batch_size"]
    ppo_config.max_grad_norm = cfg["ppo"]["max_grad_norm"]
    ppo_config.clipping_epsilon = cfg["ppo"]["clipping_epsilon"]

    # Network config
    network_config = config_dict.ConfigDict()
    network_config.policy_hidden_layer_sizes = cfg["network"]["policy_hidden_layers"]
    network_config.value_hidden_layer_sizes = cfg["network"]["value_hidden_layers"]

    print(f"\nTraining Configuration:")
    print(f"  Timesteps: {ppo_config.num_timesteps:,}")
    print(f"  Environments: {ppo_config.num_envs}")
    print(f"  Batch size: {ppo_config.batch_size}")
    print(f"  Learning rate: {ppo_config.learning_rate}")
    print(f"  Episode length: {ppo_config.episode_length}")
    print(f"  Network: {network_config.policy_hidden_layer_sizes}")

    # Setup experiment directory
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    config_name = Path(_CONFIG.value).stem  # e.g., "phase1_contact"

    if _QUICK_VERIFY.value:
        exp_name = f"quickverify_{config_name}_{timestamp}"
    else:
        exp_name = f"{config_name}_{terrain}_{timestamp}"

    amp_dir = Path(__file__).parent
    ckpt_dir = amp_dir / "checkpoints" / exp_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logdir = epath.Path(PROJECT_ROOT) / "amp" / "logs" / exp_name
    logdir.mkdir(parents=True, exist_ok=True)

    print(f"\nExperiment: {exp_name}")
    print(f"  Checkpoints: {ckpt_dir}")
    print(f"  Logs: {logdir}\n")

    # Save config
    with open(ckpt_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    # Initialize W&B
    if cfg["logging"]["use_wandb"]:
        wandb_tags = cfg["logging"].get("wandb_tags", [])
        if _QUICK_VERIFY.value:
            wandb_tags.append("quick_verify")

        wandb.init(
            project=cfg["logging"]["wandb_project"],
            entity=cfg["logging"]["wandb_entity"],
            name=exp_name,
            config=cfg,
            tags=wandb_tags,
            dir=str(logdir),
        )
        print(f"W&B initialized: {wandb.run.url}\n")

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

        # Extract metrics
        eval_reward = metrics.get("eval/episode_reward", 0.0)
        eval_length = metrics.get("eval/avg_episode_length", 0.0)
        forward_vel = metrics.get("eval/episode_forward_velocity", 0.0) / max(eval_length, 1.0)
        height = metrics.get("eval/episode_height", 0.0) / max(eval_length, 1.0)
        success = metrics.get("eval/episode_success", 0.0)
        distance = metrics.get("eval/episode_distance_walked", 0.0)

        # Progress bar
        progress_pct = (num_steps / ppo_config.num_timesteps) * 100
        sps = num_steps / elapsed if elapsed > 0 else 0
        eta_seconds = (ppo_config.num_timesteps - num_steps) / sps if sps > 0 else 0
        eta = str(datetime.timedelta(seconds=int(eta_seconds)))

        print(f"Step {num_steps:,}/{ppo_config.num_timesteps:,} ({progress_pct:.1f}%) | "
              f"Reward: {eval_reward:.2f} | Vel: {forward_vel:.3f} m/s | "
              f"Height: {height:.3f}m | Success: {success:.2%} | "
              f"SPS: {sps:.0f} | ETA: {eta}")

        # Log to W&B
        if cfg["logging"]["use_wandb"]:
            wandb.log({
                "steps": num_steps,
                "reward": eval_reward,
                "forward_velocity": forward_vel,
                "height": height,
                "success_rate": success,
                "distance_walked": distance,
                **metrics
            })

    # Train
    print("Starting training...\n")
    make_inference_fn, params, _ = ppo.train(
        environment=env,
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
        seed=cfg["training"]["seed"],
        network_factory=network_factory,
        progress_fn=progress,
        max_grad_norm=ppo_config.max_grad_norm,  # Fixed: was max_gradient_norm
        clipping_epsilon=ppo_config.clipping_epsilon,
    )

    print(f"\n{'='*80}")
    print("Training completed!")
    print(f"{'='*80}\n")

    # Save final checkpoint
    if not _QUICK_VERIFY.value:
        final_ckpt_path = ckpt_dir / "final_policy.pkl"
        with open(final_ckpt_path, "wb") as f:
            import pickle
            pickle.dump(params, f)
        print(f"Saved final checkpoint: {final_ckpt_path}\n")

    # Render videos if enabled
    if cfg["rendering"]["render_videos"] and not _QUICK_VERIFY.value:
        print("Rendering evaluation videos...")
        # TODO: Add video rendering code
        print("Video rendering not yet implemented.\n")

    if cfg["logging"]["use_wandb"]:
        wandb.finish()


if __name__ == "__main__":
    app.run(main)
