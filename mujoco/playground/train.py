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

from absl import app, flags, logging
from brax.training.agents.ppo import networks as ppo_networks, train as ppo
from etils import epath
from ml_collections import config_dict

# Add the playground directory to path so we can import wildrobot
sys.path.insert(0, str(Path(__file__).parent))

from wildrobot import config_utils, locomotion

# Project root for saving logs/checkpoints
PROJECT_ROOT = Path(__file__).parent.parent

# Configuration file
_CONFIG = flags.DEFINE_string(
    "config", None, "Path to YAML config file (default: default.yaml)"
)

# Environment flags (override config)
_TERRAIN = flags.DEFINE_string(
    "terrain", None, "Terrain type: 'flat' or 'rough' (overrides config)"
)

# Training flags (override config)
_NUM_TIMESTEPS = flags.DEFINE_integer(
    "num_timesteps", None, "Number of timesteps to train (overrides config)"
)
_NUM_ENVS = flags.DEFINE_integer(
    "num_envs", None, "Number of parallel environments (overrides config)"
)
_SEED = flags.DEFINE_integer("seed", None, "Random seed (overrides config)")
_EPISODE_LENGTH = flags.DEFINE_integer(
    "episode_length", None, "Episode length (overrides config)"
)

# PPO flags (override config)
_BATCH_SIZE = flags.DEFINE_integer("batch_size", None, "Batch size (overrides config)")
_LEARNING_RATE = flags.DEFINE_float(
    "learning_rate", None, "Learning rate (overrides config)"
)
_ENTROPY_COST = flags.DEFINE_float(
    "entropy_cost", None, "Entropy cost (overrides config)"
)

# Checkpointing
_LOAD_CHECKPOINT = flags.DEFINE_string(
    "load_checkpoint", None, "Path to checkpoint to load"
)

# Rendering
_RENDER = flags.DEFINE_boolean(
    "render", None, "Render videos after training (overrides config)"
)

# W&B Logging (override config)
_USE_WANDB = flags.DEFINE_boolean(
    "use_wandb", None, "Use Weights & Biases for logging (overrides config)"
)
_WANDB_PROJECT = flags.DEFINE_string(
    "wandb_project", None, "W&B project name (overrides config)"
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
        overrides["env.terrain"] = _TERRAIN.value
    if _NUM_TIMESTEPS.value is not None:
        overrides["training.num_timesteps"] = _NUM_TIMESTEPS.value
    if _NUM_ENVS.value is not None:
        overrides["ppo.num_envs"] = _NUM_ENVS.value
    if _SEED.value is not None:
        overrides["training.seed"] = _SEED.value
    if _EPISODE_LENGTH.value is not None:
        overrides["training.episode_length"] = _EPISODE_LENGTH.value
    if _BATCH_SIZE.value is not None:
        overrides["ppo.batch_size"] = _BATCH_SIZE.value
    if _LEARNING_RATE.value is not None:
        overrides["ppo.learning_rate"] = _LEARNING_RATE.value
    if _ENTROPY_COST.value is not None:
        overrides["ppo.entropy_cost"] = _ENTROPY_COST.value
    if _RENDER.value is not None:
        overrides["rendering.render_videos"] = _RENDER.value
    if _USE_WANDB.value is not None:
        overrides["logging.use_wandb"] = _USE_WANDB.value
    if _WANDB_PROJECT.value is not None:
        overrides["logging.wandb_project"] = _WANDB_PROJECT.value

    # Apply overrides
    cfg = config_utils.override_config(cfg, overrides)

    # Extract values from config
    terrain = cfg["env"]["terrain"]
    task_name = f"wildrobot_{terrain}"

    print(f"Training WildRobot Locomotion")
    print(f"  Terrain: {terrain}")
    print(f"  Velocity range: 0.0-1.0 m/s")
    print(f"  Config: {_CONFIG.value or 'default.yaml'}")

    # Create environment config
    env_config = config_dict.ConfigDict()
    env_config.ctrl_dt = cfg["env"]["ctrl_dt"]
    env_config.sim_dt = cfg["env"]["sim_dt"]

    # Get terminal state params (nested structure matching loco-mujoco)
    terminal_params = cfg["env"].get("terminal_state_params", {})
    env_config.min_height = terminal_params.get("min_height", 0.2)
    env_config.max_height = terminal_params.get("max_height", 0.7)

    # Create environment
    env = locomotion.WildRobotLocomotion(task=task_name, config=env_config)

    print(f"\nEnvironment: {env.__class__.__name__}")
    print(f"  Observation size: {env.observation_size}")
    print(f"  Action size: {env.action_size}")

    # PPO training parameters
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

    # Network parameters
    network_config = config_dict.ConfigDict()
    network_config.policy_hidden_layer_sizes = cfg["network"]["policy_hidden_layers"]
    network_config.value_hidden_layer_sizes = cfg["network"]["value_hidden_layers"]

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

    # Save checkpoints under playground/checkpoints/{exp_name}
    playground_dir = Path(__file__).parent
    ckpt_dir = playground_dir / "checkpoints" / exp_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Logs go to playground/training_logs
    logdir = epath.Path(PROJECT_ROOT) / "playground" / "training_logs" / exp_name
    logdir.mkdir(parents=True, exist_ok=True)

    print(f"\nExperiment: {exp_name}")
    print(f"Checkpoints: {ckpt_dir}")
    print(f"Logs: {logdir}")

    # Create training log file
    training_log_file = logdir / "training_log.txt"
    metrics_log_file = logdir / "metrics.jsonl"

    # Write initial header to log file
    with open(training_log_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write(f"WildRobot Training Log - {exp_name}\n")
        f.write("=" * 80 + "\n")
        f.write(f"Start Time: {now.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Config: {_CONFIG.value or 'default.yaml'}\n")
        f.write(f"Terrain: {terrain}\n")
        f.write(f"Total Timesteps: {ppo_config.num_timesteps:,}\n")
        f.write(f"Num Environments: {ppo_config.num_envs}\n")
        f.write(f"Episode Length: {ppo_config.episode_length}\n")
        f.write("=" * 80 + "\n\n")

    # Initialize W&B if enabled
    if cfg["logging"]["use_wandb"]:
        wandb_config = {
            "terrain": terrain,
            "num_timesteps": ppo_config.num_timesteps,
            "num_envs": ppo_config.num_envs,
            "batch_size": ppo_config.batch_size,
            "learning_rate": ppo_config.learning_rate,
            "episode_length": ppo_config.episode_length,
            "network_policy": network_config.policy_hidden_layer_sizes,
            "network_value": network_config.value_hidden_layer_sizes,
        }

        wandb.init(
            project=cfg["logging"]["wandb_project"],
            entity=cfg["logging"]["wandb_entity"],
            name=exp_name,
            config=wandb_config,
            dir=str(logdir),
        )
        print(f"W&B initialized: {wandb.run.url}")

    # Save both the config file path and overrides
    with open(ckpt_dir / "config.json", "w") as f:
        json.dump(
            {
                "config_file": _CONFIG.value or "default.yaml",
                "overrides": overrides,
                "final_config": cfg,
            },
            f,
            indent=2,
        )

    # Create network factory
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=network_config.policy_hidden_layer_sizes,
        value_hidden_layer_sizes=network_config.value_hidden_layer_sizes,
    )

    # Progress callback
    times = [time.monotonic()]
    final_metrics = {}  # Store final metrics for saving in checkpoint

    # Get validation metrics to track from config
    validation_metrics = cfg.get("validation", {}).get("metrics", {})
    if not validation_metrics:
        # Fallback to default metrics if not specified in config
        validation_metrics = {
            "reward": "eval/episode_reward",
            "episode_length": "eval/avg_episode_length",
        }

    def progress(num_steps, metrics):
        nonlocal final_metrics  # Update the outer scope variable
        times.append(time.monotonic())
        elapsed = times[-1] - times[1] if len(times) > 1 else 0

        # Extract key metrics for display
        eval_reward = metrics.get("eval/episode_reward", 0.0)
        eval_length = metrics.get("eval/avg_episode_length", 0.0)

        # Extract locomotion metrics (Brax provides SUMMED metrics, not averages)
        # We need to divide by episode length to get per-step averages
        episode_forward_vel = metrics.get("eval/episode_forward_velocity", 0.0)
        episode_height = metrics.get("eval/episode_height", 0.0)
        episode_velocity_cmd = metrics.get("eval/episode_velocity_command", 0.0)
        episode_success = metrics.get("eval/episode_success", 0.0)
        episode_distance = metrics.get("eval/episode_distance_walked", 0.0)

        # Convert summed metrics to averages
        avg_forward_vel = episode_forward_vel / max(eval_length, 1.0)
        avg_height = episode_height / max(eval_length, 1.0)
        avg_velocity_cmd = episode_velocity_cmd / max(eval_length, 1.0)
        # Calculate actual success rate as percentage (successful steps / total steps * 100)
        success_rate = (episode_success / max(eval_length, 1.0)) * 100.0
        avg_distance = episode_distance  # Already total distance per episode

        # Save final metrics based on validation config
        # (will be overwritten each time, so last call has final values)
        final_metrics = {"training_timesteps": num_steps}

        # Add all configured validation metrics
        for metric_name, metric_key in validation_metrics.items():
            if metric_key in metrics:
                final_metrics[metric_key] = metrics[metric_key]

        # Always include the raw metric keys for backward compatibility
        final_metrics["eval/episode_reward"] = eval_reward
        final_metrics["eval/avg_episode_length"] = eval_length

        # Include locomotion metrics for comprehensive tracking
        final_metrics["eval/avg_forward_velocity"] = avg_forward_vel
        final_metrics["eval/avg_height"] = avg_height
        final_metrics["eval/avg_velocity_command"] = avg_velocity_cmd
        final_metrics["eval/success_rate"] = success_rate
        final_metrics["eval/avg_distance_walked"] = avg_distance

        log_msg = (
            f"Step {num_steps:,}: reward={eval_reward:.3f}, length={eval_length:.1f}, "
            f"vel={avg_forward_vel:.3f}m/s, dist={avg_distance:.2f}m, height={avg_height:.3f}m, elapsed={elapsed:.1f}s"
        )
        print(log_msg)

        # Write to training log file
        with open(training_log_file, "a") as f:
            f.write(f"{log_msg}\n")

        # Write metrics to JSONL file for easy parsing
        with open(metrics_log_file, "a") as f:
            metrics_json = {
                "step": num_steps,
                "timestamp": time.time(),
                "elapsed": elapsed,
                "reward": float(eval_reward),
                "episode_length": float(eval_length),
                "forward_velocity": float(avg_forward_vel),
                "distance_walked": float(avg_distance),
                "height": float(avg_height),
                "velocity_command": float(avg_velocity_cmd),
                "success_rate": float(success_rate),
            }
            f.write(json.dumps(metrics_json) + "\n")

        # Log to W&B if enabled
        if cfg["logging"]["use_wandb"]:
            log_dict = {
                "train/step": num_steps,
                "train/elapsed_time": elapsed,
                "eval/episode_reward": eval_reward,
                "eval/avg_episode_length": eval_length,
            }

            # Add locomotion metrics
            log_dict["eval/avg_forward_velocity"] = avg_forward_vel
            log_dict["eval/avg_height"] = avg_height
            log_dict["eval/avg_velocity_command"] = avg_velocity_cmd
            log_dict["eval/success_rate"] = success_rate
            log_dict["eval/avg_distance_walked"] = avg_distance

            # Velocity tracking error (how well robot follows commands)
            vel_error = abs(avg_forward_vel - avg_velocity_cmd)
            log_dict["eval/velocity_tracking_error"] = vel_error

            # Log additional training metrics if available
            for key, value in metrics.items():
                if key.startswith("training/") or key.startswith("losses/"):
                    log_dict[key] = value

            # Log learning rate and other optimizer metrics
            if "learning_rate" in metrics:
                log_dict["train/learning_rate"] = metrics["learning_rate"]

            wandb.log(log_dict, step=num_steps)

    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")

    # Import wrapper function (don't pre-wrap the environment)
    from mujoco_playground import wrapper

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
        seed=cfg["training"]["seed"],
        network_factory=network_factory,
        save_checkpoint_path=ckpt_dir,
        restore_checkpoint_path=_LOAD_CHECKPOINT.value,
    )

    make_inference_fn, params, _ = train_fn(
        environment=env,  # Pass unwrapped environment
        wrap_env_fn=wrapper.wrap_for_brax_training,  # Let Brax wrap it
        progress_fn=progress,
    )

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)

    if len(times) > 1:
        print(f"Time to JIT: {times[1] - times[0]:.2f}s")
        print(f"Time to train: {times[-1] - times[1]:.2f}s")

    # Save final policy
    print("\nSaving policy...")
    import pickle

    final_checkpoint = ckpt_dir / "final_policy.pkl"

    # Include final metrics in checkpoint for automatic validation
    checkpoint_data = {
        "params": params,
        "config": cfg,
        "final_metrics": final_metrics,  # Add final training metrics
    }

    with open(final_checkpoint, "wb") as f:
        pickle.dump(checkpoint_data, f)

    print(f"Policy saved to: {final_checkpoint}")

    # Write final summary to log file
    with open(training_log_file, "a") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write("Training Complete!\n")
        f.write("=" * 80 + "\n")
        f.write(f"End Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if len(times) > 1:
            f.write(f"Time to JIT: {times[1] - times[0]:.2f}s\n")
            f.write(f"Time to train: {times[-1] - times[1]:.2f}s\n")
        f.write(f"\nPolicy saved to: {final_checkpoint}\n")
        f.write("\nFinal Training Metrics:\n")
        for key, value in final_metrics.items():
            f.write(f"  {key}: {value}\n")
        f.write("=" * 80 + "\n")

    # Print final metrics for reference
    if final_metrics:
        print(f"\nFinal Training Metrics:")
        print(
            f"  eval/episode_reward: {final_metrics.get('eval/episode_reward', 'N/A')}"
        )
        print(
            f"  eval/avg_episode_length: {final_metrics.get('eval/avg_episode_length', 'N/A')}"
        )
        print(f"  training_timesteps: {final_metrics.get('training_timesteps', 'N/A')}")

    # Render videos (Option 2: Render only at the end, like loco-mujoco)
    if cfg["rendering"]["render_videos"]:
        print("\n" + "=" * 60)
        print("Rendering final policy video...")
        print("=" * 60)

        try:
            from visualize_policy import create_env, load_policy, visualize_single_env

            video_path = logdir / "final_policy.mp4"

            # Create environment and load policy
            env = create_env()
            make_policy, params = load_policy(str(final_checkpoint), env)

            # Render a single video of the final policy
            visualize_single_env(
                env=env,
                make_policy=make_policy,
                params=params,
                output_path=str(video_path),
                n_steps=ppo_config.episode_length * 5,
                deterministic=True,
                seed=cfg["training"]["seed"],
                height=cfg["rendering"]["render_height"],
                width=cfg["rendering"]["render_width"],
                camera=None,  # WildRobot has no named cameras, use free camera
                fps=50,
            )

            # Log video to W&B
            if cfg["logging"]["use_wandb"]:
                print("\nLogging video to W&B...")
                wandb.log(
                    {
                        "videos/final_policy": wandb.Video(
                            str(video_path), fps=50, format="mp4"
                        )
                    }
                )

            print(f"\n✓ Video rendering successful: {video_path}")

        except Exception as e:
            import traceback

            print(f"\nWarning: Video recording failed: {e}")
            print("Full traceback:")
            traceback.print_exc()
            print("\nTraining completed successfully. Video recording skipped.")
            print(f"To render video later, run:")
            print(
                f"  python visualize_policy.py --checkpoint {final_checkpoint} --output videos/policy.mp4"
            )
    else:
        print("\nVideo rendering disabled (render_videos=false in config)")
        print(f"To render video later, run:")
        print(
            f"  python visualize_policy.py --checkpoint {final_checkpoint} --output videos/policy.mp4"
        )

    # Finish W&B run
    if cfg["logging"]["use_wandb"]:
        wandb.finish()
        print("W&B run finished")

    print(f"\nAll done! Results in: {logdir}")


if __name__ == "__main__":
    app.run(main)
