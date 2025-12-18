#!/usr/bin/env python3
"""Entry point for AMP+PPO training on WildRobot.

This script provides a command-line interface for training a walking policy
using PPO with Adversarial Motion Priors (AMP) for natural motion learning.

Usage:
    # Basic training (using Brax PPO)
    python train_amp.py

    # With custom configuration
    python train_amp.py --num-envs 32 --iterations 5000

    # Training with AMP (custom loop)
    python train_amp.py --use-custom-loop --amp-weight 1.0

    # Quick smoke test
    python train_amp.py --verify

Architecture:
    This script supports two training modes:

    1. Brax PPO (default): Uses Brax's battle-tested PPO trainer with
       mujoco_playground wrapper for automatic Brax compatibility.

    2. Custom AMP+PPO loop (--use-custom-loop): Uses our custom training
       loop that integrates AMP discriminator naturally.
"""

from __future__ import annotations

import argparse
import functools
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import jax
import jax.numpy as jnp
import yaml
from ml_collections import config_dict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from playground_amp.envs.wildrobot_env import WildRobotEnv, WildRobotEnvState
from playground_amp.training.experiment_tracking import ExperimentTracker

# Default training config path
DEFAULT_CONFIG_PATH = (
    Path(__file__).parent / "configs" / "wildrobot_phase3_training.yaml"
)


def load_training_config(config_path: Optional[Path] = None) -> dict:
    """Load training configuration from YAML file.

    Args:
        config_path: Path to YAML config file. Uses default if not provided.

    Returns:
        Configuration dictionary.
    """
    path = config_path or DEFAULT_CONFIG_PATH
    with open(path, "r") as f:
        return yaml.safe_load(f)


def create_env_config(training_config: dict) -> config_dict.ConfigDict:
    """Create environment config from training config.

    Args:
        training_config: Full training configuration dict.

    Returns:
        ConfigDict for WildRobotEnv.
    """
    env_cfg = training_config.get("env", {})

    config = config_dict.ConfigDict()

    # Simulation timing
    config.ctrl_dt = env_cfg.get("ctrl_dt", 0.02)
    config.sim_dt = env_cfg.get("sim_dt", 0.002)

    # Robot parameters
    config.target_height = env_cfg.get("target_height", 0.45)
    config.min_height = env_cfg.get("min_height", 0.2)
    config.max_height = env_cfg.get("max_height", 0.7)

    # Velocity command
    config.min_velocity = env_cfg.get("min_velocity", 0.5)
    config.max_velocity = env_cfg.get("max_velocity", 1.0)

    # Action filtering
    config.use_action_filter = env_cfg.get("use_action_filter", True)
    config.action_filter_alpha = env_cfg.get("action_filter_alpha", 0.7)

    # Episode length
    config.max_episode_steps = env_cfg.get("max_episode_steps", 500)

    # Reward weights (from training config reward_weights section)
    reward_cfg = training_config.get("reward_weights", {})
    config.reward_weights = config_dict.ConfigDict()
    config.reward_weights.forward_velocity = reward_cfg.get("tracking_lin_vel", 1.0)
    config.reward_weights.healthy = reward_cfg.get("base_height", 0.1)
    config.reward_weights.action_rate = reward_cfg.get("action_rate", -0.01)
    config.reward_weights.joint_velocity = reward_cfg.get("torque_penalty", -0.001)

    return config


def create_env(config_path: Optional[Path] = None):
    """Create WildRobotEnv instance with config from training YAML.

    Args:
        config_path: Optional path to training config. Uses default if not provided.

    Returns:
        WildRobotEnv: Environment extending mjx_env.MjxEnv
    """
    training_config = load_training_config(config_path)
    env_config = create_env_config(training_config)
    env = WildRobotEnv(config=env_config)
    return env


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train WildRobot with PPO+AMP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Training configuration
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="Number of training iterations",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=16,
        help="Number of parallel environments",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )

    # PPO hyperparameters
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor",
    )
    parser.add_argument(
        "--clip-epsilon",
        type=float,
        default=0.2,
        help="PPO clipping parameter",
    )
    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=0.01,
        help="Entropy bonus coefficient",
    )

    # Training mode
    parser.add_argument(
        "--use-custom-loop",
        action="store_true",
        help="Use custom AMP+PPO training loop instead of Brax PPO",
    )

    # AMP configuration (only for custom loop)
    parser.add_argument(
        "--amp-weight",
        type=float,
        default=1.0,
        help="Weight for AMP reward (custom loop only)",
    )
    parser.add_argument(
        "--disc-lr",
        type=float,
        default=1e-4,
        help="Discriminator learning rate (custom loop only)",
    )

    # Logging and checkpoints
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Log metrics every N iterations",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/wildrobot",
        help="Directory for saving checkpoints",
    )

    # Quick test mode
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run quick verification (10 iterations)",
    )

    return parser.parse_args()


def train_with_brax_ppo(args):
    """Train using Brax's PPO trainer with mujoco_playground wrapper.

    This is the recommended approach - clean and efficient.
    """
    from brax.training.agents.ppo import train as ppo
    from mujoco_playground import wrapper

    # Create environment
    print("Creating WildRobotEnv...")
    env = create_env()
    print(
        f"✓ Environment created (obs_size={env.observation_size}, action_size={env.action_size})"
    )

    # PPO configuration
    num_timesteps = args.iterations * args.num_envs * 1000

    ppo_kwargs = {
        "num_timesteps": num_timesteps,
        "num_evals": max(1, args.iterations // 20),
        "reward_scaling": 1.0,
        "episode_length": 500,
        "normalize_observations": True,
        "action_repeat": 1,
        "unroll_length": 20,
        "num_minibatches": 4,
        "num_updates_per_batch": 4,
        "discounting": args.gamma,
        "learning_rate": args.lr,
        "entropy_cost": args.entropy_coef,
        "num_envs": args.num_envs,
        "batch_size": min(1024, args.num_envs * 10),
        "seed": args.seed,
    }

    print("\nPPO Configuration:")
    for key, value in ppo_kwargs.items():
        print(f"  {key}: {value}")

    # Progress callback
    times = [time.time()]

    def progress_fn(num_steps, metrics):
        times.append(time.time())
        if num_steps % 1000 == 0 or num_steps < 5000:
            reward = metrics.get("eval/episode_reward", 0.0)
            length = metrics.get("eval/episode_length", 0.0)
            print(f"Step {num_steps:,}: reward={reward:.2f}, length={length:.1f}")

    # Train
    print("\n" + "=" * 60)
    print("Starting Brax PPO training...")
    print("=" * 60 + "\n")

    train_fn = functools.partial(ppo.train, **ppo_kwargs, progress_fn=progress_fn)

    make_inference_fn, params, metrics = train_fn(
        environment=env,
        wrap_env_fn=wrapper.wrap_for_brax_training,  # Automatic Brax compatibility!
    )

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)

    if len(times) > 1:
        print(f"Time to JIT: {times[1] - times[0]:.2f}s")
        print(f"Time to train: {times[-1] - times[1]:.2f}s")

    # Save checkpoint
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.checkpoint_dir, "final_policy.pkl")

    checkpoint_data = {
        "params": params,
        "config": ppo_kwargs,
        "final_metrics": metrics,
    }

    with open(checkpoint_path, "wb") as f:
        pickle.dump(checkpoint_data, f)

    print(f"✓ Policy saved to: {checkpoint_path}")

    return make_inference_fn, params, metrics


def train_with_custom_loop(args):
    """Train using custom AMP+PPO training loop.

    Use this when you need AMP discriminator integration.
    """
    from playground_amp.training.trainer import (
        AMPPPOConfig,
        AMPPPOState,
        train_amp_ppo,
        TrainingMetrics,
    )

    # Load training config
    training_config = load_training_config()
    trainer_cfg = training_config.get("trainer", {})
    networks_cfg = training_config.get("networks", {})
    amp_cfg = training_config.get("amp", {})

    # Create environment
    print("Creating WildRobotEnv...")
    env = create_env()
    print(
        f"✓ Environment created (obs_size={env.observation_size}, action_size={env.action_size})"
    )

    # Create configuration from YAML + CLI overrides
    config = AMPPPOConfig(
        # Environment
        obs_dim=env.observation_size,
        action_dim=env.action_size,
        num_envs=args.num_envs,
        num_steps=trainer_cfg.get("rollout_steps", 10),
        max_episode_steps=training_config.get("env", {}).get("max_episode_steps", 500),
        # PPO hyperparameters
        learning_rate=args.lr,
        gamma=args.gamma,
        gae_lambda=trainer_cfg.get("gae_lambda", 0.95),
        clip_epsilon=args.clip_epsilon,
        value_loss_coef=trainer_cfg.get("value_loss_coef", 0.5),
        entropy_coef=args.entropy_coef,
        max_grad_norm=trainer_cfg.get("max_grad_norm", 0.5),
        # PPO training
        num_minibatches=trainer_cfg.get("num_minibatches", 4),
        update_epochs=trainer_cfg.get("epochs", 4),
        # Network architecture (from YAML)
        policy_hidden_dims=tuple(
            networks_cfg.get("policy_hidden_dims", [512, 256, 128])
        ),
        value_hidden_dims=tuple(networks_cfg.get("value_hidden_dims", [512, 256, 128])),
        log_std_min=networks_cfg.get("log_std_min", -20.0),
        log_std_max=networks_cfg.get("log_std_max", 2.0),
        # AMP configuration
        enable_amp=True,
        amp_reward_weight=args.amp_weight,
        disc_learning_rate=args.disc_lr,
        disc_updates_per_iter=amp_cfg.get("update_steps", 2),
        disc_batch_size=amp_cfg.get("batch_size", 512),
        gradient_penalty_weight=amp_cfg.get("gradient_penalty_weight", 5.0),
        # AMP discriminator architecture
        disc_hidden_dims=tuple(amp_cfg.get("discriminator_hidden", [1024, 512, 256])),
        # Reference motion
        ref_buffer_size=amp_cfg.get("ref_buffer_size", 1000),
        ref_seq_len=amp_cfg.get("ref_seq_len", 32),
        ref_motion_mode=amp_cfg.get("ref_motion_mode", "walking"),
        ref_motion_path=amp_cfg.get("dataset_path"),
        # Feature normalization
        normalize_amp_features=amp_cfg.get("normalize_features", True),
        # Training
        total_iterations=args.iterations,
        seed=args.seed,
        # Logging
        log_interval=args.log_interval,
        save_interval=trainer_cfg.get("save_interval_updates", 100),
        checkpoint_dir=args.checkpoint_dir,
    )

    # Create vmapped environment functions
    def step_fn(state, action):
        new_state = env.step(state, action)
        return new_state, new_state.obs, new_state.reward, new_state.done, {}

    def reset_fn(rng):
        return env.reset(rng)

    vmapped_step = jax.vmap(step_fn, in_axes=(0, 0))
    vmapped_reset = jax.vmap(reset_fn, in_axes=(0,))

    def batched_step_fn(states, actions):
        new_states, obs, rewards, dones, _ = vmapped_step(states, actions)
        return new_states, obs, rewards, dones, {}

    def batched_reset_fn(rng):
        rngs = jax.random.split(rng, config.num_envs)
        return vmapped_reset(rngs)

    print(f"✓ Environment functions created (vmapped for {config.num_envs} envs)")

    # Training callback
    def callback(iteration, state, metrics):
        if iteration % config.log_interval == 0:
            print(
                f"Iter {iteration:5d} | "
                f"Reward: {metrics.episode_reward:8.2f} | "
                f"AMP: {metrics.amp_reward_mean:6.4f} | "
                f"Steps/s: {metrics.env_steps_per_sec:6.0f}"
            )

    # Train
    print("\n" + "=" * 60)
    print("Starting custom AMP+PPO training...")
    print("=" * 60 + "\n")

    final_state = train_amp_ppo(
        env_step_fn=batched_step_fn,
        env_reset_fn=batched_reset_fn,
        config=config,
        callback=callback,
    )

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Total steps: {final_state.total_steps}")
    print("=" * 60)

    # Save checkpoint
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.checkpoint_dir, "final_amp_policy.pkl")

    checkpoint_data = {
        "policy_params": jax.device_get(final_state.policy_params),
        "value_params": jax.device_get(final_state.value_params),
        "disc_params": jax.device_get(final_state.disc_params),
        "config": config,
    }

    with open(checkpoint_path, "wb") as f:
        pickle.dump(checkpoint_data, f)

    print(f"✓ Policy saved to: {checkpoint_path}")

    return final_state


def main():
    """Main entry point."""
    args = parse_args()

    # Quick verify mode
    if args.verify:
        args.iterations = 10
        args.num_envs = 4
        args.log_interval = 1
        print("=" * 60)
        print("VERIFICATION MODE")
        print("Running quick smoke test")
        print("=" * 60)

    print(f"\n{'=' * 60}")
    print("WildRobot Training")
    print(f"{'=' * 60}")
    print(f"  Mode: {'Custom AMP+PPO' if args.use_custom_loop else 'Brax PPO'}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Environments: {args.num_envs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Seed: {args.seed}")
    print(f"{'=' * 60}\n")

    start_time = time.time()

    try:
        if args.use_custom_loop:
            train_with_custom_loop(args)
        else:
            train_with_brax_ppo(args)

        elapsed = time.time() - start_time
        print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

        if args.verify:
            print("\n✅ VERIFICATION PASSED!")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
