"""Brax PPO training script for WildRobot locomotion.

This script integrates Brax's battle-tested PPO trainer with WildRobotEnv,
providing a production-ready training infrastructure.

Usage:
    # Smoke test (quick verification)
    python playground_amp/train_brax_ppo.py --verify

    # Full training
    python playground_amp/train_brax_ppo.py --num-iterations 3000 --num-envs 2048

Key Features:
- Brax PPO trainer (GPU-optimized, JIT-compiled)
- Config system (YAML + CLI overrides)
- Progress callbacks and checkpointing
- WandB logging support
"""

from __future__ import annotations

import argparse
import functools
import os
import time
from typing import Any, Dict

import jax
import jax.numpy as jnp

from playground_amp.brax_wrapper import BraxWildRobotWrapper
from playground_amp.envs.wildrobot_env import EnvConfig


def load_config(config_path: str) -> Dict[str, Any]:
    """Load training configuration from YAML file."""
    try:
        import yaml

        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Warning: Could not load config from {config_path}: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(description="Brax PPO training for WildRobot")
    parser.add_argument(
        "--config",
        type=str,
        default="playground_amp/configs/wildrobot_phase3_training.yaml",
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Quick verification mode (10 iterations, 4 envs)",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=None,
        help="Number of parallel environments (overrides config)",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=None,
        help="Number of training iterations (overrides config)",
    )
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB logging")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/brax_ppo",
        help="Directory for saving checkpoints",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Brax PPO Training for WildRobot")
    print("=" * 70)

    # Load configs
    training_config = load_config(args.config)
    trainer_config = training_config.get("trainer", {})

    # Create environment config
    env_config_path = training_config.get(
        "env_config_path", "playground_amp/configs/wildrobot_env.yaml"
    )
    env_cfg = EnvConfig.from_file(env_config_path)

    # Apply overrides
    if args.verify:
        print("\n[VERIFY MODE] Running quick smoke test...")
        num_envs = 4
        num_iterations = 10
    else:
        num_envs = args.num_envs or trainer_config.get("num_envs", 16)
        num_iterations = args.num_iterations or trainer_config.get("total_updates", 100)

    env_cfg.num_envs = num_envs

    print(f"\nConfiguration:")
    print(f"  Environments: {num_envs}")
    print(f"  Iterations: {num_iterations}")
    print(f"  Observation dim: 44")
    print(f"  Action dim: 11")
    print(f"  WandB logging: {'disabled' if args.no_wandb else 'enabled'}")

    # Create Brax-wrapped environment
    print(f"\nInitializing environment...")
    env = BraxWildRobotWrapper(env_cfg, autoreset=True)
    print(f"✓ Environment created (backend: {env.backend})")

    # Try to import Brax PPO
    try:
        from brax.training.agents.ppo import train as ppo

        print(f"✓ Brax PPO imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Brax PPO: {e}")
        print(f"\nInstall Brax with: pip install brax>=0.10.0")
        return

    # PPO hyperparameters (2024 best practices)
    # Note: BraxWildRobotWrapper automatically enables Pure-JAX backend (use_jax=True)
    # This enables full JIT/vmap compatibility with Brax's training infrastructure
    ppo_kwargs = {
        "num_timesteps": num_iterations * 1000,  # Total env steps
        "num_evals": max(1, num_iterations // 20),  # Evaluate every 5% of training
        "reward_scaling": 1.0,
        "episode_length": 1000,
        "normalize_observations": True,
        "action_repeat": 1,
        "unroll_length": 20,  # Steps per rollout
        "num_minibatches": 4,
        "num_updates_per_batch": 4,
        "discounting": 0.99,
        "learning_rate": 3e-4,
        "entropy_cost": 1e-2,
        "num_envs": num_envs,
        "batch_size": min(1024, num_envs * 10),
        "seed": 0,
    }

    print(f"\nPPO Configuration:")
    for key, value in ppo_kwargs.items():
        print(f"  {key}: {value}")

    # Setup logging
    if not args.no_wandb:
        try:
            import wandb

            wandb.init(
                project="wildrobot-brax-ppo",
                name=f"brax_ppo_{int(time.time())}",
                config=ppo_kwargs,
            )
            print(f"✓ WandB logging enabled")
        except Exception as e:
            print(f"Warning: WandB init failed: {e}")
            args.no_wandb = True

    # Progress callback
    def progress_fn(num_steps, metrics):
        """Called periodically during training."""
        # Log more frequently for debugging
        if num_steps % 1000 == 0 or num_steps < 5000:
            reward = metrics.get("eval/episode_reward", 0.0)
            length = metrics.get("eval/episode_length", 0.0)
            policy_loss = metrics.get("training/policy_loss", 0.0)
            value_loss = metrics.get("training/value_loss", 0.0)
            print(f"Step {num_steps:,}: reward={reward:.2f}, length={length:.1f}, "
                  f"policy_loss={policy_loss:.4f}, value_loss={value_loss:.4f}")

            if not args.no_wandb:
                try:
                    import wandb

                    wandb.log(metrics, step=num_steps)
                except Exception:
                    pass

    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print(f"\nStarting training...")
    print(f"=" * 70)

    start_time = time.time()

    try:
        # Train with Brax PPO
        train_fn = functools.partial(
            ppo.train,
            **ppo_kwargs,
            progress_fn=progress_fn,
        )

        make_inference_fn, params, metrics = train_fn(environment=env)

        elapsed = time.time() - start_time
        print(f"\n" + "=" * 70)
        print(f"Training complete!")
        print(f"  Duration: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print(f"  Final reward: {metrics.get('eval/episode_reward', 0.0):.2f}")
        print(f"  Final length: {metrics.get('eval/episode_length', 0.0):.1f}")

        # Save checkpoint
        checkpoint_path = os.path.join(
            args.checkpoint_dir, f"final_iter_{num_iterations}.pkl"
        )
        try:
            import pickle

            with open(checkpoint_path, "wb") as f:
                pickle.dump(
                    {
                        "params": params,
                        "metrics": metrics,
                        "config": ppo_kwargs,
                    },
                    f,
                )
            print(f"✓ Checkpoint saved: {checkpoint_path}")
        except Exception as e:
            print(f"Warning: Failed to save checkpoint: {e}")

        if not args.no_wandb:
            try:
                import wandb

                wandb.finish()
            except Exception:
                pass

        print(f"=" * 70)

        return make_inference_fn, params, metrics

    except KeyboardInterrupt:
        print(f"\n\nTraining interrupted by user")
        return None, None, None
    except Exception as e:
        print(f"\n\n❌ Training failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return None, None, None


if __name__ == "__main__":
    main()
