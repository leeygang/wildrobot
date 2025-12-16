"""Unified Brax PPO training script for WildRobot locomotion with AMP support.

This script integrates Brax's battle-tested PPO trainer with WildRobotEnv,
providing production-ready training infrastructure with optional AMP discriminator support.

Usage:
    # Quick verification (smoke test)
    python playground_amp/train.py --verify

    # CPU training (small scale)
    python playground_amp/train.py --num-iterations 100 --num-envs 16

    # GPU training (production scale)
    python playground_amp/train.py --num-iterations 3000 --num-envs 2048

    # Enable AMP for natural motion learning
    python playground_amp/train.py --enable-amp --amp-weight 1.0

Key Features:
- Brax PPO trainer (GPU-optimized, JIT-compiled)
- AMP discriminator support (Task 5 ready)
- Config system (YAML + CLI overrides)
- Progress callbacks and checkpointing
- WandB logging support
- Pure-JAX backend (full JIT/vmap compatibility)
"""

from __future__ import annotations

import argparse
import functools
import os
import time
from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np

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


def apply_overrides(cfg: dict, overrides: list):
    """Apply list of overrides of the form ['key.subkey=value'] to cfg dict."""
    for ov in overrides or []:
        if '=' not in ov:
            continue
        key, val = ov.split('=', 1)
        # nested keys with dot notation
        parts = key.split('.')
        d = cfg
        for p in parts[:-1]:
            if p not in d or not isinstance(d[p], dict):
                d[p] = {}
            d = d[p]
        # try to cast value to int/float/bool if appropriate
        s = val.strip()
        if s.lower() in ('true', 'false'):
            v = s.lower() == 'true'
        else:
            try:
                if '.' in s:
                    v = float(s)
                    # if integer-like, cast
                    if v == int(v):
                        v = int(v)
                else:
                    v = int(s)
            except Exception:
                v = s
        d[parts[-1]] = v
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Brax PPO training for WildRobot with AMP support")
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
    parser.add_argument(
        "--enable-amp",
        action="store_true",
        help="Enable AMP discriminator for natural motion learning",
    )
    parser.add_argument(
        "--amp-weight",
        type=float,
        default=None,
        help="AMP reward weight (default: 1.0, equal to task rewards)",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable WandB logging",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/brax_ppo",
        help="Directory for saving checkpoints",
    )
    parser.add_argument(
        "--set",
        action="append",
        help="Override config key, format: key.subkey=value (can be used multiple times)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Brax PPO Training for WildRobot" + (" + AMP" if args.enable_amp else ""))
    print("=" * 70)

    # Load configs
    training_config = load_config(args.config)
    trainer_config = training_config.get("trainer", {})

    # Apply CLI overrides to trainer config
    if args.set:
        trainer_config = apply_overrides(trainer_config, args.set)

    # Create environment config
    env_config_path = training_config.get(
        "env_config_path", "playground_amp/configs/wildrobot_env.yaml"
    )
    env_cfg = EnvConfig.from_file(env_config_path)

    # Apply quick_verify overrides if requested
    if args.verify and 'quick_verify' in training_config:
        qv = training_config['quick_verify']
        # Apply env overrides
        env_q = qv.get('env', {})
        for k, v in env_q.items():
            try:
                setattr(env_cfg, k, v)
            except Exception:
                env_cfg.__dict__[k] = v
        # Apply trainer overrides
        trainer_q = qv.get('trainer', {})
        for k, v in trainer_q.items():
            trainer_config[k] = v

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
    print(f"  AMP enabled: {args.enable_amp}")
    if args.enable_amp:
        amp_weight = args.amp_weight or training_config.get('amp', {}).get('weight', 1.0)
        print(f"  AMP weight: {amp_weight}")
    print(f"  WandB logging: {'disabled' if args.no_wandb else 'enabled'}")

    # Initialize AMP discriminator if requested (LEGACY CODE - NOT USED)
    # Note: AMP integration now uses AMPRewardWrapper instead
    amp_disc = None
    ref_buffer = None

    # Get AMP weight (used by wrapper)
    amp_weight = args.amp_weight or training_config.get('amp', {}).get('weight', 0.5)

    # Create Brax-wrapped environment
    print(f"\nInitializing environment...")
    env = BraxWildRobotWrapper(env_cfg, autoreset=True)
    print(f"✓ Environment created (backend: {env.backend})")

    # Wrap with AMP if requested
    if args.enable_amp:
        try:
            from playground_amp.amp.amp_wrapper import AMPRewardWrapper

            # Get AMP config
            amp_config = training_config.get('amp', {})

            env = AMPRewardWrapper(
                base_env=env,
                amp_weight=amp_weight,
                discriminator_lr=amp_config.get('discriminator_lr', 1e-4),
                obs_dim=44,
                ref_motion_mode="walking",  # Use synthetic walking for now
                ref_motion_num_sequences=200,
                seed=trainer_config.get("seed", 0),
            )
            print(f"✓ AMP wrapper enabled")
            print(f"  Note: AMP discriminator reward is added to task rewards")
            print(f"  Discriminator is pre-initialized (not trained during PPO)")
        except Exception as e:
            print(f"Warning: Failed to wrap environment with AMP: {e}")
            print(f"  Continuing without AMP support")
            args.enable_amp = False

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
        "episode_length": trainer_config.get("episode_length", 1000),
        "normalize_observations": True,
        "action_repeat": 1,
        "unroll_length": trainer_config.get("rollout_steps", 20),
        "num_minibatches": trainer_config.get("num_minibatches", 4),
        "num_updates_per_batch": trainer_config.get("epochs", 4),
        "discounting": trainer_config.get("gamma", 0.99),
        "learning_rate": trainer_config.get("lr", 3e-4),
        "entropy_cost": trainer_config.get("entropy_coef", 1e-2),
        "num_envs": num_envs,
        "batch_size": min(1024, num_envs * 10),
        "seed": trainer_config.get("seed", 0),
    }

    print(f"\nPPO Configuration:")
    for key, value in ppo_kwargs.items():
        print(f"  {key}: {value}")

    # Setup logging
    if not args.no_wandb:
        try:
            import wandb

            wandb_config = {**ppo_kwargs, "amp_enabled": args.enable_amp}
            if args.enable_amp:
                wandb_config["amp_weight"] = amp_weight

            wandb.init(
                project="wildrobot-brax-ppo",
                name=f"brax_ppo{'_amp' if args.enable_amp else ''}_{int(time.time())}",
                config=wandb_config,
            )
            print(f"✓ WandB logging enabled")
        except Exception as e:
            print(f"Warning: WandB init failed: {e}")
            args.no_wandb = True

    # Progress callback with AMP metrics
    def progress_fn(num_steps, metrics):
        """Called periodically during training."""
        # Log more frequently for debugging
        if num_steps % 1000 == 0 or num_steps < 5000:
            reward = metrics.get("eval/episode_reward", 0.0)
            length = metrics.get("eval/episode_length", 0.0)
            policy_loss = metrics.get("training/policy_loss", 0.0)
            value_loss = metrics.get("training/value_loss", 0.0)

            log_msg = (f"Step {num_steps:,}: reward={reward:.2f}, length={length:.1f}, "
                      f"policy_loss={policy_loss:.4f}, value_loss={value_loss:.4f}")

            # Add AMP metrics if enabled
            if args.enable_amp and amp_disc is not None:
                amp_reward = metrics.get("training/amp_reward", 0.0)
                disc_acc = metrics.get("training/discriminator_accuracy", 0.0)
                log_msg += f", amp_reward={amp_reward:.4f}, disc_acc={disc_acc:.3f}"

            print(log_msg)

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

        if args.enable_amp:
            amp_reward = metrics.get('training/amp_reward', 0.0)
            print(f"  Final AMP reward: {amp_reward:.4f}")

        # Save checkpoint
        checkpoint_path = os.path.join(
            args.checkpoint_dir, f"final_iter_{num_iterations}.pkl"
        )
        try:
            import pickle

            checkpoint_data = {
                "params": params,
                "metrics": metrics,
                "config": ppo_kwargs,
            }

            if args.enable_amp and amp_disc is not None:
                checkpoint_data["amp_discriminator"] = amp_disc
                checkpoint_data["amp_weight"] = amp_weight

            with open(checkpoint_path, "wb") as f:
                pickle.dump(checkpoint_data, f)

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
