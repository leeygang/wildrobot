#!/usr/bin/env python3
"""Entry point for WildRobot training.

This script provides a command-line interface for training walking policies
using PPO with optional Adversarial Motion Priors (AMP) for natural motion learning.

Training Modes:
    1. PPO-only (Stage 1): Learn robot-native walking without reference motion
       python train.py --config configs/ppo_walking.yaml --no-amp

    2. AMP+PPO (default): Use human motion priors for natural-looking gait
       python train.py --config configs/ppo_amass_training.yaml

Usage Examples:
    # Stage 1: PPO-only training (robot-native walking)
    python train.py --config playground_amp/configs/ppo_walking.yaml --no-amp

    # Stage 1: Quick smoke test
    python train.py --config playground_amp/configs/ppo_walking.yaml --no-amp --verify

    # AMP+PPO training with human motion priors
    python train.py --config playground_amp/configs/ppo_amass_training.yaml

    # With custom parameters
    python train.py --num-envs 512 --iterations 5000

Architecture:
    This script supports two training modes:

    1. PPO-only (--no-amp): Stage 1 training for discovering robot's natural gait.
       Uses Brax's PPO trainer with mujoco_playground wrapper. No reference motion.

    2. AMP+PPO (default): Uses custom JIT-compiled training loop that integrates
       AMP discriminator for natural motion learning from reference data.

See also:
    - playground_amp/docs/learn_first_plan.md: Roadmap for the "Learn First, Retarget Later" approach
    - playground_amp/configs/ppo_walking.yaml: Stage 1 config
    - playground_amp/configs/ppo_amass_training.yaml: AMP+PPO config
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

from ml_collections import config_dict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Default config paths
DEFAULT_TRAINING_CONFIG_PATH = (
    Path(__file__).parent / "configs" / "ppo_amass_training.yaml"
)
DEFAULT_ROBOT_CONFIG_PATH = (
    Path(__file__).parent.parent / "assets" / "robot_config.yaml"
)

# Import config loaders from configs module
from playground_amp.configs.training_config import (
    load_robot_config,
    load_training_config,
    RobotConfig,
    TrainingConfig,
    WandbConfig,
)

# Import checkpoint management
from playground_amp.training.checkpoint import (
    get_top_checkpoints_by_reward,
    load_checkpoint,
    manage_checkpoints,
    print_top_checkpoints_summary,
    save_checkpoint,
    save_checkpoint_from_cpu,
)

# Import W&B tracker
from playground_amp.training.experiment_tracking import (
    create_training_metrics,
    define_wandb_topline_metrics,
    generate_eval_video,
    save_and_upload_video,
    WandbTracker,
)


def create_env_config(training_config: dict) -> config_dict.ConfigDict:
    """Create environment config from training config.

    Args:
        training_config: Full training configuration dict.

    Returns:
        ConfigDict for WildRobotEnv.
    """
    env_cfg = training_config.get("env", {})

    config = config_dict.ConfigDict()

    # Model path (required)
    if "model_path" not in env_cfg:
        raise ValueError("model_path is required in env config")
    config.model_path = env_cfg["model_path"]

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
    from playground_amp.envs.wildrobot_env import WildRobotEnv

    # Use default path if not provided
    if config_path is None:
        config_path = DEFAULT_TRAINING_CONFIG_PATH

    training_cfg = load_training_config(config_path)
    env_config = create_env_config(training_cfg.raw_config)
    env = WildRobotEnv(config=env_config)
    return env


def parse_args():
    """Parse command-line arguments.

    CLI arguments override config file values when explicitly provided.
    """
    parser = argparse.ArgumentParser(
        description="Train WildRobot with PPO+AMP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config file (loaded first, then CLI overrides)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to training config YAML (default: configs/wildrobot_phase3_training.yaml)",
    )

    # Training configuration (CLI overrides config file)
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Number of training iterations (default from config)",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=None,
        help="Number of parallel environments (default from config)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default from config)",
    )

    # PPO hyperparameters
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (default from config)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="Discount factor (default from config)",
    )
    parser.add_argument(
        "--clip-epsilon",
        type=float,
        default=None,
        help="PPO clipping parameter (default from config)",
    )
    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=None,
        help="Entropy bonus coefficient (default from config)",
    )

    # AMP configuration (only for custom loop)
    parser.add_argument(
        "--amp-weight",
        type=float,
        default=None,
        help="Weight for AMP reward (default from config)",
    )
    parser.add_argument(
        "--amp-data",
        type=str,
        default=None,
        help="Path to AMP reference motion dataset (.pkl file)",
    )
    parser.add_argument(
        "--disc-lr",
        type=float,
        default=None,
        help="Discriminator learning rate (default from config)",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable AMP (pure PPO training)",
    )

    # Logging and checkpoints
    parser.add_argument(
        "--log-interval",
        type=int,
        default=None,
        help="Log metrics every N iterations (default from config)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory for saving checkpoints (default from config)",
    )

    # Quick test mode
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run quick verification (10 iterations, 4 envs)",
    )

    # Resume training from checkpoint
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from (e.g., checkpoints/best_checkpoint.pkl)",
    )

    # Checkpoint settings
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=None,
        help="Save checkpoint every N iterations (default from config)",
    )
    parser.add_argument(
        "--keep-checkpoints",
        type=int,
        default=None,
        help="Number of recent checkpoints to keep (default from config)",
    )

    return parser.parse_args()


def start_training(
    training_cfg: "TrainingConfig",
    wandb_tracker: Optional[WandbTracker] = None,
    use_amp: bool = True,
    checkpoint_dir: Optional[str] = None,
    amp_data_path: Optional[str] = None,
):
    """Unified training using training_loop.py for both PPO-only and AMP+PPO.

    This is the single training path for all modes:
    - Stage 1: PPO-only (use_amp=False) - robot-native walking
    - Stage 3: AMP+PPO (use_amp=True) - natural motion learning

    Args:
        training_cfg: Training configuration (single source of truth)
        wandb_tracker: Optional W&B tracker for logging
        use_amp: Whether to use AMP discriminator
        checkpoint_dir: Directory for saving checkpoints (overrides config if provided)
        amp_data_path: Path to AMP reference data (overrides config if provided)
    """
    import pickle

    import jax
    import jax.numpy as jnp
    import numpy as np

    # Check JAX backend
    print(f"\n{'=' * 60}")
    print("JAX Configuration")
    print(f"{'=' * 60}")
    print(f"  Backend: {jax.default_backend()}")
    print(f"  Devices: {jax.devices()}")
    if jax.default_backend() != "gpu":
        print("  ⚠️  WARNING: JAX is NOT using GPU!")
        print("  Install JAX with CUDA: pip install jax[cuda12]")
    else:
        print("  ✓ GPU detected")
    print(f"{'=' * 60}\n")

    from playground_amp.envs.wildrobot_env import WildRobotEnv
    from playground_amp.training.training_loop import (
        IterationMetrics,
        train,
        TrainingState,
    )

    # Create environment config and environment
    print("Creating WildRobotEnv...")
    env_config = create_env_config(training_cfg.raw_config)
    env = WildRobotEnv(config=env_config)
    print(
        f"✓ Environment created (obs_size={env.observation_size}, action_size={env.action_size})"
    )

    # Determine checkpoint directory
    final_checkpoint_dir = checkpoint_dir or training_cfg.checkpoint_dir

    # Create training job name (for checkpoint subdirectory)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_suffix = "amp" if use_amp else "ppo"
    job_name = f"wildrobot_{mode_suffix}_{timestamp}"
    job_checkpoint_dir = os.path.join(final_checkpoint_dir, job_name)

    print(f"Training job: {job_name}")
    print(f"Checkpoints will be saved to: {job_checkpoint_dir}")

    # Load reference motion data only if AMP enabled
    ref_features = None
    if use_amp:
        ref_path = amp_data_path or training_cfg.ref_motion_path
        if ref_path is None:
            raise ValueError(
                "AMP mode requires reference motion data. "
                "Set amp.dataset_path in config or pass --amp-data."
            )
        print(f"Loading reference motion data from: {ref_path}")
        from playground_amp.amp.ref_features import load_reference_features

        ref_features = load_reference_features(ref_path)
        print(f"✓ Reference features: {ref_features.shape}")

    # Create runtime config using TrainingConfig.to_runtime_config()
    config = training_cfg.to_runtime_config()

    print(
        f"✓ Environment functions created (vmapped for {training_cfg.training.num_envs} envs)"
    )

    # Create vmapped environment functions
    def batched_step_fn(state, action):
        """Batched environment step."""
        return jax.vmap(lambda s, a: env.step(s, a))(state, action)

    def batched_reset_fn(rng):
        """Batched environment reset."""
        rngs = jax.random.split(rng, config.training.num_envs)
        return jax.vmap(env.reset)(rngs)

    # Get checkpoint settings from config
    checkpoint_cfg = training_cfg.raw_config.get("checkpoints", {})
    checkpoint_interval = checkpoint_cfg.get("interval", 100)
    keep_checkpoints = checkpoint_cfg.get("keep_last_n", 5)

    # Checkpoint management state (captured by callback closure)
    best_reward = {"value": float("-inf")}
    window_best = {
        "reward": float("-inf"),
        "state": None,
        "metrics": None,
        "iteration": 0,
        "total_steps": 0,
    }

    # Training callback for W&B logging and checkpoint management
    def callback(
        iteration: int,
        state: TrainingState,
        metrics: IterationMetrics,
        steps_per_sec: float,
    ):
        # W&B logging
        if wandb_tracker is not None:
            wandb_metrics = create_training_metrics(
                iteration=iteration,
                episode_reward=float(metrics.episode_reward),
                ppo_loss=float(metrics.total_loss),
                policy_loss=float(metrics.policy_loss),
                value_loss=float(metrics.value_loss),
                entropy_loss=float(metrics.entropy_loss),
                disc_loss=float(metrics.disc_loss) if use_amp else 0.0,
                disc_accuracy=float(metrics.disc_accuracy) if use_amp else 0.5,
                amp_reward_mean=float(metrics.amp_reward_mean) if use_amp else 0.0,
                amp_reward_std=0.0,
                clip_fraction=0.0,
                approx_kl=0.0,
                env_steps_per_sec=steps_per_sec,
                forward_velocity=float(metrics.forward_velocity),
                episode_length=float(metrics.episode_length),
            )
            wandb_metrics["debug/task_reward_per_step"] = float(
                metrics.task_reward_mean
            )
            wandb_metrics["debug/robot_height"] = float(metrics.robot_height)
            wandb_tracker.log(wandb_metrics, step=int(state.total_steps))

        # Skip checkpoint tracking if disabled
        if checkpoint_interval <= 0:
            return

        current_reward = float(metrics.episode_reward)

        # Track best within current checkpoint window
        if current_reward > window_best["reward"]:
            window_best["reward"] = current_reward
            window_best["state"] = jax.device_get(state)
            window_best["metrics"] = metrics
            window_best["iteration"] = iteration
            window_best["total_steps"] = int(state.total_steps)

        # At checkpoint boundary, save the best from this window
        if iteration % checkpoint_interval == 0:
            save_state = window_best["state"]
            save_metrics = window_best["metrics"]
            save_iteration = window_best["iteration"]
            save_reward = window_best["reward"]

            is_global_best = save_reward > best_reward["value"]
            if is_global_best:
                best_reward["value"] = save_reward

            ckpt_path = save_checkpoint_from_cpu(
                state_cpu=save_state,
                config=config,
                iteration=save_iteration,
                total_steps=window_best["total_steps"],
                checkpoint_dir=job_checkpoint_dir,
                is_best=is_global_best,
                metrics=save_metrics,
            )

            if is_global_best:
                print(
                    f"  💾 Checkpoint saved (best from window, iter {save_iteration}) "
                    f"- NEW GLOBAL BEST: reward={save_reward:.2f}"
                )
            else:
                print(
                    f"  💾 Checkpoint saved (best from window, iter {save_iteration}, "
                    f"reward={save_reward:.2f})"
                )

            # Reset window tracking
            window_best["reward"] = float("-inf")
            window_best["state"] = None
            window_best["metrics"] = None
            window_best["iteration"] = 0
            window_best["total_steps"] = 0

            manage_checkpoints(job_checkpoint_dir, keep_checkpoints)

    # Train using unified trainer
    mode_str = "AMP+PPO" if use_amp else "PPO-only (Stage 1)"
    print("\n" + "=" * 60)
    print(f"Starting unified training ({mode_str})...")
    print("=" * 60 + "\n")

    final_state = train(
        env_step_fn=batched_step_fn,
        env_reset_fn=batched_reset_fn,
        config=config,
        ref_motion_data=ref_features,  # None for PPO-only
        callback=callback,
    )

    # Save final checkpoint
    os.makedirs(final_checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(
        final_checkpoint_dir, f"final_{mode_suffix}_policy.pkl"
    )

    checkpoint_data = {
        "policy_params": jax.device_get(final_state.policy_params),
        "value_params": jax.device_get(final_state.value_params),
        "disc_params": jax.device_get(final_state.disc_params) if use_amp else None,
        "feature_mean": jax.device_get(final_state.feature_mean) if use_amp else None,
        "feature_var": jax.device_get(final_state.feature_var) if use_amp else None,
        "mode": mode_suffix,
    }

    with open(checkpoint_path, "wb") as f:
        pickle.dump(checkpoint_data, f)

    print(f"✓ Policy saved to: {checkpoint_path}")

    # Print best checkpoints summary
    print_top_checkpoints_summary(job_checkpoint_dir)

    return final_state


def main():
    """Main entry point."""
    args = parse_args()

    # Load config file
    config_path = Path(args.config) if args.config else DEFAULT_TRAINING_CONFIG_PATH
    print(f"Loading config from: {config_path}")
    training_cfg = load_training_config(config_path)
    # Get raw dict for backward compatibility with existing code
    training_config = training_cfg.raw_config

    # Load robot config (generated by assets/post_process.py)
    # This must be loaded before any module that uses get_robot_config()
    if DEFAULT_ROBOT_CONFIG_PATH.exists():
        robot_cfg = load_robot_config(DEFAULT_ROBOT_CONFIG_PATH)
        print(
            f"Loaded robot config: {robot_cfg.robot_name} (action_dim={robot_cfg.action_dim})"
        )
    else:
        print(f"Warning: Robot config not found at {DEFAULT_ROBOT_CONFIG_PATH}")
        print("Run 'cd assets && python post_process.py' to generate it.")

    # Support both 'ppo:' (unified format) config
    ppo_cfg = training_config.get("ppo", {})
    amp_cfg = training_config.get("amp", {})

    # Merge config with CLI overrides (CLI takes precedence if provided)
    if args.iterations is None:
        args.iterations = ppo_cfg.get("iterations", 3000)
    if args.num_envs is None:
        args.num_envs = ppo_cfg.get("num_envs", 512)
    if args.seed is None:
        args.seed = ppo_cfg.get("seed", 42)
    if args.lr is None:
        args.lr = ppo_cfg.get("learning_rate", 3e-4)
    if args.gamma is None:
        args.gamma = ppo_cfg.get("gamma", 0.99)
    if args.clip_epsilon is None:
        args.clip_epsilon = ppo_cfg.get("clip_epsilon", 0.2)
    if args.entropy_coef is None:
        args.entropy_coef = ppo_cfg.get("entropy_coef", 0.01)
    if args.amp_weight is None:
        args.amp_weight = amp_cfg.get("weight", 1.0)
    if args.disc_lr is None:
        args.disc_lr = amp_cfg.get("disc_lr", 1e-4)
    if args.log_interval is None:
        args.log_interval = ppo_cfg.get("log_interval", 10)
    if args.checkpoint_dir is None:
        # Support 'checkpoints:' section for checkpoint directory
        checkpoints_cfg = training_config.get("checkpoints", {})
        args.checkpoint_dir = checkpoints_cfg.get("dir", "playground_amp/checkpoints")
    if args.amp_data is None:
        args.amp_data = amp_cfg.get("dataset_path")

    # Quick verify mode (override everything)
    if args.verify:
        quick_cfg = training_config.get("quick_verify", {})
        quick_ppo = quick_cfg.get("ppo", {})
        quick_amp = quick_cfg.get("amp", {})
        args.iterations = quick_ppo.get("iterations", 10)
        args.num_envs = quick_ppo.get("num_envs", 4)
        # Override disc_batch_size for quick verify (stored as temp attribute)
        args._quick_verify_disc_batch_size = quick_amp.get("batch_size", 32)
        args.log_interval = 1
        print("=" * 60)
        print("VERIFICATION MODE")
        print("Running quick smoke test")
        print("=" * 60)

    # Determine training mode: AMP+PPO (default) or PPO-only (--no-amp)
    use_amp = not args.no_amp

    print(f"\n{'=' * 60}")
    print("WildRobot Training")
    print(f"{'=' * 60}")
    print(f"  Version: {training_cfg.version} ({training_cfg.version_name})")
    print(f"  PID: {os.getpid()}  (kill -9 {os.getpid()} to terminate)")
    if use_amp:
        mode_str = "AMP+PPO (JIT-compiled - FAST)"
    else:
        mode_str = "PPO-only (Stage 1: Robot-Native Walking)"
    print(f"  Mode: {mode_str}")
    print(f"  Config: {config_path}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Environments: {args.num_envs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Seed: {args.seed}")
    if use_amp:
        print(f"  AMP weight: {args.amp_weight}")
        print(f"  AMP data: {args.amp_data}")
    print(f"  Checkpoint dir: {args.checkpoint_dir}")
    print(f"{'=' * 60}\n")

    start_time = time.time()

    # Initialize W&B tracker
    wandb_cfg = training_cfg.wandb
    wandb_tracker = None

    if wandb_cfg.enabled and not args.verify:
        # Create config dict for W&B
        wandb_config = {
            # Version tracking
            "version": training_cfg.version,
            "version_name": training_cfg.version_name,
            # Training params
            "iterations": args.iterations,
            "num_envs": args.num_envs,
            "learning_rate": args.lr,
            "gamma": args.gamma,
            "clip_epsilon": args.clip_epsilon,
            "entropy_coef": args.entropy_coef,
            "amp_weight": args.amp_weight if use_amp else 0.0,
            "disc_lr": args.disc_lr if use_amp else 0.0,
            "seed": args.seed,
            "mode": "amp+ppo" if use_amp else "ppo-only",
        }

        wandb_tracker = WandbTracker(
            project=wandb_cfg.project,
            entity=wandb_cfg.entity,
            name=wandb_cfg.name,
            config=wandb_config,
            tags=wandb_cfg.tags,
            mode=wandb_cfg.mode,
            enabled=wandb_cfg.enabled,
            log_dir=wandb_cfg.log_dir,
        )

        # Define topline metrics with exit criteria targets (Section 3.2.4)
        define_wandb_topline_metrics()

    try:
        # Apply CLI overrides to training_cfg (config is single source of truth)
        # Uses composed config structure: training_cfg.ppo.*, training_cfg.amp.*, training_cfg.training.*
        if args.iterations is not None:
            training_cfg.training.total_iterations = args.iterations
        if args.num_envs is not None:
            training_cfg.training.num_envs = args.num_envs
        if args.seed is not None:
            training_cfg.training.seed = args.seed
        if args.lr is not None:
            training_cfg.ppo.learning_rate = args.lr
        if args.gamma is not None:
            training_cfg.ppo.gamma = args.gamma
        if args.clip_epsilon is not None:
            training_cfg.ppo.clip_epsilon = args.clip_epsilon
        if args.entropy_coef is not None:
            training_cfg.ppo.entropy_coef = args.entropy_coef
        if args.log_interval is not None:
            training_cfg.training.log_interval = args.log_interval
        if args.disc_lr is not None:
            training_cfg.amp.disc_learning_rate = args.disc_lr

        # Set AMP weight based on mode
        if use_amp:
            if args.amp_weight is not None:
                training_cfg.amp.reward_weight = args.amp_weight
        else:
            training_cfg.amp.reward_weight = 0.0  # PPO-only mode

        # Unified training for both PPO-only and AMP modes
        start_training(
            training_cfg=training_cfg,
            wandb_tracker=wandb_tracker,
            use_amp=use_amp,
            checkpoint_dir=args.checkpoint_dir,
            amp_data_path=args.amp_data,
        )

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
    finally:
        # Finish W&B tracking
        if wandb_tracker is not None:
            wandb_tracker.finish()

    return 0


if __name__ == "__main__":
    sys.exit(main())
