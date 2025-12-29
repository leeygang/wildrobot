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
DEFAULT_TRAINING_CONFIG_PATH = Path(__file__).parent / "configs" / "ppo_walking.yaml"
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
    save_window_best_checkpoint,
)

# Import W&B tracker
from playground_amp.training.experiment_tracking import (
    create_training_metrics,
    create_training_metrics_from_iteration,
    define_wandb_topline_metrics,
    generate_eval_video,
    generate_job_name,
    save_and_upload_video,
    WandbTracker,
)


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
    checkpoint_dir: Optional[str] = None,
    amp_data_path: Optional[str] = None,
    resume_checkpoint_path: Optional[str] = None,
    config_name: Optional[str] = None,
):
    """Unified training using training_loop.py for both PPO-only and AMP+PPO.

    This is the single training path for all modes:
    - Stage 1: PPO-only (training_cfg.amp.enabled=False) - robot-native walking
    - Stage 3: AMP+PPO (training_cfg.amp.enabled=True) - natural motion learning

    Args:
        training_cfg: Training configuration (single source of truth)
        wandb_tracker: Optional W&B tracker for logging
        checkpoint_dir: Directory for saving checkpoints (overrides config if provided)
        amp_data_path: Path to AMP reference data (overrides config if provided)
        resume_checkpoint_path: Path to checkpoint to resume training from
        config_name: Name of the config file (without extension), e.g., "ppo_walking"
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
    env = WildRobotEnv(config=training_cfg)
    print(
        f"✓ Environment created (obs_size={env.observation_size}, action_size={env.action_size})"
    )

    # Determine checkpoint directory
    final_checkpoint_dir = checkpoint_dir or training_cfg.checkpoints.dir

    # Create training job name (for checkpoint subdirectory)
    # Format: {config_name}_v{version}_{timestamp}-{wandb_run_id}
    # Example: ppo_walking_v01005_20251228_205534-uf665cr6
    mode_suffix = "amp" if training_cfg.amp.enabled else "ppo"
    job_name = generate_job_name(
        version=training_cfg.version,
        config_name=config_name,
        wandb_tracker=wandb_tracker,
        mode_suffix=mode_suffix,
    )
    job_checkpoint_dir = os.path.join(final_checkpoint_dir, job_name)

    print(f"Training job: {job_name}")
    print(f"Checkpoints will be saved to: {job_checkpoint_dir}")

    # Load reference motion data only if AMP enabled
    ref_features = None
    if training_cfg.amp.enabled:
        ref_path = amp_data_path or training_cfg.amp.dataset_path
        if ref_path is None:
            raise ValueError(
                "AMP mode requires reference motion data. "
                "Set amp.dataset_path in config or pass --amp-data."
            )
        print(f"Loading reference motion data from: {ref_path}")
        from playground_amp.amp.ref_features import load_reference_features

        ref_features = load_reference_features(ref_path)
        print(f"✓ Reference features: {ref_features.shape}")

    print(
        f"✓ Environment functions created (vmapped for {training_cfg.ppo.num_envs} envs)"
    )

    # Create vmapped environment functions
    def batched_step_fn(state, action):
        """Batched environment step."""
        return jax.vmap(lambda s, a: env.step(s, a))(state, action)

    def batched_reset_fn(rng):
        """Batched environment reset."""
        rngs = jax.random.split(rng, training_cfg.ppo.num_envs)
        return jax.vmap(env.reset)(rngs)

    # Get checkpoint settings from config
    checkpoint_interval = training_cfg.checkpoints.interval
    keep_checkpoints = training_cfg.checkpoints.keep_last_n

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
            wandb_metrics = create_training_metrics_from_iteration(
                iteration=iteration,
                metrics=metrics,
                steps_per_sec=steps_per_sec,
                use_amp=training_cfg.amp.enabled,
            )
            wandb_metrics["debug/task_reward_per_step"] = float(
                metrics.task_reward_mean
            )
            wandb_metrics["debug/robot_height"] = float(metrics.env_metrics["height"])

            # v0.10.4: Log reward breakdown for diagnostics
            # This is critical for tuning reward weights
            reward_terms = [
                "reward/forward", "reward/lateral", "reward/healthy",
                "reward/orientation", "reward/angvel", "reward/standing",
                "reward/torque", "reward/saturation",
                "reward/action_rate", "reward/joint_vel",
                "reward/slip", "reward/clearance", "reward/gait_periodicity",
                "reward/hip_swing", "reward/knee_swing", "reward/flight_phase",
            ]
            for term in reward_terms:
                if term in metrics.env_metrics:
                    wandb_metrics[term] = float(metrics.env_metrics[term])

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
            save_window_best_checkpoint(
                window_best=window_best,
                best_reward=best_reward,
                config=training_cfg,
                checkpoint_dir=job_checkpoint_dir,
                keep_last_n=keep_checkpoints,
            )

    # Load checkpoint for resuming if provided
    resume_checkpoint = None
    if resume_checkpoint_path is not None:
        print(f"\nLoading checkpoint for resume: {resume_checkpoint_path}")
        resume_checkpoint = load_checkpoint(resume_checkpoint_path)

    # Train using unified trainer
    mode_str = "AMP+PPO" if training_cfg.amp.enabled else "PPO-only (Stage 1)"
    print("\n" + "=" * 60)
    print(f"Starting unified training ({mode_str})...")
    print("=" * 60 + "\n")

    final_state = train(
        env_step_fn=batched_step_fn,
        env_reset_fn=batched_reset_fn,
        config=training_cfg,
        ref_motion_data=ref_features,  # None for PPO-only
        callback=callback,
        resume_checkpoint=resume_checkpoint,
    )

    # Save any remaining best checkpoint from the final window
    # This fixes the bug where if training ends at iteration 740 with checkpoint_interval=100,
    # the best checkpoint between 700-740 would not be saved
    if checkpoint_interval > 0:
        save_window_best_checkpoint(
            window_best=window_best,
            best_reward=best_reward,
            config=training_cfg,
            checkpoint_dir=job_checkpoint_dir,
            keep_last_n=keep_checkpoints,
        )

    # Print best checkpoints summary
    print_top_checkpoints_summary(job_checkpoint_dir)

    return final_state


def override_config_with_cli(training_cfg: "TrainingConfig", args: argparse.Namespace) -> None:
    """Apply CLI argument overrides to training config.

    CLI arguments have highest priority and override both config file values
    and quick_verify settings. Only non-None arguments are applied.

    Args:
        training_cfg: Training configuration to modify in-place
        args: Parsed command-line arguments

    Raises:
        ValueError: If AMP configuration is invalid (enabled but weight <= 0)
    """
    # PPO parameters
    if args.iterations is not None:
        training_cfg.ppo.iterations = args.iterations
    if args.num_envs is not None:
        training_cfg.ppo.num_envs = args.num_envs
    if args.seed is not None:
        training_cfg.seed = args.seed
    if args.lr is not None:
        training_cfg.ppo.learning_rate = args.lr
    if args.gamma is not None:
        training_cfg.ppo.gamma = args.gamma
    if args.clip_epsilon is not None:
        training_cfg.ppo.clip_epsilon = args.clip_epsilon
    if args.entropy_coef is not None:
        training_cfg.ppo.entropy_coef = args.entropy_coef
    if args.log_interval is not None:
        training_cfg.ppo.log_interval = args.log_interval

    # AMP parameters
    if args.disc_lr is not None:
        training_cfg.amp.discriminator.learning_rate = args.disc_lr
    if args.amp_data is not None:
        training_cfg.amp.dataset_path = args.amp_data

    # Checkpoint parameters
    if args.checkpoint_dir is not None:
        training_cfg.checkpoints.dir = args.checkpoint_dir

    # Validate and set AMP enabled flag
    # --no-amp flag always disables AMP
    # Otherwise, validate that enabled configs have weight > 0
    if args.no_amp:
        training_cfg.amp.enabled = False
    elif training_cfg.amp.enabled and training_cfg.amp.weight <= 0:
        raise ValueError(
            f"Invalid AMP configuration: amp.enabled=True but amp.weight={training_cfg.amp.weight}. "
            "Either set amp.weight > 0 or disable AMP with --no-amp or amp.enabled=False in config."
        )


def main():
    """Main entry point."""
    args = parse_args()

    # Load config file
    config_path = Path(args.config) if args.config else DEFAULT_TRAINING_CONFIG_PATH
    print(f"Loading config from: {config_path}")
    training_cfg = load_training_config(config_path)

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

    # Quick verify mode - apply overrides from config's quick_verify section
    # Priority: CLI > Quick_Verify > Config
    # Apply quick_verify first, then CLI overrides will take precedence
    quick_verify_section = training_cfg.raw_config.get("quick_verify", {})
    quick_verify_enabled = args.verify or quick_verify_section.get("enabled", False)

    if quick_verify_enabled:
        training_cfg.apply_overrides(quick_verify_section)
        print("=" * 60)
        print("VERIFICATION MODE")
        print("Running quick smoke test")
        print("=" * 60)

    # Apply CLI overrides (highest priority - overrides both config and quick_verify)
    override_config_with_cli(training_cfg, args)

    # Freeze config after all overrides are applied
    training_cfg.freeze()

    print(f"\n{'=' * 60}")
    print("WildRobot Training")
    print(f"{'=' * 60}")
    print(f"  Version: {training_cfg.version} ({training_cfg.version_name})")
    print(f"  PID: {os.getpid()}  (kill -9 {os.getpid()} to terminate)")
    if training_cfg.amp.enabled:
        mode_str = "AMP+PPO (JIT-compiled - FAST)"
    else:
        mode_str = "PPO-only (Stage 1: Robot-Native Walking)"
    print(f"  Mode: {mode_str}")
    print(f"  Config: {config_path}")
    print(f"  Iterations: {training_cfg.ppo.iterations}")
    print(f"  Environments: {training_cfg.ppo.num_envs}")
    print(f"  Learning rate: {training_cfg.ppo.learning_rate}")
    print(f"  Seed: {training_cfg.seed}")
    if training_cfg.amp.enabled:
        print(f"  AMP weight: {training_cfg.amp.weight}")
        print(f"  AMP data: {training_cfg.amp.dataset_path}")
    print(f"  Checkpoint dir: {training_cfg.checkpoints.dir}")
    print(f"{'=' * 60}\n")

    start_time = time.time()

    # Initialize W&B tracker
    wandb_cfg = training_cfg.wandb
    wandb_tracker = None

    if wandb_cfg.enabled and not args.verify:
        wandb_tracker = WandbTracker.from_config(
            training_cfg=training_cfg,
            wandb_cfg=wandb_cfg,
        )

        # Define topline metrics with exit criteria targets (Section 3.2.4)
        define_wandb_topline_metrics()

    try:
        # Extract config name from config path (e.g., "ppo_walking" from "ppo_walking.yaml")
        config_name = config_path.stem if config_path else None

        # Unified training for both PPO-only and AMP modes
        start_training(
            training_cfg=training_cfg,
            wandb_tracker=wandb_tracker,
            checkpoint_dir=args.checkpoint_dir,
            amp_data_path=args.amp_data,
            resume_checkpoint_path=args.resume,
            config_name=config_name,
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
