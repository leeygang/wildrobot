#!/usr/bin/env python3
"""Entry point for AMP+PPO training on WildRobot.

This script provides a command-line interface for training a walking policy
using PPO with Adversarial Motion Priors (AMP) for natural motion learning.

Usage:
    # Default training with AMP+PPO (recommended)
    python train.py

    # With custom configuration
    python train.py --num-envs 512 --iterations 5000

    # Quick smoke test
    python train.py --verify

    # Debug mode: Pure PPO without AMP (Brax trainer)
    python train.py --debug-brax-ppo

Architecture:
    This script supports two training modes:

    1. AMP+PPO (default): Uses custom training loop that integrates AMP
       discriminator for natural motion learning from reference data.

    2. Brax PPO (--debug-brax-ppo): Uses Brax's PPO trainer without AMP.
       For debugging and comparison only.
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
from playground_amp.configs.config import (
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

    # Training mode - JIT AMP+PPO is default
    parser.add_argument(
        "--debug-brax-ppo",
        action="store_true",
        help="Use Brax PPO trainer without AMP (for debugging only)",
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


def train_with_jit_loop(args, wandb_tracker: Optional[WandbTracker] = None):
    """Train using fully JIT-compiled AMP+PPO training loop.

    This is the fastest training mode - entire training loop runs on GPU.
    Expected 1000x+ speedup over Python-loop based training.

    Args:
        args: Parsed command-line arguments
        wandb_tracker: Optional W&B tracker for logging
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
    from playground_amp.training.trainer_jit import (
        AMPPPOConfigJit,
        IterationMetrics,
        train_amp_ppo_jit,
        TrainingState,
    )

    # Load training config
    config_path = Path(args.config) if args.config else DEFAULT_TRAINING_CONFIG_PATH
    training_cfg = load_training_config(config_path)
    training_config = training_cfg.raw_config
    trainer_cfg = training_config.get("trainer", {})
    networks_cfg = training_config.get("networks", {})
    amp_cfg = training_config.get("amp", {})

    # Create environment
    print("Creating WildRobotEnv...")
    env = create_env()
    print(
        f"✓ Environment created (obs_size={env.observation_size}, action_size={env.action_size})"
    )

    # Get checkpoint settings from config or CLI
    checkpoint_cfg = training_config.get("checkpoints", {})
    checkpoint_interval = args.checkpoint_interval or checkpoint_cfg.get(
        "interval", 100
    )
    keep_checkpoints = args.keep_checkpoints or checkpoint_cfg.get("keep_last_n", 5)

    # Create training job name (for checkpoint subdirectory)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_name = f"wildrobot_amp_{timestamp}"
    checkpoint_dir = os.path.join(args.checkpoint_dir, job_name)

    print(f"Training job: {job_name}")
    print(f"Checkpoints will be saved to: {checkpoint_dir}")

    # Create JIT config (no checkpoint fields - handled by callback)
    config = AMPPPOConfigJit(
        # Environment
        obs_dim=env.observation_size,
        action_dim=env.action_size,
        num_envs=args.num_envs,
        num_steps=trainer_cfg.get("rollout_steps", 20),
        # PPO hyperparameters
        learning_rate=args.lr,
        gamma=args.gamma,
        gae_lambda=trainer_cfg.get("gae_lambda", 0.95),
        clip_epsilon=args.clip_epsilon,
        value_loss_coef=trainer_cfg.get("value_loss_coef", 0.5),
        entropy_coef=args.entropy_coef,
        max_grad_norm=trainer_cfg.get("max_grad_norm", 0.5),
        # PPO training
        num_minibatches=trainer_cfg.get("num_minibatches", 8),
        update_epochs=trainer_cfg.get("epochs", 4),
        # Network architecture
        policy_hidden_dims=tuple(
            networks_cfg.get("policy_hidden_dims", [512, 256, 128])
        ),
        value_hidden_dims=tuple(networks_cfg.get("value_hidden_dims", [512, 256, 128])),
        # AMP configuration
        amp_reward_weight=args.amp_weight,
        disc_learning_rate=args.disc_lr,
        disc_updates_per_iter=amp_cfg.get("update_steps", 2),
        disc_batch_size=amp_cfg.get("batch_size", 2048),
        gradient_penalty_weight=amp_cfg.get("gradient_penalty_weight", 5.0),
        disc_hidden_dims=tuple(amp_cfg.get("discriminator_hidden", [1024, 512, 256])),
        disc_input_noise_std=amp_cfg.get("disc_input_noise_std", 0.0),
        # Training
        total_iterations=args.iterations,
        seed=args.seed,
        log_interval=args.log_interval,
    )

    # Load reference motion data
    print(f"Loading reference motion data from: {args.amp_data}")
    with open(args.amp_data, "rb") as f:
        ref_data = pickle.load(f)

    # Extract AMP features from reference data
    if isinstance(ref_data, np.ndarray) and ref_data.ndim == 2:
        # Already extracted features (2D array: num_samples x feature_dim)
        ref_features = ref_data
    elif isinstance(ref_data, dict):
        # If it's a dict with 'features' key
        if "features" in ref_data:
            ref_features = np.array(ref_data["features"])
        elif "observations" in ref_data:
            # Need to convert observations to AMP features
            from playground_amp.amp.amp_features import extract_amp_features, get_default_amp_config

            amp_config = get_default_amp_config()
            obs = np.array(ref_data["observations"])
            # Extract features from observations
            features_list = []
            for seq in obs:
                for i in range(len(seq)):
                    feat = extract_amp_features(seq[i], amp_config)
                    features_list.append(feat)
            ref_features = np.array(features_list)
        else:
            raise ValueError(f"Unknown reference data format: {ref_data.keys()}")
    elif isinstance(ref_data, list):
        # List of sequences - flatten and extract features
        from playground_amp.amp.amp_features import extract_amp_features, get_default_amp_config

        amp_config = get_default_amp_config()
        features_list = []
        for seq in ref_data:
            seq = np.array(seq)
            for i in range(len(seq)):
                feat = extract_amp_features(seq[i], amp_config)
                features_list.append(feat)
        ref_features = np.array(features_list)
    else:
        ref_features = np.array(ref_data)

    print(f"✓ Reference features: {ref_features.shape}")

    # Create vmapped environment functions
    def batched_step_fn(state, action):
        """Batched environment step."""
        return jax.vmap(lambda s, a: env.step(s, a))(state, action)

    def batched_reset_fn(rng):
        """Batched environment reset."""
        rngs = jax.random.split(rng, config.num_envs)
        return jax.vmap(env.reset)(rngs)

    print(f"✓ Environment functions created (vmapped for {config.num_envs} envs)")

    # Checkpoint management state (captured by callback closure)
    # Track global best (for best_checkpoint files)
    best_reward = {"value": float("-inf")}

    # Track best within current checkpoint window
    # We save the best performing checkpoint within each N-iteration window,
    # not just the checkpoint at iteration N
    window_best = {
        "reward": float("-inf"),
        "state": None,  # Will hold CPU copy of state (via jax.device_get)
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
                disc_loss=float(metrics.disc_loss),
                disc_accuracy=float(metrics.disc_accuracy),
                amp_reward_mean=float(metrics.amp_reward_mean),
                amp_reward_std=0.0,
                clip_fraction=0.0,
                approx_kl=0.0,
                env_steps_per_sec=steps_per_sec,
                # Task reward breakdown
                forward_velocity=float(metrics.forward_velocity),
                episode_length=float(metrics.episode_length),
            )
            # Add task reward breakdown to metrics
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
            # Copy state to CPU to free GPU memory (JAX arrays are immutable)
            window_best["state"] = jax.device_get(state)
            window_best["metrics"] = metrics
            window_best["iteration"] = iteration
            window_best["total_steps"] = int(state.total_steps)

        # At checkpoint boundary, save the best from this window
        if iteration % checkpoint_interval == 0:
            # Use the best state from this window (not current state)
            save_state = window_best["state"]
            save_metrics = window_best["metrics"]
            save_iteration = window_best["iteration"]
            save_reward = window_best["reward"]

            # Check if this is also the global best
            is_global_best = save_reward > best_reward["value"]
            if is_global_best:
                best_reward["value"] = save_reward

            # Save checkpoint using the best state from this window
            ckpt_path = save_checkpoint_from_cpu(
                state_cpu=save_state,
                config=config,
                iteration=save_iteration,
                total_steps=window_best["total_steps"],
                checkpoint_dir=checkpoint_dir,
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

            # Reset window tracking for next window
            window_best["reward"] = float("-inf")
            window_best["state"] = None
            window_best["metrics"] = None
            window_best["iteration"] = 0
            window_best["total_steps"] = 0

            # Clean up old checkpoints
            manage_checkpoints(checkpoint_dir, keep_checkpoints)

    # Train
    print("\n" + "=" * 60)
    print("Starting JIT-compiled AMP+PPO training...")
    print("=" * 60 + "\n")

    final_state = train_amp_ppo_jit(
        env_step_fn=batched_step_fn,
        env_reset_fn=batched_reset_fn,
        config=config,
        ref_motion_data=ref_features,
        callback=callback,
    )

    # Save checkpoint
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.checkpoint_dir, "final_jit_policy.pkl")

    checkpoint_data = {
        "policy_params": jax.device_get(final_state.policy_params),
        "value_params": jax.device_get(final_state.value_params),
        "disc_params": jax.device_get(final_state.disc_params),
        "feature_mean": jax.device_get(final_state.feature_mean),
        "feature_var": jax.device_get(final_state.feature_var),
    }

    with open(checkpoint_path, "wb") as f:
        pickle.dump(checkpoint_data, f)

    print(f"✓ Policy saved to: {checkpoint_path}")

    # Print best checkpoints summary
    print_top_checkpoints_summary(checkpoint_dir)

    # Generate and upload evaluation video after training
    video_cfg = training_config.get("video", {})
    video_enabled = video_cfg.get("enabled", True) and not args.verify

    if video_enabled:
        try:
            import jax

            print("\n" + "=" * 60)
            print("Generating evaluation video...")
            print("=" * 60)

            # Create a single-env reset and step function for video generation
            def single_reset_fn(rng):
                return env.reset(rng)

            def single_step_fn(state, action):
                return env.step(state, action)

            # Create inference function from trained policy
            from playground_amp.training.ppo_core import create_networks, sample_actions

            ppo_network = create_networks(
                obs_dim=config.obs_dim,
                action_dim=config.action_dim,
                policy_hidden_dims=config.policy_hidden_dims,
                value_hidden_dims=config.value_hidden_dims,
            )

            def policy_inference(obs, rng):
                """Get action from trained policy."""
                action, _, _ = sample_actions(
                    final_state.processor_params,
                    final_state.policy_params,
                    ppo_network,
                    obs,
                    rng,
                )
                return action

            jit_inference_fn = jax.jit(policy_inference)

            # Get video config settings
            num_videos = video_cfg.get("num_videos", 1)
            episode_length = video_cfg.get("episode_length", 500)
            render_every = video_cfg.get("render_every", 2)
            video_width = video_cfg.get("width", 640)
            video_height = video_cfg.get("height", 480)
            video_fps = video_cfg.get("fps", 25)
            upload_to_wandb = video_cfg.get("upload_to_wandb", True) and (
                wandb_tracker is not None
            )
            output_subdir = video_cfg.get("output_subdir", "videos")

            # Generate video
            rng_key = jax.random.PRNGKey(config.seed + 1000)
            videos = generate_eval_video(
                env=env,
                inference_fn=jit_inference_fn,
                rng_key=rng_key,
                episode_length=episode_length,
                num_rollouts=num_videos,
                render_every=render_every,
                height=video_height,
                width=video_width,
            )

            # Save and upload to W&B
            if videos:
                video_dir = os.path.join(args.checkpoint_dir, output_subdir)
                os.makedirs(video_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                for i, video in enumerate(videos):
                    video_path = os.path.join(
                        video_dir, f"eval_rollout_{timestamp}_{i}.mp4"
                    )
                    wandb_key = (
                        f"eval/rollout_video_{i}" if i > 0 else "eval/final_rollout"
                    )

                    save_and_upload_video(
                        video=video,
                        filepath=video_path,
                        fps=float(video_fps),
                        upload_to_wandb=upload_to_wandb,
                        wandb_key=wandb_key,
                    )
            else:
                print("⚠️ No video frames captured (rendering may not be available)")

        except Exception as e:
            print(f"⚠️ Video generation failed: {e}")
            print("  (This is non-critical - training completed successfully)")

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

    trainer_cfg = training_config.get("trainer", {})
    amp_cfg = training_config.get("amp", {})

    # Merge config with CLI overrides (CLI takes precedence if provided)
    if args.iterations is None:
        args.iterations = trainer_cfg.get("iterations", 3000)
    if args.num_envs is None:
        args.num_envs = trainer_cfg.get("num_envs", 512)
    if args.seed is None:
        args.seed = trainer_cfg.get("seed", 42)
    if args.lr is None:
        args.lr = trainer_cfg.get("lr", 3e-4)
    if args.gamma is None:
        args.gamma = trainer_cfg.get("gamma", 0.99)
    if args.clip_epsilon is None:
        args.clip_epsilon = trainer_cfg.get("clip_epsilon", 0.2)
    if args.entropy_coef is None:
        args.entropy_coef = trainer_cfg.get("entropy_coef", 0.01)
    if args.amp_weight is None:
        args.amp_weight = amp_cfg.get("weight", 1.0)
    if args.disc_lr is None:
        args.disc_lr = amp_cfg.get("disc_lr", 1e-4)
    if args.log_interval is None:
        args.log_interval = trainer_cfg.get("log_interval", 10)
    if args.checkpoint_dir is None:
        args.checkpoint_dir = trainer_cfg.get(
            "checkpoint_dir", "playground_amp/checkpoints"
        )
    if args.amp_data is None:
        args.amp_data = amp_cfg.get("dataset_path")

    # Quick verify mode (override everything)
    if args.verify:
        quick_cfg = training_config.get("quick_verify", {})
        quick_trainer = quick_cfg.get("trainer", {})
        quick_amp = quick_cfg.get("amp", {})
        args.iterations = quick_trainer.get("iterations", 10)
        args.num_envs = quick_trainer.get("num_envs", 4)
        # Override disc_batch_size for quick verify (stored as temp attribute)
        args._quick_verify_disc_batch_size = quick_amp.get("batch_size", 32)
        args.log_interval = 1
        print("=" * 60)
        print("VERIFICATION MODE")
        print("Running quick smoke test")
        print("=" * 60)

    # Determine training mode
    use_amp = not args.debug_brax_ppo and not args.no_amp

    print(f"\n{'=' * 60}")
    print("WildRobot Training")
    print(f"{'=' * 60}")
    print(f"  Version: {training_cfg.version} ({training_cfg.version_name})")
    print(f"  PID: {os.getpid()}  (kill -9 {os.getpid()} to terminate)")
    if use_amp:
        mode_str = "AMP+PPO (JIT-compiled - FAST)"
    else:
        mode_str = "Brax PPO (debug)"
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
            "mode": "amp+ppo" if use_amp else "brax-ppo",
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
        if use_amp:
            train_with_jit_loop(args, wandb_tracker=wandb_tracker)
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
    finally:
        # Finish W&B tracking
        if wandb_tracker is not None:
            wandb_tracker.finish()

    return 0


if __name__ == "__main__":
    sys.exit(main())
