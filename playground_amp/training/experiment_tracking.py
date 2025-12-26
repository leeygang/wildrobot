"""W&B Experiment Tracking Module for WildRobot Training.

This module provides logging to Weights & Biases for experiment tracking.

Topline Metrics (Section 3.2.4 Exit Criteria):
- Forward Velocity: 0.3-0.8 m/s (target)
- Episode Reward: >350 (target)
- AMP Reward: >0.7 (target for natural gait)
- Success Rate: >85% (target)
- Average Torque: <2.8Nm (target for energy efficiency)

Usage:
    from playground_amp.training.experiment_tracking import WandbTracker

    tracker = WandbTracker(
        project="wildrobot-locomotion",
        name="ppo-amp-v1",
        config={"lr": 3e-4, "num_envs": 4096},
    )

    for iteration in range(num_iterations):
        metrics = train_step(...)
        tracker.log(metrics, step=iteration)

    tracker.finish()
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


# =============================================================================
# Exit Criteria Targets (Section 3.2.4)
# =============================================================================

EXIT_CRITERIA_TARGETS = {
    "forward_velocity": {"min": 0.3, "max": 0.8, "unit": "m/s"},
    "episode_reward": {"target": 350, "baseline": 250, "unit": ""},
    "amp_reward": {"target": 0.7, "unit": ""},
    "success_rate": {"target": 0.85, "unit": "%"},
    "avg_torque": {"target": 2.8, "max_allowed": 4.0, "unit": "Nm"},
    # Stability metrics
    "episode_length": {"target": 400, "min": 200, "unit": "steps"},
    "survival_rate": {"target": 0.95, "unit": "%"},
    "disc_accuracy": {"target_min": 0.4, "target_max": 0.7, "unit": ""},  # Should NOT be 100%
}


# Import WandbConfig from config.py (single source of truth)
# This prevents duplicate definitions and ensures consistency
from playground_amp.configs.training_config import WandbConfig


class WandbTracker:
    """W&B experiment tracker for training.

    Handles:
    - Metric logging (scalars, histograms)
    - Video logging (rollout recordings)
    - Config tracking
    - Checkpoint artifacts
    - Graceful fallback if W&B unavailable
    """

    def __init__(
        self,
        project: str = "wildrobot-locomotion",
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        notes: str = "",
        entity: Optional[str] = None,
        mode: str = "online",
        enabled: bool = True,
        log_dir: str = "playground_amp/wandb",
    ):
        """Initialize W&B tracker.

        Args:
            project: W&B project name
            name: Run name (auto-generated if None)
            config: Training configuration dict
            tags: List of tags for filtering
            notes: Notes/description for the run
            entity: W&B entity (team/user)
            mode: W&B mode ("online", "offline", "disabled")
            enabled: Whether to enable W&B logging
            log_dir: Local directory for wandb files
        """
        self.project = project
        self.name = name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.config = config or {}
        self.tags = tags or []
        self.notes = notes
        self.entity = entity
        self.mode = mode
        self.enabled = enabled
        self.log_dir = log_dir

        self._wandb_run = None
        self._run_dir = None  # Will be set after wandb.init()
        self._start_time = time.time()
        self._step = 0

        # Initialize W&B if enabled
        if enabled:
            self._init_wandb()

        # Save config locally (to wandb's run dir if available, else log_dir)
        self._save_config_local()

    def _init_wandb(self):
        """Initialize Weights & Biases tracking."""
        try:
            import wandb

            # Check if already initialized
            if wandb.run is not None:
                print(f"⚠️ W&B already initialized, using existing run")
                self._wandb_run = wandb.run
                self._run_dir = wandb.run.dir
                return

            # Set WANDB_DIR to parent of log_dir so wandb creates its subfolder there
            # wandb always creates a wandb/ subfolder, so:
            #   WANDB_DIR = "playground_amp" -> wandb creates "playground_amp/wandb/run-..."
            wandb_parent_dir = os.path.dirname(self.log_dir) or "."
            os.makedirs(wandb_parent_dir, exist_ok=True)
            os.environ["WANDB_DIR"] = wandb_parent_dir

            # Initialize W&B
            # Note: We don't set 'id' to avoid duplicate timestamps in folder name
            # wandb will generate a random 8-char ID: run-YYYYMMDD_HHMMSS-abc12xyz
            self._wandb_run = wandb.init(
                project=self.project,
                name=self.name,
                config=self.config,
                tags=self.tags,
                notes=self.notes,
                entity=self.entity,
                mode=self.mode,
            )

            # Store wandb's run directory for saving config/summary
            if self._wandb_run is not None:
                self._run_dir = self._wandb_run.dir
                print(f"✓ W&B initialized")
                print(f"  Project: {self.project}")
                print(f"  Run: {self.name}")
                print(f"  Dir: {self._run_dir}")
                if self._wandb_run.url:
                    print(f"  URL: {self._wandb_run.url}")

        except ImportError:
            print("⚠️ W&B not installed. Install with: pip install wandb")
            self.enabled = False
        except Exception as e:
            print(f"⚠️ W&B initialization failed: {e}")
            self.enabled = False

    def _save_config_local(self):
        """Save configuration to local file (in wandb's run dir if available)."""
        # Use wandb's run dir if available, otherwise create our own folder
        if self._run_dir:
            config_dir = self._run_dir
        else:
            config_dir = os.path.join(self.log_dir, self.name)
            os.makedirs(config_dir, exist_ok=True)

        config_path = os.path.join(config_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(
                {
                    "project": self.project,
                    "name": self.name,
                    "tags": self.tags,
                    "notes": self.notes,
                    "config": self.config,
                    "start_time": datetime.now().isoformat(),
                },
                f,
                indent=2,
                default=str,
            )

    def log(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        prefix: str = "",
    ):
        """Log metrics to W&B.

        Args:
            metrics: Dictionary of metric name -> value
            step: Training step (auto-increments if None)
            prefix: Prefix for metric names (e.g., "train/", "eval/")
        """
        if step is None:
            step = self._step
            self._step += 1
        else:
            self._step = step

        # Add prefix to metric names
        if prefix:
            metrics = {f"{prefix}{k}": v for k, v in metrics.items()}

        # Add elapsed time
        metrics["time/elapsed_seconds"] = time.time() - self._start_time
        metrics["time/step"] = step

        # Log to W&B
        if self.enabled and self._wandb_run is not None:
            try:
                import wandb
                wandb.log(metrics, step=step)
            except Exception as e:
                print(f"⚠️ W&B logging failed: {e}")

    def log_histogram(
        self,
        name: str,
        values: np.ndarray,
        step: Optional[int] = None,
    ):
        """Log histogram data.

        Args:
            name: Histogram name
            values: Array of values
            step: Training step
        """
        if not self.enabled or self._wandb_run is None:
            return

        step = step or self._step

        try:
            import wandb
            wandb.log({name: wandb.Histogram(values)}, step=step)
        except Exception as e:
            print(f"⚠️ W&B histogram logging failed: {e}")

    def log_video(
        self,
        name: str,
        frames: np.ndarray,
        step: Optional[int] = None,
        fps: int = 30,
    ):
        """Log video data.

        Args:
            name: Video name
            frames: Array of frames (T, H, W, C) or (T, C, H, W)
            step: Training step
            fps: Frames per second
        """
        if not self.enabled or self._wandb_run is None:
            return

        step = step or self._step

        # Ensure correct shape (T, C, H, W) for W&B
        if frames.ndim == 4:
            if frames.shape[-1] in [1, 3, 4]:  # (T, H, W, C)
                frames = np.transpose(frames, (0, 3, 1, 2))
        else:
            print(f"⚠️ Invalid video shape: {frames.shape}")
            return

        try:
            import wandb
            wandb.log({name: wandb.Video(frames, fps=fps)}, step=step)
        except Exception as e:
            print(f"⚠️ W&B video logging failed: {e}")

    def log_image(
        self,
        name: str,
        image: np.ndarray,
        step: Optional[int] = None,
    ):
        """Log image data.

        Args:
            name: Image name
            image: Image array (H, W, C) or (H, W)
            step: Training step
        """
        if not self.enabled or self._wandb_run is None:
            return

        step = step or self._step

        try:
            import wandb
            wandb.log({name: wandb.Image(image)}, step=step)
        except Exception as e:
            print(f"⚠️ W&B image logging failed: {e}")

    def log_table(
        self,
        name: str,
        columns: List[str],
        data: List[List[Any]],
    ):
        """Log tabular data.

        Args:
            name: Table name
            columns: Column names
            data: List of rows
        """
        if not self.enabled or self._wandb_run is None:
            return

        try:
            import wandb
            table = wandb.Table(columns=columns, data=data)
            wandb.log({name: table})
        except Exception as e:
            print(f"⚠️ W&B table logging failed: {e}")

    def log_summary(self, metrics: Dict[str, Any]):
        """Log summary metrics (final values).

        Args:
            metrics: Dictionary of summary metrics
        """
        if self.enabled and self._wandb_run is not None:
            try:
                import wandb
                for key, value in metrics.items():
                    wandb.run.summary[key] = value
            except Exception as e:
                print(f"⚠️ W&B summary logging failed: {e}")

        # Also save to local file (in wandb's run dir if available)
        if self._run_dir:
            summary_path = os.path.join(self._run_dir, "summary.json")
        else:
            summary_dir = os.path.join(self.log_dir, self.name)
            os.makedirs(summary_dir, exist_ok=True)
            summary_path = os.path.join(summary_dir, "summary.json")

        with open(summary_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)

    def save_artifact(
        self,
        filepath: str,
        name: str,
        artifact_type: str = "model",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Save file as W&B artifact.

        Args:
            filepath: Path to file
            name: Artifact name
            artifact_type: Type of artifact (e.g., "model", "dataset")
            metadata: Optional metadata dict
        """
        if not self.enabled or self._wandb_run is None:
            return

        try:
            import wandb
            artifact = wandb.Artifact(
                name=name,
                type=artifact_type,
                metadata=metadata or {},
            )
            artifact.add_file(filepath)
            wandb.log_artifact(artifact)
        except Exception as e:
            print(f"⚠️ W&B artifact logging failed: {e}")

    def finish(self):
        """Finalize tracking and close W&B connection."""
        # Log final elapsed time
        elapsed = time.time() - self._start_time
        self.log_summary(
            {
                "total_elapsed_seconds": elapsed,
                "total_elapsed_minutes": elapsed / 60,
                "final_step": self._step,
            }
        )

        if self.enabled and self._wandb_run is not None:
            try:
                import wandb
                wandb.finish()
                print(f"✓ W&B run finished")
            except Exception as e:
                print(f"⚠️ W&B finish failed: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()
        return False

    @property
    def url(self) -> Optional[str]:
        """Get W&B run URL."""
        if self._wandb_run is not None:
            return self._wandb_run.url
        return None


def create_training_metrics(
    iteration: int,
    episode_reward: float,
    ppo_loss: float,
    policy_loss: float,
    value_loss: float,
    entropy_loss: float,
    disc_loss: float = 0.0,
    disc_accuracy: float = 0.5,
    amp_reward_mean: float = 0.0,
    amp_reward_std: float = 0.0,
    clip_fraction: float = 0.0,
    approx_kl: float = 0.0,
    env_steps_per_sec: float = 0.0,
    forward_velocity: float = 0.0,
    success_rate: float = 0.0,
    avg_torque: float = 0.0,
    episode_length: float = 0.0,
    **extra_metrics,
) -> Dict[str, float]:
    """Create a flat dictionary of training metrics for logging.

    Includes TOPLINE metrics (Section 3.2.4 Exit Criteria) for easy tracking:
    - topline/episode_reward (target: >350)
    - topline/amp_reward (target: >0.7)
    - topline/forward_velocity (target: 0.3-0.8 m/s)
    - topline/success_rate (target: >85%)
    - topline/avg_torque (target: <2.8 Nm)

    Args:
        iteration: Current training iteration
        episode_reward: Mean episode reward
        ppo_loss: Total PPO loss
        policy_loss: Policy loss component
        value_loss: Value loss component
        entropy_loss: Entropy loss component
        disc_loss: Discriminator loss (AMP)
        disc_accuracy: Discriminator accuracy (AMP)
        amp_reward_mean: Mean AMP reward
        amp_reward_std: Std of AMP reward
        clip_fraction: PPO clip fraction
        approx_kl: Approximate KL divergence
        env_steps_per_sec: Environment steps per second
        forward_velocity: Current forward velocity (m/s)
        success_rate: Episode success rate (0-1)
        avg_torque: Average torque per joint (Nm)
        episode_length: Mean episode length (steps)
        **extra_metrics: Additional metrics

    Returns:
        Flat dictionary of metrics with prefixes
    """
    metrics = {
        # =====================================================================
        # TOPLINE METRICS (Section 3.2.4 Exit Criteria)
        # These are the key metrics to track training progress
        # =====================================================================
        "topline/episode_reward": episode_reward,      # Target: >350
        "topline/amp_reward": amp_reward_mean,          # Target: >0.7
        "topline/forward_velocity": forward_velocity,   # Target: 0.3-0.8 m/s
        "topline/success_rate": success_rate * 100,     # Target: >85%
        "topline/avg_torque": avg_torque,               # Target: <2.8 Nm
        "topline/steps_per_sec": env_steps_per_sec,     # Performance metric
        # Stability metrics
        "topline/episode_length": episode_length,       # Target: >400 steps
        "topline/disc_accuracy": disc_accuracy,         # Target: 0.4-0.7 (NOT 100%!)
        # =====================================================================

        # Environment metrics
        "env/episode_reward": episode_reward,
        "env/episode_length": episode_length,
        "env/steps_per_sec": env_steps_per_sec,
        "env/forward_velocity": forward_velocity,
        "env/success_rate": success_rate,
        "env/avg_torque": avg_torque,

        # PPO metrics
        "ppo/total_loss": ppo_loss,
        "ppo/policy_loss": policy_loss,
        "ppo/value_loss": value_loss,
        "ppo/entropy_loss": entropy_loss,
        "ppo/clip_fraction": clip_fraction,
        "ppo/approx_kl": approx_kl,

        # AMP metrics
        "amp/disc_loss": disc_loss,
        "amp/disc_accuracy": disc_accuracy,
        "amp/reward_mean": amp_reward_mean,
        "amp/reward_std": amp_reward_std,

        # Progress
        "progress/iteration": iteration,
    }

    # Add extra metrics
    for key, value in extra_metrics.items():
        if isinstance(value, (int, float, np.number)):
            metrics[key] = float(value)
        elif hasattr(value, "item"):
            metrics[key] = float(value.item())

    return metrics


def create_topline_metrics_only(
    episode_reward: float = 0.0,
    amp_reward: float = 0.0,
    forward_velocity: float = 0.0,
    success_rate: float = 0.0,
    avg_torque: float = 0.0,
    steps_per_sec: float = 0.0,
) -> Dict[str, float]:
    """Create only the topline metrics for quick logging.

    Use this when you only want to update the key exit criteria metrics.

    Args:
        episode_reward: Mean episode reward (target: >350)
        amp_reward: Mean AMP reward (target: >0.7)
        forward_velocity: Forward velocity in m/s (target: 0.3-0.8)
        success_rate: Success rate 0-1 (target: >0.85)
        avg_torque: Average torque in Nm (target: <2.8)
        steps_per_sec: Training speed

    Returns:
        Dictionary with topline/ prefixed metrics
    """
    return {
        "topline/episode_reward": episode_reward,
        "topline/amp_reward": amp_reward,
        "topline/forward_velocity": forward_velocity,
        "topline/success_rate": success_rate * 100,  # Convert to percentage
        "topline/avg_torque": avg_torque,
        "topline/steps_per_sec": steps_per_sec,
    }


def define_wandb_topline_metrics():
    """Define W&B metrics with targets for Section 3.2.4 exit criteria.

    This sets up:
    1. Metric definitions with step_metric
    2. Summary statistics (min, max, mean)
    3. Target reference lines for charts

    Should be called after wandb.init() in the training script.
    """
    try:
        import wandb

        if wandb.run is None:
            return

        # Define topline metrics to sync with global step
        wandb.define_metric("topline/*", step_metric="progress/iteration")

        # Store targets in config for reference
        wandb.config.update({
            "exit_criteria": {
                "episode_reward_target": 350,
                "episode_reward_baseline": 250,
                "amp_reward_target": 0.7,
                "forward_velocity_min": 0.3,
                "forward_velocity_max": 0.8,
                "success_rate_target": 85,  # percentage
                "avg_torque_target": 2.8,
                "avg_torque_max": 4.0,
            }
        }, allow_val_change=True)

        # Log target reference lines as constants (these appear as horizontal lines)
        # We log them once at step 0 with a special prefix
        wandb.log({
            "targets/episode_reward": 350,
            "targets/amp_reward": 0.7,
            "targets/forward_velocity_min": 0.3,
            "targets/forward_velocity_max": 0.8,
            "targets/success_rate": 85,
            "targets/avg_torque": 2.8,
        }, step=0)

        print("✓ W&B topline metrics defined with exit criteria targets")

    except ImportError:
        pass
    except Exception as e:
        print(f"⚠️ Failed to define W&B topline metrics: {e}")


def generate_eval_video(
    env,
    inference_fn,
    rng_key,
    episode_length: int = 500,
    num_rollouts: int = 1,
    render_every: int = 2,
    height: int = 480,
    width: int = 640,
) -> List[np.ndarray]:
    """Generate evaluation rollout videos.

    Reference: mujoco_playground/learning/train_jax_ppo.py

    Args:
        env: Environment instance with reset(), step(), and render() methods
        inference_fn: JIT-compiled inference function (obs) -> action
        rng_key: JAX random key
        episode_length: Number of steps per rollout
        num_rollouts: Number of rollout videos to generate
        render_every: Render every N frames (for speed)
        height: Video height in pixels
        width: Video width in pixels

    Returns:
        List of video arrays, each shape (num_frames, height, width, 3)
    """
    try:
        import jax
        import jax.numpy as jnp
        import mujoco
    except ImportError as e:
        print(f"⚠️ Cannot generate video: {e}")
        return []

    print(f"\nGenerating {num_rollouts} evaluation video(s)...")

    videos = []

    for i in range(num_rollouts):
        # Reset environment
        rng_key, reset_key, rollout_key = jax.random.split(rng_key, 3)
        state = env.reset(reset_key)

        # Collect trajectory
        frames = []
        for step in range(episode_length):
            # Get action from policy
            rollout_key, action_key = jax.random.split(rollout_key)
            action = inference_fn(state.obs, action_key)

            # Step environment
            state = env.step(state, action)

            # Render frame (every N steps for speed)
            if step % render_every == 0:
                try:
                    # Try to render using env's render method
                    if hasattr(env, 'render'):
                        frame = env.render(state, height=height, width=width)
                        if frame is not None:
                            frames.append(np.array(frame))
                except Exception as e:
                    # Skip rendering if it fails
                    if step == 0:
                        print(f"⚠️ Rendering failed: {e}")
                    break

            # Stop if episode is done
            if hasattr(state, 'done') and state.done:
                break

        if frames:
            video = np.stack(frames, axis=0)
            videos.append(video)
            print(f"  Video {i+1}: {len(frames)} frames")

    return videos


def save_and_upload_video(
    video: np.ndarray,
    filepath: str,
    fps: float = 30.0,
    upload_to_wandb: bool = True,
    wandb_key: str = "eval/rollout_video",
) -> Optional[str]:
    """Save video to file and optionally upload to W&B.

    Args:
        video: Video array, shape (num_frames, height, width, 3)
        filepath: Path to save video file (.mp4)
        fps: Frames per second
        upload_to_wandb: Whether to upload to W&B
        wandb_key: W&B metric key for the video

    Returns:
        Path to saved video file, or None if failed
    """
    if video is None or len(video) == 0:
        print("⚠️ No video frames to save")
        return None

    try:
        # Try mediapy first (preferred)
        try:
            import mediapy as media
            media.write_video(filepath, video, fps=fps)
            print(f"✓ Video saved to: {filepath}")
        except ImportError:
            # Fallback to imageio
            try:
                import imageio
                imageio.mimsave(filepath, video, fps=fps)
                print(f"✓ Video saved to: {filepath}")
            except ImportError:
                print("⚠️ Neither mediapy nor imageio installed. Cannot save video.")
                print("  Install with: pip install mediapy  OR  pip install imageio[ffmpeg]")
                return None

        # Upload to W&B
        if upload_to_wandb:
            try:
                import wandb
                if wandb.run is not None:
                    # W&B expects (time, channels, height, width) for Video
                    # Our video is (time, height, width, channels)
                    video_wandb = np.transpose(video, (0, 3, 1, 2))
                    wandb.log({wandb_key: wandb.Video(video_wandb, fps=int(fps), format="mp4")})
                    print(f"✓ Video uploaded to W&B as '{wandb_key}'")
            except Exception as e:
                print(f"⚠️ W&B video upload failed: {e}")

        return filepath

    except Exception as e:
        print(f"⚠️ Video save failed: {e}")
        return None


def generate_and_upload_eval_videos(
    env,
    policy_params,
    policy_network,
    rng_key,
    output_dir: str = "videos",
    num_videos: int = 1,
    episode_length: int = 500,
    fps: float = 25.0,
    upload_to_wandb: bool = True,
) -> List[str]:
    """Generate evaluation videos and upload to W&B.

    This is a convenience function that combines video generation, saving, and upload.

    Args:
        env: Environment instance
        policy_params: Trained policy parameters
        policy_network: Policy network module
        rng_key: JAX random key
        output_dir: Directory to save videos
        num_videos: Number of videos to generate
        episode_length: Steps per episode
        fps: Video frames per second
        upload_to_wandb: Whether to upload to W&B

    Returns:
        List of saved video file paths
    """
    try:
        import jax
    except ImportError:
        print("⚠️ JAX not available for video generation")
        return []

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create inference function
    def inference_fn(obs, rng):
        """Get action from policy."""
        # This is a simplified inference - actual implementation depends on network architecture
        try:
            action_dist = policy_network.apply(policy_params, obs)
            if hasattr(action_dist, 'sample'):
                return action_dist.sample(seed=rng)
            elif hasattr(action_dist, 'mode'):
                return action_dist.mode()
            else:
                return action_dist  # Assume it's already the action
        except Exception:
            # Fallback: assume policy_network returns action directly
            return policy_network.apply(policy_params, obs)

    jit_inference_fn = jax.jit(inference_fn)

    # Generate videos
    videos = generate_eval_video(
        env=env,
        inference_fn=jit_inference_fn,
        rng_key=rng_key,
        episode_length=episode_length,
        num_rollouts=num_videos,
        render_every=2,
    )

    # Save and upload videos
    saved_paths = []
    for i, video in enumerate(videos):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(output_dir, f"eval_rollout_{timestamp}_{i}.mp4")
        wandb_key = f"eval/rollout_video_{i}" if i > 0 else "eval/rollout_video"

        path = save_and_upload_video(
            video=video,
            filepath=filepath,
            fps=fps,
            upload_to_wandb=upload_to_wandb,
            wandb_key=wandb_key,
        )
        if path:
            saved_paths.append(path)

    return saved_paths
