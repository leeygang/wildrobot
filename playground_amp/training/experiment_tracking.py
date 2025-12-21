"""W&B Experiment Tracking Module for WildRobot Training.

This module provides logging to Weights & Biases for experiment tracking.

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


@dataclass
class WandbConfig:
    """Configuration for W&B experiment tracking."""

    # Enable/disable W&B
    enabled: bool = True

    # Project settings
    project: str = "wildrobot-locomotion"
    entity: Optional[str] = None  # W&B team/user
    name: Optional[str] = None  # Run name (auto-generated if None)
    tags: List[str] = field(default_factory=list)
    notes: str = ""

    # W&B mode: "online", "offline", "disabled"
    mode: str = "online"

    # Logging frequency
    log_frequency: int = 10  # Log every N iterations

    # Local log directory (for config backup)
    log_dir: str = "logs"


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
        log_dir: str = "logs",
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
            log_dir: Local directory for config backup
        """
        self.project = project
        self.name = name or self._generate_run_name()
        self.config = config or {}
        self.tags = tags or []
        self.notes = notes
        self.entity = entity
        self.mode = mode
        self.enabled = enabled
        self.log_dir = log_dir

        self._wandb_run = None
        self._start_time = time.time()
        self._step = 0

        # Initialize W&B if enabled
        if enabled:
            self._init_wandb()

        # Save config locally as backup
        self._save_config_local()

    def _generate_run_name(self) -> str:
        """Generate a unique run name based on timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"run_{timestamp}"

    def _init_wandb(self):
        """Initialize Weights & Biases tracking."""
        try:
            import wandb

            # Check if already initialized
            if wandb.run is not None:
                print(f"⚠️ W&B already initialized, using existing run")
                self._wandb_run = wandb.run
                return

            self._wandb_run = wandb.init(
                project=self.project,
                name=self.name,
                config=self.config,
                tags=self.tags,
                notes=self.notes,
                entity=self.entity,
                mode=self.mode,
                finish_previous=True,  # Finish any previous run before starting new one
            )
            print(f"✓ W&B initialized: {self._wandb_run.url}")

        except ImportError:
            print("⚠️ W&B not installed. Install with: pip install wandb")
            self.enabled = False
        except Exception as e:
            print(f"⚠️ W&B initialization failed: {e}")
            self.enabled = False

    def _save_config_local(self):
        """Save configuration to local file as backup."""
        config_dir = os.path.join(self.log_dir, self.project, self.name)
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

        # Also save to local file
        summary_path = os.path.join(
            self.log_dir, self.project, self.name, "summary.json"
        )
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
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
    **extra_metrics,
) -> Dict[str, float]:
    """Create a flat dictionary of training metrics for logging.

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
        **extra_metrics: Additional metrics

    Returns:
        Flat dictionary of metrics with prefixes
    """
    metrics = {
        # Environment metrics
        "env/episode_reward": episode_reward,
        "env/steps_per_sec": env_steps_per_sec,
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
