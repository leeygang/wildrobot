"""Experiment Tracking Module for WildRobot Training.

This module provides unified logging to:
- Weights & Biases (primary) - cloud-based, rich visualization
- TensorBoard (backup) - local, simple

Usage:
    from playground_amp.training.experiment_tracking import ExperimentTracker

    tracker = ExperimentTracker(
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
from typing import Any, Dict, List, Optional, Union

import numpy as np


@dataclass
class ExperimentConfig:
    """Configuration for experiment tracking."""

    # Project settings
    project: str = "wildrobot-locomotion"
    name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    notes: str = ""

    # Logging settings
    log_dir: str = "logs"
    use_wandb: bool = True
    use_tensorboard: bool = True

    # WandB settings
    wandb_entity: Optional[str] = None
    wandb_mode: str = "online"  # "online", "offline", "disabled"

    # Logging frequency
    log_frequency: int = 10  # Log every N iterations
    video_frequency: int = 100  # Log videos every N iterations

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    checkpoint_frequency: int = 100


class ExperimentTracker:
    """Unified experiment tracking for WandB and TensorBoard.

    Handles:
    - Metric logging (scalars, histograms)
    - Video logging (rollout recordings)
    - Config tracking
    - Checkpoint management
    - Graceful fallbacks if services unavailable
    """

    def __init__(
        self,
        project: str = "wildrobot-locomotion",
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        notes: str = "",
        log_dir: str = "logs",
        use_wandb: bool = True,
        use_tensorboard: bool = True,
        wandb_entity: Optional[str] = None,
        wandb_mode: str = "online",
    ):
        """Initialize experiment tracker.

        Args:
            project: Project name (WandB project, TensorBoard subdir)
            name: Run name (auto-generated if None)
            config: Training configuration dict
            tags: List of tags for filtering
            notes: Notes/description for the run
            log_dir: Base directory for logs
            use_wandb: Whether to use WandB
            use_tensorboard: Whether to use TensorBoard
            wandb_entity: WandB entity (team/user)
            wandb_mode: WandB mode ("online", "offline", "disabled")
        """
        self.project = project
        self.name = name or self._generate_run_name()
        self.config = config or {}
        self.tags = tags or []
        self.notes = notes
        self.log_dir = log_dir
        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard

        self._wandb_run = None
        self._tb_writer = None
        self._start_time = time.time()
        self._step = 0

        # Initialize backends
        if use_wandb:
            self._init_wandb(wandb_entity, wandb_mode)
        if use_tensorboard:
            self._init_tensorboard()

        # Log initial config
        self._log_config()

    def _generate_run_name(self) -> str:
        """Generate a unique run name based on timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"run_{timestamp}"

    def _init_wandb(self, entity: Optional[str], mode: str):
        """Initialize Weights & Biases tracking."""
        try:
            import wandb

            # Check if already initialized
            if wandb.run is not None:
                print(f"⚠️ WandB already initialized, using existing run")
                self._wandb_run = wandb.run
                return

            self._wandb_run = wandb.init(
                project=self.project,
                name=self.name,
                config=self.config,
                tags=self.tags,
                notes=self.notes,
                entity=entity,
                mode=mode,
                reinit=True,
            )
            print(f"✓ WandB initialized: {self._wandb_run.url}")

        except ImportError:
            print("⚠️ WandB not installed. Install with: pip install wandb")
            self.use_wandb = False
        except Exception as e:
            print(f"⚠️ WandB initialization failed: {e}")
            self.use_wandb = False

    def _init_tensorboard(self):
        """Initialize TensorBoard logging."""
        try:
            from torch.utils.tensorboard import SummaryWriter

            tb_log_dir = os.path.join(self.log_dir, self.project, self.name)
            os.makedirs(tb_log_dir, exist_ok=True)
            self._tb_writer = SummaryWriter(log_dir=tb_log_dir)
            print(f"✓ TensorBoard initialized: {tb_log_dir}")

        except ImportError:
            # Try tensorboardX as fallback
            try:
                from tensorboardX import SummaryWriter

                tb_log_dir = os.path.join(self.log_dir, self.project, self.name)
                os.makedirs(tb_log_dir, exist_ok=True)
                self._tb_writer = SummaryWriter(log_dir=tb_log_dir)
                print(f"✓ TensorBoardX initialized: {tb_log_dir}")

            except ImportError:
                print(
                    "⚠️ TensorBoard not installed. Install with: pip install tensorboard"
                )
                self.use_tensorboard = False
        except Exception as e:
            print(f"⚠️ TensorBoard initialization failed: {e}")
            self.use_tensorboard = False

    def _log_config(self):
        """Log configuration to file."""
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
        """Log metrics to all backends.

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

        # Log to WandB
        if self.use_wandb and self._wandb_run is not None:
            try:
                import wandb

                wandb.log(metrics, step=step)
            except Exception as e:
                print(f"⚠️ WandB logging failed: {e}")

        # Log to TensorBoard
        if self.use_tensorboard and self._tb_writer is not None:
            try:
                for name, value in metrics.items():
                    if isinstance(value, (int, float, np.number)):
                        self._tb_writer.add_scalar(name, value, step)
                    elif isinstance(value, np.ndarray) and value.ndim == 0:
                        self._tb_writer.add_scalar(name, float(value), step)
            except Exception as e:
                print(f"⚠️ TensorBoard logging failed: {e}")

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
        step = step or self._step

        if self.use_wandb and self._wandb_run is not None:
            try:
                import wandb

                wandb.log({name: wandb.Histogram(values)}, step=step)
            except Exception as e:
                print(f"⚠️ WandB histogram logging failed: {e}")

        if self.use_tensorboard and self._tb_writer is not None:
            try:
                self._tb_writer.add_histogram(name, values, step)
            except Exception as e:
                print(f"⚠️ TensorBoard histogram logging failed: {e}")

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
        step = step or self._step

        # Ensure correct shape (T, C, H, W) for WandB
        if frames.ndim == 4:
            if frames.shape[-1] in [1, 3, 4]:  # (T, H, W, C)
                frames_wandb = np.transpose(frames, (0, 3, 1, 2))
            else:  # Already (T, C, H, W)
                frames_wandb = frames
        else:
            print(f"⚠️ Invalid video shape: {frames.shape}")
            return

        if self.use_wandb and self._wandb_run is not None:
            try:
                import wandb

                wandb.log({name: wandb.Video(frames_wandb, fps=fps)}, step=step)
            except Exception as e:
                print(f"⚠️ WandB video logging failed: {e}")

        if self.use_tensorboard and self._tb_writer is not None:
            try:
                # TensorBoard expects (N, T, C, H, W)
                self._tb_writer.add_video(name, frames_wandb[None, ...], step, fps=fps)
            except Exception as e:
                print(f"⚠️ TensorBoard video logging failed: {e}")

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
        step = step or self._step

        if self.use_wandb and self._wandb_run is not None:
            try:
                import wandb

                wandb.log({name: wandb.Image(image)}, step=step)
            except Exception as e:
                print(f"⚠️ WandB image logging failed: {e}")

        if self.use_tensorboard and self._tb_writer is not None:
            try:
                # TensorBoard expects (C, H, W)
                if image.ndim == 3:
                    image_tb = np.transpose(image, (2, 0, 1))
                else:
                    image_tb = image[None, ...]  # Add channel dim
                self._tb_writer.add_image(name, image_tb, step)
            except Exception as e:
                print(f"⚠️ TensorBoard image logging failed: {e}")

    def log_table(
        self,
        name: str,
        columns: List[str],
        data: List[List[Any]],
    ):
        """Log tabular data (WandB only).

        Args:
            name: Table name
            columns: Column names
            data: List of rows
        """
        if self.use_wandb and self._wandb_run is not None:
            try:
                import wandb

                table = wandb.Table(columns=columns, data=data)
                wandb.log({name: table})
            except Exception as e:
                print(f"⚠️ WandB table logging failed: {e}")

    def log_summary(self, metrics: Dict[str, Any]):
        """Log summary metrics (final values).

        Args:
            metrics: Dictionary of summary metrics
        """
        if self.use_wandb and self._wandb_run is not None:
            try:
                import wandb

                for key, value in metrics.items():
                    wandb.run.summary[key] = value
            except Exception as e:
                print(f"⚠️ WandB summary logging failed: {e}")

        # Also save to file
        summary_path = os.path.join(
            self.log_dir, self.project, self.name, "summary.json"
        )
        with open(summary_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)

    def save_checkpoint(
        self,
        checkpoint_data: Dict[str, Any],
        name: str = "checkpoint",
        step: Optional[int] = None,
    ) -> str:
        """Save training checkpoint.

        Args:
            checkpoint_data: Data to save
            name: Checkpoint name
            step: Training step (appended to filename)

        Returns:
            Path to saved checkpoint
        """
        import pickle

        checkpoint_dir = os.path.join(self.log_dir, self.project, self.name, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        if step is not None:
            filename = f"{name}_step_{step:06d}.pkl"
        else:
            filename = f"{name}.pkl"

        filepath = os.path.join(checkpoint_dir, filename)

        with open(filepath, "wb") as f:
            pickle.dump(checkpoint_data, f)

        # Log to WandB as artifact
        if self.use_wandb and self._wandb_run is not None:
            try:
                import wandb

                artifact = wandb.Artifact(
                    name=f"{self.name}-checkpoint",
                    type="model",
                    metadata={"step": step},
                )
                artifact.add_file(filepath)
                wandb.log_artifact(artifact)
            except Exception as e:
                print(f"⚠️ WandB artifact logging failed: {e}")

        return filepath

    def finish(self):
        """Finalize tracking and close connections."""
        # Log final elapsed time
        elapsed = time.time() - self._start_time
        self.log_summary(
            {
                "total_elapsed_seconds": elapsed,
                "total_elapsed_minutes": elapsed / 60,
                "final_step": self._step,
            }
        )

        if self.use_wandb and self._wandb_run is not None:
            try:
                import wandb

                wandb.finish()
                print(f"✓ WandB run finished")
            except Exception as e:
                print(f"⚠️ WandB finish failed: {e}")

        if self.use_tensorboard and self._tb_writer is not None:
            try:
                self._tb_writer.close()
                print(f"✓ TensorBoard writer closed")
            except Exception as e:
                print(f"⚠️ TensorBoard close failed: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()
        return False


def create_training_metrics(
    ppo_metrics: Any,
    amp_metrics: Optional[Dict[str, float]] = None,
    env_metrics: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Create a flat dictionary of training metrics for logging.

    Args:
        ppo_metrics: PPO loss output (PPOLossOutput namedtuple)
        amp_metrics: AMP discriminator metrics
        env_metrics: Environment metrics (rewards, episode lengths)

    Returns:
        Flat dictionary of metrics
    """
    metrics = {}

    # PPO metrics
    if ppo_metrics is not None:
        if hasattr(ppo_metrics, "_asdict"):
            ppo_dict = ppo_metrics._asdict()
        elif isinstance(ppo_metrics, dict):
            ppo_dict = ppo_metrics
        else:
            ppo_dict = {}

        for key, value in ppo_dict.items():
            if isinstance(value, (int, float, np.number)):
                metrics[f"ppo/{key}"] = float(value)
            elif hasattr(value, "item"):
                metrics[f"ppo/{key}"] = float(value.item())

    # AMP metrics
    if amp_metrics is not None:
        for key, value in amp_metrics.items():
            if isinstance(value, (int, float, np.number)):
                metrics[f"amp/{key}"] = float(value)
            elif hasattr(value, "item"):
                metrics[f"amp/{key}"] = float(value.item())

    # Environment metrics
    if env_metrics is not None:
        for key, value in env_metrics.items():
            if isinstance(value, (int, float, np.number)):
                metrics[f"env/{key}"] = float(value)
            elif hasattr(value, "item"):
                metrics[f"env/{key}"] = float(value.item())

    return metrics


# Convenience functions for quick setup
def quick_wandb_init(
    name: str,
    config: Dict[str, Any],
    project: str = "wildrobot-locomotion",
    tags: Optional[List[str]] = None,
) -> ExperimentTracker:
    """Quick initialization for WandB-only tracking.

    Args:
        name: Run name
        config: Training configuration
        project: Project name
        tags: Optional tags

    Returns:
        ExperimentTracker instance
    """
    return ExperimentTracker(
        project=project,
        name=name,
        config=config,
        tags=tags or [],
        use_wandb=True,
        use_tensorboard=False,
    )


def quick_tensorboard_init(
    name: str,
    config: Dict[str, Any],
    log_dir: str = "logs",
) -> ExperimentTracker:
    """Quick initialization for TensorBoard-only tracking.

    Args:
        name: Run name
        config: Training configuration
        log_dir: Log directory

    Returns:
        ExperimentTracker instance
    """
    return ExperimentTracker(
        project="wildrobot",
        name=name,
        config=config,
        use_wandb=False,
        use_tensorboard=True,
        log_dir=log_dir,
    )


if __name__ == "__main__":
    # Test the experiment tracker
    print("Testing ExperimentTracker...")

    config = {
        "algorithm": "PPO",
        "num_envs": 4096,
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "amp_enabled": True,
    }

    # Test with TensorBoard only (doesn't require WandB login)
    tracker = ExperimentTracker(
        project="wildrobot-test",
        name="test_run",
        config=config,
        tags=["test", "ppo", "amp"],
        notes="Test run for experiment tracking",
        use_wandb=False,  # Disable WandB for test
        use_tensorboard=True,
    )

    # Simulate training loop
    for step in range(10):
        metrics = {
            "loss/policy": 0.5 - step * 0.03,
            "loss/value": 1.0 - step * 0.05,
            "reward/episode": step * 10 + np.random.randn() * 5,
            "reward/amp": -2.0 + step * 0.2,
            "performance/fps": 1000 + np.random.randint(-50, 50),
        }
        tracker.log(metrics, step=step)

    # Log summary
    tracker.log_summary(
        {
            "final_reward": 100.0,
            "final_amp_reward": 0.5,
            "success_rate": 0.85,
        }
    )

    tracker.finish()
    print("\n✅ ExperimentTracker test passed!")
