"""Checkpoint management for AMP+PPO training.

This module handles saving, loading, and managing training checkpoints.
All checkpoint I/O operations are centralized here, keeping the training
loop (trainer_jit.py) focused on pure computation.

Usage:
    from playground_amp.training.checkpoint import (
        save_checkpoint,
        load_checkpoint,
        list_best_checkpoints,
        manage_checkpoints,
        print_best_checkpoints_summary,
    )
"""

from __future__ import annotations

import os
import pickle
from typing import Any, Dict, Optional

import jax


def save_checkpoint(
    state: Any,
    config: Any,
    iteration: int,
    checkpoint_dir: str,
    is_best: bool = False,
    metrics: Optional[Any] = None,
) -> str:
    """Save training checkpoint.

    Args:
        state: Current training state (TrainingState from trainer_jit)
        config: Training configuration (AMPPPOConfigJit)
        iteration: Current iteration number
        checkpoint_dir: Directory to save checkpoints
        is_best: If True, also save as best checkpoint
        metrics: Optional IterationMetrics to save with checkpoint

    Returns:
        Path to saved checkpoint
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Store metrics if provided
    metrics_dict = None
    if metrics is not None:
        metrics_dict = {
            "episode_reward": float(metrics.episode_reward),
            "amp_reward_mean": float(metrics.amp_reward_mean),
            "disc_accuracy": float(metrics.disc_accuracy),
            "task_reward_mean": float(metrics.task_reward_mean),
            "forward_velocity": float(metrics.forward_velocity),
            "robot_height": float(metrics.robot_height),
            "episode_length": float(metrics.episode_length),
            "policy_loss": float(metrics.policy_loss),
            "value_loss": float(metrics.value_loss),
            "disc_loss": float(metrics.disc_loss),
        }

    checkpoint_data = {
        "iteration": iteration,
        "total_steps": int(state.total_steps),
        "metrics": metrics_dict,
        "policy_params": jax.device_get(state.policy_params),
        "value_params": jax.device_get(state.value_params),
        "processor_params": jax.device_get(state.processor_params),
        "disc_params": jax.device_get(state.disc_params),
        "policy_opt_state": jax.device_get(state.policy_opt_state),
        "value_opt_state": jax.device_get(state.value_opt_state),
        "disc_opt_state": jax.device_get(state.disc_opt_state),
        "feature_mean": jax.device_get(state.feature_mean),
        "feature_var": jax.device_get(state.feature_var),
        "feature_count": jax.device_get(state.feature_count),
        "rng": jax.device_get(state.rng),
        "config": {
            "obs_dim": config.obs_dim,
            "action_dim": config.action_dim,
            "num_envs": config.num_envs,
            "num_steps": config.num_steps,
            "learning_rate": config.learning_rate,
            "gamma": config.gamma,
            "gae_lambda": config.gae_lambda,
            "clip_epsilon": config.clip_epsilon,
            "value_loss_coef": config.value_loss_coef,
            "entropy_coef": config.entropy_coef,
            "max_grad_norm": config.max_grad_norm,
            "num_minibatches": config.num_minibatches,
            "update_epochs": config.update_epochs,
            "policy_hidden_dims": config.policy_hidden_dims,
            "value_hidden_dims": config.value_hidden_dims,
            "amp_reward_weight": config.amp_reward_weight,
            "disc_learning_rate": config.disc_learning_rate,
            "disc_updates_per_iter": config.disc_updates_per_iter,
            "disc_batch_size": config.disc_batch_size,
            "gradient_penalty_weight": config.gradient_penalty_weight,
            "disc_hidden_dims": config.disc_hidden_dims,
            "label_smoothing": config.label_smoothing,
            "disc_input_noise_std": config.disc_input_noise_std,
            "total_iterations": config.total_iterations,
            "seed": config.seed,
        },
    }

    # Save iteration checkpoint
    checkpoint_path = os.path.join(
        checkpoint_dir, f"checkpoint_iter_{iteration:06d}.pkl"
    )
    with open(checkpoint_path, "wb") as f:
        pickle.dump(checkpoint_data, f)

    # Save as best if requested (with iteration number in filename)
    if is_best:
        best_path = os.path.join(
            checkpoint_dir, f"best_checkpoint_iter_{iteration:06d}.pkl"
        )
        with open(best_path, "wb") as f:
            pickle.dump(checkpoint_data, f)

    return checkpoint_path


def load_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """Load training checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Dictionary containing checkpoint data
    """
    with open(checkpoint_path, "rb") as f:
        checkpoint_data = pickle.load(f)
    return checkpoint_data


def list_checkpoints(checkpoint_dir: str) -> list:
    """List all checkpoints in directory, sorted by iteration.

    Args:
        checkpoint_dir: Directory containing checkpoints

    Returns:
        List of (iteration, filename) tuples sorted by iteration
    """
    if not os.path.exists(checkpoint_dir):
        return []

    checkpoints = []
    for f in os.listdir(checkpoint_dir):
        if f.startswith("checkpoint_iter_") and f.endswith(".pkl"):
            try:
                iter_num = int(f.split("_")[-1].replace(".pkl", ""))
                checkpoints.append((iter_num, f))
            except ValueError:
                continue

    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


def list_best_checkpoints(checkpoint_dir: str) -> list:
    """List all best checkpoints in directory, sorted by iteration.

    Args:
        checkpoint_dir: Directory containing checkpoints

    Returns:
        List of (iteration, filename) tuples sorted by iteration
    """
    if not os.path.exists(checkpoint_dir):
        return []

    checkpoints = []
    for f in os.listdir(checkpoint_dir):
        if f.startswith("best_checkpoint_iter_") and f.endswith(".pkl"):
            try:
                iter_num = int(f.split("_")[-1].replace(".pkl", ""))
                checkpoints.append((iter_num, f))
            except ValueError:
                continue

    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


def manage_checkpoints(checkpoint_dir: str, keep_last_n: int = 5) -> None:
    """Remove old checkpoints, keeping only the last N of each type.

    Cleans up both regular checkpoints (checkpoint_iter_*) and
    best checkpoints (best_checkpoint_iter_*).

    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_last_n: Number of recent checkpoints to keep (for each type)
    """
    if not os.path.exists(checkpoint_dir):
        return

    # Clean up regular checkpoints
    regular_checkpoints = list_checkpoints(checkpoint_dir)
    if len(regular_checkpoints) > keep_last_n:
        for iter_num, filename in regular_checkpoints[:-keep_last_n]:
            filepath = os.path.join(checkpoint_dir, filename)
            try:
                os.remove(filepath)
            except OSError:
                pass

    # Clean up best checkpoints
    best_checkpoints = list_best_checkpoints(checkpoint_dir)
    if len(best_checkpoints) > keep_last_n:
        for iter_num, filename in best_checkpoints[:-keep_last_n]:
            filepath = os.path.join(checkpoint_dir, filename)
            try:
                os.remove(filepath)
            except OSError:
                pass


def print_best_checkpoints_summary(checkpoint_dir: str) -> None:
    """Print summary of top 5 best checkpoints with metrics.

    Args:
        checkpoint_dir: Directory containing checkpoints
    """
    best_checkpoints = list_best_checkpoints(checkpoint_dir)
    if not best_checkpoints:
        print("No best checkpoints found.")
        return

    print()
    print("📊 Top 5 Best Checkpoints:")
    print("-" * 100)
    print(
        f"{'#':<3} {'Checkpoint':<35} {'Reward':>10} {'AMP':>10} "
        f"{'Disc Acc':>10} {'Vel (m/s)':>10} {'Height':>10} {'Ep Len':>10}"
    )
    print("-" * 100)

    for i, (iter_num, filename) in enumerate(best_checkpoints[-5:], 1):
        filepath = os.path.join(checkpoint_dir, filename)
        try:
            ckpt_data = load_checkpoint(filepath)
            m = ckpt_data.get("metrics", {})
            if m:
                print(
                    f"{i:<3} {filename:<35} "
                    f"{m.get('episode_reward', 0):>10.2f} "
                    f"{m.get('amp_reward_mean', 0):>10.4f} "
                    f"{m.get('disc_accuracy', 0):>10.2f} "
                    f"{m.get('forward_velocity', 0):>10.2f} "
                    f"{m.get('robot_height', 0):>10.3f} "
                    f"{m.get('episode_length', 0):>10.1f}"
                )
            else:
                print(f"{i:<3} {filename:<35} (no metrics saved)")
        except Exception:
            print(f"{i:<3} {filename:<35} (failed to load)")

    print("-" * 100)
    print()
    print(
        f"💡 To resume from best: --resume "
        f"{os.path.join(checkpoint_dir, best_checkpoints[-1][1])}"
    )
    print()
