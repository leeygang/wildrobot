"""Checkpoint management for AMP+PPO training.

This module handles saving, loading, and managing training checkpoints.
All checkpoint I/O operations are centralized here, keeping the training
loop (trainer_jit.py) focused on pure computation.

Usage:
    from playground_amp.training.checkpoint import (
        save_checkpoint,
        load_checkpoint,
        get_top_checkpoints_by_reward,
        manage_checkpoints,
        print_top_checkpoints_summary,
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
    # Convert state to CPU and delegate to save_checkpoint_from_cpu
    state_cpu = jax.device_get(state)
    total_steps = int(state.total_steps)
    return save_checkpoint_from_cpu(
        state_cpu=state_cpu,
        config=config,
        iteration=iteration,
        total_steps=total_steps,
        checkpoint_dir=checkpoint_dir,
        is_best=is_best,
        metrics=metrics,
    )


def save_checkpoint_from_cpu(
    state_cpu: Any,
    config: Any,
    iteration: int,
    total_steps: int,
    checkpoint_dir: str,
    is_best: bool = False,
    metrics: Optional[Any] = None,
) -> str:
    """Save training checkpoint from pre-copied CPU state.

    This is useful when the state has already been copied to CPU (e.g., for
    tracking the best checkpoint within a window without repeated GPU->CPU transfers).

    Args:
        state_cpu: Training state already on CPU (via jax.device_get)
        config: Training configuration (AMPPPOConfigJit)
        iteration: Iteration number when this state was captured
        total_steps: Total steps when this state was captured
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
            "forward_velocity": float(metrics.env_metrics["forward_velocity"]),
            "robot_height": float(metrics.env_metrics["height"]),
            "episode_length": float(metrics.episode_length),
            "policy_loss": float(metrics.policy_loss),
            "value_loss": float(metrics.value_loss),
            "disc_loss": float(metrics.disc_loss),
        }

    # state_cpu is already on CPU, no need for jax.device_get
    checkpoint_data = {
        "iteration": iteration,
        "total_steps": total_steps,
        "metrics": metrics_dict,
        "policy_params": state_cpu.policy_params,
        "value_params": state_cpu.value_params,
        "processor_params": state_cpu.processor_params,
        "disc_params": state_cpu.disc_params,
        "policy_opt_state": state_cpu.policy_opt_state,
        "value_opt_state": state_cpu.value_opt_state,
        "disc_opt_state": state_cpu.disc_opt_state,
        "feature_mean": state_cpu.feature_mean,
        "feature_var": state_cpu.feature_var,
        "rng": state_cpu.rng,
        "config": {
            "num_envs": config.ppo.num_envs,
            "rollout_steps": config.ppo.rollout_steps,
            "learning_rate": config.ppo.learning_rate,
            "gamma": config.ppo.gamma,
            "gae_lambda": config.ppo.gae_lambda,
            "clip_epsilon": config.ppo.clip_epsilon,
            "value_loss_coef": config.ppo.value_loss_coef,
            "entropy_coef": config.ppo.entropy_coef,
            "max_grad_norm": config.ppo.max_grad_norm,
            "num_minibatches": config.ppo.num_minibatches,
            "epochs": config.ppo.epochs,
            "actor_hidden_sizes": config.networks.actor.hidden_sizes,
            "critic_hidden_sizes": config.networks.critic.hidden_sizes,
            "amp_weight": config.amp.weight,
            "disc_learning_rate": config.amp.discriminator.learning_rate,
            "disc_updates_per_ppo_update": config.amp.discriminator.updates_per_ppo_update,
            "disc_batch_size": config.amp.discriminator.batch_size,
            "disc_r1_gamma": config.amp.discriminator.r1_gamma,
            "disc_hidden_sizes": config.networks.discriminator.hidden_sizes,
            "disc_input_noise_std": config.amp.discriminator.input_noise_std,
            "iterations": config.ppo.iterations,
            "seed": config.seed,
        },
    }

    # Save iteration checkpoint (format: checkpoint_iteration_steps.pkl)
    checkpoint_path = os.path.join(
        checkpoint_dir, f"checkpoint_{iteration}_{total_steps}.pkl"
    )
    with open(checkpoint_path, "wb") as f:
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
        # Match format: checkpoint_iteration_steps.pkl
        if (
            f.startswith("checkpoint_")
            and not f.startswith("best_")
            and f.endswith(".pkl")
        ):
            try:
                parts = f.replace(".pkl", "").split("_")
                if len(parts) >= 3:
                    iter_num = int(parts[1])
                    checkpoints.append((iter_num, f))
            except ValueError:
                continue

    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


def get_top_checkpoints_by_reward(checkpoint_dir: str, top_n: int = 3) -> list:
    """Get top N checkpoints sorted by reward (highest first).

    Args:
        checkpoint_dir: Directory containing checkpoints
        top_n: Number of top checkpoints to return

    Returns:
        List of (reward, iteration, steps, filename, metrics) tuples sorted by reward descending
    """
    if not os.path.exists(checkpoint_dir):
        return []

    checkpoints_with_metrics = []
    for iter_num, filename in list_checkpoints(checkpoint_dir):
        filepath = os.path.join(checkpoint_dir, filename)
        try:
            ckpt_data = load_checkpoint(filepath)
            metrics = ckpt_data.get("metrics", {})
            if metrics:
                reward = metrics.get("episode_reward", float("-inf"))
                steps = ckpt_data.get("total_steps", 0)
                checkpoints_with_metrics.append(
                    (reward, iter_num, steps, filename, metrics)
                )
        except Exception:
            continue

    # Sort by reward descending
    checkpoints_with_metrics.sort(key=lambda x: x[0], reverse=True)
    return checkpoints_with_metrics[:top_n]


def manage_checkpoints(checkpoint_dir: str, keep_last_n: int = 5) -> None:
    """Remove old checkpoints, keeping only the last N.

    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_last_n: Number of recent checkpoints to keep
    """
    if not os.path.exists(checkpoint_dir):
        return

    # Clean up regular checkpoints (keep last N by iteration)
    regular_checkpoints = list_checkpoints(checkpoint_dir)
    if len(regular_checkpoints) > keep_last_n:
        for iter_num, filename in regular_checkpoints[:-keep_last_n]:
            filepath = os.path.join(checkpoint_dir, filename)
            try:
                os.remove(filepath)
            except OSError:
                pass


def print_top_checkpoints_summary(checkpoint_dir: str, top_n: int = 3) -> None:
    """Print summary of top N checkpoints ranked by reward.

    Parses all checkpoints and displays the best ones by episode reward.

    Args:
        checkpoint_dir: Directory containing checkpoints
        top_n: Number of top checkpoints to display
    """
    top_checkpoints = get_top_checkpoints_by_reward(checkpoint_dir, top_n)
    if not top_checkpoints:
        print("No checkpoints found.")
        return

    print()
    print(f"🏆 Top {len(top_checkpoints)} Checkpoints (by reward):")
    print("-" * 120)
    print(
        f"{'Rank':<5} {'Iter':>6} {'Steps':>12} "
        f"{'Reward':>10} {'AMP':>10} {'Disc Acc':>10} "
        f"{'Vel':>8} {'Height':>8} {'Ep Len':>8}"
    )
    print("-" * 120)

    for rank, (reward, iter_num, steps, filename, metrics) in enumerate(
        top_checkpoints, 1
    ):
        print(
            f"#{rank:<4} {iter_num:>6} {steps:>12,} "
            f"{reward:>10.2f} "
            f"{metrics.get('amp_reward_mean', 0):>10.4f} "
            f"{metrics.get('disc_accuracy', 0):>10.2f} "
            f"{metrics.get('forward_velocity', 0):>8.2f} "
            f"{metrics.get('robot_height', 0):>8.3f} "
            f"{metrics.get('episode_length', 0):>8.1f}"
        )

    print("-" * 120)

    # Show resume command for the best checkpoint
    if top_checkpoints:
        best_filename = top_checkpoints[0][3]
        print()
        print(
            f"💡 To resume from best: --resume "
            f"{os.path.join(checkpoint_dir, best_filename)}"
        )
    print()


def save_window_best_checkpoint(
    window_best: Dict[str, Any],
    best_reward: Dict[str, float],
    config: Any,
    checkpoint_dir: str,
    keep_last_n: int = 5,
) -> bool:
    """Save the best checkpoint from the current window.

    This function saves the best checkpoint tracked within a window of iterations,
    updates the global best reward if applicable, resets the window tracking state,
    and manages old checkpoints.

    Args:
        window_best: Dict tracking the best state within current window with keys:
            - "reward": Best reward in this window
            - "state": Training state (already on CPU via jax.device_get)
            - "metrics": IterationMetrics for the best iteration
            - "iteration": Iteration number of the best state
            - "total_steps": Total steps at the best iteration
        best_reward: Dict with "value" key tracking global best reward (mutated in place)
        config: Training configuration (TrainingConfig)
        checkpoint_dir: Directory to save checkpoints
        keep_last_n: Number of recent checkpoints to keep

    Returns:
        True if a checkpoint was saved, False if window_best was empty
    """
    if window_best["state"] is None:
        return False

    save_state = window_best["state"]
    save_metrics = window_best["metrics"]
    save_iteration = window_best["iteration"]
    save_reward = window_best["reward"]

    is_new_best = save_reward > best_reward["value"]
    if is_new_best:
        best_reward["value"] = save_reward

    save_checkpoint_from_cpu(
        state_cpu=save_state,
        config=config,
        iteration=save_iteration,
        total_steps=window_best["total_steps"],
        checkpoint_dir=checkpoint_dir,
        is_best=is_new_best,
        metrics=save_metrics,
    )

    if is_new_best:
        print(
            f"  💾 Checkpoint saved (window best, iter {save_iteration}) "
            f"- NEW BEST: reward={save_reward:.2f}"
        )
    else:
        print(
            f"  💾 Checkpoint saved (window best, iter {save_iteration}, "
            f"reward={save_reward:.2f})"
        )

    # Reset window tracking
    window_best["reward"] = float("-inf")
    window_best["state"] = None
    window_best["metrics"] = None
    window_best["iteration"] = 0
    window_best["total_steps"] = 0

    manage_checkpoints(checkpoint_dir, keep_last_n)

    return True
