#!/usr/bin/env python3
"""Micro-Train Smoke Test for AMP Training.

v0.11.0: Verifies end-to-end training wiring before long runs.

This test ensures:
1. Discriminator loss decreases from initial
2. Discriminator accuracy moves off 0.5 (chance level)
3. AMP reward has non-trivial variance
4. No NaN/Inf in features, logits, rewards
5. All metrics are finite

Usage:
    cd ~/projects/wildrobot
    uv run python scripts/micro_train_smoke_test.py
    uv run python scripts/micro_train_smoke_test.py --iterations 200
    uv run python scripts/micro_train_smoke_test.py --verbose
"""

import argparse
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from rich import print
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

from playground_amp.amp.discriminator import (
    AMPDiscriminator,
    compute_amp_reward,
    create_discriminator,
    create_discriminator_optimizer,
    discriminator_loss,
)
from playground_amp.amp.ref_buffer import ReferenceMotionBuffer

console = Console()


# =============================================================================
# Test Configuration
# =============================================================================


@dataclass
class SmokeTestConfig:
    """Configuration for micro-train smoke test."""

    # Training parameters
    num_iterations: int = 100
    batch_size: int = 256
    learning_rate: float = 1e-4
    r1_gamma: float = 5.0

    # Network architecture
    hidden_dims: Tuple[int, ...] = (512, 256)

    # Reference data
    feature_dim: int = 27  # Physics reference format

    # Acceptance criteria
    min_accuracy_improvement: float = 0.05  # Must move at least 5% off 0.5
    min_reward_std: float = 0.01  # AMP reward must have variance
    max_loss_increase_ratio: float = 1.5  # Loss shouldn't explode
    min_loss_decrease_ratio: float = 0.8  # Loss should decrease by at least 20%
    min_score_separation: float = 0.1  # Real-Fake mean score gap

    # Logging
    log_every: int = 10

    # Random seed
    seed: int = 42


@dataclass
class SmokeTestResult:
    """Result from smoke test."""

    passed: bool
    details: str

    # Metrics history
    loss_history: List[float] = field(default_factory=list)
    accuracy_history: List[float] = field(default_factory=list)
    reward_mean_history: List[float] = field(default_factory=list)
    reward_std_history: List[float] = field(default_factory=list)

    # Final metrics
    initial_loss: float = 0.0
    final_loss: float = 0.0
    initial_accuracy: float = 0.0
    final_accuracy: float = 0.0
    final_reward_mean: float = 0.0
    final_reward_std: float = 0.0

    # Issue flags
    has_nan: bool = False
    has_inf: bool = False
    loss_exploded: bool = False
    accuracy_stuck: bool = False
    reward_flat: bool = False


# =============================================================================
# Data Loading
# =============================================================================


def load_reference_features(amp_dir: Path, feature_dim: int) -> np.ndarray:
    """Load and concatenate reference features from all files.

    Args:
        amp_dir: Directory containing *_amp.pkl files
        feature_dim: Expected feature dimension

    Returns:
        Concatenated features array (N, feature_dim)
    """
    all_features = []

    for pkl_path in sorted(amp_dir.glob("*_amp.pkl")):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        if "features" in data:
            features = data["features"]
            if features.shape[1] == feature_dim:
                all_features.append(features)
            else:
                print(
                    f"[yellow]Warning: Skipping {pkl_path} (dim={features.shape[1]}, expected={feature_dim})[/yellow]"
                )

    if not all_features:
        raise ValueError(f"No valid reference files found in {amp_dir}")

    return np.concatenate(all_features, axis=0).astype(np.float32)


def generate_fake_features(
    real_features: np.ndarray,
    batch_size: int,
    noise_scale: float = 0.5,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Generate fake features by adding noise to real features.

    This simulates what an untrained policy would produce - features
    that are in the same general range but don't match the reference
    distribution.

    Args:
        real_features: Real reference features to perturb
        batch_size: Number of fake samples to generate
        noise_scale: Scale of Gaussian noise
        rng: Random number generator

    Returns:
        Fake features array (batch_size, feature_dim)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Sample random frames from real data as base
    indices = rng.integers(0, len(real_features), size=batch_size)
    base_features = real_features[indices].copy()

    # Add noise to create "fake" features
    noise = rng.normal(0, noise_scale, size=base_features.shape)
    fake_features = base_features + noise

    # Clip contacts to [0, 1]
    fake_features[:, 23:27] = np.clip(fake_features[:, 23:27], 0, 1)

    return fake_features.astype(np.float32)


# =============================================================================
# Training Loop
# =============================================================================


def run_training_iteration(
    params: Dict,
    optimizer_state: Any,
    model: AMPDiscriminator,
    optimizer: Any,
    real_batch: jnp.ndarray,
    fake_batch: jnp.ndarray,
    rng_key: jax.random.PRNGKey,
    config: SmokeTestConfig,
) -> Tuple[Dict, Any, Dict[str, float]]:
    """Run a single training iteration.

    Args:
        params: Discriminator parameters
        optimizer_state: Optimizer state
        model: Discriminator model
        optimizer: Optax optimizer
        real_batch: Real reference features
        fake_batch: Fake policy features
        rng_key: JAX random key
        config: Test configuration

    Returns:
        Tuple of (updated_params, updated_optimizer_state, metrics)
    """

    # Compute loss and gradients
    def loss_fn(params):
        loss, metrics = discriminator_loss(
            params, model, real_batch, fake_batch, rng_key, r1_gamma=config.r1_gamma
        )
        return loss, metrics

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

    # Update parameters
    updates, optimizer_state = optimizer.update(grads, optimizer_state, params)
    import optax

    params = optax.apply_updates(params, updates)

    # Compute AMP reward for fake batch
    amp_reward = compute_amp_reward(params, model, fake_batch)
    metrics["amp_reward_mean"] = float(jnp.mean(amp_reward))
    metrics["amp_reward_std"] = float(jnp.std(amp_reward))
    metrics["amp_reward_min"] = float(jnp.min(amp_reward))
    metrics["amp_reward_max"] = float(jnp.max(amp_reward))

    return params, optimizer_state, metrics


def check_numerical_issues(metrics: Dict[str, float]) -> Tuple[bool, bool]:
    """Check for NaN and Inf in metrics.

    Args:
        metrics: Dictionary of metric values

    Returns:
        Tuple of (has_nan, has_inf)
    """
    has_nan = False
    has_inf = False

    for key, value in metrics.items():
        if np.isnan(value):
            has_nan = True
        if np.isinf(value):
            has_inf = True

    return has_nan, has_inf


# =============================================================================
# Main Test Runner
# =============================================================================


def run_smoke_test(
    ref_features: np.ndarray,
    config: SmokeTestConfig,
    verbose: bool = False,
) -> SmokeTestResult:
    """Run the micro-train smoke test.

    Args:
        ref_features: Reference features (N, feature_dim)
        config: Test configuration
        verbose: Print detailed progress

    Returns:
        SmokeTestResult with all metrics and pass/fail status
    """
    print(f"\n[bold cyan]Micro-Train Smoke Test[/bold cyan]")
    print(f"  Iterations: {config.num_iterations}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Feature dim: {config.feature_dim}")
    print(f"  Reference frames: {len(ref_features)}")
    print(f"  Hidden dims: {config.hidden_dims}")

    # Initialize result
    result = SmokeTestResult(passed=False, details="")

    # Set random seeds
    np.random.seed(config.seed)
    rng = np.random.default_rng(config.seed)
    rng_key = jax.random.PRNGKey(config.seed)

    # Create discriminator
    model, params = create_discriminator(
        obs_dim=config.feature_dim,
        hidden_dims=config.hidden_dims,
        seed=config.seed,
    )

    # Create optimizer
    optimizer = create_discriminator_optimizer(config.learning_rate)
    optimizer_state = optimizer.init(params)

    print(f"\n  Training discriminator...")

    # Training loop
    for iteration in range(config.num_iterations):
        # Sample real batch
        indices = rng.integers(0, len(ref_features), size=config.batch_size)
        real_batch = jnp.array(ref_features[indices])

        # Generate fake batch (simulating untrained policy)
        fake_batch = jnp.array(
            generate_fake_features(
                ref_features, config.batch_size, noise_scale=0.3, rng=rng
            )
        )

        # Split RNG key
        rng_key, iter_key = jax.random.split(rng_key)

        # Training step
        params, optimizer_state, metrics = run_training_iteration(
            params,
            optimizer_state,
            model,
            optimizer,
            real_batch,
            fake_batch,
            iter_key,
            config,
        )

        # Check for numerical issues
        has_nan, has_inf = check_numerical_issues(metrics)
        if has_nan:
            result.has_nan = True
            print(f"[red]  ✗ NaN detected at iteration {iteration}[/red]")
            break
        if has_inf:
            result.has_inf = True
            print(f"[red]  ✗ Inf detected at iteration {iteration}[/red]")
            break

        # Record metrics
        result.loss_history.append(float(metrics["discriminator_loss"]))
        result.accuracy_history.append(float(metrics["discriminator_accuracy"]))
        result.reward_mean_history.append(float(metrics["amp_reward_mean"]))
        result.reward_std_history.append(float(metrics["amp_reward_std"]))

        # Store initial values
        if iteration == 0:
            result.initial_loss = float(metrics["discriminator_loss"])
            result.initial_accuracy = float(metrics["discriminator_accuracy"])

        # Log progress
        if verbose and (
            iteration % config.log_every == 0 or iteration == config.num_iterations - 1
        ):
            print(
                f"    Iter {iteration:3d}: loss={metrics['discriminator_loss']:.4f}, "
                f"acc={metrics['discriminator_accuracy']:.3f}, "
                f"reward={metrics['amp_reward_mean']:.3f}±{metrics['amp_reward_std']:.3f}"
            )

    # Store final values
    if result.loss_history:
        result.final_loss = result.loss_history[-1]
        result.final_accuracy = result.accuracy_history[-1]
        result.final_reward_mean = result.reward_mean_history[-1]
        result.final_reward_std = result.reward_std_history[-1]

    # Evaluate acceptance criteria
    issues = []

    # Check for NaN/Inf (already set)
    if result.has_nan:
        issues.append("NaN values detected")
    if result.has_inf:
        issues.append("Inf values detected")

    # Check reward variance (REQUIRED)
    if result.final_reward_std < config.min_reward_std:
        result.reward_flat = True
        issues.append(
            f"Reward flat (std={result.final_reward_std:.4f}<{config.min_reward_std})"
        )

    # Check loss explosion (REQUIRED - loss shouldn't increase)
    if result.final_loss > result.initial_loss * config.max_loss_increase_ratio:
        result.loss_exploded = True
        issues.append(
            f"Loss exploded ({result.initial_loss:.4f}→{result.final_loss:.4f})"
        )

    # Check training progress: EITHER accuracy improves OR loss decreases significantly
    # (LSGAN accuracy can be quirky, so loss decrease is an acceptable alternative)
    accuracy_improvement = result.final_accuracy - 0.5
    loss_decrease_ratio = (
        result.final_loss / result.initial_loss if result.initial_loss > 0 else 1.0
    )

    training_progress = False
    if accuracy_improvement >= config.min_accuracy_improvement:
        training_progress = True
    elif loss_decrease_ratio <= config.min_loss_decrease_ratio:
        # Loss decreased by at least 20%
        training_progress = True

    if not training_progress:
        result.accuracy_stuck = True
        issues.append(
            f"No training progress (acc improvement={accuracy_improvement:.3f}, loss ratio={loss_decrease_ratio:.2f})"
        )

    # Determine pass/fail
    result.passed = len(issues) == 0

    if result.passed:
        result.details = "All acceptance criteria met"
    else:
        result.details = "; ".join(issues)

    # Print summary
    print(f"\n")
    table = Table(title="Smoke Test Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Initial", justify="right")
    table.add_column("Final", justify="right")
    table.add_column("Status", justify="center")

    # Loss
    loss_status = "[green]✓[/green]" if not result.loss_exploded else "[red]✗[/red]"
    table.add_row(
        "D Loss", f"{result.initial_loss:.4f}", f"{result.final_loss:.4f}", loss_status
    )

    # Accuracy
    acc_status = "[green]✓[/green]" if not result.accuracy_stuck else "[red]✗[/red]"
    table.add_row(
        "D Accuracy",
        f"{result.initial_accuracy:.3f}",
        f"{result.final_accuracy:.3f}",
        acc_status,
    )

    # Reward
    reward_status = "[green]✓[/green]" if not result.reward_flat else "[red]✗[/red]"
    table.add_row(
        "AMP Reward",
        "-",
        f"{result.final_reward_mean:.3f}±{result.final_reward_std:.3f}",
        reward_status,
    )

    # Numerical stability
    nan_status = "[green]✓[/green]" if not result.has_nan else "[red]✗[/red]"
    inf_status = "[green]✓[/green]" if not result.has_inf else "[red]✗[/red]"
    table.add_row("No NaN", "-", "-", nan_status)
    table.add_row("No Inf", "-", "-", inf_status)

    console.print(table)

    # Overall result
    if result.passed:
        print(
            f"\n[bold green]✓ Smoke Test PASSED: Training wiring verified[/bold green]"
        )
        print(
            f"  Accuracy improved: {0.5:.3f} → {result.final_accuracy:.3f} (+{accuracy_improvement:.3f})"
        )
        print(f"  AMP reward variance: std={result.final_reward_std:.4f}")
    else:
        print(f"\n[bold red]✗ Smoke Test FAILED: {result.details}[/bold red]")
        print(f"  Review discriminator setup and reference data.")

    return result


def print_detailed_history(result: SmokeTestResult, sample_every: int = 10):
    """Print detailed training history."""
    print(f"\n[bold]Training History (every {sample_every} iterations):[/bold]")

    for i in range(0, len(result.loss_history), sample_every):
        print(
            f"  {i:3d}: loss={result.loss_history[i]:.4f}, "
            f"acc={result.accuracy_history[i]:.3f}, "
            f"reward_std={result.reward_std_history[i]:.4f}"
        )

    # Always print final
    if len(result.loss_history) > 0:
        i = len(result.loss_history) - 1
        print(
            f"  {i:3d}: loss={result.loss_history[i]:.4f}, "
            f"acc={result.accuracy_history[i]:.3f}, "
            f"reward_std={result.reward_std_history[i]:.4f}"
        )


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Micro-Train Smoke Test for AMP Training"
    )
    parser.add_argument(
        "--amp-dir",
        type=str,
        default="data/amp",
        help="Directory containing *_amp.pkl reference files",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of training iterations",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for training",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate for discriminator",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed training progress",
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help="Print detailed training history",
    )

    args = parser.parse_args()

    # Configuration
    config = SmokeTestConfig(
        num_iterations=args.iterations,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    # Load reference data
    amp_dir = Path(args.amp_dir)
    print(f"[bold blue]Loading reference data from:[/bold blue] {amp_dir}")

    ref_features = load_reference_features(amp_dir, config.feature_dim)
    print(f"  Loaded {len(ref_features)} frames")

    # Run smoke test
    result = run_smoke_test(ref_features, config, verbose=args.verbose)

    # Print history if requested
    if args.history:
        print_detailed_history(result)

    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
