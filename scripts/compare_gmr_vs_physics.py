#!/usr/bin/env python3
"""GMR vs Physics Reference Training Comparison.

This script compares training performance between:
1. GMR retargeted reference data (v0.6.x baseline)
2. Physics-generated reference data (v0.7.0)

It runs quick comparison experiments and generates a report with key metrics.

Key metrics compared:
- disc_acc: Discriminator accuracy (target: 0.55-0.75)
- D(real) / D(fake): Discriminator output distribution
- amp_reward: Style reward from discriminator
- forward_velocity: How well robot tracks commanded speed
- success_rate: Episode completion rate (not falling)
- episode_length: Average steps before termination

Usage:
    cd ~/projects/wildrobot

    # Run quick comparison (100 iterations each)
    uv run python scripts/compare_gmr_vs_physics.py --quick

    # Run full comparison (500 iterations each)
    uv run python scripts/compare_gmr_vs_physics.py --iterations 500

    # Only run physics (GMR already done)
    uv run python scripts/compare_gmr_vs_physics.py --physics-only

    # Generate report from existing logs
    uv run python scripts/compare_gmr_vs_physics.py --report-only
"""

import argparse
import json
import os
import pickle
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from rich import print
from rich.console import Console
from rich.table import Table

console = Console()


@dataclass
class ExperimentConfig:
    """Configuration for a comparison experiment."""

    name: str
    dataset_path: str
    version: str
    iterations: int = 100
    num_envs: int = 256  # Reduced for quick experiments
    rollout_steps: int = 64
    seed: int = 42


@dataclass
class ExperimentResults:
    """Results from a training experiment."""

    name: str
    version: str
    dataset_path: str
    iterations: int
    total_steps: int
    training_time_sec: float

    # Final metrics (last 10 iterations average)
    final_disc_acc: float = 0.0
    final_disc_real_mean: float = 0.0
    final_disc_fake_mean: float = 0.0
    final_amp_reward: float = 0.0
    final_forward_velocity: float = 0.0
    final_success_rate: float = 0.0
    final_episode_length: float = 0.0
    final_total_reward: float = 0.0

    # Trajectory (per-iteration)
    disc_acc_history: List[float] = field(default_factory=list)
    amp_reward_history: List[float] = field(default_factory=list)
    velocity_history: List[float] = field(default_factory=list)
    success_history: List[float] = field(default_factory=list)


def create_experiment_config(
    base_config_path: str,
    experiment: ExperimentConfig,
    output_dir: Path,
) -> Path:
    """Create a modified config file for the experiment."""
    import yaml

    with open(base_config_path, "r") as f:
        config = yaml.safe_load(f)

    # Modify for experiment
    config["version"] = experiment.version
    config["version_name"] = f"Comparison: {experiment.name}"

    # Dataset
    config["amp"]["dataset_path"] = experiment.dataset_path

    # Training params (reduced for quick experiments)
    config["trainer"]["iterations"] = experiment.iterations
    config["trainer"]["num_envs"] = experiment.num_envs
    config["trainer"]["rollout_steps"] = experiment.rollout_steps
    config["trainer"]["seed"] = experiment.seed

    # Disable W&B for comparison runs (use local logging)
    config["wandb"]["enabled"] = False

    # Checkpoint settings
    config["checkpoints"]["interval"] = max(10, experiment.iterations // 10)
    config["trainer"]["checkpoint_dir"] = str(output_dir / "checkpoints")

    # Save modified config
    config_path = (
        output_dir / f"config_{experiment.name.lower().replace(' ', '_')}.yaml"
    )
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    return config_path


def run_experiment(
    config_path: Path,
    experiment: ExperimentConfig,
    output_dir: Path,
) -> Optional[ExperimentResults]:
    """Run a training experiment and collect results."""
    import time

    print(f"\n[bold blue]Running experiment: {experiment.name}[/bold blue]")
    print(f"  Dataset: {experiment.dataset_path}")
    print(f"  Iterations: {experiment.iterations}")
    print(f"  Config: {config_path}")

    # Create log file
    log_path = output_dir / f"log_{experiment.name.lower().replace(' ', '_')}.txt"

    start_time = time.time()

    try:
        # Run training
        cmd = [
            "uv",
            "run",
            "python",
            "playground_amp/train.py",
            "--config",
            str(config_path),
        ]

        with open(log_path, "w") as log_file:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                cwd=Path(__file__).parent.parent,
            )

            # Stream output and collect metrics
            disc_acc_history = []
            amp_reward_history = []
            velocity_history = []
            success_history = []

            for line in process.stdout:
                log_file.write(line)
                log_file.flush()

                # Parse metrics from output
                if "disc_acc=" in line:
                    try:
                        # Extract metrics from trainer output
                        # Format: #iter | ... | disc_acc=X.XX | amp=X.XX | ...
                        parts = line.split("|")
                        for part in parts:
                            part = part.strip()
                            if part.startswith("disc_acc="):
                                val = float(part.split("=")[1])
                                disc_acc_history.append(val)
                            elif part.startswith("amp="):
                                val = float(part.split("=")[1])
                                amp_reward_history.append(val)
                            elif part.startswith("vel="):
                                val = float(part.split("=")[1].split("m")[0])
                                velocity_history.append(val)
                            elif part.startswith("success="):
                                val = float(part.split("=")[1].rstrip("%")) / 100
                                success_history.append(val)
                    except (ValueError, IndexError):
                        pass

                # Print progress
                if "#" in line and "disc_acc" in line:
                    print(f"  {line.strip()}")

            process.wait()

        training_time = time.time() - start_time

        if process.returncode != 0:
            print(
                f"[bold red]✗ Experiment failed (exit code {process.returncode})[/bold red]"
            )
            return None

        # Compute final metrics (average of last 10 iterations)
        def safe_mean(arr, n=10):
            if not arr:
                return 0.0
            return np.mean(arr[-n:])

        results = ExperimentResults(
            name=experiment.name,
            version=experiment.version,
            dataset_path=experiment.dataset_path,
            iterations=experiment.iterations,
            total_steps=experiment.iterations
            * experiment.num_envs
            * experiment.rollout_steps,
            training_time_sec=training_time,
            final_disc_acc=safe_mean(disc_acc_history),
            final_amp_reward=safe_mean(amp_reward_history),
            final_forward_velocity=safe_mean(velocity_history),
            final_success_rate=safe_mean(success_history),
            disc_acc_history=disc_acc_history,
            amp_reward_history=amp_reward_history,
            velocity_history=velocity_history,
            success_history=success_history,
        )

        print(
            f"[bold green]✓ Experiment completed in {training_time/60:.1f} min[/bold green]"
        )

        return results

    except Exception as e:
        print(f"[bold red]✗ Experiment error: {e}[/bold red]")
        return None


def generate_comparison_report(
    results: List[ExperimentResults],
    output_dir: Path,
):
    """Generate comparison report from experiment results."""

    print("\n" + "=" * 70)
    print("[bold]GMR vs Physics Reference Training Comparison Report[/bold]")
    print("=" * 70)

    # Summary table
    table = Table(title="Final Metrics Comparison")

    table.add_column("Metric", style="cyan")
    for r in results:
        table.add_column(r.name, style="green" if "Physics" in r.name else "yellow")

    metrics = [
        ("Discriminator Accuracy", "final_disc_acc", "{:.2f}"),
        ("AMP Reward", "final_amp_reward", "{:.3f}"),
        ("Forward Velocity (m/s)", "final_forward_velocity", "{:.2f}"),
        ("Success Rate", "final_success_rate", "{:.1%}"),
        ("Training Time (min)", "training_time_sec", lambda x: f"{x/60:.1f}"),
        ("Total Steps", "total_steps", "{:,}"),
    ]

    for metric_name, attr, fmt in metrics:
        row = [metric_name]
        for r in results:
            val = getattr(r, attr)
            if callable(fmt):
                row.append(fmt(val))
            else:
                row.append(fmt.format(val))
        table.add_row(*row)

    console.print(table)

    # Analysis
    print("\n[bold blue]Analysis[/bold blue]")

    if len(results) >= 2:
        gmr = next((r for r in results if "GMR" in r.name), None)
        physics = next((r for r in results if "Physics" in r.name), None)

        if gmr and physics:
            # Compare disc_acc
            print("\n[bold]Discriminator Accuracy:[/bold]")
            print(f"  GMR:     {gmr.final_disc_acc:.2f}")
            print(f"  Physics: {physics.final_disc_acc:.2f}")

            if 0.55 <= physics.final_disc_acc <= 0.75:
                print(
                    "  [green]✓ Physics disc_acc in healthy range (0.55-0.75)[/green]"
                )
            elif physics.final_disc_acc < 0.55:
                print(
                    "  [yellow]⚠ Physics disc_acc too low - discriminator may be collapsed[/yellow]"
                )
            else:
                print(
                    "  [yellow]⚠ Physics disc_acc too high - discriminator may be overpowering[/yellow]"
                )

            # Compare success rate
            print("\n[bold]Success Rate:[/bold]")
            print(f"  GMR:     {gmr.final_success_rate:.1%}")
            print(f"  Physics: {physics.final_success_rate:.1%}")

            if physics.final_success_rate > gmr.final_success_rate:
                improvement = (
                    physics.final_success_rate - gmr.final_success_rate
                ) * 100
                print(
                    f"  [green]✓ Physics improves success by {improvement:.1f}%[/green]"
                )
            elif physics.final_success_rate < gmr.final_success_rate:
                regression = (gmr.final_success_rate - physics.final_success_rate) * 100
                print(f"  [red]✗ Physics regresses success by {regression:.1f}%[/red]")

            # Compare velocity tracking
            print("\n[bold]Velocity Tracking:[/bold]")
            print(f"  GMR:     {gmr.final_forward_velocity:.2f} m/s")
            print(f"  Physics: {physics.final_forward_velocity:.2f} m/s")
            print(f"  Target:  0.5-1.0 m/s")

    # Recommendations
    print("\n[bold blue]Recommendations[/bold blue]")

    if len(results) >= 2 and physics:
        if physics.final_disc_acc < 0.55:
            print("  • Increase disc_lr or update_steps to strengthen discriminator")
        elif physics.final_disc_acc > 0.75:
            print(
                "  • Decrease disc_lr or increase replay_buffer_ratio to weaken discriminator"
            )

        if physics.final_success_rate < 0.1:
            print("  • Robot falling - check physics reference data quality")
            print("  • Consider increasing healthy reward weight")

        if physics.final_amp_reward < 0.1:
            print("  • AMP reward very low - check feature parity")
            print("  • Run: uv run python scripts/verify_physics_ref_parity.py")

    # Save report
    report_path = output_dir / "comparison_report.json"
    report_data = {
        "timestamp": datetime.now().isoformat(),
        "results": [
            {
                "name": r.name,
                "version": r.version,
                "dataset_path": r.dataset_path,
                "iterations": r.iterations,
                "total_steps": r.total_steps,
                "training_time_sec": r.training_time_sec,
                "final_disc_acc": r.final_disc_acc,
                "final_amp_reward": r.final_amp_reward,
                "final_forward_velocity": r.final_forward_velocity,
                "final_success_rate": r.final_success_rate,
            }
            for r in results
        ],
    }

    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2)

    print(f"\n[dim]Report saved to: {report_path}[/dim]")


def main():
    parser = argparse.ArgumentParser(
        description="GMR vs Physics Reference Training Comparison"
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of training iterations per experiment (default: 100)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick comparison (50 iterations, 128 envs)",
    )
    parser.add_argument(
        "--gmr-only",
        action="store_true",
        help="Only run GMR experiment",
    )
    parser.add_argument(
        "--physics-only",
        action="store_true",
        help="Only run Physics experiment",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Only generate report from existing results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="playground_amp/comparison_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--base-config",
        type=str,
        default="playground_amp/configs/ppo_amass_training.yaml",
        help="Base training config to modify",
    )

    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Quick mode settings
    if args.quick:
        iterations = 50
        num_envs = 128
        rollout_steps = 32
    else:
        iterations = args.iterations
        num_envs = 256
        rollout_steps = 64

    # Define experiments
    experiments = []

    if not args.physics_only:
        experiments.append(
            ExperimentConfig(
                name="GMR Baseline",
                dataset_path="playground_amp/data/walking_motions_normalized_vel.pkl",
                version="0.6.6",
                iterations=iterations,
                num_envs=num_envs,
                rollout_steps=rollout_steps,
                seed=42,
            )
        )

    if not args.gmr_only:
        experiments.append(
            ExperimentConfig(
                name="Physics Reference",
                dataset_path="playground_amp/data/physics_ref/walking_physics_merged.pkl",
                version="0.7.0",
                iterations=iterations,
                num_envs=num_envs,
                rollout_steps=rollout_steps,
                seed=42,
            )
        )

    if args.report_only:
        # Load existing results
        print("[bold blue]Loading existing results...[/bold blue]")
        report_path = output_dir / "comparison_report.json"
        if report_path.exists():
            with open(report_path, "r") as f:
                report_data = json.load(f)

            results = [ExperimentResults(**r) for r in report_data["results"]]
            generate_comparison_report(results, output_dir)
        else:
            print(f"[red]No existing results found at {report_path}[/red]")
        return 0

    print("[bold]GMR vs Physics Reference Training Comparison[/bold]")
    print(f"  Output: {output_dir}")
    print(f"  Iterations: {iterations}")
    print(f"  Envs: {num_envs}")
    print(f"  Steps/env: {rollout_steps}")
    print(f"  Total steps/experiment: {iterations * num_envs * rollout_steps:,}")

    # Check datasets exist
    for exp in experiments:
        dataset_path = Path(exp.dataset_path)
        if not dataset_path.exists():
            print(f"[red]✗ Dataset not found: {dataset_path}[/red]")
            return 1
        print(f"  [green]✓ Dataset exists: {exp.name}[/green]")

    # Run experiments
    results = []

    for experiment in experiments:
        # Create experiment config
        exp_output_dir = output_dir / experiment.name.lower().replace(" ", "_")
        exp_output_dir.mkdir(parents=True, exist_ok=True)

        config_path = create_experiment_config(
            args.base_config, experiment, exp_output_dir
        )

        # Run experiment
        result = run_experiment(config_path, experiment, exp_output_dir)

        if result:
            results.append(result)

            # Save individual result
            result_path = exp_output_dir / "results.pkl"
            with open(result_path, "wb") as f:
                pickle.dump(result, f)

    # Generate comparison report
    if results:
        generate_comparison_report(results, output_dir)
    else:
        print("[red]No experiments completed successfully[/red]")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
