#!/usr/bin/env python3
"""Check training progress for WildRobot experiments.

Usage:
    python check_training.py                              # List all experiments
    python check_training.py <experiment_name>            # Show detailed progress
    python check_training.py <experiment_name> --plot     # Show progress with plots
    python check_training.py --compare exp1 exp2          # Compare experiments
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional


def load_metrics(exp_dir: Path) -> List[dict]:
    """Load metrics from experiment directory."""
    metrics_file = exp_dir / "metrics.jsonl"
    if not metrics_file.exists():
        return []

    metrics = []
    with open(metrics_file, 'r') as f:
        for line in f:
            metrics.append(json.loads(line))
    return metrics


def load_config(exp_dir: Path) -> dict:
    """Load config from experiment directory."""
    config_file = exp_dir / "config.json"
    if not config_file.exists():
        return {}

    with open(config_file, 'r') as f:
        return json.load(f)


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time."""
    hours = seconds / 3600
    if hours < 1:
        return f"{seconds/60:.1f}m"
    elif hours < 24:
        return f"{hours:.1f}h"
    else:
        return f"{hours/24:.1f}d"


def list_experiments(training_dir: Path):
    """List all training experiments."""
    if not training_dir.exists():
        print(f"Training directory not found: {training_dir}")
        return

    experiments = sorted(training_dir.glob("*/"), key=lambda x: x.stat().st_mtime, reverse=True)

    if not experiments:
        print(f"No experiments found in {training_dir}")
        return

    print("=" * 100)
    print("AVAILABLE TRAINING EXPERIMENTS")
    print("=" * 100)
    print(f"\n{'Name':<50} {'Steps':<15} {'Success %':<12} {'Status':<10}")
    print("-" * 100)

    for exp_dir in experiments:
        metrics = load_metrics(exp_dir)
        if not metrics:
            status = "NO DATA"
            steps = "-"
            success = "-"
        else:
            latest = metrics[-1]
            steps = f"{latest['step']:,}"
            success = f"{latest['summary']['success_rate']:.1f}%"

            # Check if still running (updated in last 5 minutes)
            import time
            last_update = latest['timestamp']
            time_since_update = time.time() - last_update
            if time_since_update < 300:  # 5 minutes
                status = "RUNNING"
            else:
                status = "STOPPED"

        print(f"{exp_dir.name:<50} {steps:<15} {success:<12} {status:<10}")

    print("\n" + "=" * 100)
    print(f"\nTotal experiments: {len(experiments)}")
    print("\nTo see detailed progress: python check_training.py <experiment_name>")
    print("=" * 100)


def show_progress(exp_name: str, training_dir: Path, show_plot: bool = False):
    """Show detailed progress for an experiment."""
    exp_dir = training_dir / exp_name

    if not exp_dir.exists():
        print(f"Experiment not found: {exp_name}")
        print(f"\nSearching in: {training_dir}")
        # Try to find similar names
        similar = [d.name for d in training_dir.glob("*") if exp_name.lower() in d.name.lower()]
        if similar:
            print(f"\nDid you mean one of these?")
            for name in similar:
                print(f"  - {name}")
        return

    # Load data
    metrics = load_metrics(exp_dir)
    config = load_config(exp_dir)

    if not metrics:
        print(f"No metrics found for {exp_name}")
        return

    first = metrics[0]
    latest = metrics[-1]

    # Calculate progress
    total_steps = config.get("training", {}).get("num_timesteps", 20_000_000)
    current_steps = latest["step"]
    progress_pct = (current_steps / total_steps) * 100

    # Time calculations
    walltime_sec = latest["other"]["walltime"]
    walltime_hours = walltime_sec / 3600
    sps = latest["summary"]["sps"]
    remaining_steps = total_steps - current_steps
    eta_sec = remaining_steps / sps if sps > 0 else 0
    eta_hours = eta_sec / 3600

    # Check if running
    import time
    time_since_update = time.time() - latest['timestamp']
    is_running = time_since_update < 300  # 5 minutes

    # Improvements
    reward_improvement = latest["summary"]["reward_per_step"] - first["summary"]["reward_per_step"]
    success_improvement = latest["summary"]["success_rate"] - first["summary"]["success_rate"]
    velocity_improvement = latest["summary"]["forward_velocity"] - first["summary"]["forward_velocity"]

    # Print report
    print("=" * 100)
    print(f"TRAINING PROGRESS REPORT: {exp_name}")
    print("=" * 100)
    print(f"\nStatus: {'RUNNING ✓' if is_running else 'STOPPED ✗'}")
    print(f"Last update: {format_time(time_since_update)} ago")

    print(f"\n{'='*100}")
    print("PROGRESS")
    print(f"{'='*100}")
    print(f"  Current step:     {current_steps:,} / {total_steps:,}")
    print(f"  Completion:       {progress_pct:.2f}%")
    filled = int(progress_pct / 2)
    empty = 50 - filled
    print(f"  Progress:         [{'#' * filled}{'.' * empty}]")
    print(f"  Walltime:         {format_time(walltime_sec)} ({walltime_hours:.2f} hours)")
    print(f"  Steps/sec:        {sps:.1f}")
    print(f"  ETA:              {format_time(eta_sec)} ({eta_hours:.2f} hours)")
    print(f"  Est. total time:  {format_time(walltime_sec + eta_sec)} ({(walltime_hours + eta_hours)/24:.1f} days)")

    print(f"\n{'='*100}")
    print(f"CURRENT PERFORMANCE (Step {current_steps:,})")
    print(f"{'='*100}")
    print(f"  Reward/step:      {latest['summary']['reward_per_step']:.2f}")
    print(f"  Success rate:     {latest['summary']['success_rate']:.1f}%")
    print(f"  Forward velocity: {latest['summary']['forward_velocity']:.3f} m/s")
    print(f"  Episode length:   {latest['summary']['episode_length']:.1f} steps")
    print(f"  Distance walked:  {latest['other']['distance_walked']:.2f} m")
    print(f"  Height:           {latest['other']['height']:.3f} m")

    print(f"\n{'='*100}")
    print(f"IMPROVEMENT (vs Start)")
    print(f"{'='*100}")
    print(f"  Reward/step:      {reward_improvement:+.2f}  ({first['summary']['reward_per_step']:.2f} → {latest['summary']['reward_per_step']:.2f})")
    print(f"  Success rate:     {success_improvement:+.1f}%  ({first['summary']['success_rate']:.1f}% → {latest['summary']['success_rate']:.1f}%)")
    print(f"  Forward velocity: {velocity_improvement:+.3f} m/s  ({first['summary']['forward_velocity']:.3f} → {latest['summary']['forward_velocity']:.3f})")

    print(f"\n{'='*100}")
    print("REWARD BREAKDOWN (Weighted - Actual Training Values)")
    print(f"{'='*100}")

    # Use weighted values if available, otherwise fall back to unweighted
    rewards_weighted = latest.get('rewards_weighted', latest.get('rewards', {}))
    penalties_weighted = latest.get('penalties_weighted', latest.get('penalties', {}))

    total_rewards_weighted = rewards_weighted.get('total', latest['summary'].get('total_rewards', 0))
    total_penalties_weighted = penalties_weighted.get('total', latest['summary'].get('total_penalties', 0))

    print(f"  Total rewards:    +{total_rewards_weighted:.3f}")
    print(f"  Total penalties:  -{total_penalties_weighted:.3f}")
    print(f"  Net per step:     {latest['summary']['reward_per_step']:.3f}")

    print(f"\n  Top Rewards (weighted):")
    rewards = [(k, v) for k, v in rewards_weighted.items() if k != 'total']
    rewards.sort(key=lambda x: x[1], reverse=True)
    for name, val in rewards[:5]:
        print(f"    + {name:30s} {val:>8.3f}")

    print(f"\n  Top Penalties (weighted):")
    penalties = [(k, v) for k, v in penalties_weighted.items() if k != 'total']
    penalties.sort(key=lambda x: x[1], reverse=True)
    for name, val in penalties[:5]:
        print(f"    - {name:30s} {val:>8.3f}")

    # Show unweighted section for debugging if available
    if 'rewards' in latest and latest.get('rewards') != rewards_weighted:
        print(f"\n{'='*100}")
        print("UNWEIGHTED VALUES (for debugging - NOT used in training)")
        print(f"{'='*100}")
        print(f"  Top unweighted penalties:")
        unweighted_penalties = [(k, v) for k, v in latest['penalties'].items() if k != 'total']
        unweighted_penalties.sort(key=lambda x: x[1], reverse=True)
        for name, val in unweighted_penalties[:3]:
            # Show comparison with weighted value
            weighted_val = penalties_weighted.get(name, 0)
            print(f"    {name:30s} unweighted: {val:>10.3f} → weighted: {weighted_val:>8.3f}")

    print(f"\n{'='*100}")
    print("CONTACT METRICS")
    print(f"{'='*100}")
    print(f"  Avg air time:           {latest['contact']['avg_air_time']:.3f} s")
    print(f"  Left foot in contact:   {latest['contact']['left_in_contact']*100:.1f}%")
    print(f"  Right foot in contact:  {latest['contact']['right_in_contact']*100:.1f}%")
    print(f"  Both feet contact:      {latest['contact']['both_feet_contact']*100:.1f}%")
    print(f"  No feet contact:        {latest['contact']['no_feet_contact']*100:.1f}%")
    print(f"  Left foot force:        {latest['contact']['left_foot_force']:.2f} N")
    print(f"  Right foot force:       {latest['contact']['right_foot_force']:.2f} N")
    print(f"  Gait phase:             {latest['contact']['gait_phase']:.3f}")

    # Plot if requested
    if show_plot:
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            # Extract data for plotting
            steps = [m['step'] for m in metrics]
            rewards = [m['summary']['reward_per_step'] for m in metrics]
            success_rates = [m['summary']['success_rate'] for m in metrics]
            velocities = [m['summary']['forward_velocity'] for m in metrics]

            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Training Progress: {exp_name}', fontsize=16)

            # Reward per step
            axes[0, 0].plot(steps, rewards, 'b-', linewidth=2)
            axes[0, 0].set_xlabel('Steps')
            axes[0, 0].set_ylabel('Reward per Step')
            axes[0, 0].set_title('Reward per Step')
            axes[0, 0].grid(True, alpha=0.3)

            # Success rate
            axes[0, 1].plot(steps, success_rates, 'g-', linewidth=2)
            axes[0, 1].set_xlabel('Steps')
            axes[0, 1].set_ylabel('Success Rate (%)')
            axes[0, 1].set_title('Success Rate')
            axes[0, 1].grid(True, alpha=0.3)

            # Forward velocity
            axes[1, 0].plot(steps, velocities, 'r-', linewidth=2)
            axes[1, 0].set_xlabel('Steps')
            axes[1, 0].set_ylabel('Forward Velocity (m/s)')
            axes[1, 0].set_title('Forward Velocity')
            axes[1, 0].grid(True, alpha=0.3)

            # Episode length
            episode_lengths = [m['summary']['episode_length'] for m in metrics]
            axes[1, 1].plot(steps, episode_lengths, 'm-', linewidth=2)
            axes[1, 1].set_xlabel('Steps')
            axes[1, 1].set_ylabel('Episode Length')
            axes[1, 1].set_title('Episode Length')
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

        except ImportError:
            print("\n⚠️  matplotlib not installed. Install with: pip install matplotlib")
        except Exception as e:
            print(f"\n⚠️  Error creating plots: {e}")

    print("\n" + "=" * 100)


def compare_experiments(exp_names: List[str], training_dir: Path):
    """Compare multiple experiments."""
    print("=" * 100)
    print(f"COMPARING EXPERIMENTS: {', '.join(exp_names)}")
    print("=" * 100)

    all_metrics = {}
    all_configs = {}

    for exp_name in exp_names:
        exp_dir = training_dir / exp_name
        if not exp_dir.exists():
            print(f"\n⚠️  Experiment not found: {exp_name}")
            continue

        metrics = load_metrics(exp_dir)
        config = load_config(exp_dir)

        if not metrics:
            print(f"\n⚠️  No metrics for: {exp_name}")
            continue

        all_metrics[exp_name] = metrics
        all_configs[exp_name] = config

    if len(all_metrics) < 2:
        print("\n⚠️  Need at least 2 valid experiments to compare")
        return

    # Print comparison table
    print(f"\n{'Metric':<30}", end='')
    for exp_name in all_metrics.keys():
        print(f"{exp_name[:20]:>22}", end='')
    print()
    print("-" * 100)

    # Compare latest metrics
    metrics_to_compare = [
        ('Current Step', lambda m: f"{m[-1]['step']:,}"),
        ('Success Rate %', lambda m: f"{m[-1]['summary']['success_rate']:.1f}"),
        ('Reward/Step', lambda m: f"{m[-1]['summary']['reward_per_step']:.2f}"),
        ('Forward Vel (m/s)', lambda m: f"{m[-1]['summary']['forward_velocity']:.3f}"),
        ('Episode Length', lambda m: f"{m[-1]['summary']['episode_length']:.1f}"),
        ('Steps/Sec', lambda m: f"{m[-1]['summary']['sps']:.1f}"),
    ]

    for metric_name, extractor in metrics_to_compare:
        print(f"{metric_name:<30}", end='')
        for metrics in all_metrics.values():
            try:
                value = extractor(metrics)
                print(f"{value:>22}", end='')
            except Exception:
                print(f"{'N/A':>22}", end='')
        print()

    print("\n" + "=" * 100)


def main():
    parser = argparse.ArgumentParser(
        description="Check WildRobot training progress",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python check_training.py                                      # List all experiments
  python check_training.py phase1_contact_flat_20251129-140506  # Show detailed progress
  python check_training.py phase1_contact_flat_20251129-140506 --plot  # Show with plots
  python check_training.py --compare exp1 exp2                  # Compare experiments
        """
    )

    parser.add_argument('experiment', nargs='?', help='Experiment name to check')
    parser.add_argument('--plot', action='store_true', help='Show plots (requires matplotlib)')
    parser.add_argument('--compare', nargs='+', help='Compare multiple experiments')
    parser.add_argument('--dir', default=None, help='Training logs directory (default: ../amp/training_logs)')

    args = parser.parse_args()

    # Determine training directory
    if args.dir:
        training_dir = Path(args.dir)
    else:
        # Assume script is in common/, training_logs is in amp/
        script_dir = Path(__file__).parent
        training_dir = script_dir.parent / "amp" / "training_logs"

    # Handle different modes
    if args.compare:
        compare_experiments(args.compare, training_dir)
    elif args.experiment:
        show_progress(args.experiment, training_dir, args.plot)
    else:
        list_experiments(training_dir)


if __name__ == "__main__":
    main()
