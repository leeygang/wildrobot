#!/usr/bin/env python3
"""Monitor training progress and alert on issues.

Usage:
    python monitor_training.py <training_log_dir> [--watch] [--interval MINUTES]

Examples:
    # Single check
    python monitor_training.py training_logs/phase1_contact_flat_20251130-161234

    # Continuous monitoring (check every 30 minutes)
    python monitor_training.py training_logs/phase1_contact_flat_20251130-161234 --watch

    # Custom check interval (every 15 minutes)
    python monitor_training.py training_logs/phase1_contact_flat_20251130-161234 --watch --interval 15
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def kill_local_training() -> bool:
    """Kill the training process on the local machine.
    
    Returns:
        True if kill command was successful, False otherwise
    """
    print(f"\n🔪 FORCE STOP: Killing local training process...")
    
    # Command to kill training.py process (matches both "uv run train.py" and "python train.py")
    kill_cmd = ["pkill", "-f", "train.py.*--config"]
    
    try:
        result = subprocess.run(
            kill_cmd,
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            print(f"   ✅ Training process killed successfully")
            return True
        elif result.returncode == 1:
            # pkill returns 1 if no processes matched
            print(f"   ⚠️  No training process found (may have already stopped)")
            return True
        else:
            print(f"   ❌ Kill command failed (exit code {result.returncode})")
            if result.stderr:
                print(f"   Error: {result.stderr.strip()}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"   ❌ Kill command timed out after 5 seconds")
        return False
    except Exception as e:
        print(f"   ❌ Error executing kill command: {e}")
        return False


def monitor_training(log_dir: Path) -> dict:
    """Monitor training progress and check for issues.

    Args:
        log_dir: Path to training log directory

    Returns:
        dict with status info
    """
    metrics_file = log_dir / "metrics.jsonl"

    if not metrics_file.exists():
        print(f"❌ Metrics file not found: {metrics_file}")
        return {"status": "error", "message": "No metrics file"}

    # Read all metrics
    with open(metrics_file) as f:
        lines = f.readlines()
        if len(lines) < 2:
            print("⏳ Training just started, not enough data yet...")
            return {"status": "early", "evaluations": len(lines)}

        first = json.loads(lines[0])
        latest = json.loads(lines[-1])
        mid = json.loads(lines[len(lines)//2]) if len(lines) > 2 else first

    # Load config
    config_file = log_dir / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
        total_steps = config['training']['num_timesteps']
    else:
        total_steps = 20_000_000  # Default

    current_steps = latest['step']
    progress_pct = (current_steps / total_steps) * 100

    print("="*80)
    print(f"TRAINING MONITOR: {log_dir.name}")
    print("="*80)

    # Progress
    print(f"\n📊 PROGRESS:")
    print(f"  Steps:        {current_steps:>12,} / {total_steps:,} ({progress_pct:>5.1f}%)")
    print(f"  Evaluations:  {len(lines):>12,}")
    print(f"  Training time:{latest['other']['walltime']/3600:>12.1f} hours")
    print(f"  SPS:          {latest['summary']['sps']:>12.0f}")

    # ETA
    remaining_steps = total_steps - current_steps
    eta_seconds = remaining_steps / latest['summary']['sps'] if latest['summary']['sps'] > 0 else 0
    eta_hours = eta_seconds / 3600
    print(f"  ETA:          {eta_hours:>12.1f} hours")

    # Performance
    print(f"\n📈 PERFORMANCE:")
    reward_change = latest['summary']['reward_per_step'] - first['summary']['reward_per_step']
    vel_change = latest['summary']['forward_velocity'] - first['summary']['forward_velocity']
    success_change = latest['summary']['success_rate'] - first['summary']['success_rate']

    print(f"  Reward/step:  {first['summary']['reward_per_step']:>8.2f} → {latest['summary']['reward_per_step']:>8.2f} ({reward_change:>+7.2f})")
    print(f"  Velocity:     {first['summary']['forward_velocity']:>8.3f} → {latest['summary']['forward_velocity']:>8.3f} m/s ({vel_change:>+7.3f})")
    print(f"  Success rate: {first['summary']['success_rate']:>7.1f}% → {latest['summary']['success_rate']:>7.1f}% ({success_change:>+6.1f}%)")
    print(f"  Episode len:  {first['summary']['episode_length']:>8.1f} → {latest['summary']['episode_length']:>8.1f} steps")
    # Gating diagnostics (if present)
    gate_first = first['summary'].get('tracking_gate_active_rate')
    gate_latest = latest['summary'].get('tracking_gate_active_rate')
    scale_first = first['summary'].get('velocity_threshold_scale')
    scale_latest = latest['summary'].get('velocity_threshold_scale')
    if gate_latest is not None:
        print(f"  Gate active%: {gate_first:>8.2%} → {gate_latest:>8.2%}")
    if scale_latest is not None:
        print(f"  VelThreshScale:{scale_first:>8.2f} → {scale_latest:>8.2f}")

    # Current metrics
    print(f"\n🎯 CURRENT STATE:")
    print(f"  Reward:       {latest['summary']['reward_per_step']:>8.2f}")
    print(f"  Velocity:     {latest['summary']['forward_velocity']:>8.3f} m/s")
    print(f"  Success:      {latest['summary']['success_rate']:>7.1f}%")
    print(f"  Episode len:  {latest['summary']['episode_length']:>8.1f} steps")
    if gate_latest is not None:
        print(f"  Gate active%: {gate_latest:>8.2%}")
    if scale_latest is not None:
        print(f"  VelThreshScale:{scale_latest:>8.2f}")

    # Top penalties and rewards
    print(f"\n🔴 TOP PENALTIES:")
    penalties = sorted(
        [(k, v) for k, v in latest['penalties_weighted'].items() if k != 'total'],
        key=lambda x: x[1],
        reverse=True
    )[:3]
    for name, val in penalties:
        print(f"  {name:30s} -{val:>8.2f}")

    print(f"\n🟢 TOP REWARDS:")
    rewards = sorted(
        [(k, v) for k, v in latest['rewards_weighted'].items() if k != 'total'],
        key=lambda x: x[1],
        reverse=True
    )[:3]
    for name, val in rewards:
        print(f"  {name:30s} +{val:>8.2f}")

    # Special check for forward_velocity_bonus (Phase 1E fix)
    fwd_bonus = latest['rewards_weighted'].get('forward_velocity_bonus', None)
    if fwd_bonus is not None:
        print(f"\n⚡ FORWARD VELOCITY BONUS:")
        print(f"  forward_velocity_bonus       +{fwd_bonus:>8.2f}")
        if fwd_bonus > 0:
            print(f"  ✅ Robot getting forward motion bonus!")
        else:
            print(f"  ⚠️  Zero bonus (not moving forward or standing still)")

    # Special check for velocity_threshold_penalty (Phase 1G Option 3 fix)
    vel_penalty = latest['penalties_weighted'].get('velocity_threshold_penalty', None)
    if vel_penalty is not None:
        print(f"\n🚨 VELOCITY THRESHOLD PENALTY (Phase 1G):")
        print(f"  velocity_threshold_penalty   -{vel_penalty:>8.2f}")
        if vel_penalty > 5.0:
            print(f"  ⚠️  HIGH penalty - robot moving too slowly!")
            print(f"     (Penalty > 5.0 means velocity < 0.2 m/s)")
        elif vel_penalty > 0.1:
            print(f"  ⚠️  Moderate penalty - robot below threshold")
            print(f"     (Penalty active, velocity < 0.3 m/s)")
        else:
            print(f"  ✅ No penalty - robot moving fast enough!")
            print(f"     (Velocity >= 0.3 m/s threshold)")

    # Health checks
    print(f"\n🔍 HEALTH CHECKS:")
    issues = []
    warnings = []
    good_signs = []

    # Critical issues
    if latest['summary']['forward_velocity'] < 0:
        issues.append(f"❌ CRITICAL: Moving BACKWARD ({latest['summary']['forward_velocity']:.3f} m/s)")
    elif latest['summary']['forward_velocity'] < 0.1:
        warnings.append(f"⚠️  Velocity very low ({latest['summary']['forward_velocity']:.3f} m/s)")
    elif latest['summary']['forward_velocity'] > 0.3:
        good_signs.append(f"✅ Good forward velocity ({latest['summary']['forward_velocity']:.3f} m/s)")

    if reward_change < -2.0:
        issues.append(f"❌ CRITICAL: Reward degrading ({reward_change:+.2f})")
    elif reward_change < 0:
        warnings.append(f"⚠️  Reward declining slightly ({reward_change:+.2f})")
    elif reward_change > 1.0:
        good_signs.append(f"✅ Reward improving ({reward_change:+.2f})")

    if latest['summary']['success_rate'] < 40:
        issues.append(f"❌ Low success rate ({latest['summary']['success_rate']:.1f}%)")
    elif latest['summary']['success_rate'] > 60:
        good_signs.append(f"✅ Good success rate ({latest['summary']['success_rate']:.1f}%)")

    # Check if velocity is positive (key fix indicator)
    if latest['summary']['forward_velocity'] > 0:
        good_signs.append(f"✅ Robot moving FORWARD (fix working!)")

    # Check if reward is positive or improving
    if latest['summary']['reward_per_step'] > 0:
        good_signs.append(f"✅ Positive reward ({latest['summary']['reward_per_step']:.2f})")

    # Gating health: if gate still heavily active late in training, warn
    if gate_latest is not None and progress_pct > 25.0:
        if gate_latest > 0.50:
            warnings.append(f"⚠️  Gate active {gate_latest:.0%} (forward velocity below threshold often)")
        elif gate_latest < 0.10:
            good_signs.append(f"✅ Gate rarely active ({gate_latest:.0%})")

    # Velocity threshold decay scale: if still high late training, warn
    if scale_latest is not None and progress_pct > 50.0:
        if scale_latest > 0.50:
            warnings.append(f"⚠️  Velocity threshold penalty not decayed yet (scale={scale_latest:.2f})")
        elif scale_latest < 0.20:
            good_signs.append(f"✅ Velocity threshold penalty mostly decayed (scale={scale_latest:.2f})")

    # Print results
    if issues:
        for issue in issues:
            print(f"  {issue}")
    if warnings:
        for warning in warnings:
            print(f"  {warning}")
    if good_signs:
        for sign in good_signs:
            print(f"  {sign}")

    if not issues and not warnings and not good_signs:
        print(f"  ⏳ Early in training, trends not clear yet")

    # Recommendations
    print(f"\n💡 RECOMMENDATION:")
    if issues:
        print(f"  🚨 STOP TRAINING - Critical issues detected!")
        print(f"     Review reward configuration")
        status = "critical"
    elif progress_pct < 10:
        print(f"  ⏳ Early phase - continue monitoring")
        print(f"     Check again at 10% progress (~2M steps)")
        status = "early"
    elif progress_pct < 50:
        print(f"  ✅ Continue training")
        print(f"     Check again at 50% progress (~10M steps)")
        status = "good"
    else:
        print(f"  ✅ Training progressing well - continue to completion")
        status = "good"

    print("="*80)

    return {
        "status": status,
        "progress_pct": progress_pct,
        "current_steps": current_steps,
        "reward": latest['summary']['reward_per_step'],
        "velocity": latest['summary']['forward_velocity'],
        "success_rate": latest['summary']['success_rate'],
        "issues": issues,
        "warnings": warnings,
        "good_signs": good_signs,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Monitor training progress and alert on issues.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single check
  python monitor_training.py training_logs/phase1_contact_flat_20251130-161234

  # Continuous monitoring (check every 30 minutes)
  python monitor_training.py training_logs/phase1_contact_flat_20251130-161234 --watch

  # Custom check interval (every 15 minutes)
  python monitor_training.py training_logs/phase1_contact_flat_20251130-161234 --watch --interval 15

  # Auto-kill training on critical issues (runs locally)
  python monitor_training.py training_logs/phase1_contact_flat_20251130-161234 --watch --force_stop
        """
    )
    parser.add_argument("log_dir", type=str, help="Path to training log directory")
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Continuously monitor training (check every --interval minutes)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=15,
        help="Check interval in minutes (default: 15)",
    )
    parser.add_argument(
        "--force_stop",
        action="store_true",
        help="Automatically kill local training process when critical issues detected",
    )

    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"❌ Directory not found: {log_dir}")
        sys.exit(1)

    if args.watch:
        print(f"🔄 Continuous monitoring enabled")
        print(f"   Checking every {args.interval} minutes")
        print(f"   Press Ctrl+C to stop\n")

        check_count = 0
        try:
            while True:
                check_count += 1

                # Clear screen for readability (works on macOS/Linux)
                if check_count > 1:
                    os.system('clear' if os.name != 'nt' else 'cls')

                # Print timestamp
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"\n{'='*80}")
                print(f"CHECK #{check_count} at {now}")
                print(f"{'='*80}\n")

                # Monitor training
                result = monitor_training(log_dir)

                # Check if training is complete or has critical issues
                if result.get("progress_pct", 0) >= 99.9:
                    print(f"\n🎉 TRAINING COMPLETE!")
                    print(f"   Final check performed.")
                    break

                if result["status"] == "critical":
                    print(f"\n🚨 CRITICAL ISSUES DETECTED!")
                    
                    # Force stop if enabled
                    if args.force_stop:
                        print(f"   --force_stop enabled: Attempting to kill training...")
                        success = kill_local_training()
                        
                        if success:
                            print(f"\n   ✅ Training stopped successfully")
                            print(f"   Stopping monitor.")
                            sys.exit(1)
                        else:
                            print(f"\n   ❌ Failed to stop training automatically")
                            print(f"   Please manually run: pkill -f train.py")
                            print(f"   Stopping monitor anyway.")
                            sys.exit(1)
                    else:
                        print(f"   Stopping monitor. Please review the training.")
                        print(f"   Hint: Use --force_stop to automatically kill training on critical issues")
                        sys.exit(1)

                # Wait for next check
                print(f"\n⏳ Next check in {args.interval} minutes...")
                print(f"   (Press Ctrl+C to stop monitoring)")
                time.sleep(args.interval * 60)

        except KeyboardInterrupt:
            print(f"\n\n⚠️  Monitoring stopped by user (Ctrl+C)")
            print(f"   Total checks performed: {check_count}")
            sys.exit(0)
    else:
        # Single check mode
        result = monitor_training(log_dir)

        # Exit code based on status
        if result["status"] == "critical":
            sys.exit(1)
        else:
            sys.exit(0)


if __name__ == "__main__":
    main()
