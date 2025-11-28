#!/usr/bin/env python3
"""Verify quick test results and decide if ready for full training.

Usage:
    # Parse from console output
    python verify_quick_test.py

    # Or provide output file
    python verify_quick_test.py output.txt

    # Or check latest training_logs
    python verify_quick_test.py --check-logs
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple


# Decision thresholds
THRESHOLDS = {
    "reward": {
        "excellent": -300,
        "good": -500,
        "marginal": -800,
        "same_as_before": -1000,
    },
    "success_rate": {
        "excellent": 0.50,
        "good": 0.30,
        "marginal": 0.20,
    },
    "forward_velocity": {
        "good": 0.3,
        "marginal": 0.1,
    },
    "height": {
        "min": 0.35,
        "target_min": 0.40,
        "target_max": 0.45,
        "max": 0.50,
    },
}


def parse_console_line(line: str) -> Optional[Dict[str, float]]:
    """Parse metrics from a training console line.

    Example line:
    Step 10,000/10,000 (100.0%) | Reward: -320.15 | Vel: 0.520 m/s |
    Height: 0.428m | Success: 52.00% | SPS: 2500 | ETA: 0:00:00
    """
    # Pattern to match the metrics
    pattern = r"Step\s+([\d,]+)/([\d,]+).*?Reward:\s+([-\d.]+).*?Vel:\s+([\d.]+)\s+m/s.*?Height:\s+([\d.]+)m.*?Success:\s+([\d.]+)%"

    match = re.search(pattern, line)
    if match:
        steps_current = int(match.group(1).replace(",", ""))
        steps_total = int(match.group(2).replace(",", ""))
        reward = float(match.group(3))
        velocity = float(match.group(4))
        height = float(match.group(5))
        success = float(match.group(6)) / 100.0  # Convert percentage to decimal

        return {
            "steps_current": steps_current,
            "steps_total": steps_total,
            "reward": reward,
            "velocity": velocity,
            "height": height,
            "success_rate": success,
        }
    return None


def parse_output(text: str) -> Optional[Dict[str, float]]:
    """Parse training output and extract final metrics."""
    lines = text.strip().split('\n')

    # Find the last line with metrics
    for line in reversed(lines):
        metrics = parse_console_line(line)
        if metrics and metrics["steps_current"] == metrics["steps_total"]:
            return metrics

    return None


def check_metrics(metrics: Dict[str, float]) -> Tuple[str, str, Dict[str, str]]:
    """Check metrics against thresholds.

    Returns:
        (decision, explanation, details)
        decision: "GO", "NO-GO", or "INVESTIGATE"
    """
    reward = metrics["reward"]
    success = metrics["success_rate"]
    velocity = metrics["velocity"]
    height = metrics["height"]

    details = {}
    issues = []
    good_signs = []

    # Check reward
    if reward > THRESHOLDS["reward"]["excellent"]:
        details["reward"] = "✅ EXCELLENT"
        good_signs.append(f"Reward {reward:.1f} is excellent (>{THRESHOLDS['reward']['excellent']})")
    elif reward > THRESHOLDS["reward"]["good"]:
        details["reward"] = "✅ GOOD"
        good_signs.append(f"Reward {reward:.1f} is good (improved from -800!)")
    elif reward > THRESHOLDS["reward"]["marginal"]:
        details["reward"] = "⚠️ MARGINAL"
        issues.append(f"Reward {reward:.1f} is marginal (target: >{THRESHOLDS['reward']['good']})")
    elif reward > THRESHOLDS["reward"]["same_as_before"]:
        details["reward"] = "⚠️ SAME AS BEFORE"
        issues.append(f"Reward {reward:.1f} is same as previous run (no improvement)")
    else:
        details["reward"] = "❌ WORSE"
        issues.append(f"Reward {reward:.1f} is WORSE than before (<-1000)")

    # Check success rate
    if success > THRESHOLDS["success_rate"]["excellent"]:
        details["success_rate"] = "✅ EXCELLENT"
        good_signs.append(f"Success rate {success:.1%} is excellent")
    elif success > THRESHOLDS["success_rate"]["good"]:
        details["success_rate"] = "✅ GOOD"
        good_signs.append(f"Success rate {success:.1%} is good")
    elif success > THRESHOLDS["success_rate"]["marginal"]:
        details["success_rate"] = "⚠️ MARGINAL"
        issues.append(f"Success rate {success:.1%} is marginal (target: >{THRESHOLDS['success_rate']['good']:.0%})")
    else:
        details["success_rate"] = "❌ BAD"
        issues.append(f"Success rate {success:.1%} is too low (robot falling)")

    # Check velocity
    if velocity > THRESHOLDS["forward_velocity"]["good"]:
        details["velocity"] = "✅ GOOD"
        good_signs.append(f"Velocity {velocity:.2f} m/s is good")
    elif velocity > THRESHOLDS["forward_velocity"]["marginal"]:
        details["velocity"] = "⚠️ MARGINAL"
        issues.append(f"Velocity {velocity:.2f} m/s is marginal (target: >{THRESHOLDS['forward_velocity']['good']})")
    else:
        details["velocity"] = "❌ BAD"
        issues.append(f"Velocity {velocity:.2f} m/s is too low (not moving)")

    # Check height
    if THRESHOLDS["height"]["target_min"] <= height <= THRESHOLDS["height"]["target_max"]:
        details["height"] = "✅ GOOD"
        good_signs.append(f"Height {height:.3f}m is in target range")
    elif THRESHOLDS["height"]["min"] <= height <= THRESHOLDS["height"]["max"]:
        details["height"] = "⚠️ OK"
        issues.append(f"Height {height:.3f}m is acceptable but not ideal")
    else:
        details["height"] = "❌ BAD"
        issues.append(f"Height {height:.3f}m is out of range ({THRESHOLDS['height']['min']}-{THRESHOLDS['height']['max']})")

    # Make decision
    critical_failures = [
        reward < THRESHOLDS["reward"]["same_as_before"],
        success < THRESHOLDS["success_rate"]["marginal"],
        velocity < THRESHOLDS["forward_velocity"]["marginal"],
    ]

    marginal_issues = [
        reward < THRESHOLDS["reward"]["good"],
        success < THRESHOLDS["success_rate"]["good"],
    ]

    if any(critical_failures):
        decision = "NO-GO"
        explanation = "❌ CRITICAL ISSUES FOUND - Do not start full training"
    elif sum(marginal_issues) >= 2:
        decision = "INVESTIGATE"
        explanation = "⚠️ MARGINAL RESULTS - Check contact metrics before deciding"
    else:
        decision = "GO"
        explanation = "✅ LOOKS GOOD - Ready for full training!"

    return decision, explanation, details


def find_latest_quickverify() -> Optional[Path]:
    """Find the latest quickverify training log."""
    logs_dir = Path(__file__).parent / "training_logs"
    if not logs_dir.exists():
        return None

    # Find all quickverify directories
    quickverify_dirs = sorted(logs_dir.glob("quickverify_*"))
    if not quickverify_dirs:
        return None

    # Return the most recent
    return quickverify_dirs[-1]


def check_training_logs(folder_name: Optional[str] = None) -> Optional[Dict[str, float]]:
    """Check training logs for metrics.

    Args:
        folder_name: Specific folder name (e.g., 'quickverify_phase1_contact_20251128-112811')
                    If None, uses latest quickverify folder
    """
    logs_dir = Path(__file__).parent / "training_logs"

    if not logs_dir.exists():
        print(f"Error: {logs_dir} does not exist")
        return None

    # Determine which folder to check
    if folder_name:
        log_folder = logs_dir / folder_name
        if not log_folder.exists():
            print(f"Error: Folder '{folder_name}' not found in training_logs/")
            print(f"\nAvailable folders:")
            for folder in sorted(logs_dir.iterdir()):
                if folder.is_dir():
                    print(f"  - {folder.name}")
            return None
    else:
        log_folder = find_latest_quickverify()
        if not log_folder:
            print("No quickverify training logs found in training_logs/")
            return None

    print(f"\nChecking: {log_folder.name}")

    # Try to find config
    config_file = log_folder / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
        print(f"  Config: {config.get('env', {}).get('terrain', 'unknown')}")

    # Look for log files in logs/ subdirectory
    logs_subdir = log_folder / "logs"

    # Try to find wandb output or other log files
    log_files_to_check = []

    if logs_subdir.exists():
        # Check for wandb debug logs
        wandb_dirs = list(logs_subdir.glob("wandb/debug-*.log"))
        log_files_to_check.extend(wandb_dirs)

        # Check for any .log or .txt files
        log_files_to_check.extend(logs_subdir.glob("*.log"))
        log_files_to_check.extend(logs_subdir.glob("*.txt"))

        # Check wandb run directories
        for run_dir in logs_subdir.glob("wandb/run-*"):
            if run_dir.is_dir():
                log_files_to_check.extend(run_dir.glob("*.log"))

    # Also check root of log_folder
    log_files_to_check.extend(log_folder.glob("*.log"))
    log_files_to_check.extend(log_folder.glob("*.txt"))
    log_files_to_check.extend(log_folder.glob("output*.txt"))

    # Try to parse each log file
    for log_file in log_files_to_check:
        try:
            with open(log_file) as f:
                content = f.read()

            metrics = parse_output(content)
            if metrics:
                print(f"  Found metrics in: {log_file.name}")
                return metrics
        except Exception:
            continue

    print("  Could not find parseable metrics in log files")
    print("  Please run with console output instead:")
    print(f"    python verify_quick_test.py")
    print("  Then paste the last training line\n")

    return None


def generate_diagnostics(metrics: Dict[str, float], decision: str) -> Dict[str, any]:
    """Generate structured diagnostic information for Devmate."""
    reward = metrics["reward"]
    success = metrics["success_rate"]
    velocity = metrics["velocity"]
    height = metrics["height"]

    diagnostics = {
        "decision": decision,
        "metrics": metrics,
        "issues": [],
        "potential_causes": [],
        "config_checks": [],
        "code_checks": [],
        "suggested_fixes": [],
    }

    # Diagnose reward issues
    if reward < -1000:
        diagnostics["issues"].append(f"CRITICAL: Reward {reward:.1f} is worse than baseline (-800 to -1000)")
        diagnostics["potential_causes"].extend([
            "Config file may not have uploaded (still using old z_velocity=1.0)",
            "Code changes not applied correctly",
            "New bug introduced in recent changes",
        ])
        diagnostics["config_checks"].extend([
            {"file": "phase1_contact.yaml", "key": "reward_weights.z_velocity", "expected": 0.1, "critical": True},
            {"file": "phase1_contact.yaml", "key": "reward_weights.roll_pitch_position", "expected": 2.0, "critical": True},
            {"file": "phase1_contact.yaml", "key": "reward_weights.foot_air_time", "expected": 3.0, "critical": False},
        ])
        diagnostics["code_checks"].extend([
            {"file": "amp/walk.py", "function": "reset()", "check": "All 40 metrics initialized in reset()"},
            {"file": "amp/walk.py", "function": "step()", "check": "Air time state tracking working"},
            {"file": "amp/rewards/contact_rewards.py", "class": "FootAirTimeReward", "check": "Reward computation correct"},
        ])
        diagnostics["suggested_fixes"].append({
            "priority": "HIGH",
            "action": "Verify config uploaded",
            "command": "ssh remote 'cat ~/projects/wildrobot/amp/phase1_contact.yaml | grep z_velocity'",
        })

    elif reward < -800:
        diagnostics["issues"].append(f"WARNING: Reward {reward:.1f} is marginal (target: >-500)")
        diagnostics["potential_causes"].extend([
            "Z-velocity penalty still too high",
            "Contact rewards not strong enough",
            "Robot still unstable (high z-velocity)",
        ])
        diagnostics["config_checks"].extend([
            {"file": "phase1_contact.yaml", "key": "reward_weights.z_velocity", "expected": 0.1, "critical": True},
            {"file": "phase1_contact.yaml", "key": "reward_weights.foot_contact", "expected": 5.0, "critical": False},
            {"file": "phase1_contact.yaml", "key": "reward_weights.tracking_exp_xy", "expected": 25.0, "critical": False},
        ])
        diagnostics["suggested_fixes"].append({
            "priority": "MEDIUM",
            "action": "Reduce z_velocity further if robot bouncing",
            "file": "phase1_contact.yaml",
            "change": "z_velocity: 0.05  # Reduce from 0.1",
        })

    # Diagnose success rate issues
    if success < 0.20:
        diagnostics["issues"].append(f"CRITICAL: Success rate {success:.1%} is too low (robot falling)")
        diagnostics["potential_causes"].extend([
            "Upright penalty too weak",
            "Z-velocity causing instability",
            "Contact rewards causing robot to tip over",
        ])
        diagnostics["suggested_fixes"].extend([
            {
                "priority": "HIGH",
                "action": "Increase upright penalty",
                "file": "phase1_contact.yaml",
                "change": "roll_pitch_position: 3.0  # Increase from 2.0",
            },
            {
                "priority": "HIGH",
                "action": "Reduce z_velocity penalty",
                "file": "phase1_contact.yaml",
                "change": "z_velocity: 0.05  # Reduce from 0.1",
            },
        ])

    elif success < 0.30:
        diagnostics["issues"].append(f"WARNING: Success rate {success:.1%} is marginal (target: >30%)")
        diagnostics["potential_causes"].append("Robot stability marginal")
        diagnostics["suggested_fixes"].append({
            "priority": "MEDIUM",
            "action": "Monitor stability, may need upright penalty increase",
            "file": "phase1_contact.yaml",
            "change": "roll_pitch_position: 2.5  # Increase from 2.0",
        })

    # Diagnose velocity issues
    if velocity < 0.1:
        diagnostics["issues"].append(f"CRITICAL: Velocity {velocity:.3f} m/s is too low (not moving)")
        diagnostics["potential_causes"].extend([
            "Tracking rewards too weak",
            "Robot spending energy on stability instead of movement",
            "Contact penalties preventing forward motion",
        ])
        diagnostics["suggested_fixes"].append({
            "priority": "HIGH",
            "action": "Increase tracking rewards",
            "file": "phase1_contact.yaml",
            "change": "tracking_exp_xy: 35.0  # Increase from 25.0",
        })

    # Diagnose height issues
    if height < 0.35 or height > 0.50:
        diagnostics["issues"].append(f"WARNING: Height {height:.3f}m is out of range (0.35-0.50)")
        if height < 0.35:
            diagnostics["potential_causes"].append("Robot collapsing or crouching")
            diagnostics["suggested_fixes"].append({
                "priority": "HIGH",
                "action": "Increase upright penalty to prevent collapse",
                "file": "phase1_contact.yaml",
                "change": "roll_pitch_position: 3.0  # Increase from 2.0",
            })
        else:
            diagnostics["potential_causes"].append("Robot standing too tall or bouncing")

    return diagnostics


def print_report(metrics: Dict[str, float], decision: str, explanation: str, details: Dict[str, str]):
    """Print a formatted report with Devmate-friendly diagnostics."""
    print("\n" + "="*70)
    print("🧪 QUICK VERIFY RESULTS ANALYSIS")
    print("="*70)

    print("\n📊 METRICS:")
    print(f"  Reward:        {metrics['reward']:>8.2f}  {details['reward']}")
    print(f"  Success Rate:  {metrics['success_rate']:>7.1%}  {details['success_rate']}")
    print(f"  Velocity:      {metrics['velocity']:>8.3f} m/s  {details['velocity']}")
    print(f"  Height:        {metrics['height']:>8.3f} m    {details['height']}")

    print(f"\n📋 DECISION: {decision}")
    print(f"  {explanation}")

    # Generate diagnostics
    diagnostics = generate_diagnostics(metrics, decision)

    # Print diagnostics if issues found
    if diagnostics["issues"] or decision != "GO":
        print("\n" + "="*70)
        print("🔍 DIAGNOSTICS FOR DEVMATE")
        print("="*70)

        if diagnostics["issues"]:
            print("\n❌ ISSUES DETECTED:")
            for i, issue in enumerate(diagnostics["issues"], 1):
                print(f"  {i}. {issue}")

        if diagnostics["potential_causes"]:
            print("\n🔎 POTENTIAL CAUSES:")
            for i, cause in enumerate(diagnostics["potential_causes"], 1):
                print(f"  {i}. {cause}")

        if diagnostics["config_checks"]:
            print("\n📝 CONFIG VALUES TO CHECK:")
            for check in diagnostics["config_checks"]:
                critical = " [CRITICAL]" if check.get("critical") else ""
                print(f"  • {check['file']}: {check['key']} = {check['expected']}{critical}")

        if diagnostics["code_checks"]:
            print("\n💻 CODE TO VERIFY:")
            for check in diagnostics["code_checks"]:
                if "function" in check:
                    print(f"  • {check['file']}: {check['function']} - {check['check']}")
                elif "class" in check:
                    print(f"  • {check['file']}: {check['class']} - {check['check']}")

        if diagnostics["suggested_fixes"]:
            print("\n🔧 SUGGESTED FIXES (Priority Order):")
            for i, fix in enumerate(sorted(diagnostics["suggested_fixes"],
                                          key=lambda x: 0 if x['priority']=='HIGH' else 1 if x['priority']=='MEDIUM' else 2), 1):
                print(f"\n  {i}. [{fix['priority']}] {fix['action']}")
                if "file" in fix and "change" in fix:
                    print(f"     File: {fix['file']}")
                    print(f"     Change: {fix['change']}")
                elif "command" in fix:
                    print(f"     Command: {fix['command']}")

    print("\n" + "="*70)

    if decision == "GO":
        print("\n✅ RECOMMENDATION: START FULL TRAINING")
        print("\nRun on remote:")
        print("  cd ~/projects/wildrobot/amp")
        print("  python train.py --config phase1_contact.yaml")
        print("\nMonitor these W&B metrics:")
        print("  • topline/reward (should improve to -200 to -400)")
        print("  • topline/success_rate (should reach 50-70%)")
        print("  • topline/reward_z_velocity (should be >-5)")
        print("  • topline/avg_air_time (should reach >0.15)")

    elif decision == "NO-GO":
        print("\n❌ RECOMMENDATION: DO NOT START FULL TRAINING")
        print("\n📋 NEXT STEPS FOR USER:")
        print("  1. Copy this entire output")
        print("  2. Paste into Devmate chat")
        print("  3. Say: 'Fix the issues found in quick verify'")
        print("  4. Devmate will diagnose and fix automatically")

    else:  # INVESTIGATE
        print("\n⚠️ RECOMMENDATION: INVESTIGATE BEFORE DECIDING")
        print("\n📋 NEXT STEPS:")
        print("  1. Review the diagnostics above")
        print("  2. Check if suggested fixes make sense")
        print("  3. Copy this output to Devmate for analysis")
        print("  4. Or run full training and monitor W&B closely")

    print("\n" + "="*70 + "\n")

    # Print machine-readable summary for easy parsing
    print("=" * 70)
    print("MACHINE-READABLE SUMMARY (for Devmate)")
    print("=" * 70)
    print(f"DECISION: {decision}")
    print(f"REWARD: {metrics['reward']:.2f}")
    print(f"SUCCESS_RATE: {metrics['success_rate']:.3f}")
    print(f"VELOCITY: {metrics['velocity']:.3f}")
    print(f"HEIGHT: {metrics['height']:.3f}")
    print(f"ISSUES_COUNT: {len(diagnostics['issues'])}")
    print(f"FIXES_COUNT: {len(diagnostics['suggested_fixes'])}")
    if diagnostics["config_checks"]:
        print("\nCONFIG_TO_CHECK:")
        for check in diagnostics["config_checks"]:
            print(f"  {check['key']}={check['expected']}")
    if diagnostics["suggested_fixes"]:
        print("\nSUGGESTED_CHANGES:")
        for fix in diagnostics["suggested_fixes"]:
            if "file" in fix and "change" in fix:
                print(f"  {fix['file']}: {fix['change']}")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Verify quick test results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (paste console line)
  python verify_quick_test.py

  # Check specific folder
  python verify_quick_test.py quickverify_phase1_contact_20251128-112811

  # Check latest quickverify folder
  python verify_quick_test.py --check-logs

  # Read from file
  python verify_quick_test.py output.txt

  # Paste full output
  python verify_quick_test.py --paste
        """
    )
    parser.add_argument("input", nargs="?", help="Folder name, file path, or console output")
    parser.add_argument("--check-logs", action="store_true", help="Check latest quickverify folder")
    parser.add_argument("--paste", action="store_true", help="Read from stdin (paste console output)")

    args = parser.parse_args()

    metrics = None

    # Determine input type
    if args.check_logs:
        # Check latest folder
        metrics = check_training_logs()
        if not metrics:
            print("Could not extract metrics from logs. Please provide console output instead.")
            return 1

    elif args.paste:
        # Read from stdin
        print("Paste console output (Ctrl+D when done):")
        text = sys.stdin.read()
        metrics = parse_output(text)

    elif args.input:
        # Could be: folder name, file path, or nothing
        input_path = Path(args.input)

        # Check if it's a folder name in training_logs/
        logs_dir = Path(__file__).parent / "training_logs"
        if (logs_dir / args.input).exists() and (logs_dir / args.input).is_dir():
            # It's a folder name
            metrics = check_training_logs(args.input)
            if not metrics:
                return 1

        # Check if it's a file path
        elif input_path.exists() and input_path.is_file():
            try:
                with open(input_path) as f:
                    text = f.read()
                metrics = parse_output(text)
            except Exception as e:
                print(f"Error reading file: {e}")
                return 1

        else:
            print(f"Error: '{args.input}' is not a valid folder name or file path")
            print(f"\nAvailable folders in training_logs/:")
            if logs_dir.exists():
                for folder in sorted(logs_dir.iterdir()):
                    if folder.is_dir():
                        print(f"  - {folder.name}")
            return 1

    else:
        # Interactive mode - ask for last line
        print("="*70)
        print("QUICK VERIFY RESULT CHECKER")
        print("="*70)
        print("\nPaste the last training line from console output:")
        print("(Example: Step 10,000/10,000 (100.0%) | Reward: -320.15 | ...)")
        print()

        line = input("> ")
        metrics = parse_console_line(line)

    if not metrics:
        print("\n❌ Error: Could not parse metrics from input")
        print("\nMake sure you provide a line like:")
        print("Step 10,000/10,000 (100.0%) | Reward: -320.15 | Vel: 0.520 m/s | Height: 0.428m | Success: 52.00% | ...")
        return 1

    # Check metrics and make decision
    decision, explanation, details = check_metrics(metrics)

    # Print report
    print_report(metrics, decision, explanation, details)

    # Exit code: 0 for GO, 1 for NO-GO, 2 for INVESTIGATE
    exit_codes = {"GO": 0, "NO-GO": 1, "INVESTIGATE": 2}
    return exit_codes[decision]


if __name__ == "__main__":
    sys.exit(main())
