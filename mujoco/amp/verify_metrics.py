#!/usr/bin/env python3
"""Verify metrics logging correctness in training runs.

Usage:
    python verify_metrics.py <training_log_dir>

Example:
    python verify_metrics.py training_logs/quickverify_phase1_contact_20251130-150951
"""

import json
import sys
from pathlib import Path


def verify_metrics(log_dir: Path) -> bool:
    """Verify that metrics logging is correct.

    Args:
        log_dir: Path to training log directory

    Returns:
        True if metrics are correct, False otherwise
    """
    metrics_file = log_dir / "metrics.jsonl"

    if not metrics_file.exists():
        print(f"❌ Metrics file not found: {metrics_file}")
        return False

    print("="*80)
    print(f"METRICS VERIFICATION: {log_dir.name}")
    print("="*80)

    with open(metrics_file) as f:
        lines = f.readlines()
        if not lines:
            print("❌ Metrics file is empty!")
            return False
        final = json.loads(lines[-1])

    print(f"\n✅ 1. CHECK: Does summary use WEIGHTED values?")
    print(f"\n  Unweighted totals (raw physical values):")
    print(f"    rewards.total:         {final['rewards']['total']:>10.2f}")
    print(f"    penalties.total:       {final['penalties']['total']:>10,.2f}")

    print(f"\n  Weighted totals (actual training impact):")
    print(f"    rewards_weighted.total:    {final['rewards_weighted']['total']:>10.2f}")
    print(f"    penalties_weighted.total:  {final['penalties_weighted']['total']:>10.2f}")

    print(f"\n  Summary section:")
    print(f"    summary.total_rewards:     {final['summary']['total_rewards']:>10.2f}")
    print(f"    summary.total_penalties:   {final['summary']['total_penalties']:>10.2f}")
    print(f"    summary.reward_per_step:   {final['summary']['reward_per_step']:>10.2f}")

    # Check if summary matches weighted
    summary_uses_weighted = (
        abs(final['summary']['total_rewards'] - final['rewards_weighted']['total']) < 0.01 and
        abs(final['summary']['total_penalties'] - final['penalties_weighted']['total']) < 0.01
    )

    print(f"\n  Verification:")
    if summary_uses_weighted:
        print(f"    ✅ summary.total_rewards matches rewards_weighted.total")
        print(f"    ✅ summary.total_penalties matches penalties_weighted.total")
    else:
        print(f"    ❌ summary.total_rewards does NOT match rewards_weighted.total")
        print(f"    ❌ summary.total_penalties does NOT match penalties_weighted.total")
        print(f"    ❌ Summary is using UNWEIGHTED values (BUG!)")

    print(f"\n✅ 2. CHECK: Does the math work out?")
    calculated = final['rewards_weighted']['total'] - final['penalties_weighted']['total']
    actual = final['summary']['reward_per_step']
    print(f"  {final['rewards_weighted']['total']:.2f} - {final['penalties_weighted']['total']:.2f} = {calculated:.2f}")
    print(f"  reward_per_step = {actual:.2f}")

    math_works = abs(calculated - actual) < 0.01
    if math_works:
        print(f"  ✅ MATH CHECKS OUT! ({calculated:.2f} ≈ {actual:.2f})")
    else:
        print(f"  ❌ MATH DOESN'T MATCH! ({calculated:.2f} ≠ {actual:.2f})")

    print(f"\n📊 3. TRAINING PERFORMANCE:")
    print(f"  Reward/step:       {final['summary']['reward_per_step']:>8.2f}")
    print(f"  Success rate:      {final['summary']['success_rate']:>8.1f}%")
    print(f"  Forward velocity:  {final['summary']['forward_velocity']:>8.3f} m/s")
    print(f"  Episode length:    {final['summary']['episode_length']:>8.1f} steps")

    print(f"\n🔍 4. TOP WEIGHTED PENALTIES:")
    penalties_sorted = sorted(
        [(k, v) for k, v in final['penalties_weighted'].items() if k != 'total'],
        key=lambda x: x[1],
        reverse=True
    )[:5]
    for name, val in penalties_sorted:
        print(f"  {name:30s} -{val:>8.2f}")

    print(f"\n🔍 5. TOP WEIGHTED REWARDS:")
    rewards_sorted = sorted(
        [(k, v) for k, v in final['rewards_weighted'].items() if k != 'total'],
        key=lambda x: x[1],
        reverse=True
    )[:5]
    for name, val in rewards_sorted:
        print(f"  {name:30s} +{val:>8.2f}")

    print("\n" + "="*80)
    print("CONCLUSION:")
    print("="*80)

    all_correct = summary_uses_weighted and math_works
    if all_correct:
        print("✅ METRICS LOGGING IS CORRECT!")
        print("✅ Summary uses weighted values")
        print("✅ Math checks out")
    else:
        print("❌ METRICS HAVE ISSUES:")
        if not summary_uses_weighted:
            print("  - Summary is using unweighted values instead of weighted")
        if not math_works:
            print("  - Reward calculation doesn't add up")
    print("="*80)

    return all_correct


def main():
    if len(sys.argv) != 2:
        print("Usage: python verify_metrics.py <training_log_dir>")
        print("\nExample:")
        print("  python verify_metrics.py training_logs/quickverify_phase1_contact_20251130-150951")
        sys.exit(1)

    log_dir = Path(sys.argv[1])
    if not log_dir.exists():
        print(f"❌ Directory not found: {log_dir}")
        sys.exit(1)

    success = verify_metrics(log_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
