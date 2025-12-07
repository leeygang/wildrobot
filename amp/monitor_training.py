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
from datetime import datetime, timedelta
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


def _detect_phase(config: dict, log_dir: Path, cli_phase: str | None = None) -> str:
    """Detect training phase from CLI override, wandb tags, or folder naming.

    Returns: 'phase0', 'phase1', or 'unknown'
    """
    if cli_phase and cli_phase in {"phase0", "phase1"}:
        return cli_phase

    # Try wandb tags
    try:
        tags = config.get("logging", {}).get("wandb_tags", []) or []
        if any(str(t).lower() == "phase0" for t in tags):
            return "phase0"
        if any(str(t).lower().startswith("phase1") for t in tags):
            return "phase1"
    except Exception:
        pass

    # Fallback: directory name heuristics
    name = log_dir.name.lower()
    if "phase0" in name:
        return "phase0"
    if "phase1" in name:
        return "phase1"

    return "unknown"


def _phase0_checks(latest: dict, first: dict, mid: dict, config: dict, progress_pct: float) -> tuple[list[str], list[str], list[str]]:
    """Phase 0 specific health checks and guidance.

    Returns: (issues, warnings, good_signs)
    """
    issues: list[str] = []
    warnings: list[str] = []
    good: list[str] = []

    # Extract helpful fields with guards
    summary = latest.get("summary", {})
    first_s = first.get("summary", {})
    gate = summary.get("tracking_gate_active_rate")
    scale = summary.get("velocity_threshold_scale")
    velocity = summary.get("forward_velocity")
    episode_len = summary.get("episode_length") or config.get("training", {}).get("episode_length", 600)
    ctrl_dt = config.get("env", {}).get("ctrl_dt", 0.02)

    # Dynamic distance proxy if not logged explicitly
    distance_logged = summary.get("distance_walked")
    if distance_logged is None and velocity is not None:
        distance_logged = velocity * episode_len * ctrl_dt

    # Stage thresholds that ramp with progress
    # Early (<300k steps): focus on positive velocity and decay kickoff soon
    # Mid (>=300k): velocity >= 0.40 m/s, gate < 30%
    # Late (>=1.0M): velocity >= 0.55 m/s, gate < 15%
    # Use progress percentage of total steps as proxy for absolute steps
    total_steps = config.get("training", {}).get("num_timesteps", 20_000_000)
    # Try to infer current steps if available
    current_steps = latest.get("step", progress_pct * total_steps / 100)

    mid_threshold = 300_000
    late_threshold = 1_000_000

    # Velocity checks
    if velocity is not None:
        if current_steps >= late_threshold and velocity < 0.55:
            warnings.append(f"Phase0: velocity {velocity:.3f} < 0.55 m/s target for exit")
        elif current_steps >= mid_threshold and velocity < 0.40:
            warnings.append(f"Phase0: velocity {velocity:.3f} < 0.40 m/s mid-phase target")
        elif velocity > 0.4:
            good.append(f"Phase0: good forward velocity ({velocity:.3f} m/s)")

    # Gate activity checks
    if gate is not None:
        if current_steps >= late_threshold and gate > 0.15:
            warnings.append(f"Phase0: gate active {gate:.0%} > 15% exit target")
        elif current_steps >= mid_threshold and gate > 0.30:
            warnings.append(f"Phase0: gate active {gate:.0%} > 30% mid-phase target")
        elif gate < 0.20:
            good.append(f"Phase0: gate rarely active ({gate:.0%})")

    # Decay activation check (scale < 1 after some steps)
    if scale is not None:
        if current_steps > 200_000 and scale >= 0.99:
            warnings.append(f"Phase0: velocity-threshold decay not active yet (scale={scale:.2f})")
        elif scale < 0.9:
            good.append(f"Phase0: decay active (scale={scale:.2f})")

    # Distance consistency check against threshold velocity * horizon (10% tolerance)
    if distance_logged is not None and episode_len and ctrl_dt:
        # Expected distance for 0.55 m/s target
        expected = 0.55 * episode_len * ctrl_dt
        if current_steps >= late_threshold and distance_logged < 0.9 * expected:
            warnings.append(f"Phase0: distance {distance_logged:.2f}m < {0.9*expected:.2f}m target")

    # Success rate lenient floor in Phase 0
    success = summary.get("success_rate")
    if success is not None:
        if current_steps >= mid_threshold and success < 50:
            warnings.append(f"Phase0: low success rate ({success:.1f}%) mid-phase")
        elif success >= 60:
            good.append(f"Phase0: good success rate ({success:.1f}%)")

    return issues, warnings, good


def _detect_catastrophic_forgetting(metrics: list, current_steps: int) -> tuple[bool, dict]:
    """Detect catastrophic forgetting by tracking peak performance degradation.
    
    Based on analysis of 3 trainings:
    - BASELINE: peaks 8.8M (31.10), collapses by 11M (<50% peak) - 64% wasted compute
    - NEW: peaks 15.9M (28.96), collapses by 19M (<50% peak) - 30% wasted compute
    - LATEST: peaks 0.1M (22.57), collapses by 0.9M (<50% peak) - 98% wasted compute
    
    Strategy:
    - Track best reward in last 2M steps (rolling window for recent peak)
    - If current reward < 60% of recent peak AND stayed low for 1M steps, trigger stop
    - Prevents early false positives while catching sustained degradation
    - Minimum 2M steps before allowing early stop
    
    Returns:
        (should_stop, info_dict)
    """
    if len(metrics) < 10:
        return False, {"reason": "insufficient_data", "evals": len(metrics)}
    
    rewards = [entry['summary']['reward_per_step'] for entry in metrics]
    steps = [entry['step'] for entry in metrics]
    
    # Find global peak
    peak_reward = max(rewards)
    peak_idx = rewards.index(peak_reward)
    peak_step = steps[peak_idx]
    
    # Find recent peak (last 2M steps)
    recent_window_start = current_steps - 2_000_000
    recent_indices = [i for i, s in enumerate(steps) if s >= recent_window_start]
    
    if len(recent_indices) < 5:
        return False, {"reason": "insufficient_recent_data", "recent_evals": len(recent_indices)}
    
    recent_rewards = [rewards[i] for i in recent_indices]
    recent_peak = max(recent_rewards)
    
    # Check if in sustained degradation (last 1M steps all below threshold)
    sustained_window_start = current_steps - 1_000_000
    sustained_indices = [i for i, s in enumerate(steps) if s >= sustained_window_start]
    
    if len(sustained_indices) < 3:
        return False, {"reason": "insufficient_sustained_window", "sustained_evals": len(sustained_indices)}
    
    sustained_rewards = [rewards[i] for i in sustained_indices]
    sustained_max = max(sustained_rewards)
    current_reward = rewards[-1]
    
    # Trigger conditions:
    # 1. Peak was significant (>10 reward) - not just noise
    # 2. Degraded below 50% of global peak
    # 3. Best in last 1M steps is also below 60% of global peak (sustained degradation)
    # 4. Current reward hasn't recovered (< 40% of global peak)
    
    # Use stricter thresholds based on global peak for clearer signal
    degraded_threshold = 0.50  # Current < 50% of peak
    sustained_threshold = 0.60  # Sustained max < 60% of peak
    critical_threshold = 0.40   # Current < 40% of peak (very degraded)
    
    is_peak_significant = peak_reward > 10.0
    is_degraded = current_reward < (peak_reward * degraded_threshold)
    is_sustained = sustained_max < (peak_reward * sustained_threshold)
    is_critical = current_reward < (peak_reward * critical_threshold)
    
    should_stop = (
        is_peak_significant and 
        ((is_degraded and is_sustained) or is_critical) and  # Either sustained degradation OR critical drop
        current_steps >= 2_000_000  # Minimum 2M steps before allowing early stop
    )
    
    info = {
        "peak_reward": peak_reward,
        "peak_step": peak_step,
        "recent_peak": recent_peak,
        "current_reward": current_reward,
        "sustained_max": sustained_max,
        "degradation_pct": ((current_reward - peak_reward) / peak_reward * 100) if peak_reward > 0 else 0,
        "current_vs_peak_pct": (current_reward / peak_reward * 100) if peak_reward > 0 else 0,
        "sustained_vs_peak_pct": (sustained_max / peak_reward * 100) if peak_reward > 0 else 0,
        "is_peak_significant": is_peak_significant,
        "is_degraded": is_degraded,
        "is_sustained": is_sustained,
        "is_critical": is_critical,
        "degraded_threshold": degraded_threshold,
        "sustained_threshold": sustained_threshold,
        "critical_threshold": critical_threshold,
    }
    
    return should_stop, info


def monitor_training(log_dir: Path, phase_override: str | None = None) -> dict:
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

        metrics_list = [json.loads(line) for line in lines]
        first = metrics_list[0]
        latest = metrics_list[-1]
        mid = metrics_list[len(metrics_list)//2] if len(metrics_list) > 2 else first

    # Load config
    config_file = log_dir / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
        total_steps = config['training']['num_timesteps']
    else:
        total_steps = 20_000_000  # Default
        config = {}

    current_steps = latest['step']
    progress_pct = (current_steps / total_steps) * 100

    # Check for catastrophic forgetting
    should_early_stop, forgetting_info = _detect_catastrophic_forgetting(metrics_list, current_steps)

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
    
    # Phase 1 contact metrics (if available)
    alternation_ratio = latest['summary'].get('contact_alternation_ratio')
    if alternation_ratio is not None:
        print(f"  Alternation:  {alternation_ratio:>8.2%} (target: >85%)")

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

    # Early stopping analysis
    if forgetting_info.get('reason') != 'insufficient_data':
        print(f"\n🧠 EARLY STOPPING ANALYSIS:")
        if should_early_stop:
            print(f"  🚨 CATASTROPHIC FORGETTING DETECTED!")
            print(f"  Peak reward:      {forgetting_info['peak_reward']:>8.2f} @ {forgetting_info['peak_step']/1e6:.1f}M steps")
            print(f"  Recent peak:      {forgetting_info['recent_peak']:>8.2f} (last 2M steps)")
            print(f"  Current reward:   {forgetting_info['current_reward']:>8.2f}")
            print(f"  Sustained max:    {forgetting_info['sustained_max']:>8.2f} (last 1M steps)")
            print(f"  Degradation:      {forgetting_info['degradation_pct']:>8.1f}% from recent peak")
            print(f"  Current vs peak:  {forgetting_info['current_vs_peak_pct']:>8.1f}% of global peak")
            print(f"  💡 Training has degraded and not recovered for 1M+ steps")
        else:
            print(f"  Peak reward:      {forgetting_info['peak_reward']:>8.2f} @ {forgetting_info['peak_step']/1e6:.1f}M steps")
            print(f"  Recent peak:      {forgetting_info['recent_peak']:>8.2f} (last 2M steps)")
            print(f"  Current reward:   {forgetting_info['current_reward']:>8.2f}")
            print(f"  Status:           ✅ No sustained degradation detected")
            degraded_flags = []
            if not forgetting_info['is_peak_significant']:
                degraded_flags.append("peak too low")
            if not forgetting_info['is_degraded']:
                degraded_flags.append("not degraded")
            if not forgetting_info['is_sustained']:
                degraded_flags.append("not sustained")
            if degraded_flags:
                print(f"  Reason:           {', '.join(degraded_flags)}")

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

    # Phase-specific checks (Phase 0 foundations)
    phase = _detect_phase(config, log_dir, phase_override)
    if phase == "phase0":
        print("\n🧭 PHASE 0 CHECKS:")
        p0_issues, p0_warnings, p0_good = _phase0_checks(latest, first, mid, config, progress_pct)
        for m in p0_issues:
            print(f"  {m}")
        for m in p0_warnings:
            print(f"  {m}")
        for m in p0_good:
            print(f"  {m}")
        issues.extend(p0_issues)
        warnings.extend(p0_warnings)
        good_signs.extend(p0_good)

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
    
    # Priority 1: Check for catastrophic forgetting (overrides all other checks)
    if should_early_stop:
        print(f"  🚨 STOP TRAINING IMMEDIATELY - Catastrophic forgetting detected!")
        print(f"     Peak: {forgetting_info['peak_reward']:.2f} @ {forgetting_info['peak_step']/1e6:.1f}M steps")
        print(f"     Current: {forgetting_info['current_reward']:.2f} ({forgetting_info['current_vs_peak_pct']:.1f}% of peak)")
        print(f"     Degraded for 1M+ steps with no recovery")
        print(f"     💡 Load checkpoint from {forgetting_info['peak_step']/1e6:.1f}M steps and adjust hyperparameters")
        status = "catastrophic_forgetting"
        return {
            "status": status,
            "current_steps": current_steps,
            "progress_pct": progress_pct,
            "issues": issues,
            "warnings": warnings,
            "good_signs": good_signs,
            "forgetting_detected": True,
            "forgetting_info": forgetting_info,
        }
    
    # Phase-aware tightening: For Phase 0, only mark critical on sustained stall
    phase = _detect_phase(config, log_dir, phase_override)
    latest_vel = latest['summary'].get('forward_velocity', 0.0)
    latest_gate = latest['summary'].get('tracking_gate_active_rate', None)
    mid_vel = mid.get('summary', {}).get('forward_velocity', None)
    mid_gate = mid.get('summary', {}).get('tracking_gate_active_rate', None)

    sustained_stall = False
    if phase == "phase0" and latest_gate is not None and mid_gate is not None and mid_vel is not None:
        sustained_stall = (latest_gate > 0.65 and latest_vel < 0.02 and mid_gate > 0.65 and mid_vel < 0.02)

    # Recovery detection: gate moderated and small positive velocity across two checkpoints
    recovery_detected = False
    if phase == "phase0" and latest_gate is not None and mid_gate is not None and mid_vel is not None:
        recovery_detected = (latest_gate < 0.55 and latest_vel > 0.08 and mid_gate < 0.55 and (mid_vel or 0) > 0.08)
        if recovery_detected:
            print("  🟢 RECOVERY DETECTED - Gate moderated and velocity positive across checkpoints")
            good_signs.append("✅ Recovery: gate < 55% and vel > 0.08 m/s (mid+latest)")

    if issues:
        if sustained_stall:
            print(f"  🚨 STOP TRAINING - Sustained Phase 0 stall detected (gate>65%, vel<0.02 m/s)")
            print(f"     Consider softening gating and boosting forward bonus")
            status = "critical"
        else:
            # Downgrade to warning in Phase 0 to allow recovery window
            if phase == "phase0":
                if recovery_detected:
                    print(f"  ✅ Continue training - Recovery trend detected")
                    status = "good"
                else:
                    print(f"  ⏸️  RECOVERY WINDOW - Issues detected but not sustained; continue monitoring")
                    status = "warning"
            else:
                print(f"  🚨 STOP TRAINING - Critical issues detected!")
                print(f"     Review reward configuration")
                status = "critical"
    else:
        # Phase-aware decision
        if phase == "phase0":
            # If key Phase 0 goals not trending, suggest pause
            gate_latest = latest_gate
            scale_latest = latest['summary'].get('velocity_threshold_scale', None)

            mid_ok = (latest_vel >= 0.40) or (progress_pct < 1.5)  # allow very early grace period
            gate_ok = (gate_latest is None) or (gate_latest <= 0.30) or (progress_pct < 1.5)
            decay_ok = (scale_latest is None) or (scale_latest < 0.99) or (progress_pct < 1.0)

            if progress_pct >= 5 and not (mid_ok and gate_ok and decay_ok):
                print("  ⏸️  PAUSE SOON IF NO IMPROVEMENT - Phase 0 targets not trending")
                print("     Next step: verify env_global_step wiring and gating params")
                status = "warning"
            else:
                if progress_pct < 10:
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
        else:
            if progress_pct < 10:
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
    # No separate fallback; status already set above

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
        "forgetting_detected": should_early_stop,
        "forgetting_info": forgetting_info if should_early_stop else None,
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
        "--phase",
        type=str,
        default=None,
        choices=["phase0", "phase1", "auto", None],
        help="Override phase detection (default: auto)",
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
        # Require multiple consecutive confirmations before treating a run as complete.
        completion_streak = 0
        completion_required = 3
        completion_threshold = 99.9
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
                result = monitor_training(log_dir, None if args.phase in (None, "auto") else args.phase)

                # Check if training is complete or has critical issues
                pct = float(result.get("progress_pct", 0.0))
                if pct >= completion_threshold:
                    completion_streak += 1
                    print(f"\n🎉 Progress at {pct:.1f}% >= {completion_threshold}%. Confirmation {completion_streak}/{completion_required}.")
                    # If the run reports progress >100% this is suspicious (mismatched config/metrics);
                    # warn the user but still require confirmations before exiting.
                    if pct > 100.0:
                        print(f"   ⚠️  Progress ({pct:.1f}%) exceeds configured total_steps. Check run `config.json` and `metrics.jsonl` semantics.")

                    if completion_streak >= completion_required:
                        print(f"\n🎉 TRAINING COMPLETE!")
                        print(f"   Final check performed.")
                        break
                    else:
                        print("   Waiting for consecutive confirmations before exiting the monitor.")
                else:
                    # Reset streak when progress falls below threshold
                    if pct > 100.0:
                        print(f"\n⚠️  Progress ({pct:.1f}%) exceeds configured total_steps - treating as suspicious and continuing to monitor.")
                    completion_streak = 0

                # Check for catastrophic forgetting (highest priority)
                if result["status"] == "catastrophic_forgetting":
                    print(f"\n🚨 CATASTROPHIC FORGETTING DETECTED!")
                    
                    if args.force_stop:
                        print(f"   --force_stop enabled: Attempting to kill training...")
                        success = kill_local_training()
                        
                        if success:
                            print(f"\n   ✅ Training stopped successfully")
                            print(f"   Load checkpoint from {result['forgetting_info']['peak_step']/1e6:.1f}M steps")
                            print(f"   Stopping monitor.")
                            sys.exit(1)
                        else:
                            print(f"\n   ⚠️  Could not kill training automatically")
                            print(f"   Please stop manually and load checkpoint from {result['forgetting_info']['peak_step']/1e6:.1f}M steps")
                            sys.exit(1)
                    else:
                        print(f"   💡 Run with --force_stop to automatically kill training")
                        print(f"   Or manually stop and load checkpoint from {result['forgetting_info']['peak_step']/1e6:.1f}M steps")
                        print(f"   ⏱  Continuing to monitor (no --force_stop). Will re-check at the next interval.")
                        # Do not stop the monitor loop; allow users to intervene manually
                        # Continue to next check
                        continue

                if result["status"] == "critical":
                    print(f"\n🚨 CRITICAL ISSUES DETECTED!")

                    # Define an 'early' guard: don't kill too early in training
                    progress_pct = float(result.get("progress_pct", 0.0))
                    is_early = progress_pct < 10.0  # Treat <10% progress as 'too early' to force-stop

                    # Force stop if enabled and not early
                    if args.force_stop:
                        if is_early:
                            print(f"   ⏸️  --force_stop ignored: training is too early ({progress_pct:.1f}% done)")
                            print(f"   Will continue monitoring without killing the training process.")
                            # Do not exit; keep monitoring
                            # Sleep until next interval
                        else:
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
                            print(f"   ⚠️  Recommendation: Stop training and inspect logs/checkpoints")
                            print(f"   Hint: Run this script with --force_stop to automatically kill local training when critical issues are detected")
                            print(f"   ⏱  Continuing to monitor (no --force_stop). Will re-check at the next interval.")
                            # Do not exit; continue monitoring so user can intervene
                            continue

                # Wait for next check (show clock time for next check)
                next_time = datetime.now() + timedelta(minutes=args.interval)
                print(f"\n⏳ Next check in {args.interval} minutes ({next_time:%H:%M})...")
                print(f"   (Press Ctrl+C to stop monitoring)")
                time.sleep(args.interval * 60)

        except KeyboardInterrupt:
            print(f"\n\n⚠️  Monitoring stopped by user (Ctrl+C)")
            print(f"   Total checks performed: {check_count}")
            sys.exit(0)
    else:
        # Single check mode
        result = monitor_training(log_dir, None if args.phase in (None, "auto") else args.phase)

        # Exit code based on status
        if result["status"] == "critical":
            sys.exit(1)
        else:
            sys.exit(0)


if __name__ == "__main__":
    main()
