#!/usr/bin/env python3
"""Multi-seed nominal-to-PPO handoff gate evaluator (v0.19.4-C/D).

Runs the nominal-only probe across N seeds and evaluates against the
quantitative handoff gate defined in walking_training.md and
reference_design.md.

Usage:
    JAX_PLATFORMS=cpu uv run python training/eval/eval_handoff_gate.py
    JAX_PLATFORMS=cpu uv run python training/eval/eval_handoff_gate.py \
        --config training/configs/ppo_walking_v0194c.yaml \
        --seeds 10 --forward-cmd 0.15 --horizon 500

Gate criteria (all must pass):
    1. Multi-seed robustness: >=80% seeds survive >=400/500, no seed <300,
       mean survival >=430
    2. Forward velocity: mean >=0.075 m/s (50% of cmd=0.15)
    3. Step generation: commanded >=0.045m, realized >=0.03m or ratio >=0.5
    4. Stability: pitch p95 <=0.20, roll p95 <=0.12, term fracs <=0.20
    5. Gait quality: stance switches on all passing seeds
    6. Lateral drift: direction varies across seeds (not systematic)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from training.eval.eval_loc_ref_probe import run_nominal_probe


def _evaluate_gate(
    results: list[dict[str, Any]],
    *,
    forward_cmd: float,
    horizon: int,
) -> dict[str, Any]:
    """Evaluate quantitative handoff gate across multi-seed results."""
    n_seeds = len(results)
    if n_seeds == 0:
        return {"pass": False, "reason": "no results"}

    # Per-seed metrics
    survivals = []
    forward_vels = []
    step_lengths = []
    foothold_cmds = []
    pitch_vals = []
    roll_vals = []
    drift_dirs = []
    stance_switches = []
    term_types = []

    for r in results:
        done_step = int(r.get("done_step", 0))
        done_reached = bool(r.get("done_reached", False))
        survival = done_step if done_reached else horizon
        survivals.append(survival)
        forward_vels.append(float(r.get("forward_velocity_mean", 0.0)))
        step_lengths.append(float(r.get("debug_step_length_m_mean", 0.0)))
        # Use raw nominal foothold (before support scaling), not the gated swing target
        foothold_cmds.append(float(r.get("debug_loc_ref_foothold_x_raw_mean",
                                         r.get("debug_loc_ref_swing_x_target_mean", 0.0))))
        pitch_vals.append(float(r.get("debug_loc_ref_root_pitch_abs_p95", 0.0)))
        roll_vals.append(float(r.get("debug_loc_ref_root_roll_abs_p95", 0.0)))
        # Lateral drift direction from mean lateral velocity
        lat_vel = float(r.get("lateral_velocity_mean", 0.0))
        drift_dirs.append(1 if lat_vel > 0.005 else (-1 if lat_vel < -0.005 else 0))
        switches = int(r.get("stance_switch_count", 0))
        stance_switches.append(switches)
        term_types.append(str(r.get("dominant_termination", "none")))

    survivals_arr = np.array(survivals)
    forward_vels_arr = np.array(forward_vels)

    gate = {}
    gate["n_seeds"] = n_seeds
    gate["forward_cmd"] = forward_cmd
    gate["horizon"] = horizon

    # 1. Multi-seed robustness
    survive_400 = int(np.sum(survivals_arr >= 400))
    survive_400_frac = survive_400 / n_seeds
    min_survival = int(np.min(survivals_arr))
    mean_survival = float(np.mean(survivals_arr))
    gate["robustness"] = {
        "survive_400_frac": survive_400_frac,
        "survive_400_count": survive_400,
        "min_survival": min_survival,
        "mean_survival": mean_survival,
        "pass_survive_400": survive_400_frac >= 0.80,
        "pass_min_300": min_survival >= 300,
        "pass_mean_430": mean_survival >= 430,
        "pass": survive_400_frac >= 0.80 and min_survival >= 300 and mean_survival >= 430,
    }

    # 2. Forward velocity
    mean_fwd = float(np.mean(forward_vels_arr))
    target_fwd = 0.5 * forward_cmd
    gate["forward_velocity"] = {
        "mean": mean_fwd,
        "target": target_fwd,
        "tracking_ratio": mean_fwd / max(forward_cmd, 1e-6),
        "pass": mean_fwd >= target_fwd,
    }

    # 3. Step generation
    mean_foothold_cmd = float(np.mean(foothold_cmds))
    mean_step_len = float(np.mean(step_lengths))
    step_ratio = mean_step_len / max(mean_foothold_cmd, 1e-6)
    gate["step_generation"] = {
        "mean_foothold_cmd_m": mean_foothold_cmd,
        "mean_step_length_m": mean_step_len,
        "step_ratio": step_ratio,
        "pass_cmd_adequate": mean_foothold_cmd >= 0.045,
        "pass_realized": mean_step_len >= 0.03 or step_ratio >= 0.5,
        "pass": mean_foothold_cmd >= 0.045 and (mean_step_len >= 0.03 or step_ratio >= 0.5),
    }

    # 4. Stability
    mean_pitch_p95 = float(np.mean(pitch_vals))
    mean_roll_p95 = float(np.mean(roll_vals))
    term_pitch_count = sum(1 for t in term_types if t == "term/pitch")
    term_roll_count = sum(1 for t in term_types if t == "term/roll")
    term_pitch_frac = term_pitch_count / n_seeds
    term_roll_frac = term_roll_count / n_seeds
    gate["stability"] = {
        "mean_pitch_p95": mean_pitch_p95,
        "mean_roll_p95": mean_roll_p95,
        "term_pitch_frac": term_pitch_frac,
        "term_roll_frac": term_roll_frac,
        "pass_pitch_p95": mean_pitch_p95 <= 0.20,
        "pass_roll_p95": mean_roll_p95 <= 0.12,
        "pass_term_pitch": term_pitch_frac <= 0.20,
        "pass_term_roll": term_roll_frac <= 0.20,
        "pass": (mean_pitch_p95 <= 0.20 and mean_roll_p95 <= 0.12
                 and term_pitch_frac <= 0.20 and term_roll_frac <= 0.20),
    }

    # 5. Gait quality
    passing_seeds = [i for i, s in enumerate(survivals) if s >= 400]
    all_passing_have_switches = all(stance_switches[i] > 0 for i in passing_seeds) if passing_seeds else False
    gate["gait_quality"] = {
        "passing_seeds_count": len(passing_seeds),
        "all_passing_have_switches": all_passing_have_switches,
        "min_switches_passing": min((stance_switches[i] for i in passing_seeds), default=0),
        "pass": all_passing_have_switches,
    }

    # 6. Lateral drift
    nonzero_drifts = [d for d in drift_dirs if d != 0]
    if len(nonzero_drifts) >= 2:
        drift_systematic = all(d == nonzero_drifts[0] for d in nonzero_drifts)
    else:
        drift_systematic = False
    gate["lateral_drift"] = {
        "drift_directions": drift_dirs,
        "systematic": drift_systematic,
        "pass": not drift_systematic,
    }

    # Overall
    all_pass = all([
        gate["robustness"]["pass"],
        gate["forward_velocity"]["pass"],
        gate["step_generation"]["pass"],
        gate["stability"]["pass"],
        gate["gait_quality"]["pass"],
        gate["lateral_drift"]["pass"],
    ])
    gate["overall_pass"] = all_pass

    # Per-seed summary
    gate["per_seed"] = []
    for i, r in enumerate(results):
        gate["per_seed"].append({
            "seed": int(r.get("seed", i)),
            "survival_steps": survivals[i],
            "forward_vel": forward_vels[i],
            "step_length": step_lengths[i],
            "foothold_cmd": foothold_cmds[i],
            "pitch_p95": pitch_vals[i],
            "roll_p95": roll_vals[i],
            "drift_dir": drift_dirs[i],
            "stance_switches": stance_switches[i],
            "term_type": term_types[i],
        })

    return gate


def run_handoff_gate(
    *,
    config_path: str,
    forward_cmd: float = 0.15,
    horizon: int = 500,
    n_seeds: int = 5,
    seed_start: int = 0,
    output_dir: str = "/tmp/handoff_gate",
    step_time_override: float | None = None,
) -> dict[str, Any]:
    """Run multi-seed nominal probe and evaluate handoff gate."""
    from training.configs.training_config import load_training_config

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    results = []
    for i in range(n_seeds):
        seed = seed_start + i
        print(f"\n--- Seed {seed} ({i+1}/{n_seeds}) ---")

        summary = run_nominal_probe(
            config_path=config_path,
            forward_cmd=forward_cmd,
            horizon=horizon,
            seed=seed,
            residual_scale_override=0.0,
            num_envs=1,
            trace_output=str(out / f"trace_seed{seed}.json"),
            stop_on_done=True,
        )
        summary["seed"] = seed

        # Compute additional metrics from the trace
        trace_path = out / f"trace_seed{seed}.json"
        if trace_path.exists():
            trace = json.loads(trace_path.read_text(encoding="utf-8"))
            # Pitch p95
            if "root_pitch" in trace:
                pitch_abs = np.abs(np.array(trace["root_pitch"]))
                summary["debug_loc_ref_root_pitch_abs_p95"] = float(np.percentile(pitch_abs, 95))
            # Step length mean (touchdown-only — nonzero entries)
            if "step_length_m" in trace:
                sl = np.array(trace.get("step_length_m", []))
                td_sl = sl[sl > 0.001]  # touchdown-only
                summary["debug_step_length_m_mean"] = float(np.mean(td_sl)) if len(td_sl) > 0 else 0.0
            # Lateral velocity mean
            if "lateral_velocity" in trace:
                summary["lateral_velocity_mean"] = float(np.mean(trace["lateral_velocity"]))
            # Roll p95
            if "root_roll" in trace:
                roll_abs = np.abs(np.array(trace["root_roll"]))
                summary["debug_loc_ref_root_roll_abs_p95"] = float(np.percentile(roll_abs, 95))
            # Stance switch count
            if "stance_foot" in trace:
                sf = np.array(trace["stance_foot"])
                switches = int(np.sum(np.abs(np.diff(sf)) > 0.5))
                summary["stance_switch_count"] = switches

        # Save per-seed summary
        seed_json = out / f"seed_{seed}.json"
        seed_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        results.append(summary)

    # Evaluate gate
    gate = _evaluate_gate(results, forward_cmd=forward_cmd, horizon=horizon)

    # Save aggregate
    aggregate = {
        "config": config_path,
        "forward_cmd": forward_cmd,
        "horizon": horizon,
        "n_seeds": n_seeds,
        "gate": gate,
    }
    agg_path = out / "handoff_gate.json"
    agg_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8")

    # Print report
    _print_gate_report(gate)

    return aggregate


def _print_gate_report(gate: dict[str, Any]) -> None:
    """Print human-readable gate report."""
    def _pf(passed: bool) -> str:
        return "PASS" if passed else "FAIL"

    print("\n" + "=" * 60)
    print("NOMINAL-TO-PPO HANDOFF GATE EVALUATION")
    print("=" * 60)

    r = gate["robustness"]
    print(f"\n1. Multi-seed robustness [{_pf(r['pass'])}]")
    print(f"   Survive >=400 steps: {r['survive_400_count']}/{gate['n_seeds']} "
          f"({r['survive_400_frac']:.0%}) [need >=80%] {_pf(r['pass_survive_400'])}")
    print(f"   Min survival: {r['min_survival']} [need >=300] {_pf(r['pass_min_300'])}")
    print(f"   Mean survival: {r['mean_survival']:.1f} [need >=430] {_pf(r['pass_mean_430'])}")

    f = gate["forward_velocity"]
    print(f"\n2. Forward velocity [{_pf(f['pass'])}]")
    print(f"   Mean: {f['mean']:.4f} m/s [need >={f['target']:.4f}] "
          f"({f['tracking_ratio']:.0%} of cmd)")

    s = gate["step_generation"]
    print(f"\n3. Step generation [{_pf(s['pass'])}]")
    print(f"   Commanded foothold: {s['mean_foothold_cmd_m']:.4f} m "
          f"[need >=0.045] {_pf(s['pass_cmd_adequate'])}")
    print(f"   Realized step: {s['mean_step_length_m']:.4f} m "
          f"(ratio: {s['step_ratio']:.2f}) {_pf(s['pass_realized'])}")

    st = gate["stability"]
    print(f"\n4. Stability [{_pf(st['pass'])}]")
    print(f"   Pitch p95: {st['mean_pitch_p95']:.3f} rad [<=0.20] {_pf(st['pass_pitch_p95'])}")
    print(f"   Roll p95: {st['mean_roll_p95']:.3f} rad [<=0.12] {_pf(st['pass_roll_p95'])}")
    print(f"   Term pitch frac: {st['term_pitch_frac']:.2f} [<=0.20] {_pf(st['pass_term_pitch'])}")
    print(f"   Term roll frac: {st['term_roll_frac']:.2f} [<=0.20] {_pf(st['pass_term_roll'])}")

    g = gate["gait_quality"]
    print(f"\n5. Gait quality [{_pf(g['pass'])}]")
    print(f"   Passing seeds with switches: {g['all_passing_have_switches']}")
    print(f"   Min switches (passing): {g['min_switches_passing']}")

    d = gate["lateral_drift"]
    print(f"\n6. Lateral drift [{_pf(d['pass'])}]")
    print(f"   Systematic: {d['systematic']} (directions: {d['drift_directions']})")

    print(f"\n{'=' * 60}")
    overall = gate["overall_pass"]
    print(f"OVERALL: {_pf(overall)}")
    print(f"{'=' * 60}")

    if gate.get("per_seed"):
        print("\nPer-seed summary:")
        print(f"{'seed':>5} {'surv':>5} {'fwd_v':>7} {'step_l':>7} {'fh_cmd':>7} "
              f"{'pitch95':>8} {'roll95':>7} {'drift':>6} {'sw':>4} {'term':>12}")
        for ps in gate["per_seed"]:
            print(f"{ps['seed']:5d} {ps['survival_steps']:5d} "
                  f"{ps['forward_vel']:7.4f} {ps['step_length']:7.4f} "
                  f"{ps['foothold_cmd']:7.4f} {ps['pitch_p95']:8.4f} "
                  f"{ps['roll_p95']:7.4f} {ps['drift_dir']:6d} "
                  f"{ps['stance_switches']:4d} {ps['term_type']:>12}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Nominal-to-PPO handoff gate evaluator")
    p.add_argument("--config", type=str, default="training/configs/ppo_walking_v0193a.yaml")
    p.add_argument("--forward-cmd", type=float, default=0.15)
    p.add_argument("--horizon", type=int, default=500)
    p.add_argument("--seeds", type=int, default=5, help="Number of seeds to run")
    p.add_argument("--seed-start", type=int, default=0, help="Starting seed")
    p.add_argument("--output-dir", type=str, default="/tmp/handoff_gate")
    p.add_argument("--step-time", type=float, default=None,
                   help="Override step time for step-amplitude diagnosis")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    aggregate = run_handoff_gate(
        config_path=args.config,
        forward_cmd=args.forward_cmd,
        horizon=args.horizon,
        n_seeds=args.seeds,
        seed_start=args.seed_start,
        output_dir=args.output_dir,
        step_time_override=args.step_time,
    )
    return 0 if aggregate["gate"]["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
