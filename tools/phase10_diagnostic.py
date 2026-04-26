#!/usr/bin/env python3
"""Phase 10 diagnostic — explain the WR P1 closed-loop FAIL pattern.

Phase 10 is the diagnostic phase from
``training/docs/reference_architecture_comparison.md``.  It runs the
WR closed-loop replay with per-step trace capture and produces a
verdict on the P1 FAIL root cause among four hypotheses:

- **cycle-0-bound** — WR retains the first-cycle from-rest transient
  while TB truncates ``ceil(cycle_time / control_dt) = 36`` steps.
  Signal: large gap between contact_match on the first cycle vs the
  rest of the episode.
- **planner-shape-bound** — the swing-z envelope still produces
  mid-swing tap-down events.  Signal: the swing foot enters physical
  contact at non-boundary swing frames.
- **Phase-7-boundary-bound** — Phase 7's triangle envelope crosses
  the 0.025 m plantarflex threshold in a single frame between
  ``i_swing=1`` (z=0.013) and ``i_swing=2`` (z=0.026), simultaneously
  with the discrete 0.15 rad ankle plantarflex flip.  Signal: actuator
  saturation events clustered at ``i_swing in {2, 9}``.
- **actuator-stack-bound** — WR position actuators vs TB torque+PD
  produce structurally different per-step joint-error patterns at
  identical reference quality.  Signal: D1 / D2 clean and D4
  saturation distribution is approximately uniform across the swing
  window.

Verdicts inform the next slice:
- cycle-0-bound → mirror TB truncation in WR ``_generate_at_step_length``
  (Phase 12 candidate)
- Phase-7-boundary-bound → soften the plantarflex schedule or shape
  the triangle's first / last 1-2 frames (Phase 12 candidate)
- actuator-stack-bound → activate H6 once the parity tool also ships
  the composite gauge (per H6 activation rule)
- mixed → multiple Phase 12-class follow-ups required

The diagnostic does NOT modify any planner code; it only reads the
post-Phase-7 state.  It runs WR-only, so it does not need the TB
venv (same constraint pattern as ``tools/phase6_wr_probe.py``).

Usage::

    uv run python tools/phase10_diagnostic.py
    uv run python tools/phase10_diagnostic.py --vx 0.15 --horizon 200
    uv run python tools/phase10_diagnostic.py --json-out tools/phase10_diagnostic.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Reuse parity-tool helpers so the diagnostic is byte-for-byte
# consistent with the parity-tool's WR closed-loop replay.  Importing
# only WR-side helpers means the diagnostic does NOT need the TB venv.
from tools.reference_geometry_parity import (  # noqa: E402
    _SAT_MARGIN_RAD,
    _SHARED_PITCH_FAIL_RAD,
    _SHARED_ROLL_FAIL_RAD,
    _WR_ROOT_Z_FAIL_M,
    _WR_SHARED_LEG_NAMES,
    _foot_floor_in_contact,
    _init_wr_standing,
    _load_wr_assets,
    _quat_to_pitch_roll,
    _resolve_wr_reference_library,
    _wr_collision_geoms,
    _wr_foot_center_pos,
    _wr_traj,
)


# Phase 7 swing-z envelope structure.  Mirrors
# ``control/zmp/zmp_walk.py:617-635`` (Phase 7 single-peaked
# triangle).  These constants are used to identify the
# plantarflex-threshold-crossing frames within the swing window so
# D3 / D4 can bucket sat events by ``i_swing``.
_PLANTARFLEX_THRESHOLD_M = 0.025  # control/zmp/zmp_walk.py:611
_FOOT_STEP_HEIGHT_M = 0.04        # control/zmp/zmp_walk.py:95
_SWING_FLOOR_CLEARANCE_M = 0.025  # control/zmp/zmp_walk.py:611
# Triangle parameters at WR cycle (cycle_time=0.72, dt=0.02):
#   T_half = 0.36, n_half = 18, ds = 0.12, n_ds = 6, swing_window = 12
_SWING_WINDOW = 12
_SWING_HALF = _SWING_WINDOW // 2  # = 6
_UP_DELTA = (_FOOT_STEP_HEIGHT_M + _SWING_FLOOR_CLEARANCE_M) / max(1, _SWING_HALF - 1)
_SWING_Z_TABLE = _UP_DELTA * np.concatenate(
    (
        np.arange(_SWING_HALF, dtype=np.float64),
        np.arange(_SWING_WINDOW - _SWING_HALF - 1, -1, -1, dtype=np.float64),
    )
)
# Plantarflex-threshold-crossing frames (where commanded swing-z
# crosses 0.025 m): up-cross at i=2, down-cross at i=9.
_PLANTARFLEX_UP_CROSS_IDX = int(np.argmax(_SWING_Z_TABLE > _PLANTARFLEX_THRESHOLD_M))  # 2
# down-cross is the last index where swing-z > threshold
_PLANTARFLEX_DOWN_CROSS_IDX = int(
    len(_SWING_Z_TABLE) - 1 - np.argmax(_SWING_Z_TABLE[::-1] > _PLANTARFLEX_THRESHOLD_M)
)  # 9

# TB cycle-0 truncation length (toddlerbot/algorithms/zmp_walk.py:138).
# WR retains this; D1 splits on this boundary.
_FIRST_CYCLE_STEPS = 36  # ceil(cycle_time=0.72 / control_dt=0.02)


@dataclass
class PerStepTrace:
    """Per-step capture from a single WR closed-loop replay."""

    step_idx: list[int] = field(default_factory=list)
    contact_left_phys: list[bool] = field(default_factory=list)
    contact_right_phys: list[bool] = field(default_factory=list)
    contact_left_ref: list[bool] = field(default_factory=list)
    contact_right_ref: list[bool] = field(default_factory=list)
    foot_z_left_phys_m: list[float] = field(default_factory=list)
    foot_z_right_phys_m: list[float] = field(default_factory=list)
    i_swing_left: list[int] = field(default_factory=list)  # -1 if in stance
    i_swing_right: list[int] = field(default_factory=list)
    sat_per_joint: list[list[bool]] = field(default_factory=list)
    q_err_per_joint_rad: list[list[float]] = field(default_factory=list)
    pitch_rad: list[float] = field(default_factory=list)
    roll_rad: list[float] = field(default_factory=list)
    root_z_m: list[float] = field(default_factory=list)
    termination_mode: str = "ok"
    survived_steps: int = 0
    horizon: int = 0


def _capture_replay(lib, vx: float, horizon: int) -> PerStepTrace:
    """Run the WR closed-loop replay with per-step trace capture.

    Mirrors ``tools/reference_geometry_parity.py::_closed_loop_wildrobot``
    so the trace is byte-for-byte the same trajectory the parity tool
    sees, with extra per-step capture.
    """
    model, data, mapper, act_to_qpos, geom_ids, body_ids, shared_joint_limits = _load_wr_assets()
    traj = _wr_traj(lib, vx)
    horizon = min(int(horizon), int(traj.n_steps))
    _init_wr_standing(model, data)
    physics_steps_per_ref = max(1, int(round(float(traj.dt) / float(model.opt.timestep))))
    standing_q = data.qpos[act_to_qpos][mapper.policy_to_mj_order].astype(np.float64)
    first_q = np.asarray(traj.q_ref[0], dtype=np.float64)
    for ramp_idx in range(50):
        alpha = (ramp_idx + 1) / 50.0
        blended = standing_q * (1.0 - alpha) + first_q * alpha
        mapper.set_all_ctrl(data, blended)
        for _ in range(physics_steps_per_ref):
            mujoco.mj_step(model, data)

    left_contact_geoms = _wr_collision_geoms(model, body_ids["left_foot"])
    right_contact_geoms = _wr_collision_geoms(model, body_ids["right_foot"])
    trace = PerStepTrace(horizon=horizon)
    i_swing_l = -1
    i_swing_r = -1
    for step_idx in range(horizon):
        idx = min(step_idx, traj.n_steps - 1)
        q_ref = np.asarray(traj.q_ref[idx], dtype=np.float64)
        mapper.set_all_ctrl(data, q_ref)
        for _ in range(physics_steps_per_ref):
            mujoco.mj_step(model, data)

        actual_policy_q = data.qpos[act_to_qpos][mapper.policy_to_mj_order][:8].astype(np.float64)
        q_err = (actual_policy_q - q_ref[:8]).tolist()
        sat_lo = actual_policy_q <= shared_joint_limits[:, 0] + _SAT_MARGIN_RAD
        sat_hi = actual_policy_q >= shared_joint_limits[:, 1] - _SAT_MARGIN_RAD
        sat_flags = (sat_lo | sat_hi).tolist()

        left_in = _foot_floor_in_contact(data, geom_ids["floor"], left_contact_geoms)
        right_in = _foot_floor_in_contact(data, geom_ids["floor"], right_contact_geoms)
        ref_contact = np.asarray(traj.contact_mask[idx], dtype=np.float64) > 0.5
        ref_left = bool(ref_contact[0])
        ref_right = bool(ref_contact[1])

        # Track i_swing per foot: increments while the planner says
        # "this foot is in swing", resets to -1 the moment the planner
        # commands stance again.
        i_swing_l = i_swing_l + 1 if not ref_left else -1
        i_swing_r = i_swing_r + 1 if not ref_right else -1

        left_center, right_center = _wr_foot_center_pos(data, geom_ids)
        # Foot z is reported as the geom-pos z of the heel-toe midpoint.
        # This is consistent with the parity tool's foot-z usage.
        trace.foot_z_left_phys_m.append(float(left_center[2]))
        trace.foot_z_right_phys_m.append(float(right_center[2]))
        trace.contact_left_phys.append(bool(left_in))
        trace.contact_right_phys.append(bool(right_in))
        trace.contact_left_ref.append(ref_left)
        trace.contact_right_ref.append(ref_right)
        trace.i_swing_left.append(int(i_swing_l))
        trace.i_swing_right.append(int(i_swing_r))
        trace.sat_per_joint.append(sat_flags)
        trace.q_err_per_joint_rad.append(q_err)
        trace.step_idx.append(int(step_idx))

        pitch, roll = _quat_to_pitch_roll(np.asarray(data.qpos[3:7], dtype=np.float64))
        trace.pitch_rad.append(float(pitch))
        trace.roll_rad.append(float(roll))
        trace.root_z_m.append(float(data.qpos[2]))
        trace.survived_steps = step_idx + 1
        if abs(pitch) > _SHARED_PITCH_FAIL_RAD:
            trace.termination_mode = "pitch"
            break
        if abs(roll) > _SHARED_ROLL_FAIL_RAD:
            trace.termination_mode = "roll"
            break
        if float(data.qpos[2]) < _WR_ROOT_Z_FAIL_M:
            trace.termination_mode = "height"
            break

    return trace


def _diag_d1_contact_match_split(trace: PerStepTrace) -> dict:
    """D1: contact_match split into [first cycle] vs [steps after]."""
    n = trace.survived_steps
    matches = np.array(
        [
            (trace.contact_left_phys[i] == trace.contact_left_ref[i])
            and (trace.contact_right_phys[i] == trace.contact_right_ref[i])
            for i in range(n)
        ],
        dtype=np.float64,
    )
    overall = float(matches.mean()) if n else float("nan")
    if n <= _FIRST_CYCLE_STEPS:
        first = float(matches.mean()) if n else float("nan")
        rest = float("nan")
    else:
        first = float(matches[:_FIRST_CYCLE_STEPS].mean())
        rest = float(matches[_FIRST_CYCLE_STEPS:].mean())
    gap = (rest - first) if not np.isnan(rest) else float("nan")
    return {
        "overall_match_frac": overall,
        "first_cycle_match_frac": first,
        "rest_match_frac": rest,
        "rest_minus_first_gap": gap,
        "first_cycle_step_count": min(n, _FIRST_CYCLE_STEPS),
        "rest_step_count": max(0, n - _FIRST_CYCLE_STEPS),
        "interpretation": (
            "cycle-0-bound" if (not np.isnan(gap)) and gap >= 0.10
            else "no-cycle-0-drag" if not np.isnan(gap)
            else "insufficient-data"
        ),
    }


def _diag_d2_mid_swing_tap_down(trace: PerStepTrace) -> dict:
    """D2: count mid-swing tap-down events per foot.

    A tap-down event = ``ref_contact[foot] == 0`` (commanded swing)
    AND ``phys_contact[foot] == 1`` (foot dipped into floor).  Boundary
    swing frames (``i_swing in {0, swing_window-1}``) are excluded
    from the "mid-swing" count because the planner intentionally
    commands z=0 there; only frames with i_swing in [1, swing_window-2]
    count as a mid-swing tap-down.
    """
    n = trace.survived_steps
    events = {"left": [], "right": []}
    boundary = {0, _SWING_WINDOW - 1}
    for i in range(n):
        for foot, ref_arr, phys_arr, i_swing_arr in [
            ("left", trace.contact_left_ref, trace.contact_left_phys, trace.i_swing_left),
            ("right", trace.contact_right_ref, trace.contact_right_phys, trace.i_swing_right),
        ]:
            if (not ref_arr[i]) and phys_arr[i]:
                i_swing = i_swing_arr[i]
                if 0 <= i_swing < _SWING_WINDOW and i_swing not in boundary:
                    events[foot].append({"step_idx": i, "i_swing": int(i_swing)})
    total = len(events["left"]) + len(events["right"])
    # Express as events per swing cycle (rough denominator).
    n_swing_cycles = max(1, n // _SWING_WINDOW)
    return {
        "left_event_count": len(events["left"]),
        "right_event_count": len(events["right"]),
        "total_event_count": total,
        "events_per_swing_cycle": float(total / n_swing_cycles),
        "left_events": events["left"][:20],   # first 20 for inspection
        "right_events": events["right"][:20],
        "interpretation": (
            "planner-shape-bound" if total >= n_swing_cycles  # ≥1 event/cycle
            else "no-mid-swing-tap-down"
        ),
    }


def _diag_d3_d4_saturation_by_i_swing(trace: PerStepTrace) -> dict:
    """D3 + D4: saturation event distribution across the swing window.

    Buckets ``any-joint-saturated`` events by ``i_swing`` index
    (combined across both feet — the question is whether the
    plantarflex-crossing frames specifically push the actuator stack).
    """
    n = trace.survived_steps
    bucket_count = np.zeros(_SWING_WINDOW, dtype=np.int64)
    bucket_total = np.zeros(_SWING_WINDOW, dtype=np.int64)
    sat_at_threshold_up = 0
    sat_at_threshold_down = 0
    threshold_up_visits = 0
    threshold_down_visits = 0
    for i in range(n):
        any_sat = any(trace.sat_per_joint[i])
        for i_swing in (trace.i_swing_left[i], trace.i_swing_right[i]):
            if 0 <= i_swing < _SWING_WINDOW:
                bucket_total[i_swing] += 1
                if any_sat:
                    bucket_count[i_swing] += 1
                if i_swing == _PLANTARFLEX_UP_CROSS_IDX:
                    threshold_up_visits += 1
                    if any_sat:
                        sat_at_threshold_up += 1
                if i_swing == _PLANTARFLEX_DOWN_CROSS_IDX:
                    threshold_down_visits += 1
                    if any_sat:
                        sat_at_threshold_down += 1
    rate_per_idx = (bucket_count / np.maximum(bucket_total, 1)).tolist()
    overall_rate = float(bucket_count.sum() / max(bucket_total.sum(), 1))
    threshold_up_rate = float(sat_at_threshold_up / max(threshold_up_visits, 1))
    threshold_down_rate = float(sat_at_threshold_down / max(threshold_down_visits, 1))
    # "Cluster" interpretation: threshold-frame sat rates are >= 1.5x
    # the swing-window mean AND the swing-window mean is >= 0.05.
    cluster = (
        overall_rate >= 0.05
        and (threshold_up_rate >= 1.5 * overall_rate or threshold_down_rate >= 1.5 * overall_rate)
    )
    return {
        "swing_z_table_m": _SWING_Z_TABLE.tolist(),
        "plantarflex_up_cross_i_swing": _PLANTARFLEX_UP_CROSS_IDX,
        "plantarflex_down_cross_i_swing": _PLANTARFLEX_DOWN_CROSS_IDX,
        "sat_rate_by_i_swing": rate_per_idx,
        "sat_count_by_i_swing": bucket_count.tolist(),
        "visits_by_i_swing": bucket_total.tolist(),
        "swing_window_mean_sat_rate": overall_rate,
        "sat_rate_at_up_cross": threshold_up_rate,
        "sat_rate_at_down_cross": threshold_down_rate,
        "interpretation": (
            "Phase-7-boundary-bound" if cluster
            else "uniform-sat-distribution" if overall_rate >= 0.05
            else "no-significant-sat"
        ),
    }


def _verdict(d1: dict, d2: dict, d3d4: dict) -> dict:
    """Compose the four sub-verdicts into a single decision."""
    components = {
        "cycle_0_bound": d1["interpretation"] == "cycle-0-bound",
        "planner_shape_bound": d2["interpretation"] == "planner-shape-bound",
        "phase_7_boundary_bound": d3d4["interpretation"] == "Phase-7-boundary-bound",
        "actuator_stack_bound": (
            d1["interpretation"] != "cycle-0-bound"
            and d2["interpretation"] != "planner-shape-bound"
            and d3d4["interpretation"] in ("uniform-sat-distribution", "no-significant-sat")
        ),
    }
    active = [k for k, v in components.items() if v]
    if not active:
        verdict = "inconclusive"
    elif len(active) == 1:
        verdict = active[0]
    else:
        verdict = "mixed: " + " + ".join(active)
    return {"verdict": verdict, "components": components}


def _summarize_one_vx(lib, vx: float, horizon: int) -> dict:
    trace = _capture_replay(lib, vx, horizon)
    d1 = _diag_d1_contact_match_split(trace)
    d2 = _diag_d2_mid_swing_tap_down(trace)
    d3d4 = _diag_d3_d4_saturation_by_i_swing(trace)
    return {
        "vx": vx,
        "horizon": trace.horizon,
        "survived_steps": trace.survived_steps,
        "termination_mode": trace.termination_mode,
        "d1_contact_match_split": d1,
        "d2_mid_swing_tap_down": d2,
        "d3_d4_saturation_by_i_swing": d3d4,
        "verdict": _verdict(d1, d2, d3d4),
    }


def _print_human(result: dict) -> None:
    print()
    print(f"=== Phase 10 diagnostic — vx={result['vx']:.2f} ===")
    print(
        f"  survived steps   = {result['survived_steps']}/{result['horizon']}"
        f"  (termination={result['termination_mode']})"
    )
    d1 = result["d1_contact_match_split"]
    print()
    print("  D1: ref_contact_match_frac split (cycle 0 retention)")
    print(f"     overall                         = {d1['overall_match_frac']:.4f}")
    print(
        f"     first cycle ({d1['first_cycle_step_count']} steps)"
        f"          = {d1['first_cycle_match_frac']:.4f}"
    )
    print(
        f"     rest ({d1['rest_step_count']} steps)              "
        f"= {d1['rest_match_frac']:.4f}"
    )
    print(f"     rest - first_cycle gap          = {d1['rest_minus_first_gap']:+.4f}")
    print(f"     verdict                         = {d1['interpretation']}")
    d2 = result["d2_mid_swing_tap_down"]
    print()
    print("  D2: mid-swing tap-down events (planner-shape regression)")
    print(f"     left foot tap-down events       = {d2['left_event_count']}")
    print(f"     right foot tap-down events      = {d2['right_event_count']}")
    print(f"     total                           = {d2['total_event_count']}")
    print(f"     per swing cycle                 = {d2['events_per_swing_cycle']:.3f}")
    print(f"     verdict                         = {d2['interpretation']}")
    d3 = result["d3_d4_saturation_by_i_swing"]
    print()
    print("  D3+D4: actuator saturation by i_swing (Phase-7-boundary check)")
    print(f"     swing-window mean sat rate      = {d3['swing_window_mean_sat_rate']:.4f}")
    print(
        f"     sat rate at up-cross (i={d3['plantarflex_up_cross_i_swing']})    "
        f"     = {d3['sat_rate_at_up_cross']:.4f}"
    )
    print(
        f"     sat rate at down-cross (i={d3['plantarflex_down_cross_i_swing']})  "
        f"     = {d3['sat_rate_at_down_cross']:.4f}"
    )
    print(f"     sat rate by i_swing             =")
    for i, rate in enumerate(d3["sat_rate_by_i_swing"]):
        marker = ""
        if i == d3["plantarflex_up_cross_i_swing"]:
            marker = "  <- plantarflex up-cross"
        elif i == d3["plantarflex_down_cross_i_swing"]:
            marker = "  <- plantarflex down-cross"
        print(
            f"        i_swing={i:2d}  z_cmd={d3['swing_z_table_m'][i]:.4f}m"
            f"  sat_rate={rate:.4f}  visits={d3['visits_by_i_swing'][i]}"
            f"{marker}"
        )
    print(f"     verdict                         = {d3['interpretation']}")
    v = result["verdict"]
    print()
    print(f"  → composite verdict: {v['verdict']}")
    for component, active in v["components"].items():
        flag = "ACTIVE" if active else "  -   "
        print(f"     [{flag}] {component}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--vx",
        nargs="+",
        type=float,
        default=[0.10, 0.15, 0.20],
        help="Closed-loop replay vx bins (default: 0.10 0.15 0.20).",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=200,
        help="Closed-loop replay horizon in control steps (default: 200).",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default="tools/phase10_diagnostic.json",
        help="Path to write the per-vx diagnostic JSON.",
    )
    parser.add_argument(
        "--wr-library-path",
        type=str,
        default=None,
        help="Optional pinned WR reference library directory.",
    )
    parser.add_argument(
        "--wr-library-rebuild",
        action="store_true",
        help="Force-rebuild the WR reference library cache.",
    )
    args = parser.parse_args()

    # Fingerprint reference library so we use the same WR library the
    # parity tool would use.  Phase 5 cache picks up the Phase 7
    # planner change automatically.
    lib_args = argparse.Namespace(
        wr_library_path=args.wr_library_path,
        wr_library_rebuild=args.wr_library_rebuild,
        wr_library_no_cache=False,
        wr_library_cache_dir=".cache/wr_reference_libraries",
        vx=0.15,
        nominal_only=False,
    )
    lib, lifecycle = _resolve_wr_reference_library(lib_args)
    print(f"WR library: source={lifecycle.source} cache_key={lifecycle.cache_key}")

    payload = {
        "swing_z_table_m": _SWING_Z_TABLE.tolist(),
        "plantarflex_threshold_m": _PLANTARFLEX_THRESHOLD_M,
        "first_cycle_steps": _FIRST_CYCLE_STEPS,
        "vx_bins": list(args.vx),
        "horizon": args.horizon,
        "results": [],
    }
    for vx in args.vx:
        print(f"\n  Running closed-loop replay at vx=+{vx:.3f} m/s ...")
        result = _summarize_one_vx(lib, vx, args.horizon)
        _print_human(result)
        payload["results"].append(result)

    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
