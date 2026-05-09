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
from control.zmp.zmp_walk import ZMPWalkGenerator  # noqa: E402


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
    (
        model, data, mapper, act_to_qpos, geom_ids, body_ids,
        shared_joint_limits, shared_leg_idx,
    ) = _load_wr_assets()
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

        # Slice the 8 shared leg joints by name-resolved column index
        # (post the v20 ankle_roll merge legs sit at non-contiguous
        # indices in the 21-wide PolicySpec order).  shared_joint_limits
        # is already in _WR_SHARED_LEG_NAMES order, so the fancy-index
        # selection produces an 8-vector aligned with the limits and
        # with downstream sat_per_joint / q_err_per_joint consumers
        # that index by joint-name position.
        actual_policy_q = (
            data.qpos[act_to_qpos][mapper.policy_to_mj_order][shared_leg_idx]
            .astype(np.float64)
        )
        q_err = (actual_policy_q - q_ref[shared_leg_idx]).tolist()
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


def _collapse_tap_down_events(
    ref_contact: list[bool],
    phys_contact: list[bool],
    i_swing: list[int],
    n_steps: int,
) -> list[dict]:
    """Collapse contiguous (commanded-swing AND physical-contact) frames
    into distinct tap-down events.

    A "tap-down event" = a maximal contiguous span of consecutive
    steps where ``not ref_contact[i]`` AND ``phys_contact[i]``.
    Counting per-step rather than per-event would overcount: a single
    tap that lasts 3 frames is one physical event, not three.

    Boundary-only events (where every frame in the span has
    ``i_swing in {0, swing_window-1}``) are filtered out because the
    planner commands z=0 at boundary frames and physical contact
    there is expected behaviour, not a tap-down.
    """
    boundary = {0, _SWING_WINDOW - 1}
    events: list[dict] = []
    in_event = False
    span_start = -1
    span_i_swings: list[int] = []
    for i in range(n_steps):
        is_tap = (not ref_contact[i]) and phys_contact[i]
        if is_tap and not in_event:
            in_event = True
            span_start = i
            span_i_swings = [int(i_swing[i])]
        elif is_tap and in_event:
            span_i_swings.append(int(i_swing[i]))
        elif (not is_tap) and in_event:
            in_event = False
            non_boundary = [j for j in span_i_swings if j not in boundary and 0 <= j < _SWING_WINDOW]
            if non_boundary:
                events.append(
                    {
                        "step_start": span_start,
                        "step_end": i - 1,
                        "duration_frames": len(span_i_swings),
                        "i_swing_start": span_i_swings[0],
                        "i_swing_end": span_i_swings[-1],
                        "i_swing_min": min(span_i_swings),
                        "i_swing_max": max(span_i_swings),
                    }
                )
    if in_event:
        non_boundary = [j for j in span_i_swings if j not in boundary and 0 <= j < _SWING_WINDOW]
        if non_boundary:
            events.append(
                {
                    "step_start": span_start,
                    "step_end": n_steps - 1,
                    "duration_frames": len(span_i_swings),
                    "i_swing_start": span_i_swings[0],
                    "i_swing_end": span_i_swings[-1],
                    "i_swing_min": min(span_i_swings),
                    "i_swing_max": max(span_i_swings),
                }
            )
    return events


def _diag_d2_mid_swing_tap_down(trace: PerStepTrace) -> dict:
    """D2: count distinct mid-swing tap-down events per foot.

    A tap-down event = a maximal contiguous span where the planner
    commanded swing (``ref_contact[foot] == 0``) AND the physics
    reported floor contact (``phys_contact[foot] == 1``).  The
    contiguous-span collapsing avoids overcounting a single physical
    tap that spans multiple frames.

    Boundary-only events (entire span at ``i_swing in {0, swing_window-1}``)
    are filtered because the planner commands z=0 there and contact
    is expected.
    """
    n = trace.survived_steps
    left_events = _collapse_tap_down_events(
        trace.contact_left_ref, trace.contact_left_phys, trace.i_swing_left, n
    )
    right_events = _collapse_tap_down_events(
        trace.contact_right_ref, trace.contact_right_phys, trace.i_swing_right, n
    )
    total = len(left_events) + len(right_events)
    n_swing_cycles = max(1, n // _SWING_WINDOW)
    events_per_cycle = float(total / n_swing_cycles)
    # Mean / max event duration as secondary signals — a long tap
    # suggests dragging, a short tap suggests a bounce.
    all_durations = [e["duration_frames"] for e in left_events + right_events]
    mean_dur = float(np.mean(all_durations)) if all_durations else 0.0
    max_dur = int(np.max(all_durations)) if all_durations else 0
    return {
        "left_event_count": len(left_events),
        "right_event_count": len(right_events),
        "total_event_count": total,
        "events_per_swing_cycle": events_per_cycle,
        "mean_event_duration_frames": mean_dur,
        "max_event_duration_frames": max_dur,
        "left_events": left_events[:20],   # first 20 for inspection
        "right_events": right_events[:20],
        "interpretation": (
            # Threshold tuned to corrected metric (distinct events,
            # not per-step contacts).  >= 0.5 distinct events/cycle is
            # a meaningful planner-shape signal; <= 0.3 is the Phase
            # 12A exit criterion.
            "planner-shape-bound" if events_per_cycle >= 0.5
            else "borderline" if events_per_cycle > 0.3
            else "clear"
        ),
    }


def _diag_d3_d4_saturation_by_i_swing(trace: PerStepTrace) -> dict:
    """D3 + D4: saturation distribution across the swing window, broken
    down per joint AND per swing side.

    The earlier "any-joint" reduction proved timing-coincidence between
    the down-cross frame and saturation but did not establish mechanism:
    "any joint" could be a stance-side hip or knee, not the swing-side
    ankle the un-plantarflex hypothesis predicts.  This refactored
    diagnostic reports:

    - per-joint sat rate by ``i_swing`` (8 joints × ``swing_window``)
    - **swing-side ankle** sat rate by ``i_swing`` — the joint whose
      saturation directly supports the un-plantarflex hypothesis
    - swing-side hip-pitch / knee sat rates for comparison
    - any-joint sat rate (kept for backward comparison with the prior
      diagnostic output)

    The "Phase-7-boundary-bound" interpretation now requires the
    **swing-side ankle** to cluster at the down-cross, not just any
    joint.  If any-joint clusters but swing-side ankle does not, the
    interpretation is "down-cross-correlated (mechanism-unclear)" and
    the un-plantarflex hypothesis is *suggestive*, not confirmed.
    """
    n = trace.survived_steps
    n_joints = len(_WR_SHARED_LEG_NAMES)

    # joint_idx_for_swing_side: when LEFT is in swing, the swing-side
    # ankle is left_ankle_pitch (idx 6); same for right.
    swing_side_ankle = {"left": 6, "right": 7}
    swing_side_hip_pitch = {"left": 0, "right": 1}
    swing_side_knee = {"left": 4, "right": 5}

    # Per-joint × i_swing bucketed counts.
    per_joint_count = np.zeros((n_joints, _SWING_WINDOW), dtype=np.int64)
    per_joint_total = np.zeros((n_joints, _SWING_WINDOW), dtype=np.int64)

    # Swing-side-specific (the side that's actually swinging this frame).
    swing_ankle_count = np.zeros(_SWING_WINDOW, dtype=np.int64)
    swing_ankle_total = np.zeros(_SWING_WINDOW, dtype=np.int64)
    swing_hip_pitch_count = np.zeros(_SWING_WINDOW, dtype=np.int64)
    swing_hip_pitch_total = np.zeros(_SWING_WINDOW, dtype=np.int64)
    swing_knee_count = np.zeros(_SWING_WINDOW, dtype=np.int64)
    swing_knee_total = np.zeros(_SWING_WINDOW, dtype=np.int64)

    # Backward-compatible "any-joint" bucket so the regression vs the
    # prior diagnostic is visible.
    any_count = np.zeros(_SWING_WINDOW, dtype=np.int64)
    any_total = np.zeros(_SWING_WINDOW, dtype=np.int64)

    for i in range(n):
        sat = trace.sat_per_joint[i]
        any_sat = any(sat)
        # Iterate per (foot, i_swing) so a frame where both feet are in
        # "swing" (shouldn't happen in proper alternating gait, but
        # defend against it) still gets attributed correctly per side.
        for foot, i_swing_arr in (("left", trace.i_swing_left), ("right", trace.i_swing_right)):
            i_swing = i_swing_arr[i]
            if not (0 <= i_swing < _SWING_WINDOW):
                continue
            # Per-joint × i_swing
            for j in range(n_joints):
                per_joint_total[j, i_swing] += 1
                if sat[j]:
                    per_joint_count[j, i_swing] += 1
            # Swing-side-specific buckets
            ankle_idx = swing_side_ankle[foot]
            hip_idx = swing_side_hip_pitch[foot]
            knee_idx = swing_side_knee[foot]
            swing_ankle_total[i_swing] += 1
            if sat[ankle_idx]:
                swing_ankle_count[i_swing] += 1
            swing_hip_pitch_total[i_swing] += 1
            if sat[hip_idx]:
                swing_hip_pitch_count[i_swing] += 1
            swing_knee_total[i_swing] += 1
            if sat[knee_idx]:
                swing_knee_count[i_swing] += 1
            # any-joint backward compatibility
            any_total[i_swing] += 1
            if any_sat:
                any_count[i_swing] += 1

    def _rate(count: np.ndarray, total: np.ndarray) -> list[float]:
        return (count / np.maximum(total, 1)).tolist()

    swing_ankle_rate = _rate(swing_ankle_count, swing_ankle_total)
    any_rate = _rate(any_count, any_total)
    overall_any = float(any_count.sum() / max(any_total.sum(), 1))
    overall_swing_ankle = float(swing_ankle_count.sum() / max(swing_ankle_total.sum(), 1))

    # Cluster interpretation: the SWING-SIDE ANKLE must cluster at
    # the down-cross (mechanism-supporting) for the verdict to be
    # "Phase-7-boundary-bound".  If any-joint clusters but swing-side
    # ankle doesn't, the verdict is "down-cross-correlated" — timing
    # coincidence without joint-level mechanism evidence.
    swing_ankle_at_up = swing_ankle_rate[_PLANTARFLEX_UP_CROSS_IDX]
    swing_ankle_at_down = swing_ankle_rate[_PLANTARFLEX_DOWN_CROSS_IDX]
    any_at_down = any_rate[_PLANTARFLEX_DOWN_CROSS_IDX]
    any_at_up = any_rate[_PLANTARFLEX_UP_CROSS_IDX]

    swing_ankle_cluster = (
        overall_swing_ankle >= 0.05
        and (
            swing_ankle_at_up >= 1.5 * overall_swing_ankle
            or swing_ankle_at_down >= 1.5 * overall_swing_ankle
        )
    )
    any_cluster = (
        overall_any >= 0.05
        and (any_at_up >= 1.5 * overall_any or any_at_down >= 1.5 * overall_any)
    )

    if swing_ankle_cluster:
        interpretation = "Phase-7-boundary-bound"
    elif any_cluster:
        interpretation = "down-cross-correlated-mechanism-unclear"
    elif overall_any >= 0.05:
        interpretation = "uniform-sat-distribution"
    else:
        interpretation = "no-significant-sat"

    return {
        "swing_z_table_m": _SWING_Z_TABLE.tolist(),
        "plantarflex_up_cross_i_swing": _PLANTARFLEX_UP_CROSS_IDX,
        "plantarflex_down_cross_i_swing": _PLANTARFLEX_DOWN_CROSS_IDX,
        "joint_names": list(_WR_SHARED_LEG_NAMES),
        # Backward-compat: any-joint bucket
        "any_joint_sat_rate_by_i_swing": any_rate,
        "any_joint_sat_count_by_i_swing": any_count.tolist(),
        "any_joint_visits_by_i_swing": any_total.tolist(),
        "swing_window_mean_any_sat_rate": overall_any,
        "any_sat_rate_at_up_cross": any_at_up,
        "any_sat_rate_at_down_cross": any_at_down,
        # Per-joint × i_swing breakdown (8 joints × swing_window)
        "per_joint_sat_rate_by_i_swing": {
            name: _rate(per_joint_count[j], per_joint_total[j])
            for j, name in enumerate(_WR_SHARED_LEG_NAMES)
        },
        # Mechanism-supporting: swing-side ankle (the joint whose
        # un-plantarflex is the hypothesised root cause)
        "swing_side_ankle_sat_rate_by_i_swing": swing_ankle_rate,
        "swing_window_mean_swing_ankle_sat_rate": overall_swing_ankle,
        "swing_side_ankle_sat_at_up_cross": swing_ankle_at_up,
        "swing_side_ankle_sat_at_down_cross": swing_ankle_at_down,
        # Comparison: swing-side hip-pitch / knee
        "swing_side_hip_pitch_sat_rate_by_i_swing": _rate(swing_hip_pitch_count, swing_hip_pitch_total),
        "swing_side_knee_sat_rate_by_i_swing": _rate(swing_knee_count, swing_knee_total),
        "interpretation": interpretation,
    }


def _verdict(d1: dict, d2: dict, d3d4: dict) -> dict:
    """Compose the four sub-verdicts into a single decision.

    Verdict labels:
    - ``cycle_0_bound`` — D1 shows large positive gap (rest >> first)
    - ``planner_shape_bound`` — D2 distinct events ≥ 0.5 / cycle
    - ``phase_7_boundary_bound`` — D3+D4 swing-side ANKLE clusters at
      down-cross (mechanism-supporting evidence)
    - ``down_cross_correlated_mechanism_unclear`` — D3+D4 any-joint
      clusters at down-cross but swing-side ankle does not (timing
      coincidence without joint-level mechanism evidence)
    - ``actuator_stack_bound`` — D1 / D2 clean and D3+D4 sat
      distribution is uniform across the swing window
    """
    d3d4_interp = d3d4["interpretation"]
    components = {
        "cycle_0_bound": d1["interpretation"] == "cycle-0-bound",
        "planner_shape_bound": d2["interpretation"] == "planner-shape-bound",
        "phase_7_boundary_bound": d3d4_interp == "Phase-7-boundary-bound",
        "down_cross_correlated_mechanism_unclear": (
            d3d4_interp == "down-cross-correlated-mechanism-unclear"
        ),
        "actuator_stack_bound": (
            d1["interpretation"] != "cycle-0-bound"
            and d2["interpretation"] != "planner-shape-bound"
            and d3d4_interp in ("uniform-sat-distribution", "no-significant-sat")
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
    print("  D2: distinct mid-swing tap-down events (collapsed contiguous spans)")
    print(f"     left foot distinct events       = {d2['left_event_count']}")
    print(f"     right foot distinct events      = {d2['right_event_count']}")
    print(f"     total distinct events           = {d2['total_event_count']}")
    print(f"     events per swing cycle          = {d2['events_per_swing_cycle']:.3f}")
    print(f"     mean event duration (frames)    = {d2['mean_event_duration_frames']:.2f}")
    print(f"     max event duration (frames)     = {d2['max_event_duration_frames']}")
    print(f"     verdict                         = {d2['interpretation']}")
    d3 = result["d3_d4_saturation_by_i_swing"]
    print()
    print("  D3+D4: actuator saturation by i_swing — per joint AND swing-side")
    print()
    print("    [swing-side ANKLE — mechanism-supporting joint for un-plantarflex hypothesis]")
    print(
        f"     mean sat rate (swing-side ankle) = "
        f"{d3['swing_window_mean_swing_ankle_sat_rate']:.4f}"
    )
    print(
        f"     swing-side ankle at up-cross   (i={d3['plantarflex_up_cross_i_swing']:2d})"
        f" = {d3['swing_side_ankle_sat_at_up_cross']:.4f}"
    )
    print(
        f"     swing-side ankle at down-cross (i={d3['plantarflex_down_cross_i_swing']:2d})"
        f" = {d3['swing_side_ankle_sat_at_down_cross']:.4f}"
    )
    print()
    print("    [any-joint — backward-compat, prior diagnostic's metric]")
    print(
        f"     mean sat rate (any joint)        = "
        f"{d3['swing_window_mean_any_sat_rate']:.4f}"
    )
    print(
        f"     any-joint at up-cross   (i={d3['plantarflex_up_cross_i_swing']:2d})"
        f"     = {d3['any_sat_rate_at_up_cross']:.4f}"
    )
    print(
        f"     any-joint at down-cross (i={d3['plantarflex_down_cross_i_swing']:2d})"
        f"     = {d3['any_sat_rate_at_down_cross']:.4f}"
    )
    print()
    print("    [per-joint sat rate by i_swing — ankle (swing-side picks): see above]")
    swing_ankle_rate = d3["swing_side_ankle_sat_rate_by_i_swing"]
    swing_hip_rate = d3["swing_side_hip_pitch_sat_rate_by_i_swing"]
    swing_knee_rate = d3["swing_side_knee_sat_rate_by_i_swing"]
    any_rate = d3["any_joint_sat_rate_by_i_swing"]
    visits = d3["any_joint_visits_by_i_swing"]
    print(
        "        i_swing | z_cmd(m) | swing-ankle | swing-hip | swing-knee | any-joint | visits"
    )
    for i in range(_SWING_WINDOW):
        marker = ""
        if i == d3["plantarflex_up_cross_i_swing"]:
            marker = " <- up-cross"
        elif i == d3["plantarflex_down_cross_i_swing"]:
            marker = " <- down-cross"
        print(
            f"        {i:7d} | {d3['swing_z_table_m'][i]:8.4f} |"
            f" {swing_ankle_rate[i]:11.4f} | {swing_hip_rate[i]:9.4f} |"
            f" {swing_knee_rate[i]:10.4f} | {any_rate[i]:9.4f} | {visits[i]:6d}"
            f"{marker}"
        )
    print(f"     verdict                         = {d3['interpretation']}")
    print()
    print("    [non-zero per-joint sat rates — ALL 8 shared leg joints]")
    pj = d3["per_joint_sat_rate_by_i_swing"]
    nonzero_joints = [name for name, rates in pj.items() if max(rates) > 0.0]
    if not nonzero_joints:
        print("        (no joint saturated at any i_swing — sat is zero everywhere)")
    else:
        for name in nonzero_joints:
            rates = pj[name]
            peak_i = int(np.argmax(rates))
            peak_v = float(max(rates))
            print(
                f"        {name:24s} peak={peak_v:.4f} at i_swing={peak_i:2d}"
                f" | full: [{', '.join(f'{r:.3f}' for r in rates)}]"
            )
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
        default=[0.15, 0.20, 0.25],
        help="Closed-loop replay vx bins (default: 0.15 0.20 0.25 — Phase 9D bracket around the vx=0.20 operating point, TB-step/leg-matched at the WR-pendulum-scaled cycle_time=0.96 s).",
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

    # Build the reference library inline for the exact requested vx
    # bins.  This bypasses the parity tool's cache-resolver default
    # bin set ({0.10, 0.15, 0.20, 0.25} as of Phase 9D) so the
    # diagnostic can be run at the operating point + neighbours
    # without rebuilding the cached parity-tool library.  When the
    # requested bins are a subset of the parity tool's defaults, the
    # cached library is used; otherwise an inline build runs.
    requested_bins = sorted(round(float(v), 4) for v in args.vx)
    parity_default_bins = sorted([0.10, 0.15, 0.20, 0.25])
    if set(requested_bins) <= set(parity_default_bins):
        lib_args = argparse.Namespace(
            wr_library_path=args.wr_library_path,
            wr_library_rebuild=args.wr_library_rebuild,
            wr_library_no_cache=False,
            wr_library_cache_dir=".cache/wr_reference_libraries",
            vx=0.20,
            nominal_only=False,
        )
        lib, lifecycle = _resolve_wr_reference_library(lib_args)
        print(f"WR library: source={lifecycle.source} cache_key={lifecycle.cache_key}")
    else:
        # Inline build for non-default bins (e.g., shuffling-regime
        # comparison at vx ∈ {0.18, 0.20}).
        print(f"WR library: inline build for non-default bins {requested_bins}")
        lib = ZMPWalkGenerator().build_library_for_vx_values(requested_bins)

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
