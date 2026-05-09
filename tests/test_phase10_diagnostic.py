"""Unit tests for tools/phase10_diagnostic.py metric definitions.

These tests anchor the corrected D2 (distinct-event collapsing) and
D3+D4 (per-joint + swing-side attribution) metrics so a future
refactor cannot silently regress them.

The tests build small synthetic ``PerStepTrace`` instances with known
contact / sat patterns and verify the metric values directly.  They
do NOT spin up MuJoCo or the reference library — that is what the
end-to-end ``tools/phase10_diagnostic.py`` run does.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.phase10_diagnostic import (  # noqa: E402
    _FIRST_CYCLE_STEPS,
    _PLANTARFLEX_DOWN_CROSS_IDX,
    _PLANTARFLEX_UP_CROSS_IDX,
    _SWING_WINDOW,
    PerStepTrace,
    _collapse_tap_down_events,
    _diag_d1_contact_match_split,
    _diag_d2_mid_swing_tap_down,
    _diag_d3_d4_saturation_by_i_swing,
    _verdict,
)


# Sanity-check the WR-cycle constants the diagnostic depends on.
# Phase 9D-revisited (2026-05-09): cycle_time scaled 0.72 → 0.96; the
# constants are now derived from the live ZMPWalkConfig, so this test
# pins the post-9D values.  Pre-9D values (swing_window=12, up=2,
# down=9) are recoverable by setting ZMPWalkConfig.cycle_time_s = 0.72.
def test_swing_window_constants_match_wr_cycle():
    # WR cycle (Phase 9D): cycle_time=0.96, dt=0.02, single_double_ratio=2.0
    # -> T_single = cycle/2 * 2/3 = 0.32 s -> swing_window = 16.
    # Triangle: foot_step_height_m = 0.045; total apex = 0.045 + 0.025
    # clearance = 0.07; up-delta = 0.07 / (8 - 1) = 0.01; threshold-cross
    # at z > 0.025 -> first index = 3 (z=0.030), last = 12 (z=0.030).
    assert _SWING_WINDOW == 16
    assert _PLANTARFLEX_UP_CROSS_IDX == 3
    assert _PLANTARFLEX_DOWN_CROSS_IDX == 12


# ---------------------------------------------------------------------
# _collapse_tap_down_events — the core D2 fix
# ---------------------------------------------------------------------


def _build_swing_window_pattern(
    contiguous_run_lengths: list[int],
    gaps: list[int],
) -> tuple[list[bool], list[bool], list[int]]:
    """Build a synthetic single-foot trace where the foot is always in
    commanded swing (ref_contact=False) and physical contact toggles
    according to the alternating run / gap pattern.

    ``run_lengths`` and ``gaps`` interleave: run, gap, run, gap, ...
    starting with a run.  i_swing increments uniformly from 0.

    Used to construct known-cardinality tap-down spans for the
    collapse function.
    """
    ref: list[bool] = []
    phys: list[bool] = []
    i_swing: list[int] = []
    idx = 0
    for k, length in enumerate(contiguous_run_lengths):
        # run of physical contact
        for _ in range(length):
            ref.append(False)  # commanded swing
            phys.append(True)  # physical contact => tap-down frame
            i_swing.append(min(idx, _SWING_WINDOW - 1))
            idx += 1
        if k < len(gaps):
            for _ in range(gaps[k]):
                ref.append(False)
                phys.append(False)  # no contact => not a tap-down frame
                i_swing.append(min(idx, _SWING_WINDOW - 1))
                idx += 1
    return ref, phys, i_swing


def test_collapse_single_long_span_is_one_event():
    # 5-frame run of contact across i_swing 1..5 (all mid-swing) -> 1 event
    ref, phys, i_swing = _build_swing_window_pattern([5], [])
    # shift i_swing so the run is at indices 1..5 (mid-swing)
    i_swing = [1, 2, 3, 4, 5]
    events = _collapse_tap_down_events(ref, phys, i_swing, len(ref))
    assert len(events) == 1
    assert events[0]["duration_frames"] == 5
    assert events[0]["i_swing_start"] == 1
    assert events[0]["i_swing_end"] == 5


def test_collapse_two_separated_spans_are_two_events():
    # Two contact runs (3 frames + 2 frames) separated by a non-contact gap
    ref = [False] * 8
    phys = [True, True, True, False, False, True, True, False]
    i_swing = [1, 2, 3, 4, 5, 6, 7, 8]
    events = _collapse_tap_down_events(ref, phys, i_swing, 8)
    assert len(events) == 2
    assert events[0]["duration_frames"] == 3
    assert events[1]["duration_frames"] == 2


def test_collapse_excludes_boundary_only_spans():
    # A span that only touches i_swing=0 or i_swing=swing_window-1 is
    # boundary-only and should NOT count (planner intentionally
    # commands z=0 there).
    ref = [False, False, False]
    phys = [True, True, True]
    i_swing = [0, _SWING_WINDOW - 1, 0]  # all boundary frames
    events = _collapse_tap_down_events(ref, phys, i_swing, 3)
    assert events == []


def test_collapse_keeps_span_that_extends_into_mid_swing():
    # A span that starts at boundary but extends into mid-swing IS
    # a real tap-down event.
    ref = [False] * 4
    phys = [True, True, True, True]
    i_swing = [0, 1, 2, 3]  # touches boundary at 0, but 1,2,3 are mid-swing
    events = _collapse_tap_down_events(ref, phys, i_swing, 4)
    assert len(events) == 1
    assert events[0]["duration_frames"] == 4


def test_collapse_open_span_at_end_of_trace_is_closed_correctly():
    # If the trace ends mid-tap-down, the open span should still be
    # recorded as one event.
    ref = [False] * 5
    phys = [False, True, True, True, True]
    i_swing = [1, 2, 3, 4, 5]
    events = _collapse_tap_down_events(ref, phys, i_swing, 5)
    assert len(events) == 1
    assert events[0]["duration_frames"] == 4
    assert events[0]["step_start"] == 1
    assert events[0]["step_end"] == 4


def test_collapse_ignores_stance_frames():
    # ref_contact == True (stance) should never produce a tap-down
    # event regardless of phys_contact value.
    ref = [True, True, True]
    phys = [True, True, True]
    i_swing = [-1, -1, -1]
    events = _collapse_tap_down_events(ref, phys, i_swing, 3)
    assert events == []


def test_collapse_ignores_swing_frames_without_contact():
    # ref_contact == False (swing), phys_contact == False -> no tap-down
    ref = [False, False, False]
    phys = [False, False, False]
    i_swing = [1, 2, 3]
    events = _collapse_tap_down_events(ref, phys, i_swing, 3)
    assert events == []


# ---------------------------------------------------------------------
# D2 metric (uses _collapse_tap_down_events)
# ---------------------------------------------------------------------


def _make_trace(n_steps: int = 24) -> PerStepTrace:
    """Empty PerStepTrace with n_steps stance frames (no contact, no
    saturation) to be filled in by individual tests.
    """
    trace = PerStepTrace()
    trace.survived_steps = n_steps
    trace.horizon = n_steps
    for i in range(n_steps):
        trace.contact_left_phys.append(False)
        trace.contact_right_phys.append(False)
        trace.contact_left_ref.append(True)  # stance
        trace.contact_right_ref.append(True)
        trace.foot_z_left_phys_m.append(0.0)
        trace.foot_z_right_phys_m.append(0.0)
        trace.i_swing_left.append(-1)
        trace.i_swing_right.append(-1)
        trace.sat_per_joint.append([False] * 8)
        trace.q_err_per_joint_rad.append([0.0] * 8)
        trace.pitch_rad.append(0.0)
        trace.roll_rad.append(0.0)
        trace.root_z_m.append(0.4)
        trace.step_idx.append(i)
    return trace


def test_d2_per_step_overcount_bug_is_fixed():
    # OLD bug: a 4-frame contiguous tap-down was reported as 4 events.
    # NEW: it should be 1 distinct event.
    trace = _make_trace(n_steps=12)
    # Configure left foot in commanded swing for all 12 frames, with
    # physical contact in frames 2..5 (4-frame contiguous tap, all
    # mid-swing).
    for i in range(12):
        trace.contact_left_ref[i] = False
        trace.i_swing_left[i] = i
    for i in (2, 3, 4, 5):
        trace.contact_left_phys[i] = True
    d2 = _diag_d2_mid_swing_tap_down(trace)
    assert d2["left_event_count"] == 1, "4-frame contiguous tap should be 1 event"
    assert d2["right_event_count"] == 0
    assert d2["total_event_count"] == 1
    assert d2["mean_event_duration_frames"] == pytest.approx(4.0)
    assert d2["max_event_duration_frames"] == 4


def test_d2_two_taps_in_one_swing_count_as_two():
    # If the foot taps, lifts off, taps again — that's two events.
    trace = _make_trace(n_steps=12)
    for i in range(12):
        trace.contact_left_ref[i] = False
        trace.i_swing_left[i] = i
    # Two separated taps: frames 2..3, then 7..8
    for i in (2, 3, 7, 8):
        trace.contact_left_phys[i] = True
    d2 = _diag_d2_mid_swing_tap_down(trace)
    assert d2["left_event_count"] == 2
    assert d2["mean_event_duration_frames"] == pytest.approx(2.0)


def test_d2_clear_verdict_with_no_taps():
    trace = _make_trace(n_steps=24)
    d2 = _diag_d2_mid_swing_tap_down(trace)
    assert d2["total_event_count"] == 0
    assert d2["interpretation"] == "clear"


def test_d2_planner_shape_bound_threshold():
    # 2 swing cycles, 1 distinct event each -> 1.0 events/cycle ->
    # planner-shape-bound (>= 0.5)
    trace = _make_trace(n_steps=24)
    for i in range(24):
        trace.contact_left_ref[i] = False
        trace.i_swing_left[i] = i % 12
    # One tap per cycle (mid-swing)
    for i in (3, 15):
        trace.contact_left_phys[i] = True
    d2 = _diag_d2_mid_swing_tap_down(trace)
    assert d2["events_per_swing_cycle"] >= 0.5
    assert d2["interpretation"] == "planner-shape-bound"


# ---------------------------------------------------------------------
# D3+D4 — per-joint + swing-side attribution
# ---------------------------------------------------------------------


def test_d3d4_swing_side_ankle_only_uses_swinging_foot():
    """When LEFT is in swing, the swing-side ankle is left_ankle_pitch
    (joint idx 6).  Saturation on right_ankle (idx 7) should NOT be
    attributed to the swing-side ankle bucket while left is swinging.
    """
    trace = _make_trace(n_steps=12)
    # All 12 frames: LEFT in swing (i_swing=0..11), RIGHT in stance
    for i in range(12):
        trace.contact_left_ref[i] = False
        trace.i_swing_left[i] = i
    # Saturate ONLY right_ankle (idx 7) at every frame
    for i in range(12):
        sat = [False] * 8
        sat[7] = True
        trace.sat_per_joint[i] = sat
    d = _diag_d3_d4_saturation_by_i_swing(trace)
    # Swing-side ankle should be 0.0 everywhere (we only saturated the
    # NON-swing side)
    assert d["swing_window_mean_swing_ankle_sat_rate"] == pytest.approx(0.0)
    # Any-joint should be 1.0 everywhere (right_ankle counts as "any")
    assert d["swing_window_mean_any_sat_rate"] == pytest.approx(1.0)
    # The per-joint breakdown should show right_ankle_pitch saturated
    assert d["per_joint_sat_rate_by_i_swing"]["right_ankle_pitch"][3] == pytest.approx(1.0)
    assert d["per_joint_sat_rate_by_i_swing"]["left_ankle_pitch"][3] == pytest.approx(0.0)


def test_d3d4_swing_side_ankle_attribution_when_correct_side_saturates():
    """When LEFT is in swing AND left_ankle (idx 6) saturates, the
    swing-side ankle bucket should fire.
    """
    # Trace length is _SWING_WINDOW so the down-cross frame fits.  This
    # makes the test cycle-independent: pre-9D (swing_window=12) needs
    # 12 frames; post-9D (swing_window=16) needs 16.
    trace = _make_trace(n_steps=_SWING_WINDOW)
    for i in range(_SWING_WINDOW):
        trace.contact_left_ref[i] = False
        trace.i_swing_left[i] = i
    # Saturate left_ankle (idx 6) only at i_swing = down-cross
    for i in range(_SWING_WINDOW):
        sat = [False] * 8
        if i == _PLANTARFLEX_DOWN_CROSS_IDX:
            sat[6] = True
        trace.sat_per_joint[i] = sat
    d = _diag_d3_d4_saturation_by_i_swing(trace)
    assert d["swing_side_ankle_sat_at_down_cross"] == pytest.approx(1.0)
    assert d["swing_side_ankle_sat_at_up_cross"] == pytest.approx(0.0)


def test_d3d4_phase_7_boundary_bound_requires_swing_side_ankle():
    """Verdict ``Phase-7-boundary-bound`` should require the SWING-SIDE
    ankle to cluster at the down-cross, not just any joint.
    """
    # Two full swing windows so the i_swing wrap-around test is meaningful.
    n_steps = 2 * _SWING_WINDOW
    trace = _make_trace(n_steps=n_steps)
    for i in range(n_steps):
        trace.contact_left_ref[i] = False
        trace.i_swing_left[i] = i % _SWING_WINDOW
    # Saturate hip_pitch (idx 0 / 1, NOT ankle) at every down-cross
    # frame.  Should produce any-joint clustering at down-cross but
    # NOT swing-side-ankle clustering.
    for i in range(n_steps):
        sat = [False] * 8
        if (i % _SWING_WINDOW) == _PLANTARFLEX_DOWN_CROSS_IDX:
            sat[0] = True  # left_hip_pitch
        trace.sat_per_joint[i] = sat
    d = _diag_d3_d4_saturation_by_i_swing(trace)
    # Swing-side ankle is NOT clustered (we never saturated the ankle).
    assert d["swing_side_ankle_sat_at_down_cross"] == pytest.approx(0.0)
    # Any-joint IS clustered (we saturated hip_pitch at down-cross).
    assert d["any_sat_rate_at_down_cross"] > d["swing_window_mean_any_sat_rate"]
    # Verdict should NOT be Phase-7-boundary-bound (mechanism unclear).
    assert d["interpretation"] == "down-cross-correlated-mechanism-unclear"


def test_d3d4_uniform_sat_when_no_clustering():
    """If saturation is spread uniformly across the swing window, the
    verdict should be ``uniform-sat-distribution``.
    """
    trace = _make_trace(n_steps=24)
    for i in range(24):
        trace.contact_left_ref[i] = False
        trace.i_swing_left[i] = i % 12
    # Saturate ankle at every frame (uniform across swing)
    for i in range(24):
        sat = [False] * 8
        sat[6] = True  # left ankle
        trace.sat_per_joint[i] = sat
    d = _diag_d3_d4_saturation_by_i_swing(trace)
    # Swing-side-ankle rate is 1.0 EVERYWHERE; up/down cross rates
    # both equal the mean -> not a cluster.
    assert d["interpretation"] == "uniform-sat-distribution"


def test_d3d4_no_significant_sat():
    """If overall sat rate < 0.05, verdict is ``no-significant-sat``."""
    trace = _make_trace(n_steps=24)
    for i in range(24):
        trace.contact_left_ref[i] = False
        trace.i_swing_left[i] = i % 12
    d = _diag_d3_d4_saturation_by_i_swing(trace)
    assert d["interpretation"] == "no-significant-sat"


# ---------------------------------------------------------------------
# D1 — contact match split
# ---------------------------------------------------------------------


def test_d1_no_drag_when_first_better_than_rest():
    """If the first cycle is BETTER than the rest, the gap is negative
    and the verdict is ``no-cycle-0-drag`` (rules out the cycle-0
    truncation hypothesis).
    """
    n = _FIRST_CYCLE_STEPS + 30  # need enough rest steps
    trace = _make_trace(n_steps=n)
    # First-cycle steps: all matched (ref==phys)
    for i in range(_FIRST_CYCLE_STEPS):
        trace.contact_left_ref[i] = True
        trace.contact_left_phys[i] = True
        trace.contact_right_ref[i] = True
        trace.contact_right_phys[i] = True
    # Rest: contact mismatch on every step
    for i in range(_FIRST_CYCLE_STEPS, n):
        trace.contact_left_ref[i] = True
        trace.contact_left_phys[i] = False  # mismatch
        trace.contact_right_ref[i] = True
        trace.contact_right_phys[i] = True
    d1 = _diag_d1_contact_match_split(trace)
    assert d1["first_cycle_match_frac"] == pytest.approx(1.0)
    assert d1["rest_match_frac"] == pytest.approx(0.0)
    assert d1["rest_minus_first_gap"] == pytest.approx(-1.0)
    assert d1["interpretation"] == "no-cycle-0-drag"


def test_d1_cycle_0_bound_when_rest_better_than_first():
    """If the rest is much better than the first cycle, the gap is
    positive and the verdict is ``cycle-0-bound``.

    D1's match metric AND-s left and right foot contact matches per
    step, so the test sets both feet's ref/phys consistently per
    epoch.
    """
    n = _FIRST_CYCLE_STEPS + 30
    trace = _make_trace(n_steps=n)
    # First cycle: BOTH feet mismatched on every step
    for i in range(_FIRST_CYCLE_STEPS):
        trace.contact_left_ref[i] = True
        trace.contact_left_phys[i] = False  # mismatch
        trace.contact_right_ref[i] = True
        trace.contact_right_phys[i] = False  # mismatch
    # Rest: BOTH feet matched
    for i in range(_FIRST_CYCLE_STEPS, n):
        trace.contact_left_ref[i] = True
        trace.contact_left_phys[i] = True
        trace.contact_right_ref[i] = True
        trace.contact_right_phys[i] = True
    d1 = _diag_d1_contact_match_split(trace)
    assert d1["first_cycle_match_frac"] == pytest.approx(0.0)
    assert d1["rest_match_frac"] == pytest.approx(1.0)
    assert d1["rest_minus_first_gap"] >= 0.10
    assert d1["interpretation"] == "cycle-0-bound"


# ---------------------------------------------------------------------
# Composite verdict
# ---------------------------------------------------------------------


def test_verdict_composes_components_correctly():
    d1 = {"interpretation": "no-cycle-0-drag"}
    d2 = {"interpretation": "planner-shape-bound"}
    d3 = {"interpretation": "Phase-7-boundary-bound"}
    v = _verdict(d1, d2, d3)
    assert v["components"]["planner_shape_bound"]
    assert v["components"]["phase_7_boundary_bound"]
    assert not v["components"]["cycle_0_bound"]
    assert not v["components"]["actuator_stack_bound"]
    assert "mixed" in v["verdict"]


def test_verdict_actuator_stack_bound_requires_clean_d1_and_d2():
    d1 = {"interpretation": "no-cycle-0-drag"}
    d2 = {"interpretation": "clear"}
    d3 = {"interpretation": "uniform-sat-distribution"}
    v = _verdict(d1, d2, d3)
    assert v["components"]["actuator_stack_bound"]
    assert v["verdict"] == "actuator_stack_bound"


def test_verdict_distinguishes_mechanism_unclear():
    """When D3+D4 reports any-joint clustering but swing-side ankle is
    clean, the verdict is ``down_cross_correlated_mechanism_unclear``
    — NOT ``phase_7_boundary_bound``.  This is the key fix from the
    review feedback.
    """
    d1 = {"interpretation": "no-cycle-0-drag"}
    d2 = {"interpretation": "clear"}
    d3 = {"interpretation": "down-cross-correlated-mechanism-unclear"}
    v = _verdict(d1, d2, d3)
    assert v["components"]["down_cross_correlated_mechanism_unclear"]
    assert not v["components"]["phase_7_boundary_bound"]
