#!/usr/bin/env python3
"""Compare WR's two candidate runtime reference target classes.

Phase 2 of training/docs/reference_architecture_comparison.md decides
whether WR's body-pose reward target should remain the **planner-phase**
trajectory (today's ``win["pelvis_pos"]`` from the LIPM-LQR COM plan)
or move to the **command-integrated** path (TB-style
``path_state.path_pos`` integrated at runtime from ``velocity_cmd``).

The two are nearly identical in the y axis (LIPM lateral oscillation is
small) but diverge in the x axis whenever the planner deviates from a
linear ``vx · t`` advance.  The dominant divergence source is the
from-rest cycle 0 transient that WR retains and TB truncates: the
planner spends the first ~one cycle ramping up to steady-state speed,
which leaves the planner trajectory permanently offset behind the
command-integrated path even after steady state is reached.

This script reports both the steady-state and worst-case divergence
numbers cited in the Phase 2 closeout so the rationale stays
reproducible.

Usage::

    uv run python tools/compare_reference_target_semantics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from control.zmp.zmp_walk import ZMPWalkGenerator


_VX_BINS = (0.15, 0.20, 0.25)  # Phase 9D: bracket around vx=0.20 operating point (TB-step/leg-matched at cycle_time=0.96 s)


def _summary_for_vx(vx: float) -> dict:
    """Per-vx divergence summary between planner-phase and command-integrated."""
    traj = ZMPWalkGenerator().generate(vx)
    n = traj.n_steps
    dt = float(traj.dt)
    cycle_steps = int(round(traj.cycle_time / dt))

    t = np.arange(n) * dt
    planner_xy = np.asarray(traj.pelvis_pos[:, :2], dtype=np.float64)
    # WR smoke runs yaw_rate=0 and vy=0; the TB-style command-integrated
    # path reduces to (vx*t, 0) for that configuration.  When yaw is
    # added (v0.20.4), the helper must integrate via SO(3) composition
    # like ``toddlerbot/reference/motion_ref.py::integrate_path_state``.
    cmd_xy = np.zeros_like(planner_xy)
    cmd_xy[:, 0] = vx * t
    cmd_xy[:, 1] = 0.0

    diff = planner_xy - cmd_xy
    diff_norm = np.linalg.norm(diff, axis=1)

    # Steady-state slice = everything past the first cycle (where the
    # from-rest transient settles).
    ss = slice(cycle_steps, None)

    return {
        "vx": vx,
        "n_steps": n,
        "duration_s": n * dt,
        "cycle_steps": cycle_steps,
        "x_mean_m": float(diff[:, 0].mean()),
        "x_max_abs_m": float(np.abs(diff[:, 0]).max()),
        "x_rms_m": float(np.sqrt((diff[:, 0] ** 2).mean())),
        "x_final_delta_m": float(diff[-1, 0]),
        "y_mean_m": float(diff[:, 1].mean()),
        "y_max_abs_m": float(np.abs(diff[:, 1]).max()),
        "y_rms_m": float(np.sqrt((diff[:, 1] ** 2).mean())),
        "l2_max_m": float(diff_norm.max()),
        "l2_rms_m": float(np.sqrt((diff_norm ** 2).mean())),
        "ss_x_mean_m": float(diff[ss, 0].mean()),
        "ss_x_rms_m": float(np.sqrt((diff[ss, 0] ** 2).mean())),
        "ss_y_max_abs_m": float(np.abs(diff[ss, 1]).max()),
        "ss_y_rms_m": float(np.sqrt((diff[ss, 1] ** 2).mean())),
        "worst_l2_idx": int(np.argmax(diff_norm)),
    }


def main() -> int:
    print("Phase 2 reference-target-semantics divergence probe")
    print()
    print("Compares WR's planner-phase target (traj.pelvis_pos[:, :2]) to")
    print("TB-style command-integrated target (vx * t_since_reset, 0)")
    print("across the smoke vx range.  WR yaw_rate = 0 so the integrated")
    print("path reduces to a linear x advance; the closeout calls out that")
    print("yaw-enabled v0.20.4 will need full SO(3) integration.")
    print()

    rows = [_summary_for_vx(vx) for vx in _VX_BINS]

    print(
        f"{'vx':>5}  {'dur(s)':>7}  {'cyc':>4}  "
        f"{'x_mean(m)':>10}  {'x_max|·|':>9}  {'x_rms':>9}  {'x_final':>9}  "
        f"{'y_max|·|':>9}  {'y_rms':>9}  {'L2_max':>8}"
    )
    print("-" * 110)
    for r in rows:
        print(
            f"{r['vx']:>5.2f}  {r['duration_s']:>7.2f}  {r['cycle_steps']:>4d}  "
            f"{r['x_mean_m']:>+10.4f}  {r['x_max_abs_m']:>9.4f}  "
            f"{r['x_rms_m']:>9.4f}  {r['x_final_delta_m']:>+9.4f}  "
            f"{r['y_max_abs_m']:>9.4f}  {r['y_rms_m']:>9.4f}  "
            f"{r['l2_max_m']:>8.4f}"
        )

    print()
    print("Steady-state slice (frames after cycle 0, where the from-rest")
    print("transient has settled).  TB truncates cycle 0; WR retains it.")
    print()
    print(
        f"{'vx':>5}  {'ss_x_mean(m)':>13}  {'ss_x_rms':>10}  "
        f"{'ss_y_max|·|':>12}  {'ss_y_rms':>10}"
    )
    print("-" * 60)
    for r in rows:
        print(
            f"{r['vx']:>5.2f}  {r['ss_x_mean_m']:>+13.5f}  "
            f"{r['ss_x_rms_m']:>10.5f}  {r['ss_y_max_abs_m']:>12.5f}  "
            f"{r['ss_y_rms_m']:>10.5f}"
        )

    print()
    print("Reading the numbers:")
    print("- x_mean / x_rms: dominated by the from-rest cycle 0 lag.  After")
    print("  cycle 0 settles, the planner trajectory is offset behind the")
    print("  command-integrated path by ~one half-cycle of cumulative ramp.")
    print("- y_max / y_rms: pure LIPM lateral oscillation under Q=I, R=0.1.")
    print("  Sub-cm peak, sub-mm RMS — small enough to ignore for reward")
    print("  shaping but visible in the trajectory.")
    print("- The two semantics are NOT equivalent up to noise even in")
    print("  steady state; the cycle-0-retention difference produces a")
    print("  permanent x offset.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
