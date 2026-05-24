"""v0.21.0 P2.5a-r2: one-shot library build-time + memory measurement (M3).

This is a measurement test — the printed line is captured for the
smoke1 PR description.  Assertions are intentionally loose so the
test does not flake on slow CI runners; tighten only if a regression
is observed.

Reviewer round-2 update: the canonical library is now TB-style
split-cmd (linear-walk + pure-yaw - dedup), so the canonical grid
shrinks from 100 bins to 24 bins and the build time follows.
"""

from __future__ import annotations

import time

from control.zmp.zmp_walk import ZMPWalkGenerator


def test_build_canonical_library_measure_time_and_memory(capsys) -> None:
    """M3 + reviewer round-2: one-shot measurement.  Library uses TB-style
    split-cmd factory so the bin count is small (≤24 for the canonical
    grid).  Loose budgets — tighten only if regression is observed.
    """
    gen = ZMPWalkGenerator()
    t0 = time.perf_counter()
    lib = gen.build_library_for_3d_values(
        vx_values=[0.0, 0.18, 0.22, 0.26],
        vy_values=[-0.10, -0.05, 0.0, 0.05, 0.10],
        yaw_rate_values=[-0.20, -0.10, 0.0, 0.10, 0.20],
    )
    elapsed_s = time.perf_counter() - t0
    n_bins = len(lib)
    # 4 vx * 5 vy = 20 linear + 5 pure-yaw - 1 dedup = 24 bins.
    assert n_bins == 24, f"expected 24 split-cmd bins, got {n_bins}"
    total_bytes = sum(
        arr.nbytes
        for traj in lib._entries.values()
        for arr in (traj.q_ref, traj.pelvis_pos, traj.left_foot_pos,
                    traj.right_foot_pos, traj.pelvis_rpy)
        if hasattr(arr, "nbytes")
    )
    mb = total_bytes / (1024 * 1024)
    print(
        f"\n[v0.21.0 P2.5a-r2] library: bins={n_bins}, build_s={elapsed_s:.2f}, "
        f"approx_mb={mb:.2f}"
    )
    # Loose budgets sized for split-cmd library.
    assert elapsed_s < 60.0, f"build too slow: {elapsed_s:.1f}s"
    assert mb < 60.0, f"library too large: {mb:.1f} MB"
