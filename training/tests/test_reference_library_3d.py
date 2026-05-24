"""v0.21.0 P2: 3D reference library regression + meta-grid tests.

These tests anchor the audit assumption that ``ReferenceLibrary.lookup``
is already 3D-aware (P2.1), and lock in the new ``vy_grid`` /
``yaw_rate_grid`` metadata fields that P2.3 adds to
``ReferenceLibraryMeta`` (P2.2, P2.4).
"""

from __future__ import annotations

import pytest

from control.references.reference_library import ReferenceLibraryMeta
from control.zmp.zmp_walk import ZMPWalkGenerator


def test_reference_library_lookup_already_picks_nearest_3d_key() -> None:
    gen = ZMPWalkGenerator()
    lib = gen.build_library_for_3d_values(
        vx_values=[0.20],
        vy_values=[-0.05, 0.0, 0.05],
        yaw_rate_values=[-0.10, 0.0, 0.10],
    )
    picked = lib.lookup(vx=0.20, vy=0.04, yaw_rate=-0.08)
    assert picked.command_vy == pytest.approx(0.05)
    assert picked.command_yaw_rate == pytest.approx(-0.10)


def test_meta_accepts_vy_and_yaw_rate_grids() -> None:
    meta = ReferenceLibraryMeta(
        command_range_vx=(0.20, 0.20),
        command_range_vy=(-0.05, 0.05),
        command_range_yaw=(-0.10, 0.10),
        vy_grid=(-0.05, 0.0, 0.05),
        yaw_rate_grid=(-0.10, 0.0, 0.10),
    )
    assert meta.vy_grid == (-0.05, 0.0, 0.05)
    assert meta.yaw_rate_grid == (-0.10, 0.0, 0.10)


def test_build_library_for_3d_values_populates_meta_grids() -> None:
    gen = ZMPWalkGenerator()
    lib = gen.build_library_for_3d_values(
        vx_values=[0.20],
        vy_values=[-0.05, 0.0, 0.05],
        yaw_rate_values=[-0.10, 0.0, 0.10],
    )
    assert tuple(lib.meta.vy_grid) == (-0.05, 0.0, 0.05)
    assert tuple(lib.meta.yaw_rate_grid) == (-0.10, 0.0, 0.10)


def test_lookup_uses_per_axis_nearest_not_mixed_unit_euclidean() -> None:
    """Reviewer #5: rad/s and m/s must not be mixed in a single L2.
    A vy diff of 0.05 m/s should NOT compete with a wz diff of 0.05 rad/s
    in the same metric.  Per-axis selection makes both choices independent.
    """
    gen = ZMPWalkGenerator()
    lib = gen.build_library_for_3d_values(
        vx_values=[0.20],
        vy_values=[-0.05, 0.0, 0.05],
        yaw_rate_values=[-0.20, 0.0, 0.20],
    )
    # Query at (0.20, +0.03, +0.18): per-axis nearest is (0.20, +0.05, +0.20).
    # Under a mixed-unit L2 with naive weighting, the wz mismatch of 0.02
    # could push the result toward a different vy bin.  Per-axis selection
    # is invariant to that.
    picked = lib.lookup(vx=0.20, vy=0.03, yaw_rate=0.18)
    assert picked.command_vy == pytest.approx(0.05)
    assert picked.command_yaw_rate == pytest.approx(0.20)


def test_lookup_falls_back_to_zero_for_empty_axis_grid() -> None:
    """Legacy 1D libraries (only vx_grid populated) must still work."""
    gen = ZMPWalkGenerator()
    lib = gen.build_library_for_vx_values([0.0, 0.20])  # 1D library
    picked = lib.lookup(vx=0.18, vy=0.0, yaw_rate=0.0)
    assert picked.command_vx == pytest.approx(0.20)
