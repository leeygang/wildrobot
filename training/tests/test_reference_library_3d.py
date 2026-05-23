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
