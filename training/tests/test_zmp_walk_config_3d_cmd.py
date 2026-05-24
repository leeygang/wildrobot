from __future__ import annotations
import pytest
from control.zmp.zmp_walk import ZMPWalkConfig, ZMPWalkGenerator


def test_zmp_walk_config_has_rotation_radius_field() -> None:
    cfg = ZMPWalkConfig()
    assert cfg.rotation_radius_m == pytest.approx(0.10)


def test_zmp_walk_config_accepts_rotation_radius_override() -> None:
    cfg = ZMPWalkConfig(rotation_radius_m=0.15)
    assert cfg.rotation_radius_m == pytest.approx(0.15)


def test_generate_accepts_vy_and_yaw_rate_kwargs() -> None:
    gen = ZMPWalkGenerator()
    traj = gen.generate(command_vx=0.20, command_vy=0.05, command_yaw_rate=0.0)
    assert traj.command_vx == pytest.approx(0.20)
    assert traj.command_vy == pytest.approx(0.05)
    assert traj.command_yaw_rate == pytest.approx(0.0)


def test_generate_defaults_vy_and_yaw_to_zero() -> None:
    gen = ZMPWalkGenerator()
    traj = gen.generate(command_vx=0.20)
    assert traj.command_vy == pytest.approx(0.0)
    assert traj.command_yaw_rate == pytest.approx(0.0)


def test_pure_lateral_command_does_not_short_circuit_to_standing() -> None:
    """H1: generate(vx=0, vy=0.05) MUST produce a non-standing trajectory.
    Under the old gate `abs(command_vx) < 1e-4`, this would return
    `_generate_standing()` and the lateral footstep block would never run.
    """
    gen = ZMPWalkGenerator()
    traj = gen.generate(command_vx=0.0, command_vy=0.05, command_yaw_rate=0.0)
    foot_y = traj.left_foot_pos[:, 1]
    assert (foot_y.max() - foot_y.min()) > 0.001, (
        "H1 fix missing: pure-lateral command short-circuited to standing."
    )


def test_pure_yaw_command_does_not_short_circuit_to_standing() -> None:
    """H1: generate(vx=0, wz=0.10) MUST produce non-trivial yaw."""
    gen = ZMPWalkGenerator()
    traj = gen.generate(command_vx=0.0, command_vy=0.0, command_yaw_rate=0.10)
    yaw = traj.pelvis_rpy[:, 2]
    assert abs(yaw[-1] - yaw[0]) > 1e-3


def test_negative_vy_flips_lateral_stride_sign() -> None:
    """H1 follow-on: line 609 currently flips step_length on vx < 0 only.
    Verify the equivalent per-axis sign flip works for vy < 0 too. Check
    pelvis world-y rather than foot y because foot_pos is stored
    COM-relative in WR (zmp_walk.py:1218-1220), so foot y does not flip
    even when the underlying world motion does."""
    gen = ZMPWalkGenerator()
    traj_pos = gen.generate(command_vx=0.0, command_vy=+0.05)
    traj_neg = gen.generate(command_vx=0.0, command_vy=-0.05)
    assert traj_pos.pelvis_pos[-1, 1] * traj_neg.pelvis_pos[-1, 1] < 0


def test_planner_produces_nonzero_lateral_foot_displacement_for_vy() -> None:
    """Tightened from -final.md: now uses vx=0 (was vx=0.0001) since the
    H1 gate fix lets a pure-lateral command through."""
    gen = ZMPWalkGenerator()
    traj = gen.generate(command_vx=0.0, command_vy=0.05)
    foot_y = traj.left_foot_pos[:, 1]
    assert (foot_y.max() - foot_y.min()) > 0.005


def test_planner_yaws_pelvis_for_nonzero_yaw_rate() -> None:
    """Tightened: uses vx=vy=0 (was vy=0); requires the H1 gate fix."""
    gen = ZMPWalkGenerator()
    traj = gen.generate(command_vx=0.0, command_vy=0.0, command_yaw_rate=0.10)
    yaw = traj.pelvis_rpy[:, 2]
    assert abs(yaw[-1] - yaw[0]) > 1e-3


def test_generated_trajectory_carries_all_three_cmd_axes() -> None:
    gen = ZMPWalkGenerator()
    traj = gen.generate(command_vx=0.20, command_vy=0.04, command_yaw_rate=0.0)
    assert traj.command_vx == pytest.approx(0.20)
    assert traj.command_vy == pytest.approx(0.04)
    assert traj.command_yaw_rate == pytest.approx(0.0)


def test_build_library_for_3d_values_yields_one_entry_per_grid_point() -> None:
    gen = ZMPWalkGenerator()
    lib = gen.build_library_for_3d_values(
        vx_values=[0.0, 0.20],  # H2: include static bin
        vy_values=[-0.05, 0.0, 0.05],
        yaw_rate_values=[-0.10, 0.0, 0.10],
    )
    # Reviewer fix (round 2): library is TB-style split commands, not full
    # Cartesian.  Linear: 2 vx × 3 vy = 6.  Pure-yaw: 3 wz.
    # Overlap on (0, 0, 0) → dedup 1.  Total = 8.
    assert len(lib) == 8
    assert (0.20, 0.0, 0.10) not in lib, "mixed translation+yaw bin forbidden"
    assert (0.20, 0.0, 0.0) in lib, "linear-walk bin must exist"
    assert (0.0, 0.0, 0.10) in lib, "pure-yaw bin must exist"
    assert (0.0, 0.0, 0.0) in lib, "H2: must include the static-pose bin"


def test_library_factory_uses_tb_style_split_commands() -> None:
    """Reviewer (round 2): library must NOT store mixed translation+yaw
    bins because the planner only applies yaw in the pure-rotation branch
    (vx≈vy≈0). A key like (0.20, 0.05, 0.20) would have command_yaw_rate
    in metadata but zero actual yaw in the trajectory.

    TB's approach: linear-walk bins (vx, vy, 0) + pure-yaw bins (0, 0, wz),
    dedup (0, 0, 0).
    """
    gen = ZMPWalkGenerator()
    lib = gen.build_library_for_3d_values(
        vx_values=[0.0, 0.20],
        vy_values=[-0.05, 0.0, 0.05],
        yaw_rate_values=[-0.10, 0.0, 0.10],
    )
    # Linear: 2 vx × 3 vy = 6.  Pure-yaw: 3 wz.  Overlap on (0, 0, 0).
    # Total = 6 + 3 - 1 = 8.
    assert len(lib) == 8, (
        f"expected 8 split-cmd bins (linear 2x3=6 + pure-yaw 3 - dedup 1), "
        f"got {len(lib)}"
    )
    # Required bins present:
    assert (0.0, 0.0, 0.0) in lib, "static bin missing"
    assert (0.20, 0.05, 0.0) in lib, "linear-walk bin missing"
    assert (0.0, 0.0, 0.10) in lib, "pure-yaw bin missing"
    # Forbidden mixed bins absent:
    assert (0.20, 0.05, 0.10) not in lib, (
        "mixed translation+yaw bin must NOT exist (planner can't realize it)"
    )
    assert (0.20, 0.0, 0.10) not in lib, (
        "mixed vx+yaw bin must NOT exist"
    )


def test_library_factory_no_stored_key_combines_translation_and_yaw() -> None:
    """Invariant: for every stored key (vx, vy, wz), either translation is
    zero (||vx, vy|| < eps) or yaw is zero (|wz| < eps).  Mixed keys would
    be silently mislabeled."""
    import math
    gen = ZMPWalkGenerator()
    lib = gen.build_library_for_3d_values(
        vx_values=[0.0, 0.18, 0.22, 0.26],
        vy_values=[-0.10, -0.05, 0.0, 0.05, 0.10],
        yaw_rate_values=[-0.20, -0.10, 0.0, 0.10, 0.20],
    )
    eps = 1e-9
    for key in lib._entries.keys():
        vx, vy, wz = key
        translation_magnitude = math.hypot(vx, vy)
        assert translation_magnitude < eps or abs(wz) < eps, (
            f"key {key} combines translation (||vx,vy||={translation_magnitude}) "
            f"and yaw (|wz|={abs(wz)}); planner cannot realize this."
        )


def test_library_factory_canonical_grid_has_24_bins() -> None:
    """v0.21.0 canonical grid: 4 vx × 5 vy + 5 wz - 1 dedup = 24 bins."""
    gen = ZMPWalkGenerator()
    lib = gen.build_library_for_3d_values(
        vx_values=[0.0, 0.18, 0.22, 0.26],
        vy_values=[-0.10, -0.05, 0.0, 0.05, 0.10],
        yaw_rate_values=[-0.20, -0.10, 0.0, 0.10, 0.20],
    )
    assert len(lib) == 24


def test_build_library_for_vx_values_still_builds_forward_only_library() -> None:
    """P1.12 regression: the legacy 1D ``build_library_for_vx_values``
    factory must continue to build a forward-only library after the
    P1 (vx, vy, wz) widening landed.  This guards the existing v0.20.x
    smoke / runtime paths that still call the 1D factory."""
    gen = ZMPWalkGenerator()
    lib = gen.build_library_for_vx_values([0.0, 0.20])
    assert len(lib) == 2
    assert (0.0, 0.0, 0.0) in lib
    assert (0.20, 0.0, 0.0) in lib


def test_pure_yaw_command_propagates_to_left_foot_rpy_yaw() -> None:
    """Reviewer #3: yaw must propagate into the foot orientation trajectory,
    not stop at the pelvis.  Under pure-yaw cmd, left_foot_rpy[:, 2] must
    accumulate yaw across the cycle."""
    gen = ZMPWalkGenerator()
    traj = gen.generate(command_vx=0.0, command_vy=0.0, command_yaw_rate=0.10)
    foot_yaw = traj.left_foot_rpy[:, 2]
    assert abs(foot_yaw[-1] - foot_yaw[0]) > 1e-3, (
        "Reviewer #3 fix missing: foot yaw stayed at zero under pure-yaw cmd."
    )


def test_pure_yaw_command_propagates_to_right_foot_rpy_yaw() -> None:
    gen = ZMPWalkGenerator()
    traj = gen.generate(command_vx=0.0, command_vy=0.0, command_yaw_rate=0.10)
    foot_yaw = traj.right_foot_rpy[:, 2]
    assert abs(foot_yaw[-1] - foot_yaw[0]) > 1e-3


def test_combined_translation_keeps_foot_yaw_at_zero() -> None:
    """Combined vx+vy with wz=0 must not introduce spurious foot yaw."""
    gen = ZMPWalkGenerator()
    traj = gen.generate(command_vx=0.15, command_vy=0.05, command_yaw_rate=0.0)
    assert abs(traj.left_foot_rpy[:, 2]).max() < 1e-4
    assert abs(traj.right_foot_rpy[:, 2]).max() < 1e-4


def test_pure_yaw_ik_produces_nonzero_hip_roll_or_hip_pitch_pattern_distinct_from_no_yaw() -> None:
    """Sanity check that the IK actually consumes the yaw orientation.
    If IK ignores yaw, the q_ref under wz=0.10 (with foot yaw = 0.5*0.10*dt
    per swing) would be indistinguishable from the q_ref under wz=0
    (forward-only).  This test asserts they differ."""
    gen = ZMPWalkGenerator()
    traj_yaw = gen.generate(command_vx=0.0, command_vy=0.0, command_yaw_rate=0.10)
    traj_zero = gen.generate(command_vx=0.0, command_vy=0.0, command_yaw_rate=0.0)
    # Either zero is _generate_standing() (q_ref = home), traj_yaw must
    # diverge from home_pose over the cycle.
    import numpy as np
    diff = np.abs(np.asarray(traj_yaw.q_ref) - np.asarray(traj_zero.q_ref)).max()
    assert diff > 1e-3, (
        "IK appears to ignore foot yaw orientation under pure-yaw cmd."
    )
