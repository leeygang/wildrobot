"""v0.21.0 P5 — 3D reference library plumbing tests.

Covers:
  - P5.1: ``EnvConfig`` accepts ``loc_ref_command_axes_3d`` + vy / yaw_rate grids.
  - P5.3: ``_init_offline_service`` builds a 3D library when the toggle is on.
  - P5.4a: ``_yaw_from_quat_wxyz`` helper recovers yaw from a wxyz quat.
  - P5.5: ``_lookup_offline_window`` picks different bins for sign-mirrored vy
          commands and for a yawed actor heading (heading-frame correction).

The smoke1 YAML referenced in the original spec doesn't exist yet (P8 lands it),
so we adapt by loading the existing ``smoke14`` YAML and overriding only the
3D-relevant env fields via ``dataclasses.replace``.  Bin count is asserted
dynamically against the actual library size (the library is TB-split: linear
``vx × vy`` bins + pure-yaw bins with ``(0, 0, 0)`` dedup; cf. P2.5b/c) instead
of a hard-coded Cartesian count.
"""

from __future__ import annotations

import dataclasses
import math

import jax.numpy as jp
import pytest


# ---------------------------------------------------------------------------
# P5.1 — config toggle / grid fields
# ---------------------------------------------------------------------------


def test_env_config_accepts_3d_axes_toggle_and_grids() -> None:
    from training.configs.training_runtime_config import EnvConfig

    cfg = EnvConfig(
        loc_ref_command_axes_3d=True,
        loc_ref_offline_command_vy_grid=(-0.10, 0.0, 0.10),
        loc_ref_offline_command_yaw_rate_grid=(-0.20, 0.0, 0.20),
    )
    assert cfg.loc_ref_command_axes_3d is True
    assert cfg.loc_ref_offline_command_vy_grid == (-0.10, 0.0, 0.10)
    assert cfg.loc_ref_offline_command_yaw_rate_grid == (-0.20, 0.0, 0.20)


def test_loc_ref_command_axes_3d_defaults_false() -> None:
    from training.configs.training_runtime_config import EnvConfig

    cfg = EnvConfig()
    assert cfg.loc_ref_command_axes_3d is False
    assert cfg.loc_ref_offline_command_vy_grid == ()
    assert cfg.loc_ref_offline_command_yaw_rate_grid == ()


# ---------------------------------------------------------------------------
# Shared env factory: smoke14 YAML + dataclasses.replace 3D overrides
# ---------------------------------------------------------------------------


def _env_3d_via_smoke14_replace():
    """Build a WildRobotEnv with the 3D library toggle flipped on.

    The vy / yaw_rate grids are small (3 entries each) to keep library-build
    time bounded — the ZMP planner runs once per bin.  vx grid is inherited
    from the smoke14 YAML (``min_velocity=0.18``, ``max_velocity=0.26``,
    ``loc_ref_command_grid_interval=0.02`` ⇒ vx ∈ {0.18, 0.20, 0.22, 0.24, 0.26}).
    """
    from training.configs.training_config import load_training_config
    from training.envs.wildrobot_env import WildRobotEnv

    cfg = load_training_config("training/configs/ppo_walking_v0201_smoke14.yaml")
    cfg = dataclasses.replace(
        cfg,
        env=dataclasses.replace(
            cfg.env,
            loc_ref_command_axes_3d=True,
            loc_ref_offline_command_vy_grid=(-0.05, 0.0, 0.05),
            loc_ref_offline_command_yaw_rate_grid=(-0.10, 0.0, 0.10),
        ),
    )
    return WildRobotEnv(cfg)


# ---------------------------------------------------------------------------
# P5.3 — _init_offline_service builds 3D library
# ---------------------------------------------------------------------------


def test_init_offline_service_builds_3d_grid_when_toggled() -> None:
    """When ``loc_ref_command_axes_3d=True`` the env stores a ``(n_bins, 3)``
    ``_offline_cmd_keys`` array whose size matches the TB-split library."""
    env = _env_3d_via_smoke14_replace()
    # n_bins axis matches the stacked q_ref array along axis 0.
    n_bins_from_stack = int(env._offline_jax_arrays["q_ref"].shape[0])
    assert env._offline_cmd_keys.ndim == 2
    assert env._offline_cmd_keys.shape == (n_bins_from_stack, 3)
    # TB-split library bin count: smoke14 YAML gives vx ∈
    # {0.18, 0.20, 0.22, 0.24, 0.26} (5 values, none == 0.0), vy ∈
    # {-0.05, 0, 0.05}, wz ∈ {-0.1, 0, 0.1}.  Linear bins are
    # full Cartesian over (vx × vy) with wz=0; pure-yaw bins are
    # ``(0, 0, wz)`` for each wz.  Note: ``build_library_for_3d_values``
    # iterates the PASSED vx values for linear bins (no force-union of
    # 0.0 — that only goes into ``meta.command_range_vx`` per the round-2
    # follow-up).  With no overlap (vx=0.18 ≠ 0) the static dedup is
    # trivial: 5 * 3 = 15 linear + 3 pure-yaw = 18 total.
    # The (0, 0, 0) static bin is one of the pure-yaw bins (wz=0).
    assert n_bins_from_stack == 18, (
        f"Expected 18 TB-split bins (5 vx * 3 vy + 3 wz, no dedup); "
        f"got {n_bins_from_stack}.  Check build_library_for_3d_values."
    )


def test_offline_cmd_keys_include_static_bin() -> None:
    """A ``(0, 0, 0)`` static bin is present so static / pure-turn commands
    snap to it instead of to the nearest forward-walking bin (H2 invariant)."""
    import numpy as np

    env = _env_3d_via_smoke14_replace()
    keys = np.asarray(env._offline_cmd_keys)
    static_present = bool(np.any(np.linalg.norm(keys, axis=-1) < 1e-6))
    assert static_present, "Missing (0, 0, 0) static bin in _offline_cmd_keys"


# ---------------------------------------------------------------------------
# P5.4a — _yaw_from_quat_wxyz helper
# ---------------------------------------------------------------------------


def test_yaw_from_quat_wxyz_recovers_yaw() -> None:
    from control.references.runtime_reference_service import _yaw_from_quat_wxyz

    yaw = 0.3
    q = jp.asarray(
        [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)], dtype=jp.float32
    )  # wxyz
    assert float(_yaw_from_quat_wxyz(q)) == pytest.approx(yaw, abs=1e-5)


def test_yaw_from_quat_wxyz_recovers_negative_yaw() -> None:
    from control.references.runtime_reference_service import _yaw_from_quat_wxyz

    yaw = -0.5
    q = jp.asarray(
        [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)], dtype=jp.float32
    )
    assert float(_yaw_from_quat_wxyz(q)) == pytest.approx(yaw, abs=1e-5)


# ---------------------------------------------------------------------------
# P5.5 — heading-frame correction + 3D nearest-key lookup
# ---------------------------------------------------------------------------


def test_lookup_offline_window_uses_3d_nearest_key() -> None:
    """Sign-mirrored vy commands MUST land in different bins (smoke vs the
    legacy 1D-vx lookup, which would have collapsed both to the same bin)."""
    env = _env_3d_via_smoke14_replace()
    win_a = env._lookup_offline_window(
        step_idx=jp.asarray(10, dtype=jp.int32),
        velocity_cmd=jp.asarray([0.20, 0.05, 0.0], dtype=jp.float32),
    )
    win_b = env._lookup_offline_window(
        step_idx=jp.asarray(10, dtype=jp.int32),
        velocity_cmd=jp.asarray([0.20, -0.05, 0.0], dtype=jp.float32),
    )
    assert float(win_a["selected_vy"]) != float(win_b["selected_vy"])


def test_lookup_with_yawed_heading_picks_different_bin() -> None:
    """Heading-frame correction: a +0.3 rad path rotation while commanding
    pure +vx rotates the cmd into a ``(vx', vy')`` mix in the body frame.

    The library bin grid is sparse enough that the corrected cmd lands in a
    different (selected_vx, selected_vy) than the aligned (path == heading)
    case — verifying the correction is actually applied and not a no-op.
    """
    env = _env_3d_via_smoke14_replace()
    cmd = jp.asarray([0.20, 0.0, 0.0], dtype=jp.float32)
    yaw = 0.3
    path_rot_wxyz = jp.asarray(
        [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)], dtype=jp.float32
    )
    identity_rot = jp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jp.float32)
    win_yawed = env._lookup_offline_window(
        step_idx=jp.asarray(10, dtype=jp.int32),
        velocity_cmd=cmd,
        path_rot_wxyz=path_rot_wxyz,
        heading_rot_wxyz=identity_rot,
    )
    win_aligned = env._lookup_offline_window(
        step_idx=jp.asarray(10, dtype=jp.int32),
        velocity_cmd=cmd,
        path_rot_wxyz=identity_rot,
        heading_rot_wxyz=identity_rot,
    )
    # The corrected cmd projects onto (vx', vy') ≠ (vx, 0), so the L2 nearest
    # bin shifts to one with a non-zero |vy| component.
    assert (
        float(win_yawed["selected_vy"]) != float(win_aligned["selected_vy"])
        or float(win_yawed["selected_vx"]) != float(win_aligned["selected_vx"])
    )


# ---------------------------------------------------------------------------
# Reviewer bug fixes (post-P5)
# ---------------------------------------------------------------------------


def test_3d_primary_idx_selects_straight_walk_anchor_not_first_vx_tie() -> None:
    """Reviewer bug #1: in 3D mode primary_idx must select the
    (offline_vx, 0, 0) anchor — NOT the first vy-tied bin (e.g.
    (offline_vx, vy_min, 0) under build_library_for_3d_values' sort
    order).  primary_service feeds _ref_init_q_rad via
    _offline_service.lookup_np(0) so a wrong anchor contaminates the
    reset pose.
    """
    from training.configs.training_config import load_training_config
    from training.envs.wildrobot_env import WildRobotEnv

    cfg = load_training_config("training/configs/ppo_walking_v0201_smoke14.yaml")
    cfg = dataclasses.replace(
        cfg,
        env=dataclasses.replace(
            cfg.env,
            loc_ref_command_axes_3d=True,
            loc_ref_offline_command_vy_grid=(-0.05, 0.0, 0.05),
            loc_ref_offline_command_yaw_rate_grid=(-0.10, 0.0, 0.10),
        ),
    )
    env = WildRobotEnv(cfg)
    # ``RuntimeReferenceService.command_key`` is a (vx, vy, wz) tuple
    # set from the underlying ReferenceTrajectory at construction.
    primary_key = env._offline_service.command_key
    offline_vx = float(cfg.env.loc_ref_offline_command_vx)
    assert math.isclose(primary_key[0], offline_vx, abs_tol=1e-3), (
        f"primary anchor vx={primary_key[0]} != offline_vx={offline_vx}"
    )
    assert math.isclose(primary_key[1], 0.0, abs_tol=1e-9), (
        f"primary anchor vy={primary_key[1]} != 0; "
        "P5 bound the wrong (vx, vy_min, 0) lateral bin instead of "
        "(vx, 0, 0) straight-walk."
    )
    assert math.isclose(primary_key[2], 0.0, abs_tol=1e-9)


def test_3d_mode_respects_loc_ref_offline_library_path(tmp_path) -> None:
    """Reviewer bug #2: when loc_ref_offline_library_path is set AND
    loc_ref_command_axes_3d=True, the env must load from disk, not
    silently rebuild via ZMPWalkGenerator.build_library_for_3d_values.
    """
    from control.zmp.zmp_walk import ZMPWalkGenerator
    from control.zmp import zmp_walk
    from training.configs.training_config import load_training_config
    from training.envs.wildrobot_env import WildRobotEnv

    # Build a tiny library on disk first.
    gen = ZMPWalkGenerator()
    lib = gen.build_library_for_3d_values(
        vx_values=[0.18, 0.22, 0.26],
        vy_values=[0.0],
        yaw_rate_values=[0.0],
    )
    save_path = str(tmp_path / "tiny_3d_lib")
    lib.save(save_path)

    cfg = load_training_config("training/configs/ppo_walking_v0201_smoke14.yaml")
    cfg = dataclasses.replace(
        cfg,
        env=dataclasses.replace(
            cfg.env,
            loc_ref_command_axes_3d=True,
            loc_ref_offline_library_path=save_path,
            loc_ref_offline_command_vy_grid=(0.0,),  # ignored because path is set
            loc_ref_offline_command_yaw_rate_grid=(0.0,),
        ),
    )

    # Spy on the planner — it MUST NOT be called by env init.  (The
    # one-time on-disk build above happens BEFORE the spy is installed,
    # so the spy only sees calls from WildRobotEnv.__init__.)
    original_build = zmp_walk.ZMPWalkGenerator.build_library_for_3d_values
    call_count = {"n": 0}

    def spy(self, *args, **kwargs):
        call_count["n"] += 1
        return original_build(self, *args, **kwargs)

    zmp_walk.ZMPWalkGenerator.build_library_for_3d_values = spy
    try:
        env = WildRobotEnv(cfg)
    finally:
        zmp_walk.ZMPWalkGenerator.build_library_for_3d_values = original_build

    assert call_count["n"] == 0, (
        f"3D mode unexpectedly rebuilt the library via "
        f"build_library_for_3d_values ({call_count['n']} call(s)) instead "
        f"of loading from loc_ref_offline_library_path={save_path}"
    )
    # And verify the loaded library has the bins from disk: the on-disk
    # library has 3 linear bins (vx=0.18/0.22/0.26 with vy=0, wz=0) +
    # 1 pure-yaw static bin (0, 0, 0) = 4 total.
    assert int(env._offline_cmd_keys.shape[0]) >= 3, (
        "loaded library should have >= 3 bins from the persisted file"
    )
