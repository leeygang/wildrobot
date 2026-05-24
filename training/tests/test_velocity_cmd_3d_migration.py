"""v0.21.0 P3 — WildRobotInfo.velocity_cmd 3-vector migration tests.

Foundational data-shape change: ``WildRobotInfo.velocity_cmd`` moves
from a scalar ``()`` (forward vx only) to a length-3 vector
``(vx, vy, wz)``.  Companion bumps:

* ``EnvConfig.cmd_deadzone`` and ``EnvConfig.eval_velocity_cmd`` become
  three-tuples (H3: ``eval_velocity_cmd`` scalar broadcasts to
  ``(s, 0, 0)``, not ``(s, s, s)``).
* ``compute_command_integrated_path_state`` takes ``velocity_cmd_y_mps``
  and ``yaw_rate_cmd_rps``.
* ``WildRobotInfo`` carries ``path_state_torso_pos`` (3,) and
  ``path_state_path_rot`` (4,) for runtime-correct incremental
  integration under ``cmd_resample_steps > 0`` (H7).

All sub-step regression tests live here; the implementation lands in one
atomic commit per the plan's foundational-shape-change directive.
"""

from __future__ import annotations

import dataclasses
import math

import jax
import jax.numpy as jp
import numpy as np
import pytest

from training.envs.env_info import WR_INFO_KEY, get_expected_shapes


# ---------------------------------------------------------------------------
# P3.1 — WildRobotInfo.velocity_cmd shape
# ---------------------------------------------------------------------------


def test_wildrobot_info_velocity_cmd_expected_shape_is_3() -> None:
    shapes = get_expected_shapes(action_size=13)
    assert shapes["velocity_cmd"] == (3,)


# ---------------------------------------------------------------------------
# P3.3 — EnvConfig.cmd_deadzone default-tuple
# ---------------------------------------------------------------------------


def test_env_config_cmd_deadzone_default_is_three_tuple() -> None:
    from training.configs.training_runtime_config import EnvConfig

    cfg = EnvConfig()
    assert tuple(cfg.cmd_deadzone) == (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# P3.4a — _load_three_axis helper (H3 + NEW-4)
# ---------------------------------------------------------------------------


def test_load_three_axis_accepts_scalar_and_broadcasts_for_deadzone() -> None:
    from training.configs.training_config import _load_three_axis

    out = _load_three_axis(
        0.05, default_scalar=0.0, scalar_broadcast_axis="all",
    )
    assert out == (0.05, 0.05, 0.05)


def test_load_three_axis_scalar_broadcast_for_eval_cmd() -> None:
    """H3: scalar ``0.26`` for eval_velocity_cmd must map to
    ``(0.26, 0.0, 0.0)``, NOT ``(0.26, 0.26, 0.26)``.  vy / wz
    default to ``0.0``."""
    from training.configs.training_config import _load_three_axis

    out = _load_three_axis(
        0.26, default_scalar=-1.0, scalar_broadcast_axis="vx_only",
    )
    assert out == (0.26, 0.0, 0.0)


def test_load_three_axis_negative_scalar_for_eval_cmd_sentinel() -> None:
    """H3: ``-1.0`` sentinel scalar -> ``(-1.0, 0.0, 0.0)``.  Sentinel
    detection in env code reads only the [0]th axis (vx) per H3."""
    from training.configs.training_config import _load_three_axis

    out = _load_three_axis(
        -1.0, default_scalar=-1.0, scalar_broadcast_axis="vx_only",
    )
    assert out == (-1.0, 0.0, 0.0)


def test_load_three_axis_accepts_three_tuple() -> None:
    from training.configs.training_config import _load_three_axis

    out = _load_three_axis(
        [0.02, 0.03, 0.04], default_scalar=0.0, scalar_broadcast_axis="all",
    )
    assert out == (0.02, 0.03, 0.04)


def test_load_three_axis_uses_default_on_none_for_deadzone() -> None:
    from training.configs.training_config import _load_three_axis

    out = _load_three_axis(
        None, default_scalar=0.0, scalar_broadcast_axis="all",
    )
    assert out == (0.0, 0.0, 0.0)


def test_load_three_axis_uses_default_on_none_for_eval_cmd() -> None:
    from training.configs.training_config import _load_three_axis

    out = _load_three_axis(
        None, default_scalar=-1.0, scalar_broadcast_axis="vx_only",
    )
    assert out == (-1.0, 0.0, 0.0)


def test_load_three_axis_rejects_two_tuple() -> None:
    from training.configs.training_config import _load_three_axis

    with pytest.raises(ValueError, match="length 3"):
        _load_three_axis(
            [0.02, 0.03], default_scalar=0.0, scalar_broadcast_axis="all",
        )


def test_smoke14_yaml_eval_cmd_scalar_broadcasts_to_vx_only() -> None:
    """H3 + S1: smoke14's scalar ``eval_velocity_cmd: 0.26`` MUST map
    to ``(0.26, 0.0, 0.0)``, NOT ``(0.26, 0.26, 0.26)``; the scalar
    ``cmd_deadzone: 0.0666666667`` symmetric-broadcasts to all axes."""
    from training.configs.training_config import load_training_config

    cfg = load_training_config(
        "training/configs/ppo_walking_v0201_smoke14.yaml"
    )
    assert tuple(cfg.env.eval_velocity_cmd) == pytest.approx(
        (0.26, 0.0, 0.0)
    )
    assert tuple(cfg.env.cmd_deadzone) == pytest.approx(
        (0.0666666667, 0.0666666667, 0.0666666667), abs=1e-7,
    )


# ---------------------------------------------------------------------------
# P3.5 — EnvConfig.eval_velocity_cmd default-tuple (H3)
# ---------------------------------------------------------------------------


def test_env_config_eval_velocity_cmd_default_is_vx_only_sentinel() -> None:
    """H3: default tuple is ``(-1.0, 0.0, 0.0)``: vx uses the sentinel,
    vy / wz default to ``0.0`` (not ``-1.0``)."""
    from training.configs.training_runtime_config import EnvConfig

    cfg = EnvConfig()
    assert tuple(cfg.eval_velocity_cmd) == (-1.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# P3.7 — Env step fanout under 3-vec velocity_cmd
# ---------------------------------------------------------------------------


def test_env_step_does_not_raise_on_three_vec_velocity_cmd() -> None:
    from training.configs.training_config import load_training_config
    from training.envs.wildrobot_env import WildRobotEnv

    cfg = load_training_config(
        "training/configs/ppo_walking_v0201_smoke14.yaml"
    )
    env = WildRobotEnv(cfg)
    state = env.reset(jax.random.PRNGKey(0))
    assert state.info[WR_INFO_KEY].velocity_cmd.shape == (3,)
    _ = env.step(state, jp.zeros(env.action_size))


# ---------------------------------------------------------------------------
# P3.8a — Per-callsite pass-through shape test
# ---------------------------------------------------------------------------


def test_pass_through_callsites_receive_three_vec() -> None:
    """Verify each pass-through callsite forwards velocity_cmd as
    ``(3,)``.  Mutates the reset-state's velocity_cmd to a non-trivial
    3-vec and steps the env; if any pass-through callsite bottlenecks
    to scalar, this raises a shape mismatch deep inside ``step``.
    """
    from training.configs.training_config import load_training_config
    from training.envs.wildrobot_env import WildRobotEnv

    cfg = load_training_config(
        "training/configs/ppo_walking_v0201_smoke14.yaml"
    )
    env = WildRobotEnv(cfg)
    state = env.reset(jax.random.PRNGKey(0))
    new_cmd = jp.array([0.1, 0.05, 0.02], dtype=jp.float32)
    wr = state.info[WR_INFO_KEY]
    new_wr = wr.replace(velocity_cmd=new_cmd)
    state = state.replace(
        info={**state.info, WR_INFO_KEY: new_wr}
    )
    state2 = env.step(state, jp.zeros(env.action_size))
    assert state2.info[WR_INFO_KEY].velocity_cmd.shape == (3,)


# ---------------------------------------------------------------------------
# P3.9 — compute_command_integrated_path_state 3D extension
# ---------------------------------------------------------------------------


def test_path_state_integrates_lateral_velocity() -> None:
    from control.references.runtime_reference_service import (
        RuntimeReferenceService,
    )

    out = RuntimeReferenceService.compute_command_integrated_path_state(
        t_since_reset_s=1.0,
        velocity_cmd_mps=0.0,
        velocity_cmd_y_mps=0.10,
        yaw_rate_cmd_rps=0.0,
        dt_s=0.02,
    )
    assert float(out["path_pos"][1]) == pytest.approx(0.10, abs=0.01)
    assert abs(float(out["path_pos"][0])) < 0.01


def test_path_state_integrates_yaw_rate_for_pure_turn() -> None:
    from control.references.runtime_reference_service import (
        RuntimeReferenceService,
    )

    out = RuntimeReferenceService.compute_command_integrated_path_state(
        t_since_reset_s=1.0,
        velocity_cmd_mps=0.0,
        velocity_cmd_y_mps=0.0,
        yaw_rate_cmd_rps=0.10,
        dt_s=0.02,
    )
    expected_qz = math.sin(0.10 / 2.0)
    assert float(out["path_rot"][3]) == pytest.approx(expected_qz, abs=0.005)


def test_path_state_2d_collapses_to_1d_when_vy_zero() -> None:
    """C10 backward-compat regression: with ``vy=0`` and ``wz != 0``,
    the new closed form must produce the same ``path_pos[1]`` as the
    legacy vx-only closed-form."""
    from control.references.runtime_reference_service import (
        RuntimeReferenceService,
    )

    out_new = RuntimeReferenceService.compute_command_integrated_path_state(
        t_since_reset_s=1.0,
        velocity_cmd_mps=0.10,
        velocity_cmd_y_mps=0.0,
        yaw_rate_cmd_rps=0.10,
        dt_s=0.02,
    )
    dt, vx, wz = 0.02, 0.10, 0.10
    n = round(1.0 / dt)
    theta = wz * dt
    sin_half = math.sin(theta / 2)
    ratio = math.sin(n * theta / 2) / sin_half
    legacy_y = vx * dt * ratio * math.sin((n + 1) * theta / 2)
    assert float(out_new["path_pos"][1]) == pytest.approx(legacy_y, abs=1e-6)


def test_incremental_path_state_matches_closed_form_for_constant_cmd() -> None:
    """H7: ``incremental_path_state_step`` iterated N times must match
    the closed-form helper when cmd is constant.  Equivalence proof
    that lets the env use the incremental form at runtime without
    losing parity with TB's analytic prior."""
    from control.references.runtime_reference_service import (
        RuntimeReferenceService,
    )

    dt = 0.02
    vx, vy, wz = 0.10, 0.05, 0.10
    n = 50
    torso_pos = jp.zeros(3, dtype=jp.float32)
    path_rot = jp.array([1.0, 0.0, 0.0, 0.0], dtype=jp.float32)
    cmd = jp.array([vx, vy, wz], dtype=jp.float32)
    for _ in range(n):
        torso_pos, path_rot = (
            RuntimeReferenceService.incremental_path_state_step(
                prev_torso_pos=torso_pos,
                prev_path_rot_wxyz=path_rot,
                velocity_cmd=cmd,
                dt_s=dt,
            )
        )
    closed = RuntimeReferenceService.compute_command_integrated_path_state(
        t_since_reset_s=n * dt,
        velocity_cmd_mps=vx,
        velocity_cmd_y_mps=vy,
        yaw_rate_cmd_rps=wz,
        dt_s=dt,
    )
    np.testing.assert_allclose(
        np.asarray(torso_pos), np.asarray(closed["path_pos"]), atol=1e-5,
    )


# ---------------------------------------------------------------------------
# P3.10a — WildRobotInfo carries incremental path-state
# ---------------------------------------------------------------------------


def test_wildrobot_info_carries_incremental_path_state() -> None:
    """H7: ``path_state_torso_pos`` (3,) and ``path_state_path_rot``
    (4,) wxyz."""
    shapes = get_expected_shapes(action_size=13)
    assert shapes["path_state_torso_pos"] == (3,)
    assert shapes["path_state_path_rot"] == (4,)


# ---------------------------------------------------------------------------
# P3.10b — reset initializes path-state to origin / identity
# ---------------------------------------------------------------------------


def test_reset_initializes_path_state_to_origin_and_identity() -> None:
    from training.configs.training_config import load_training_config
    from training.envs.wildrobot_env import WildRobotEnv

    cfg = load_training_config(
        "training/configs/ppo_walking_v0201_smoke14.yaml"
    )
    env = WildRobotEnv(cfg)
    state = env.reset(jax.random.PRNGKey(0))
    wr = state.info[WR_INFO_KEY]
    np.testing.assert_allclose(
        np.asarray(wr.path_state_torso_pos), np.zeros(3)
    )
    np.testing.assert_allclose(
        np.asarray(wr.path_state_path_rot),
        np.array([1.0, 0.0, 0.0, 0.0]),
    )


# ---------------------------------------------------------------------------
# P3.10c — incremental_path_state_step unit tests
# ---------------------------------------------------------------------------


def test_incremental_step_translates_along_world_x_under_identity_rotation() -> None:
    from control.references.runtime_reference_service import (
        RuntimeReferenceService,
    )

    new_pos, new_rot = RuntimeReferenceService.incremental_path_state_step(
        prev_torso_pos=jp.zeros(3, dtype=jp.float32),
        prev_path_rot_wxyz=jp.array(
            [1.0, 0.0, 0.0, 0.0], dtype=jp.float32
        ),
        velocity_cmd=jp.array([0.10, 0.0, 0.0], dtype=jp.float32),
        dt_s=0.02,
    )
    assert float(new_pos[0]) == pytest.approx(0.10 * 0.02, abs=1e-7)
    assert float(new_pos[1]) == pytest.approx(0.0, abs=1e-7)
    # Identity rot unchanged since wz=0:
    assert float(new_rot[0]) == pytest.approx(1.0, abs=1e-7)
    assert float(new_rot[3]) == pytest.approx(0.0, abs=1e-7)


def test_incremental_step_rotates_around_z_under_pure_yaw_cmd() -> None:
    from control.references.runtime_reference_service import (
        RuntimeReferenceService,
    )

    _, new_rot = RuntimeReferenceService.incremental_path_state_step(
        prev_torso_pos=jp.zeros(3, dtype=jp.float32),
        prev_path_rot_wxyz=jp.array(
            [1.0, 0.0, 0.0, 0.0], dtype=jp.float32
        ),
        velocity_cmd=jp.array([0.0, 0.0, 0.10], dtype=jp.float32),
        dt_s=0.02,
    )
    expected_qz = math.sin(0.10 * 0.02 / 2.0)
    assert float(new_rot[3]) == pytest.approx(expected_qz, abs=1e-6)


# ---------------------------------------------------------------------------
# P3.10d — path-state continuity under cmd resample
# ---------------------------------------------------------------------------


def test_path_state_continuity_under_cmd_resample() -> None:
    """H7: a mid-episode cmd resample must NOT reset path_state to
    origin.  Run 50 steps with ``cmd_resample_steps=10`` and assert
    ``path_state_torso_pos[0]`` never discontinuously goes negative."""
    from training.configs.training_config import load_training_config
    from training.envs.wildrobot_env import WildRobotEnv

    cfg = load_training_config(
        "training/configs/ppo_walking_v0201_smoke14.yaml"
    )
    cfg.env = dataclasses.replace(cfg.env, cmd_resample_steps=10)
    env = WildRobotEnv(cfg)
    state = env.reset(jax.random.PRNGKey(0))
    last_x = float(state.info[WR_INFO_KEY].path_state_torso_pos[0])
    for _ in range(50):
        state = env.step(state, jp.zeros(env.action_size))
        cur_x = float(state.info[WR_INFO_KEY].path_state_torso_pos[0])
        # Under any non-zero cmd, x should grow (or hold if cmd
        # resamples to zero).  Assertion: never goes negative
        # discontinuously by more than 0.05 m in a single step.
        assert cur_x >= last_x - 0.05, (
            "path_state_torso_pos reset / drifted under cmd resample"
        )
        last_x = cur_x
