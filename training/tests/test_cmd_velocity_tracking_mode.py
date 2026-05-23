from __future__ import annotations

import math
from pathlib import Path

import pytest

from training.configs.training_config import load_training_config
from training.configs.training_runtime_config import RewardWeightsConfig
from training.core.metrics_registry import METRIC_SPEC_BY_NAME, Reducer
from training.envs.wildrobot_env import WildRobotEnv


_SMOKE13 = Path("training/configs/ppo_walking_v0201_smoke13.yaml")


def test_cmd_velocity_track_dim_defaults_to_scalar_mode() -> None:
    assert RewardWeightsConfig().cmd_velocity_track_dim == 1


def test_smoke13_sets_2d_cmd_velocity_tracking_mode() -> None:
    if not _SMOKE13.exists():
        pytest.skip(f"{_SMOKE13.name} not found")
    cfg = load_training_config(str(_SMOKE13))
    assert cfg.reward_weights.cmd_velocity_track_dim == 2


def test_scalar_cmd_tracking_ignores_lateral_velocity() -> None:
    """Dim=1 reward must be ``exp(-alpha * (vx_actual - velocity_cmd)^2)``
    and ignore both ``lateral_velocity`` and the reference velocity
    inputs in the reward path.  ``cmd_velocity_xy_err`` is the
    actual-vs-commanded diagnostic (kept under both dims);
    ``ref_velocity_xy_err`` is the diagnostic-only actual-vs-ref
    distance and is non-zero whenever a ref velocity is supplied
    (it is NOT a function of track_dim post-fix)."""
    reward, xy_err, lat_abs, ref_err = (
        WildRobotEnv._cmd_forward_velocity_track_reward(
            forward_velocity=0.1333333333,
            lateral_velocity=0.1,
            velocity_cmd=0.1333333333,
            # Adversarial: huge ref velocity.  Under dim=1 the reward
            # ignores it; only the cmd axis matters.
            ref_forward_velocity=99.0,
            ref_lateral_velocity=-99.0,
            alpha=562.5,
            track_dim=1,
        )
    )
    assert float(reward) == pytest.approx(1.0)
    assert float(xy_err) == pytest.approx(0.1)
    assert float(lat_abs) == pytest.approx(0.1)
    # Diagnostic-only ref err is non-zero (huge), confirming it's the
    # actual-vs-ref distance rather than a track_dim gate.
    assert float(ref_err) > 1.0


def test_2d_cmd_tracking_targets_command_not_reference() -> None:
    """TB parity (post-fix): dim=2 reward compares actual (vx, vy)
    to (velocity_cmd, 0), NOT to the selected reference's pelvis
    velocity.  Pin this by feeding actual == velocity_cmd but with
    a wildly off reference, and asserting reward == 1.0."""
    reward, xy_err, lat_abs, ref_err = (
        WildRobotEnv._cmd_forward_velocity_track_reward(
            forward_velocity=0.10,
            lateral_velocity=0.0,
            velocity_cmd=0.10,
            # Reference at a totally different bin's finite-diff
            # velocity.  Under the pre-fix dim=2 path this would
            # crater the reward.  Under TB-parity it MUST NOT matter.
            ref_forward_velocity=0.133,
            ref_lateral_velocity=0.02,
            alpha=562.5,
            track_dim=2,
        )
    )
    assert float(reward) == pytest.approx(1.0), (
        "dim=2 reward must be 1.0 when actual == (velocity_cmd, 0) "
        "regardless of the reference's pelvis velocity — proves the "
        "reward target is the COMMAND, not the cmd-conditioned ref."
    )
    # cmd-distance diagnostic is the reward-target error under dim=2.
    assert float(xy_err) == pytest.approx(0.0)
    assert float(lat_abs) == pytest.approx(0.0)
    # ref-distance diagnostic is non-zero and equals the cmd vs ref
    # distance: sqrt((0.10-0.133)^2 + (0.0-0.02)^2).
    expected_ref_err = math.sqrt((0.10 - 0.133) ** 2 + (0.0 - 0.02) ** 2)
    assert float(ref_err) == pytest.approx(expected_ref_err, abs=1e-6)


def test_2d_cmd_tracking_penalizes_lateral_drift_against_zero_cmd() -> None:
    """Under dim=2 (post-fix) the lateral axis is penalized against
    ``vy_cmd == 0`` — WR has no lateral command.  Forward axis at
    velocity_cmd contributes zero error; only the lateral drift
    drives the loss."""
    reward, xy_err, lat_abs, _ref_err = (
        WildRobotEnv._cmd_forward_velocity_track_reward(
            forward_velocity=0.10,
            lateral_velocity=0.05,
            velocity_cmd=0.10,
            # Reference irrelevant to the reward under the TB-parity
            # fix; pin one that matches actual so the test is
            # focused on the cmd-vs-actual loss.
            ref_forward_velocity=0.10,
            ref_lateral_velocity=0.05,
            alpha=562.5,
            track_dim=2,
        )
    )
    # err_sq = 0 + (0.05)^2 = 0.0025 → reward = exp(-562.5 * 0.0025)
    expected = math.exp(-562.5 * 0.0025)
    assert float(reward) == pytest.approx(expected)
    assert float(reward) < 1.0
    assert float(xy_err) == pytest.approx(0.05)
    assert float(lat_abs) == pytest.approx(0.05)


def test_metrics_registry_has_cmd_and_ref_velocity_err() -> None:
    xy = METRIC_SPEC_BY_NAME["tracking/cmd_velocity_xy_err"]
    lat = METRIC_SPEC_BY_NAME["tracking/lateral_velocity_abs"]
    ref = METRIC_SPEC_BY_NAME["tracking/ref_velocity_xy_err"]
    sel = METRIC_SPEC_BY_NAME["tracking/ref_selected_vx"]
    err = METRIC_SPEC_BY_NAME["tracking/ref_cmd_bin_abs_err"]
    assert xy.reducer == Reducer.MEAN
    assert lat.reducer == Reducer.MEAN
    assert ref.reducer == Reducer.MEAN
    assert sel.reducer == Reducer.MEAN
    assert err.reducer == Reducer.MEAN



# ---------------------------------------------------------------------------
# Selected-bin diagnostics — the env exposes ``selected_vx`` and
# ``cmd_bin_abs_err`` on every ``_lookup_offline_window`` call so the
# bin choice is observable in the training metrics.
# ---------------------------------------------------------------------------
def _make_env_from(path: Path):
    if not path.exists():
        pytest.skip(f"{path.name} not found")
    cfg = load_training_config(str(path))
    return WildRobotEnv(config=cfg)


def test_cmd_conditioned_lookup_exposes_selected_bin_diagnostics() -> None:
    """Smoke13 grid (post commit 80c8058 + grid_interval=0.02) is
    [0.08, 0.10, 0.12, 0.1333333].  ``cmd=0.10`` snaps to bin 0.10
    with zero error; ``cmd=0.107`` snaps to the analytical nearest
    bin (0.10 vs 0.12: |0.107-0.10|=0.007 < |0.107-0.12|=0.013, so
    0.10 wins) with error 0.007."""
    import jax.numpy as jp
    import numpy as np

    env = _make_env_from(_SMOKE13)
    assert env._offline_command_conditioned
    grid = np.asarray(env._offline_vx_grid)
    expected_grid = np.array([0.08, 0.10, 0.12, 0.1333333333], dtype=np.float32)
    np.testing.assert_allclose(grid, expected_grid, atol=1e-5)

    step = jp.asarray(50, dtype=jp.int32)

    win_exact = env._lookup_offline_window(step, velocity_cmd=jp.float32(0.10))
    assert float(win_exact["selected_vx"]) == pytest.approx(0.10, abs=1e-5)
    assert float(win_exact["cmd_bin_abs_err"]) == pytest.approx(0.0, abs=1e-7)

    # cmd=0.107 → analytical nearest bin = 0.10 (|0.107-0.10|=0.007
    # < |0.107-0.12|=0.013).  Pin the exact expected bin & error.
    cmd = 0.107
    nearest_bin_idx = int(np.argmin(np.abs(grid - cmd)))
    expected_bin = float(grid[nearest_bin_idx])
    expected_err = abs(expected_bin - cmd)
    win_near = env._lookup_offline_window(step, velocity_cmd=jp.float32(cmd))
    assert float(win_near["selected_vx"]) == pytest.approx(expected_bin, abs=1e-5)
    assert float(win_near["cmd_bin_abs_err"]) == pytest.approx(expected_err, abs=1e-6)


def test_legacy_mode_selected_bin_diagnostics_use_configured_vx() -> None:
    """Under legacy single-bin mode (smoke12b: loc_ref_command_conditioned
    defaults to False) ``selected_vx`` must equal
    ``loc_ref_offline_command_vx`` and ``cmd_bin_abs_err`` is zero
    when no command is passed; non-zero is the |configured -
    velocity_cmd| distance when a command IS passed."""
    import jax.numpy as jp

    smoke12b_path = Path("training/configs/ppo_walking_v0201_smoke12b.yaml")
    env = _make_env_from(smoke12b_path)
    assert not env._offline_command_conditioned
    configured_vx = float(env._config.env.loc_ref_offline_command_vx)
    step = jp.asarray(10, dtype=jp.int32)

    win_no_cmd = env._lookup_offline_window(step)
    assert float(win_no_cmd["selected_vx"]) == pytest.approx(configured_vx, abs=1e-6)
    assert float(win_no_cmd["cmd_bin_abs_err"]) == pytest.approx(0.0, abs=1e-7)

    probe_cmd = configured_vx + 0.03
    win_with_cmd = env._lookup_offline_window(step, velocity_cmd=jp.float32(probe_cmd))
    assert float(win_with_cmd["selected_vx"]) == pytest.approx(configured_vx, abs=1e-6)
    assert float(win_with_cmd["cmd_bin_abs_err"]) == pytest.approx(
        abs(configured_vx - probe_cmd), abs=1e-6
    )
