"""Tests for v0.21.0 P6 — vy_cmd plumbed into the cmd-velocity reward
helper, and the new yaw-rate tracking term.

P6 wires the 3-axis ``velocity_cmd`` (vx, vy, wz) into the reward
path:
  * P6.1/P6.2 — ``_cmd_forward_velocity_track_reward`` accepts an
    optional ``vy_cmd`` kwarg (defaults to 0.0 so legacy callers
    are byte-for-byte identical — NEW-1 / C16 mitigation).
  * P6.3 — ``_compute_reward_terms`` passes ``velocity_cmd[1]`` as
    ``vy_cmd``; the resulting ``tracking/cmd_velocity_xy_err``
    metric now uses ``vy_actual - vy_cmd`` (instead of just
    ``vy_actual``) so it reflects the commanded lateral.
  * P6.4 — new static helper ``_yaw_rate_track_reward`` +
    ``cmd_yaw_rate_alpha`` (TB walk.gin
    ang_vel_tracking_sigma=4.0 → α=1/4.0=0.25, H5) +
    ``cmd_yaw_rate_track`` weight (default 0.0 for back-compat) +
    ``tracking/yaw_rate_err`` metric.

Pre-existing tests in ``test_cmd_velocity_tracking_mode.py`` are
expected to keep passing unchanged: they call the helper without
``vy_cmd`` and the default of 0.0 preserves prior behavior.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import pytest

from training.configs.training_config import load_training_config
from training.configs.training_runtime_config import RewardWeightsConfig
from training.core.metrics_registry import METRIC_SPEC_BY_NAME, Reducer
from training.envs.wildrobot_env import WildRobotEnv


_SMOKE14 = Path("training/configs/ppo_walking_v0201_smoke14.yaml")


# ---------------------------------------------------------------------------
# P6.1 / P6.2 — vy_cmd kwarg
# ---------------------------------------------------------------------------
def test_2d_cmd_tracking_uses_supplied_vy_cmd() -> None:
    """P6.1: helper accepts ``vy_cmd`` and the dim=2 reward
    penalizes ``(vy_actual - vy_cmd)`` rather than ``vy_actual``.
    Pin reward == 1.0 when both axes match the command exactly."""
    reward, xy_err, lat_abs, _ = WildRobotEnv._cmd_forward_velocity_track_reward(
        forward_velocity=0.10,
        lateral_velocity=0.05,
        velocity_cmd=0.10,
        vy_cmd=0.05,
        ref_forward_velocity=0.10,
        ref_lateral_velocity=0.05,
        alpha=562.5,
        track_dim=2,
    )
    assert float(reward) == pytest.approx(1.0)
    # Under dim=2 with vy_cmd matched, the lateral err contribution
    # is zero — but the diagnostic ``cmd_velocity_xy_err`` is the
    # full 2D distance from actual to the (vx_cmd, vy_cmd) target,
    # so it should also be zero here.
    assert float(xy_err) == pytest.approx(0.0, abs=1e-6)
    # ``lateral_velocity_abs`` is unconditionally |vy_actual|, not
    # |vy_err|, so it stays at 0.05.
    assert float(lat_abs) == pytest.approx(0.05)


def test_default_vy_cmd_preserves_legacy_dim2_behavior() -> None:
    """C16 mitigation: legacy callers that omit ``vy_cmd`` get
    ``vy_cmd = 0`` and therefore identical numerics to the
    pre-P6.2 ``vy_err_cmd = lateral_velocity`` form."""
    # Re-use the exact inputs from
    # ``test_2d_cmd_tracking_penalizes_lateral_drift_against_zero_cmd``.
    reward, xy_err, lat_abs, _ = (
        WildRobotEnv._cmd_forward_velocity_track_reward(
            forward_velocity=0.10,
            lateral_velocity=0.05,
            velocity_cmd=0.10,
            # vy_cmd intentionally omitted — default must be 0.0.
            ref_forward_velocity=0.10,
            ref_lateral_velocity=0.05,
            alpha=562.5,
            track_dim=2,
        )
    )
    expected = math.exp(-562.5 * 0.0025)
    assert float(reward) == pytest.approx(expected)
    assert float(xy_err) == pytest.approx(0.05)
    assert float(lat_abs) == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# P6.3 — call site wires real vy_cmd from velocity_cmd[1]
# ---------------------------------------------------------------------------
def _load_smoke14_dim2_lateral_active_cfg():
    """Load smoke14 and patch the reward-weights block so this test
    actually exercises the new lateral path:
      * cmd_velocity_track_dim = 2 (lateral axis matters)
      * cmd_forward_velocity_track > 0 (kept from smoke14)
      * cmd_yaw_rate_track > 0 (new term, exercised separately)
    smoke14 itself defaults to dim=1 (no smoke1.yaml exists yet —
    P8 creates it); dataclass-replace is the supported override."""
    if not _SMOKE14.exists():
        pytest.skip(f"{_SMOKE14.name} not found")
    cfg = load_training_config(str(_SMOKE14))
    new_rw = dataclasses.replace(
        cfg.reward_weights,
        cmd_velocity_track_dim=2,
        cmd_yaw_rate_track=1.0,
    )
    return dataclasses.replace(cfg, reward_weights=new_rw)


def test_env_reward_consumes_real_vy_cmd_from_3vec() -> None:
    """P6.3 E2E: forcing ``velocity_cmd = (0, 0.10, 0)`` after reset
    must produce ``tracking/cmd_velocity_xy_err ~= 0.10`` on the
    first step (vx_actual ~ 0, vy_actual ~ 0, so the 2D distance
    to (0, 0.10) is 0.10).  Pre-P6.3 the metric was
    sqrt(vx_err^2 + vy_actual^2) and would read ~0.0 here."""
    import jax
    import jax.numpy as jp

    from training.core.metrics_registry import METRICS_VEC_KEY, unpack_metrics
    from training.envs.env_info import WR_INFO_KEY

    cfg = _load_smoke14_dim2_lateral_active_cfg()
    env = WildRobotEnv(config=cfg)
    state = env.reset(jax.random.PRNGKey(0))
    wr = state.info[WR_INFO_KEY]
    new_wr = wr.replace(
        velocity_cmd=jp.array([0.0, 0.10, 0.0], dtype=jp.float32)
    )
    state = state.replace(info={**state.info, WR_INFO_KEY: new_wr})
    state2 = env.step(state, jp.zeros(env.action_size))
    metrics = unpack_metrics(state2.metrics[METRICS_VEC_KEY])
    err = metrics["tracking/cmd_velocity_xy_err"]
    # First-step actor velocities are ~0 (the policy zero-action
    # doesn't move the COM in a single ctrl step), so the 2D
    # distance to (vx_cmd=0, vy_cmd=0.10) is dominated by the
    # |0 - 0.10| = 0.10 lateral component.
    assert float(err) == pytest.approx(0.10, abs=0.02)
    # And the new yaw-rate diagnostic should be present and bounded.
    # The first physics step's body-frame gyro reads non-zero noise
    # (~0.2 rad/s under the zero-action settling transient) but is
    # always |ang_vel_z - cmd_wz|; with cmd_wz=0 it equals
    # |gyro[2]|, which a settled robot would drive back to 0.
    yaw_err = metrics["tracking/yaw_rate_err"]
    assert float(yaw_err) < 2.0  # sanity bound, not a tracking gate


# ---------------------------------------------------------------------------
# P6.4 — yaw-rate tracking term (TB-aligned alpha=0.25, H5)
# ---------------------------------------------------------------------------
def test_yaw_rate_alpha_field_exists_with_tb_aligned_default() -> None:
    """H5: ``cmd_yaw_rate_alpha`` is a NEW field, NOT a reuse of
    ``cmd_forward_velocity_alpha``.  TB walk.gin
    ang_vel_tracking_sigma=4.0 → α = 1/4.0 = 0.25."""
    rw = RewardWeightsConfig()
    assert rw.cmd_yaw_rate_alpha == pytest.approx(0.25)


def test_yaw_rate_track_weight_defaults_to_zero_for_back_compat() -> None:
    """Default weight must be 0.0 so legacy YAMLs (smoke14, smoke12b,
    smoke7, …) do NOT silently start tracking yaw rate."""
    rw = RewardWeightsConfig()
    assert rw.cmd_yaw_rate_track == pytest.approx(0.0)


def test_yaw_rate_track_reward_peaks_at_cmd() -> None:
    reward = WildRobotEnv._yaw_rate_track_reward(
        ang_vel_z=0.10,
        yaw_rate_cmd=0.10,
        alpha=0.25,
    )
    assert float(reward) == pytest.approx(1.0)


def test_yaw_rate_track_reward_decays_off_cmd_with_tb_alpha() -> None:
    """H5: at α=0.25 (TB-aligned) an error of 0.10 rad/s yields
    reward = exp(-0.25 * 0.01) = exp(-0.0025) ≈ 0.9975 — a much
    broader basin than the -final.md draft's α=562.5 (which
    would have given exp(-5.625) ≈ 0.0036)."""
    reward = WildRobotEnv._yaw_rate_track_reward(
        ang_vel_z=0.0,
        yaw_rate_cmd=0.10,
        alpha=0.25,
    )
    assert float(reward) == pytest.approx(math.exp(-0.0025), rel=1e-5)


def test_yaw_rate_err_metric_registered_with_mean_reducer() -> None:
    spec = METRIC_SPEC_BY_NAME["tracking/yaw_rate_err"]
    assert spec.reducer == Reducer.MEAN


def test_training_config_loader_reads_yaw_rate_fields() -> None:
    """The YAML reward loader must round-trip both new fields with
    their TB-aligned defaults when the YAML omits them."""
    if not _SMOKE14.exists():
        pytest.skip(f"{_SMOKE14.name} not found")
    cfg = load_training_config(str(_SMOKE14))
    # smoke14 does NOT set these — defaults must hold.
    assert cfg.reward_weights.cmd_yaw_rate_alpha == pytest.approx(0.25)
    assert cfg.reward_weights.cmd_yaw_rate_track == pytest.approx(0.0)
