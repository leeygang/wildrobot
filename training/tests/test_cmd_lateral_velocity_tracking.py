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


# ---------------------------------------------------------------------------
# v0.21.0 smoke2 follow-up — strict reward_weights loader, end-to-end
# emission of cmd_yaw_rate_track, and dim=2 vy_cmd contract.
# ---------------------------------------------------------------------------
def test_reward_weights_loader_rejects_unknown_key(tmp_path) -> None:
    """A YAML key under ``reward_weights:`` that does NOT correspond
    to a ``RewardWeightsConfig`` dataclass field must raise at load
    time.  The previous silent-accept behaviour let typos (e.g.
    ``cmd_velocity_track`` when the real field is
    ``cmd_velocity_track_dim`` / ``cmd_forward_velocity_track``)
    become dead config — see v0.21.0 smoke1 post-mortem."""
    if not _SMOKE14.exists():
        pytest.skip(f"{_SMOKE14.name} not found")

    # Build a smoke14 copy with one bogus reward_weights key.
    import yaml as _yaml

    raw = _yaml.safe_load(_SMOKE14.read_text())
    raw["reward_weights"]["cmd_velocity_track"] = 2.0  # the smoke1 typo
    bad = tmp_path / "smoke14_with_bogus_reward.yaml"
    bad.write_text(_yaml.safe_dump(raw))

    with pytest.raises(ValueError, match="Unknown reward_weights key"):
        load_training_config(str(bad))


def test_reward_weights_loader_accepts_all_valid_dataclass_fields() -> None:
    """Sanity: a real smoke YAML (smoke14) MUST load cleanly under
    the strict validator.  Catches regressions where the loader
    drops a legitimate field from its accepted set."""
    if not _SMOKE14.exists():
        pytest.skip(f"{_SMOKE14.name} not found")
    cfg = load_training_config(str(_SMOKE14))
    # If we got here, the strict validator passed every smoke14 key.
    assert cfg.reward_weights is not None


def test_reward_cmd_yaw_rate_track_emits_when_weight_positive() -> None:
    """End-to-end: when ``cmd_yaw_rate_track > 0`` and the env is
    stepped with a non-zero yaw cmd, the per-step
    ``reward/cmd_yaw_rate_track`` metric MUST be present in the
    emitted ``metrics_vec`` and become non-zero capable (positive
    weight × non-zero kernel value).  This catches the smoke1
    silent-drop where the term was wired in the dataclass + env +
    metrics registry but never appeared in the W&B logs."""
    import jax
    import jax.numpy as jp

    from training.core.metrics_registry import METRICS_VEC_KEY, unpack_metrics
    from training.envs.env_info import WR_INFO_KEY

    cfg = _load_smoke14_dim2_lateral_active_cfg()  # turns cmd_yaw_rate_track=1.0
    env = WildRobotEnv(config=cfg)
    state = env.reset(jax.random.PRNGKey(0))
    wr = state.info[WR_INFO_KEY]
    new_wr = wr.replace(
        velocity_cmd=jp.array([0.0, 0.0, 0.10], dtype=jp.float32)
    )
    state = state.replace(info={**state.info, WR_INFO_KEY: new_wr})
    state2 = env.step(state, jp.zeros(env.action_size))
    metrics = unpack_metrics(state2.metrics[METRICS_VEC_KEY])

    # Key MUST exist (registered + emitted via terminal_metrics_dict).
    assert "reward/cmd_yaw_rate_track" in metrics, (
        "reward/cmd_yaw_rate_track was registered + emitted in env "
        "code but is missing from unpacked metrics_vec.  This is the "
        "smoke1 silent-drop signature."
    )
    val = float(metrics["reward/cmd_yaw_rate_track"])
    # Non-zero capable: weight=1.0, TB alpha=0.25, settling gyro_z ~ 0,
    # cmd_wz=0.10 → unweighted kernel = exp(-0.25 * 0.10^2) ≈ 0.9975;
    # post-dt rescale (ctrl_dt = 0.02, CLAUDE.md "reward = sum(...) * dt")
    # gives per-step contribution ≈ 0.020.  Floor the assertion well
    # below that so first-step kernel jitter doesn't fail this
    # regression — the point is to catch silent ZERO (the smoke1
    # W&B-log signature), not to gate magnitude.
    assert abs(val) > 1e-3, (
        f"reward/cmd_yaw_rate_track={val} is effectively zero with "
        f"weight=1.0 and a tracking cmd.  Silent zero means the "
        f"weight × reward multiply was bypassed somewhere."
    )


def test_dim2_makes_reward_metric_reflect_vy_cmd() -> None:
    """Sharper variant of ``test_env_reward_consumes_real_vy_cmd_from_3vec``.
    When ``cmd_velocity_track_dim == 2``, the EMITTED
    ``reward/cmd_forward_velocity_track`` value MUST depend on
    ``vy_cmd``.  Concretely: at a fixed vx_cmd, swapping vy_cmd
    between 0 and 0.10 must change the emitted reward (otherwise
    the 2D path silently degenerated to 1D)."""
    import jax
    import jax.numpy as jp

    from training.core.metrics_registry import METRICS_VEC_KEY, unpack_metrics
    from training.envs.env_info import WR_INFO_KEY

    cfg = _load_smoke14_dim2_lateral_active_cfg()
    env = WildRobotEnv(config=cfg)

    def _step_with_cmd(vy_cmd: float) -> float:
        state = env.reset(jax.random.PRNGKey(0))
        wr = state.info[WR_INFO_KEY]
        new_wr = wr.replace(
            velocity_cmd=jp.array([0.20, vy_cmd, 0.0], dtype=jp.float32)
        )
        state = state.replace(info={**state.info, WR_INFO_KEY: new_wr})
        state2 = env.step(state, jp.zeros(env.action_size))
        m = unpack_metrics(state2.metrics[METRICS_VEC_KEY])
        return float(m["reward/cmd_forward_velocity_track"])

    r_vy0 = _step_with_cmd(0.0)
    r_vy10 = _step_with_cmd(0.10)
    # If dim=2 actually consumes vy_cmd, the 0.10 m/s lateral target
    # MUST change the emitted reward vs vy_cmd=0.  smoke14 has
    # alpha=562.5 + weight 2.0 + post-dt rescale (CLAUDE.md), so the
    # absolute kernel value at large error (~0.20) is ~exp(-562.5 *
    # 0.04) ≈ 1.7e-10 — tiny in absolute terms.  Assert RELATIVE
    # dependence (the dim=2 path must change reward as vy_cmd
    # changes); abs threshold floored at numerical noise.
    assert max(abs(r_vy0), abs(r_vy10)) > 0, (
        "both rewards were exactly zero; cannot evaluate dependence"
    )
    denom = max(abs(r_vy0), abs(r_vy10))
    rel_diff = abs(r_vy0 - r_vy10) / denom
    assert rel_diff > 0.05, (
        f"reward/cmd_forward_velocity_track did NOT vary with vy_cmd "
        f"(r_vy0={r_vy0}, r_vy10={r_vy10}, rel_diff={rel_diff}); "
        f"cmd_velocity_track_dim=2 has degenerated to dim=1."
    )
