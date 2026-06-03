"""Pin the v0.21.0 smoke6 (TB-contract) contract + the home-base RSI behavior.

smoke6 = smoke5 (RSI, forward-first 1D control) but switches the action base
from the time-varying q_ref to a STATIC `home` base — ToddlerBot's recipe
(build the gait from a static home base + RSI).  Adopted on the TB recipe's
merits, NOT failure-recovery: smoke5's q_ref base already walks deterministically
and is the proven fallback.  Levers vs smoke5:
  * env.loc_ref_residual_base: q_ref -> home   (THE lever)
  * env.loc_ref_reset_base:    ref_init -> home
  * residual scales: coverage 0.75 -> 1.1, symmetric per L/R pair

The load-bearing new invariant (vs smoke5's q_ref base): with the home base the
zero-residual target is `home`, so the controller does NOT sit on the RSI frame
at step 0 (like TB).  Instead the coverage-1.1 scales make every RSI frame
REACHABLE from home — |q_ref(f) - home| <= residual_scale per joint — so the
policy CAN hold the mid-stride pose.  That reachability is what this test pins.

Reference: training/CHANGELOG.md [v0.21.0-smoke5-result + smoke6-plan].
"""

from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
import pytest

from training.configs.training_config import load_training_config
from training.envs.wildrobot_env import WildRobotEnv

_CFG = Path("training/configs/ppo_walking_v0210_smoke6_home_rsi.yaml")
_SMOKE5 = Path("training/configs/ppo_walking_v0210_smoke5_rsi.yaml")
_WR = "wr"


@pytest.fixture(scope="module")
def cfg():
    if not _CFG.exists():
        pytest.skip(f"{_CFG.name} not found")
    return load_training_config(str(_CFG))


@pytest.fixture(scope="module")
def env(cfg):
    return WildRobotEnv(config=cfg)


def _root_qvel(state) -> np.ndarray:
    data = getattr(state, "pipeline_state", None) or getattr(state, "data", None)
    return np.asarray(data.qvel)[:3]


# ---------------------------------------------------------------------------
# Config contract
# ---------------------------------------------------------------------------
def test_smoke6_levers(cfg) -> None:
    assert cfg.env.loc_ref_rsi_enabled is True
    # THE lever: static home base (TB contract), not the time-varying q_ref.
    assert cfg.env.loc_ref_residual_base == "home"
    assert cfg.env.loc_ref_reset_base == "home"


def test_smoke6_scales_symmetric_and_coverage11(cfg) -> None:
    """Residual scales must be symmetric per L/R pair AND coverage-1.1 sized so
    the prior is reachable from home (knee swing 0.889 -> >= 0.889)."""
    s = cfg.env.loc_ref_residual_scale_per_joint
    for joint in ("hip_pitch", "knee_pitch", "ankle_pitch", "hip_roll", "ankle_roll"):
        assert s[f"left_{joint}"] == pytest.approx(s[f"right_{joint}"]), (
            f"{joint} L/R scales must be symmetric for straight walking"
        )
    # sized to cover the prior's MAX swing over the FULL command range (all
    # library bins), not just the single-vx 0.1333 derivation point:
    # measured all-bins max — knee 0.889, hip_pitch 0.601, ankle_pitch 0.308.
    assert s["left_knee_pitch"] >= 0.889, "knee scale must cover the all-bins prior swing"
    assert s["left_hip_pitch"] >= 0.601, "hip_pitch scale must cover the all-bins prior swing"
    assert s["left_ankle_pitch"] >= 0.308, "ankle_pitch scale must cover the all-bins prior swing"


def test_smoke6_keeps_forward_first(cfg) -> None:
    e, rw = cfg.env, cfg.reward_weights
    assert e.cmd_sampler_3d_branched is False
    assert rw.cmd_velocity_track_dim == 1
    assert e.min_velocity == pytest.approx(0.065)
    assert e.max_velocity == pytest.approx(0.26)
    assert rw.cmd_forward_velocity_alpha == pytest.approx(562.5)


# ---------------------------------------------------------------------------
# Home-base RSI behavior (the lever) — validated empirically
# ---------------------------------------------------------------------------
def test_smoke6_train_reset_starts_moving_forward(env) -> None:
    n_steps = int(env._offline_service.n_steps)
    seen = set()
    for s in range(6):
        st = env.reset(jax.random.PRNGKey(s))
        vx, _, _ = _root_qvel(st)
        sidx = int(np.asarray(st.info[_WR].loc_ref_offline_step_idx))
        assert vx > 0.05, f"seed {s}: expected forward root velocity, got {vx}"
        assert 0 <= sidx < n_steps
        seen.add(sidx)
    assert len(seen) > 1  # frame randomized


def test_smoke6_all_bins_reachable_from_home(env) -> None:
    """Load-bearing reachability invariant — the GENERAL contract (not just a
    few sampled resets): EVERY library bin's per-joint prior swing from home must
    fit within the residual scale, so NO RSI reset can land off-manifold.  This
    mirrors the env-init guard and is what makes the static home base + RSI safe.
    (Computed directly from the library, so it covers high-vx bins that random
    sampling can miss — the gap that made the old 6-sample check insufficient.)"""
    qref = np.asarray(env._offline_jax_arrays["q_ref"])  # (bins, steps, n_act)
    home = np.asarray(env._home_q_rad)
    scale = np.asarray(env._residual_q_scale_per_joint)
    amp = np.max(np.abs(qref - home[None, None, :]), axis=(0, 1))  # per joint
    over = amp - scale
    j = int(over.argmax())
    assert over.max() <= 1e-3, (
        f"library bin unreachable from home: joint {j} prior swing "
        f"{amp[j]:.3f} > scale {scale[j]:.3f} (short by {over.max():.3f}) — widen scale"
    )


def test_smoke6_rsi_undersized_scales_raise(cfg) -> None:
    """The env-init reachability guard MUST hard-fail home+RSI with undersized
    scales (the original off-manifold bug).  Shrink the knee scale below the
    prior swing (0.889) and confirm init raises."""
    import dataclasses

    bad_scales = dict(cfg.env.loc_ref_residual_scale_per_joint)
    bad_scales["left_knee_pitch"] = 0.3
    bad_scales["right_knee_pitch"] = 0.3
    bad_env = dataclasses.replace(
        cfg.env, loc_ref_residual_scale_per_joint=bad_scales
    )
    bad_cfg = dataclasses.replace(cfg, env=bad_env)
    with pytest.raises(ValueError, match="prior swing"):
        WildRobotEnv(config=bad_cfg)


def test_smoke6_eval_reset_is_static(env) -> None:
    st = env.reset_for_eval(jax.random.PRNGKey(0))
    vx, vy, vz = _root_qvel(st)
    assert abs(vx) < 1e-6 and abs(vy) < 1e-6 and abs(vz) < 1e-6
    assert int(np.asarray(st.info[_WR].loc_ref_offline_step_idx)) == 0


# ---------------------------------------------------------------------------
# Guard: RSI now allows {q_ref, home}; ref_init still rejected
# ---------------------------------------------------------------------------
def test_rsi_allows_home_base(cfg) -> None:
    """home + RSI must NOT raise (the smoke5 guard that forbade it was built on
    a falsified premise; TB runs RSI from a static home base)."""
    WildRobotEnv(config=cfg)  # constructed in fixture too; explicit here


def test_rsi_rejects_ref_init_base(cfg) -> None:
    import dataclasses

    bad_env = dataclasses.replace(cfg.env, loc_ref_residual_base="ref_init")
    bad_cfg = dataclasses.replace(cfg, env=bad_env)
    with pytest.raises(ValueError, match="loc_ref_rsi_enabled"):
        WildRobotEnv(config=bad_cfg)


def test_smoke6_lateral_factor_neutral_under_dim1(env, cfg) -> None:
    """smoke8 fix: on a track_dim==1 (forward-only) config the lateral axis is
    NOT in the reward, so reward/cmd_lateral_velocity_track must be a NEUTRAL 1.0
    (not the misleading exp(-alpha_x*vy^2) it would be if it fell back to the
    forward alpha).  tracking/cmd_vy_err is still emitted (always-valid)."""
    import jax.numpy as jp
    from training.core.metrics_registry import METRICS_VEC_KEY, unpack_metrics

    assert cfg.reward_weights.cmd_velocity_track_dim == 1  # precondition

    st = env.reset(jax.random.PRNGKey(0))
    m = unpack_metrics(st.metrics[METRICS_VEC_KEY])
    assert float(m["reward/cmd_lateral_velocity_track"]) == pytest.approx(1.0)
    assert "tracking/cmd_vy_err" in m  # still present (diagnostic)

    act = jp.zeros(env.action_size)
    for _ in range(3):
        st = env.step(st, act)
    m = unpack_metrics(st.metrics[METRICS_VEC_KEY])
    assert float(m["reward/cmd_lateral_velocity_track"]) == pytest.approx(1.0)
