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
    # coverage-1.1 sizing on the load-bearing walking joints
    assert s["left_knee_pitch"] >= 0.889, "knee scale must cover the prior swing"
    assert s["left_hip_pitch"] >= 0.514, "hip_pitch scale must cover the prior swing"


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


def test_smoke6_rsi_frame_reachable_from_home(env) -> None:
    """Load-bearing coverage-1.1 invariant: every RSI reset frame's actuated
    pose is within the per-joint residual scale of `home`, so the home + ±scale
    controller CAN hold it (no permanent step-0 snap).  This is what coverage
    1.1 buys vs smoke5's coverage 0.75 (where a static base was unreachable)."""
    home = np.asarray(env._home_q_rad)
    scale = np.asarray(env._residual_q_scale_per_joint)
    addrs = np.asarray(env._actuator_qpos_addrs)
    worst = 0.0
    for s in range(6):
        st = env.reset(jax.random.PRNGKey(s))
        data = getattr(st, "pipeline_state", None) or getattr(st, "data", None)
        body_act = np.asarray(data.qpos)[addrs]
        over = np.abs(body_act - home) - scale  # <=0 means reachable
        worst = max(worst, float(over.max()))
        assert over.max() <= 1e-3, (
            f"seed {s}: RSI frame unreachable from home by {over.max():.3f} rad "
            f"(joint {int(over.argmax())}) — widen coverage"
        )


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
