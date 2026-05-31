"""Pin the v0.21.0 smoke5 (RSI) contract + the RSI reset behavior.

smoke5 = smoke4A (forward-first 1D control) + Reference State Initialization:
TRAIN episodes reset onto a random frame of the moving reference gait (qpos +
reference qvel) so the velocity reward is live at step 0; eval resets stay
static (from rest).  Levers vs smoke4A:
  * env.loc_ref_rsi_enabled: false -> true
  * env.loc_ref_residual_base: home -> ref_init  (reachability; TB first_frame_ref analog)
  * env.loc_ref_reset_base:    home -> ref_init

Reference: training/docs/smoke5_rsi_proposal.md.
"""

from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
import pytest

from training.configs.training_config import load_training_config
from training.envs.wildrobot_env import WildRobotEnv

_CFG = Path("training/configs/ppo_walking_v0210_smoke5_rsi.yaml")
_SMOKE4 = Path("training/configs/ppo_walking_v0210_smoke4_forward_first.yaml")
_WR = "wr"  # WildRobotInfo key in state.info


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
def test_smoke5_rsi_levers(cfg) -> None:
    assert cfg.env.loc_ref_rsi_enabled is True
    # base must FOLLOW q_ref(t): WR gait amplitude (0.91 rad) >> the ±0.25
    # residual bound, so a static base (home/ref_init) is off-manifold/unreachable.
    assert cfg.env.loc_ref_residual_base == "q_ref"
    assert cfg.env.loc_ref_reset_base == "ref_init"


def test_smoke5_keeps_smoke4a_forward_first(cfg) -> None:
    """Everything that defines smoke4A's forward-first 1D control is intact."""
    e, rw = cfg.env, cfg.reward_weights
    assert e.cmd_sampler_3d_branched is False
    assert rw.cmd_velocity_track_dim == 1
    assert rw.cmd_yaw_rate_track == pytest.approx(0.0)
    assert e.min_velocity == pytest.approx(0.065)
    assert e.max_velocity == pytest.approx(0.26)
    assert e.loc_ref_command_grid_interval == pytest.approx(0.065)
    assert rw.cmd_forward_velocity_alpha == pytest.approx(562.5)


# ---------------------------------------------------------------------------
# RSI reset behavior (the lever) — validated empirically
# ---------------------------------------------------------------------------
def test_smoke5_train_reset_starts_moving_forward(env) -> None:
    """TRAIN reset (perturb_pose default True) seeds the reference velocity:
    the robot starts moving forward at ≈cmd, at a RANDOM gait frame."""
    n_steps = int(env._offline_service.n_steps)
    seen_idx = set()
    for s in range(6):
        st = env.reset(jax.random.PRNGKey(s))
        vx, _, _ = _root_qvel(st)
        sidx = int(np.asarray(st.info[_WR].loc_ref_offline_step_idx))
        assert vx > 0.05, f"seed {s}: expected forward root velocity, got {vx}"
        assert 0 <= sidx < n_steps
        seen_idx.add(sidx)
    assert len(seen_idx) > 1  # frame is randomized across resets


def test_smoke5_rsi_reset_is_on_manifold(env) -> None:
    """Coherence (load-bearing): the step-0 control base must MATCH the sampled
    RSI frame's pose, else the controller pulls the body off the gait manifold
    at step 0.  With the q_ref base, ctrl@reset = q_ref(f) ≈ body actuated qpos
    (both at frame f).  A static base would be off by ~0.5 rad (> residual bound).
    """
    for s in range(4):
        st = env.reset(jax.random.PRNGKey(s))
        data = getattr(st, "pipeline_state", None) or getattr(st, "data", None)
        ctrl = np.asarray(data.ctrl)
        body_act = np.asarray(data.qpos)[np.asarray(env._actuator_qpos_addrs)]
        # zero-residual control target tracks the RSI frame's joint pose
        assert np.max(np.abs(ctrl - body_act)) < 0.25, (
            f"seed {s}: control base off the RSI frame by "
            f"{np.max(np.abs(ctrl - body_act)):.3f} rad (> residual bound)"
        )


def test_smoke5_eval_reset_is_static(env) -> None:
    """Eval reset (perturb_pose=False) stays at rest, frame 0 — deployment
    starts from a standstill."""
    st = env.reset_for_eval(jax.random.PRNGKey(0))
    vx, vy, vz = _root_qvel(st)
    assert abs(vx) < 1e-6 and abs(vy) < 1e-6 and abs(vz) < 1e-6
    assert int(np.asarray(st.info[_WR].loc_ref_offline_step_idx)) == 0


def test_smoke4a_reset_is_static_rsi_off() -> None:
    """Off-by-default regression: smoke4A (loc_ref_rsi_enabled absent/False)
    resets at rest, frame 0 — RSI changes nothing when disabled."""
    if not _SMOKE4.exists():
        pytest.skip("smoke4A config not found")
    env4 = WildRobotEnv(config=load_training_config(str(_SMOKE4)))
    st = env4.reset(jax.random.PRNGKey(0))
    assert np.allclose(_root_qvel(st), 0.0, atol=1e-6)
    assert int(np.asarray(st.info[_WR].loc_ref_offline_step_idx)) == 0


# ---------------------------------------------------------------------------
# Required residual-base alignment (env init guard)
# ---------------------------------------------------------------------------
def test_rsi_with_home_residual_base_raises(cfg) -> None:
    """RSI + home control base is rejected at env init (home can sit >0.25 rad
    from mid-stride frames → unreachable within the residual bound)."""
    import dataclasses

    bad_env = dataclasses.replace(cfg.env, loc_ref_residual_base="home")
    bad_cfg = dataclasses.replace(cfg, env=bad_env)
    with pytest.raises(ValueError, match="loc_ref_rsi_enabled"):
        WildRobotEnv(config=bad_cfg)
