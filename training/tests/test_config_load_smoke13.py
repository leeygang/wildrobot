"""Contract tests for smoke13 (TB-style direct-policy walking migration).

Pin the six architectural invariants from the migration plan:

  1. Config loads and selects ``action_mode == "direct"``.
  2. Zero action targets exactly ``_home_q_rad`` (no residual_base
     switching, no q_ref offset).
  3. Action path does not depend on ``loc_ref_residual_base`` —
     flipping that field while ``action_mode == "direct"`` leaves
     ``target_q`` unchanged.
  4. Actor observation layout is phase + proprio only
     (``wr_obs_v7_phase_proprio``), with no offline-reference window
     channels in the actor stream.
  5. Native-MuJoCo eval adapter (``V6EvalAdapter``) routes the same
     direct-policy semantics so visualize_policy.py / eval ladders
     are contract-matched.
  6. Reward block matches TB-active (walk.gin:110-131); every
     hybrid/imitation term is explicitly zeroed.

Spec: /tmp/wr_tb_style_direct_policy_migration_plan.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from training.configs.training_config import load_training_config


_SMOKE13 = Path("training/configs/ppo_walking_v0201_smoke13.yaml")


@pytest.fixture(scope="module")
def smoke13_cfg():
    if not _SMOKE13.exists():
        pytest.skip(f"{_SMOKE13.name} not found")
    return load_training_config(str(_SMOKE13))


# ---------------------------------------------------------------------------
# 1. Config loads and selects the new direct-policy action mode.
# ---------------------------------------------------------------------------
def test_smoke13_loads(smoke13_cfg) -> None:
    assert smoke13_cfg is not None


def test_smoke13_selects_action_mode_direct(smoke13_cfg) -> None:
    """The architectural promise of smoke13.  If this flips back to
    'residual' the entire migration is vacuous."""
    assert smoke13_cfg.env.action_mode == "direct"


# ---------------------------------------------------------------------------
# 2. Action path: zero action -> exactly q_default (= _home_q_rad).
# 3. Action path independent of loc_ref_residual_base.
# ---------------------------------------------------------------------------
def _make_env(cfg):
    # Importing the env is heavy (mjx + mujoco + JAX); keep it inside
    # the test functions so import-only test discovery stays cheap.
    from training.envs.wildrobot_env import WildRobotEnv
    return WildRobotEnv(config=cfg)


def test_smoke13_zero_action_targets_home(smoke13_cfg) -> None:
    """Zero policy_action under direct mode must compose to exactly
    ``_home_q_rad`` (clipped to joint range, which is a no-op because
    home is already inside the range).  Pins the TB-equivalence
    invariant ``target_q = home + 0.25 * 0 = home``."""
    import jax.numpy as jp

    env = _make_env(smoke13_cfg)
    zero_action = jp.zeros((env.action_size,), dtype=jp.float32)
    win0 = env._lookup_offline_window(jp.asarray(0, dtype=jp.int32))
    target_q, residual_delta = env._compose_target_q_from_residual(
        policy_action=zero_action, nominal_q_ref=win0["q_ref"]
    )
    assert np.allclose(np.asarray(residual_delta), 0.0, atol=1e-7)
    assert np.allclose(
        np.asarray(target_q), np.asarray(env._home_q_rad), atol=1e-6
    ), "smoke13 direct mode: zero action must target exactly home_q_rad"


def test_smoke13_action_path_ignores_residual_base(smoke13_cfg) -> None:
    """Under ``action_mode == "direct"`` the env's
    ``_compose_target_q_from_residual`` must NOT consult
    ``_residual_base_mode``.  Verified by flipping the cached base mode
    on a built env and checking that ``target_q`` for the same
    non-zero action is unchanged.

    This guards against a future regression that drops the
    ``action_mode == "direct"`` short-circuit and reintroduces the
    residual_base switch."""
    import jax.numpy as jp

    env = _make_env(smoke13_cfg)
    action = jp.full((env.action_size,), 0.3, dtype=jp.float32)
    win0 = env._lookup_offline_window(jp.asarray(0, dtype=jp.int32))

    original_base_mode = env._residual_base_mode
    target_home, _ = env._compose_target_q_from_residual(
        policy_action=action, nominal_q_ref=win0["q_ref"]
    )
    try:
        env._residual_base_mode = "q_ref"
        target_qref, _ = env._compose_target_q_from_residual(
            policy_action=action, nominal_q_ref=win0["q_ref"]
        )
        env._residual_base_mode = "ref_init"
        target_refinit, _ = env._compose_target_q_from_residual(
            policy_action=action, nominal_q_ref=win0["q_ref"]
        )
    finally:
        env._residual_base_mode = original_base_mode

    th = np.asarray(target_home)
    assert np.allclose(np.asarray(target_qref), th, atol=1e-6), (
        "smoke13: action path must not depend on residual_base when "
        "action_mode == 'direct'"
    )
    assert np.allclose(np.asarray(target_refinit), th, atol=1e-6), (
        "smoke13: action path must not depend on residual_base when "
        "action_mode == 'direct'"
    )


# ---------------------------------------------------------------------------
# 4. Actor observation layout contains no hybrid offline-ref channels.
# ---------------------------------------------------------------------------
def test_smoke13_actor_obs_layout_is_phase_proprio_only(smoke13_cfg) -> None:
    """The actor observation must be ``wr_obs_v7_phase_proprio`` —
    v3 base (gravity + gyro + jpos + jvel + foot_switches +
    prev_action + velocity_cmd) + phase(2) + proprio_history.  No
    offline-reference window channels (loc_ref_q_ref, pelvis,
    per-foot pos/vel, contact_mask) in the actor stream.

    The layout dispatch in ``policy_contract/numpy/obs.py`` for
    ``wr_obs_v7_phase_proprio`` already enforces this — pinning the
    config selector here keeps the contract anchored at the config
    layer too."""
    assert (
        smoke13_cfg.env.actor_obs_layout_id == "wr_obs_v7_phase_proprio"
    )


# ---------------------------------------------------------------------------
# 5. Native-MuJoCo eval adapter uses the same direct-policy semantics.
# ---------------------------------------------------------------------------
def test_smoke13_v6_adapter_uses_direct_mode(smoke13_cfg) -> None:
    """``V6EvalAdapter`` must read the same ``action_mode`` field and
    bypass the residual_base switch, so visualize_policy.py and any
    native-MuJoCo eval ladder produces the same target_q as the
    training env."""
    import mujoco
    from training.eval.v6_eval_adapter import V6EvalAdapter
    from training.policy_spec_utils import build_policy_spec_from_training_config
    from training.sim_adapter.mujoco_signals import MujocoSignalsAdapter
    from training.configs.training_config import load_robot_config

    mj_model = mujoco.MjModel.from_xml_path(str(Path(smoke13_cfg.env.model_path)))
    rcfg = load_robot_config(smoke13_cfg.env.robot_config_path)
    spec = build_policy_spec_from_training_config(
        training_cfg=smoke13_cfg, robot_cfg=rcfg, action_filter_alpha=0.0
    )
    sigA = MujocoSignalsAdapter(
        mj_model=mj_model,
        robot_config=rcfg,
        policy_spec=spec,
        foot_switch_threshold=smoke13_cfg.env.foot_switch_threshold,
    )
    adapter = V6EvalAdapter(
        training_cfg=smoke13_cfg,
        mj_model=mj_model,
        policy_spec=spec,
        signals_adapter=sigA,
        action_dim=int(spec.model.action_dim),
    )
    assert adapter._action_mode == "direct"
    # Direct mode's ctrl_init must be home, not ref_init / q_ref.
    np.testing.assert_allclose(
        adapter._ctrl_init_policy_order(), adapter._home_q_rad, atol=1e-6
    )


# ---------------------------------------------------------------------------
# 6. Reward block matches TB-active; hybrid/imitation terms absent or zero.
# ---------------------------------------------------------------------------
_TB_ACTIVE_NONZERO = {
    # walk.gin row -> expected smoke13 weight
    "alive": 1.0,                       # walk.gin:127
    "cmd_forward_velocity_track": 2.0,  # walk.gin:111 (lin_vel_xy)
    "ref_body_quat_track": 2.5,         # walk.gin:117 (torso_quat)
    "ang_vel_xy": 1.0,                  # walk.gin:115 (penalty_ang_vel_xy)
    "action_rate": -2.0,                # walk.gin:119 (penalty_action_rate)
    "penalty_pose": -0.5,               # walk.gin:120
    "penalty_close_feet_xy": 10.0,      # walk.gin:126
    "feet_phase": 7.5,                  # walk.gin:128
    "penalty_feet_ori": 5.0,            # walk.gin:129
}

_HYBRID_MUST_BE_ZERO = (
    "ref_q_track",          # joint-imitation; TB-active drops
    "ref_contact_match",    # contact-imitation
    "ref_feet_z_track",     # foot-z imitation
    "torso_pos_xy",         # torso-XY imitation
    "lin_vel_z",            # vertical-bobbing imitation
    "feet_air_time",        # sparse; replaced by feet_phase
    "feet_clearance",       # sparse; replaced by feet_phase
    "feet_distance",        # superseded by penalty_close_feet_xy
    "torso_pitch_soft",     # superseded by penalty_ang_vel_xy
    "torso_roll_soft",      # superseded by penalty_ang_vel_xy
    "torque",               # not in TB-active
    "joint_velocity",       # not in TB-active
    "slip",                 # not in TB-active
    "pitch_rate",           # M1 diagnostic hook only
)


def test_smoke13_reward_block_matches_tb_active(smoke13_cfg) -> None:
    w = smoke13_cfg.reward_weights
    for name, expected in _TB_ACTIVE_NONZERO.items():
        got = getattr(w, name)
        assert got == pytest.approx(expected), (
            f"smoke13 reward_weights.{name} = {got}, expected {expected} "
            "(TB-active walk.gin:110-131)"
        )


def test_smoke13_hybrid_rewards_explicitly_zero(smoke13_cfg) -> None:
    w = smoke13_cfg.reward_weights
    for name in _HYBRID_MUST_BE_ZERO:
        got = getattr(w, name)
        assert got == pytest.approx(0.0), (
            f"smoke13 reward_weights.{name} = {got}, must be 0.0 "
            "(hybrid / imitation term; explicitly disabled in TB-active migration)"
        )


# ---------------------------------------------------------------------------
# Anchor / scale / reset invariants (cross-checks the architecture).
# ---------------------------------------------------------------------------
def test_smoke13_uses_scalar_tb_action_scale(smoke13_cfg) -> None:
    """TB-faithful scalar 0.25 (mjx_config.py:103).  No per-joint
    widening in this branch — that was a smoke12 diagnostic for the
    residual contract."""
    env = smoke13_cfg.env
    assert env.loc_ref_residual_scale == pytest.approx(0.25)
    assert env.loc_ref_residual_scale_per_joint == {}


def test_smoke13_reset_and_penalty_pose_anchored_to_home(smoke13_cfg) -> None:
    env = smoke13_cfg.env
    assert env.loc_ref_reset_base == "home"
    assert env.loc_ref_penalty_pose_anchor == "home"
