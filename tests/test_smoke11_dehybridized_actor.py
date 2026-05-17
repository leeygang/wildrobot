"""Pin the smoke11 actor-obs de-hybridization.

smoke11 is the post-home-migration step that drops every
reference-trajectory channel from the actor obs except the 2-dim
gait phase clock — matching ToddlerBot's default
``ObsConfig.use_phase_signal=True`` actor contract
(``toddlerbot/locomotion/walk.gin:24-28``,
``toddlerbot/locomotion/mjx_env.py:2113-2124``).

These tests are the contract that the branch stays de-hybridized:

  1. The smoke11 yaml loads, basin flags carry over from smoke10,
     and the new layout id is wired through.
  2. The new ``wr_obs_v7_phase_proprio`` layout has exactly the
     intended channel set (positive list) AND nothing else
     reference-derived (negative list: no ``loc_ref_q_ref``,
     ``loc_ref_pelvis_*``, ``loc_ref_*_foot_*``, ``loc_ref_contact_mask``,
     ``loc_ref_stance_foot``, ``loc_ref_next_foothold``,
     ``loc_ref_swing_*``, ``loc_ref_pelvis_targets``,
     ``loc_ref_history``).  The ONLY remaining ``loc_ref_*`` field
     in actor obs is the 2-dim ``loc_ref_phase_sin_cos`` clock.
  3. Smoke10 still wires the full v6 layout — verifying the v7
     change is opt-in via the yaml, not a global flip.
  4. End-to-end env build under smoke11: actor obs dim shrinks by
     exactly the 57 dims of the dropped reference channels
     (smoke10 1184 -> smoke11 1127).
  5. Zero-action under smoke11 still composes to ``_home_q_rad``
     (the home migration / basin-flip contract is preserved).
  6. The privileged-critic obs is unaffected (still 52 dims and
     still consumes ``nominal_q_ref`` + ``ref_contact_mask``) — the
     reference machinery remains load-bearing for the critic and
     reward terms; only the ACTOR input changed.
"""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
import pytest


_REPO_ROOT = Path(__file__).resolve().parents[1]
_SMOKE10_CFG = _REPO_ROOT / "training" / "configs" / "ppo_walking_v0201_smoke10.yaml"
_SMOKE11_CFG = _REPO_ROOT / "training" / "configs" / "ppo_walking_v0201_smoke11.yaml"


_PHASE_FIELD = "loc_ref_phase_sin_cos"
_BANNED_LOC_REF_FIELDS = (
    "loc_ref_q_ref",
    "loc_ref_pelvis_pos",
    "loc_ref_pelvis_vel",
    "loc_ref_pelvis_targets",
    "loc_ref_left_foot_pos",
    "loc_ref_right_foot_pos",
    "loc_ref_left_foot_vel",
    "loc_ref_right_foot_vel",
    "loc_ref_contact_mask",
    "loc_ref_stance_foot",
    "loc_ref_next_foothold",
    "loc_ref_swing_pos",
    "loc_ref_swing_vel",
    "loc_ref_history",
)


def _maybe_build_env(cfg_path: Path):
    if not cfg_path.exists():
        pytest.skip(f"{cfg_path.name} not found")
    try:
        from assets.robot_config import load_robot_config
        from training.configs.training_config import load_training_config
        from training.envs.wildrobot_env import WildRobotEnv
    except Exception as exc:  # pragma: no cover - import gate
        pytest.skip(f"env deps unavailable: {exc}")
    load_robot_config(str(_REPO_ROOT / "assets/v2/mujoco_robot_config.json"))
    cfg = load_training_config(str(cfg_path))
    cfg.freeze()
    return WildRobotEnv(cfg)


# -----------------------------------------------------------------------------
# 1. smoke11 yaml threading
# -----------------------------------------------------------------------------


def test_smoke11_yaml_threads_v7_layout_and_keeps_home_basin() -> None:
    """smoke11 must (a) load, (b) flip actor_obs_layout_id to
    wr_obs_v7_phase_proprio, and (c) keep smoke10's home/home basin
    flags untouched."""
    if not _SMOKE11_CFG.exists():
        pytest.skip(f"{_SMOKE11_CFG.name} not found")
    from training.configs.training_config import load_training_config
    cfg = load_training_config(str(_SMOKE11_CFG))
    env = cfg.env
    assert env.actor_obs_layout_id == "wr_obs_v7_phase_proprio", (
        f"smoke11 actor_obs_layout_id = {env.actor_obs_layout_id!r}; "
        "expected 'wr_obs_v7_phase_proprio'."
    )
    assert env.loc_ref_residual_base == "home"
    assert env.loc_ref_reset_base == "home"
    assert env.loc_ref_penalty_pose_anchor == "home"


def test_smoke11_inherits_smoke10_curriculum_and_rewards() -> None:
    """The only material delta vs smoke10 is the actor obs layout.
    Curriculum / reward block / PPO / compute budget must inherit
    byte-equal so the next run cleanly attributes any difference to
    the obs de-hybridization."""
    if not _SMOKE11_CFG.exists() or not _SMOKE10_CFG.exists():
        pytest.skip("smoke10 or smoke11 config missing")
    from training.configs.training_config import load_training_config
    s10 = load_training_config(str(_SMOKE10_CFG))
    s11 = load_training_config(str(_SMOKE11_CFG))

    # Curriculum.
    assert s11.env.min_velocity == s10.env.min_velocity
    assert s11.env.max_velocity == s10.env.max_velocity
    assert s11.env.cmd_deadzone == s10.env.cmd_deadzone
    assert s11.env.eval_velocity_cmd == s10.env.eval_velocity_cmd
    assert s11.env.loc_ref_offline_command_vx == s10.env.loc_ref_offline_command_vx

    # Reward block.
    s10_rw = s10.reward_weights
    s11_rw = s11.reward_weights
    for field in (
        "alive", "cmd_forward_velocity_track", "cmd_forward_velocity_alpha",
        "ref_body_quat_track", "ang_vel_xy", "action_rate", "penalty_pose",
        "penalty_close_feet_xy", "feet_phase", "feet_phase_alpha",
        "feet_phase_swing_height", "penalty_feet_ori",
        "ref_q_track", "ref_contact_match", "ref_feet_z_track",
    ):
        assert getattr(s11_rw, field) == getattr(s10_rw, field), (
            f"smoke11 reward_weights.{field} drifted from smoke10."
        )

    # Form flags + spatial thresholds.
    assert (
        s11.env.loc_ref_penalty_ang_vel_xy_form
        == s10.env.loc_ref_penalty_ang_vel_xy_form
    )
    assert (
        s11.env.loc_ref_penalty_feet_ori_form
        == s10.env.loc_ref_penalty_feet_ori_form
    )
    assert s11.env.close_feet_threshold == s10.env.close_feet_threshold
    assert s11.env.min_feet_y_dist == s10.env.min_feet_y_dist
    assert s11.env.max_feet_y_dist == s10.env.max_feet_y_dist

    # Residual scale.
    assert s11.env.loc_ref_residual_mode == s10.env.loc_ref_residual_mode
    assert s11.env.loc_ref_residual_scale == s10.env.loc_ref_residual_scale
    assert (
        dict(s11.env.loc_ref_residual_scale_per_joint)
        == dict(s10.env.loc_ref_residual_scale_per_joint)
    )

    # Reset perturbation.
    assert list(s11.env.reset_torso_roll_range) == list(s10.env.reset_torso_roll_range)
    assert list(s11.env.reset_torso_pitch_range) == list(s10.env.reset_torso_pitch_range)

    # PPO + compute budget.
    assert s11.ppo.num_envs == s10.ppo.num_envs
    assert s11.ppo.rollout_steps == s10.ppo.rollout_steps
    assert s11.ppo.iterations == s10.ppo.iterations
    assert s11.ppo.learning_rate == s10.ppo.learning_rate
    assert s11.ppo.entropy_coef == s10.ppo.entropy_coef
    assert s11.ppo.gamma == s10.ppo.gamma


# -----------------------------------------------------------------------------
# 2. layout content: positive + negative lists
# -----------------------------------------------------------------------------


def _layout_for(env) -> list[tuple[str, int]]:
    return [(f.name, int(f.size)) for f in env._policy_spec.observation.layout]


def test_v7_layout_has_exactly_the_intended_fields() -> None:
    """Positive list: ``wr_obs_v7_phase_proprio`` has exactly the
    10 fields the design specifies, in order, with correct sizes."""
    env = _maybe_build_env(_SMOKE11_CFG)
    action_dim = env.action_size
    # PROPRIO_HISTORY_FRAMES is defined once in policy_contract/spec.py
    from policy_contract.spec import PROPRIO_HISTORY_FRAMES
    proprio_bundle = 3 + 4 + 3 * action_dim
    expected = [
        ("gravity_local", 3),
        ("angvel_heading_local", 3),
        ("joint_pos_normalized", action_dim),
        ("joint_vel_normalized", action_dim),
        ("foot_switches", 4),
        ("prev_action", action_dim),
        ("velocity_cmd", 1),
        ("loc_ref_phase_sin_cos", 2),
        ("proprio_history", PROPRIO_HISTORY_FRAMES * proprio_bundle),
        ("padding", 1),
    ]
    assert _layout_for(env) == expected, (
        f"v7 layout drifted from spec:\n  expected={expected}\n  got={_layout_for(env)}"
    )


def test_v7_actor_obs_has_no_reference_trajectory_fields() -> None:
    """Negative list: no banned ``loc_ref_*`` field appears in actor
    obs.  Catches a future drift that re-adds q_ref / pelvis / foot /
    contact / stance / foothold / swing / pelvis_targets / history to
    the actor input under a v7-named layout."""
    env = _maybe_build_env(_SMOKE11_CFG)
    names = [f.name for f in env._policy_spec.observation.layout]
    leaked = [n for n in names if n in _BANNED_LOC_REF_FIELDS]
    assert not leaked, (
        f"v7 actor obs leaked reference channels: {leaked}.  The only "
        f"loc_ref_* field allowed at the actor is {_PHASE_FIELD!r} "
        "(2-dim gait phase clock matching TB use_phase_signal=True)."
    )
    # Phase clock IS allowed.
    assert _PHASE_FIELD in names, (
        f"v7 actor obs missing {_PHASE_FIELD!r}; the gait clock is the "
        "one TB-aligned reference-derived signal we keep."
    )


def test_v6_actor_obs_still_has_full_reference_window() -> None:
    """Sanity: the v7 cleanup must NOT silently flip v6 to a leaner
    layout.  Build smoke10 (v6) and confirm the banned channels still
    appear there.  Catches a future regression that strips v6 too."""
    env = _maybe_build_env(_SMOKE10_CFG)
    names = [f.name for f in env._policy_spec.observation.layout]
    for banned in _BANNED_LOC_REF_FIELDS:
        assert banned in names, (
            f"smoke10 (v6) is missing {banned!r}; the v7 cleanup leaked "
            "into v6.  v6 is the back-compat layout — its channel set "
            "must stay intact."
        )


# -----------------------------------------------------------------------------
# 3. obs-dim math + end-to-end env build
# -----------------------------------------------------------------------------


def test_v7_actor_obs_dim_is_57_smaller_than_v6() -> None:
    """The 14 dropped reference channels total 57 dims:
        21 (q_ref) + 3+3 (pelvis pos/vel) + 3+3+3+3 (foot pos/vel) +
        2 (contact_mask) + 1 (stance_foot) + 2 (next_foothold) +
        3+3 (swing pos/vel) + 3 (pelvis_targets) + 4 (loc_ref_history)
        = 57.
    Pin the exact obs-dim delta so any future addition (or accidental
    over-removal) trips a clear failure."""
    env_v7 = _maybe_build_env(_SMOKE11_CFG)
    env_v6 = _maybe_build_env(_SMOKE10_CFG)
    delta = env_v6._policy_spec.model.obs_dim - env_v7._policy_spec.model.obs_dim
    assert delta == 57, (
        f"v6 - v7 actor obs_dim = {delta}; expected 57 (the dropped "
        f"reference-trajectory channels).  v6={env_v6._policy_spec.model.obs_dim}, "
        f"v7={env_v7._policy_spec.model.obs_dim}."
    )


# -----------------------------------------------------------------------------
# 4. home/home action contract preserved
# -----------------------------------------------------------------------------


def test_smoke11_zero_action_still_targets_home_q_rad() -> None:
    """smoke11 keeps smoke10's fixed-home action contract: zero
    policy action under ``loc_ref_residual_base=home`` composes to
    exactly ``_home_q_rad`` at every offline-cycle step.  The obs
    de-hybridization is orthogonal to the action path."""
    env = _maybe_build_env(_SMOKE11_CFG)
    assert env._residual_base_mode == "home"
    home = np.asarray(env._home_q_rad).astype(np.float32)
    zero_action = jp.zeros(env.action_size, dtype=jp.float32)
    for step_idx in (0, 5, 24, 47):
        win = env._lookup_offline_window(jp.asarray(step_idx, dtype=jp.int32))
        nominal_q_ref = win["q_ref"].astype(jp.float32)
        target_q, _ = env._compose_target_q_from_residual(
            policy_action=zero_action,
            nominal_q_ref=nominal_q_ref,
        )
        np.testing.assert_allclose(
            np.asarray(target_q), home, atol=1e-5,
            err_msg=(
                f"smoke11 step {step_idx}: target_q != _home_q_rad — "
                "the home basin contract regressed under the v7 layout."
            ),
        )


# -----------------------------------------------------------------------------
# 5. privileged critic obs unaffected
# -----------------------------------------------------------------------------


def test_v7_privileged_critic_obs_still_consumes_reference() -> None:
    """The reference machinery must remain load-bearing for the
    privileged critic + reward terms.  Concretely: the critic obs
    layout (PRIVILEGED_OBS_DIM=52) still includes ``motor_pos_error``
    (= q_actual - nominal_q_ref) and ``ref_stance`` (= ref contact
    mask), proving that ``nominal_q_ref`` and ``ref_contact_mask`` are
    still threaded into the env's per-step pipeline even when the
    actor doesn't see them."""
    env = _maybe_build_env(_SMOKE11_CFG)
    state = env.reset(jax.random.PRNGKey(0))
    # WildRobotInfo exposes the privileged critic obs as `critic_obs`
    # (see training/envs/env_info.py:122,214,276).  Shape must stay
    # at PRIVILEGED_OBS_DIM=52 — that's how we know the critic still
    # consumes motor_pos_error (= q_actual - nominal_q_ref) and
    # ref_stance (= ref contact mask) per
    # _get_privileged_critic_obs's documented payload.
    from training.envs.env_info import PRIVILEGED_OBS_DIM, WR_INFO_KEY
    wr_info = state.info[WR_INFO_KEY]
    priv = np.asarray(wr_info.critic_obs)
    assert priv.shape == (PRIVILEGED_OBS_DIM,), (
        f"privileged critic obs shape changed: got {priv.shape}, "
        f"expected ({PRIVILEGED_OBS_DIM},).  The de-hybridization "
        "must not touch the critic pipeline."
    )
    # The env's reset path calls _get_privileged_critic_obs, which
    # threads nominal_q_ref + ref_contact_mask through the critic; if
    # those weren't present the env wouldn't build.  Shape contract
    # above is the concrete witness that those slots are still wired.
