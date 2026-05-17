"""Pin smoke12: per-joint residual scales derived from the WR walking prior.

smoke11 (the de-hybridized actor branch) converged to a standing local
minimum at vx ≈ -0.01 m/s, step_length ≈ 0.5 mm.  Root cause measured
against the WR offline prior at v0.1333: the flat ±0.25 rad/joint
residual cap was 1.8-3.6× too tight on the leg-pitch chain.  smoke12
widens the residual via per-joint scales derived by
``assets/derive_residual_scales.py --coverage 0.75``.

These tests pin both halves of the contract:

  1. The smoke12 yaml loads, basin stays ``home`` / ``home``, actor obs
     layout stays ``wr_obs_v7_phase_proprio``, everything else inherits
     byte-equal from smoke11.
  2. The per-joint residual scales in the yaml exactly match what
     ``derive_residual_scales.py`` produces today at
     ``--coverage 0.75 --floor 0.25`` against the current prior + home.
     If the prior is regenerated and someone forgets to re-run the
     script, this test fails loudly.
  3. The widened scales actually move every leg-pitch joint above the
     0.25 floor (the whole point of smoke12) AND stay below 1.0 rad
     so the residual remains a bounded correction, not unbounded
     direct control.
  4. Env builds, zero-action under home basin still hits ``_home_q_rad``
     exactly at every offline-cycle step.
"""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
import pytest


_REPO_ROOT = Path(__file__).resolve().parents[1]
_SMOKE11_CFG = _REPO_ROOT / "training" / "configs" / "ppo_walking_v0201_smoke11.yaml"
_SMOKE12_CFG = _REPO_ROOT / "training" / "configs" / "ppo_walking_v0201_smoke12.yaml"

_LEG_PITCH_JOINTS = (
    "left_hip_pitch", "left_knee_pitch", "left_ankle_pitch",
    "right_hip_pitch", "right_knee_pitch", "right_ankle_pitch",
)


def _load_cfg(path: Path):
    if not path.exists():
        pytest.skip(f"{path.name} not found")
    from training.configs.training_config import load_training_config
    return load_training_config(str(path))


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
# 1. yaml threading + smoke11 inheritance
# -----------------------------------------------------------------------------


def test_smoke12_loads_and_inherits_smoke11_basin_and_obs() -> None:
    cfg = _load_cfg(_SMOKE12_CFG)
    env = cfg.env
    assert env.actor_obs_layout_id == "wr_obs_v7_phase_proprio"
    assert env.loc_ref_residual_base == "home"
    assert env.loc_ref_reset_base == "home"
    assert env.loc_ref_penalty_pose_anchor == "home"


def test_smoke12_inherits_smoke11_curriculum_reward_ppo() -> None:
    """smoke12's ONLY material delta vs smoke11 is the per-joint
    residual scale block.  Everything else inherits byte-equal so a
    smoke12 vs smoke11 outcome diff cleanly attributes to the residual
    widening."""
    s11 = _load_cfg(_SMOKE11_CFG)
    s12 = _load_cfg(_SMOKE12_CFG)

    # Curriculum.
    assert s12.env.min_velocity == s11.env.min_velocity
    assert s12.env.max_velocity == s11.env.max_velocity
    assert s12.env.cmd_deadzone == s11.env.cmd_deadzone
    assert s12.env.eval_velocity_cmd == s11.env.eval_velocity_cmd
    assert s12.env.loc_ref_offline_command_vx == s11.env.loc_ref_offline_command_vx

    # Reward block (full TB-active set; imitation kept disabled).
    for field in (
        "alive", "cmd_forward_velocity_track", "cmd_forward_velocity_alpha",
        "ref_body_quat_track", "ang_vel_xy", "action_rate", "penalty_pose",
        "penalty_close_feet_xy", "feet_phase", "feet_phase_alpha",
        "feet_phase_swing_height", "penalty_feet_ori",
        "ref_q_track", "ref_contact_match", "ref_feet_z_track",
    ):
        assert getattr(s12.reward_weights, field) == getattr(s11.reward_weights, field)

    # Form flags + spatial thresholds.
    assert s12.env.loc_ref_penalty_ang_vel_xy_form == s11.env.loc_ref_penalty_ang_vel_xy_form
    assert s12.env.loc_ref_penalty_feet_ori_form == s11.env.loc_ref_penalty_feet_ori_form
    assert s12.env.close_feet_threshold == s11.env.close_feet_threshold

    # Reset perturbation carries over.
    assert list(s12.env.reset_torso_roll_range) == list(s11.env.reset_torso_roll_range)
    assert list(s12.env.reset_torso_pitch_range) == list(s11.env.reset_torso_pitch_range)

    # PPO + compute budget.
    assert s12.ppo.num_envs == s11.ppo.num_envs
    assert s12.ppo.rollout_steps == s11.ppo.rollout_steps
    assert s12.ppo.iterations == s11.ppo.iterations
    assert s12.ppo.learning_rate == s11.ppo.learning_rate
    assert s12.ppo.entropy_coef == s11.ppo.entropy_coef
    assert s12.ppo.gamma == s11.ppo.gamma


# -----------------------------------------------------------------------------
# 2. residual scales match the derivation script's output for coverage=0.75
# -----------------------------------------------------------------------------


def test_smoke12_residual_scales_match_derivation_script() -> None:
    """Re-running ``derive_residual_scales.py`` against the current
    prior + home keyframe must reproduce the per-joint scales in the
    smoke12 yaml exactly (to 6 decimals).  Catches "someone regenerated
    the prior but forgot to re-derive the scales" drift.
    """
    import sys
    sys.path.insert(0, str(_REPO_ROOT))
    from assets.derive_residual_scales import (
        DEFAULT_DERIVATION_VX,
        _derive_per_joint_scales,
        _load_home_pose,
        _load_prior_q_ref,
    )

    q_ref, joint_order = _load_prior_q_ref(DEFAULT_DERIVATION_VX)
    home = _load_home_pose(joint_order, "home")
    derived, _ = _derive_per_joint_scales(
        q_ref, home, joint_order, coverage=0.75, floor=0.25,
    )

    s12 = _load_cfg(_SMOKE12_CFG)
    on_disk = dict(s12.env.loc_ref_residual_scale_per_joint)

    for joint, expected in derived.items():
        actual = float(on_disk[joint])
        # 4 decimals — matches the yaml emitter's round(..., 4).
        assert abs(actual - round(expected, 4)) < 5e-5, (
            f"smoke12 residual_scale[{joint}] = {actual:.6f}, but "
            f"derive_residual_scales.py at coverage=0.75 produces "
            f"{expected:.6f}.  Re-run "
            "`uv run python assets/derive_residual_scales.py --coverage 0.75 "
            "--write training/configs/ppo_walking_v0201_smoke12.yaml`."
        )


# -----------------------------------------------------------------------------
# 3. widened scales are above smoke11 floor and below an unbounded cap
# -----------------------------------------------------------------------------


def test_smoke12_widens_leg_pitch_above_smoke11_floor() -> None:
    """smoke12 must actually widen each leg-pitch joint above the
    smoke11 0.25 cap (the whole point of the branch).  Ankle is the
    one already-covered joint and is allowed to stay at the 0.25
    floor."""
    s12 = _load_cfg(_SMOKE12_CFG)
    per_joint = dict(s12.env.loc_ref_residual_scale_per_joint)
    for joint in ("left_hip_pitch", "right_hip_pitch",
                  "left_knee_pitch", "right_knee_pitch"):
        v = float(per_joint[joint])
        assert v > 0.25, (
            f"smoke12 residual_scale[{joint}] = {v:.4f}; smoke11 used 0.25 "
            "and smoke12 must widen above that — derivation script may be "
            "running with the wrong coverage."
        )
    # Ankle is allowed to stay at floor (prior amplitude already covered).
    assert per_joint["left_ankle_pitch"] >= 0.25
    assert per_joint["right_ankle_pitch"] >= 0.25


def test_smoke12_residual_stays_below_unbounded_cap() -> None:
    """Sanity: residual remains a bounded correction, not unbounded
    direct control.  1.0 rad is the upper sanity bound — anything
    above would saturate the WR knee range (0..2.094 rad) on one
    side of home and lose all anti-exploit pressure.  Today the
    knees clock in at ~0.67 with coverage=0.75; this gives ~33%
    headroom before the bound triggers."""
    s12 = _load_cfg(_SMOKE12_CFG)
    per_joint = dict(s12.env.loc_ref_residual_scale_per_joint)
    for joint, scale in per_joint.items():
        assert float(scale) < 1.0, (
            f"smoke12 residual_scale[{joint}] = {scale:.4f} >= 1.0 rad — "
            "the residual is no longer a bounded correction.  If this is "
            "intentional, raise the sanity bound here AND update the "
            "G5 anti-exploit gate in walking_training.md."
        )


# -----------------------------------------------------------------------------
# 4. env build + zero-action contract preserved
# -----------------------------------------------------------------------------


def test_smoke12_env_builds_and_zero_action_targets_home() -> None:
    env = _maybe_build_env(_SMOKE12_CFG)
    assert env._residual_base_mode == "home"
    assert env._reset_base_mode == "home"
    # The per-joint scales loaded into the env match the yaml block —
    # if they don't, an unrelated runtime config path is shadowing them.
    names = list(env._policy_spec.robot.actuator_names)
    runtime = np.asarray(env._residual_q_scale_per_joint, dtype=np.float64)
    on_disk = dict(_load_cfg(_SMOKE12_CFG).env.loc_ref_residual_scale_per_joint)
    for joint in _LEG_PITCH_JOINTS:
        assert abs(runtime[names.index(joint)] - float(on_disk[joint])) < 1e-6

    # Zero-action under home basin still hits _home_q_rad exactly.
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
                f"smoke12 step {step_idx}: zero-action target_q != "
                "_home_q_rad — widening the residual scale broke the home "
                "basin invariant."
            ),
        )


def test_smoke12_jit_reset_still_traces() -> None:
    """Regression carryover from dc04595: the reset body must trace
    cleanly under jax.jit even with the widened residual scales (the
    fix is independent of residual_scale, but pin it again here so a
    future scale-related refactor can't silently re-introduce the
    float(tracer) issue)."""
    env = _maybe_build_env(_SMOKE12_CFG)
    assert env._reset_perturbation_enabled() is True
    reset_jit = jax.jit(env.reset)
    state = reset_jit(jax.random.PRNGKey(0))
    obs_array = state.obs["state"] if isinstance(state.obs, dict) else state.obs
    assert np.asarray(obs_array).shape == (env.observation_size,)
