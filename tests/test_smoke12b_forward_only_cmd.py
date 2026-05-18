"""Pin smoke12b: forward-only cmd range bug fix on smoke12.

smoke12 was intended as the basin-break bootstrap but inherited the
symmetric ``[-0.1333, +0.1333]`` cmd range from smoke9c/10/11.  With
``cmd_zero_chance: 0.0`` every episode demanded nonzero motion, and
~50% of episodes asked for backward motion.  Combined with smoke12's
widened residual + halved action_rate + baseline-subtract feet_phase,
PPO collapsed into a backward-stepping gait:
    forward_velocity ≈ -0.35 m/s, step_length_event ≈ -1 mm,
    ep_len ≈ 54/500, term_height_low_frac = 100%.
Iter-0 (random policy) already showed forward_velocity ≈ -0.14 m/s,
confirming the bias was in the curriculum rather than env physics.

smoke12b fixes the curriculum to forward-only
(``min_velocity: 0.08, max_velocity: 0.1333333333``) and keeps every
other smoke12 knob byte-equal.  These tests pin both halves:

  1. The smoke12b yaml has a strictly forward cmd range with the
     other smoke12 basin-break knobs intact.
  2. Every other field (basin / obs / reward / residual / PPO /
     reset perturbation) inherits byte-equal from smoke12 — the
     bug fix is surgical.
  3. Env builds, zero-action under home basin still hits
     ``_home_q_rad``.
"""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
import pytest


_REPO_ROOT = Path(__file__).resolve().parents[1]
_SMOKE12_CFG = _REPO_ROOT / "training" / "configs" / "ppo_walking_v0201_smoke12.yaml"
_SMOKE12B_CFG = _REPO_ROOT / "training" / "configs" / "ppo_walking_v0201_smoke12b.yaml"


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
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"env deps unavailable: {exc}")
    load_robot_config(str(_REPO_ROOT / "assets/v2/mujoco_robot_config.json"))
    cfg = load_training_config(str(cfg_path))
    cfg.freeze()
    return WildRobotEnv(cfg)


# -----------------------------------------------------------------------------
# 1. forward-only cmd range (the bug-fix)
# -----------------------------------------------------------------------------


def test_smoke12b_cmd_range_is_strictly_forward() -> None:
    """The whole point of smoke12b: cmd range must be > 0 on both
    bounds (no backward cmd ever sampled).  Direct pin so a future
    edit that re-introduces a negative ``min_velocity`` trips here
    immediately.
    """
    cfg = _load_cfg(_SMOKE12B_CFG)
    env = cfg.env
    assert env.min_velocity > 0.0, (
        f"smoke12b min_velocity = {env.min_velocity}; must be > 0 to "
        "guarantee forward-only cmd sampling."
    )
    assert env.max_velocity > 0.0, (
        f"smoke12b max_velocity = {env.max_velocity}; must be > 0."
    )
    assert env.max_velocity > env.min_velocity, (
        f"smoke12b max_velocity ({env.max_velocity}) must exceed "
        f"min_velocity ({env.min_velocity}) so the sampler has a "
        "valid open interval."
    )
    # Also pin cmd_zero_chance=0.0 (carried from smoke12) so there's
    # no "stand still" escape on top of the forward-only range.
    assert env.cmd_zero_chance == 0.0, (
        f"smoke12b cmd_zero_chance = {env.cmd_zero_chance}; must be "
        "0.0 (basin-break is paired with forward-only cmd range)."
    )


def test_smoke12b_eval_cmd_stays_in_training_range() -> None:
    """G4/G5 require the eval cmd to be inside the training range so
    a passing eval reflects on-distribution performance.  Pin this
    explicitly so a future ``max_velocity`` reduction can't silently
    push the eval cmd out of training distribution."""
    cfg = _load_cfg(_SMOKE12B_CFG)
    env = cfg.env
    assert env.min_velocity <= env.eval_velocity_cmd <= env.max_velocity, (
        f"smoke12b eval_velocity_cmd ({env.eval_velocity_cmd}) is "
        f"outside the training range [{env.min_velocity}, "
        f"{env.max_velocity}]."
    )


def test_smoke12b_deadzone_is_inert_under_forward_only_range() -> None:
    """smoke12b retains ``cmd_deadzone: 0.0666666667`` from smoke12
    for back-compat with the sampler path, but with min_velocity =
    0.08 > 0.0667 the deadzone snap branch is unreachable.  This
    test pins that property so a future deadzone tightening above
    0.08 silently re-activates the snap branch.
    """
    cfg = _load_cfg(_SMOKE12B_CFG)
    env = cfg.env
    assert env.cmd_deadzone < env.min_velocity, (
        f"smoke12b cmd_deadzone ({env.cmd_deadzone}) must stay below "
        f"min_velocity ({env.min_velocity}) so the sampler's "
        "deadzone snap is inert.  If the deadzone is raised above "
        "min_velocity, ALL forward cmds would snap to 0 and break "
        "the basin-break invariant."
    )


# -----------------------------------------------------------------------------
# 2. byte-equal inheritance from smoke12 (surgical fix)
# -----------------------------------------------------------------------------


def test_smoke12b_inherits_smoke12_basin_break_byte_equal() -> None:
    """Surgical-fix discipline: smoke12b changes ONLY ``min_velocity``
    and ``max_velocity``.  Every other smoke12 basin-break knob must
    inherit byte-equal so the smoke12b vs smoke12 outcome diff
    attributes cleanly to the cmd-range fix.

    The four smoke12 deltas (vs smoke11) are all carried over:
      1. widened residual_scale_per_joint
      2. cmd_zero_chance = 0.0
      3. action_rate = -1.0
      4. _feet_phase_reward + loc_ref_feet_phase_zero_on_standing
    """
    s12 = _load_cfg(_SMOKE12_CFG)
    s12b = _load_cfg(_SMOKE12B_CFG)

    # The ONE intended delta.
    assert s12b.env.min_velocity != s12.env.min_velocity, (
        "smoke12b min_velocity should DIFFER from smoke12 (forward-only "
        f"bug fix).  smoke12={s12.env.min_velocity}, "
        f"smoke12b={s12b.env.min_velocity}."
    )
    # smoke12b max_velocity is intentionally the same as smoke12's
    # (= eval cmd = 0.1333); only the LOWER bound moved.
    assert s12b.env.max_velocity == s12.env.max_velocity, (
        f"smoke12b max_velocity ({s12b.env.max_velocity}) drifted from "
        f"smoke12 ({s12.env.max_velocity}); the surgical fix only "
        "moves min_velocity."
    )

    # Everything else inherits byte-equal.
    # Curriculum (except the two range bounds).
    assert s12b.env.cmd_deadzone == s12.env.cmd_deadzone
    assert s12b.env.cmd_resample_steps == s12.env.cmd_resample_steps
    assert s12b.env.cmd_zero_chance == s12.env.cmd_zero_chance
    assert s12b.env.cmd_turn_chance == s12.env.cmd_turn_chance
    assert s12b.env.eval_velocity_cmd == s12.env.eval_velocity_cmd
    assert s12b.env.loc_ref_offline_command_vx == s12.env.loc_ref_offline_command_vx

    # Basin / obs.
    assert s12b.env.actor_obs_layout_id == s12.env.actor_obs_layout_id
    assert s12b.env.loc_ref_residual_base == s12.env.loc_ref_residual_base
    assert s12b.env.loc_ref_reset_base == s12.env.loc_ref_reset_base
    assert s12b.env.loc_ref_penalty_pose_anchor == s12.env.loc_ref_penalty_pose_anchor

    # Reward block.
    for field in (
        "alive", "cmd_forward_velocity_track", "cmd_forward_velocity_alpha",
        "ref_body_quat_track", "ang_vel_xy", "action_rate", "penalty_pose",
        "penalty_close_feet_xy", "feet_phase", "feet_phase_alpha",
        "feet_phase_swing_height", "penalty_feet_ori",
        "ref_q_track", "ref_contact_match", "ref_feet_z_track",
    ):
        assert getattr(s12b.reward_weights, field) == getattr(s12.reward_weights, field), (
            f"smoke12b reward_weights.{field} drifted from smoke12; "
            "only min/max_velocity should differ between the two."
        )

    # Form flags + spatial thresholds.
    assert s12b.env.loc_ref_penalty_ang_vel_xy_form == s12.env.loc_ref_penalty_ang_vel_xy_form
    assert s12b.env.loc_ref_penalty_feet_ori_form == s12.env.loc_ref_penalty_feet_ori_form
    assert s12b.env.loc_ref_feet_phase_zero_on_standing == s12.env.loc_ref_feet_phase_zero_on_standing
    assert s12b.env.close_feet_threshold == s12.env.close_feet_threshold

    # Residual scale block (smoke12's widened values).
    assert s12b.env.loc_ref_residual_mode == s12.env.loc_ref_residual_mode
    assert s12b.env.loc_ref_residual_scale == s12.env.loc_ref_residual_scale
    assert (
        dict(s12b.env.loc_ref_residual_scale_per_joint)
        == dict(s12.env.loc_ref_residual_scale_per_joint)
    )

    # Reset perturbation.
    assert list(s12b.env.reset_torso_roll_range) == list(s12.env.reset_torso_roll_range)
    assert list(s12b.env.reset_torso_pitch_range) == list(s12.env.reset_torso_pitch_range)

    # PPO + compute budget.
    assert s12b.ppo.num_envs == s12.ppo.num_envs
    assert s12b.ppo.rollout_steps == s12.ppo.rollout_steps
    assert s12b.ppo.iterations == s12.ppo.iterations
    assert s12b.ppo.learning_rate == s12.ppo.learning_rate
    assert s12b.ppo.entropy_coef == s12.ppo.entropy_coef
    assert s12b.ppo.gamma == s12.ppo.gamma


# -----------------------------------------------------------------------------
# 3. env build + zero-action contract preserved
# -----------------------------------------------------------------------------


def test_smoke12b_env_builds_and_zero_action_targets_home() -> None:
    env = _maybe_build_env(_SMOKE12B_CFG)
    assert env._residual_base_mode == "home"
    assert env._reset_base_mode == "home"
    # smoke12-inherited basin-break flag still set.
    assert env._feet_phase_zero_on_standing is True

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
                f"smoke12b step {step_idx}: zero-action target_q != "
                "_home_q_rad — the home basin invariant regressed under "
                "the cmd-range fix."
            ),
        )


def test_smoke12b_sampler_never_produces_negative_cmd() -> None:
    """End-to-end probe of the velocity sampler: 1024 draws from the
    smoke12b sampler must all be strictly positive (≥ min_velocity).
    Catches a future sampler change that could re-introduce backward
    cmds even with a positive min_velocity (e.g. accidentally adding
    a negation, or the deadzone snap branch firing incorrectly)."""
    env = _maybe_build_env(_SMOKE12B_CFG)
    sample_fn = jax.jit(env._sample_velocity_cmd)
    rng = jax.random.PRNGKey(0)
    samples = []
    for i in range(1024):
        rng, sub = jax.random.split(rng)
        samples.append(float(sample_fn(sub)))
    samples = np.asarray(samples)
    assert samples.min() >= 0.08 - 1e-6, (
        f"smoke12b velocity sampler returned min={samples.min():+.4f}; "
        "expected >= 0.08 (min_velocity)."
    )
    assert samples.max() <= 0.1333333333 + 1e-6, (
        f"smoke12b velocity sampler returned max={samples.max():+.4f}; "
        "expected <= 0.1333333333 (max_velocity)."
    )
    assert (samples > 0).all(), (
        "smoke12b velocity sampler produced a non-positive cmd — "
        "backward-cmd regression."
    )
