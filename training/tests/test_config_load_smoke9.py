"""YAML contract assertions for v0.20.1 smoke9 (full TB walk.gin recipe).

Smoke9 implements the 4 TB rewards that smoke8b deferred
(penalty_close_feet_xy, feet_phase, true penalty_ang_vel_xy via form
flag, true penalty_feet_ori via form flag) and enables them at TB
walk.gin magnitudes.  These tests pin every TB-mapping claim so a
future yaml edit can't silently regress one.

Spec: training/CHANGELOG.md [v0.20.1-smoke9-tb-faithful-rewards].
"""

from __future__ import annotations

from pathlib import Path

import pytest

from training.configs.training_config import load_training_config

_SMOKE9 = Path("training/configs/ppo_walking_v0201_smoke9.yaml")


@pytest.fixture(scope="module")
def smoke9_cfg():
    if not _SMOKE9.exists():
        pytest.skip(f"{_SMOKE9.name} not found")
    return load_training_config(str(_SMOKE9))


def test_smoke9_loads(smoke9_cfg) -> None:
    assert smoke9_cfg is not None


def test_smoke9_inherits_smoke8b_architecture(smoke9_cfg) -> None:
    """Smoke9 keeps smoke8b's architectural choices: home-base residual,
    home-anchored penalty_pose, TB-aligned PPO hyperparams.  Only the
    reward family expands."""
    env = smoke9_cfg.env
    assert env.loc_ref_residual_base == "home"
    assert env.loc_ref_penalty_pose_anchor == "home"
    ppo = smoke9_cfg.ppo
    assert ppo.learning_rate == 3.0e-5
    assert ppo.entropy_coef == 5.0e-4
    assert ppo.num_envs == 4096
    assert ppo.rollout_steps == 20
    assert ppo.gamma == 0.97
    assert ppo.max_grad_norm == 1.0


def test_smoke9_form_flags_are_tb(smoke9_cfg) -> None:
    """Smoke9 specifically sets the new env form flags to TB:
    - loc_ref_penalty_ang_vel_xy_form: tb_neg_squared
        (env emits -sum(rate²); positive yaml weight matches TB sign)
    - loc_ref_penalty_feet_ori_form: tb_linear_lateral
        (env emits -(sqrt(gx²+gy²)_L + sqrt(gx²+gy²)_R); positive yaml
        weight matches TB sign)
    - close_feet_threshold: 0.06 (TB walk.gin:125)
    """
    env = smoke9_cfg.env
    assert env.loc_ref_penalty_ang_vel_xy_form == "tb_neg_squared", (
        f"smoke9 must set form=tb_neg_squared; got "
        f"{env.loc_ref_penalty_ang_vel_xy_form!r}.  Without this the "
        f"reward is the bounded Gaussian, not TB's unbounded negative."
    )
    assert env.loc_ref_penalty_feet_ori_form == "tb_linear_lateral", (
        f"smoke9 must set form=tb_linear_lateral; got "
        f"{env.loc_ref_penalty_feet_ori_form!r}.  Without this the "
        f"reward is WR's quadratic 3-axis, not TB's linear lateral L2."
    )
    assert env.close_feet_threshold == 0.06, (
        f"smoke9 close_feet_threshold = {env.close_feet_threshold}; "
        f"expected 0.06 (TB walk.gin:125)."
    )


def test_smoke9_full_tb_active_rewards_present(smoke9_cfg) -> None:
    """All 8 TB walk.gin:110-131 active rewards (the 9th, ang_vel_z,
    is consistently deferred since WR doesn't sample yaw cmds) must
    be ENABLED at TB-equivalent magnitudes.  Smoke8b had 5; smoke9
    adds 3 (penalty_close_feet_xy, feet_phase, ang_vel_xy form-corrected,
    penalty_feet_ori form-corrected).
    """
    rw = smoke9_cfg.reward_weights
    expected = {
        # Round-1 (smoke8b) faithful terms — kept as-is.
        "alive": 1.0,                              # TB alive = 1.0
        "cmd_forward_velocity_track": 2.0,         # TB lin_vel_xy = 2.0
        "cmd_forward_velocity_alpha": 1000.0,      # TB lin_vel_tracking_sigma = 1000
        "ref_body_quat_track": 2.5,                # TB torso_quat = 2.5
        "action_rate": -2.0,                       # TB penalty_action_rate = 2.0
        "penalty_pose": -0.5,                      # TB penalty_pose = 0.5
        # Smoke9 additions — TB-faithful via env form flags / new rewards.
        "ang_vel_xy": 1.0,                         # TB penalty_ang_vel_xy = 1.0
        "penalty_close_feet_xy": 10.0,             # TB penalty_close_feet_xy = 10.0
        "feet_phase": 7.5,                         # TB feet_phase = 7.5
        "feet_phase_alpha": 1428.6,                # 1 / TB feet_phase_tracking_sigma=0.0007
        "feet_phase_swing_height": 0.04,           # TB swing_height = 0.04
        "penalty_feet_ori": 5.0,                   # TB penalty_feet_ori = 5.0
                                                   # (POSITIVE under
                                                   # tb_linear_lateral form)
    }
    failures = []
    for key, expected_value in expected.items():
        actual = getattr(rw, key)
        # Use tolerance for floating-point fields (alpha, sigma);
        # exact match for integer-like weights.
        if isinstance(expected_value, float) and abs(expected_value) > 1.0:
            ok = abs(actual - expected_value) <= 0.1 * abs(expected_value)
        else:
            ok = actual == expected_value
        if not ok:
            failures.append(f"  {key}: expected {expected_value}, got {actual}")
    if failures:
        pytest.fail(
            "smoke9 TB-active rewards must match TB walk.gin:110-131:\n"
            + "\n".join(failures)
        )


def test_smoke9_imitation_and_extras_disabled(smoke9_cfg) -> None:
    """Same disabled-list as smoke8b.  TB walk.gin doesn't use any of:
    ref_q_track, ref_contact_match, ref_feet_z_track, torso_pos_xy,
    lin_vel_z, feet_air_time, feet_clearance, feet_distance,
    torso_pitch_soft, torso_roll_soft, torque, joint_velocity, slip,
    pitch_rate.
    """
    rw = smoke9_cfg.reward_weights
    must_be_zero = [
        "ref_q_track", "ref_contact_match", "ref_feet_z_track",
        "torso_pos_xy", "lin_vel_z", "feet_air_time", "feet_clearance",
        "feet_distance", "torso_pitch_soft", "torso_roll_soft",
        "torque", "joint_velocity", "slip", "pitch_rate",
    ]
    nonzero = [(k, getattr(rw, k)) for k in must_be_zero if getattr(rw, k) != 0.0]
    if nonzero:
        pytest.fail(
            "smoke9: the following reward weights must be 0.0 (TB "
            "walk.gin doesn't use them):\n  "
            + "\n  ".join(f"{k} = {v}" for k, v in nonzero)
        )


def test_smoke9_compute_budget_preserved(smoke9_cfg) -> None:
    """Same compute as smoke7/8/8a/8b: 4096 × 20 × 240 = 19,660,800."""
    ppo = smoke9_cfg.ppo
    total = ppo.num_envs * ppo.rollout_steps * ppo.iterations
    assert total == 19_660_800, (
        f"smoke9 compute budget = {total:,}; expected 19,660,800."
    )


# -----------------------------------------------------------------------------
# Back-compat: smoke7/8/8a default to non-TB form flags
# -----------------------------------------------------------------------------


def test_smoke7_default_form_flags_unchanged() -> None:
    """smoke7/8/8a yamls don't set the smoke9 form flags.  Their loaded
    config must default to the historical (non-TB) forms so the
    architectural change is opt-in only, not silent."""
    smoke7 = Path("training/configs/ppo_walking_v0201_smoke.yaml")
    if not smoke7.exists():
        pytest.skip("smoke7 yaml missing")
    cfg = load_training_config(str(smoke7))
    env = cfg.env
    assert env.loc_ref_penalty_ang_vel_xy_form == "gaussian", (
        "smoke7 must keep the historical Gaussian ang_vel_xy form; "
        f"got {env.loc_ref_penalty_ang_vel_xy_form!r}.  If this fails, "
        "the dataclass default flipped — check loc_ref_penalty_ang_vel_xy_form."
    )
    assert env.loc_ref_penalty_feet_ori_form == "wr_quad_3axis", (
        "smoke7 must keep the historical quadratic feet_ori form; "
        f"got {env.loc_ref_penalty_feet_ori_form!r}."
    )
