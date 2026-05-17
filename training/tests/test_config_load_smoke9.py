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
    home-anchored penalty_pose, TB-compatible 12GB-safe PPO shape.
    Only the reward family expands."""
    env = smoke9_cfg.env
    assert env.loc_ref_residual_base == "home"
    assert env.loc_ref_penalty_pose_anchor == "home"
    ppo = smoke9_cfg.ppo
    assert ppo.learning_rate == 3.0e-5
    assert ppo.entropy_coef == 5.0e-4
    assert ppo.num_envs == 2048
    assert ppo.rollout_steps == 20
    assert ppo.iterations == 480
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
    """Same compute as smoke7/8/8a/8b: 2048 × 20 × 480 = 19,660,800."""
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


# =============================================================================
# Smoke9b — TB-aligned + TB-matched vx range
# =============================================================================
# Smoke9 (run `58bv3zzm`) failed because cmd_forward_velocity_track had
# zero gradient at WR's vx range (random policy too far from cmd basin
# for TB's tight alpha=1000).  Smoke9b narrows vx to TB's [-0.1, 0.1]
# so the random policy starts inside the basin.  These tests pin the
# specific vx changes to prevent silent regression.

_SMOKE9B = Path("training/configs/ppo_walking_v0201_smoke9b.yaml")


@pytest.fixture(scope="module")
def smoke9b_cfg():
    if not _SMOKE9B.exists():
        pytest.skip(f"{_SMOKE9B.name} not found")
    return load_training_config(str(_SMOKE9B))


def test_smoke9b_loads(smoke9b_cfg) -> None:
    assert smoke9b_cfg is not None


def test_smoke9b_vx_range_matches_tb(smoke9b_cfg) -> None:
    """TB walk.gin:48 sets command_range[5] = [-0.1, 0.1] for vx.
    Smoke9b matches this exactly so random-policy WR starts at err
    ≤ 0.10 m/s, where TB's tight alpha=1000 reward is non-zero
    (exp(-10) = 4.5e-5; small but PPO can build on it).

    Smoke7/8/8a/8b/9 used [0, 0.30] which put random policy at
    err ≈ 0.20 — exp(-40) = 0, no gradient, smoke9 failed.
    """
    env = smoke9b_cfg.env
    assert env.min_velocity == -0.1, (
        f"smoke9b min_velocity = {env.min_velocity}; expected -0.1 "
        f"(TB walk.gin:48 command_range[5][0])."
    )
    assert env.max_velocity == 0.1, (
        f"smoke9b max_velocity = {env.max_velocity}; expected 0.1 "
        f"(TB walk.gin:48 command_range[5][1])."
    )


def test_smoke9b_eval_velocity_matches_max_tb_band(smoke9b_cfg) -> None:
    """Eval pinned to 0.10 (top of the TB-aligned vx range).  G4
    promotion gate becomes |achieved - 0.10| ≤ 0.05 (= 0.5 × 0.10).
    """
    assert smoke9b_cfg.env.eval_velocity_cmd == 0.10, (
        f"smoke9b eval_velocity_cmd = {smoke9b_cfg.env.eval_velocity_cmd}; "
        f"expected 0.10."
    )


def test_smoke9b_offline_command_vx_aligned(smoke9b_cfg) -> None:
    """Offline reference trajectory built for vx=0.10.  Phase clock +
    feet_phase expected-z curves are vx-independent (cycle period set
    by ZMPWalkConfig.cycle_time_s, not by speed), but aligning the
    library bin makes obs / diagnostics match the operating point."""
    assert smoke9b_cfg.env.loc_ref_offline_command_vx == 0.10, (
        f"smoke9b loc_ref_offline_command_vx = "
        f"{smoke9b_cfg.env.loc_ref_offline_command_vx}; expected 0.10 "
        f"(matches eval_velocity_cmd)."
    )


def test_smoke9b_inherits_smoke9_architecture(smoke9b_cfg) -> None:
    """Everything smoke9 had stays identical — only the vx block changes.
    Catches accidental regression of any TB-faithful smoke9 setting.
    """
    env = smoke9b_cfg.env
    assert env.loc_ref_residual_base == "home"
    assert env.loc_ref_reset_base == "home"
    assert env.loc_ref_penalty_pose_anchor == "home"
    assert env.loc_ref_penalty_ang_vel_xy_form == "tb_neg_squared"
    assert env.loc_ref_penalty_feet_ori_form == "tb_linear_lateral"
    assert env.close_feet_threshold == 0.06
    ppo = smoke9b_cfg.ppo
    assert ppo.learning_rate == 3.0e-5
    assert ppo.entropy_coef == 5.0e-4
    assert ppo.num_envs == 2048
    assert ppo.gamma == 0.97


def test_smoke9b_full_tb_active_rewards_present(smoke9b_cfg) -> None:
    """Same TB walk.gin:110-131 active block as smoke9, at the same
    magnitudes.  Only the training distribution changes."""
    rw = smoke9b_cfg.reward_weights
    assert rw.alive == 1.0
    assert rw.cmd_forward_velocity_track == 2.0
    assert rw.cmd_forward_velocity_alpha == 1000.0
    assert rw.ref_body_quat_track == 2.5
    assert rw.ang_vel_xy == 1.0
    assert rw.action_rate == -2.0
    assert rw.penalty_pose == -0.5
    assert rw.penalty_close_feet_xy == 10.0
    assert rw.feet_phase == 7.5
    assert rw.penalty_feet_ori == 5.0


def test_smoke9b_compute_budget_preserved(smoke9b_cfg) -> None:
    """Same compute as smoke9: 2048 × 20 × 480 = 19,660,800."""
    ppo = smoke9b_cfg.ppo
    assert ppo.num_envs * ppo.rollout_steps * ppo.iterations == 19_660_800


# =============================================================================
# Smoke9c — TB-aligned rewards + WR size-normalized TB curriculum + ref-init
# =============================================================================

_SMOKE9C = Path("training/configs/ppo_walking_v0201_smoke9c.yaml")


@pytest.fixture(scope="module")
def smoke9c_cfg():
    if not _SMOKE9C.exists():
        pytest.skip(f"{_SMOKE9C.name} not found")
    return load_training_config(str(_SMOKE9C))


def test_smoke9c_loads(smoke9c_cfg) -> None:
    assert smoke9c_cfg is not None


def test_smoke9c_uses_wr_size_normalized_tb_curriculum(smoke9c_cfg) -> None:
    env = smoke9c_cfg.env
    assert env.min_velocity == -0.1333333333
    assert env.max_velocity == 0.1333333333
    assert env.cmd_deadzone == 0.0666666667
    assert env.eval_velocity_cmd == 0.1333333333
    assert env.loc_ref_offline_command_vx == 0.1333333333


def test_smoke9c_uses_ref_init_bases(smoke9c_cfg) -> None:
    env = smoke9c_cfg.env
    assert env.loc_ref_residual_base == "ref_init"
    assert env.loc_ref_reset_base == "ref_init"
    assert env.loc_ref_penalty_pose_anchor == "home"


def test_smoke9c_keeps_tb_active_reward_weights_with_normalized_alpha(smoke9c_cfg) -> None:
    rw = smoke9c_cfg.reward_weights
    assert rw.cmd_forward_velocity_alpha == 562.5
    assert rw.cmd_forward_velocity_track == 2.0
    assert rw.feet_phase == 7.5
    assert rw.penalty_feet_ori == 5.0
    assert rw.penalty_pose == -0.5
    assert rw.penalty_close_feet_xy == 10.0


def test_smoke9c_compute_budget_preserved(smoke9c_cfg) -> None:
    ppo = smoke9c_cfg.ppo
    assert ppo.num_envs == 2048
    assert ppo.rollout_steps == 20
    assert ppo.iterations == 480
    assert ppo.num_envs * ppo.rollout_steps * ppo.iterations == 19_660_800


def test_smoke9c_spatial_normalization_thresholds(smoke9c_cfg) -> None:
    """smoke9c follow-up: lateral-foot geometry thresholds copied from
    TB literal meters are body-bound; the spatial-normalization pass
    rescales them by the TB→WR stance-width fraction (TB 0.074 m → WR
    0.18056 m).  Pin the derived values so a future yaml edit can't
    silently regress one back to TB meters.
    """
    env = smoke9c_cfg.env
    # close_feet_threshold: TB 0.06 / 0.074 * 0.18056 ≈ 0.146
    assert env.close_feet_threshold == pytest.approx(0.146, abs=5e-4), (
        f"smoke9c close_feet_threshold = {env.close_feet_threshold}; "
        f"expected 0.146 (WR-normalized from TB 0.06 m)."
    )
    # min_feet_y_dist: TB 0.07 / 0.074 * 0.18056 ≈ 0.171
    assert env.min_feet_y_dist == pytest.approx(0.171, abs=5e-4), (
        f"smoke9c min_feet_y_dist = {env.min_feet_y_dist}; "
        f"expected 0.171 (WR-normalized from TB 0.07 m).  "
        f"feet_distance is inactive but defaults are normalized to "
        f"prevent silent reactivation with TB meters."
    )
    # max_feet_y_dist: TB 0.13 / 0.074 * 0.18056 ≈ 0.317
    assert env.max_feet_y_dist == pytest.approx(0.317, abs=5e-4), (
        f"smoke9c max_feet_y_dist = {env.max_feet_y_dist}; "
        f"expected 0.317 (WR-normalized from TB 0.13 m)."
    )


def test_smoke9c_feet_phase_normalized_basin_preserved(smoke9c_cfg) -> None:
    """smoke9c follow-up: feet_phase swing target = 0.05 m taken from
    the WR-validated ZMP prior (control/zmp/zmp_walk.py:134
    foot_step_height_m) — NOT a strict Hof-1996 body-size rescale of
    TB's 0.04 m (which would give ~0.07-0.08 m on WR; see Appendix C
    in walking_training.md for the explicit deviation rationale).
    ``alpha`` is rescaled by (0.04/0.05)² so the dimensionless basin
    ``alpha * swing_height² ≈ 2.286`` is preserved, which keeps the
    Gaussian reward SHAPE invariant under the target change (flat-foot
    baseline + peak-tracking reward stay calibrated).
    """
    rw = smoke9c_cfg.reward_weights
    assert rw.feet_phase_swing_height == pytest.approx(0.05, abs=1e-6), (
        f"smoke9c feet_phase_swing_height = {rw.feet_phase_swing_height}; "
        f"expected 0.05 m (WR ZMP prior foot_step_height_m)."
    )
    assert rw.feet_phase_alpha == pytest.approx(914.304, abs=1e-2), (
        f"smoke9c feet_phase_alpha = {rw.feet_phase_alpha}; "
        f"expected 914.304 = 1428.6 * (0.04/0.05)² (preserves "
        f"alpha * swing_height² = 2.286)."
    )
    # Dimensionless basin invariant.
    tb_basin = 1428.6 * 0.04 * 0.04
    wr_basin = rw.feet_phase_alpha * rw.feet_phase_swing_height ** 2
    assert wr_basin == pytest.approx(tb_basin, rel=5e-3), (
        f"feet_phase normalized basin shifted: tb={tb_basin:.4f}, "
        f"wr={wr_basin:.4f}.  Hof (1996) body-size normalization "
        f"requires alpha * swing_height² be invariant."
    )


def test_smoke9c_reset_perturbation_enabled(smoke9c_cfg) -> None:
    """smoke9c follow-up: TB-style reset-time torso roll / pitch
    perturbation is enabled.  Default ranges should mirror TB's bipedal
    DomainRandConfig (mjx_config.py:211-212 ``[-0.1, 0.1]`` rad each).
    """
    env = smoke9c_cfg.env
    assert list(env.reset_torso_roll_range) == [-0.1, 0.1], (
        f"smoke9c reset_torso_roll_range = "
        f"{list(env.reset_torso_roll_range)}; expected [-0.1, 0.1] "
        f"(matches TB DomainRandConfig.torso_roll_range)."
    )
    assert list(env.reset_torso_pitch_range) == [-0.1, 0.1], (
        f"smoke9c reset_torso_pitch_range = "
        f"{list(env.reset_torso_pitch_range)}; expected [-0.1, 0.1] "
        f"(matches TB DomainRandConfig.torso_pitch_range)."
    )


# =============================================================================
# Smoke10 — post-home-migration TB-style fixed-home anchor
# =============================================================================
# smoke10 is the immediate follow-up to the 2026-05-17 ``home`` keyframe
# migration.  Discipline (tracking note §1.2): change ONLY the nominal
# action anchor; every other env / reward / PPO field must remain
# byte-equal to smoke9c.  The tests below pin both halves: (1) the basin
# flags really did flip, (2) nothing else regressed.

_SMOKE10 = Path("training/configs/ppo_walking_v0201_smoke10.yaml")


@pytest.fixture(scope="module")
def smoke10_cfg():
    if not _SMOKE10.exists():
        pytest.skip(f"{_SMOKE10.name} not found")
    return load_training_config(str(_SMOKE10))


def test_smoke10_loads(smoke10_cfg) -> None:
    assert smoke10_cfg is not None


def test_smoke10_flips_basin_to_home(smoke10_cfg) -> None:
    """The one material delta vs smoke9c: residual + reset bases are now
    ``home`` (post-home-migration TB-style fixed-home anchor)."""
    env = smoke10_cfg.env
    assert env.loc_ref_residual_base == "home", (
        f"smoke10 loc_ref_residual_base = {env.loc_ref_residual_base}; "
        "expected 'home' (the whole point of smoke10)."
    )
    assert env.loc_ref_reset_base == "home", (
        f"smoke10 loc_ref_reset_base = {env.loc_ref_reset_base}; "
        "expected 'home' (post-home-migration reset anchor)."
    )
    # Penalty-pose anchor was already `home` in smoke9c — verify it
    # stays there (semantics unchanged across the basin flip).
    assert env.loc_ref_penalty_pose_anchor == "home"


def test_smoke10_command_curriculum_inherited_from_smoke9c(smoke10_cfg) -> None:
    """smoke10 must inherit smoke9c's WR size-normalized command block
    verbatim.  Discipline: do not change command distribution in the
    same run as the basin flip."""
    env = smoke10_cfg.env
    assert env.min_velocity == -0.1333333333
    assert env.max_velocity == 0.1333333333
    assert env.cmd_deadzone == 0.0666666667
    assert env.eval_velocity_cmd == 0.1333333333
    assert env.loc_ref_offline_command_vx == 0.1333333333


def test_smoke10_reward_block_inherited_from_smoke9c(smoke10_cfg) -> None:
    """smoke10 must keep smoke9c's TB-active reward weights + the
    WR-normalized alpha / feet_phase basin.  Catches accidental
    reward-formula coupling with the basin flip."""
    rw = smoke10_cfg.reward_weights
    assert rw.cmd_forward_velocity_track == 2.0
    assert rw.cmd_forward_velocity_alpha == 562.5
    assert rw.ref_body_quat_track == 2.5
    assert rw.ang_vel_xy == 1.0
    assert rw.action_rate == -2.0
    assert rw.penalty_pose == -0.5
    assert rw.penalty_close_feet_xy == 10.0
    assert rw.feet_phase == 7.5
    assert rw.feet_phase_alpha == pytest.approx(914.304, abs=1e-2)
    assert rw.feet_phase_swing_height == pytest.approx(0.05, abs=1e-6)
    assert rw.penalty_feet_ori == 5.0
    # Imitation terms stay disabled (TB doesn't use them).
    assert rw.ref_q_track == 0.0
    assert rw.ref_contact_match == 0.0
    assert rw.ref_feet_z_track == 0.0


def test_smoke10_form_flags_and_spatial_thresholds_inherited(smoke10_cfg) -> None:
    """TB-faithful reward formula selectors + WR-normalized spatial
    geometry (close_feet_threshold, min/max_feet_y_dist) must carry
    over byte-equal from smoke9c."""
    env = smoke10_cfg.env
    assert env.loc_ref_penalty_ang_vel_xy_form == "tb_neg_squared"
    assert env.loc_ref_penalty_feet_ori_form == "tb_linear_lateral"
    assert env.close_feet_threshold == pytest.approx(0.146, abs=5e-4)
    assert env.min_feet_y_dist == pytest.approx(0.171, abs=5e-4)
    assert env.max_feet_y_dist == pytest.approx(0.317, abs=5e-4)


def test_smoke10_residual_scale_inherited(smoke10_cfg) -> None:
    """Per-joint residual authority is unchanged — the action anchor
    moves but the per-joint scale window does not.  Catches any
    coupled "rescue knob" change at the residual layer."""
    env = smoke10_cfg.env
    assert env.loc_ref_residual_mode == "absolute"
    assert env.loc_ref_residual_scale == 0.20
    per_joint = dict(env.loc_ref_residual_scale_per_joint)
    for joint in (
        "left_hip_pitch", "right_hip_pitch",
        "left_hip_roll", "right_hip_roll",
        "left_knee_pitch", "right_knee_pitch",
        "left_ankle_pitch", "right_ankle_pitch",
        "left_ankle_roll", "right_ankle_roll",
    ):
        assert per_joint[joint] == 0.25, (
            f"smoke10 residual_scale_per_joint[{joint}] = "
            f"{per_joint[joint]}; expected 0.25 (smoke9c value)."
        )


def test_smoke10_reset_perturbation_preserved(smoke10_cfg) -> None:
    """Tracking note §2.3: keep reset perturbation, do not revert.
    smoke10 must carry smoke9c's TB-style torso roll/pitch perturbation
    intact — the basin flip is orthogonal to the perturbation signal."""
    env = smoke10_cfg.env
    assert list(env.reset_torso_roll_range) == [-0.1, 0.1]
    assert list(env.reset_torso_pitch_range) == [-0.1, 0.1]


def test_smoke10_ppo_and_compute_budget_inherited(smoke10_cfg) -> None:
    """PPO hyperparameters + compute budget must match smoke9c.
    Discipline: do not co-tune PPO with the basin flip."""
    ppo = smoke10_cfg.ppo
    assert ppo.num_envs == 2048
    assert ppo.rollout_steps == 20
    assert ppo.iterations == 480
    assert ppo.learning_rate == 3.0e-5
    assert ppo.entropy_coef == 5.0e-4
    assert ppo.gamma == 0.97
    assert ppo.num_envs * ppo.rollout_steps * ppo.iterations == 19_660_800
