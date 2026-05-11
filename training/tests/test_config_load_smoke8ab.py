"""YAML contract assertions for v0.20.1 smoke8a / smoke8b configs.

Smoke8a and smoke8b are TB-alignment variants of smoke8.  These tests
pin the specific differences each variant promised:

  - smoke8a (Track A): TB-aligned PPO hyperparameters (learning_rate,
    entropy, num_envs, etc.) corrected from smoke7/8's misattributed
    values.  Reward family unchanged from smoke8.
  - smoke8b (Track A + B): smoke8a + TB-pure reward family (imitation
    block zeroed out, only TB walk.gin's active rewards remain).

If a future edit accidentally regresses a smoke8a value back to the
smoke7 misattribution, OR turns on an imitation reward in smoke8b that
TB doesn't use, these tests fail loud at config-load time — before
any compute is wasted.

Spec: training/CHANGELOG.md [v0.20.1-smoke8ab-tb-alignment-audit].
"""

from __future__ import annotations

from pathlib import Path

import pytest

from training.configs.training_config import load_training_config

_SMOKE8A = Path("training/configs/ppo_walking_v0201_smoke8a.yaml")
_SMOKE8B = Path("training/configs/ppo_walking_v0201_smoke8b.yaml")


# -----------------------------------------------------------------------------
# Smoke8a — Track A: TB-aligned PPO hyperparams
# -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def smoke8a_cfg():
    if not _SMOKE8A.exists():
        pytest.skip(f"{_SMOKE8A.name} not found")
    return load_training_config(str(_SMOKE8A))


def test_smoke8a_loads(smoke8a_cfg) -> None:
    """Yaml parses cleanly into the training config."""
    assert smoke8a_cfg is not None
    assert smoke8a_cfg.env is not None
    assert smoke8a_cfg.ppo is not None


def test_smoke8a_residual_base_home(smoke8a_cfg) -> None:
    """Smoke8a inherits smoke8's home-base architecture."""
    assert smoke8a_cfg.env.loc_ref_residual_base == "home"


def test_smoke8a_ppo_hyperparams_match_tb(smoke8a_cfg) -> None:
    """Track A correction: PPO hyperparams must match TB's
    ``toddlerbot/locomotion/ppo_config.py`` defaults, NOT smoke7/8's
    misattributed g1/mujoco_playground baselines.
    """
    ppo = smoke8a_cfg.ppo
    expected = {
        "num_envs": 4096,
        "rollout_steps": 20,
        "iterations": 240,
        "learning_rate": 3.0e-5,
        "gamma": 0.97,
        "gae_lambda": 0.95,
        "entropy_coef": 5.0e-4,
        "clip_epsilon": 0.2,
        "epochs": 4,
        "num_minibatches": 16,
        "max_grad_norm": 1.0,
    }
    failures = []
    for key, expected_value in expected.items():
        actual = getattr(ppo, key)
        if actual != expected_value:
            failures.append(f"  ppo.{key}: expected {expected_value}, got {actual}")
    if failures:
        pytest.fail(
            "smoke8a PPO hyperparams must match TB defaults "
            "(toddlerbot/locomotion/ppo_config.py).  Smoke7/8 cited "
            "these as 'TB defaults' but used g1/mujoco_playground "
            "values; smoke8a corrects this.\n" + "\n".join(failures)
        )


def test_smoke8a_compute_budget_preserved(smoke8a_cfg) -> None:
    """Track A reshapes per-iter (more envs, shorter rollout, more
    iters) while preserving the total smoke compute budget.
    Smoke7/8 = 1024 × 128 × 150 = 19,660,800 transitions.
    Smoke8a  = 4096 × 20  × 240 = 19,660,800 transitions.
    """
    ppo = smoke8a_cfg.ppo
    total = ppo.num_envs * ppo.rollout_steps * ppo.iterations
    assert total == 19_660_800, (
        f"smoke8a compute budget = {total:,} transitions; expected "
        f"19,660,800 (= smoke7/8 budget).  If you changed any of "
        f"num_envs/rollout_steps/iterations, the others must be "
        f"adjusted to keep the product constant."
    )


def test_smoke8a_log_std_init_matches_tb(smoke8a_cfg) -> None:
    """TB ppo_config.py: init_noise_std = 0.5 → log(0.5) ≈ -0.6931.
    smoke7/8 used -1.0 → exp(-1.0) ≈ 0.37 (less initial exploration).
    """
    actor = smoke8a_cfg.networks.actor
    assert abs(actor.log_std_init - (-0.693)) < 0.001, (
        f"smoke8a actor.log_std_init = {actor.log_std_init}; expected "
        f"-0.693 (= log(0.5), matching TB init_noise_std = 0.5)."
    )


def test_smoke8a_reward_family_unchanged_from_smoke8(smoke8a_cfg) -> None:
    """Track A keeps the imitation reward block — that's a Track B
    decision.  Confirm key imitation weights are still active.
    """
    rw = smoke8a_cfg.reward_weights
    assert rw.ref_q_track == 5.0
    assert rw.ref_body_quat_track == 2.5
    assert rw.ref_feet_z_track == 5.0
    assert rw.ref_contact_match == 1.0
    assert rw.feet_air_time == 500.0


# -----------------------------------------------------------------------------
# Smoke8b — Track A + B: TB-pure reward family
# -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def smoke8b_cfg():
    if not _SMOKE8B.exists():
        pytest.skip(f"{_SMOKE8B.name} not found")
    return load_training_config(str(_SMOKE8B))


def test_smoke8b_loads(smoke8b_cfg) -> None:
    assert smoke8b_cfg is not None


def test_smoke8b_residual_base_home(smoke8b_cfg) -> None:
    assert smoke8b_cfg.env.loc_ref_residual_base == "home"


def test_smoke8b_ppo_hyperparams_match_tb(smoke8b_cfg) -> None:
    """Smoke8b inherits Track A's TB-aligned PPO hyperparams."""
    ppo = smoke8b_cfg.ppo
    assert ppo.learning_rate == 3.0e-5
    assert ppo.entropy_coef == 5.0e-4
    assert ppo.num_envs == 4096
    assert ppo.rollout_steps == 20
    assert ppo.num_minibatches == 16
    assert ppo.gamma == 0.97
    assert ppo.max_grad_norm == 1.0


def test_smoke8b_imitation_rewards_disabled(smoke8b_cfg) -> None:
    """Track B: imitation reward block must be DISABLED (weight 0).
    TB walk.gin's active recipe doesn't use ref_q_track,
    ref_contact_match, torso_pos_xy.
    """
    rw = smoke8b_cfg.reward_weights
    must_be_zero = {
        "ref_q_track": rw.ref_q_track,
        "ref_contact_match": rw.ref_contact_match,
        "torso_pos_xy": rw.torso_pos_xy,
        "feet_air_time": rw.feet_air_time,
        "feet_clearance": rw.feet_clearance,
        "lin_vel_z": rw.lin_vel_z,
        "torso_pitch_soft": rw.torso_pitch_soft,
        "torso_roll_soft": rw.torso_roll_soft,
        "torque": rw.torque,
        "joint_velocity": rw.joint_velocity,
    }
    nonzero = {k: v for k, v in must_be_zero.items() if v != 0.0}
    if nonzero:
        pytest.fail(
            "smoke8b: the following reward weights must be 0.0 (TB "
            "walk.gin doesn't use them):\n  "
            + "\n  ".join(f"{k} = {v}" for k, v in nonzero.items())
        )


def test_smoke8b_tb_active_rewards_present(smoke8b_cfg) -> None:
    """Track B: keep TB walk.gin's active reward block at TB magnitudes.
    Mapping documented in smoke8b.yaml header.

    Audit revision: feet_distance and ref_feet_z_track DISABLED here
    (set to 0).  WR's implementations are not faithful TB analogs
    (positive band reward vs binary close-feet penalty;
    ZMP-imitation vs phase-derived foot height).  Implementing true
    TB-equivalent terms is a smoke9 candidate.
    """
    rw = smoke8b_cfg.reward_weights
    expected = {
        # alive: TB RewardScales.alive = 1.0
        "alive": 1.0,
        # cmd_forward_velocity_track: TB lin_vel_xy = 2.0
        "cmd_forward_velocity_track": 2.0,
        # ref_body_quat_track: TB torso_quat = 2.5 (target IS identity quat,
        # geodesic angle, alpha=20 default = TB rot_tracking_sigma)
        "ref_body_quat_track": 2.5,
        # ang_vel_xy: TB penalty_ang_vel_xy = 1.0
        "ang_vel_xy": 1.0,
        # action_rate: TB penalty_action_rate = 2.0 (sign convention diff)
        "action_rate": -2.0,
        # penalty_pose: TB walk.gin = 0.5 (sign convention diff).
        # See test_smoke8b_penalty_pose_anchor_is_home for the critical
        # env flag that makes this term TB-aligned (anchor home, not q_ref).
        "penalty_pose": -0.5,
        # penalty_feet_ori: TB = 5.0 (sign convention diff)
        "penalty_feet_ori": -5.0,
    }
    failures = []
    for key, expected_value in expected.items():
        actual = getattr(rw, key)
        if actual != expected_value:
            failures.append(f"  {key}: expected {expected_value}, got {actual}")
    if failures:
        pytest.fail(
            "smoke8b TB-active rewards must match TB walk.gin:110-131 "
            "magnitudes:\n" + "\n".join(failures)
        )


def test_smoke8b_tracking_sigma_matches_tb(smoke8b_cfg) -> None:
    """TB lin_vel_xy reward (mjx_env.py:2346) is
    ``exp(-self.lin_vel_tracking_sigma * error**2)`` — multiplicative.
    WR's formula (wildrobot_env.py:1014) is
    ``exp(-cmd_forward_velocity_alpha * err²)`` — same form.
    So TB-equivalent alpha = TB sigma = 1000.

    smoke7/8 used alpha=200 (~5× looser than TB).  An earlier draft
    of smoke8b set alpha=0.001 based on inverted dimensional
    reasoning; this test pins the corrected value.
    """
    rw = smoke8b_cfg.reward_weights
    assert rw.cmd_forward_velocity_alpha == 1000.0, (
        f"smoke8b cmd_forward_velocity_alpha = "
        f"{rw.cmd_forward_velocity_alpha}; expected 1000.0 "
        f"(= TB lin_vel_tracking_sigma).  Both TB and WR use "
        f"exp(-sigma * err²) — multiplicative — so alpha=sigma."
    )


def test_smoke8b_penalty_pose_anchor_is_home(smoke8b_cfg) -> None:
    """TB penalty_pose (mjx_env.py:2700-2703) anchors to default_motor_pos
    (constant home), NOT to the moving reference.  WR's default
    behavior is q_err = q_actual - q_ref(t) — that's a hidden
    joint-imitation penalty even when ref_q_track is disabled.

    smoke8b sets env.loc_ref_penalty_pose_anchor: home so the penalty
    matches TB exactly (q_err = q_actual - home_q_rad).
    """
    assert smoke8b_cfg.env.loc_ref_penalty_pose_anchor == "home", (
        f"smoke8b env.loc_ref_penalty_pose_anchor = "
        f"{smoke8b_cfg.env.loc_ref_penalty_pose_anchor!r}; expected "
        f"'home'.  Without this flag, penalty_pose is a hidden "
        f"joint-imitation term against the ZMP trajectory — not "
        f"TB-aligned even with ref_q_track=0."
    )


def test_smoke8b_non_tb_aligned_foot_rewards_disabled(smoke8b_cfg) -> None:
    """feet_distance and ref_feet_z_track are NOT faithful TB analogs:
    - feet_distance (env:1063) = positive smooth band reward over
      [min_y, max_y].  TB penalty_close_feet_xy (mjx_env.py:2710) =
      binary -1.0 when distance < threshold, no upper bound.
    - ref_feet_z_track (env:1125) tracks ZMP-prior foot-z.  TB
      feet_phase (walk_env.py:631) is phase-derived expected height,
      command-gated.  Different mechanisms.

    Disabled in smoke8b until WR implements true TB-equivalent terms
    (smoke9 candidate).  Setting these to nonzero in smoke8b would
    silently re-introduce non-TB reward semantics.
    """
    rw = smoke8b_cfg.reward_weights
    failures = []
    if rw.feet_distance != 0.0:
        failures.append(
            f"  feet_distance = {rw.feet_distance}; must be 0 in smoke8b "
            f"(WR implementation is shape-wrong vs TB penalty_close_feet_xy)"
        )
    if rw.ref_feet_z_track != 0.0:
        failures.append(
            f"  ref_feet_z_track = {rw.ref_feet_z_track}; must be 0 in "
            f"smoke8b (WR implementation is ZMP-imitation, not TB's "
            f"phase-derived feet_phase)"
        )
    if failures:
        pytest.fail(
            "smoke8b: foot rewards without faithful TB equivalents "
            "must be disabled (set to 0):\n" + "\n".join(failures)
        )
