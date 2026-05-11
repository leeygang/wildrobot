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
    """
    rw = smoke8b_cfg.reward_weights
    expected = {
        # alive: TB RewardScales.alive = 1.0
        "alive": 1.0,
        # cmd_forward_velocity_track: TB lin_vel_xy = 2.0
        "cmd_forward_velocity_track": 2.0,
        # ref_body_quat_track: TB torso_quat = 2.5
        "ref_body_quat_track": 2.5,
        # ang_vel_xy: TB penalty_ang_vel_xy = 1.0
        "ang_vel_xy": 1.0,
        # action_rate: TB penalty_action_rate = 2.0 (sign convention diff)
        "action_rate": -2.0,
        # penalty_pose: TB walk.gin = 0.5 (sign convention diff)
        "penalty_pose": -0.5,
        # feet_distance: TB penalty_close_feet_xy = 10.0
        "feet_distance": 10.0,
        # ref_feet_z_track: TB feet_phase = 7.5
        "ref_feet_z_track": 7.5,
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
    """TB lin_vel_tracking_sigma = 1000 ⇒ exp(-err²/1000).
    WR formula: exp(-alpha * err²) ⇒ alpha = 1/sigma = 0.001.

    smoke7/8 used alpha=200 (effective sigma=1/200=0.005), which is
    ~450× tighter than TB.  smoke8b loosens to TB's 0.001.
    """
    rw = smoke8b_cfg.reward_weights
    assert rw.cmd_forward_velocity_alpha == 0.001, (
        f"smoke8b cmd_forward_velocity_alpha = "
        f"{rw.cmd_forward_velocity_alpha}; expected 0.001 "
        f"(= 1 / TB lin_vel_tracking_sigma=1000).  smoke7/8 used 200, "
        f"which was ~450× tighter than TB."
    )


def test_smoke8b_close_feet_threshold_matches_tb(smoke8b_cfg) -> None:
    """TB close_feet_threshold = 0.06 (walk.gin:125).  smoke7/8 used
    min_feet_y_dist = 0.07 (which sets the lower bound of WR's
    feet_distance band penalty).  smoke8b tightens to TB's 0.06.
    """
    assert smoke8b_cfg.env.min_feet_y_dist == 0.06, (
        f"smoke8b env.min_feet_y_dist = "
        f"{smoke8b_cfg.env.min_feet_y_dist}; expected 0.06 (TB "
        f"close_feet_threshold)."
    )
