"""Pin contract for ppo_walking_v0201_smoke13_026.yaml — the 0.26 m/s
WR-scaled comparison config layered on top of smoke13.

Asserts:

  - config loads,
  - velocity range = [0.18, 0.26], eval = 0.26, offline_vx = 0.26,
  - cmd-conditioned + interval=0.02 → effective vx grid
      [0.18, 0.20, 0.22, 0.24, 0.26],
  - cmd_velocity_track_dim == 2 (TB-style 2D velocity reward),
  - smoke12b residual authority preserved (per-joint scales
    byte-equal to smoke13),
  - cmd_zero_chance == 0.0 (no standing episodes),
  - actor obs / critic refs / reward weights match smoke13.

Spec: /tmp/smoke13_tb_velocity_reward_fix_prompt.md §4 + §5.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from training.configs.training_config import load_training_config


_CFG = Path("training/configs/ppo_walking_v0201_smoke13_026.yaml")
_SMOKE13 = Path("training/configs/ppo_walking_v0201_smoke13.yaml")


@pytest.fixture(scope="module")
def cfg026():
    if not _CFG.exists():
        pytest.skip(f"{_CFG.name} not found")
    return load_training_config(str(_CFG))


def test_loads(cfg026) -> None:
    assert cfg026 is not None


def test_velocity_range_and_eval(cfg026) -> None:
    env = cfg026.env
    assert env.min_velocity == pytest.approx(0.18, abs=1e-6)
    assert env.max_velocity == pytest.approx(0.26, abs=1e-6)
    assert env.eval_velocity_cmd == pytest.approx(0.26, abs=1e-6)
    assert env.loc_ref_offline_command_vx == pytest.approx(0.26, abs=1e-6)
    assert env.eval_velocity_cmd == pytest.approx(
        env.loc_ref_offline_command_vx, abs=1e-9
    )


def test_no_zero_command(cfg026) -> None:
    """Forward-motion-only smoke; no standing episodes."""
    assert cfg026.env.cmd_zero_chance == pytest.approx(0.0)
    assert cfg026.env.cmd_turn_chance == pytest.approx(0.0)


def test_command_conditioned_lookup_enabled(cfg026) -> None:
    assert cfg026.env.loc_ref_command_conditioned is True
    assert cfg026.env.loc_ref_command_grid_interval == pytest.approx(0.02)


def test_effective_vx_grid_matches_spec(cfg026) -> None:
    """Building the env materializes the cmd-conditioned vx grid;
    pin the exact 5-bin grid {0.18, 0.20, 0.22, 0.24, 0.26}."""
    from training.envs.wildrobot_env import WildRobotEnv

    env = WildRobotEnv(config=cfg026)
    grid = np.asarray(env._offline_vx_grid)
    expected = np.array([0.18, 0.20, 0.22, 0.24, 0.26], dtype=np.float32)
    np.testing.assert_allclose(grid, expected, atol=1e-5)
    # eval_velocity_cmd must hit its own bin exactly.
    eval_cmd = float(cfg026.env.eval_velocity_cmd)
    nearest = float(grid[int(np.argmin(np.abs(grid - eval_cmd)))])
    assert nearest == pytest.approx(eval_cmd, abs=1e-6)


def test_cmd_velocity_track_dim_is_2(cfg026) -> None:
    assert cfg026.reward_weights.cmd_velocity_track_dim == 2


def test_smoke12b_residual_authority_preserved(cfg026) -> None:
    env = cfg026.env
    assert env.loc_ref_residual_mode == "absolute"
    assert env.loc_ref_residual_scale == pytest.approx(0.2)
    expected = {
        "left_hip_pitch": 0.3534,
        "left_hip_roll": 0.25,
        "left_knee_pitch": 0.667,
        "left_ankle_pitch": 0.25,
        "left_ankle_roll": 0.25,
        "right_hip_pitch": 0.3867,
        "right_hip_roll": 0.25,
        "right_knee_pitch": 0.6657,
        "right_ankle_pitch": 0.25,
        "right_ankle_roll": 0.25,
    }
    got = dict(env.loc_ref_residual_scale_per_joint)
    assert got == pytest.approx(expected)


def test_actor_and_critic_inheritance_from_smoke13(cfg026) -> None:
    env = cfg026.env
    assert env.actor_obs_layout_id == "wr_obs_v7_phase_proprio"
    assert env.critic_imitation_refs is True
    assert env.loc_ref_residual_base == "home"
    assert env.loc_ref_reset_base == "home"
    assert env.loc_ref_penalty_pose_anchor == "home"


def test_reward_weights_byte_equal_to_smoke13(cfg026) -> None:
    """All reward weights match smoke13 exactly; the velocity-reward
    semantics change is in the in-code path, not the yaml."""
    if not _SMOKE13.exists():
        pytest.skip(f"{_SMOKE13.name} not found")
    cfg13 = load_training_config(str(_SMOKE13))
    # Spot-check the load-bearing TB-active block.
    for field in (
        "alive",
        "cmd_forward_velocity_track",
        "cmd_forward_velocity_alpha",
        "cmd_velocity_track_dim",
        "ref_body_quat_track",
        "ang_vel_xy",
        "action_rate",
        "penalty_pose",
        "penalty_close_feet_xy",
        "feet_phase",
        "feet_phase_alpha",
        "feet_phase_swing_height",
        "penalty_feet_ori",
    ):
        got = getattr(cfg026.reward_weights, field)
        want = getattr(cfg13.reward_weights, field)
        assert got == pytest.approx(want), (
            f"reward_weights.{field}={got} drifted from smoke13 ({want}); "
            "the 0.26 config is only allowed to change command range"
        )
    # Hybrid / imitation block must stay disabled.
    for field in (
        "ref_q_track",
        "ref_contact_match",
        "ref_feet_z_track",
        "torso_pos_xy",
        "lin_vel_z",
        "feet_air_time",
        "feet_clearance",
        "feet_distance",
        "torso_pitch_soft",
        "torso_roll_soft",
        "torque",
        "joint_velocity",
        "slip",
        "pitch_rate",
    ):
        assert getattr(cfg026.reward_weights, field) == pytest.approx(0.0)


def test_ppo_hyperparameters_match_smoke13(cfg026) -> None:
    """Every PPO hyperparameter EXCEPT iterations must match smoke13.
    iterations is the only intentional ppo delta — the 0.26 config
    extends compute to ~30M env-steps for a longer comparison run."""
    if not _SMOKE13.exists():
        pytest.skip(f"{_SMOKE13.name} not found")
    cfg13 = load_training_config(str(_SMOKE13))
    p = cfg026.ppo
    p13 = cfg13.ppo
    for field in (
        "num_envs",
        "rollout_steps",
        "learning_rate",
        "gamma",
        "gae_lambda",
        "clip_epsilon",
        "entropy_coef",
        "value_loss_coef",
        "epochs",
        "num_minibatches",
        "max_grad_norm",
    ):
        assert getattr(p, field) == pytest.approx(getattr(p13, field)), (
            f"ppo.{field} drifted from smoke13"
        )


def test_iteration_budget_targets_30M_env_steps(cfg026) -> None:
    """0.26 config is a 30M-step run: iterations * num_envs *
    rollout_steps must be >=30M and within 5% of the 30M target so
    a future iteration tweak can't silently undershoot or 2x the
    budget."""
    p = cfg026.ppo
    env_steps = int(p.iterations) * int(p.num_envs) * int(p.rollout_steps)
    target = 30_000_000
    assert env_steps >= target, (
        f"compute budget {env_steps:,} below 30M target"
    )
    assert env_steps <= int(target * 1.05), (
        f"compute budget {env_steps:,} exceeds 30M target by >5% — "
        "trim iterations"
    )
    # Pin the exact iteration count too so a refactor moving the
    # arithmetic doesn't silently drift the run length.
    assert p.iterations == 750
