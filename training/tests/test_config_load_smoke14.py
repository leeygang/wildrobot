"""Pin combined-fix contract for smoke14.yaml.

Smoke14 = smoke13_026 (velocity-reward TB-parity fix + 0.18-0.26 m/s
WR-scaled command range + 30M compute budget) PLUS critic obs
temporal stacking x15 (TB ``c_frame_stack`` parity).

These tests confirm the combined config carries BOTH fixes intact:

  - smoke13_026 parity: velocity range / eval / grid / iterations /
    smoke12b residual authority / actor obs layout / critic imitation
    refs / reward weights all match smoke13_026.
  - smoke14 delta: env.critic_obs_history_frames == 15.

The critic-stacking behavior itself is covered by
test_critic_obs_stacking.py.  This file pins the config-layer
contract.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from training.configs.training_config import load_training_config


_SMOKE13 = Path("training/configs/ppo_walking_v0201_smoke13.yaml")
_SMOKE13_026 = Path("training/configs/ppo_walking_v0201_smoke13_026.yaml")
_SMOKE14 = Path("training/configs/ppo_walking_v0201_smoke14.yaml")


@pytest.fixture(scope="module")
def cfg14():
    if not _SMOKE14.exists():
        pytest.skip(f"{_SMOKE14.name} not found")
    return load_training_config(str(_SMOKE14))


def test_loads(cfg14) -> None:
    assert cfg14 is not None


# ---------------------------------------------------------------------------
# Velocity range / eval / grid — must match smoke13_026 exactly.
# ---------------------------------------------------------------------------
def test_velocity_range_matches_smoke13_026(cfg14) -> None:
    env = cfg14.env
    assert env.min_velocity == pytest.approx(0.18, abs=1e-6)
    assert env.max_velocity == pytest.approx(0.26, abs=1e-6)
    # v0.21.0 P3 / H3: scalar YAML ``eval_velocity_cmd`` broadcasts
    # to ``(vx, 0.0, 0.0)`` (vy / wz pin to 0 — sentinel detection
    # reads only the vx axis).
    assert tuple(env.eval_velocity_cmd) == pytest.approx(
        (0.26, 0.0, 0.0), abs=1e-6
    )
    assert env.loc_ref_offline_command_vx == pytest.approx(0.26, abs=1e-6)
    assert env.cmd_zero_chance == pytest.approx(0.0)
    assert env.cmd_turn_chance == pytest.approx(0.0)
    assert env.loc_ref_command_conditioned is True
    assert env.loc_ref_command_grid_interval == pytest.approx(0.02)


def test_effective_vx_grid_018_to_026(cfg14) -> None:
    """Effective bin set from arange(0.18, 0.26+eps, 0.02) ∪ {0.26}.
    Pin the exact 5-bin grid so a future grid_interval tweak can't
    silently densify or sparsify the run."""
    from training.envs.wildrobot_env import WildRobotEnv

    env = WildRobotEnv(config=cfg14)
    grid = np.asarray(env._offline_vx_grid)
    expected = np.array([0.18, 0.20, 0.22, 0.24, 0.26], dtype=np.float32)
    np.testing.assert_allclose(grid, expected, atol=1e-5)
    # v0.21.0 P3 / H3: ``eval_velocity_cmd`` is (vx, vy, wz); use vx
    # (the [0]th axis) for the grid bin selection.
    eval_cmd = float(cfg14.env.eval_velocity_cmd[0])
    nearest = float(grid[int(np.argmin(np.abs(grid - eval_cmd)))])
    assert nearest == pytest.approx(eval_cmd, abs=1e-6)


# ---------------------------------------------------------------------------
# Combined fix invariants — both knobs must be on.
# ---------------------------------------------------------------------------
def test_critic_obs_history_frames_is_15(cfg14) -> None:
    assert cfg14.env.critic_obs_history_frames == 15


def test_cmd_velocity_track_dim_is_1(cfg14) -> None:
    """Smoke14 reverts to legacy scalar-vx tracking (dim=1).  WR has
    no lateral command, so dim=2's vy penalty against vy_cmd=0 is
    effectively a sway-suppression term orthogonal to the forward
    objective.  Dim=1 keeps the velocity reward focused on vx
    while the critic-stacking change is on trial."""
    assert cfg14.reward_weights.cmd_velocity_track_dim == 1


def test_smoke14_diverges_from_smoke13_026_only_on_critic_and_dim(cfg14) -> None:
    """Pin the EXACT delta set between smoke14 and smoke13_026.
    Anything else drifting is a config bug — those two configs are
    A/B-paired and the attribution depends on a 2-D delta."""
    if not _SMOKE13_026.exists():
        pytest.skip(f"{_SMOKE13_026.name} not found")
    cfg026 = load_training_config(str(_SMOKE13_026))
    # smoke14-only deltas:
    assert cfg14.env.critic_obs_history_frames == 15
    assert cfg026.env.critic_obs_history_frames == 1
    assert cfg14.reward_weights.cmd_velocity_track_dim == 1
    assert cfg026.reward_weights.cmd_velocity_track_dim == 2
    # Velocity range parity (the load-bearing inheritance):
    assert cfg14.env.min_velocity == pytest.approx(cfg026.env.min_velocity)
    assert cfg14.env.max_velocity == pytest.approx(cfg026.env.max_velocity)
    # v0.21.0 P3 / H3: tuple comparison via ``tuple(...)``.
    assert tuple(cfg14.env.eval_velocity_cmd) == pytest.approx(
        tuple(cfg026.env.eval_velocity_cmd)
    )
    assert cfg14.env.loc_ref_offline_command_vx == pytest.approx(
        cfg026.env.loc_ref_offline_command_vx
    )


# ---------------------------------------------------------------------------
# Compute budget — 30M target.
# ---------------------------------------------------------------------------
def test_iteration_budget_targets_30M_env_steps(cfg14) -> None:
    p = cfg14.ppo
    env_steps = int(p.iterations) * int(p.num_envs) * int(p.rollout_steps)
    target = 30_000_000
    assert env_steps >= target
    assert env_steps <= int(target * 1.05)
    assert p.iterations == 750


# ---------------------------------------------------------------------------
# Inheritance from smoke12b authority + smoke13 obs/critic decisions.
# ---------------------------------------------------------------------------
def test_smoke12b_residual_authority_preserved(cfg14) -> None:
    env = cfg14.env
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


def test_actor_and_critic_inheritance(cfg14) -> None:
    env = cfg14.env
    assert env.actor_obs_layout_id == "wr_obs_v7_phase_proprio"
    assert env.critic_imitation_refs is True
    assert env.loc_ref_residual_base == "home"
    assert env.loc_ref_reset_base == "home"
    assert env.loc_ref_penalty_pose_anchor == "home"


# ---------------------------------------------------------------------------
# Reward weights byte-equal to smoke13_026.
# ---------------------------------------------------------------------------
def test_reward_weights_byte_equal_to_smoke13_026_except_dim(cfg14) -> None:
    """All reward weights match smoke13_026 EXCEPT
    cmd_velocity_track_dim (intentional smoke14 delta: 2 → 1).
    Anything else drifting is a config bug."""
    if not _SMOKE13_026.exists():
        pytest.skip(f"{_SMOKE13_026.name} not found")
    cfg026 = load_training_config(str(_SMOKE13_026))
    for field in (
        "alive",
        "cmd_forward_velocity_track",
        "cmd_forward_velocity_alpha",
        # cmd_velocity_track_dim deliberately excluded — smoke14 delta.
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
        got = getattr(cfg14.reward_weights, field)
        want = getattr(cfg026.reward_weights, field)
        assert got == pytest.approx(want), (
            f"reward_weights.{field}={got} drifted from smoke13_026 ({want}); "
            "smoke14 is only allowed to change critic stacking + "
            "cmd_velocity_track_dim"
        )
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
        assert getattr(cfg14.reward_weights, field) == pytest.approx(0.0)


def test_ppo_hyperparameters_match_smoke13_026_except_iterations(cfg14) -> None:
    """All ppo knobs byte-equal to smoke13_026 (iterations also matches,
    both at 750)."""
    if not _SMOKE13_026.exists():
        pytest.skip(f"{_SMOKE13_026.name} not found")
    cfg026 = load_training_config(str(_SMOKE13_026))
    p = cfg14.ppo
    p026 = cfg026.ppo
    for field in (
        "num_envs",
        "rollout_steps",
        "iterations",
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
        assert getattr(p, field) == pytest.approx(getattr(p026, field)), (
            f"ppo.{field} drifted from smoke13_026"
        )


# ---------------------------------------------------------------------------
# Smoke13 retains the legacy depth (critic stacking is the smoke14 delta).
# ---------------------------------------------------------------------------
def test_smoke13_still_legacy_depth() -> None:
    if not _SMOKE13.exists():
        pytest.skip(f"{_SMOKE13.name} not found")
    cfg13 = load_training_config(str(_SMOKE13))
    assert cfg13.env.critic_obs_history_frames == 1, (
        "smoke13 must keep the legacy single-frame critic obs so prior "
        "runs are bit-equal; critic stacking is the smoke14-only delta"
    )


def test_smoke13_026_still_legacy_depth() -> None:
    if not _SMOKE13_026.exists():
        pytest.skip(f"{_SMOKE13_026.name} not found")
    cfg026 = load_training_config(str(_SMOKE13_026))
    assert cfg026.env.critic_obs_history_frames == 1, (
        "smoke13_026 is the velocity-fix-only isolation; it must NOT "
        "carry the smoke14 critic-stacking change"
    )
