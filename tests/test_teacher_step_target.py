from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import pytest

from assets.robot_config import load_robot_config
from training.configs.training_config import load_training_config
from training.core.metrics_registry import METRICS_VEC_KEY, unpack_metrics
from training.envs.teacher_step_target import (
    compute_teacher_step_required_reward,
    compute_teacher_swing_foot_reward,
    compute_teacher_step_target,
    compute_teacher_target_step_xy_reward,
)
from training.envs.teacher_whole_body_target import (
    compute_teacher_whole_body_target,
    compute_teacher_recovery_height_reward,
)
from training.envs.wildrobot_env import WildRobotEnv
from training.policy_spec_utils import build_policy_spec_from_training_config
from policy_contract.spec import policy_spec_hash


def test_teacher_target_clipping_and_reachability() -> None:
    out = compute_teacher_step_target(
        teacher_enabled=True,
        push_active=jnp.asarray(True),
        recovery_active=jnp.asarray(False),
        need_step=jnp.asarray(1.0, dtype=jnp.float32),
        pitch=jnp.asarray(0.5, dtype=jnp.float32),
        roll=jnp.asarray(0.4, dtype=jnp.float32),
        pitch_rate=jnp.asarray(2.0, dtype=jnp.float32),
        forward_vel=jnp.asarray(1.0, dtype=jnp.float32),
        lateral_vel=jnp.asarray(0.7, dtype=jnp.float32),
        capture_error_xy=jnp.asarray([0.4, 0.3], dtype=jnp.float32),
        hard_threshold=0.6,
        x_min=-0.10,
        x_max=0.10,
        left_y_min=0.02,
        left_y_max=0.06,
        right_y_min=-0.06,
        right_y_max=-0.02,
    )
    assert float(out.teacher_active) == 1.0
    assert float(out.active_count) == 1.0
    assert float(out.target_step_x) >= -0.100001
    assert float(out.target_step_x) <= 0.100001
    if int(out.swing_foot) == 0:
        assert 0.02 <= float(out.target_step_y) <= 0.06
    else:
        assert -0.06 <= float(out.target_step_y) <= -0.02
    assert float(out.target_reachable) in (0.0, 1.0)


def test_teacher_inactive_in_quiet_case() -> None:
    out = compute_teacher_step_target(
        teacher_enabled=True,
        push_active=jnp.asarray(False),
        recovery_active=jnp.asarray(False),
        need_step=jnp.asarray(0.0, dtype=jnp.float32),
        pitch=jnp.asarray(0.0, dtype=jnp.float32),
        roll=jnp.asarray(0.0, dtype=jnp.float32),
        pitch_rate=jnp.asarray(0.0, dtype=jnp.float32),
        forward_vel=jnp.asarray(0.0, dtype=jnp.float32),
        lateral_vel=jnp.asarray(0.0, dtype=jnp.float32),
        capture_error_xy=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        hard_threshold=0.6,
        x_min=-0.10,
        x_max=0.10,
        left_y_min=0.02,
        left_y_max=0.06,
        right_y_min=-0.06,
        right_y_max=-0.02,
    )
    assert float(out.teacher_active) == 0.0
    assert float(out.active_count) == 0.0
    assert float(out.step_required_soft) == 0.0
    assert float(out.raw_step_required_soft) == 0.0
    assert int(out.swing_foot) == -1


def test_teacher_active_in_disturbed_case() -> None:
    out = compute_teacher_step_target(
        teacher_enabled=True,
        push_active=jnp.asarray(True),
        recovery_active=jnp.asarray(False),
        need_step=jnp.asarray(0.8, dtype=jnp.float32),
        pitch=jnp.asarray(0.2, dtype=jnp.float32),
        roll=jnp.asarray(0.1, dtype=jnp.float32),
        pitch_rate=jnp.asarray(0.8, dtype=jnp.float32),
        forward_vel=jnp.asarray(0.2, dtype=jnp.float32),
        lateral_vel=jnp.asarray(0.2, dtype=jnp.float32),
        capture_error_xy=jnp.asarray([0.05, 0.02], dtype=jnp.float32),
        hard_threshold=0.6,
        x_min=-0.10,
        x_max=0.10,
        left_y_min=0.02,
        left_y_max=0.06,
        right_y_min=-0.06,
        right_y_max=-0.02,
    )
    assert float(out.teacher_active) == 1.0
    assert float(out.active_count) == 1.0
    assert float(out.step_required_soft) > 0.0
    assert float(out.raw_step_required_soft) >= float(out.step_required_soft)


def test_teacher_xy_reward_semantics() -> None:
    reward_near, err_near = compute_teacher_target_step_xy_reward(
        touchdown_x=jnp.asarray(0.05, dtype=jnp.float32),
        touchdown_y=jnp.asarray(0.03, dtype=jnp.float32),
        target_x=jnp.asarray(0.05, dtype=jnp.float32),
        target_y=jnp.asarray(0.03, dtype=jnp.float32),
        teacher_active=jnp.asarray(1.0, dtype=jnp.float32),
        first_recovery_touchdown=jnp.asarray(1.0, dtype=jnp.float32),
        sigma=0.06,
    )
    reward_far, err_far = compute_teacher_target_step_xy_reward(
        touchdown_x=jnp.asarray(0.10, dtype=jnp.float32),
        touchdown_y=jnp.asarray(0.06, dtype=jnp.float32),
        target_x=jnp.asarray(-0.10, dtype=jnp.float32),
        target_y=jnp.asarray(-0.06, dtype=jnp.float32),
        teacher_active=jnp.asarray(1.0, dtype=jnp.float32),
        first_recovery_touchdown=jnp.asarray(1.0, dtype=jnp.float32),
        sigma=0.06,
    )
    reward_off, _ = compute_teacher_target_step_xy_reward(
        touchdown_x=jnp.asarray(0.05, dtype=jnp.float32),
        touchdown_y=jnp.asarray(0.03, dtype=jnp.float32),
        target_x=jnp.asarray(0.05, dtype=jnp.float32),
        target_y=jnp.asarray(0.03, dtype=jnp.float32),
        teacher_active=jnp.asarray(0.0, dtype=jnp.float32),
        first_recovery_touchdown=jnp.asarray(1.0, dtype=jnp.float32),
        sigma=0.06,
    )
    assert float(err_near) < float(err_far)
    assert float(reward_near) > float(reward_far)
    assert float(reward_off) == 0.0


def test_teacher_step_required_reward_semantics() -> None:
    reward_liftoff, event_liftoff = compute_teacher_step_required_reward(
        teacher_active=jnp.asarray(1.0, dtype=jnp.float32),
        step_required_soft=jnp.asarray(0.75, dtype=jnp.float32),
        first_recovery_liftoff=jnp.asarray(1.0, dtype=jnp.float32),
    )
    reward_clean, event_clean = compute_teacher_step_required_reward(
        teacher_active=jnp.asarray(0.0, dtype=jnp.float32),
        step_required_soft=jnp.asarray(1.0, dtype=jnp.float32),
        first_recovery_liftoff=jnp.asarray(1.0, dtype=jnp.float32),
    )
    reward_none, event_none = compute_teacher_step_required_reward(
        teacher_active=jnp.asarray(1.0, dtype=jnp.float32),
        step_required_soft=jnp.asarray(1.0, dtype=jnp.float32),
        first_recovery_liftoff=jnp.asarray(0.0, dtype=jnp.float32),
    )
    assert float(event_liftoff) == 1.0
    assert float(reward_liftoff) == pytest.approx(0.75)
    assert float(event_clean) == 0.0
    assert float(reward_clean) == 0.0
    assert float(event_none) == 0.0
    assert float(reward_none) == 0.0


def test_teacher_swing_foot_reward_semantics() -> None:
    reward_match, event_match = compute_teacher_swing_foot_reward(
        touchdown_foot=jnp.asarray(0, dtype=jnp.int32),
        teacher_swing_foot=jnp.asarray(0, dtype=jnp.int32),
        teacher_active=jnp.asarray(1.0, dtype=jnp.float32),
        teacher_step_required_hard=jnp.asarray(1.0, dtype=jnp.float32),
        first_recovery_touchdown=jnp.asarray(1.0, dtype=jnp.float32),
    )
    reward_mismatch, event_mismatch = compute_teacher_swing_foot_reward(
        touchdown_foot=jnp.asarray(1, dtype=jnp.int32),
        teacher_swing_foot=jnp.asarray(0, dtype=jnp.int32),
        teacher_active=jnp.asarray(1.0, dtype=jnp.float32),
        teacher_step_required_hard=jnp.asarray(1.0, dtype=jnp.float32),
        first_recovery_touchdown=jnp.asarray(1.0, dtype=jnp.float32),
    )
    reward_clean, event_clean = compute_teacher_swing_foot_reward(
        touchdown_foot=jnp.asarray(0, dtype=jnp.int32),
        teacher_swing_foot=jnp.asarray(0, dtype=jnp.int32),
        teacher_active=jnp.asarray(0.0, dtype=jnp.float32),
        teacher_step_required_hard=jnp.asarray(1.0, dtype=jnp.float32),
        first_recovery_touchdown=jnp.asarray(1.0, dtype=jnp.float32),
    )
    assert float(event_match) == 1.0
    assert float(reward_match) == 1.0
    assert float(event_mismatch) == 1.0
    assert float(reward_mismatch) == 0.0
    assert float(event_clean) == 0.0
    assert float(reward_clean) == 0.0


def test_v0174t_observation_and_teacher_metrics_present() -> None:
    cfg = load_training_config("training/configs/ppo_standing_v0174t.yaml")
    load_robot_config(Path(cfg.env.assets_root) / "mujoco_robot_config.json")
    cfg.freeze()

    env = WildRobotEnv(config=cfg)
    rng = jax.random.PRNGKey(7)
    state = jax.jit(env.reset)(rng)
    obs_dim_before = int(state.obs.shape[-1])
    action = jnp.zeros((env.action_size,), dtype=jnp.float32)
    state = jax.jit(env.step)(state, action)
    obs_dim_after = int(state.obs.shape[-1])
    assert obs_dim_before == obs_dim_after == env.observation_size

    metrics = unpack_metrics(state.metrics[METRICS_VEC_KEY])
    for key in (
        "teacher/active_frac",
        "teacher/step_required_mean",
        "teacher/target_step_x_mean",
        "teacher/target_step_y_mean",
        "reward/teacher_target_step_xy",
    ):
        assert key in metrics
        assert jnp.isfinite(jnp.asarray(metrics[key]))


def test_v0174t_step_observation_contract_and_clean_gating() -> None:
    cfg_base = load_training_config("training/configs/ppo_standing_v0174t.yaml")
    cfg_step = load_training_config("training/configs/ppo_standing_v0174t_step.yaml")
    load_robot_config(Path(cfg_step.env.assets_root) / "mujoco_robot_config.json")
    cfg_base.freeze()
    cfg_step.freeze()

    env_base = WildRobotEnv(config=cfg_base)
    env_step = WildRobotEnv(config=cfg_step)
    assert env_base.observation_size == env_step.observation_size

    rng = jax.random.PRNGKey(11)
    state = jax.jit(env_step.reset)(rng)
    obs_dim_before = int(state.obs.shape[-1])
    action = jnp.zeros((env_step.action_size,), dtype=jnp.float32)
    state = jax.jit(env_step.step)(state, action)
    obs_dim_after = int(state.obs.shape[-1])
    assert obs_dim_before == obs_dim_after == env_step.observation_size

    metrics = unpack_metrics(state.metrics[METRICS_VEC_KEY])
    assert float(metrics["teacher/active_frac"]) == 0.0
    assert float(metrics["teacher/teacher_active_during_clean_frac"]) == 0.0
    for key in (
        "reward/teacher_step_required",
        "reward/teacher_swing_foot",
        "reward/teacher_recovery_height",
        "reward/teacher_com_velocity_reduction",
        "teacher/step_required_event_rate",
        "teacher/swing_foot_match_events",
        "teacher/whole_body_active_frac",
        "teacher/whole_body_active_during_clean_frac",
        "teacher/whole_body_active_during_push_frac",
        "teacher/recovery_height_target_mean",
        "teacher/recovery_height_error",
        "teacher/recovery_height_in_band_frac",
        "teacher/com_velocity_target_mean",
        "teacher/com_velocity_error",
        "teacher/com_velocity_target_hit_frac",
        "recovery/visible_step_rate",
        "visible_step_rate_hard",
    ):
        assert key in metrics
        assert jnp.isfinite(jnp.asarray(metrics[key]))


def test_whole_body_teacher_disturbed_hard_gate_activation() -> None:
    out = compute_teacher_whole_body_target(
        whole_body_teacher_enabled=True,
        push_active=jnp.asarray(True),
        recovery_active=jnp.asarray(False),
        step_required_hard=jnp.asarray(1.0, dtype=jnp.float32),
        height=jnp.asarray(0.40, dtype=jnp.float32),
        horizontal_speed=jnp.asarray(0.22, dtype=jnp.float32),
        height_target_min=0.39,
        height_target_max=0.42,
        height_hard_gate=True,
        com_vel_target=0.12,
    )
    assert float(out.whole_body_active) == 1.0
    assert float(out.whole_body_active_count) == 1.0
    assert 0.39 <= float(out.recovery_height_target) <= 0.42
    assert float(out.recovery_height_error) == pytest.approx(0.0)
    assert float(out.recovery_height_in_band) == 1.0


def test_whole_body_teacher_clean_window_leakage_zero() -> None:
    out = compute_teacher_whole_body_target(
        whole_body_teacher_enabled=True,
        push_active=jnp.asarray(False),
        recovery_active=jnp.asarray(False),
        step_required_hard=jnp.asarray(1.0, dtype=jnp.float32),
        height=jnp.asarray(0.35, dtype=jnp.float32),
        horizontal_speed=jnp.asarray(0.35, dtype=jnp.float32),
        height_target_min=0.39,
        height_target_max=0.42,
        height_hard_gate=True,
        com_vel_target=0.12,
    )
    assert float(out.whole_body_active) == 0.0
    assert float(out.whole_body_active_count) == 0.0
    assert float(out.recovery_height_target) == 0.0
    assert float(out.recovery_height_error) == 0.0
    assert float(out.com_velocity_error) == 0.0


def test_whole_body_teacher_hard_gate_blocks_when_step_not_required() -> None:
    out = compute_teacher_whole_body_target(
        whole_body_teacher_enabled=True,
        push_active=jnp.asarray(True),
        recovery_active=jnp.asarray(False),
        step_required_hard=jnp.asarray(0.0, dtype=jnp.float32),
        height=jnp.asarray(0.36, dtype=jnp.float32),
        horizontal_speed=jnp.asarray(0.28, dtype=jnp.float32),
        height_target_min=0.39,
        height_target_max=0.42,
        height_hard_gate=True,
        com_vel_target=0.12,
    )
    assert float(out.whole_body_active) == 0.0
    assert float(out.recovery_height_error) == 0.0
    reward = compute_teacher_recovery_height_reward(
        height_error=out.recovery_height_error,
        teacher_active=out.whole_body_active,
        sigma=0.03,
    )
    assert float(reward) == 0.0


def test_crouch_teacher_policy_contract_hash_unchanged_vs_crouch() -> None:
    cfg_a = load_training_config("training/configs/ppo_standing_v0174t_crouch.yaml")
    cfg_b = load_training_config("training/configs/ppo_standing_v0174t_crouch_teacher.yaml")
    cfg_a.freeze()
    cfg_b.freeze()
    robot_cfg = load_robot_config(Path(cfg_a.env.assets_root) / "mujoco_robot_config.json")
    spec_a = build_policy_spec_from_training_config(training_cfg=cfg_a, robot_cfg=robot_cfg)
    spec_b = build_policy_spec_from_training_config(training_cfg=cfg_b, robot_cfg=robot_cfg)
    assert policy_spec_hash(spec_a) == policy_spec_hash(spec_b)


def test_crouch_teacher_config_parses_whole_body_keys() -> None:
    cfg = load_training_config("training/configs/ppo_standing_v0174t_crouch_teacher.yaml")
    assert cfg.env.whole_body_teacher_enabled is True
    assert float(cfg.env.whole_body_teacher_height_target_min) == pytest.approx(0.39)
    assert float(cfg.env.whole_body_teacher_height_target_max) == pytest.approx(0.42)
    assert cfg.env.whole_body_teacher_height_hard_gate is True
    assert float(cfg.reward_weights.teacher_recovery_height) > 0.0
    assert float(cfg.reward_weights.teacher_com_velocity_reduction) == pytest.approx(0.0)
