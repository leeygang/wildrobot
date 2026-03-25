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
    compute_teacher_step_target,
    compute_teacher_target_step_xy_reward,
)
from training.envs.wildrobot_env import WildRobotEnv


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
