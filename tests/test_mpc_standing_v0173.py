"""Smoke tests for v0.17.3 architecture-pivot standing bring-up path."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jp

from assets.robot_config import load_robot_config
from training.configs.training_config import load_training_config
from training.envs.wildrobot_env import WildRobotEnv
from training.core.metrics_registry import METRICS_VEC_KEY, unpack_metrics


def test_v0173_config_loads():
    cfg = load_training_config("training/configs/mpc_standing_v0173.yaml")
    assert cfg.version == "0.17.3"
    assert cfg.env.controller_stack == "mpc_standing"
    assert cfg.env.base_ctrl_enabled is False
    assert cfg.env.fsm_enabled is False
    assert cfg.env.mpc_residual_scale > 0.0


def test_v0173_env_init_and_step():
    cfg = load_training_config("training/configs/mpc_standing_v0173.yaml")
    load_robot_config(Path(cfg.env.assets_root) / "mujoco_robot_config.json")
    cfg.freeze()

    env = WildRobotEnv(config=cfg)
    rng = jax.random.PRNGKey(123)
    state = jax.jit(env.reset)(rng)
    action = jp.zeros((env.action_size,), dtype=jp.float32)
    state = jax.jit(env.step)(state, action)

    assert jp.isfinite(state.obs).all()
    assert jp.isfinite(state.reward)

    metrics = unpack_metrics(state.metrics[METRICS_VEC_KEY])
    assert "debug/mpc_planner_active" in metrics
    assert "debug/mpc_controller_active" in metrics
    assert "debug/mpc_step_requested" in metrics
    assert float(metrics["debug/mpc_planner_active"]) >= 0.0
    assert float(metrics["debug/mpc_controller_active"]) >= 0.0

