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


def test_invalid_controller_stack_fails_fast():
    cfg = load_training_config("training/configs/mpc_standing_v0173.yaml")
    cfg.env.controller_stack = "mpc_standng"
    cfg_path = Path("training/configs/__tmp_invalid_controller_stack.yaml")
    try:
        cfg_path.write_text(
            Path("training/configs/mpc_standing_v0173.yaml")
            .read_text()
            .replace("controller_stack: mpc_standing", "controller_stack: mpc_standng"),
            encoding="utf-8",
        )
        try:
            load_training_config(str(cfg_path))
            assert False, "Expected ValueError for invalid controller_stack"
        except ValueError as exc:
            assert "controller_stack" in str(exc)
    finally:
        if cfg_path.exists():
            cfg_path.unlink()


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
    assert float(metrics["debug/mpc_planner_active"]) == 1.0
    assert float(metrics["debug/mpc_controller_active"]) == 1.0
