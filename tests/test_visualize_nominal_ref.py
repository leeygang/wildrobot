from __future__ import annotations

import jax.numpy as jnp

from training.core.metrics_registry import METRIC_INDEX
from training.configs.training_config import load_training_config
from training.eval.visualize_nominal_ref import (
    _configure_nominal_only,
    _dominant_termination_from_metrics,
    parse_args,
)


def test_parse_args_nominal_ref_viewer() -> None:
    args = parse_args(
        [
            "--config",
            "training/configs/ppo_walking_v0193a.yaml",
            "--forward-cmd",
            "0.10",
            "--horizon",
            "64",
            "--headless",
            "--print-every",
            "4",
            "--seed",
            "7",
            "--no-stop-on-done",
        ]
    )
    assert args.config.endswith("ppo_walking_v0193a.yaml")
    assert abs(args.forward_cmd - 0.10) < 1e-9
    assert args.horizon == 64
    assert args.headless is True
    assert args.print_every == 4
    assert args.seed == 7
    assert args.stop_on_done is False


def test_dominant_termination_from_metrics() -> None:
    vec = jnp.zeros((len(METRIC_INDEX),), dtype=jnp.float32)
    assert _dominant_termination_from_metrics(vec) == "none"

    vec = vec.at[METRIC_INDEX["term/pitch"]].set(1.0)
    vec = vec.at[METRIC_INDEX["term/height_low"]].set(0.5)
    assert _dominant_termination_from_metrics(vec) == "term/pitch"


def test_configure_nominal_only_forces_no_push_and_no_action_delay() -> None:
    cfg = load_training_config("training/configs/ppo_walking_v0193a.yaml")
    # Deliberately set conflicting values to verify normalization.
    cfg.env.push_enabled = True
    cfg.env.action_delay_steps = 1
    cfg.env.loc_ref_enabled = False
    cfg.env.base_ctrl_enabled = True
    cfg.env.fsm_enabled = True
    _configure_nominal_only(cfg, forward_cmd=0.10, horizon=32)
    assert cfg.ppo.num_envs == 1
    assert cfg.ppo.rollout_steps == 32
    assert abs(cfg.env.min_velocity - 0.10) < 1e-9
    assert abs(cfg.env.max_velocity - 0.10) < 1e-9
    assert cfg.env.loc_ref_enabled is True
    assert cfg.env.controller_stack == "ppo"
    assert cfg.env.base_ctrl_enabled is False
    assert cfg.env.fsm_enabled is False
    assert cfg.env.push_enabled is False
    assert cfg.env.action_delay_steps == 0
