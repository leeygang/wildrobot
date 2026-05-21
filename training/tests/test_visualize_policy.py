from __future__ import annotations

from pathlib import Path

import mujoco
import pytest

from assets.robot_config import load_robot_config
from training.configs.training_config import load_training_config
from training.eval.v6_eval_adapter import V6EvalAdapter
from training.eval.visualize_policy import (
    _adapter_layout_enabled,
    _is_terminated_from_pose,
    _network_activation_name,
    _validate_user_fixed_velocity_cmd,
)
from training.policy_spec_utils import build_policy_spec_from_training_config
from training.sim_adapter.mujoco_signals import MujocoSignalsAdapter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SMOKE11_CFG = (
    PROJECT_ROOT / "training" / "configs" / "ppo_walking_v0201_smoke11.yaml"
)
SMOKE12B_CFG = (
    PROJECT_ROOT / "training" / "configs" / "ppo_walking_v0201_smoke12b.yaml"
)


def test_visualizer_adapter_layout_supports_v6_and_v7() -> None:
    assert _adapter_layout_enabled("wr_obs_v6_offline_ref_history") is True
    assert _adapter_layout_enabled("wr_obs_v7_phase_proprio") is True
    assert _adapter_layout_enabled("wr_obs_v4") is False


def test_visualizer_threads_actor_activation_from_training_config() -> None:
    cfg = load_training_config(str(SMOKE12B_CFG))
    assert _network_activation_name(cfg) == "elu"
    cfg.networks.critic.activation = "silu"
    with pytest.raises(ValueError, match="must match"):
        _network_activation_name(cfg)


def test_visualizer_rejects_out_of_range_fixed_velocity() -> None:
    cfg = load_training_config(str(SMOKE12B_CFG))
    assert _validate_user_fixed_velocity_cmd(cfg, 0.10) == pytest.approx(0.10)
    with pytest.raises(ValueError, match="outside configured range"):
        _validate_user_fixed_velocity_cmd(cfg, float(cfg.env.min_velocity) - 1e-3)
    with pytest.raises(ValueError, match="outside configured range"):
        _validate_user_fixed_velocity_cmd(cfg, float(cfg.env.max_velocity) + 1e-3)


def test_visualizer_relaxed_termination_ignores_pitch_roll() -> None:
    cfg = load_training_config(str(SMOKE12B_CFG))
    assert cfg.env.use_relaxed_termination is True
    assert (
        _is_terminated_from_pose(
            height=0.46,
            roll_rad=0.0,
            pitch_rad=1.2,
            step_count=10,
            training_cfg=cfg,
        )
        is False
    )
    assert (
        _is_terminated_from_pose(
            height=float(cfg.env.min_height) - 0.01,
            roll_rad=0.0,
            pitch_rad=0.0,
            step_count=10,
            training_cfg=cfg,
        )
        is True
    )


def test_visualizer_v7_layout_uses_native_eval_adapter_contract() -> None:
    cfg = load_training_config(str(SMOKE11_CFG))
    robot_cfg_path = Path(cfg.env.robot_config_path)
    if not robot_cfg_path.is_absolute():
        robot_cfg_path = PROJECT_ROOT / robot_cfg_path
    robot_cfg = load_robot_config(str(robot_cfg_path))

    scene_path = Path(cfg.env.scene_xml_path)
    if not scene_path.is_absolute():
        scene_path = PROJECT_ROOT / scene_path
    mj_model = mujoco.MjModel.from_xml_path(str(scene_path))
    mj_data = mujoco.MjData(mj_model)

    policy_spec = build_policy_spec_from_training_config(
        training_cfg=cfg,
        robot_cfg=robot_cfg,
        action_filter_alpha=float(cfg.env.action_filter_alpha),
    )
    signals_adapter = MujocoSignalsAdapter(
        mj_model=mj_model,
        robot_config=robot_cfg,
        policy_spec=policy_spec,
        foot_switch_threshold=cfg.env.foot_switch_threshold,
    )
    adapter = V6EvalAdapter(
        training_cfg=cfg,
        mj_model=mj_model,
        policy_spec=policy_spec,
        signals_adapter=signals_adapter,
        action_dim=int(policy_spec.model.action_dim),
    )
    adapter.reset_native_mj_state(mj_data, apply_noise=False, rng=None)
    obs = adapter.compute_obs(
        mj_data=mj_data,
        velocity_cmd=float(cfg.env.eval_velocity_cmd),
    )
    assert obs.shape == (policy_spec.model.obs_dim,)
