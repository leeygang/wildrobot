from __future__ import annotations

from pathlib import Path

import jax.numpy as jp
import mujoco
import numpy as np
import pytest

from assets.robot_config import load_robot_config
from training.configs.training_config import load_training_config
from training.envs.wildrobot_env import WildRobotEnv
from training.eval.v6_eval_adapter import V6EvalAdapter


CONFIG = Path("training/configs/ppo_walking_v0210_17d9_narrow_walking_base.yaml")


@pytest.fixture(scope="module")
def cfg():
    return load_training_config(CONFIG)


@pytest.fixture(scope="module")
def env(cfg):
    load_robot_config(cfg.env.robot_config_path)
    cfg.freeze()
    return WildRobotEnv(config=cfg)


def test_17d9_is_a_short_geometry_only_finetune(cfg) -> None:
    assert cfg.version == "0.21.0-17d9"
    assert cfg.ppo.iterations == 20
    assert cfg.checkpoints.interval == 5
    assert cfg.ppo.eval.post_training_top_k == 4
    assert cfg.reward_weights.saturation == pytest.approx(0.0)
    assert cfg.env.close_feet_threshold == pytest.approx(0.127)
    assert cfg.env.loc_ref_rsi_probability == pytest.approx(0.75)


def test_17d9_offsets_are_symmetric_and_only_touch_leg_roll(cfg) -> None:
    assert cfg.env.loc_ref_walking_joint_offsets_rad == {
        "left_hip_roll": 0.03,
        "right_hip_roll": -0.03,
        "left_ankle_roll": -0.03,
        "right_ankle_roll": 0.03,
    }


def test_17d9_zero_action_and_rsi_reference_share_walking_offset(env) -> None:
    names = tuple(env._policy_spec.robot.actuator_names)
    expected_offset = np.zeros(env.action_size, dtype=np.float32)
    for name, value in {
        "left_hip_roll": 0.03,
        "right_hip_roll": -0.03,
        "left_ankle_roll": -0.03,
        "right_ankle_roll": 0.03,
    }.items():
        expected_offset[names.index(name)] = value

    np.testing.assert_allclose(
        np.asarray(env._walking_home_q_rad) - np.asarray(env._home_q_rad),
        expected_offset,
        atol=1e-7,
    )

    raw_ref = np.asarray(env._offline_service.lookup_np(0).q_ref, dtype=np.float32)
    shifted_ref = np.asarray(
        env._lookup_offline_window(jp.asarray(0, dtype=jp.int32))["q_ref"]
    )
    np.testing.assert_allclose(shifted_ref - raw_ref, expected_offset, atol=1e-7)

    target, residual = env._compose_target_q_from_residual(
        policy_action=jp.zeros(env.action_size, dtype=jp.float32),
        nominal_q_ref=jp.asarray(shifted_ref),
    )
    np.testing.assert_allclose(target, env._walking_home_q_rad, atol=1e-7)
    np.testing.assert_allclose(residual, 0.0, atol=1e-7)


def test_17d9_native_eval_zero_action_matches_training_walking_base(env) -> None:
    adapter = V6EvalAdapter(
        training_cfg=env._config,
        mj_model=env._mj_model,
        policy_spec=env._policy_spec,
        signals_adapter=env._signals_adapter,
        action_dim=env.action_size,
    )
    data = mujoco.MjData(env._mj_model)
    mujoco.mj_resetDataKeyframe(env._mj_model, data, 0)
    adapter.apply_action(data, np.zeros(env.action_size, dtype=np.float32))

    target_policy_order = np.asarray(data.ctrl)[
        np.asarray(env._ctrl_mapper.policy_to_mj_order_jax)
    ]
    np.testing.assert_allclose(
        target_policy_order,
        np.asarray(env._walking_home_q_rad),
        atol=1e-6,
    )
