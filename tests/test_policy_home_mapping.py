from __future__ import annotations

import numpy as np


def test_build_policy_spec_threads_home_ctrl_from_training_config() -> None:
    from assets.robot_config import load_robot_config, get_robot_config
    from training.configs.training_config import load_training_config
    from training.policy_spec_utils import (
        build_policy_spec_from_training_config,
        maybe_get_home_ctrl_from_training_config,
    )

    load_robot_config("assets/v2/mujoco_robot_config.json")
    training_cfg = load_training_config("training/configs/ppo_standing_v0173a.yaml")
    robot_cfg = get_robot_config()

    spec = build_policy_spec_from_training_config(
        training_cfg=training_cfg,
        robot_cfg=robot_cfg,
    )
    expected_home = maybe_get_home_ctrl_from_training_config(
        training_cfg=training_cfg,
        robot_cfg=robot_cfg,
    )

    assert spec.action.mapping_id == "pos_target_home_v1"
    assert spec.robot.home_ctrl_rad is not None
    np.testing.assert_allclose(spec.robot.home_ctrl_rad, expected_home, atol=1e-8)


def test_home_mapping_zero_action_maps_to_home_ctrl() -> None:
    from policy_contract.calib import JaxCalibOps, NumpyCalibOps
    from policy_contract.spec_builder import build_policy_spec

    actuated_joint_specs = [
        {
            "name": "joint",
            "range": [-0.2, 1.0],
            "policy_action_sign": 1.0,
            "max_velocity": 10.0,
        }
    ]
    spec = build_policy_spec(
        robot_name="wildrobot",
        actuated_joint_specs=actuated_joint_specs,
        action_filter_alpha=0.0,
        layout_id="wr_obs_v1",
        mapping_id="pos_target_home_v1",
        home_ctrl_rad=[0.3],
    )

    zero = np.zeros((1,), dtype=np.float32)
    ctrl_np = NumpyCalibOps.action_to_ctrl(spec=spec, action=zero)
    ctrl_jax = np.asarray(JaxCalibOps.action_to_ctrl(spec=spec, action=zero))

    np.testing.assert_allclose(ctrl_np, np.array([0.3], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(ctrl_jax, np.array([0.3], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(
        NumpyCalibOps.ctrl_to_policy_action(spec=spec, ctrl_rad=np.array([0.3], dtype=np.float32)),
        zero,
        atol=1e-6,
    )
