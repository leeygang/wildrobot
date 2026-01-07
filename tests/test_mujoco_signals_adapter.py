from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from assets.robot_config import load_robot_config
from policy_contract.spec import PolicySpec
from training.sim_adapter.mujoco_signals import MujocoSignalsAdapter


def _load_assets():
    mujoco = pytest.importorskip("mujoco")
    root = Path(__file__).resolve().parents[1]
    xml_path = root / "assets" / "wildrobot.xml"
    cfg_path = root / "assets" / "robot_config.yaml"
    policy_path = root / "policy_contract" / "policy_spec.json"

    mj_model = mujoco.MjModel.from_xml_path(str(xml_path))
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, mj_data)

    robot_cfg = load_robot_config(cfg_path)
    policy_spec = PolicySpec.from_json(policy_path)
    return mujoco, mj_model, mj_data, robot_cfg, policy_spec


def test_mujoco_signals_adapter_shapes() -> None:
    mujoco, mj_model, mj_data, robot_cfg, policy_spec = _load_assets()
    adapter = MujocoSignalsAdapter(
        mj_model=mj_model,
        robot_config=robot_cfg,
        policy_spec=policy_spec,
        foot_switch_threshold=10.0,
    )

    signals = adapter.read(mj_data)
    assert signals.quat_xyzw.shape == (4,)
    assert signals.gyro_rad_s.shape == (3,)
    assert signals.joint_pos_rad.shape == (len(policy_spec.robot.actuator_names),)
    assert signals.joint_vel_rad_s.shape == (len(policy_spec.robot.actuator_names),)
    assert signals.foot_switches.shape == (4,)
    assert signals.quat_xyzw.dtype == signals.joint_pos_rad.dtype


def test_mujoco_signals_adapter_missing_actuator() -> None:
    _ = pytest.importorskip("mujoco")
    _, mj_model, _, robot_cfg, policy_spec = _load_assets()
    bad_spec = replace(
        policy_spec,
        robot=replace(policy_spec.robot, actuator_names=["missing_actuator"]),
    )
    with pytest.raises(ValueError, match="Actuator 'missing_actuator' not found"):
        MujocoSignalsAdapter(
            mj_model=mj_model,
            robot_config=robot_cfg,
            policy_spec=bad_spec,
            foot_switch_threshold=10.0,
        )
