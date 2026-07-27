from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from assets.robot_config import load_robot_config
from policy_contract.spec import PolicySpec
from training.sim_adapter.mjx_signals import MjxSignalsAdapter
from training.sim_adapter.mujoco_signals import MujocoSignalsAdapter


def _load_assets():
    mujoco = pytest.importorskip("mujoco")
    root = Path(__file__).resolve().parents[1]
    xml_path = root / "assets" / "v2" / "wildrobot.xml"
    cfg_path = root / "assets" / "v2" / "mujoco_robot_config.json"
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


def test_mujoco_signals_adapter_converts_framequat_to_xyzw() -> None:
    mujoco, mj_model, mj_data, robot_cfg, policy_spec = _load_assets()
    adapter = MujocoSignalsAdapter(
        mj_model=mj_model,
        robot_config=robot_cfg,
        policy_spec=policy_spec,
        foot_switch_threshold=10.0,
    )

    sensor_id = mujoco.mj_name2id(
        mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "chest_imu_quat"
    )
    sensor_adr = int(mj_model.sensor_adr[sensor_id])
    quat_wxyz = mj_data.sensordata[sensor_adr : sensor_adr + 4]
    expected_xyzw = np.concatenate([quat_wxyz[1:4], quat_wxyz[0:1]])

    signals = adapter.read(mj_data)

    np.testing.assert_allclose(signals.quat_xyzw, expected_xyzw, atol=1e-7)
    from policy_contract.numpy.frames import gravity_local_from_quat

    gravity_local = gravity_local_from_quat(signals.quat_xyzw)
    assert gravity_local[2] < -0.99


def test_mjx_signals_adapter_matches_native_quaternion_contract() -> None:
    mujoco, mj_model, mj_data, robot_cfg, _ = _load_assets()
    mjx = pytest.importorskip("mujoco.mjx")
    adapter = MjxSignalsAdapter(
        mj_model=mj_model,
        robot_config=robot_cfg,
        foot_switch_threshold=10.0,
    )

    signals = adapter.read(mjx.put_data(mj_model, mj_data))

    sensor_id = mujoco.mj_name2id(
        mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "chest_imu_quat"
    )
    sensor_adr = int(mj_model.sensor_adr[sensor_id])
    quat_wxyz = mj_data.sensordata[sensor_adr : sensor_adr + 4]
    expected_xyzw = np.concatenate([quat_wxyz[1:4], quat_wxyz[0:1]])
    np.testing.assert_allclose(
        np.asarray(signals.quat_xyzw), expected_xyzw, atol=1e-7
    )


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
