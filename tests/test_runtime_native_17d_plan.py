from __future__ import annotations

import pytest

from assets.robot_config import load_robot_config
from training.configs.training_config import load_training_config
from training.policy_spec_utils import build_policy_spec_from_training_config
from wr_runtime.control.mock_robot_io import MockRobotIO
from wr_runtime.control.run_policy import _PolicySubsetRobotIO, _walking_runtime_plan


def test_native_17d_runtime_maps_policy_directly_to_hardware() -> None:
    cfg = load_training_config(
        "training/configs/ppo_walking_v0210_smoke6_home_rsi.yaml"
    )
    robot_cfg = load_robot_config("assets/v2/mujoco_robot_config.json")
    spec = build_policy_spec_from_training_config(
        training_cfg=cfg, robot_cfg=robot_cfg
    )

    hardware_names, home, mins, maxs = _walking_runtime_plan(spec)

    assert hardware_names == list(spec.robot.actuator_names)
    assert len(hardware_names) == 17
    assert home is not None and home.shape == (17,)
    assert mins.shape == maxs.shape == (17,)
    assert not any("wrist" in name for name in hardware_names)


def test_leg_only_runtime_holds_excluded_actuators_at_home() -> None:
    cfg = load_training_config(
        "training/configs/ppo_walking_v0210_tb1_direct_ppo.yaml"
    )
    robot_cfg = load_robot_config("assets/v2/mujoco_robot_config.json")
    spec = build_policy_spec_from_training_config(
        training_cfg=cfg, robot_cfg=robot_cfg
    )
    hardware_names, home, mins, maxs = _walking_runtime_plan(spec)

    assert len(hardware_names) == 17
    assert home.shape == mins.shape == maxs.shape == (17,)
    assert len(spec.robot.actuator_names) == 10

    base = MockRobotIO(
        actuator_names=hardware_names,
        control_dt=0.02,
        home_q_rad=home,
    )
    projected = _PolicySubsetRobotIO(
        base,
        policy_actuator_names=spec.robot.actuator_names,
        hardware_actuator_names=hardware_names,
        hardware_home_q_rad=home,
    )
    active_target = projected.read().joint_pos_rad + 0.01
    projected.write_ctrl(active_target)

    full_written = base.written[-1]
    active_set = set(spec.robot.actuator_names)
    for idx, name in enumerate(hardware_names):
        expected = (
            active_target[list(spec.robot.actuator_names).index(name)]
            if name in active_set
            else home[idx]
        )
        assert full_written[idx] == pytest.approx(expected)


def test_toddlerbot_leg_policy_reads_all_hardware_actuators() -> None:
    cfg = load_training_config(
        "training/configs/ppo_walking_v0210_tb2_rsl_parity.yaml"
    )
    robot_cfg = load_robot_config("assets/v2/mujoco_robot_config.json")
    spec = build_policy_spec_from_training_config(
        training_cfg=cfg, robot_cfg=robot_cfg
    )
    hardware_names, home, _, _ = _walking_runtime_plan(spec)
    observed_names = spec.robot.observation_actuator_names
    assert observed_names is not None

    base = MockRobotIO(
        actuator_names=hardware_names,
        control_dt=0.02,
        home_q_rad=home,
    )
    projected = _PolicySubsetRobotIO(
        base,
        policy_actuator_names=spec.robot.actuator_names,
        hardware_actuator_names=hardware_names,
        hardware_home_q_rad=home,
        observation_actuator_names=observed_names,
    )

    signals = projected.read()
    assert signals.joint_pos_rad.shape == (17,)
    assert signals.joint_vel_rad_s.shape == (17,)
    projected.write_ctrl(home[[hardware_names.index(n) for n in spec.robot.actuator_names]])
    assert base.written[-1].shape == (17,)
