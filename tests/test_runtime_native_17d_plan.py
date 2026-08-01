from __future__ import annotations

from assets.robot_config import load_robot_config
from training.configs.training_config import load_training_config
from training.policy_spec_utils import build_policy_spec_from_training_config
from wr_runtime.control.run_policy import _walking_runtime_plan


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
