from __future__ import annotations

from assets.robot_config import load_robot_config
from training.configs.training_config import load_training_config
from training.policy_spec_utils import build_policy_spec_from_training_config
from wr_runtime.control.run_policy import _walking_runtime_plan


WRIST_NAMES = (
    "left_wrist_yaw",
    "left_wrist_pitch",
    "right_wrist_yaw",
    "right_wrist_pitch",
)


def test_native_17d_runtime_accepts_external_fixed_wrists_without_wrapper() -> None:
    cfg = load_training_config(
        "training/configs/ppo_walking_v0210_smoke6_home_rsi.yaml"
    )
    cfg.env.policy_excluded_actuator_names = WRIST_NAMES
    robot_cfg = load_robot_config("assets/v2/mujoco_robot_config.json")
    spec = build_policy_spec_from_training_config(
        training_cfg=cfg, robot_cfg=robot_cfg
    )

    hardware_names, home, mins, maxs, external_home = _walking_runtime_plan(
        spec, externally_managed_actuator_names=WRIST_NAMES
    )

    assert hardware_names == list(spec.robot.actuator_names)
    assert len(hardware_names) == 17
    assert home is not None and home.shape == (17,)
    assert mins.shape == maxs.shape == (17,)
    assert external_home == {}
