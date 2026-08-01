from __future__ import annotations

import json
from pathlib import Path

import mujoco
import pytest

from assets.post_process import remove_deprecated_wrist_dofs
from control.zmp.zmp_walk import ZMPWalkGenerator
from training.configs.training_config import _parse_env_config


_ROOT = Path(__file__).resolve().parents[1]


def _order_file_names(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def test_policy_robot_and_runtime_share_native_17d_order() -> None:
    policy_order = _order_file_names(
        _ROOT / "assets/v2/actuator_order.txt"
    )
    robot_config = json.loads(
        (_ROOT / "assets/v2/mujoco_robot_config.json").read_text()
    )
    hardware_config = json.loads(
        (_ROOT / "runtime/configs/hardware_config.json").read_text()
    )

    robot_names = [item["name"] for item in robot_config["actuated_joint_specs"]]
    servo_names = list(hardware_config["servo_controller"]["servos"])

    assert len(policy_order) == 17
    assert policy_order == robot_names
    assert set(servo_names) == set(policy_order)
    assert not any("wrist" in name for name in policy_order)
    assert "externally_managed_actuator_names" not in hardware_config


def test_mujoco_model_exactly_matches_native_policy_actuators() -> None:
    model = mujoco.MjModel.from_xml_path(
        str(_ROOT / "assets/v2/scene_flat_terrain.xml")
    )
    model_names = [str(model.actuator(index).name) for index in range(model.nu)]
    policy_names = _order_file_names(
        _ROOT / "assets/v2/actuator_order.txt"
    )

    assert (model.nq, model.nv, model.nu) == (24, 23, 17)
    assert policy_names == model_names
    assert not any("wrist" in name for name in model_names)
    assert model.key_qpos.shape == (2, 24)


def test_post_process_removes_wrist_dofs_but_keeps_hand_bodies(
    tmp_path: Path,
) -> None:
    xml_path = tmp_path / "model.xml"
    xml_path.write_text(
        "<mujoco><worldbody><body name='hand'>"
        "<joint name='left_wrist_yaw'/></body></worldbody>"
        "<actuator><position name='left_wrist_yaw' joint='left_wrist_yaw'/>"
        "</actuator></mujoco>"
    )

    remove_deprecated_wrist_dofs(xml_path)

    text = xml_path.read_text()
    assert "name=\"hand\"" in text
    assert "left_wrist_yaw" not in text


def test_training_config_rejects_legacy_exclusion_key() -> None:
    with pytest.raises(ValueError, match="was removed"):
        _parse_env_config({"env": {"policy_excluded_actuator_names": []}})


def test_distillation_generator_can_select_archived_21d_teacher_config() -> None:
    generator = ZMPWalkGenerator(
        robot_config_path=(
            _ROOT
            / "runtime/bundles/walking_v0210_smoke6_ckpt1650/"
            "mujoco_robot_config.json"
        )
    )

    assert generator._load_actuator_layout()["n_joints"] == 21
