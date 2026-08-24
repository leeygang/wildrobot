from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest

from assets.post_process import remove_deprecated_wrist_dofs
from control.zmp.zmp_walk import ZMPWalkGenerator
from training.configs.training_config import _parse_env_config
from training.envs.env_info import (
    PRIVILEGED_OBS_HISTORY_FRAMES,
    get_expected_shapes,
)
from training.scripts.distill_walking_21d_to_17d import (
    DEFAULT_TEACHER_POLICY_SPEC,
    DEFAULT_TEACHER_ROBOT_XML,
    _build_legacy_teacher_scene,
)


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
    joint_names = [str(model.joint(index).name) for index in range(model.njnt)]
    assert not any("wrist" in name for name in joint_names)
    assert np.count_nonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE) == 1
    assert model.key_qpos.shape == (2, 24)


def test_onshape_export_is_native_17d_with_one_root_body() -> None:
    root = ET.parse(_ROOT / "assets/v2/onshape_export/wildrobot.xml").getroot()
    worldbody = root.find("worldbody")
    actuator = root.find("actuator")

    assert worldbody is not None
    assert actuator is not None
    assert len(worldbody.findall("body")) == 1
    assert len(root.findall(".//freejoint")) == 1

    joint_names = [joint.get("name", "") for joint in root.findall(".//joint")]
    actuator_names = [item.get("name", "") for item in list(actuator)]
    assert len(actuator_names) == 17
    assert not any("wrist" in name for name in joint_names + actuator_names)


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


def test_archived_teacher_scene_does_not_depend_on_active_visual_meshes() -> None:
    scene_path = _build_legacy_teacher_scene(
        robot_xml_path=DEFAULT_TEACHER_ROBOT_XML,
        policy_spec_path=DEFAULT_TEACHER_POLICY_SPEC,
    )
    try:
        model = mujoco.MjModel.from_xml_path(str(scene_path))
    finally:
        scene_path.unlink(missing_ok=True)

    assert model.nu == 21
    assert not np.any(model.geom_type == mujoco.mjtGeom.mjGEOM_MESH)


def test_archived_teacher_critic_history_uses_21d_width() -> None:
    assert get_expected_shapes(action_size=21)["critic_obs_history"] == (
        PRIVILEGED_OBS_HISTORY_FRAMES,
        52,
    )
