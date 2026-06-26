import json
from pathlib import Path

import numpy as np
import pytest

from runtime.configs import WildRobotRuntimeConfig, ServoConfig, ServoControllerConfig, ServoSpec
from runtime.wr_runtime.hardware.actuators import (
    ServoModel,
    joint_target_rad_to_servo_pos_elec_units,
    servo_pos_elect_units_to_joint_target_rad,
)


def _write_config(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    return path


def _base_config() -> dict:
    return {
        "mjcf_path": "model.xml",
        "policy_onnx_path": "policy.onnx",
        "realism_profile_path": "assets/v2/realism_profile_v0.19.1.json",
    }


def test_canonical_servo_controller_parses(tmp_path: Path) -> None:
    cfg_dict = _base_config() | {
        "servo_controller": {
            "type": "hiwonder",
            "port": "/dev/ttyUSB0",
            "baudrate": 9600,
            "default_move_time_ms": 900,
            "servos": {
                "left_hip_pitch": {"id": 1, "servo_offset_unit": 10, "motor_unit_direction": 1, "joint_angle_at_zero_unit_deg": 0},
                "right_hip_pitch": {"id": 2, "servo_offset_unit": -20, "motor_unit_direction": -1, "joint_angle_at_zero_unit_deg": 90},
            },
        }
    }
    cfg = WildRobotRuntimeConfig.load(_write_config(tmp_path, cfg_dict))

    assert cfg.servo_controller.type == "hiwonder"
    assert cfg.servo_controller.port == "/dev/ttyUSB0"
    assert cfg.servo_controller.baudrate == 9600
    assert cfg.servo_controller.default_move_time_ms == 900
    assert cfg.servo_controller.servo_ids == {"left_hip_pitch": 1, "right_hip_pitch": 2}
    assert cfg.servo_controller.joint_servo_offset_units["left_hip_pitch"] == 10
    assert cfg.servo_controller.joint_motor_unit_directions["right_hip_pitch"] == -1.0
    assert cfg.servo_controller.joint_angle_at_zero_unit_deg["right_hip_pitch"] == 90.0
    assert cfg.realism_profile_path == "assets/v2/realism_profile_v0.19.1.json"


def test_legacy_servo_keys_still_load_and_serialize_to_new_names(tmp_path: Path) -> None:
    cfg_dict = _base_config() | {
        "servo_controller": {
            "servos": {
                "left": {"id": 1, "offset_unit": 7, "motor_sign": -1, "motor_center_mujoco_deg": 12},
            }
        }
    }
    cfg = WildRobotRuntimeConfig.load(_write_config(tmp_path, cfg_dict))
    servo = cfg.servo_controller.get_servo("left")
    assert servo.servo_offset_unit == 7
    assert servo.motor_unit_direction == -1.0
    assert servo.joint_angle_at_zero_unit_deg == 12.0
    out = cfg.to_dict()
    assert out["servo_controller"]["servos"]["left"] == {
        "id": 1,
        "servo_offset_unit": 7,
        "motor_unit_direction": -1.0,
        "joint_angle_at_zero_unit_deg": 12.0,
    }


def test_servo_unit_direction_alias_loads_and_serializes_to_canonical(tmp_path: Path) -> None:
    cfg_dict = _base_config() | {
        "servo_controller": {
            "servos": {
                "left": {
                    "id": 1,
                    "servo_offset_unit": 0,
                    "servo_unit_direction": -1,
                },
            }
        }
    }
    cfg = WildRobotRuntimeConfig.load(_write_config(tmp_path, cfg_dict))
    servo = cfg.servo_controller.get_servo("left")
    assert servo.servo_unit_direction == -1.0
    assert cfg.servo_controller.joint_servo_unit_directions["left"] == -1.0

    out = cfg.to_dict()
    assert "servo_unit_direction" not in out["servo_controller"]["servos"]["left"]
    assert out["servo_controller"]["servos"]["left"]["motor_unit_direction"] == -1.0


def test_conflicting_servo_unit_direction_alias_raises(tmp_path: Path) -> None:
    cfg_dict = _base_config() | {
        "servo_controller": {
            "servos": {
                "left": {
                    "id": 1,
                    "motor_unit_direction": 1,
                    "servo_unit_direction": -1,
                },
            }
        }
    }
    with pytest.raises(ValueError):
        WildRobotRuntimeConfig.load(_write_config(tmp_path, cfg_dict))


def test_left_handed_bno_axis_map_raises(tmp_path: Path) -> None:
    cfg_dict = _base_config() | {
        "servo_controller": {
            "servos": {
                "left_hip_pitch": {"id": 1},
            },
        },
        "bno085": {
            "axis_map": ["+X", "-Y", "+Z"],
        }
    }

    with pytest.raises(ValueError, match="right-handed"):
        WildRobotRuntimeConfig.load(_write_config(tmp_path, cfg_dict))


def test_legacy_blocks_round_trip_to_canonical(tmp_path: Path) -> None:
    canonical = ServoControllerConfig(
        type="hiwonder",
        port="/dev/ttyUSB0",
        baudrate=9600,
        servos={
            "left": ServoSpec(id=1, servo_offset_unit=0, motor_unit_direction=1),
            "right": ServoSpec(id=2, servo_offset_unit=5, motor_unit_direction=-1),
        },
    )

    legacy_hiwonder_cfg = _base_config() | {
        "hiwonder": {
            "port": canonical.port,
            "baudrate": canonical.baudrate,
            "servo_ids": {"left": 1, "right": 2},
            "joint_offsets_rad": {"right": 5},
            "joint_directions": {"right": -1},
        }
    }

    with pytest.raises(KeyError):
        WildRobotRuntimeConfig.load(_write_config(tmp_path, legacy_hiwonder_cfg))

    legacy_controller_cfg = _base_config() | {
        "Hiwonder_controller": {
            "port": canonical.port,
            "baudrate": canonical.baudrate,
            "servos": {
                "left": {"id": 1, "offset": 0, "motor_sign": 1},
                "right": {"id": 2, "offset_rad": 5, "motor_sign": -1},
            },
        }
    }

    with pytest.raises(KeyError):
        WildRobotRuntimeConfig.load(_write_config(tmp_path, legacy_controller_cfg))


def test_conflicting_servo_definitions_raise(tmp_path: Path) -> None:
    cfg_dict = _base_config() | {
        "servo_controller": {
            "port": "/dev/ttyUSB0",
            "servos": {"left": {"id": 1}},
        },
        "hiwonder": {
            "port": "/dev/ttyUSB1",
            "servo_ids": {"left": 1},
        },
    }

    with pytest.raises(ValueError) as excinfo:
        WildRobotRuntimeConfig.load(_write_config(tmp_path, cfg_dict))

    assert "Conflicting servo controller definitions" in str(excinfo.value)


def test_rad_units_round_trip_with_offset_unit() -> None:
    servo_model = ServoModel()
    offsets = np.array([10.0], dtype=np.float32)
    motor_signs = np.array([1.0], dtype=np.float32)
    centers_rad = np.array([0.0], dtype=np.float32)
    target_rad = np.array([0.2], dtype=np.float32)

    units = joint_target_rad_to_servo_pos_elec_units(target_rad, offsets, motor_signs, centers_rad, servo_model)
    back = servo_pos_elect_units_to_joint_target_rad(units, offsets, motor_signs, centers_rad, servo_model)

    np.testing.assert_allclose(back, target_rad, atol=1e-6)


def test_center_deg_maps_to_servo_center() -> None:
    servo_model = ServoModel()
    offsets = np.array([0.0], dtype=np.float32)
    motor_signs = np.array([1.0], dtype=np.float32)
    centers_rad = np.array([np.deg2rad(90.0)], dtype=np.float32)
    target_rad = np.array([np.pi / 2], dtype=np.float32)

    units = joint_target_rad_to_servo_pos_elec_units(target_rad, offsets, motor_signs, centers_rad, servo_model)
    assert np.allclose(units, np.array([servo_model.units_center], dtype=np.float32), atol=1e-3)

    back = servo_pos_elect_units_to_joint_target_rad(units, offsets, motor_signs, centers_rad, servo_model)
    np.testing.assert_allclose(back, target_rad, atol=1e-6)


def test_shoulder_center_deg_units_endpoints() -> None:
    servo_model = ServoModel()
    offsets = np.array([0.0], dtype=np.float32)
    motor_signs = np.array([1.0], dtype=np.float32)
    centers_rad = np.array([np.deg2rad(-90.0)], dtype=np.float32)

    target_max = np.array([np.deg2rad(30.0)], dtype=np.float32)
    target_min = np.array([np.deg2rad(-180.0)], dtype=np.float32)

    units_max = joint_target_rad_to_servo_pos_elec_units(target_max, offsets, motor_signs, centers_rad, servo_model)
    units_min = joint_target_rad_to_servo_pos_elec_units(target_min, offsets, motor_signs, centers_rad, servo_model)

    np.testing.assert_allclose(units_max, np.array([1000.0], dtype=np.float32), atol=1e-3)
    np.testing.assert_allclose(units_min, np.array([125.0], dtype=np.float32), atol=1e-3)


def test_offset_conflicts_raise(tmp_path: Path) -> None:
    cfg_dict = _base_config() | {
        "servo_controller": {
            "servos": {
                "left": {"id": 1, "offset_unit": 1, "offset": 2},
            }
        }
    }
    with pytest.raises(ValueError):
        WildRobotRuntimeConfig.load(_write_config(tmp_path, cfg_dict))


def test_offset_rad_migrates_to_unit_and_serializes(tmp_path: Path) -> None:
    cfg_dict = _base_config() | {
        "servo_controller": {
            "servos": {
                "left": {"id": 1, "offset_rad": 7},
            }
        }
    }
    cfg = WildRobotRuntimeConfig.load(_write_config(tmp_path, cfg_dict))
    assert cfg.servo_controller.joint_servo_offset_units["left"] == 7
    out = cfg.to_dict()
    assert out["servo_controller"]["servos"]["left"]["servo_offset_unit"] == 7
