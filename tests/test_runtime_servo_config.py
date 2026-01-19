import json
from pathlib import Path

import numpy as np
import pytest

from runtime.configs import WildRobotRuntimeConfig, ServoControllerConfig, ServoSpec
from runtime.wr_runtime.hardware.actuators import ServoModel, rad_to_servo_units, servo_units_to_rad


def _write_config(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    return path


def _base_config() -> dict:
    return {
        "mjcf_path": "model.xml",
        "policy_onnx_path": "policy.onnx",
    }


def test_canonical_servo_controller_parses(tmp_path: Path) -> None:
    cfg_dict = _base_config() | {
        "servo_controller": {
            "type": "hiwonder",
            "port": "/dev/ttyUSB0",
            "baudrate": 9600,
            "default_move_time_ms": 900,
            "servos": {
                "left_hip_pitch": {"id": 1, "offset_unit": 10, "direction": 1},
                "right_hip_pitch": {"id": 2, "offset_unit": -20, "direction": -1},
            },
        }
    }
    cfg = WildRobotRuntimeConfig.load(_write_config(tmp_path, cfg_dict))

    assert cfg.servo_controller.type == "hiwonder"
    assert cfg.servo_controller.port == "/dev/ttyUSB0"
    assert cfg.servo_controller.baudrate == 9600
    assert cfg.servo_controller.default_move_time_ms == 900
    assert cfg.servo_controller.servo_ids == {"left_hip_pitch": 1, "right_hip_pitch": 2}
    assert cfg.servo_controller.joint_offset_units["left_hip_pitch"] == 10
    assert cfg.servo_controller.joint_directions["right_hip_pitch"] == -1


def test_legacy_blocks_round_trip_to_canonical(tmp_path: Path) -> None:
    canonical = ServoControllerConfig(
        type="hiwonder",
        port="/dev/ttyUSB0",
        baudrate=9600,
        servos={
            "left": ServoSpec(id=1, offset_unit=0, direction=1),
            "right": ServoSpec(id=2, offset_unit=5, direction=-1),
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

    cfg = WildRobotRuntimeConfig.load(_write_config(tmp_path, legacy_hiwonder_cfg))
    assert cfg.servo_controller == canonical

    legacy_controller_cfg = _base_config() | {
        "Hiwonder_controller": {
            "port": canonical.port,
            "baudrate": canonical.baudrate,
            "servos": {
                "left": {"id": 1, "offset": 0, "direction": 1},
                "right": {"id": 2, "offset_rad": 5, "direction": -1},
            },
        }
    }

    cfg2 = WildRobotRuntimeConfig.load(_write_config(tmp_path, legacy_controller_cfg))
    assert cfg2.servo_controller == canonical


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
    directions = np.array([1.0], dtype=np.float32)
    target_rad = np.array([0.2], dtype=np.float32)

    units = rad_to_servo_units(target_rad, offsets, directions, servo_model)
    back = servo_units_to_rad(units, offsets, directions, servo_model)

    np.testing.assert_allclose(back, target_rad, atol=1e-6)


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
    assert cfg.servo_controller.joint_offset_units["left"] == 7
    out = cfg.to_dict()
    assert out["servo_controller"]["servos"]["left"]["offset_unit"] == 7
