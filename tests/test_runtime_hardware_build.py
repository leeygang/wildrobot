"""Hardware-mode startup wiring.

Regression for the review finding: ``_build_hardware_robot_io`` called
``Actuators.from_config`` / ``Imu.from_config`` / ``FootSwitches.from_config``
which do not exist, so real-robot execution raised ``AttributeError`` before the
first control tick.  These tests assert the builder wires the concrete hardware
classes with the right kwargs (mocked — no servos/IMU/GPIO required).
"""

from __future__ import annotations

import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import make_v8_spec

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TEMPLATE = _REPO_ROOT / "runtime" / "configs" / "runtime_config_template.json"
_V2_CONFIG = _REPO_ROOT / "runtime" / "configs" / "runtime_config_v2.json"
_CALIBRATE = _REPO_ROOT / "runtime" / "scripts" / "calibrate.py"


@pytest.mark.parametrize("config_path", [_TEMPLATE, _V2_CONFIG])
def test_runtime_config_covers_all_v8_actuators(config_path: Path) -> None:
    """Regression: the template (which export copies into every bundle) and the
    v2 sample must define a servo for every actuator in the v8 21-joint spec.

    Reproduces the review finding: the template omitted left/right_ankle_roll, so
    hardware startup raised KeyError "Servo ID missing for joint
    'left_ankle_roll'" before the first control tick.
    """
    servos = json.loads(config_path.read_text())["servo_controller"]["servos"]
    spec_names = set(make_v8_spec().robot.actuator_names)
    missing = sorted(spec_names - set(servos))
    assert not missing, f"{config_path.name} missing servo entries for {missing}"
    # ankle_roll specifically (the regressed joints) must be present.
    assert "left_ankle_roll" in servos
    assert "right_ankle_roll" in servos


def test_calibrate_default_config_path_is_repo_root_relative(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    ns = runpy.run_path(str(_CALIBRATE))
    config_path = ns["resolve_config_path"](SimpleNamespace(config=None, bundle=None))
    assert config_path == _V2_CONFIG


def test_hardware_protocols_have_no_from_config() -> None:
    """The protocols never had a ``from_config`` factory — pin it so the builder
    can't regress back to calling one."""
    from wr_runtime.hardware.actuators import Actuators
    from wr_runtime.hardware.imu import Imu

    assert not hasattr(Actuators, "from_config")
    assert not hasattr(Imu, "from_config")


def _fake_runtime_config() -> SimpleNamespace:
    servo_controller = SimpleNamespace(
        servo_ids={"j": 1},
        port="/dev/ttyUSB0",
        baudrate=9600,
        default_move_time_ms=None,
        joint_offset_units={"j": 3},
        joint_motor_signs={"j": -1.0},
        joint_motor_center_mujoco_deg={"j": 0.0},
    )
    bno085 = SimpleNamespace(
        i2c_address=0x4B,
        upside_down=False,
        axis_map=["+X", "-Y", "+Z"],
        suppress_debug=True,
        i2c_frequency_hz=100_000,
        init_retries=3,
    )
    foot_switches = SimpleNamespace(
        get_all_pins=lambda: {
            "left_toe": "D5",
            "left_heel": "D6",
            "right_toe": "D13",
            "right_heel": "D19",
        }
    )
    return SimpleNamespace(
        servo_controller=servo_controller,
        bno085=bno085,
        foot_switches=foot_switches,
    )


def test_build_hardware_robot_io_wires_concrete_classes(monkeypatch) -> None:
    import configs
    import wr_runtime.hardware.actuators as act_mod
    import wr_runtime.hardware.bno085 as bno_mod
    import wr_runtime.hardware.foot_switches as fs_mod
    from wr_runtime.control import run_policy

    captured = {}

    class _FakeActuators:
        def __init__(self, **kwargs):
            captured["actuators"] = kwargs

    class _FakeImu:
        def __init__(self, **kwargs):
            captured["imu"] = kwargs

    class _FakeFootSwitches:
        def __init__(self, **kwargs):
            captured["foot"] = kwargs

    monkeypatch.setattr(configs.WrRuntimeConfig, "load", staticmethod(
        lambda path: _fake_runtime_config()
    ))
    monkeypatch.setattr(act_mod, "HiwonderBoardActuators", _FakeActuators)
    monkeypatch.setattr(bno_mod, "BNO085IMU", _FakeImu)
    monkeypatch.setattr(fs_mod, "FootSwitches", _FakeFootSwitches)

    io = run_policy._build_hardware_robot_io(
        runtime_config_path="ignored",
        actuator_names=["j"],
        control_dt=0.02,
    )

    # HardwareRobotIO is a real dataclass; construction must not touch hardware.
    assert io.actuator_names == ["j"]
    assert io.control_dt == pytest.approx(0.02)

    a = captured["actuators"]
    assert a["servo_ids"] == {"j": 1}
    assert a["port"] == "/dev/ttyUSB0"
    assert a["baudrate"] == 9600
    # default_move_time_ms None -> one control period (20 ms at 50 Hz).
    assert a["default_move_time_ms"] == 20
    assert a["joint_offset_units"] == {"j": 3}

    assert captured["imu"]["i2c_address"] == 0x4B
    assert captured["imu"]["sampling_hz"] == 50  # round(1/0.02)
    assert captured["foot"]["pins"]["left_toe"] == "D5"


def test_build_hardware_robot_io_fails_fast_on_missing_servo(monkeypatch) -> None:
    """A config missing a spec actuator must raise an actionable error before
    touching hardware (not a bare KeyError deep in the actuator constructor)."""
    import configs
    from wr_runtime.control import run_policy

    cfg = _fake_runtime_config()  # servo_ids only has {'j': 1}
    monkeypatch.setattr(
        configs.WrRuntimeConfig, "load", staticmethod(lambda path: cfg)
    )
    with pytest.raises(SystemExit, match="left_ankle_roll"):
        run_policy._build_hardware_robot_io(
            runtime_config_path="ignored",
            actuator_names=["j", "left_ankle_roll"],
            control_dt=0.02,
        )
