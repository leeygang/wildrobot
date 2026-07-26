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
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import make_v8_spec
from wr_runtime.hardware.imu import ImuSample
from wr_runtime.hardware.robot_io import HardwareRobotIO

_REPO_ROOT = Path(__file__).resolve().parents[1]
_HARDWARE_CONFIG = _REPO_ROOT / "runtime" / "configs" / "hardware_config.json"
_DEPLOYED_HARDWARE_CONFIG = (
    _REPO_ROOT
    / "runtime"
    / "bundles"
    / "standing_walk_v0222"
    / "hardware_config.json"
)
_CALIBRATE = _REPO_ROOT / "runtime" / "scripts" / "calibrate.py"


def test_hardware_config_covers_all_v8_actuators() -> None:
    """The hardware template must cover every actuator exported to a bundle.

    Reproduces the review finding: an old template omitted left/right_ankle_roll, so
    hardware startup raised KeyError "Servo ID missing for joint
    'left_ankle_roll'" before the first control tick.
    """
    servos = json.loads(_HARDWARE_CONFIG.read_text())["servo_controller"]["servos"]
    spec_names = set(make_v8_spec().robot.actuator_names)
    missing = sorted(spec_names - set(servos))
    assert not missing, f"{_HARDWARE_CONFIG.name} missing servo entries for {missing}"
    # ankle_roll specifically (the regressed joints) must be present.
    assert "left_ankle_roll" in servos
    assert "right_ankle_roll" in servos


def test_hardware_config_template_matches_current_deployment_calibration() -> None:
    template = json.loads(_HARDWARE_CONFIG.read_text())
    deployed = json.loads(_DEPLOYED_HARDWARE_CONFIG.read_text())
    for payload in (template, deployed):
        payload.pop("realism_profile_path")
        payload.pop("robot_config_path")
    assert template == deployed

    raw_template = json.loads(_HARDWARE_CONFIG.read_text())
    for key in ("realism_profile_path", "robot_config_path"):
        assert (_HARDWARE_CONFIG.parent / raw_template[key]).resolve().is_file()


def test_calibrate_default_config_path_is_repo_root_relative(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    ns = runpy.run_path(str(_CALIBRATE))
    config_path = ns["resolve_config_path"](SimpleNamespace(config=None, bundle=None))
    assert config_path == _HARDWARE_CONFIG


def test_calibrate_go_home_skips_global_zero_prompt(monkeypatch) -> None:
    """With --go-home, calibration should preserve home pose instead of asking
    to immediately move every joint away to MuJoCo zero."""
    import builtins
    import runtime.scripts.calibrate as calibrate_mod

    class _FakeController:
        def move_servos(self, cmds, move_ms):
            pass

        def read_servo_positions(self, servo_ids):
            return [(int(servo_id), 500) for servo_id in servo_ids]

        def unload_servos(self, servo_ids):
            pass

        def close(self):
            pass

    prompts: list[str] = []
    responses = iter(["y", "q"])

    def _input(prompt: str = "") -> str:
        prompts.append(prompt)
        return next(responses)

    monkeypatch.setattr(calibrate_mod, "build_calibration_controller", lambda config: _FakeController())
    monkeypatch.setattr(builtins, "input", _input)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "calibrate.py",
            "--config",
            str(_HARDWARE_CONFIG),
            "--calibrate",
            "--go-home",
            "--pause-s",
            "0",
        ],
    )

    calibrate_mod.main()

    assert any("Move all servos to home pose now?" in prompt for prompt in prompts)
    assert not any("Move all joints to MuJoCo joint_pos_deg 0" in prompt for prompt in prompts)


def test_calibrate_home_is_top_level_command(monkeypatch, tmp_path) -> None:
    import builtins
    import runtime.scripts.calibrate as calibrate_mod

    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / "policy_spec.json").write_text(
        json.dumps(
            {
                "robot": {
                    "actuator_names": ["waist_yaw"],
                    "home_ctrl_rad": [0.0],
                }
            }
        )
    )

    class _FakeController:
        def __init__(self):
            self.moves = []
            self.unloaded = False

        def move_servos(self, cmds, move_ms):
            self.moves.append((list(cmds), int(move_ms)))

        def read_servo_positions(self, servo_ids):
            return [(int(servo_id), 500) for servo_id in servo_ids]

        def unload_servos(self, servo_ids):
            self.unloaded = True

        def close(self):
            pass

    controller = _FakeController()
    prompts: list[str] = []
    responses = iter(["q"])

    def _input(prompt: str = "") -> str:
        prompts.append(prompt)
        return next(responses)

    monkeypatch.setattr(calibrate_mod, "build_calibration_controller", lambda config: controller)
    monkeypatch.setattr(builtins, "input", _input)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "calibrate.py",
            "--config",
            str(_HARDWARE_CONFIG),
            "--bundle",
            str(bundle_dir),
            "--calibrate-home",
            "--pause-s",
            "0",
        ],
    )

    calibrate_mod.main()

    assert controller.moves
    assert controller.unloaded
    assert any("Select servo # to adjust home pose" in prompt for prompt in prompts)
    assert not any("Select servo #" in prompt and "save+quit" in prompt for prompt in prompts)


def test_calibrate_zero_pose_commands_can_ignore_or_apply_offset() -> None:
    import runtime.scripts.calibrate as calibrate_mod
    from configs.config import WrRuntimeConfig

    cfg = WrRuntimeConfig.load(_HARDWARE_CONFIG)
    joint = next(
        name
        for name, servo in cfg.hiwonder_controller.servos.items()
        if int(servo.offset) != 0
    )
    servo = cfg.hiwonder_controller.servos[joint]
    states = {
        joint: calibrate_mod.normalize_joint_state(
            offset=servo.offset,
            motor_sign=servo.motor_sign,
        )
    }

    absolute = calibrate_mod.build_zero_pose_commands(
        joint_names=[joint],
        servo_cfgs=cfg.hiwonder_controller.servos,
        states=states,
        use_offset=False,
    )
    with_offset = calibrate_mod.build_zero_pose_commands(
        joint_names=[joint],
        servo_cfgs=cfg.hiwonder_controller.servos,
        states=states,
        use_offset=True,
    )

    assert absolute == [
        (
            servo.id,
            servo.joint_target_rad_to_elect_unit_for_calibrate(
                0.0,
                motor_sign=states[joint].motor_sign,
                offset=0,
            ),
        )
    ]
    assert with_offset == [
        (
            servo.id,
            servo.joint_target_rad_to_elect_unit_for_calibrate(
                0.0,
                motor_sign=states[joint].motor_sign,
                offset=states[joint].offset,
            ),
        )
    ]
    assert absolute != with_offset


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("a", "a"), ("o", "o"), ("n", "n"), ("", "o"), ("bad", "o")],
)
def test_calibrate_zero_centering_prompt(monkeypatch, raw: str, expected: str) -> None:
    import runtime.scripts.calibrate as calibrate_mod

    monkeypatch.setattr("builtins.input", lambda _prompt="": raw)

    assert calibrate_mod.prompt_zero_centering_mode(default="o") == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("z", "z"), ("r", "r"), ("q", "q"), ("", "z"), ("bad", "z")],
)
def test_calibrate_offset_reference_prompt(monkeypatch, raw: str, expected: str) -> None:
    import runtime.scripts.calibrate as calibrate_mod
    from configs.config import ServoConfig

    monkeypatch.setattr("builtins.input", lambda _prompt="": raw)

    assert calibrate_mod.prompt_offset_reference_mode(servo=ServoConfig(id=1), default="z") == expected


def test_calibrate_offset_from_zero_degree_reference_handles_shoulder_center() -> None:
    import runtime.scripts.calibrate as calibrate_mod
    from configs.config import ServoConfig

    servo = ServoConfig(
        id=21,
        motor_unit_direction=-1.0,
        joint_angle_at_zero_unit_deg=90.0,
    )
    offset = 14
    zero_deg_units = servo.joint_target_rad_to_elect_unit_for_calibrate(
        0.0,
        motor_sign=-1,
        offset=offset,
    )
    raw_center_units = int(ServoConfig.UNITS_CENTER) + offset

    assert zero_deg_units != raw_center_units
    assert (
        calibrate_mod.offset_from_reference_pose_units(
            servo,
            zero_deg_units,
            motor_sign=-1,
            target_rad=0.0,
        )
        == offset
    )
    assert (
        calibrate_mod.offset_from_reference_pose_units(
            servo,
            raw_center_units,
            motor_sign=-1,
            target_rad=servo.center_rad,
        )
        == offset
    )


def test_write_bundle_home_ctrl_rad_updates_policy_spec(tmp_path) -> None:
    import runtime.scripts.calibrate as calibrate_mod

    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    spec_path = bundle_dir / "policy_spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "robot": {
                    "actuator_names": ["j1", "j2"],
                    "home_ctrl_rad": [0.1, 0.2],
                }
            }
        )
    )

    written_path = calibrate_mod.write_bundle_home_ctrl_rad(bundle_dir, [0.1, 0.35])

    assert written_path == spec_path
    data = json.loads(spec_path.read_text())
    assert data["robot"]["home_ctrl_rad"] == [0.1, 0.35]


def test_home_pose_units_uses_current_servo_calibration() -> None:
    import runtime.scripts.calibrate as calibrate_mod
    from configs.config import ServoConfig

    servo = ServoConfig(
        id=32,
        servo_offset_unit=-38,
        motor_unit_direction=-1.0,
        joint_angle_at_zero_unit_deg=85.0,
    )
    state = calibrate_mod.JointState(offset=-38, motor_sign=-1)

    assert calibrate_mod._home_pose_units(servo, state, 0.0) == servo.joint_target_rad_to_elect_unit_for_calibrate(
        0.0,
        motor_sign=-1,
        offset=-38,
    )


def test_home_pose_status_line_prints_current_requested_and_config() -> None:
    import runtime.scripts.calibrate as calibrate_mod
    from configs.config import ServoConfig

    servo = ServoConfig(id=1, motor_unit_direction=1.0, joint_angle_at_zero_unit_deg=0.0)
    state = calibrate_mod.JointState(offset=10, motor_sign=1)
    requested_rad = float(np.deg2rad(5.0))
    home_rad = float(np.deg2rad(3.0))
    readback_units = servo.joint_target_rad_to_elect_unit_for_calibrate(
        float(np.deg2rad(4.0)),
        motor_sign=1,
        offset=10,
    )

    line = calibrate_mod._format_home_pose_line(
        joint="j",
        servo=servo,
        state=state,
        requested_rad=requested_rad,
        home_rad=home_rad,
        readback_units=readback_units,
    )

    assert "current= +4.08deg" in line
    assert "requested= +5.00deg" in line
    assert "requested_raw=" in line
    assert "home= +3.00deg" in line
    assert "home_raw=" in line


def test_calibrate_home_imu_status_prints_body_angle() -> None:
    import runtime.scripts.calibrate as calibrate_mod

    sample = SimpleNamespace(
        valid=True,
        fresh=True,
        timestamp_s=12.5,
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        gyro_rad_s=np.array([0.0, 0.0, 0.0], dtype=np.float32),
    )

    line = calibrate_mod._format_calibrate_home_imu_status(sample)

    assert "IMU body_angle: valid=True fresh=True" in line
    assert "rpy_deg=[+0.0, +0.0, +0.0]" in line
    assert "tilt_deg=0.0" in line
    assert "gyro_norm_rad_s=0.000" in line


def test_ttl_calibration_controller_uses_per_servo_protocol() -> None:
    from wr_runtime.hardware.ttl_servo_controller import TtlServoController

    class _FakeTransport:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class _FakeRawBus:
        def __init__(self):
            self.transport = _FakeTransport()
            self.moves = []
            self.unloads = []

        def move_time_write(self, servo_id, position, time_ms):
            self.moves.append((int(servo_id), int(position), int(time_ms)))

        def read_positions(self, servo_ids):
            return {int(servo_id): 500 + int(servo_id) for servo_id in servo_ids}

        def unload(self, servo_id):
            self.unloads.append(int(servo_id))

    raw_bus = _FakeRawBus()
    controller = TtlServoController(raw_bus)

    assert controller.move_servos([(1, 410), (8, 582)], 1500)
    assert raw_bus.moves == [(1, 410, 1500), (8, 582, 1500)]
    assert controller.read_servo_positions([1, 8]) == [(1, 501), (8, 508)]
    assert controller.unload_servos([1, 8])
    assert raw_bus.unloads == [1, 8]
    controller.close()
    assert raw_bus.transport.closed


def test_ttl_servo_controller_rejects_deprecated_board_type() -> None:
    from wr_runtime.hardware.ttl_servo_controller import build_ttl_servo_controller

    cfg = SimpleNamespace(
        type="hiwonder",
        port="/dev/ttyUSB0",
        baudrate=9600,
    )

    with pytest.raises(ValueError, match="hiwonder_ttl_bus"):
        build_ttl_servo_controller(cfg)


def test_multi_board_calibration_controller_routes_by_servo_id(monkeypatch) -> None:
    import wr_runtime.hardware.ttl_servo_controller as controller_mod

    class _FakeController:
        def __init__(self, port):
            self.port = port
            self.moves = []
            self.closed = False

        def move_servos(self, commands, time_ms):
            self.moves.append((list(commands), int(time_ms)))
            return True

        def read_servo_positions(self, servo_ids):
            return [(int(servo_id), 500 + int(servo_id)) for servo_id in servo_ids]

        def unload_servos(self, servo_ids):
            return True

        def get_battery_voltage(self):
            return None

        def close(self):
            self.closed = True

    controllers = {}

    def _build(*, port, baudrate):
        controllers[port] = _FakeController(port)
        return controllers[port]

    monkeypatch.setattr(controller_mod, "_build_single_ttl_servo_controller", _build)
    cfg = SimpleNamespace(
        type="hiwonder_ttl_bus",
        baudrate=115200,
        boards=(
            SimpleNamespace(
                name="left_leg_board", port="/dev/ttyUSB0", servo_ids=(1,)
            ),
            SimpleNamespace(
                name="right_leg_board", port="/dev/ttyUSB1", servo_ids=(2,)
            ),
        ),
    )
    controller = controller_mod.build_ttl_servo_controller(cfg)
    try:
        assert controller.move_servos([(1, 510), (2, 520)], 20)
        assert controller.read_servo_positions([2, 1]) == [(2, 502), (1, 501)]
    finally:
        controller.close()

    assert controllers["/dev/ttyUSB0"].moves == [([(1, 510)], 20)]
    assert controllers["/dev/ttyUSB1"].moves == [([(2, 520)], 20)]
    assert all(item.closed for item in controllers.values())


def test_calibrate_servo_board_discovers_complete_mapping_and_writes_config(
    tmp_path,
) -> None:
    from runtime.scripts import calibrate

    positions_by_port = {
        "/dev/ttyUSB0": {1: 501, 2: 502},
        "/dev/ttyUSB1": {5: 505},
        "/dev/ttyUSB2": {21: 521},
        "/dev/ttyUSB3": {},
    }

    class _FakeController:
        def __init__(self, port):
            self.port = port

        def read_servo_positions(self, servo_ids):
            positions = positions_by_port[self.port]
            return [
                (int(servo_id), positions[int(servo_id)])
                for servo_id in servo_ids
                if int(servo_id) in positions
            ] or None

        def close(self):
            pass

    boards = calibrate.discover_servo_boards(
        servo_ids_by_joint={
            "left_hip_pitch": 1,
            "left_hip_roll": 2,
            "right_hip_pitch": 5,
            "left_shoulder_pitch": 21,
        },
        controller_type="hiwonder_ttl_bus",
        baudrate=115200,
        ports=positions_by_port,
        controller_factory=lambda cfg: _FakeController(cfg.port),
    )

    assert [(board.name, board.port, board.servo_ids) for board in boards] == [
        ("left_leg_board", "/dev/ttyUSB0", (1, 2)),
        ("right_leg_board", "/dev/ttyUSB1", (5,)),
        ("upper_body_board", "/dev/ttyUSB2", (21,)),
    ]
    output = tmp_path / "config.json"
    raw = {"servo_controller": {"port": "/dev/ttyUSB-old", "servos": {}}}
    calibrate.write_servo_board_config(raw, output, boards)
    saved = json.loads(output.read_text())
    assert "port" not in saved["servo_controller"]
    assert saved["servo_controller"]["boards"] == [
        {
            "name": "left_leg_board",
            "port": "/dev/ttyUSB0",
            "servo_ids": [1, 2],
        },
        {
            "name": "right_leg_board",
            "port": "/dev/ttyUSB1",
            "servo_ids": [5],
        },
        {
            "name": "upper_body_board",
            "port": "/dev/ttyUSB2",
            "servo_ids": [21],
        },
    ]


def test_calibrate_servo_board_detects_only_usb_serial_ports(monkeypatch) -> None:
    from serial.tools import list_ports
    from runtime.scripts import calibrate

    ports = [
        SimpleNamespace(device="/dev/ttyS0", vid=None, hwid="PNP0501"),
        SimpleNamespace(device="/dev/ttyUSB1", vid=0x1A86, hwid="USB VID:PID=1A86:7523"),
        SimpleNamespace(device="/dev/ttyACM0", vid=None, hwid="USB CDC"),
    ]
    monkeypatch.setattr(list_ports, "comports", lambda: ports)
    monkeypatch.setattr(calibrate, "_stable_serial_port", lambda device: device)

    assert calibrate.detect_usb_serial_ports() == ["/dev/ttyACM0", "/dev/ttyUSB1"]


def test_calibrate_servo_board_rejects_duplicate_servo_response() -> None:
    from runtime.scripts import calibrate

    class _FakeController:
        def __init__(self, port):
            self.port = port

        def read_servo_positions(self, servo_ids):
            return [(int(servo_ids[0]), 500)]

        def close(self):
            pass

    with pytest.raises(RuntimeError, match="duplicate"):
        calibrate.discover_servo_boards(
            servo_ids_by_joint={"left_hip_pitch": 1},
            controller_type="hiwonder_ttl_bus",
            baudrate=115200,
            ports=["/dev/ttyUSB0", "/dev/ttyUSB1"],
            controller_factory=lambda cfg: _FakeController(cfg.port),
        )


def test_calibrate_servo_board_rejects_split_leg_group() -> None:
    from runtime.scripts import calibrate

    positions_by_port = {
        "/dev/ttyUSB0": {1: 501},
        "/dev/ttyUSB1": {2: 502},
    }

    class _FakeController:
        def __init__(self, port):
            self.port = port

        def read_servo_positions(self, servo_ids):
            positions = positions_by_port[self.port]
            return [
                (int(servo_id), positions[int(servo_id)])
                for servo_id in servo_ids
                if int(servo_id) in positions
            ] or None

        def close(self):
            pass

    with pytest.raises(RuntimeError, match="physical joint group"):
        calibrate.discover_servo_boards(
            servo_ids_by_joint={"left_hip_pitch": 1, "left_hip_roll": 2},
            controller_type="hiwonder_ttl_bus",
            baudrate=115200,
            ports=positions_by_port,
            controller_factory=lambda cfg: _FakeController(cfg.port),
        )


def test_calibrate_servo_board_is_top_level_cli_mode(monkeypatch, tmp_path) -> None:
    from runtime.scripts import calibrate

    config_path = tmp_path / "config.json"
    config_path.write_text(_HARDWARE_CONFIG.read_text())
    discovered = [
        calibrate.DetectedServoBoard(
            "left_leg_board", "/dev/ttyUSB0", (1, 2, 3, 4, 9)
        ),
        calibrate.DetectedServoBoard(
            "right_leg_board", "/dev/ttyUSB1", (5, 6, 7, 8, 10)
        ),
        calibrate.DetectedServoBoard(
            "upper_body_board",
            "/dev/ttyUSB2",
            (40, 21, 22, 23, 24, 25, 31, 32, 33, 34, 35),
        ),
    ]
    monkeypatch.setattr(calibrate, "detect_usb_serial_ports", lambda: [b.port for b in discovered])
    monkeypatch.setattr(calibrate, "discover_servo_boards", lambda **kwargs: discovered)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "calibrate.py",
            "--config",
            str(config_path),
            "--calibrate-servo-board",
        ],
    )

    calibrate.main()

    saved = json.loads(config_path.read_text())["servo_controller"]
    assert "port" not in saved
    assert len(saved["boards"]) == 3
    assert saved["boards"][2]["servo_ids"] == list(discovered[2].servo_ids)


def test_hardware_protocols_have_no_from_config() -> None:
    """The protocols never had a ``from_config`` factory — pin it so the builder
    can't regress back to calling one."""
    from wr_runtime.hardware.actuators import Actuators
    from wr_runtime.hardware.imu import Imu

    assert not hasattr(Actuators, "from_config")
    assert not hasattr(Imu, "from_config")


def _fake_runtime_config() -> SimpleNamespace:
    servo_controller = SimpleNamespace(
        type="hiwonder_ttl_bus",
        servo_ids={"j": 1},
        port="/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0",
        baudrate=115200,
        default_move_time_ms=None,
        joint_servo_offset_units={"j": 3},
        joint_motor_unit_directions={"j": -1.0},
        joint_angle_at_zero_unit_deg={"j": 0.0},
    )
    bno085 = SimpleNamespace(
        transport="spi",
        i2c_address=0x4B,
        upside_down=False,
        axis_map=["+X", "-Y", "-Z"],
        suppress_debug=True,
        i2c_frequency_hz=100_000,
        spi_baudrate=1_000_000,
        spi_read_skip_bytes=2,
        spi_cs_pin="D8",
        spi_int_pin="D17",
        spi_reset_pin="D27",
        spi_wake_pin="D25",
        init_retries=3,
        sampling_hz=None,
        enable_rotation_vector=True,
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


def _patch_ttl_servo_backend(monkeypatch, captured):
    import wr_runtime.hardware.actuators as act_mod
    import wr_runtime.hardware.hiwonder_ttl_bus as ttl_mod
    import wr_runtime.hardware.servo_io_worker as worker_mod

    class _FakeTransport:
        def __init__(self, config):
            captured["transport_config"] = config
            captured.setdefault("transport_configs", []).append(config)
            self.port = config.port
            self.baudrate = config.baudrate

    class _FakeRawBus:
        def __init__(self, transport, config):
            captured["raw_bus_config"] = config
            self.transport = transport

    class _FakeServoIOWorker:
        def __init__(self, raw_bus, config, *, worker_name="servo_board"):
            captured["worker_config"] = config
            captured.setdefault("worker_configs", []).append(config)
            captured.setdefault("workers", []).append(self)
            self.raw_bus = raw_bus
            self.config = config
            self.servo_ids = tuple(config.servo_ids)
            self.worker_name = worker_name
            self.started = False

        def start(self):
            self.started = True
            captured["worker_started"] = True

        def stop(self, *, timeout_s=1.0):
            self.started = False

        def close(self):
            self.started = False

    class _FakeActuators:
        def __init__(self, **kwargs):
            captured["actuators"] = kwargs

    monkeypatch.setattr(ttl_mod, "SerialTransport", _FakeTransport)
    monkeypatch.setattr(ttl_mod, "RawServoBus", _FakeRawBus)
    monkeypatch.setattr(worker_mod, "ServoIOWorker", _FakeServoIOWorker)
    monkeypatch.setattr(act_mod, "HiwonderCachedActuators", _FakeActuators)


def test_build_hardware_robot_io_wires_concrete_classes(monkeypatch) -> None:
    import configs
    import wr_runtime.hardware.bno085 as bno_mod
    import wr_runtime.hardware.foot_switches as fs_mod
    from wr_runtime.control import run_policy

    captured = {}
    _patch_ttl_servo_backend(monkeypatch, captured)

    class _FakeImu:
        def __init__(self, **kwargs):
            captured["imu"] = kwargs

    class _FakeFootSwitches:
        def __init__(self, **kwargs):
            captured["foot"] = kwargs

    monkeypatch.setattr(configs.WrRuntimeConfig, "load", staticmethod(
        lambda path: _fake_runtime_config()
    ))
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
    assert captured["transport_config"].port == "/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0"
    assert captured["transport_config"].baudrate == 115200
    # default_move_time_ms None -> one control period (20 ms at 50 Hz).
    assert a["default_move_time_ms"] == 20
    assert a["joint_servo_offset_units"] == {"j": 3}
    assert a["port"] == "/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0"
    assert a["baudrate"] == 115200
    assert captured["worker_started"] is True

    assert captured["imu"]["transport"] == "spi"
    assert captured["imu"]["i2c_address"] == 0x4B
    assert captured["imu"]["spi_baudrate"] == 1_000_000
    assert captured["imu"]["spi_read_skip_bytes"] == 2
    assert captured["imu"]["spi_cs_pin"] == "D8"
    assert captured["imu"]["spi_int_pin"] == "D17"
    assert captured["imu"]["spi_reset_pin"] == "D27"
    assert captured["imu"]["spi_wake_pin"] == "D25"
    assert captured["imu"]["sampling_hz"] == 50  # round(1/0.02)
    assert captured["imu"]["enable_rotation_vector"] is True
    assert captured["foot"]["pins"]["left_toe"] == "D5"


def test_build_hardware_robot_io_uses_bno_sampling_override(monkeypatch) -> None:
    import configs
    import wr_runtime.hardware.bno085 as bno_mod
    import wr_runtime.hardware.foot_switches as fs_mod
    from wr_runtime.control import run_policy

    captured = {}
    _patch_ttl_servo_backend(monkeypatch, captured)
    cfg = _fake_runtime_config()
    cfg.bno085.sampling_hz = 20
    cfg.bno085.enable_rotation_vector = False

    class _FakeImu:
        def __init__(self, **kwargs):
            captured["imu"] = kwargs

    class _FakeFootSwitches:
        def __init__(self, **kwargs):
            pass

    monkeypatch.setattr(configs.WrRuntimeConfig, "load", staticmethod(lambda path: cfg))
    monkeypatch.setattr(bno_mod, "BNO085IMU", _FakeImu)
    monkeypatch.setattr(fs_mod, "FootSwitches", _FakeFootSwitches)

    run_policy._build_hardware_robot_io(
        runtime_config_path="ignored",
        actuator_names=["j"],
        control_dt=0.02,
    )

    assert captured["imu"]["sampling_hz"] == 20
    assert captured["imu"]["enable_rotation_vector"] is False


def test_build_hardware_robot_io_creates_one_parallel_worker_per_board(
    monkeypatch,
) -> None:
    import configs
    import wr_runtime.hardware.bno085 as bno_mod
    import wr_runtime.hardware.foot_switches as fs_mod
    from wr_runtime.control import run_policy
    from wr_runtime.hardware.servo_io_worker import MultiBoardServoIO

    captured = {}
    _patch_ttl_servo_backend(monkeypatch, captured)
    cfg = _fake_runtime_config()
    cfg.servo_controller.servo_ids = {"j1": 1, "j2": 2, "j3": 3}
    cfg.servo_controller.joint_servo_offset_units = {"j1": 0, "j2": 0, "j3": 0}
    cfg.servo_controller.joint_motor_unit_directions = {
        "j1": 1.0,
        "j2": 1.0,
        "j3": 1.0,
    }
    cfg.servo_controller.joint_angle_at_zero_unit_deg = {
        "j1": 0.0,
        "j2": 0.0,
        "j3": 0.0,
    }
    cfg.servo_controller.effective_boards = (
        SimpleNamespace(
            name="left_leg_board", port="/dev/ttyUSB0", servo_ids=(1,)
        ),
        SimpleNamespace(
            name="right_leg_board", port="/dev/ttyUSB1", servo_ids=(2,)
        ),
        SimpleNamespace(
            name="upper_body_board", port="/dev/ttyUSB2", servo_ids=(3,)
        ),
    )

    class _FakeImu:
        def __init__(self, **kwargs):
            pass

    class _FakeFootSwitches:
        def __init__(self, **kwargs):
            pass

    monkeypatch.setattr(configs.WrRuntimeConfig, "load", staticmethod(lambda path: cfg))
    monkeypatch.setattr(bno_mod, "BNO085IMU", _FakeImu)
    monkeypatch.setattr(fs_mod, "FootSwitches", _FakeFootSwitches)

    run_policy._build_hardware_robot_io(
        runtime_config_path="ignored",
        actuator_names=["j1", "j2", "j3"],
        control_dt=0.02,
    )

    assert [item.port for item in captured["transport_configs"]] == [
        "/dev/ttyUSB0",
        "/dev/ttyUSB1",
        "/dev/ttyUSB2",
    ]
    assert all(worker.started for worker in captured["workers"])
    assert [worker.worker_name for worker in captured["workers"]] == [
        "left_leg_board",
        "right_leg_board",
        "upper_body_board",
    ]
    assert isinstance(captured["actuators"]["servo_io"], MultiBoardServoIO)
    assert captured["actuators"]["port"] == (
        "left_leg_board=/dev/ttyUSB0,right_leg_board=/dev/ttyUSB1,"
        "upper_body_board=/dev/ttyUSB2"
    )


def test_build_hardware_robot_io_builds_one_read_set_for_board_actuators(
    monkeypatch,
) -> None:
    import configs
    import wr_runtime.hardware.bno085 as bno_mod
    import wr_runtime.hardware.foot_switches as fs_mod
    from wr_runtime.control import run_policy

    captured = {}
    _patch_ttl_servo_backend(monkeypatch, captured)
    cfg = _fake_runtime_config()
    cfg.servo_read_schedule = SimpleNamespace(
        max_cache_age_s={"default": 0.25},
    )

    class _FakeImu:
        def __init__(self, **kwargs):
            pass

    class _FakeFootSwitches:
        def __init__(self, **kwargs):
            pass

    monkeypatch.setattr(configs.WrRuntimeConfig, "load", staticmethod(lambda path: cfg))
    monkeypatch.setattr(bno_mod, "BNO085IMU", _FakeImu)
    monkeypatch.setattr(fs_mod, "FootSwitches", _FakeFootSwitches)

    run_policy._build_hardware_robot_io(
        runtime_config_path="ignored",
        actuator_names=["j"],
        control_dt=0.02,
    )

    worker_config = captured["worker_config"]
    assert tuple(worker_config.read_group_schedule) == ("group_0",)
    assert len(worker_config.read_groups) == 1
    assert worker_config.read_groups[0].servo_ids == (1,)
    assert captured["actuators"]["cache_age_limits_s"] == {"default": 0.25}


def test_build_hardware_robot_io_rejects_deprecated_hiwonder_board(monkeypatch) -> None:
    import configs
    from wr_runtime.control import run_policy

    cfg = _fake_runtime_config()
    cfg.servo_controller.type = "hiwonder"
    monkeypatch.setattr(configs.WrRuntimeConfig, "load", staticmethod(lambda path: cfg))

    with pytest.raises(SystemExit, match="deprecated LSC controller-board"):
        run_policy._build_hardware_robot_io(
            runtime_config_path="ignored",
            actuator_names=["j"],
            control_dt=0.02,
        )


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


def test_hardware_robot_io_waits_for_first_valid_imu_sample() -> None:
    valid_sample = ImuSample(
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        gyro_rad_s=np.zeros(3, dtype=np.float32),
        timestamp_s=1.0,
        valid=True,
    )
    invalid_sample = ImuSample(
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        gyro_rad_s=np.zeros(3, dtype=np.float32),
        timestamp_s=0.0,
        valid=False,
    )

    class _FakeImu:
        def __init__(self) -> None:
            self.samples = [
                invalid_sample,
                invalid_sample,
                valid_sample,
                valid_sample,
                valid_sample,
                valid_sample,
                valid_sample,
                valid_sample,
            ]

        def read(self):
            return self.samples.pop(0)

    robot_io = HardwareRobotIO(
        actuator_names=["j"],
        control_dt=0.02,
        actuators=SimpleNamespace(),
        imu=_FakeImu(),
        foot_switches=SimpleNamespace(),
    )

    robot_io.wait_for_valid_imu_sample(timeout_s=1.0, poll_s=0.0)

    assert robot_io._last_fresh_imu_sample is valid_sample
    assert robot_io._imu_nonfresh_consecutive == 0
    assert robot_io._last_fresh_imu_wall_time_s is not None


def test_hardware_robot_io_wait_rejects_startup_gyro_integrated_imu_sample() -> None:
    integrated_sample = ImuSample(
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        gyro_rad_s=np.zeros(3, dtype=np.float32),
        timestamp_s=1.0,
        valid=True,
        fresh=True,
    )
    direct_sample = ImuSample(
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        gyro_rad_s=np.zeros(3, dtype=np.float32),
        timestamp_s=2.0,
        valid=True,
        fresh=True,
    )

    class _FakeImu:
        def __init__(self) -> None:
            self.samples = [
                ("integrated_from_gyro_after_missing", integrated_sample),
                ("normalized", direct_sample),
                ("normalized", direct_sample),
                ("normalized", direct_sample),
                ("normalized", direct_sample),
                ("normalized", direct_sample),
                ("normalized", direct_sample),
            ]
            self.diag = {}

        def read(self):
            quat_status, sample = self.samples.pop(0)
            self.diag = {"quat_status": quat_status, "gyro_status": "raw"}
            return sample

    robot_io = HardwareRobotIO(
        actuator_names=["j"],
        control_dt=0.02,
        actuators=SimpleNamespace(),
        imu=_FakeImu(),
        foot_switches=SimpleNamespace(),
    )

    robot_io.wait_for_valid_imu_sample(timeout_s=1.0, poll_s=0.0)

    assert robot_io._last_fresh_imu_sample is direct_sample


def test_hardware_robot_io_wait_counts_direct_samples_across_cached_reads() -> None:
    direct_sample = ImuSample(
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        gyro_rad_s=np.zeros(3, dtype=np.float32),
        timestamp_s=1.0,
        valid=True,
        fresh=True,
    )
    cached_sample = ImuSample(
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        gyro_rad_s=np.zeros(3, dtype=np.float32),
        timestamp_s=1.0,
        valid=True,
        fresh=False,
    )

    class _FakeImu:
        def __init__(self) -> None:
            self.samples = [
                direct_sample,
                cached_sample,
                cached_sample,
                direct_sample,
                cached_sample,
                direct_sample,
                cached_sample,
                direct_sample,
                cached_sample,
                direct_sample,
                cached_sample,
                direct_sample,
            ]
            self.diag = {"quat_status": "normalized", "gyro_status": "raw"}

        def read(self):
            return self.samples.pop(0)

    robot_io = HardwareRobotIO(
        actuator_names=["j"],
        control_dt=0.02,
        actuators=SimpleNamespace(),
        imu=_FakeImu(),
        foot_switches=SimpleNamespace(),
    )

    robot_io.wait_for_valid_imu_sample(timeout_s=1.0, poll_s=0.0)

    assert robot_io._last_fresh_imu_sample is direct_sample


def test_hardware_robot_io_reuses_recent_cached_imu_sample(monkeypatch) -> None:
    import wr_runtime.hardware.robot_io as robot_io_mod

    valid_sample = ImuSample(
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        gyro_rad_s=np.zeros(3, dtype=np.float32),
        timestamp_s=1.0,
        valid=True,
    )
    stale_sample = ImuSample(
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        gyro_rad_s=np.zeros(3, dtype=np.float32),
        timestamp_s=1.0,
        valid=True,
        fresh=False,
    )
    now = [0.0]
    monkeypatch.setattr(robot_io_mod.time, "monotonic", lambda: now[0])

    class _FakeImu:
        def __init__(self) -> None:
            self.first = True
            self.error_count = 0
            self.last_error = None

        def read(self):
            if self.first:
                self.first = False
                return valid_sample
            return stale_sample

    robot_io = HardwareRobotIO(
        actuator_names=["j"],
        control_dt=0.02,
        actuators=SimpleNamespace(
            get_positions_rad=lambda: np.array([0.0], dtype=np.float32),
            estimate_velocities_rad_s=lambda dt: np.array([0.0], dtype=np.float32),
            set_targets_rad=lambda targets, move_time_ms=None: None,
        ),
        imu=_FakeImu(),
        foot_switches=SimpleNamespace(
            read=lambda: SimpleNamespace(switches=[True, True, True, True])
        ),
        max_cached_imu_age_s=0.25,
    )

    robot_io.read()
    assert robot_io.last_timing_s["imu_read"] >= 0.0
    assert robot_io.last_timing_s["actuator_read"] >= 0.0
    assert robot_io.last_timing_s["footswitch_read"] >= 0.0
    for t in (0.02, 0.04, 0.06, 0.08, 0.10, 0.12):
        now[0] = t
        signals = robot_io.read()
        assert np.allclose(signals.quat_xyzw, valid_sample.quat_xyzw)

    assert robot_io._imu_nonfresh_consecutive == 6
    robot_io.write_ctrl(np.array([0.1], dtype=np.float32))
    assert robot_io.last_timing_s["write_ctrl"] >= 0.0

    now[0] = 0.251
    with pytest.raises(RuntimeError, match="IMU cached sample is too old"):
        robot_io.read()


def test_hardware_preflight_prints_all_statuses(capsys) -> None:
    from wr_runtime.control import run_policy

    sample = ImuSample(
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        gyro_rad_s=np.array([0.01, 0.02, 0.03], dtype=np.float32),
        timestamp_s=1.0,
        valid=True,
    )

    class _FakeActuators:
        port = "/dev/ttyUSB0"
        baudrate = 9600
        servo_ids_list = [7, 8]
        controller = SimpleNamespace(get_battery_voltage=lambda: 7.4)

        def get_positions_rad(self):
            return np.array([0.0, 0.1], dtype=np.float32)

    class _FakeFootSwitches:
        def read(self):
            return SimpleNamespace(switches=[True, True, True, True])

    class _FakeRobotIO:
        actuators = _FakeActuators()
        imu = SimpleNamespace(
            read=lambda: sample,
            error_count=0,
            last_error=None,
            diag={"quat_source": "game_quaternion"},
        )
        foot_switches = _FakeFootSwitches()
        _last_fresh_imu_sample = None

        def wait_for_valid_imu_sample(self, *, timeout_s):
            self._last_fresh_imu_sample = sample

    run_policy._run_hardware_preflight(
        robot_io=_FakeRobotIO(),
        actuator_names=["left_hip_pitch", "right_hip_pitch"],
        home_q_rad=np.array([0.0, 0.1], dtype=np.float32),
        joint_min_rad=np.array([-1.0, -1.0], dtype=np.float32),
        joint_max_rad=np.array([1.0, 1.0], dtype=np.float32),
        imu_startup_timeout_s=0.1,
        require_all_footswitches=True,
        home_tolerance_deg=25.0,
    )

    out = capsys.readouterr().out
    assert "Servo bus: port=/dev/ttyUSB0 baud=9600 voltage=7.40V" in out
    assert "left_hip_pitch" in out
    assert "right_hip_pitch" in out
    assert "IMU: valid=True" in out
    assert (
        "Footswitches: left_toe=1, left_heel=1, right_toe=1, right_heel=1" in out
    )
    assert "Hardware preflight OK." in out


def test_hardware_preflight_fails_on_open_footswitch(capsys) -> None:
    from wr_runtime.control import run_policy

    sample = ImuSample(
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        gyro_rad_s=np.zeros(3, dtype=np.float32),
        timestamp_s=1.0,
        valid=True,
    )

    class _FakeActuators:
        port = "/dev/ttyUSB0"
        baudrate = 9600
        servo_ids_list = [7]
        controller = SimpleNamespace(get_battery_voltage=lambda: None)

        def get_positions_rad(self):
            return np.array([0.0], dtype=np.float32)

    class _FakeFootSwitches:
        def read(self):
            return SimpleNamespace(switches=[True, True, True, False])

    class _FakeRobotIO:
        actuators = _FakeActuators()
        imu = SimpleNamespace(
            read=lambda: sample,
            error_count=0,
            last_error=None,
            diag={},
        )
        foot_switches = _FakeFootSwitches()
        _last_fresh_imu_sample = None

        def wait_for_valid_imu_sample(self, *, timeout_s):
            self._last_fresh_imu_sample = sample

    with pytest.raises(SystemExit, match="right_heel"):
        run_policy._run_hardware_preflight(
            robot_io=_FakeRobotIO(),
            actuator_names=["left_hip_pitch"],
            home_q_rad=np.array([0.0], dtype=np.float32),
            joint_min_rad=np.array([-1.0], dtype=np.float32),
            joint_max_rad=np.array([1.0], dtype=np.float32),
            imu_startup_timeout_s=0.1,
            require_all_footswitches=True,
            home_tolerance_deg=25.0,
        )

    out = capsys.readouterr().out
    assert "right_heel=0" in out
    assert "footswitches open at walk start" in out


def test_hardware_preflight_warns_on_open_footswitch_when_allowed(capsys) -> None:
    from wr_runtime.control import run_policy

    sample = ImuSample(
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        gyro_rad_s=np.zeros(3, dtype=np.float32),
        timestamp_s=1.0,
        valid=True,
    )

    class _FakeActuators:
        port = "/dev/ttyUSB0"
        baudrate = 9600
        servo_ids_list = [7]
        controller = SimpleNamespace(get_battery_voltage=lambda: None)

        def get_positions_rad(self):
            return np.array([0.0], dtype=np.float32)

    class _FakeFootSwitches:
        def read(self):
            return SimpleNamespace(switches=[True, True, False, True])

    class _FakeRobotIO:
        actuators = _FakeActuators()
        imu = SimpleNamespace(
            read=lambda: sample,
            error_count=0,
            last_error=None,
            diag={},
        )
        foot_switches = _FakeFootSwitches()
        _last_fresh_imu_sample = None

        def wait_for_valid_imu_sample(self, *, timeout_s):
            self._last_fresh_imu_sample = sample

    run_policy._run_hardware_preflight(
        robot_io=_FakeRobotIO(),
        actuator_names=["left_hip_pitch"],
        home_q_rad=np.array([0.0], dtype=np.float32),
        joint_min_rad=np.array([-1.0], dtype=np.float32),
        joint_max_rad=np.array([1.0], dtype=np.float32),
        imu_startup_timeout_s=0.1,
        require_all_footswitches=False,
        home_tolerance_deg=25.0,
    )

    out = capsys.readouterr().out
    assert "right_toe=0" in out
    assert "\033[33mWARNING: initial footswitches open at walk start" in out
    assert "Hardware preflight OK." in out


def test_hardware_preflight_warns_on_initial_servo_out_of_range(capsys) -> None:
    from wr_runtime.control import run_policy

    sample = ImuSample(
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        gyro_rad_s=np.zeros(3, dtype=np.float32),
        timestamp_s=1.0,
        valid=True,
    )

    class _FakeActuators:
        port = "/dev/ttyUSB0"
        baudrate = 9600
        servo_ids_list = [2]
        controller = SimpleNamespace(get_battery_voltage=lambda: None)

        def get_positions_rad(self):
            return np.array([1.2], dtype=np.float32)

    class _FakeFootSwitches:
        def read(self):
            return SimpleNamespace(switches=[True, True, True, True])

    class _FakeRobotIO:
        actuators = _FakeActuators()
        imu = SimpleNamespace(
            read=lambda: sample,
            error_count=0,
            last_error=None,
            diag={},
        )
        foot_switches = _FakeFootSwitches()
        _last_fresh_imu_sample = None

        def wait_for_valid_imu_sample(self, *, timeout_s):
            self._last_fresh_imu_sample = sample

    run_policy._run_hardware_preflight(
        robot_io=_FakeRobotIO(),
        actuator_names=["left_hip_roll"],
        home_q_rad=np.array([0.0], dtype=np.float32),
        joint_min_rad=np.array([-1.0], dtype=np.float32),
        joint_max_rad=np.array([1.0], dtype=np.float32),
        imu_startup_timeout_s=0.1,
        require_all_footswitches=True,
        home_tolerance_deg=25.0,
    )

    out = capsys.readouterr().out
    assert "left_hip_roll" in out
    assert "WARN" in out
    assert "\033[33mWARNING: left_hip_roll servo id=2 readback" in out
    assert "is outside policy range" in out
    assert "Hardware preflight OK." in out
