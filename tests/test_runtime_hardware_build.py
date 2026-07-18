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


def test_calibrate_go_home_skips_global_zero_prompt(monkeypatch) -> None:
    """With --go-home, calibration should preserve home pose instead of asking
    to immediately move every joint away to MuJoCo zero."""
    import builtins
    import runtime.scripts.calibrate as calibrate_mod
    import wr_runtime.hardware.hiwonder_board_controller as hw_mod

    class _FakeController:
        def __init__(self, config):
            self.config = config

        def move_servos(self, cmds, move_ms):
            pass

        def unload_servos(self, servo_ids):
            pass

        def close(self):
            pass

    prompts: list[str] = []
    responses = iter(["y", "q"])

    def _input(prompt: str = "") -> str:
        prompts.append(prompt)
        return next(responses)

    monkeypatch.setattr(hw_mod, "HiwonderBoardController", _FakeController)
    monkeypatch.setattr(builtins, "input", _input)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "calibrate.py",
            "--config",
            str(_V2_CONFIG),
            "--calibrate",
            "--go-home",
            "--pause-s",
            "0",
        ],
    )

    calibrate_mod.main()

    assert any("Move all servos to home pose now?" in prompt for prompt in prompts)
    assert not any("Move all joints to MuJoCo joint_pos_deg 0" in prompt for prompt in prompts)


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
            self.port = config.port
            self.baudrate = config.baudrate

    class _FakeRawBus:
        def __init__(self, transport, config):
            captured["raw_bus_config"] = config
            self.transport = transport

    class _FakeServoIOWorker:
        def __init__(self, raw_bus, config):
            captured["worker_config"] = config
            self.raw_bus = raw_bus
            self.started = False

        def start(self):
            self.started = True
            captured["worker_started"] = True

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


def test_build_hardware_robot_io_passes_servo_read_schedule(monkeypatch) -> None:
    import configs
    import wr_runtime.hardware.bno085 as bno_mod
    import wr_runtime.hardware.foot_switches as fs_mod
    from wr_runtime.control import run_policy

    captured = {}
    _patch_ttl_servo_backend(monkeypatch, captured)
    cfg = _fake_runtime_config()
    cfg.servo_read_schedule = SimpleNamespace(
        mode="staggered",
        groups=[["j"]],
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
