from __future__ import annotations

import json
import runpy
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


_SCRIPT = Path(__file__).resolve().parents[1] / "runtime" / "scripts" / "probe_bno085.py"


def _probe_symbols() -> dict[str, object]:
    return runpy.run_path(str(_SCRIPT), run_name="probe_bno085_test")


def test_resolve_bundle_dir_infers_config_parent(tmp_path: Path) -> None:
    symbols = _probe_symbols()
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / "policy_spec.json").write_text('{"robot": {}}')

    resolved = symbols["_resolve_bundle_dir"](bundle_dir / "wildrobot_config.json", None)

    assert resolved == bundle_dir


def test_home_servo_commands_use_runtime_servo_conversion(tmp_path: Path) -> None:
    symbols = _probe_symbols()
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / "policy_spec.json").write_text(
        json.dumps(
            {
                "robot": {
                    "actuator_names": ["left_hip_pitch", "right_hip_pitch"],
                    "home_ctrl_rad": [0.1, -0.2],
                }
            }
        )
    )

    class FakeServo:
        def __init__(self, servo_id: int):
            self.id = servo_id
            self.calls: list[float] = []

        def joint_target_rad_to_elect_unit(self, target_rad: float) -> int:
            self.calls.append(float(target_rad))
            return int(round(500.0 + 1000.0 * float(target_rad)))

    left = FakeServo(1)
    right = FakeServo(5)
    cfg = SimpleNamespace(
        servo_controller=SimpleNamespace(
            servos={
                "left_hip_pitch": left,
                "right_hip_pitch": right,
            }
        )
    )

    commands = symbols["_home_servo_commands"](cfg, bundle_dir)

    assert commands == [(1, 600), (5, 300)]
    assert left.calls == [0.1]
    assert right.calls == [-0.2]


def test_home_servo_commands_reject_missing_servo(tmp_path: Path) -> None:
    symbols = _probe_symbols()
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / "policy_spec.json").write_text(
        json.dumps(
            {
                "robot": {
                    "actuator_names": ["left_hip_pitch"],
                    "home_ctrl_rad": [0.1],
                }
            }
        )
    )
    cfg = SimpleNamespace(servo_controller=SimpleNamespace(servos={}))

    with pytest.raises(ValueError, match="left_hip_pitch"):
        symbols["_home_servo_commands"](cfg, bundle_dir)


def test_home_command_thread_repeats_until_stopped() -> None:
    symbols = _probe_symbols()

    class FakeController:
        def __init__(self):
            self.calls: list[tuple[list[tuple[int, int]], int]] = []

        def move_servos(self, commands: list[tuple[int, int]], time_ms: int) -> bool:
            self.calls.append((list(commands), int(time_ms)))
            return True

    controller = FakeController()
    stop_event = threading.Event()
    thread, stats = symbols["_start_home_command_thread"](
        controller,
        [(1, 500)],
        start_s=time.monotonic(),
        home_after_s=0.0,
        home_move_ms=20,
        repeat_home_hz=100.0,
        stop_event=stop_event,
    )

    time.sleep(0.035)
    stop_event.set()
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert len(controller.calls) >= 2
    assert controller.calls[0] == ([(1, 500)], 20)
    assert stats["sent"] == len(controller.calls)
    assert stats["error"] is None


def test_quat_tilt_reports_body_inclination() -> None:
    symbols = _probe_symbols()
    half_angle = np.deg2rad(30.0) / 2.0
    quat = np.array([np.cos(half_angle), 0.0, np.sin(half_angle), 0.0])

    tilt_rad = symbols["_quat_tilt_rad"](quat)

    assert tilt_rad is not None
    assert np.rad2deg(tilt_rad) == pytest.approx(30.0)


def test_quat_rpy_reports_pitch() -> None:
    symbols = _probe_symbols()
    half_angle = np.deg2rad(12.0) / 2.0
    quat = np.array([np.cos(half_angle), 0.0, np.sin(half_angle), 0.0])

    rpy_deg = symbols["_quat_rpy_deg"](quat)

    np.testing.assert_allclose(rpy_deg, [0.0, 12.0, 0.0], atol=1e-6)


def test_home_state_diagnostics_logs_runtime_signals_and_unloads(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    symbols = _probe_symbols()
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / "policy_spec.json").write_text(
        json.dumps(
            {
                "robot": {
                    "actuator_names": ["left_hip_pitch", "right_hip_pitch"],
                    "home_ctrl_rad": [0.1, -0.2],
                }
            }
        )
    )

    half_angle = np.deg2rad(5.0) / 2.0

    class FakeActuators:
        def __init__(self):
            self.targets = []

        def wait_for_initial_cache(self, *, timeout_s):
            return True

        def set_targets_rad(self, targets, *, move_time_ms):
            self.targets.append((np.asarray(targets).copy(), move_time_ms))

    class FakeRobotIO:
        def __init__(self):
            self.actuators = FakeActuators()
            self.closed = False
            self.last_servo_metrics = {
                "servo_cache_age_max_s": 0.02,
                "servo_cache_age_leg_max_s": 0.02,
                "servo_read_fail_count": 0,
            }
            self.read_count = 0

        def wait_for_valid_imu_sample(self, *, timeout_s):
            return None

        def read(self):
            self.read_count += 1
            return SimpleNamespace(
                quat_wxyz=np.array(
                    [np.cos(half_angle), 0.0, np.sin(half_angle), 0.0],
                    dtype=np.float32,
                ),
                gyro_rad_s=np.array([0.0, 0.1, 0.0], dtype=np.float32),
                joint_pos_rad=np.array([0.11, -0.19], dtype=np.float32),
                foot_switches=np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32),
                timestamp_s=float(self.read_count),
            )

        def close(self):
            self.closed = True

    robot_io = FakeRobotIO()
    import wr_runtime.control.run_policy as run_policy

    monkeypatch.setattr(
        run_policy,
        "_build_hardware_robot_io",
        lambda **_kwargs: robot_io,
    )

    result = symbols["_run_home_state_diagnostics"](
        config_path=tmp_path / "hardware_config.json",
        cfg=SimpleNamespace(),
        bundle_dir=bundle_dir,
        total=2,
        dt_s=0.001,
        home_after_s=0.0,
        home_move_ms=2000,
        max_tilt_deg=15.0,
        print_every=0,
    )

    output = capsys.readouterr().out
    assert result == 0
    assert output.count("HOME_DIAGNOSTIC_SAMPLE ") == 2
    assert '"footswitches":[1,1,0,0]' in output
    assert '"status":"complete"' in output
    assert len(robot_io.actuators.targets) == 1
    assert robot_io.actuators.targets[0][1] == 2000
    assert robot_io.closed
