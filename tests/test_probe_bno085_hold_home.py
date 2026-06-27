from __future__ import annotations

import json
import runpy
import threading
import time
from pathlib import Path
from types import SimpleNamespace

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
