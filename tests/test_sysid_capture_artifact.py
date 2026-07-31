from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.sysid.run_capture import build_capture_artifact
from tools.sysid.run_capture import _capture_measured_response_hardware
from tools.sysid.run_capture import _read_servo_position_units


def test_sysid_capture_artifact_shapes() -> None:
    artifact = build_capture_artifact(
        mode="step",
        joint_name="left_knee_pitch",
        sample_rate_hz=50.0,
        duration_s=2.0,
        amplitude_rad=0.2,
        hold_rad=0.0,
        step_start_s=0.5,
        chirp_start_hz=0.2,
        chirp_end_hz=2.0,
        model_delay_steps=1,
        model_backlash_rad=0.01,
        model_tau_s=0.08,
        model_noise_std=0.001,
        seed=1,
        metadata={"asset_version": "v2"},
    )
    artifact.validate()
    t = artifact.timestamps_s.shape[0]
    assert t > 10
    assert artifact.command_rad.shape == (t,)
    assert artifact.measured_position_rad.shape == (t,)
    assert artifact.measured_velocity_rad_s.shape == (t,)
    assert np.all(np.isfinite(artifact.command_rad))


class _FakeController:
    def __init__(self) -> None:
        self.calls = 0

    def read_servo_positions(self, ids):
        self.calls += 1
        if self.calls < 3:
            return None
        return [(int(ids[0]), 512)]


def test_read_servo_position_units_retries_and_succeeds() -> None:
    controller = _FakeController()
    units = _read_servo_position_units(
        controller,
        servo_id=7,
        retries=5,
        retry_sleep_s=0.0,
    )
    assert units == 512


def test_hardware_capture_uses_runtime_ttl_controller(monkeypatch) -> None:
    class _FakeServo:
        id = 3

        @staticmethod
        def joint_target_rad_to_elect_unit(value):
            return int(round(500.0 + (100.0 * value)))

        @staticmethod
        def servo_elect_units_to_joint_target_rad(value):
            return (float(value) - 500.0) / 100.0

    class _FakeServoControllerConfig:
        @staticmethod
        def get_servo(_joint_name):
            return _FakeServo()

    class _FakeConfig:
        servo_controller = _FakeServoControllerConfig()

    class _FakeTtlController:
        def __init__(self) -> None:
            self.position = 500
            self.closed = False
            self.move_commands = []

        def move_servos(self, commands, time_ms):
            assert time_ms == 20
            self.move_commands.append(commands[0])
            self.position = int(commands[0][1])
            return True

        def read_servo_positions(self, ids):
            return [(int(ids[0]), self.position)]

        def close(self):
            self.closed = True

    controller = _FakeTtlController()
    monkeypatch.setattr(
        "tools.sysid.run_capture.build_ttl_servo_controller",
        lambda config: controller,
    )

    position, velocity, timestamps = _capture_measured_response_hardware(
        cfg=_FakeConfig(),
        joint_name="left_knee_pitch",
        command_rad=np.array([0.0, 0.0, 0.1, 0.1], dtype=np.float32),
        sample_rate_hz=1000.0,
        move_time_ms=20,
        read_retries=1,
        read_retry_sleep_s=0.0,
        settle_s=0.0,
        return_to_hold=True,
    )

    np.testing.assert_allclose(position, [0.0, 0.0, 0.1, 0.1], atol=1e-6)
    assert velocity.shape == (4,)
    assert timestamps.shape == (4,)
    assert controller.position == 500
    assert controller.move_commands == [(3, 500), (3, 510), (3, 500)]
    assert controller.closed
