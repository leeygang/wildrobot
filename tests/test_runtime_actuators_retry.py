import numpy as np
import pytest

from runtime.wr_runtime.hardware.actuators import HiwonderBoardActuators, ServoModel, servo_units_to_rad


class FakeController:
    def __init__(self, move_failures: int = 0, read_failures: int = 0, read_response=None):
        self.move_failures = move_failures
        self.read_failures = read_failures
        self.read_response = read_response
        self.move_calls = 0
        self.read_calls = 0
        self.unload_calls = 0
        self.last_commands = None
        self.last_time_ms = None

    def move_servos(self, commands, time_ms):
        self.move_calls += 1
        if self.move_failures > 0:
            self.move_failures -= 1
            raise IOError("write failed")
        self.last_commands = commands
        self.last_time_ms = time_ms
        return True

    def read_servo_positions(self, servo_ids):
        self.read_calls += 1
        if self.read_failures > 0:
            self.read_failures -= 1
            return None
        return self.read_response

    def unload_servos(self, servo_ids):
        self.unload_calls += 1
        return True

    def close(self):
        return True


def _make_actuators(controller: FakeController, max_retries: int = 3):
    return HiwonderBoardActuators(
        actuator_names=["j1", "j2"],
        servo_ids={"j1": 1, "j2": 2},
        joint_offsets_rad={"j1": 0.0, "j2": 0.0},
        port="/dev/null",
        baudrate=9600,
        default_move_time_ms=20,
        servo_model=ServoModel(units_min=0, units_max=1000, units_center=500, units_per_rad=100.0),
        max_retries=max_retries,
        retry_backoff_s=0.0,
        controller=controller,
    )


def test_set_targets_retries_and_succeeds():
    controller = FakeController(move_failures=2)
    actuators = _make_actuators(controller, max_retries=3)
    actuators.set_targets_rad(np.array([0.0, 0.0], dtype=np.float32), move_time_ms=30)

    assert controller.move_calls == 3
    assert controller.last_time_ms == 30
    assert controller.last_commands == [(1, 500), (2, 500)]


def test_set_targets_raises_after_exhausted_retries():
    controller = FakeController(move_failures=3)
    actuators = _make_actuators(controller, max_retries=2)
    with pytest.raises(RuntimeError):
        actuators.set_targets_rad(np.array([0.0, 0.0], dtype=np.float32))


def test_get_positions_retries_then_succeeds():
    controller = FakeController(
        read_failures=2,
        read_response=[(1, 600), (2, 400)],
    )
    actuators = _make_actuators(controller, max_retries=3)

    positions = actuators.get_positions_rad()
    assert controller.read_calls == 3
    assert positions is not None
    expected = servo_units_to_rad(
        np.array([600, 400], dtype=np.float32),
        offsets_rad=np.zeros(2, dtype=np.float32),
        directions=np.ones(2, dtype=np.float32),
        servo_model=actuators.servo_model,
    )
    np.testing.assert_allclose(positions, expected)


def test_get_positions_returns_none_after_exhausted_retries():
    controller = FakeController(read_failures=3, read_response=None)
    actuators = _make_actuators(controller, max_retries=2)
    positions = actuators.get_positions_rad()
    assert positions is None
