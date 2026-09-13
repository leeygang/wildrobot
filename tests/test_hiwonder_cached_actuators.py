from types import SimpleNamespace

import numpy as np

from runtime.wr_runtime.hardware.actuators import HiwonderCachedActuators, ServoModel
from runtime.wr_runtime.hardware.servo_io_worker import CachedServoState, ServoIOMetrics


class FakeServoIO:
    def __init__(self, state, *, write_deadband_units=0):
        self.state = state
        self.config = SimpleNamespace(write_deadband_units=int(write_deadband_units))
        self.submitted = []
        self.closed = False
        self.stopped = False
        self.unload_report = {
            "attempted_servo_ids": [1, 2],
            "commanded_servo_ids": [1, 2],
            "confirmed_servo_ids": [1, 2],
            "unverified_servo_ids": [],
            "failed_servo_ids": [],
            "errors": [],
        }

    def submit_targets_units(self, positions_by_servo_id, *, move_time_ms):
        self.submitted.append((dict(positions_by_servo_id), int(move_time_ms)))

    def get_cached_servo_state(self):
        return self.state

    def get_metrics(self):
        return ServoIOMetrics(read_success=3, read_failures=1)

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True

    def unload_servos(self, servo_ids):
        assert list(servo_ids) == [1, 2]
        return self.unload_report


def _state():
    return CachedServoState(
        servo_ids=(1, 2),
        position_units=np.array([600.0, 400.0], dtype=np.float32),
        velocity_units_s=np.array([10.0, -20.0], dtype=np.float32),
        position_age_s=np.array([0.01, 0.02], dtype=np.float32),
        read_fail_count=np.array([0, 0], dtype=np.int32),
        last_update_time_s=np.array([1.0, 1.0], dtype=np.float64),
        temperature_c=np.array([42.0, 47.0], dtype=np.float32),
        voltage_v=np.array([11.2, 10.8], dtype=np.float32),
        torque_enabled_state=np.array([1, 0], dtype=np.int8),
        temperature_age_s=np.array([0.4, 0.5], dtype=np.float32),
        voltage_age_s=np.array([0.3, 0.6], dtype=np.float32),
        torque_enabled_age_s=np.array([0.2, 0.7], dtype=np.float32),
        health_read_fail_count=np.array([0, 2], dtype=np.int32),
        last_read_group="test",
        last_read_servo_id=2,
        last_error=None,
    )


def _actuators(fake_io):
    return HiwonderCachedActuators(
        actuator_names=["j1", "j2"],
        servo_ids={"j1": 1, "j2": 2},
        default_move_time_ms=20,
        joint_servo_offset_units={"j1": 0, "j2": 0},
        joint_motor_unit_directions={"j1": 1.0, "j2": -1.0},
        servo_model=ServoModel(units_min=0, units_max=1000, units_center=500, units_per_rad=100.0),
        servo_io=fake_io,
    )


def test_cached_actuators_convert_targets_to_servo_units():
    fake_io = FakeServoIO(_state())
    actuators = _actuators(fake_io)

    actuators.set_targets_rad(np.array([0.5, -0.5], dtype=np.float32), move_time_ms=30)

    assert fake_io.submitted == [({1: 550, 2: 550}, 30)]


def test_cached_actuators_convert_cached_units_to_radians():
    fake_io = FakeServoIO(_state())
    actuators = _actuators(fake_io)

    positions = actuators.get_positions_rad()
    velocities = actuators.estimate_velocities_rad_s(dt=0.02)

    np.testing.assert_allclose(positions, np.array([1.0, 1.0], dtype=np.float32))
    np.testing.assert_allclose(velocities, np.array([0.1, 0.2], dtype=np.float32))
    diagnostics = actuators.last_position_diagnostics
    np.testing.assert_array_equal(diagnostics["servo_ids"], [1, 2])
    np.testing.assert_allclose(diagnostics["position_units"], [600.0, 400.0])
    np.testing.assert_allclose(diagnostics["velocity_units_s"], [10.0, -20.0])
    np.testing.assert_allclose(diagnostics["position_age_s"], [0.01, 0.02])
    np.testing.assert_allclose(diagnostics["temperature_c"], [42.0, 47.0])
    np.testing.assert_allclose(diagnostics["voltage_v"], [11.2, 10.8])
    np.testing.assert_array_equal(diagnostics["torque_enabled_state"], [1, 0])
    np.testing.assert_array_equal(diagnostics["health_read_fail_count"], [0, 2])


def test_cached_actuators_close_delegates_to_worker():
    fake_io = FakeServoIO(_state())
    actuators = _actuators(fake_io)

    actuators.close()

    assert fake_io.closed is True


def test_cached_actuators_records_verified_shutdown_unload():
    fake_io = FakeServoIO(_state())
    actuators = _actuators(fake_io)

    actuators.disable()

    assert fake_io.stopped is True
    assert actuators.last_shutdown_diagnostics["confirmed_servo_ids"] == [1, 2]


def test_cached_actuators_reports_last_read_servo_id():
    fake_io = FakeServoIO(_state(), write_deadband_units=3)
    actuators = _actuators(fake_io)

    metrics = actuators.get_servo_cache_metrics()

    assert metrics["servo_read_group"] == "test"
    assert metrics["servo_read_ids"] == [2]
    assert metrics["servo_write_deadband_units"] == 3


def test_cached_actuators_initial_cache_requires_fresh_age():
    state = _state()
    state.position_age_s[:] = 2.0
    fake_io = FakeServoIO(state)
    actuators = _actuators(fake_io)

    assert actuators.wait_for_initial_cache(timeout_s=0.001) is False
    assert "did not become fresh" in repr(actuators._last_error)
