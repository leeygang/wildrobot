import numpy as np
import pytest

import runtime.wr_runtime.hardware.actuators as act_mod
from runtime.wr_runtime.hardware.actuators import HiwonderBoardActuators, ServoModel


class FakeController:
    def __init__(self, positions):
        self.positions = dict(positions)
        self.read_calls = []
        self.unload_calls = []

    def read_servo_positions(self, servo_ids):
        ids = [int(sid) for sid in servo_ids]
        self.read_calls.append(ids)
        return [(sid, int(self.positions[sid])) for sid in ids if sid in self.positions]

    def move_servos(self, commands, time_ms):
        return True

    def unload_servos(self, servo_ids):
        self.unload_calls.append(list(servo_ids))
        return True

    def close(self):
        return True


def _make_actuators(controller: FakeController):
    return HiwonderBoardActuators(
        actuator_names=["left_hip_pitch", "right_hip_pitch", "waist_yaw"],
        servo_ids={"left_hip_pitch": 1, "right_hip_pitch": 2, "waist_yaw": 3},
        port="/dev/null",
        baudrate=9600,
        default_move_time_ms=20,
        joint_servo_offset_units={
            "left_hip_pitch": 0,
            "right_hip_pitch": 0,
            "waist_yaw": 0,
        },
        servo_model=ServoModel(
            units_min=0,
            units_max=1000,
            units_center=500,
            units_per_rad=100.0,
        ),
        controller=controller,
        retry_backoff_s=0.0,
        read_schedule_mode="staggered",
        read_schedule_groups=[["left_hip_pitch"], ["right_hip_pitch"]],
        read_schedule_max_cache_age_s={"leg": 1.0, "default": 1.0},
    )


def test_staggered_read_initializes_full_cache_then_reads_one_group(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(act_mod.time, "monotonic", lambda: now[0])
    controller = FakeController({1: 600, 2: 400, 3: 500})
    actuators = _make_actuators(controller)

    first = actuators.get_positions_rad()
    assert controller.read_calls == [[1, 2, 3]]
    np.testing.assert_allclose(first, np.array([1.0, -1.0, 0.0], dtype=np.float32))

    now[0] = 0.05
    controller.positions[1] = 650
    second = actuators.get_positions_rad()

    assert controller.read_calls[-1] == [1]
    np.testing.assert_allclose(second, np.array([1.5, -1.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(
        actuators.estimate_velocities_rad_s(0.02),
        np.array([10.0, 0.0, 0.0], dtype=np.float32),
    )

    metrics = actuators.get_servo_cache_metrics()
    assert metrics["servo_read_mode"] == "staggered"
    assert metrics["servo_read_group"] == "left_leg"
    assert metrics["servo_read_ids"] == [1]
    assert metrics["servo_read_count"] == 1
    assert metrics["servo_cache_uninitialized_count"] == 0


def test_staggered_read_aborts_when_cache_age_exceeds_limit(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(act_mod.time, "monotonic", lambda: now[0])
    controller = FakeController({1: 600, 2: 400, 3: 500})
    actuators = _make_actuators(controller)
    actuators._cache_age_limit_s[:] = 0.01

    assert actuators.get_positions_rad() is not None
    now[0] = 0.02

    with pytest.raises(RuntimeError, match="servo cache age exceeded"):
        actuators.get_positions_rad()
