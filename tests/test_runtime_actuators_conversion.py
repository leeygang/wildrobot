import numpy as np

from runtime.wr_runtime.hardware.actuators import (
    ServoModel,
    rad_to_servo_units,
    servo_units_to_rad,
)


def test_rad_to_units_and_back_identity():
    servo_model = ServoModel()
    offsets = np.array([0.0, 0.1], dtype=np.float32)
    directions = np.array([1.0, -1.0], dtype=np.float32)
    targets = np.array([0.0, 0.2], dtype=np.float32)

    units = rad_to_servo_units(targets, offsets, directions, servo_model)
    expected_units = servo_model.units_center + directions * (targets + offsets) * servo_model.units_per_rad
    np.testing.assert_allclose(units, expected_units)

    recovered = servo_units_to_rad(units, offsets, directions, servo_model)
    np.testing.assert_allclose(recovered, targets, atol=1e-6)


def test_rad_to_units_clamps_to_limits():
    servo_model = ServoModel(units_min=0, units_max=1000, units_center=500, units_per_rad=10.0)
    offsets = np.zeros(2, dtype=np.float32)
    directions = np.ones(2, dtype=np.float32)
    targets = np.array([100.0, -100.0], dtype=np.float32)

    units = rad_to_servo_units(targets, offsets, directions, servo_model)
    assert units[0] == servo_model.units_max
    assert units[1] == servo_model.units_min
