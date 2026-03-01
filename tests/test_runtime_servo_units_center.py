import numpy as np

from runtime.configs.config import ServoConfig
from runtime.wr_runtime.hardware.actuators import (
    ServoModel,
    joint_target_rad_to_servo_pos_elec_units,
    servo_pos_elect_units_to_joint_target_rad,
)


def test_rad_to_units_center_uses_center_rad_not_joint_range_mid() -> None:
    # Pick a joint range whose midpoint is NOT zero.
    rad_range = (-1.0, 2.0)
    assert (rad_range[0] + rad_range[1]) / 2.0 == 0.5

    # Keep motor_center_mujoco_deg at 0 so center_rad == 0.
    servo = ServoConfig(
        id=1,
        offset=17,
        motor_sign=1.0,
        motor_center_mujoco_deg=0.0,
        rad_range=rad_range,
    )

    # Servo electrical units are centered at 500 + offset when target_rad == center_rad.
    assert servo.center_rad == 0.0
    assert servo.ctrl_center == 0.5

    units_at_center_rad = servo.rad_to_units(servo.center_rad)
    assert units_at_center_rad == ServoConfig.UNITS_CENTER + servo.offset

    # But the joint-range midpoint (ctrl_center) is a different angle, so it should
    # generally NOT map to 500 (unless by coincidence).
    units_at_ctrl_center = servo.rad_to_units(servo.ctrl_center)
    assert units_at_ctrl_center != ServoConfig.UNITS_CENTER + servo.offset


def test_vectorized_mapping_center_is_units_center_plus_offset() -> None:
    servo_model = ServoModel(units_center=500)

    targets_rad = np.asarray([0.25, -0.75], dtype=np.float32)
    centers_rad = np.asarray([0.25, -0.75], dtype=np.float32)
    offsets_unit = np.asarray([10.0, -20.0], dtype=np.float32)
    motor_signs = np.asarray([1.0, -1.0], dtype=np.float32)

    units = joint_target_rad_to_servo_pos_elec_units(
        targets_rad=targets_rad,
        offsets_unit=offsets_unit,
        motor_signs=motor_signs,
        centers_rad=centers_rad,
        servo_model=servo_model,
    )

    # When target_rad == center_rad, delta == 0 so units == units_center + offset.
    assert np.allclose(units, np.asarray([510.0, 480.0], dtype=np.float32))

    # Round-trip should land back at centers_rad.
    back_rad = servo_pos_elect_units_to_joint_target_rad(
        units=units,
        offsets_unit=offsets_unit,
        motor_signs=motor_signs,
        centers_rad=centers_rad,
        servo_model=servo_model,
    )
    assert np.allclose(back_rad, centers_rad)
