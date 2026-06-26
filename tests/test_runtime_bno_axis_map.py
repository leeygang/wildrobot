import numpy as np
import pytest


def test_axis_map_to_matrix_identity() -> None:
    from runtime.wr_runtime.hardware.bno085 import _axis_map_to_r_bs

    r = _axis_map_to_r_bs(["+X", "+Y", "+Z"])
    np.testing.assert_allclose(r, np.eye(3, dtype=np.float32))


def test_axis_map_to_matrix_rejects_duplicates() -> None:
    from runtime.wr_runtime.hardware.bno085 import _axis_map_to_r_bs

    with pytest.raises(ValueError):
        _axis_map_to_r_bs(["+X", "-X", "+Z"])


def test_axis_map_to_matrix_rejects_bad_format() -> None:
    from runtime.wr_runtime.hardware.bno085 import _axis_map_to_r_bs

    with pytest.raises(ValueError):
        _axis_map_to_r_bs(["X", "Y", "Z"])


def test_axis_map_applies_rotation() -> None:
    from runtime.wr_runtime.hardware.bno085 import _axis_map_to_r_bs

    r = _axis_map_to_r_bs(["+Y", "+Z", "+X"])  # body_x=imu_y, body_y=imu_z, body_z=imu_x
    vec_sensor = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    vec_body = r @ vec_sensor
    np.testing.assert_allclose(vec_body, np.array([2.0, 3.0, 1.0], dtype=np.float32))


def test_axis_map_sign_flip_right_handed() -> None:
    from runtime.wr_runtime.hardware.bno085 import _axis_map_to_r_bs

    r = _axis_map_to_r_bs(["-X", "-Y", "+Z"])
    vec_sensor = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    vec_body = r @ vec_sensor
    np.testing.assert_allclose(vec_body, np.array([-1.0, 0.0, 0.0], dtype=np.float32))


def test_axis_map_rejects_left_handed_reflection() -> None:
    from runtime.wr_runtime.hardware.bno085 import _axis_map_to_r_bs

    with pytest.raises(ValueError, match="right-handed"):
        _axis_map_to_r_bs(["+X", "-Y", "+Z"])


def test_imu_payload_changed_only_when_report_values_change() -> None:
    from runtime.wr_runtime.hardware.bno085 import _imu_payload_changed
    from runtime.wr_runtime.hardware.imu import ImuSample

    sample = ImuSample(
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        gyro_rad_s=np.array([0.1, 0.0, 0.0], dtype=np.float32),
        timestamp_s=1.0,
        valid=True,
    )

    assert not _imu_payload_changed(
        sample,
        np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        np.array([0.1, 0.0, 0.0], dtype=np.float32),
    )
    assert _imu_payload_changed(
        sample,
        np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        np.array([0.2, 0.0, 0.0], dtype=np.float32),
    )


def test_input_report_sequence_reads_adafruit_channel_counter() -> None:
    from runtime.wr_runtime.hardware.bno085 import _input_report_sequence

    class FakeAdafruitImu:
        _sequence_number = [0, 1, 2, 42, 4, 5]

    assert _input_report_sequence(FakeAdafruitImu()) == 42
