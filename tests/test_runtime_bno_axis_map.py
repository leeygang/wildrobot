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


def test_enable_feature_retries_unknown_0x7b_report() -> None:
    from runtime.wr_runtime.hardware.bno085 import _enable_feature_allowing_unknown_reports

    class FakeAdafruitImu:
        def __init__(self):
            self.calls = 0

        def enable_feature(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise KeyError(0x7B)

    imu = FakeAdafruitImu()

    _enable_feature_allowing_unknown_reports(imu, 0x02, report_interval=5000)

    assert imu.calls == 2


def test_enable_feature_does_not_hide_other_key_errors() -> None:
    from runtime.wr_runtime.hardware.bno085 import _enable_feature_allowing_unknown_reports

    class FakeAdafruitImu:
        def enable_feature(self, *args, **kwargs):
            raise KeyError(0x7C)

    with pytest.raises(KeyError):
        _enable_feature_allowing_unknown_reports(FakeAdafruitImu(), 0x02)


def test_bno_read_rejects_bad_quaternion_norm() -> None:
    from runtime.wr_runtime.hardware.bno085 import BNO085IMU
    from runtime.wr_runtime.hardware.imu import ImuSample

    class FakeAdafruitImu:
        _sequence_number = [0, 0, 0, 0]

        @property
        def game_quaternion(self):
            self._sequence_number[3] += 1
            return (0.001, 0.001, 0.001, 0.001)

        @property
        def gyro(self):
            return (0.0, 0.0, 0.0)

    imu = BNO085IMU.__new__(BNO085IMU)
    imu._imu = FakeAdafruitImu()
    imu._latest = ImuSample(
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        gyro_rad_s=np.zeros(3, dtype=np.float32),
        timestamp_s=None,
        valid=False,
    )
    imu._use_game_quat = True
    imu.enable_rotation_vector = False
    imu.max_quat_norm_deviation = 0.1
    imu.suppress_debug = True
    imu.upside_down = False
    imu._r_bs = None
    imu._diag = {}

    sample = imu._read_sample_once()

    assert not sample.valid
    assert sample.timestamp_s is None
    np.testing.assert_allclose(sample.quat_xyzw, [0.0, 0.0, 0.0, 1.0])
    assert imu.diag["quat_status"] == "bad_norm"


def test_bno_read_integrates_gyro_when_runtime_quaternion_is_bad() -> None:
    import time

    from runtime.wr_runtime.hardware.bno085 import BNO085IMU
    from runtime.wr_runtime.hardware.imu import ImuSample

    class FakeAdafruitImu:
        _sequence_number = [0, 0, 0, 0]

        @property
        def game_quaternion(self):
            self._sequence_number[3] += 1
            return (0.0, 0.0, 0.0, 2.0)

        @property
        def gyro(self):
            return (0.0, 0.0, 1.0)

    previous = ImuSample(
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        gyro_rad_s=np.zeros(3, dtype=np.float32),
        timestamp_s=time.monotonic() - 0.02,
        valid=True,
        fresh=True,
    )
    imu = BNO085IMU.__new__(BNO085IMU)
    imu._imu = FakeAdafruitImu()
    imu._latest = previous
    imu._last_report_sample = previous
    imu._use_game_quat = True
    imu.enable_rotation_vector = False
    imu.max_quat_norm_deviation = 0.1
    imu.suppress_debug = True
    imu.upside_down = False
    imu._r_bs = None
    imu._diag = {}

    sample = imu._read_sample_once()

    assert sample.valid is True
    assert sample.fresh is True
    assert sample.timestamp_s is not None
    assert sample.timestamp_s > previous.timestamp_s
    np.testing.assert_allclose(np.linalg.norm(sample.quat_xyzw), 1.0, rtol=1e-5)
    assert imu.diag["quat_status"] == "integrated_from_gyro_after_bad_norm"


def test_bno_read_does_not_integrate_from_old_cached_quaternion() -> None:
    import time

    from runtime.wr_runtime.hardware.bno085 import BNO085IMU
    from runtime.wr_runtime.hardware.imu import ImuSample

    class FakeAdafruitImu:
        _sequence_number = [0, 0, 0, 0]

        @property
        def game_quaternion(self):
            self._sequence_number[3] += 1
            return (0.0, 0.0, 0.0, 2.0)

        @property
        def gyro(self):
            return (0.0, 0.0, 1.0)

    previous = ImuSample(
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        gyro_rad_s=np.zeros(3, dtype=np.float32),
        timestamp_s=time.monotonic() - 1.0,
        valid=True,
        fresh=True,
    )
    imu = BNO085IMU.__new__(BNO085IMU)
    imu._imu = FakeAdafruitImu()
    imu._latest = previous
    imu._last_report_sample = previous
    imu._use_game_quat = True
    imu.enable_rotation_vector = False
    imu.max_quat_norm_deviation = 0.1
    imu.suppress_debug = True
    imu.upside_down = False
    imu._r_bs = None
    imu._diag = {}

    sample = imu._read_sample_once()

    assert sample.valid is False
    assert sample.fresh is False
    assert sample.timestamp_s == previous.timestamp_s
    assert imu.diag["quat_status"] == "bad_norm"
    assert float(imu.diag["quat_integration_skipped_dt_s"]) > 0.25


def test_bno_read_does_not_probe_game_quaternion_property_twice() -> None:
    from runtime.wr_runtime.hardware.bno085 import BNO085IMU
    from runtime.wr_runtime.hardware.imu import ImuSample

    class FakeAdafruitImu:
        _sequence_number = [0, 0, 0, 0]

        def __init__(self) -> None:
            self.game_quaternion_reads = 0

        @property
        def game_quaternion(self):
            self.game_quaternion_reads += 1
            self._sequence_number[3] += 1
            return (0.0, 0.0, 0.0, 1.0)

        @property
        def gyro(self):
            return (0.0, 0.0, 0.0)

    adafruit_imu = FakeAdafruitImu()
    imu = BNO085IMU.__new__(BNO085IMU)
    imu._imu = adafruit_imu
    imu._latest = ImuSample(
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        gyro_rad_s=np.zeros(3, dtype=np.float32),
        timestamp_s=None,
        valid=False,
        fresh=False,
    )
    imu._last_report_sample = imu._latest
    imu._use_game_quat = True
    imu.enable_rotation_vector = False
    imu.max_quat_norm_deviation = 0.1
    imu.suppress_debug = True
    imu.upside_down = False
    imu._r_bs = None
    imu._diag = {}

    sample = imu._read_sample_once()

    assert sample.valid is True
    assert sample.fresh is True
    assert adafruit_imu.game_quaternion_reads == 1


def test_bno_background_read_marks_cached_sample_not_fresh() -> None:
    from queue import Queue

    from runtime.wr_runtime.hardware.bno085 import BNO085IMU
    from runtime.wr_runtime.hardware.imu import ImuSample

    imu = BNO085IMU.__new__(BNO085IMU)
    imu.polling_mode = False
    imu._q = Queue(maxsize=1)
    imu._latest = ImuSample(
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        gyro_rad_s=np.zeros(3, dtype=np.float32),
        timestamp_s=1.0,
        valid=True,
        fresh=True,
    )
    imu._diag = {}

    sample = imu.read()

    assert sample.valid is True
    assert sample.fresh is False
    assert sample.timestamp_s == 1.0
    assert imu.diag["payload_status"] == "stale"
    assert imu.diag["read_status"] == "cached"
