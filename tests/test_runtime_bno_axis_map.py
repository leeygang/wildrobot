from types import SimpleNamespace

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


def test_resolve_board_pin_accepts_blinka_and_gpio_names() -> None:
    from runtime.wr_runtime.hardware.bno085 import _resolve_board_pin

    class FakeBoard:
        D8 = object()
        D17 = object()

    assert _resolve_board_pin(FakeBoard, "D8") is FakeBoard.D8
    assert _resolve_board_pin(FakeBoard, "board.D8") is FakeBoard.D8
    assert _resolve_board_pin(FakeBoard, "GPIO17") is FakeBoard.D17
    with pytest.raises(ValueError, match="Unknown Blinka board pin"):
        _resolve_board_pin(FakeBoard, "NOPE")


def test_spi_init_failure_detail_mentions_corrupt_header() -> None:
    from runtime.wr_runtime.hardware.bno085 import _format_init_failure_detail

    detail = _format_init_failure_detail(
        transport="spi",
        i2c_address=0x4B,
        i2c_frequency_hz=100_000,
        spi_baudrate=1_000_000,
        spi_read_skip_bytes=2,
        spi_cs_pin="D8",
        spi_int_pin="D17",
        spi_reset_pin="D27",
        last_exc=IndexError("list assignment index out of range"),
    )

    assert "corrupt SHTP header" in detail
    assert "spi_read_skip_bytes=2" in detail
    assert "MISO/MOSI" in detail
    assert "PS0/PS1" in detail


def test_spi_read_skip_reads_packet_in_one_transaction(monkeypatch) -> None:
    from runtime.wr_runtime.hardware.bno085 import _make_bno08x_spi_read_skip_class

    class FakePacketError(Exception):
        pass

    class FakePacket:
        def __init__(self, buf):
            self.data = bytes(buf[:20])

        @staticmethod
        def header_from_buffer(buf):
            return SimpleNamespace(
                packet_byte_count=((int(buf[1]) << 8) | int(buf[0])) & 0x7FFF,
                channel_number=int(buf[2]),
                sequence_number=int(buf[3]),
            )

    class FakeBase:
        def __init__(self, *args, **kwargs):
            self._data_buffer = bytearray(64)
            self._sequence_number = [0] * 6
            self._debug = False
            self._int = SimpleNamespace(value=False)
            self.updated_packet = None

        def _read_packet(self):
            raise NotImplementedError

        def _dbg(self, *args):
            pass

        def _update_sequence_number(self, packet):
            self.updated_packet = packet

    class FakeSpi:
        def __init__(self):
            self.chunks = [
                bytes([0x00, 0x00, 0x14, 0x00, 0x01, 0x00]),
                bytes([0x01, 0x04, 0x00, 0x00, 0x00, 0x00, 0x80, 0x06,
                       0x31, 0x2E, 0x30, 0x2E, 0x30, 0x00, 0x02, 0x02]),
            ]
            self.reads = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def readinto(self, buf, start=0, end=None, write_value=0):
            if end is None:
                end = len(buf)
            self.reads.append((int(start), int(end), int(write_value)))
            data = self.chunks.pop(0)
            buf[start:end] = data[: end - start]

    monkeypatch.setitem(globals(), "Packet", FakePacket)
    monkeypatch.setitem(globals(), "PacketError", FakePacketError)

    spi_cls = _make_bno08x_spi_read_skip_class(FakeBase)
    imu = spi_cls(read_skip_bytes=2)
    fake_spi = FakeSpi()
    imu._spi = fake_spi

    packet = imu._read_packet()

    assert packet.data[:4] == bytes([0x14, 0x00, 0x01, 0x00])
    assert imu._sequence_number[1] == 0
    assert imu.updated_packet is packet
    assert fake_spi.reads == [(0, 6, 0), (4, 20, 0)]


def test_spi_read_skip_rejects_invalid_channel_as_packet_error(monkeypatch) -> None:
    from runtime.wr_runtime.hardware.bno085 import _make_bno08x_spi_read_skip_class

    class FakePacketError(Exception):
        pass

    class FakePacket:
        @staticmethod
        def header_from_buffer(buf):
            return SimpleNamespace(
                packet_byte_count=((int(buf[1]) << 8) | int(buf[0])) & 0x7FFF,
                channel_number=int(buf[2]),
                sequence_number=int(buf[3]),
            )

    class FakeBase:
        def __init__(self, *args, **kwargs):
            self._data_buffer = bytearray(64)
            self._sequence_number = [0] * 6
            self._debug = False
            self._int = SimpleNamespace(value=False)

        def _read_packet(self):
            raise NotImplementedError

        def _wait_for_int(self):
            pass

        def _dbg(self, *args):
            pass

    class FakeSpi:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def readinto(self, buf, start=0, end=None, write_value=0):
            if end is None:
                end = len(buf)
            buf[start:end] = bytes([0x00, 0x00, 0x80, 0x01, 0x80, 0x00])[: end - start]

    monkeypatch.setitem(globals(), "Packet", FakePacket)
    monkeypatch.setitem(globals(), "PacketError", FakePacketError)

    spi_cls = _make_bno08x_spi_read_skip_class(FakeBase)
    imu = spi_cls(read_skip_bytes=2)
    imu._spi = FakeSpi()

    with pytest.raises(FakePacketError, match="channel=128"):
        imu._read_packet()


def test_spi_read_skip_data_ready_follows_int_pin(monkeypatch) -> None:
    from runtime.wr_runtime.hardware.bno085 import _make_bno08x_spi_read_skip_class

    class FakePacketError(Exception):
        pass

    class FakePacket:
        pass

    class FakeBase:
        def __init__(self, *args, **kwargs):
            self._int = SimpleNamespace(value=True)

        def _read_packet(self):
            raise NotImplementedError

    monkeypatch.setitem(globals(), "Packet", FakePacket)
    monkeypatch.setitem(globals(), "PacketError", FakePacketError)

    spi_cls = _make_bno08x_spi_read_skip_class(FakeBase)
    imu = spi_cls(read_skip_bytes=2)

    assert imu._data_ready is False
    imu._int.value = False
    assert imu._data_ready is True


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


def test_bno_read_records_property_exception_details() -> None:
    from runtime.wr_runtime.hardware.bno085 import BNO085IMU
    from runtime.wr_runtime.hardware.imu import ImuSample

    class FakeAdafruitImu:
        _sequence_number = [0, 0, 0, 0]

        @property
        def game_quaternion(self):
            raise RuntimeError("no quaternion report")

        @property
        def gyro(self):
            raise RuntimeError("no gyro report")

    imu = BNO085IMU.__new__(BNO085IMU)
    imu._imu = FakeAdafruitImu()
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

    assert sample.valid is False
    assert imu.diag["quat_exception_type"] == "RuntimeError"
    assert imu.diag["quat_exception_msg"] == "no quaternion report"
    assert imu.diag["gyro_exception_type"] == "RuntimeError"
    assert imu.diag["gyro_exception_msg"] == "no gyro report"


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
