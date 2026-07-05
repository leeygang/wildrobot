from __future__ import annotations

import contextlib
import io
import os
import traceback
from dataclasses import replace
from queue import Queue
from threading import Thread
from typing import Optional

import numpy as np
from .imu import Imu, ImuSample


_INPUT_SENSOR_REPORT_CHANNEL = 3
_IGNORABLE_UNKNOWN_SENSOR_REPORT_IDS = {0x7B}
_MAX_GYRO_INTEGRATION_DT_S = 0.25


def _imu_payload_changed(previous: ImuSample, quat_xyzw: np.ndarray, gyro_rad_s: np.ndarray) -> bool:
    if previous.timestamp_s is None:
        return True
    prev_quat = np.asarray(previous.quat_xyzw, dtype=np.float32)
    prev_gyro = np.asarray(previous.gyro_rad_s, dtype=np.float32)
    next_quat = np.asarray(quat_xyzw, dtype=np.float32)
    next_gyro = np.asarray(gyro_rad_s, dtype=np.float32)
    return not (
        np.array_equal(prev_quat, next_quat)
        and np.array_equal(prev_gyro, next_gyro)
    )


def _quat_multiply_xyzw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = [float(v) for v in a]
    bx, by, bz, bw = [float(v) for v in b]
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=np.float32,
    )


def _integrate_quat_xyzw_body_gyro(
    quat_xyzw: np.ndarray,
    gyro_rad_s: np.ndarray,
    dt_s: float,
) -> np.ndarray:
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float32).reshape(4)
    gyro_rad_s = np.asarray(gyro_rad_s, dtype=np.float32).reshape(3)
    dt_s = min(max(float(dt_s), 0.0), _MAX_GYRO_INTEGRATION_DT_S)

    omega_norm = float(np.linalg.norm(gyro_rad_s))
    if not np.isfinite(omega_norm) or omega_norm <= 1e-9 or dt_s <= 0.0:
        return quat_xyzw.copy()

    angle = omega_norm * dt_s
    axis = gyro_rad_s / omega_norm
    half_angle = 0.5 * angle
    sin_half = float(np.sin(half_angle))
    delta = np.array(
        [
            float(axis[0]) * sin_half,
            float(axis[1]) * sin_half,
            float(axis[2]) * sin_half,
            float(np.cos(half_angle)),
        ],
        dtype=np.float32,
    )
    integrated = _quat_multiply_xyzw(quat_xyzw, delta)
    norm = float(np.linalg.norm(integrated))
    if np.isfinite(norm) and norm > 1e-6:
        return (integrated / norm).astype(np.float32)
    return quat_xyzw.copy()


def _record_diag_exception(diag: dict[str, object], prefix: str, exc: BaseException) -> None:
    diag[f"{prefix}_exception"] = True
    diag[f"{prefix}_exception_type"] = type(exc).__name__
    diag[f"{prefix}_exception_msg"] = str(exc)


def _input_report_sequence(imu) -> Optional[int]:
    seq = getattr(imu, "_sequence_number", None)
    if not isinstance(seq, list) or len(seq) <= _INPUT_SENSOR_REPORT_CHANNEL:
        return None
    try:
        return int(seq[_INPUT_SENSOR_REPORT_CHANNEL])
    except Exception:
        return None


def _is_ignorable_unknown_sensor_report(exc: BaseException) -> bool:
    return (
        isinstance(exc, KeyError)
        and len(getattr(exc, "args", ())) == 1
        and exc.args[0] in _IGNORABLE_UNKNOWN_SENSOR_REPORT_IDS
    )


def _enable_feature_allowing_unknown_reports(imu, *args, max_unknown_reports: int = 5, **kwargs) -> None:
    last_exc: BaseException | None = None
    for _ in range(max(1, int(max_unknown_reports))):
        try:
            imu.enable_feature(*args, **kwargs)
            return
        except Exception as exc:
            if not _is_ignorable_unknown_sensor_report(exc):
                raise
            last_exc = exc
    if last_exc is not None:
        raise last_exc


def _resolve_board_pin(board_module, pin_name: str):
    name = str(pin_name).strip()
    if name.lower().startswith("board."):
        name = name.split(".", 1)[1]
    upper_name = name.upper()
    if upper_name.startswith("GPIO") and upper_name[4:].isdigit():
        name = f"D{upper_name[4:]}"
    try:
        return getattr(board_module, name)
    except AttributeError as exc:
        raise ValueError(f"Unknown Blinka board pin for BNO085 SPI: {pin_name!r}") from exc


def _format_init_failure_detail(
    *,
    transport: str,
    i2c_address: int,
    i2c_frequency_hz: int,
    spi_baudrate: int,
    spi_read_skip_bytes: int,
    spi_cs_pin: str,
    spi_int_pin: str,
    spi_reset_pin: str,
    last_exc: BaseException | None,
) -> str:
    if transport == "spi":
        detail = (
            f"spi_baudrate={spi_baudrate} "
            f"spi_read_skip_bytes={spi_read_skip_bytes} cs={spi_cs_pin} "
            f"int={spi_int_pin} reset={spi_reset_pin}"
        )
        if isinstance(last_exc, IndexError):
            detail += (
                "; SPI reset read returned a corrupt SHTP header "
                "(check MISO/MOSI, CS, INT, RST, PS0/PS1, and common ground)"
            )
        return detail
    return f"address=0x{i2c_address:02X} freq={i2c_frequency_hz}Hz"


def _make_bno08x_spi_read_skip_class(base_cls):
    packet_cls = base_cls._read_packet.__globals__["Packet"]
    packet_error_cls = base_cls._read_packet.__globals__["PacketError"]

    # Some Pi/BNO08x breakout combinations return a fixed leading preamble before SHTP bytes.
    class _BNO08XSPIReadSkip(base_cls):
        def __init__(self, *args, read_skip_bytes: int = 0, **kwargs):
            self._wr_spi_read_skip_bytes = max(0, int(read_skip_bytes))
            super().__init__(*args, **kwargs)

        def _readinto_after_skip(
            self,
            spi,
            buf,
            *,
            start: int = 0,
            end=None,
            write_value: int = 0x00,
        ) -> None:
            if end is None:
                end = len(buf)
            start = int(start)
            end = int(end)
            read_len = max(0, end - start)
            skip = int(self._wr_spi_read_skip_bytes)
            if skip <= 0:
                spi.readinto(buf, start=start, end=end, write_value=write_value)
                return

            raw = bytearray(skip + read_len)
            spi.readinto(raw, end=len(raw), write_value=write_value)
            buf[start:end] = raw[skip:]

        def _read_into(self, buf, start=0, end=None):
            self._wait_for_int()

            with self._spi as spi:
                self._readinto_after_skip(spi, buf, start=start, end=end, write_value=0x00)

        def _read_header(self):
            self._wait_for_int()

            with self._spi as spi:
                self._readinto_after_skip(spi, self._data_buffer, start=0, end=4, write_value=0x00)
            self._dbg("")
            self._dbg("SHTP READ packet header: ", [hex(x) for x in self._data_buffer[0:4]])

        def _read_packet(self):
            self._wait_for_int()

            skip = int(self._wr_spi_read_skip_bytes)
            with self._spi as spi:
                raw_header = bytearray(skip + 4)
                spi.readinto(raw_header, end=len(raw_header), write_value=0x00)
                self._data_buffer[0:4] = raw_header[skip:]
                self._dbg("")
                self._dbg("SHTP READ packet header: ", [hex(x) for x in self._data_buffer[0:4]])
                if self._debug:
                    print([hex(x) for x in self._data_buffer[0:4]])

                halfpacket = bool(self._data_buffer[1] & 0x80)
                header = packet_cls.header_from_buffer(self._data_buffer)
                packet_byte_count = header.packet_byte_count
                channel_number = header.channel_number
                sequence_number = header.sequence_number

                self._sequence_number[channel_number] = sequence_number
                if packet_byte_count == 0:
                    raise packet_error_cls("No packet available")

                self._dbg("channel %d has %d bytes available" % (channel_number, packet_byte_count - 4))

                if packet_byte_count > len(self._data_buffer):
                    header_bytes = bytes(self._data_buffer[0:4])
                    self._data_buffer = bytearray(packet_byte_count)
                    self._data_buffer[0:4] = header_bytes

                if packet_byte_count > 4:
                    spi.readinto(
                        self._data_buffer,
                        start=4,
                        end=packet_byte_count,
                        write_value=0x00,
                    )

            if halfpacket:
                raise packet_error_cls("read partial packet")
            new_packet = packet_cls(self._data_buffer)
            if self._debug:
                print(new_packet)
            self._update_sequence_number(new_packet)
            return new_packet

        def _read(self, requested_read_length):
            self._dbg("trying to read", requested_read_length, "bytes")
            unread_bytes = 0
            total_read_length = requested_read_length + 4
            if total_read_length > len(self._data_buffer):
                unread_bytes = total_read_length - len(self._data_buffer)
                total_read_length = len(self._data_buffer)

            with self._spi as spi:
                self._readinto_after_skip(
                    spi,
                    self._data_buffer,
                    start=0,
                    end=total_read_length,
                    write_value=0x00,
                )
            return unread_bytes > 0

    return _BNO08XSPIReadSkip


class BNO085IMU(Imu):
    """BNO085 IMU wrapper.

    Designed to be compatible with the observation layout used in training:
    - quaternion (xyzw)
    - gyro (rad/s)

    read() returns body-frame quat/gyro with upside_down and axis_map already applied.

    Runs a tiny background thread to keep latest reading available.
    """

    def __init__(
        self,
        i2c_address: int = 0x4A,
        upside_down: bool = False,
        sampling_hz: int = 50,
        *,
        transport: str = "i2c",
        axis_map: Optional[list[str]] = None,
        suppress_debug: bool = True,
        max_quat_norm_deviation: float = 0.1,
        i2c_frequency_hz: int = 100000,
        spi_baudrate: int = 1_000_000,
        spi_read_skip_bytes: int = 0,
        spi_cs_pin: str = "D8",
        spi_int_pin: str = "D17",
        spi_reset_pin: str = "D27",
        init_retries: int = 3,
        polling_mode: bool = False,
        enable_rotation_vector: bool = True,
    ):
        self.transport = str(transport).strip().lower()
        if self.transport not in {"i2c", "spi"}:
            raise ValueError("BNO085IMU transport must be 'i2c' or 'spi'")
        self.i2c_address = int(i2c_address)
        self.upside_down = bool(upside_down)
        self.sampling_hz = int(sampling_hz)
        self.max_quat_norm_deviation = float(max_quat_norm_deviation)
        self.suppress_debug = bool(suppress_debug)
        self.i2c_frequency_hz = int(i2c_frequency_hz)
        self.spi_baudrate = int(spi_baudrate)
        self.spi_read_skip_bytes = max(0, int(spi_read_skip_bytes))
        self.spi_cs_pin = str(spi_cs_pin)
        self.spi_int_pin = str(spi_int_pin)
        self.spi_reset_pin = str(spi_reset_pin)
        self.init_retries = int(init_retries)
        self.polling_mode = bool(polling_mode)
        self.enable_rotation_vector = bool(enable_rotation_vector)
        self._axis_map = list(axis_map) if axis_map is not None else None
        self._r_bs = _axis_map_to_r_bs(self._axis_map) if self._axis_map is not None else None

        # Lazy imports so this module can be imported on non-RPi machines.
        import board
        import busio
        import time
        from adafruit_bno08x import BNO_REPORT_GYROSCOPE, BNO_REPORT_ROTATION_VECTOR
        from adafruit_bno08x.i2c import BNO08X_I2C

        try:
            # Newer adafruit_bno08x exposes GAME_ROTATION_VECTOR (mag-free quaternion).
            from adafruit_bno08x import BNO_REPORT_GAME_ROTATION_VECTOR as _GAME_QUAT_REPORT  # type: ignore
        except Exception:
            _GAME_QUAT_REPORT = None
        self._use_game_quat = _GAME_QUAT_REPORT is not None

        def _enable_feature_best_effort(imu, report_id: int) -> None:
            # Different adafruit_bno08x versions accept different arguments for report interval.
            # Try a couple of common patterns; fall back to default interval.
            interval_us = int(1_000_000 / max(1, self.sampling_hz))
            for kwargs in (
                {"report_interval_us": interval_us},
                {"report_interval": interval_us},
                {"report_interval_ms": max(1, int(1000 / max(1, self.sampling_hz)))},
            ):
                try:
                    _enable_feature_allowing_unknown_reports(imu, report_id, **kwargs)
                    return
                except TypeError:
                    pass
            for args in ((report_id, interval_us), (report_id,)):
                try:
                    _enable_feature_allowing_unknown_reports(imu, *args)
                    return
                except TypeError:
                    pass
            _enable_feature_allowing_unknown_reports(imu, report_id)

        # The Adafruit driver can throw during enable_feature() when it drains
        # stale/unknown packets from the input report channel. We retry by
        # reinitializing the bus+device if the per-feature retry above cannot clear it.
        last_exc: Exception | None = None
        self._bus = None
        self._digital_pins = ()
        self._imu = None  # type: ignore[assignment]
        for attempt in range(max(1, self.init_retries)):
            bus = None
            digital_pins = ()
            try:
                with self._maybe_silence_debug_output():
                    if self.transport == "spi":
                        import digitalio
                        from adafruit_bno08x.spi import BNO08X_SPI

                        bus = busio.SPI(board.SCK, MOSI=board.MOSI, MISO=board.MISO)
                        cs = digitalio.DigitalInOut(_resolve_board_pin(board, self.spi_cs_pin))
                        int_pin = digitalio.DigitalInOut(_resolve_board_pin(board, self.spi_int_pin))
                        reset = digitalio.DigitalInOut(_resolve_board_pin(board, self.spi_reset_pin))
                        digital_pins = (cs, int_pin, reset)
                        spi_cls = (
                            _make_bno08x_spi_read_skip_class(BNO08X_SPI)
                            if self.spi_read_skip_bytes > 0
                            else BNO08X_SPI
                        )
                        kwargs = {
                            "baudrate": self.spi_baudrate,
                            "debug": not self.suppress_debug,
                        }
                        if self.spi_read_skip_bytes > 0:
                            kwargs["read_skip_bytes"] = self.spi_read_skip_bytes
                        imu = spi_cls(
                            bus,
                            cs,
                            int_pin,
                            reset,
                            **kwargs,
                        )
                    else:
                        bus = busio.I2C(board.SCL, board.SDA, frequency=self.i2c_frequency_hz)
                        imu = BNO08X_I2C(bus, address=self.i2c_address)
                    if _GAME_QUAT_REPORT is not None:
                        _enable_feature_best_effort(imu, _GAME_QUAT_REPORT)
                    if self.enable_rotation_vector or _GAME_QUAT_REPORT is None:
                        _enable_feature_best_effort(imu, BNO_REPORT_ROTATION_VECTOR)
                    _enable_feature_best_effort(imu, BNO_REPORT_GYROSCOPE)
                self._bus = bus
                self._digital_pins = digital_pins
                self._imu = imu
                time.sleep(0.05)
                break
            except Exception as exc:
                last_exc = exc
                self._deinit_transport(bus, digital_pins)
                time.sleep(0.2 * (attempt + 1))
        if self._imu is None:
            detail = _format_init_failure_detail(
                transport=self.transport,
                i2c_address=self.i2c_address,
                i2c_frequency_hz=self.i2c_frequency_hz,
                spi_baudrate=self.spi_baudrate,
                spi_read_skip_bytes=self.spi_read_skip_bytes,
                spi_cs_pin=self.spi_cs_pin,
                spi_int_pin=self.spi_int_pin,
                spi_reset_pin=self.spi_reset_pin,
                last_exc=last_exc,
            )
            raise RuntimeError(
                "Failed to initialize BNO085 IMU (enable_feature failed). "
                "Check IMU power, wiring, protocol-select pins, and runtime transport config. "
                f"transport={self.transport} {detail} retries={self.init_retries}"
            ) from last_exc
        if self.suppress_debug:
            self._try_disable_debug()

        self._latest: ImuSample = ImuSample(
            quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            gyro_rad_s=np.zeros(3, dtype=np.float32),
            timestamp_s=None,
            valid=False,
            fresh=False,
        )
        self._last_report_sample = self._latest

        self._q: Optional[Queue[ImuSample]] = None
        self._running = False
        self._worker: Optional[Thread] = None
        self._error_count = 0
        self._last_error: Optional[str] = None
        self._last_traceback: Optional[str] = None
        self._diag: dict[str, object] = {}
        if not self.polling_mode:
            self._q = Queue(maxsize=1)
            self._running = True
            self._worker = Thread(target=self._loop, daemon=True)
            self._worker.start()

    @property
    def error_count(self) -> int:
        return int(self._error_count)

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    @property
    def last_traceback(self) -> Optional[str]:
        return self._last_traceback

    @property
    def diag(self) -> dict[str, object]:
        # Shallow copy for safe external printing.
        return dict(self._diag)

    def close(self) -> None:
        try:
            self._running = False
            if self._worker is not None:
                self._worker.join(timeout=1.0)
        except Exception:
            pass
        try:
            self._deinit_transport(getattr(self, "_bus", None), getattr(self, "_digital_pins", ()))
        except Exception:
            pass

    @staticmethod
    def _deinit_transport(bus, digital_pins=()) -> None:
        for pin in digital_pins or ():
            try:
                pin.deinit()
            except Exception:
                pass
        try:
            if bus is not None:
                bus.deinit()
        except Exception:
            pass

    def read(self) -> ImuSample:
        if self.polling_mode:
            self._latest = self._read_sample_once()
            return self._latest

        # Drain queue to keep freshest sample
        if self._q is None:
            return replace(self._latest, fresh=False)
        drained = False
        try:
            while True:
                self._latest = self._q.get_nowait()
                drained = True
        except Exception:
            pass
        if not drained:
            self._diag = {
                **self._diag,
                "payload_status": "stale",
                "read_status": "cached",
            }
            return replace(self._latest, fresh=False)
        return self._latest

    def _try_disable_debug(self) -> None:
        """Best-effort: disable noisy debug prints from the Adafruit BNO08X stack."""
        for attr in ("debug", "_debug", "DEBUG", "_DEBUG"):
            try:
                if hasattr(self._imu, attr):
                    setattr(self._imu, attr, False)
            except Exception:
                pass

        try:
            import adafruit_bno08x.adafruit_bno08x as core  # type: ignore

            for attr in ("debug", "_debug", "DEBUG", "_DEBUG"):
                try:
                    if hasattr(core, attr):
                        setattr(core, attr, False)
                except Exception:
                    pass
        except Exception:
            pass

    @contextlib.contextmanager
    def _maybe_silence_debug_output(self):
        """Best-effort suppression for chatty library debug prints.

        Some BNO08X stacks print verbose packet dumps directly to stdout/stderr when
        debug is enabled internally. We treat this as a hard runtime concern (timing/log noise).
        """
        if not self.suppress_debug:
            yield
            return

        sink = io.StringIO()
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            yield

    def _read_sample_once(self) -> ImuSample:
        import time

        now_s = time.monotonic()
        diag: dict[str, object] = {"quat_source": None, "quat_exception": False, "gyro_exception": False}
        seq_before = _input_report_sequence(self._imu)
        with self._maybe_silence_debug_output():
            quat = None
            gyro = None

            # The Adafruit driver raises RuntimeError if a report isn't available yet.
            # Treat that as an invalid sample rather than crashing the reader loop.
            try:
                if self._use_game_quat:
                    quat = self._imu.game_quaternion
                    diag["quat_source"] = "game_quaternion"
            except Exception as exc:
                _record_diag_exception(diag, "quat", exc)
                quat = None

            if quat is None and (self.enable_rotation_vector or not self._use_game_quat):
                try:
                    quat = self._imu.quaternion
                    diag["quat_source"] = "quaternion"
                except Exception as exc:
                    _record_diag_exception(diag, "quat", exc)
                    quat = None

            try:
                gyro = self._imu.gyro
            except Exception as exc:
                _record_diag_exception(diag, "gyro", exc)
                gyro = None
        seq_after = _input_report_sequence(self._imu)
        seq_available = seq_before is not None and seq_after is not None
        input_report_advanced = bool(seq_available and seq_after != seq_before)
        diag["input_report_seq"] = seq_after
        diag["input_report_advanced"] = input_report_advanced if seq_available else None

        if quat is None:
            quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            quat_valid = False
            diag["quat_status"] = "missing"
        else:
            # adafruit_bno08x returns quaternion as (i, j, k, real)
            quat_xyzw = np.array([quat[0], quat[1], quat[2], quat[3]], dtype=np.float32)
            quat_valid = True
            diag["quat_status"] = "raw"

        if gyro is None:
            gyro_rad_s = np.zeros(3, dtype=np.float32)
            gyro_valid = False
            diag["gyro_status"] = "missing"
        else:
            gyro_rad_s = np.array(gyro, dtype=np.float32)
            gyro_valid = bool(gyro_rad_s.shape == (3,) and np.all(np.isfinite(gyro_rad_s)))
            diag["gyro_status"] = "raw" if gyro_valid else "bad"

        raw_norm = float(np.linalg.norm(quat_xyzw))
        diag["quat_norm"] = raw_norm

        if quat_valid:
            if (
                np.isfinite(raw_norm)
                and raw_norm > 1e-6
                and abs(raw_norm - 1.0) <= self.max_quat_norm_deviation
            ):
                quat_xyzw = (quat_xyzw / raw_norm).astype(np.float32)
                diag["quat_status"] = "normalized"
            else:
                quat_valid = False
                diag["quat_status"] = "bad_norm"

        if self.upside_down:
            if quat_valid:
                quat_xyzw = np.array(
                    [quat_xyzw[0], -quat_xyzw[1], -quat_xyzw[2], quat_xyzw[3]],
                    dtype=np.float32,
                )
            if gyro_valid:
                gyro_rad_s = np.array(
                    [gyro_rad_s[0], -gyro_rad_s[1], -gyro_rad_s[2]],
                    dtype=np.float32,
                )

        if self._r_bs is not None:
            if gyro_valid:
                gyro_rad_s = (self._r_bs @ gyro_rad_s.reshape(3, 1)).reshape(3).astype(np.float32)
            if quat_valid:
                r_ws = _quat_xyzw_to_rotmat(quat_xyzw)
                r_wb = r_ws @ self._r_bs.T
                quat_xyzw = _rotmat_to_quat_xyzw(r_wb)
                mapped_norm = float(np.linalg.norm(quat_xyzw))
                if np.isfinite(mapped_norm) and mapped_norm > 1e-6:
                    quat_xyzw = (quat_xyzw / mapped_norm).astype(np.float32)
                else:
                    quat_valid = False
                    diag["quat_status"] = "bad_axis_map_norm"

        last_report_sample = getattr(self, "_last_report_sample", self._latest)
        quat_integrated = False
        if quat_valid:
            quat_out = quat_xyzw
        elif (
            gyro_valid
            and bool(getattr(last_report_sample, "valid", False))
            and last_report_sample.timestamp_s is not None
        ):
            dt_s = max(0.0, now_s - float(last_report_sample.timestamp_s))
            if dt_s <= _MAX_GYRO_INTEGRATION_DT_S:
                quat_out = _integrate_quat_xyzw_body_gyro(
                    np.asarray(last_report_sample.quat_xyzw, dtype=np.float32),
                    gyro_rad_s,
                    dt_s,
                )
                quat_integrated = True
                diag["quat_status"] = f"integrated_from_gyro_after_{diag['quat_status']}"
                diag["quat_integration_dt_s"] = dt_s
            else:
                quat_out = last_report_sample.quat_xyzw
                diag["quat_integration_skipped_dt_s"] = dt_s
        else:
            quat_out = last_report_sample.quat_xyzw
        valid = bool(gyro_valid and (quat_valid or quat_integrated))
        payload_changed = _imu_payload_changed(last_report_sample, quat_out, gyro_rad_s)
        report_fresh = (
            valid
            and (
                last_report_sample.timestamp_s is None
                or input_report_advanced
                or (not seq_available and payload_changed)
            )
        )
        diag["payload_changed"] = payload_changed
        if valid and not report_fresh:
            diag["payload_status"] = "stale"
        else:
            diag["payload_status"] = "fresh" if valid else "invalid"

        self._diag = diag
        sample = ImuSample(
            quat_xyzw=quat_out,
            gyro_rad_s=gyro_rad_s,
            timestamp_s=now_s if report_fresh else last_report_sample.timestamp_s,
            valid=bool(valid),
            fresh=bool(report_fresh),
        )
        if report_fresh:
            self._last_report_sample = sample
        return sample

    def _loop(self) -> None:
        import time

        period = 1.0 / max(1, self.sampling_hz)
        while self._running:
            try:
                sample = self._read_sample_once()
                try:
                    if self._q is not None:
                        if self._q.full():
                            self._q.get_nowait()
                        self._q.put_nowait(sample)
                except Exception:
                    pass
            except Exception:
                # Keep running; hardware IO can be flaky at boot.
                # Surface the failure to callers by emitting an invalid sample instead of
                # leaving a stale "valid=True" reading in place.
                try:
                    self._error_count += 1
                    self._last_error = "BNO085IMU._read_sample_once failed"
                    self._last_traceback = traceback.format_exc()
                    if os.environ.get("WR_BNO085_DEBUG_EXCEPTIONS", "").strip() not in ("", "0", "false", "False"):
                        print(self._last_traceback, flush=True)
                    sample = ImuSample(
                        quat_xyzw=self._latest.quat_xyzw,
                        gyro_rad_s=self._latest.gyro_rad_s,
                        timestamp_s=time.monotonic(),
                        valid=False,
                        fresh=False,
                    )
                    if self._q is not None:
                        if self._q.full():
                            self._q.get_nowait()
                        self._q.put_nowait(sample)
                except Exception:
                    pass

            time.sleep(period)


def _axis_map_to_r_bs(axis_map: Optional[list[str]]) -> np.ndarray:
    """Convert axis_map strings into a 3x3 matrix R_bs (body <- sensor)."""
    if axis_map is None:
        raise ValueError("axis_map is None")
    if len(axis_map) != 3:
        raise ValueError("axis_map must have length 3")

    axis_to_vec = {
        "X": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "Y": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "Z": np.array([0.0, 0.0, 1.0], dtype=np.float32),
    }

    rows = []
    used = set()
    for entry in axis_map:
        s = str(entry).strip().upper()
        if len(s) != 2 or s[0] not in {"+", "-"} or s[1] not in {"X", "Y", "Z"}:
            raise ValueError(f"Invalid axis_map entry: {entry!r} (expected '+X', '-Y', etc)")
        sign = 1.0 if s[0] == "+" else -1.0
        axis = s[1]
        if axis in used:
            raise ValueError(f"axis_map uses axis '{axis}' more than once: {axis_map}")
        used.add(axis)
        rows.append(sign * axis_to_vec[axis])

    r_bs = np.stack(rows, axis=0).astype(np.float32)
    det = float(np.linalg.det(r_bs))
    if abs(det - 1.0) > 1e-3:
        raise ValueError(
            "axis_map must be a right-handed rotation with determinant +1; "
            f"det={det:.3f} axis_map={axis_map}. Flip two axes, not one."
        )
    return r_bs


def _quat_xyzw_to_rotmat(quat_xyzw: np.ndarray) -> np.ndarray:
    """Quaternion (xyzw) to rotation matrix R (sensor/body -> world)."""
    x, y, z, w = [float(v) for v in quat_xyzw]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def _rotmat_to_quat_xyzw(r: np.ndarray) -> np.ndarray:
    """Rotation matrix to quaternion (xyzw)."""
    r = np.asarray(r, dtype=np.float32).reshape(3, 3)
    t = float(np.trace(r))
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    else:
        if r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
            s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
            w = (r[2, 1] - r[1, 2]) / s
            x = 0.25 * s
            y = (r[0, 1] + r[1, 0]) / s
            z = (r[0, 2] + r[2, 0]) / s
        elif r[1, 1] > r[2, 2]:
            s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
            w = (r[0, 2] - r[2, 0]) / s
            x = (r[0, 1] + r[1, 0]) / s
            y = 0.25 * s
            z = (r[1, 2] + r[2, 1]) / s
        else:
            s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
            w = (r[1, 0] - r[0, 1]) / s
            x = (r[0, 2] + r[2, 0]) / s
            y = (r[1, 2] + r[2, 1]) / s
            z = 0.25 * s
    return np.array([x, y, z, w], dtype=np.float32)
