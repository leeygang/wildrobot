from __future__ import annotations

import contextlib
import io
import os
import traceback
from queue import Queue
from threading import Thread
from typing import Optional

import numpy as np
from .imu import Imu, ImuSample


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
        axis_map: Optional[list[str]] = None,
        suppress_debug: bool = True,
        max_quat_norm_deviation: float = 0.1,
        i2c_frequency_hz: int = 100000,
        init_retries: int = 3,
        polling_mode: bool = False,
    ):
        self.i2c_address = int(i2c_address)
        self.upside_down = bool(upside_down)
        self.sampling_hz = int(sampling_hz)
        self.max_quat_norm_deviation = float(max_quat_norm_deviation)
        self.suppress_debug = bool(suppress_debug)
        self.i2c_frequency_hz = int(i2c_frequency_hz)
        self.init_retries = int(init_retries)
        self.polling_mode = bool(polling_mode)
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
                    imu.enable_feature(report_id, **kwargs)
                    return
                except TypeError:
                    pass
            for args in ((report_id, interval_us), (report_id,)):
                try:
                    imu.enable_feature(*args)
                    return
                except TypeError:
                    pass
            imu.enable_feature(report_id)

        # The Adafruit driver occasionally throws IndexError during enable_feature() if the
        # I2C stream returns a corrupted/partial packet (common on marginal wiring/power).
        # We retry by reinitializing the bus+device.
        last_exc: Exception | None = None
        self._i2c = None
        self._imu = None  # type: ignore[assignment]
        for attempt in range(max(1, self.init_retries)):
            try:
                i2c = busio.I2C(board.SCL, board.SDA, frequency=self.i2c_frequency_hz)
                imu = BNO08X_I2C(i2c, address=self.i2c_address)
                with self._maybe_silence_debug_output():
                    # Always enable ROTATION_VECTOR because the library's `.quaternion` accessor
                    # expects that report. If GAME_ROTATION_VECTOR is available, enable it too
                    # and prefer `.game_quaternion` at read time (mag-free, typically more robust).
                    _enable_feature_best_effort(imu, BNO_REPORT_ROTATION_VECTOR)
                    if _GAME_QUAT_REPORT is not None:
                        _enable_feature_best_effort(imu, _GAME_QUAT_REPORT)
                    _enable_feature_best_effort(imu, BNO_REPORT_GYROSCOPE)
                self._i2c = i2c
                self._imu = imu
                time.sleep(0.05)
                break
            except Exception as exc:
                last_exc = exc
                try:
                    i2c.deinit()
                except Exception:
                    pass
                time.sleep(0.2 * (attempt + 1))
        if self._imu is None:
            raise RuntimeError(
                "Failed to initialize BNO085 IMU (enable_feature failed). "
                "This often indicates I2C noise/power issues or an incompatible address. "
                f"address=0x{self.i2c_address:02X} freq={self.i2c_frequency_hz}Hz retries={self.init_retries}"
            ) from last_exc
        if self.suppress_debug:
            self._try_disable_debug()

        self._latest: ImuSample = ImuSample(
            quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            gyro_rad_s=np.zeros(3, dtype=np.float32),
            timestamp_s=None,
            valid=False,
        )

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
            if getattr(self, "_i2c", None) is not None:
                self._i2c.deinit()  # type: ignore[union-attr]
        except Exception:
            pass

    def read(self) -> ImuSample:
        if self.polling_mode:
            self._latest = self._read_sample_once()
            return self._latest

        # Drain queue to keep freshest sample
        if self._q is None:
            return self._latest
        try:
            while True:
                self._latest = self._q.get_nowait()
        except Exception:
            pass
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

        valid = True
        diag: dict[str, object] = {"quat_source": None, "quat_exception": False, "gyro_exception": False}
        with self._maybe_silence_debug_output():
            quat = None
            gyro = None

            # The Adafruit driver raises RuntimeError if a report isn't available yet.
            # Treat that as an invalid sample rather than crashing the reader loop.
            try:
                if self._use_game_quat and hasattr(self._imu, "game_quaternion"):
                    quat = self._imu.game_quaternion
                    diag["quat_source"] = "game_quaternion"
            except Exception:
                diag["quat_exception"] = True
                quat = None

            if quat is None:
                try:
                    quat = self._imu.quaternion
                    diag["quat_source"] = "quaternion"
                except Exception:
                    diag["quat_exception"] = True
                    quat = None

            try:
                gyro = self._imu.gyro
            except Exception:
                diag["gyro_exception"] = True
                gyro = None

        if quat is None:
            quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            valid = False
            diag["quat_status"] = "missing"
        else:
            # adafruit_bno08x returns quaternion as (i, j, k, real)
            quat_xyzw = np.array([quat[0], quat[1], quat[2], quat[3]], dtype=np.float32)
            diag["quat_status"] = "raw"

        if gyro is None:
            gyro_rad_s = np.zeros(3, dtype=np.float32)
            valid = False
            diag["gyro_status"] = "missing"
        else:
            gyro_rad_s = np.array(gyro, dtype=np.float32)
            diag["gyro_status"] = "raw"

        raw_norm = float(np.linalg.norm(quat_xyzw))
        diag["quat_norm"] = raw_norm

        # Always renormalize quaternions if they are finite and non-zero.
        # Some BNO08X report paths (or library versions) can yield non-unit quats.
        if np.isfinite(raw_norm) and raw_norm > 1e-6:
            quat_xyzw = (quat_xyzw / raw_norm).astype(np.float32)
            diag["quat_status"] = "normalized"
        else:
            valid = False
            diag["quat_status"] = "bad_norm"

        if self.upside_down:
            quat_xyzw = np.array([quat_xyzw[0], -quat_xyzw[1], -quat_xyzw[2], quat_xyzw[3]], dtype=np.float32)
            gyro_rad_s = np.array([gyro_rad_s[0], -gyro_rad_s[1], -gyro_rad_s[2]], dtype=np.float32)

        if self._r_bs is not None:
            gyro_rad_s = (self._r_bs @ gyro_rad_s.reshape(3, 1)).reshape(3).astype(np.float32)
            r_ws = _quat_xyzw_to_rotmat(quat_xyzw)
            r_wb = r_ws @ self._r_bs.T
            quat_xyzw = _rotmat_to_quat_xyzw(r_wb)

        quat_out = quat_xyzw if valid else self._latest.quat_xyzw

        self._diag = diag
        return ImuSample(
            quat_xyzw=quat_out,
            gyro_rad_s=gyro_rad_s,
            timestamp_s=time.monotonic(),
            valid=bool(valid),
        )

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
    if abs(abs(det) - 1.0) > 1e-3:
        raise ValueError(f"axis_map must form an orthonormal basis; det={det:.3f} axis_map={axis_map}")
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
