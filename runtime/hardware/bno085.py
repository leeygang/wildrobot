from __future__ import annotations

from dataclasses import dataclass
from queue import Queue
from threading import Thread
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class ImuSample:
    quat_xyzw: np.ndarray  # (4,) float32
    gyro_rad_s: np.ndarray  # (3,) float32


class BNO085IMU:
    """BNO085 IMU wrapper.

    Designed to be compatible with the observation layout used in training:
    - quaternion (xyzw)
    - gyro (rad/s)

    Runs a tiny background thread to keep latest reading available.
    """

    def __init__(self, i2c_address: int = 0x4A, upside_down: bool = False, sampling_hz: int = 50):
        self.i2c_address = int(i2c_address)
        self.upside_down = bool(upside_down)
        self.sampling_hz = int(sampling_hz)

        # Lazy imports so this module can be imported on non-RPi machines.
        import board
        import busio
        from adafruit_bno08x import (
            BNO_REPORT_ROTATION_VECTOR,
            BNO_REPORT_GYROSCOPE,
        )
        from adafruit_bno08x.i2c import BNO08X_I2C

        i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)
        self._imu = BNO08X_I2C(i2c, address=self.i2c_address)
        self._imu.enable_feature(BNO_REPORT_ROTATION_VECTOR)
        self._imu.enable_feature(BNO_REPORT_GYROSCOPE)

        self._latest: ImuSample = ImuSample(
            quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            gyro_rad_s=np.zeros(3, dtype=np.float32),
        )

        self._q: Queue[ImuSample] = Queue(maxsize=1)
        self._running = True
        self._worker = Thread(target=self._loop, daemon=True)
        self._worker.start()

    def close(self) -> None:
        self._running = False
        try:
            self._worker.join(timeout=1.0)
        except Exception:
            pass

    def read(self) -> ImuSample:
        # Drain queue to keep freshest sample
        try:
            while True:
                self._latest = self._q.get_nowait()
        except Exception:
            pass
        return self._latest

    def _loop(self) -> None:
        import time

        period = 1.0 / max(1, self.sampling_hz)
        while self._running:
            try:
                quat = self._imu.quaternion
                gyro = self._imu.gyro

                if quat is None:
                    quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
                else:
                    # adafruit_bno08x returns quaternion as (i, j, k, real)
                    quat_xyzw = np.array([quat[0], quat[1], quat[2], quat[3]], dtype=np.float32)

                if gyro is None:
                    gyro_rad_s = np.zeros(3, dtype=np.float32)
                else:
                    gyro_rad_s = np.array(gyro, dtype=np.float32)

                if self.upside_down:
                    # Simple axis flip for upside-down mounting.
                    # If you need a different remap, do it here.
                    quat_xyzw = np.array([quat_xyzw[0], -quat_xyzw[1], -quat_xyzw[2], quat_xyzw[3]], dtype=np.float32)
                    gyro_rad_s = np.array([gyro_rad_s[0], -gyro_rad_s[1], -gyro_rad_s[2]], dtype=np.float32)

                sample = ImuSample(quat_xyzw=quat_xyzw, gyro_rad_s=gyro_rad_s)

                try:
                    if self._q.full():
                        self._q.get_nowait()
                    self._q.put_nowait(sample)
                except Exception:
                    pass
            except Exception:
                # Keep running; hardware IO can be flaky at boot.
                pass

            time.sleep(period)
