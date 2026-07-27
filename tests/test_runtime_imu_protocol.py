import numpy as np
from typing import runtime_checkable

from runtime.wr_runtime.hardware.imu import Imu, ImuSample


class DummyImu:
    def __init__(self) -> None:
        self._called = 0

    def read(self) -> ImuSample:
        self._called += 1
        return ImuSample(
            quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            gyro_rad_s=np.zeros(3, dtype=np.float32),
            timestamp_s=0.0,
            valid=True,
        )

    def close(self) -> None:
        self._called += 1


def test_dummy_imu_satisfies_protocol() -> None:
    imu: Imu = DummyImu()
    sample = imu.read()
    assert sample.quat_wxyz.shape == (4,)
    assert sample.gyro_rad_s.shape == (3,)
    assert sample.valid is True
    assert sample.fresh is True


class InvalidImu:
    def read(self) -> ImuSample:
        return ImuSample(
            quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            gyro_rad_s=np.zeros(3, dtype=np.float32),
            timestamp_s=None,
            valid=False,
        )

    def close(self) -> None:
        pass


def test_invalid_flag_propagates() -> None:
    imu: Imu = InvalidImu()
    sample = imu.read()
    assert sample.valid is False
    assert sample.fresh is True
    assert sample.timestamp_s is None
