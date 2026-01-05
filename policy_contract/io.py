from __future__ import annotations

from typing import Protocol, TypeVar

SignalsT = TypeVar("SignalsT")


class SignalsProvider(Protocol[SignalsT]):
    def read(self, *args, **kwargs) -> SignalsT:
        ...


class RobotIO(Protocol[SignalsT]):
    actuator_names: list[str]
    control_dt: float

    def read(self) -> SignalsT:
        ...

    def write_ctrl(self, ctrl_targets_rad) -> None:
        ...


class NumpyRobotIO(RobotIO[SignalsT]):
    def read(self) -> SignalsT:
        raise NotImplementedError

    def write_ctrl(self, ctrl_targets_rad) -> None:
        raise NotImplementedError


class NumpySignalsProvider(SignalsProvider[SignalsT]):
    def read(self, *args, **kwargs) -> SignalsT:
        raise NotImplementedError
