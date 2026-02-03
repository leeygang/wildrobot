from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Sequence

import numpy as np

from .hiwonder_board_controller import HiwonderBoardController


class Actuators(Protocol):
    """Hardware actuators interface for the runtime control loop."""

    def set_targets_rad(self, targets_rad: np.ndarray, *, move_time_ms: int | None = None) -> None:
        ...

    def get_positions_rad(self) -> Optional[np.ndarray]:
        ...

    def estimate_velocities_rad_s(self, dt: float) -> np.ndarray:
        ...

    def disable(self) -> None:
        ...

    def close(self) -> None:
        ...


@dataclass(frozen=True)
class ServoModel:
    units_min: int = 0
    units_max: int = 1000
    units_center: int = 500
    units_per_rad: float = 500.0 / np.deg2rad(120.0)  # ~238.732


def rad_to_servo_units(
    targets_rad: np.ndarray,
    offsets_unit: np.ndarray,
    directions: np.ndarray,
    centers_rad: np.ndarray,
    servo_model: ServoModel,
) -> np.ndarray:
    delta = targets_rad - centers_rad
    units = servo_model.units_center + offsets_unit + directions * (delta * servo_model.units_per_rad)
    return np.clip(units, servo_model.units_min, servo_model.units_max)


def servo_units_to_rad(
    units: np.ndarray,
    offsets_unit: np.ndarray,
    directions: np.ndarray,
    centers_rad: np.ndarray,
    servo_model: ServoModel,
) -> np.ndarray:
    delta_units = units - servo_model.units_center - offsets_unit
    return centers_rad + directions * (delta_units / servo_model.units_per_rad)


class HiwonderBoardActuators(Actuators):
    """Actuator implementation backed by the Hiwonder servo board."""

    def __init__(
        self,
        actuator_names: Sequence[str],
        servo_ids: Dict[str, int],
        port: str,
        baudrate: int,
        default_move_time_ms: Optional[int],
        joint_offset_units: Dict[str, int],
        joint_directions: Optional[Dict[str, float]] = None,
        joint_center_deg: Optional[Dict[str, float]] = None,
        servo_model: ServoModel | None = None,
        max_retries: int = 3,
        retry_backoff_s: float = 0.002,
        controller: Optional[HiwonderBoardController] = None,
    ) -> None:
        self.actuator_names: List[str] = list(actuator_names)
        self.port = port
        self.baudrate = baudrate
        self.default_move_time_ms = default_move_time_ms
        self.servo_model = servo_model or ServoModel()
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s

        self.servo_ids_list: List[int] = []
        offsets: List[int] = []
        directions: List[float] = []
        centers_rad: List[float] = []
        joint_directions = joint_directions or {}
        joint_center_deg = joint_center_deg or {}
        for name in self.actuator_names:
            if name not in servo_ids:
                raise KeyError(f"Servo ID missing for joint '{name}'")
            self.servo_ids_list.append(int(servo_ids[name]))
            offsets.append(int(joint_offset_units.get(name, 0)))
            directions.append(float(joint_directions.get(name, 1.0)))
            centers_rad.append(float(np.deg2rad(joint_center_deg.get(name, 0.0))))

        self.offsets_unit = np.asarray(offsets, dtype=np.float32)
        self.directions = np.asarray(directions, dtype=np.float32)
        self.centers_rad = np.asarray(centers_rad, dtype=np.float32)

        self.controller = controller or HiwonderBoardController(port=port, baudrate=baudrate)
        self._last_positions: Optional[np.ndarray] = None
        self._prev_positions: Optional[np.ndarray] = None
        self._last_error: Optional[Exception] = None

    def set_targets_rad(self, targets_rad: np.ndarray, *, move_time_ms: int | None = None) -> None:
        targets = np.asarray(targets_rad, dtype=np.float32)
        if targets.shape[0] != len(self.servo_ids_list):
            raise ValueError(f"Expected {len(self.servo_ids_list)} targets, got {targets.shape[0]}")

        move_time = move_time_ms if move_time_ms is not None else self.default_move_time_ms
        if move_time is None:
            raise ValueError("move_time_ms must be provided when no default_move_time_ms is set")

        units = rad_to_servo_units(
            targets,
            self.offsets_unit,
            self.directions,
            self.centers_rad,
            self.servo_model,
        )
        commands = list(zip(self.servo_ids_list, np.rint(units).astype(int).tolist()))

        last_err: Optional[Exception] = None
        self._last_error = None
        for attempt in range(self.max_retries):
            try:
                self.controller.move_servos(commands, time_ms=move_time)
                self._last_error = None
                return
            except Exception as exc:
                last_err = exc
                self._last_error = exc
                time.sleep(self.retry_backoff_s)
        raise RuntimeError(
            f"Failed to set targets for joints {self.actuator_names} "
            f"on port {self.port} baud {self.baudrate} after {self.max_retries} attempts"
        ) from last_err

    def get_positions_rad(self) -> Optional[np.ndarray]:
        last_err: Optional[Exception] = None
        self._last_error = None
        for _ in range(self.max_retries):
            try:
                resp = self.controller.read_servo_positions(self.servo_ids_list)
            except Exception as exc:
                last_err = exc
                time.sleep(self.retry_backoff_s)
                continue

            if resp is None or len(resp) != len(self.servo_ids_list):
                last_err = RuntimeError("Servo position response missing or incomplete")
                time.sleep(self.retry_backoff_s)
                continue

            try:
                pos_map = {sid: pos for sid, pos in resp}
                units = np.asarray([pos_map[sid] for sid in self.servo_ids_list], dtype=np.float32)
            except Exception as exc:
                last_err = exc
                time.sleep(self.retry_backoff_s)
                continue

            radians = servo_units_to_rad(
                units,
                self.offsets_unit,
                self.directions,
                self.centers_rad,
                self.servo_model,
            ).astype(np.float32)
            self._prev_positions = self._last_positions
            self._last_positions = radians
            self._last_error = None
            return radians

        self._last_error = last_err
        return None

    def estimate_velocities_rad_s(self, dt: float) -> np.ndarray:
        if self._last_positions is None or self._prev_positions is None or dt <= 0.0:
            return np.zeros(len(self.servo_ids_list), dtype=np.float32)
        return (self._last_positions - self._prev_positions) / float(dt)

    def disable(self) -> None:
        try:
            self.controller.unload_servos(self.servo_ids_list)
        except Exception:
            pass

    def close(self) -> None:
        try:
            self.controller.close()
        except Exception:
            pass
