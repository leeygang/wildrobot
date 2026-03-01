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


def joint_target_rad_to_servo_pos_elec_units(
    targets_rad: np.ndarray,
    offsets_unit: np.ndarray,
    motor_signs: np.ndarray,
    centers_rad: np.ndarray,
    servo_model: ServoModel,
) -> np.ndarray:
    delta = targets_rad - centers_rad
    units = servo_model.units_center + offsets_unit + motor_signs * (delta * servo_model.units_per_rad)
    return np.clip(units, servo_model.units_min, servo_model.units_max)


def servo_pos_elect_units_to_joint_target_rad(
    units: np.ndarray,
    offsets_unit: np.ndarray,
    motor_signs: np.ndarray,
    centers_rad: np.ndarray,
    servo_model: ServoModel,
) -> np.ndarray:
    delta_units = units - servo_model.units_center - offsets_unit
    return centers_rad + motor_signs * (delta_units / servo_model.units_per_rad)


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
        joint_motor_signs: Optional[Dict[str, float]] = None,
        joint_motor_center_mujoco_deg: Optional[Dict[str, float]] = None,
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
        motor_signs: List[float] = []
        centers_rad: List[float] = []
        joint_motor_signs = joint_motor_signs or {}
        joint_motor_center_mujoco_deg = joint_motor_center_mujoco_deg or {}
        for name in self.actuator_names:
            if name not in servo_ids:
                raise KeyError(f"Servo ID missing for joint '{name}'")
            self.servo_ids_list.append(int(servo_ids[name]))
            offsets.append(int(joint_offset_units.get(name, 0)))
            motor_signs.append(float(joint_motor_signs.get(name, 1.0)))
            centers_rad.append(
                float(np.deg2rad(joint_motor_center_mujoco_deg.get(name, 0.0)))
            )

        self.offsets_unit = np.asarray(offsets, dtype=np.float32)
        self.motor_signs = np.asarray(motor_signs, dtype=np.float32)
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

        units = joint_target_rad_to_servo_pos_elec_units(
            targets,
            self.offsets_unit,
            self.motor_signs,
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
        for attempt in range(self.max_retries):
            try:
                resp = self._read_positions_complete()
            except Exception as exc:
                last_err = exc
                time.sleep(self.retry_backoff_s)
                continue

            if resp is None:
                last_err = RuntimeError(
                    f"Servo position response missing or incomplete (got 0/{len(self.servo_ids_list)})"
                )
                time.sleep(self.retry_backoff_s)
                continue

            pos_map: dict[int, int]
            try:
                pos_map = {int(sid): int(pos) for sid, pos in resp}
            except Exception as exc:
                last_err = exc
                time.sleep(self.retry_backoff_s)
                continue

            missing = [int(sid) for sid in self.servo_ids_list if int(sid) not in pos_map]
            if missing:
                # On the final attempt, run a quick diagnostic to pinpoint which IDs respond at all.
                diag = ""
                if attempt >= self.max_retries - 1:
                    try:
                        responding, not_responding = self._diagnose_individual_ids()
                        diag = f"; responding_ids={responding}; nonresponding_ids={not_responding}"
                    except Exception as exc:
                        diag = f"; diagnose_error={repr(exc)}"
                last_err = RuntimeError(
                    "Servo position response missing or incomplete "
                    f"(got {len(pos_map)}/{len(self.servo_ids_list)}; missing_ids={missing}){diag}"
                )
                time.sleep(self.retry_backoff_s)
                continue

            try:
                units = np.asarray([pos_map[int(sid)] for sid in self.servo_ids_list], dtype=np.float32)
            except Exception as exc:
                last_err = exc
                time.sleep(self.retry_backoff_s)
                continue

            radians = servo_pos_elect_units_to_joint_target_rad(
                units,
                self.offsets_unit,
                self.motor_signs,
                self.centers_rad,
                self.servo_model,
            ).astype(np.float32)
            self._prev_positions = self._last_positions
            self._last_positions = radians
            self._last_error = None
            return radians

        self._last_error = last_err
        return None

    def _read_positions_complete(self) -> Optional[List[tuple[int, int]]]:
        """Read all requested servo IDs, tolerating partial replies.

        Many boards/links will sometimes reply with only a subset of IDs.
        We merge whatever we got, then retry only the missing IDs.
        """
        ids = [int(x) for x in self.servo_ids_list]
        results: dict[int, int] = {}

        # First try: request all IDs at once.
        resp = self.controller.read_servo_positions(ids)
        if resp:
            for sid, units in resp:
                results[int(sid)] = int(units)

        missing = [sid for sid in ids if sid not in results]
        if not missing:
            return [(sid, results[sid]) for sid in ids]

        # Retry only missing IDs (no proactive batching). Some boards respond on a second try.
        for _ in range(2):
            time.sleep(self.retry_backoff_s)
            resp2 = self.controller.read_servo_positions(missing)
            if resp2:
                for sid, units in resp2:
                    results[int(sid)] = int(units)
            missing = [sid for sid in ids if sid not in results]
            if not missing:
                return [(sid, results[sid]) for sid in ids]

        # Last resort: single-ID reads for whatever is still missing.
        for sid in list(missing):
            time.sleep(self.retry_backoff_s)
            resp3 = self.controller.read_servo_positions([sid])
            if resp3 and len(resp3) == 1:
                results[int(resp3[0][0])] = int(resp3[0][1])

        missing = [sid for sid in ids if sid not in results]
        if missing:
            return [(sid, results[sid]) for sid in ids if sid in results]
        return [(sid, results[sid]) for sid in ids]

    def _diagnose_individual_ids(self) -> tuple[List[int], List[int]]:
        responding: List[int] = []
        not_responding: List[int] = []
        for sid in self.servo_ids_list:
            try:
                resp = self.controller.read_servo_positions([int(sid)])
            except Exception:
                resp = None
            if resp and len(resp) == 1:
                responding.append(int(sid))
            else:
                not_responding.append(int(sid))
            time.sleep(self.retry_backoff_s)
        return responding, not_responding

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
