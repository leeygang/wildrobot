from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Sequence

import numpy as np

from .servo_io_worker import ServoIOWorker


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


class HiwonderCachedActuators(Actuators):
    """Actuator adapter backed by a ServoIOWorker cache.

    The worker owns raw serial IO and stores servo electrical units. This
    adapter keeps the runtime-facing actuator API in calibrated joint radians.
    """

    def __init__(
        self,
        actuator_names: Sequence[str],
        servo_ids: Dict[str, int],
        default_move_time_ms: Optional[int],
        joint_servo_offset_units: Dict[str, int],
        servo_io: ServoIOWorker,
        joint_motor_unit_directions: Optional[Dict[str, float]] = None,
        joint_angle_at_zero_unit_deg: Optional[Dict[str, float]] = None,
        servo_model: ServoModel | None = None,
        cache_age_limits_s: Optional[Dict[str, float]] = None,
        port: Optional[str] = None,
        baudrate: Optional[int] = None,
    ) -> None:
        self.actuator_names: List[str] = list(actuator_names)
        self.default_move_time_ms = default_move_time_ms
        self.servo_io = servo_io
        self.servo_model = servo_model or ServoModel()
        transport = getattr(getattr(servo_io, "raw_bus", None), "transport", None)
        self.port = port or getattr(transport, "port", "unknown")
        self.baudrate = int(baudrate or getattr(transport, "baudrate", 0) or 0)
        self._last_error: Optional[Exception] = None

        self.servo_ids_list: List[int] = []
        offsets: List[int] = []
        motor_signs: List[float] = []
        centers_rad: List[float] = []
        joint_motor_unit_directions = joint_motor_unit_directions or {}
        joint_angle_at_zero_unit_deg = joint_angle_at_zero_unit_deg or {}
        for name in self.actuator_names:
            if name not in servo_ids:
                raise KeyError(f"Servo ID missing for joint '{name}'")
            self.servo_ids_list.append(int(servo_ids[name]))
            offsets.append(int(joint_servo_offset_units.get(name, 0)))
            motor_signs.append(float(joint_motor_unit_directions.get(name, 1.0)))
            centers_rad.append(float(np.deg2rad(joint_angle_at_zero_unit_deg.get(name, 0.0))))

        self.offsets_unit = np.asarray(offsets, dtype=np.float32)
        self.motor_signs = np.asarray(motor_signs, dtype=np.float32)
        self.centers_rad = np.asarray(centers_rad, dtype=np.float32)
        self._cache_age_limit_s = self._build_cache_age_limits(cache_age_limits_s or {})

    def _build_cache_age_limits(self, limits: Dict[str, float]) -> np.ndarray:
        defaults = {"leg": 0.12, "arm": 0.75, "wrist": 1.00, "default": 0.75}
        merged = {**defaults, **{str(k): float(v) for k, v in limits.items()}}
        age_limits = []
        for name in self.actuator_names:
            if any(part in name for part in ("hip", "knee", "ankle")):
                key = "leg"
            elif "wrist" in name:
                key = "wrist"
            elif any(part in name for part in ("shoulder", "elbow")):
                key = "arm"
            else:
                key = "default"
            age_limits.append(float(merged.get(key, merged["default"])))
        return np.asarray(age_limits, dtype=np.float32)

    def wait_for_initial_cache(self, *, timeout_s: float = 3.0) -> bool:
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        while time.monotonic() < deadline:
            if self._cached_state_ready():
                self._last_error = None
                return True
            time.sleep(0.002)
        self._last_error = RuntimeError(
            f"servo cache did not become fresh within {float(timeout_s):.2f}s"
        )
        return False

    def _cached_state_ready(self) -> bool:
        state = self.servo_io.get_cached_servo_state()
        index_by_servo_id = {int(sid): i for i, sid in enumerate(state.servo_ids)}
        for joint_idx, sid in enumerate(self.servo_ids_list):
            idx = index_by_servo_id.get(int(sid))
            if idx is None:
                return False
            if not np.isfinite(float(state.position_units[idx])):
                return False
            age_s = float(state.position_age_s[idx])
            limit_s = float(self._cache_age_limit_s[joint_idx])
            if not np.isfinite(age_s) or age_s > limit_s:
                return False
        return True

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
        positions_by_servo_id = dict(
            zip(self.servo_ids_list, np.rint(units).astype(int).tolist())
        )
        self.servo_io.submit_targets_units(positions_by_servo_id, move_time_ms=int(move_time))

    def get_positions_rad(self) -> Optional[np.ndarray]:
        state = self.servo_io.get_cached_servo_state()
        index_by_servo_id = {int(sid): i for i, sid in enumerate(state.servo_ids)}
        unit_values: list[float] = []
        for joint_idx, sid in enumerate(self.servo_ids_list):
            idx = index_by_servo_id.get(int(sid))
            if idx is None:
                self._last_error = RuntimeError(f"missing cached servo id {sid}")
                return None
            value = float(state.position_units[idx])
            if not np.isfinite(value):
                self._last_error = RuntimeError(f"missing cached position for servo id {sid}")
                return None
            age_s = float(state.position_age_s[idx])
            limit_s = float(self._cache_age_limit_s[joint_idx])
            if not np.isfinite(age_s) or age_s > limit_s:
                self._last_error = RuntimeError(
                    f"servo cache age exceeded for id {sid}: "
                    f"age_s={age_s:.3f} limit_s={limit_s:.3f}"
                )
                return None
            unit_values.append(value)

        units = np.asarray(unit_values, dtype=np.float32)
        self._last_error = None
        return servo_pos_elect_units_to_joint_target_rad(
            units,
            self.offsets_unit,
            self.motor_signs,
            self.centers_rad,
            self.servo_model,
        ).astype(np.float32)

    def estimate_velocities_rad_s(self, dt: float) -> np.ndarray:
        state = self.servo_io.get_cached_servo_state()
        index_by_servo_id = {int(sid): i for i, sid in enumerate(state.servo_ids)}
        velocities: list[float] = []
        for i, sid in enumerate(self.servo_ids_list):
            idx = index_by_servo_id.get(int(sid))
            if idx is None:
                velocities.append(0.0)
                continue
            units_s = float(state.velocity_units_s[idx])
            if not np.isfinite(units_s):
                velocities.append(0.0)
                continue
            velocities.append(float(self.motor_signs[i]) * units_s / float(self.servo_model.units_per_rad))
        return np.asarray(velocities, dtype=np.float32)

    def get_servo_cache_metrics(self) -> dict:
        state = self.servo_io.get_cached_servo_state()
        metrics = self.servo_io.get_metrics()
        index_by_servo_id = {int(sid): i for i, sid in enumerate(state.servo_ids)}
        age = np.full(len(self.servo_ids_list), np.inf, dtype=np.float32)
        fail_count = np.zeros(len(self.servo_ids_list), dtype=np.int32)
        for i, sid in enumerate(self.servo_ids_list):
            idx = index_by_servo_id.get(int(sid))
            if idx is None:
                continue
            age[i] = float(state.position_age_s[idx])
            fail_count[i] = int(state.read_fail_count[idx])

        finite_age = age[np.isfinite(age)]
        max_age = float(np.max(finite_age)) if finite_age.size else float("inf")
        leg_mask = np.asarray(
            [
                any(part in name for part in ("hip", "knee", "ankle"))
                for name in self.actuator_names
            ],
            dtype=bool,
        )
        arm_mask = np.asarray(
            [
                any(part in name for part in ("shoulder", "elbow"))
                for name in self.actuator_names
            ],
            dtype=bool,
        )
        wrist_mask = np.asarray(["wrist" in name for name in self.actuator_names], dtype=bool)

        def _masked_max(mask: np.ndarray) -> float:
            vals = age[mask & np.isfinite(age)]
            return float(np.max(vals)) if vals.size else 0.0

        servo_io_config = getattr(self.servo_io, "config", None)
        write_deadband_units = int(getattr(servo_io_config, "write_deadband_units", 0))

        return {
            "servo_read_mode": "ttl_worker",
            "servo_read_group": state.last_read_group,
            "servo_read_ids": (
                [] if state.last_read_servo_id is None else [int(state.last_read_servo_id)]
            ),
            "servo_read_count": int(metrics.read_success),
            "servo_read_fail_count": int(metrics.read_failures),
            "servo_cache_age_max_s": max_age,
            "servo_cache_age_leg_max_s": _masked_max(leg_mask),
            "servo_cache_age_arm_max_s": _masked_max(arm_mask),
            "servo_cache_age_wrist_max_s": _masked_max(wrist_mask),
            "servo_cache_stale_joint_count": int(np.count_nonzero(age > self._cache_age_limit_s)),
            "servo_cache_uninitialized_count": int(np.count_nonzero(~np.isfinite(age))),
            "servo_read_fail_count_total": int(np.sum(fail_count)),
            "servo_write_targets_submitted": int(metrics.write_targets_submitted),
            "servo_write_targets_replaced": int(metrics.write_targets_replaced),
            "servo_write_commands": int(metrics.write_commands),
            "servo_write_commands_skipped": int(metrics.write_commands_skipped),
            "servo_write_deadband_units": write_deadband_units,
            "servo_write_failures": int(metrics.write_failures),
            "servo_cache_deadline_reads": int(metrics.cache_deadline_reads),
            "servo_forced_read_after_write": int(metrics.forced_read_after_write),
            "servo_forced_read_after_write_missed": int(
                metrics.forced_read_after_write_missed
            ),
            "servo_latest_write_queue_latency_s": float(
                metrics.latest_write_queue_latency_s
            ),
            "servo_latest_write_latency_s": float(metrics.latest_write_latency_s),
            "servo_latest_read_latency_s": float(metrics.latest_read_latency_s),
        }

    def disable(self) -> None:
        self.servo_io.stop()
        for sid in self.servo_ids_list:
            try:
                self.servo_io.raw_bus.unload(int(sid))
            except Exception:
                pass

    def close(self) -> None:
        self.servo_io.close()
