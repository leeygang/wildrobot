from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Sequence

import numpy as np

from .hiwonder_board_controller import HiwonderBoardController
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


class HiwonderBoardActuators(Actuators):
    """Deprecated actuator implementation for the old Hiwonder LSC board."""

    def __init__(
        self,
        actuator_names: Sequence[str],
        servo_ids: Dict[str, int],
        port: str,
        baudrate: int,
        default_move_time_ms: Optional[int],
        joint_servo_offset_units: Dict[str, int],
        joint_motor_unit_directions: Optional[Dict[str, float]] = None,
        joint_angle_at_zero_unit_deg: Optional[Dict[str, float]] = None,
        servo_model: ServoModel | None = None,
        max_retries: int = 3,
        retry_backoff_s: float = 0.002,
        controller: Optional[HiwonderBoardController] = None,
        read_schedule_mode: str = "full",
        read_schedule_groups: Optional[Sequence[Sequence[str]]] = None,
        read_schedule_max_cache_age_s: Optional[Dict[str, float]] = None,
    ) -> None:
        warnings.warn(
            "HiwonderBoardActuators is deprecated and is not used by policy "
            "runtime. Use HiwonderCachedActuators with the raw TTL servo bus.",
            DeprecationWarning,
            stacklevel=2,
        )
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
        joint_motor_unit_directions = joint_motor_unit_directions or {}
        joint_angle_at_zero_unit_deg = joint_angle_at_zero_unit_deg or {}
        for name in self.actuator_names:
            if name not in servo_ids:
                raise KeyError(f"Servo ID missing for joint '{name}'")
            self.servo_ids_list.append(int(servo_ids[name]))
            offsets.append(int(joint_servo_offset_units.get(name, 0)))
            motor_signs.append(float(joint_motor_unit_directions.get(name, 1.0)))
            centers_rad.append(
                float(np.deg2rad(joint_angle_at_zero_unit_deg.get(name, 0.0)))
            )

        self.offsets_unit = np.asarray(offsets, dtype=np.float32)
        self.motor_signs = np.asarray(motor_signs, dtype=np.float32)
        self.centers_rad = np.asarray(centers_rad, dtype=np.float32)

        self.controller = controller or HiwonderBoardController(port=port, baudrate=baudrate)
        self._last_positions: Optional[np.ndarray] = None
        self._prev_positions: Optional[np.ndarray] = None
        self._last_error: Optional[Exception] = None
        self.read_schedule_mode = str(read_schedule_mode or "full").lower()
        self._read_schedule_groups = self._build_read_schedule_groups(
            read_schedule_groups or []
        )
        self._read_group_cursor = 0
        self._position_cache_rad = np.full(len(self.servo_ids_list), np.nan, dtype=np.float32)
        self._velocity_cache_rad_s = np.zeros(len(self.servo_ids_list), dtype=np.float32)
        self._prev_position_cache_rad = np.full(len(self.servo_ids_list), np.nan, dtype=np.float32)
        self._last_update_wall_time_s = np.full(len(self.servo_ids_list), np.nan, dtype=np.float64)
        self._read_fail_count_by_joint = np.zeros(len(self.servo_ids_list), dtype=np.int32)
        self._last_read_group_name: Optional[str] = None
        self._last_read_group_ids: List[int] = []
        self._last_read_count = 0
        self._last_read_fail_count = 0
        self._last_cache_age_s = np.full(len(self.servo_ids_list), np.inf, dtype=np.float32)
        self._cache_age_limit_s = self._build_cache_age_limits(
            read_schedule_max_cache_age_s or {}
        )
        if self.read_schedule_mode not in {"full", "staggered"}:
            raise ValueError("read_schedule_mode must be 'full' or 'staggered'")
        if self.read_schedule_mode == "staggered" and not self._read_schedule_groups:
            raise ValueError("staggered read schedule requires at least one group")

    def _build_read_schedule_groups(
        self, groups: Sequence[Sequence[str]]
    ) -> list[tuple[str, list[int], list[int]]]:
        by_name = {name: i for i, name in enumerate(self.actuator_names)}
        out: list[tuple[str, list[int], list[int]]] = []
        for group_idx, names in enumerate(groups):
            indices: list[int] = []
            group_names: list[str] = []
            for name in names:
                name_s = str(name)
                if name_s not in by_name:
                    raise KeyError(f"servo_read_schedule group references unknown joint '{name_s}'")
                idx = by_name[name_s]
                if idx in indices:
                    continue
                indices.append(idx)
                group_names.append(name_s)
            if not indices:
                raise ValueError(f"servo_read_schedule group {group_idx} is empty")
            label = self._infer_read_group_label(group_idx, group_names)
            servo_ids = [int(self.servo_ids_list[i]) for i in indices]
            out.append((label, indices, servo_ids))
        return out

    @staticmethod
    def _infer_read_group_label(group_idx: int, names: Sequence[str]) -> str:
        names_set = set(names)
        if names_set and all(name.startswith("left_") for name in names_set):
            if any(part in name for name in names_set for part in ("hip", "knee", "ankle")):
                return "left_leg"
        if names_set and all(name.startswith("right_") for name in names_set):
            if any(part in name for name in names_set for part in ("hip", "knee", "ankle")):
                return "right_leg"
        if any("wrist" in name or "shoulder" in name or "elbow" in name for name in names_set):
            return "torso_arms"
        return f"group_{group_idx}"

    def _build_cache_age_limits(self, limits: Dict[str, float]) -> np.ndarray:
        defaults = {"leg": 0.12, "arm": 0.25, "wrist": 0.50, "default": 0.25}
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

    def _cache_is_initialized(self) -> bool:
        return bool(np.all(np.isfinite(self._position_cache_rad)))

    def _cache_age_s(self, now_s: float | None = None) -> np.ndarray:
        now = time.monotonic() if now_s is None else float(now_s)
        age = now - self._last_update_wall_time_s
        age[~np.isfinite(self._last_update_wall_time_s)] = np.inf
        return age.astype(np.float32)

    def _update_cache_from_units(
        self,
        *,
        indices: Sequence[int],
        units_by_servo_id: Dict[int, int],
        now_s: float,
    ) -> int:
        updated = 0
        for idx in indices:
            servo_id = int(self.servo_ids_list[idx])
            if servo_id not in units_by_servo_id:
                self._read_fail_count_by_joint[idx] += 1
                continue
            units = np.asarray([float(units_by_servo_id[servo_id])], dtype=np.float32)
            q_rad = float(
                servo_pos_elect_units_to_joint_target_rad(
                    units,
                    self.offsets_unit[idx : idx + 1],
                    self.motor_signs[idx : idx + 1],
                    self.centers_rad[idx : idx + 1],
                    self.servo_model,
                )[0]
            )
            prev_q = float(self._position_cache_rad[idx])
            prev_time = float(self._last_update_wall_time_s[idx])
            self._prev_position_cache_rad[idx] = prev_q
            self._position_cache_rad[idx] = q_rad
            if np.isfinite(prev_q) and np.isfinite(prev_time) and now_s > prev_time:
                self._velocity_cache_rad_s[idx] = (q_rad - prev_q) / (now_s - prev_time)
            self._last_update_wall_time_s[idx] = now_s
            self._read_fail_count_by_joint[idx] = 0
            updated += 1
        self._last_cache_age_s = self._cache_age_s(now_s)
        return updated

    def _update_full_read_cache(self, units: np.ndarray, radians: np.ndarray) -> None:
        now_s = time.monotonic()
        previous = self._position_cache_rad.copy()
        previous_time = self._last_update_wall_time_s.copy()
        self._prev_position_cache_rad = previous
        self._position_cache_rad = np.asarray(radians, dtype=np.float32).copy()
        for idx, q_rad in enumerate(self._position_cache_rad):
            prev_q = float(previous[idx])
            prev_t = float(previous_time[idx])
            if np.isfinite(prev_q) and np.isfinite(prev_t) and now_s > prev_t:
                self._velocity_cache_rad_s[idx] = (float(q_rad) - prev_q) / (now_s - prev_t)
        self._last_update_wall_time_s[:] = now_s
        self._read_fail_count_by_joint[:] = 0
        self._last_read_group_name = "full"
        self._last_read_group_ids = [int(sid) for sid in self.servo_ids_list]
        self._last_read_count = int(units.size)
        self._last_read_fail_count = 0
        self._last_cache_age_s = self._cache_age_s(now_s)

    def _check_cache_age_or_raise(self) -> None:
        age = self._cache_age_s()
        self._last_cache_age_s = age
        uninitialized = np.where(~np.isfinite(age))[0].tolist()
        stale = np.where(age > self._cache_age_limit_s)[0].tolist()
        if uninitialized:
            names = [self.actuator_names[i] for i in uninitialized]
            raise RuntimeError(f"servo cache has uninitialized joints: {names}")
        if stale:
            details = [
                f"{self.actuator_names[i]} age_s={float(age[i]):.3f} "
                f"limit_s={float(self._cache_age_limit_s[i]):.3f}"
                for i in stale
            ]
            raise RuntimeError("servo cache age exceeded: " + "; ".join(details))

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
        if self.read_schedule_mode == "staggered":
            return self._get_positions_rad_staggered()
        return self._get_positions_rad_full()

    def _get_positions_rad_full(self) -> Optional[np.ndarray]:
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
            self._update_full_read_cache(units, radians)
            self._last_error = None
            return radians

        self._last_error = last_err
        return None

    def _get_positions_rad_staggered(self) -> Optional[np.ndarray]:
        if not self._cache_is_initialized():
            return self._initialize_position_cache()

        group_name, indices, servo_ids = self._read_schedule_groups[
            self._read_group_cursor % len(self._read_schedule_groups)
        ]
        self._read_group_cursor += 1
        self._last_read_group_name = group_name
        self._last_read_group_ids = [int(sid) for sid in servo_ids]
        self._last_read_count = 0
        self._last_read_fail_count = len(servo_ids)
        self._last_error = None

        try:
            resp = self._read_positions_complete(servo_ids)
        except Exception as exc:
            self._last_error = exc
            resp = None

        pos_map: dict[int, int] = {}
        if resp:
            pos_map = {int(sid): int(pos) for sid, pos in resp}
        now_s = time.monotonic()
        updated = self._update_cache_from_units(
            indices=indices,
            units_by_servo_id=pos_map,
            now_s=now_s,
        )
        self._last_read_count = int(updated)
        self._last_read_fail_count = len(indices) - int(updated)
        if self._last_read_fail_count > 0 and self._last_error is None:
            missing = [int(sid) for sid in servo_ids if int(sid) not in pos_map]
            self._last_error = RuntimeError(
                f"staggered servo read incomplete group={group_name} missing_ids={missing}"
            )
        self._check_cache_age_or_raise()
        self._last_positions = self._position_cache_rad.copy()
        return self._position_cache_rad.copy()

    def _initialize_position_cache(self) -> Optional[np.ndarray]:
        previous_mode = self.read_schedule_mode
        self.read_schedule_mode = "full"
        try:
            positions = self._get_positions_rad_full()
        finally:
            self.read_schedule_mode = previous_mode
        if positions is None:
            return None
        return np.asarray(positions, dtype=np.float32).copy()

    def _read_positions_complete(
        self, servo_ids: Optional[Sequence[int]] = None
    ) -> Optional[List[tuple[int, int]]]:
        """Read all requested servo IDs, tolerating partial replies.

        Many boards/links will sometimes reply with only a subset of IDs.
        We merge whatever we got, then retry only the missing IDs.
        """
        ids = [int(x) for x in (servo_ids if servo_ids is not None else self.servo_ids_list)]
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
        if self.read_schedule_mode == "staggered":
            return self._velocity_cache_rad_s.copy()
        if self._last_positions is None or self._prev_positions is None or dt <= 0.0:
            return np.zeros(len(self.servo_ids_list), dtype=np.float32)
        return (self._last_positions - self._prev_positions) / float(dt)

    def get_servo_cache_metrics(self) -> dict:
        age = self._cache_age_s()
        self._last_cache_age_s = age
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

        uninitialized = int(np.count_nonzero(~np.isfinite(age)))
        stale = int(np.count_nonzero(age > self._cache_age_limit_s))
        return {
            "servo_read_mode": self.read_schedule_mode,
            "servo_read_group": self._last_read_group_name,
            "servo_read_ids": list(self._last_read_group_ids),
            "servo_read_count": int(self._last_read_count),
            "servo_read_fail_count": int(self._last_read_fail_count),
            "servo_cache_age_max_s": max_age,
            "servo_cache_age_leg_max_s": _masked_max(leg_mask),
            "servo_cache_age_arm_max_s": _masked_max(arm_mask),
            "servo_cache_age_wrist_max_s": _masked_max(wrist_mask),
            "servo_cache_stale_joint_count": stale,
            "servo_cache_uninitialized_count": uninitialized,
            "servo_read_fail_count_total": int(np.sum(self._read_fail_count_by_joint)),
        }

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
        defaults = {"leg": 0.12, "arm": 0.25, "wrist": 0.50, "default": 0.25}
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
            f"servo cache did not initialize within {float(timeout_s):.2f}s"
        )
        return False

    def _cached_state_ready(self) -> bool:
        state = self.servo_io.get_cached_servo_state()
        index_by_servo_id = {int(sid): i for i, sid in enumerate(state.servo_ids)}
        for sid in self.servo_ids_list:
            idx = index_by_servo_id.get(int(sid))
            if idx is None:
                return False
            if not np.isfinite(float(state.position_units[idx])):
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

        return {
            "servo_read_mode": "ttl_worker",
            "servo_read_group": state.last_read_group,
            "servo_read_ids": [],
            "servo_read_count": int(metrics.read_success),
            "servo_read_fail_count": int(metrics.read_failures),
            "servo_cache_age_max_s": max_age,
            "servo_cache_age_leg_max_s": _masked_max(leg_mask),
            "servo_cache_age_arm_max_s": _masked_max(arm_mask),
            "servo_cache_age_wrist_max_s": _masked_max(wrist_mask),
            "servo_cache_stale_joint_count": int(np.count_nonzero(age > self._cache_age_limit_s)),
            "servo_cache_uninitialized_count": int(np.count_nonzero(~np.isfinite(age))),
            "servo_read_fail_count_total": int(np.sum(fail_count)),
            "servo_write_commands": int(metrics.write_commands),
            "servo_write_failures": int(metrics.write_failures),
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
