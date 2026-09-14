from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, replace
from logging import Logger
from typing import Optional, Sequence

import numpy as np

from .hiwonder_ttl_bus import RawServoBus


@dataclass(frozen=True)
class ServoReadGroup:
    name: str
    servo_ids: tuple[int, ...]
    retry_cache_age_s: float | None = None
    max_cache_age_s: float | None = None


@dataclass(frozen=True)
class ServoIOWorkerConfig:
    servo_ids: tuple[int, ...] = ()
    read_groups: tuple[ServoReadGroup, ...] = ()
    read_group_schedule: tuple[str, ...] = ()
    max_write_attempts: int = 2
    write_deadband_units: int = 3
    max_read_attempts: int = 2
    retry_cache_age_s: float = 0.08
    max_cache_age_s: float = 0.25
    min_reads_after_write: int = 2
    idle_sleep_s: float = 0.0005
    stale_log_period_s: float = 1.0
    health_poll_interval_s: float = 0.0


@dataclass(frozen=True)
class CachedServoState:
    servo_ids: tuple[int, ...]
    position_units: np.ndarray
    velocity_units_s: np.ndarray
    position_age_s: np.ndarray
    read_fail_count: np.ndarray
    last_update_time_s: np.ndarray
    temperature_c: np.ndarray
    voltage_v: np.ndarray
    torque_enabled_state: np.ndarray
    temperature_age_s: np.ndarray
    voltage_age_s: np.ndarray
    torque_enabled_age_s: np.ndarray
    health_read_fail_count: np.ndarray
    last_read_group: str | None
    last_read_servo_id: int | None
    last_error: str | None


@dataclass(frozen=True)
class ServoIOMetrics:
    write_targets_submitted: int = 0
    write_targets_replaced: int = 0
    write_commands: int = 0
    write_commands_skipped: int = 0
    write_failures: int = 0
    cache_deadline_reads: int = 0
    forced_read_after_write: int = 0
    forced_read_after_write_missed: int = 0
    read_success: int = 0
    read_failures: int = 0
    stale_cache_errors: int = 0
    health_read_success: int = 0
    health_read_failures: int = 0
    unexpected_unload_events: int = 0
    unload_commands: int = 0
    unload_failures: int = 0
    unload_unverified: int = 0
    latest_write_queue_latency_s: float = 0.0
    latest_write_latency_s: float = 0.0
    latest_read_latency_s: float = 0.0
    latest_health_read_latency_s: float = 0.0


class ServoIOWorker:
    """Single-owner runtime IO worker for raw HTD/Hiwonder servo buses.

    The worker accepts latest-wins target writes and otherwise polls one servo
    position at a time into a full cache. Optional health reads are individually
    staggered across the configured interval and yield to target writes and
    position-cache deadlines; the worker never performs a blocking health sweep.
    """

    _CACHE_DEADLINE_READ_FRACTION = 0.70
    _HEALTH_FIELDS = ("temperature", "voltage", "torque_enabled")

    def __init__(
        self,
        raw_bus: RawServoBus,
        config: ServoIOWorkerConfig,
        *,
        logger: Logger | None = None,
        worker_name: str = "servo_board",
    ) -> None:
        self.raw_bus = raw_bus
        self.config = config
        self.logger = logger
        self.worker_name = str(worker_name)

        self._read_groups = self._normalize_read_groups(config)
        self._read_group_by_name = {group.name: group for group in self._read_groups}
        self._schedule = self._normalize_schedule(config, self._read_groups)
        self._group_offsets = {group.name: 0 for group in self._read_groups}
        self._schedule_index = 0

        servo_ids = self._servo_ids_from_config(config, self._read_groups)
        if not servo_ids:
            raise ValueError("ServoIOWorker requires at least one servo id")
        self.servo_ids = tuple(servo_ids)
        self._id_to_index = {sid: i for i, sid in enumerate(self.servo_ids)}
        self._group_for_servo = self._build_group_for_servo(self._read_groups)
        self._max_cache_age_by_servo = self._build_max_cache_age_by_servo(
            self.servo_ids, self._read_groups, config
        )

        n = len(self.servo_ids)
        self._position_units = np.full(n, np.nan, dtype=np.float32)
        self._velocity_units_s = np.zeros(n, dtype=np.float32)
        self._last_update_time_s = np.full(n, np.nan, dtype=np.float64)
        self._read_fail_count = np.zeros(n, dtype=np.int32)
        self._temperature_c = np.full(n, np.nan, dtype=np.float32)
        self._voltage_v = np.full(n, np.nan, dtype=np.float32)
        self._torque_enabled_state = np.full(n, -1, dtype=np.int8)
        self._torque_disabled_active = np.zeros(n, dtype=bool)
        self._temperature_update_time_s = np.full(n, np.nan, dtype=np.float64)
        self._voltage_update_time_s = np.full(n, np.nan, dtype=np.float64)
        self._torque_enabled_update_time_s = np.full(n, np.nan, dtype=np.float64)
        self._health_read_fail_count = np.zeros(n, dtype=np.int32)

        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._pending_target: tuple[dict[int, int], int, float] | None = None
        self._retry_queue: list[int] = []
        self._retry_attempts: dict[int, int] = {}
        self._stale_log_time: dict[int, float] = {}
        self._last_read_group: str | None = None
        self._last_read_servo_id: int | None = None
        self._last_error: str | None = None
        self._last_written_target_by_servo: dict[int, tuple[int, int]] = {}
        self._health_cursor = 0
        self._next_health_read_s = time.monotonic()
        self._metrics = ServoIOMetrics()

    @staticmethod
    def _normalize_read_groups(config: ServoIOWorkerConfig) -> tuple[ServoReadGroup, ...]:
        if config.read_groups:
            return tuple(config.read_groups)
        if config.servo_ids:
            return (ServoReadGroup(name="all", servo_ids=tuple(config.servo_ids)),)
        return ()

    @staticmethod
    def _normalize_schedule(
        config: ServoIOWorkerConfig, read_groups: Sequence[ServoReadGroup]
    ) -> tuple[str, ...]:
        if config.read_group_schedule:
            names = tuple(config.read_group_schedule)
        else:
            names = tuple(group.name for group in read_groups)
        known = {group.name for group in read_groups}
        unknown = [name for name in names if name not in known]
        if unknown:
            raise ValueError(f"read_group_schedule references unknown groups: {unknown}")
        return names

    @staticmethod
    def _servo_ids_from_config(
        config: ServoIOWorkerConfig, read_groups: Sequence[ServoReadGroup]
    ) -> list[int]:
        ordered: list[int] = []
        for sid in config.servo_ids:
            sid_int = int(sid)
            if sid_int not in ordered:
                ordered.append(sid_int)
        for group in read_groups:
            for sid in group.servo_ids:
                sid_int = int(sid)
                if sid_int not in ordered:
                    ordered.append(sid_int)
        return ordered

    @staticmethod
    def _build_group_for_servo(read_groups: Sequence[ServoReadGroup]) -> dict[int, ServoReadGroup]:
        group_for_servo: dict[int, ServoReadGroup] = {}
        for group in read_groups:
            for sid in group.servo_ids:
                group_for_servo.setdefault(int(sid), group)
        return group_for_servo

    @staticmethod
    def _build_max_cache_age_by_servo(
        servo_ids: Sequence[int],
        read_groups: Sequence[ServoReadGroup],
        config: ServoIOWorkerConfig,
    ) -> dict[int, float]:
        max_age_by_servo = {int(sid): float(config.max_cache_age_s) for sid in servo_ids}
        for group in read_groups:
            if group.max_cache_age_s is None:
                continue
            max_age_s = float(group.max_cache_age_s)
            for sid in group.servo_ids:
                max_age_by_servo[int(sid)] = max_age_s
        return max_age_by_servo

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name=f"ServoIOWorker-{self.worker_name}",
            daemon=True,
        )
        self._thread.start()

    def stop(self, *, timeout_s: float = 1.0) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=float(timeout_s))
            self._thread = None

    def submit_targets_units(
        self,
        positions_by_servo_id: dict[int, int],
        *,
        move_time_ms: int,
    ) -> None:
        target = {int(sid): int(pos) for sid, pos in positions_by_servo_id.items()}
        with self._lock:
            replaced = self._pending_target is not None
            self._pending_target = (target, int(move_time_ms), time.monotonic())
            self._metrics = replace(
                self._metrics,
                write_targets_submitted=self._metrics.write_targets_submitted + 1,
                write_targets_replaced=self._metrics.write_targets_replaced + (1 if replaced else 0),
            )
        self._wake.set()

    def get_cached_servo_state(self) -> CachedServoState:
        now = time.monotonic()
        with self._lock:
            position = self._position_units.copy()
            velocity = self._velocity_units_s.copy()
            last_update = self._last_update_time_s.copy()
            fail_count = self._read_fail_count.copy()
            temperature_c = self._temperature_c.copy()
            voltage_v = self._voltage_v.copy()
            torque_enabled_state = self._torque_enabled_state.copy()
            temperature_update = self._temperature_update_time_s.copy()
            voltage_update = self._voltage_update_time_s.copy()
            torque_enabled_update = self._torque_enabled_update_time_s.copy()
            health_fail_count = self._health_read_fail_count.copy()
            last_group = self._last_read_group
            last_read_servo_id = self._last_read_servo_id
            last_error = self._last_error
        age = now - last_update
        age[~np.isfinite(last_update)] = np.inf
        temperature_age = now - temperature_update
        temperature_age[~np.isfinite(temperature_update)] = np.inf
        voltage_age = now - voltage_update
        voltage_age[~np.isfinite(voltage_update)] = np.inf
        torque_enabled_age = now - torque_enabled_update
        torque_enabled_age[~np.isfinite(torque_enabled_update)] = np.inf
        return CachedServoState(
            servo_ids=self.servo_ids,
            position_units=position,
            velocity_units_s=velocity,
            position_age_s=age.astype(np.float32),
            read_fail_count=fail_count,
            last_update_time_s=last_update,
            temperature_c=temperature_c,
            voltage_v=voltage_v,
            torque_enabled_state=torque_enabled_state,
            temperature_age_s=temperature_age.astype(np.float32),
            voltage_age_s=voltage_age.astype(np.float32),
            torque_enabled_age_s=torque_enabled_age.astype(np.float32),
            health_read_fail_count=health_fail_count,
            last_read_group=last_group,
            last_read_servo_id=last_read_servo_id,
            last_error=last_error,
        )

    def get_metrics(self) -> ServoIOMetrics:
        with self._lock:
            return self._metrics

    def close(self) -> None:
        self.stop()
        try:
            self.raw_bus.transport.close()
        except Exception:
            pass

    def unload_servos(self, servo_ids: Sequence[int]) -> dict[str, object]:
        attempted_ids: list[int] = []
        commanded_ids: list[int] = []
        confirmed_ids: list[int] = []
        unverified_ids: list[int] = []
        failed_ids: list[int] = []
        errors: list[str] = []
        for servo_id in servo_ids:
            sid = int(servo_id)
            attempted_ids.append(sid)
            try:
                self.raw_bus.unload(sid)
                commanded_ids.append(sid)
            except Exception as exc:
                failed_ids.append(sid)
                errors.append(f"servo_id={sid} unload command failed: {exc!r}")
                continue
            try:
                loaded = self.raw_bus.read_loaded(sid)
            except Exception as exc:
                loaded = None
                errors.append(f"servo_id={sid} unload verification failed: {exc!r}")
            if loaded is False:
                confirmed_ids.append(sid)
            elif loaded is None:
                unverified_ids.append(sid)
            else:
                failed_ids.append(sid)
                errors.append(f"servo_id={sid} remained torque-enabled after unload")

        with self._lock:
            self._metrics = replace(
                self._metrics,
                unload_commands=self._metrics.unload_commands + len(commanded_ids),
                unload_failures=self._metrics.unload_failures + len(failed_ids),
                unload_unverified=self._metrics.unload_unverified + len(unverified_ids),
            )
            if errors:
                self._last_error = errors[-1]
        for message in errors:
            self._log_error(message)
        return {
            "attempted_servo_ids": attempted_ids,
            "commanded_servo_ids": commanded_ids,
            "confirmed_servo_ids": confirmed_ids,
            "unverified_servo_ids": unverified_ids,
            "failed_servo_ids": failed_ids,
            "errors": errors,
        }

    def _run(self) -> None:
        reads_after_write_remaining = 0
        while not self._stop.is_set():
            if reads_after_write_remaining > 0:
                if self._read_next_available():
                    reads_after_write_remaining -= 1
                    with self._lock:
                        self._metrics = replace(
                            self._metrics,
                            forced_read_after_write=(
                                self._metrics.forced_read_after_write + 1
                            ),
                        )
                    continue
                reads_after_write_remaining = 0
                with self._lock:
                    self._metrics = replace(
                        self._metrics,
                        forced_read_after_write_missed=(
                            self._metrics.forced_read_after_write_missed + 1
                        ),
                    )

            target = self._pop_pending_target()
            if target is not None:
                self._write_target(target)
                reads_after_write_remaining = max(
                    0, int(self.config.min_reads_after_write)
                )
                continue

            if self._health_read_due():
                self._read_next_health()
                continue

            if self._read_next_available():
                continue

            self._wake.wait(timeout=max(0.0, float(self.config.idle_sleep_s)))
            self._wake.clear()

    def _health_read_due(self) -> bool:
        interval_s = float(self.config.health_poll_interval_s)
        if not math.isfinite(interval_s) or interval_s <= 0.0:
            return False
        now = time.monotonic()
        with self._lock:
            if self._pending_target is not None:
                return False
            if not bool(np.all(np.isfinite(self._last_update_time_s))):
                return False
            if now < self._next_health_read_s:
                return False
            for servo_id, idx in self._id_to_index.items():
                max_age_s = float(
                    self._max_cache_age_by_servo.get(
                        int(servo_id), self.config.max_cache_age_s
                    )
                )
                if now - float(self._last_update_time_s[idx]) >= (
                    self._CACHE_DEADLINE_READ_FRACTION * max_age_s
                ):
                    return False
        return True

    def _read_next_health(self) -> None:
        task_count = len(self.servo_ids) * len(self._HEALTH_FIELDS)
        if task_count <= 0:
            return
        task_index = self._health_cursor % task_count
        servo_id = int(self.servo_ids[task_index // len(self._HEALTH_FIELDS)])
        field_name = self._HEALTH_FIELDS[task_index % len(self._HEALTH_FIELDS)]
        self._health_cursor += 1
        spacing_s = max(0.0, float(self.config.health_poll_interval_s)) / task_count
        self._next_health_read_s = time.monotonic() + spacing_s
        self._read_health_one(servo_id, field_name)

    def _read_health_one(self, servo_id: int, field_name: str) -> None:
        start_s = time.monotonic()
        try:
            if field_name == "temperature":
                value = self.raw_bus.read_temperature_c(int(servo_id))
            elif field_name == "voltage":
                value = self.raw_bus.read_voltage_v(int(servo_id))
            else:
                value = self.raw_bus.read_loaded(int(servo_id))
        except Exception as exc:
            self._record_health_read_failure(
                servo_id, field_name, exc, latency_s=time.monotonic() - start_s
            )
            return
        if value is None:
            self._record_health_read_failure(
                servo_id, field_name, None, latency_s=time.monotonic() - start_s
            )
            return

        now = time.monotonic()
        idx = self._id_to_index[int(servo_id)]
        with self._lock:
            if field_name == "temperature":
                self._temperature_c[idx] = float(value)
                self._temperature_update_time_s[idx] = now
            elif field_name == "voltage":
                self._voltage_v[idx] = float(value)
                self._voltage_update_time_s[idx] = now
            else:
                current = 1 if bool(value) else 0
                self._torque_enabled_state[idx] = current
                self._torque_enabled_update_time_s[idx] = now
                was_commanded = int(servo_id) in self._last_written_target_by_servo
                if current == 0 and was_commanded and not bool(
                    self._torque_disabled_active[idx]
                ):
                    self._torque_disabled_active[idx] = True
                    self._metrics = replace(
                        self._metrics,
                        unexpected_unload_events=(
                            self._metrics.unexpected_unload_events + 1
                        ),
                    )
                elif current == 1:
                    self._torque_disabled_active[idx] = False
            self._metrics = replace(
                self._metrics,
                health_read_success=self._metrics.health_read_success + 1,
                latest_health_read_latency_s=time.monotonic() - start_s,
            )

    def _record_health_read_failure(
        self,
        servo_id: int,
        field_name: str,
        exc: Exception | None,
        *,
        latency_s: float,
    ) -> None:
        idx = self._id_to_index[int(servo_id)]
        message = f"health read failed servo_id={servo_id} field={field_name}"
        if exc is not None:
            message += f" error={exc!r}"
        with self._lock:
            self._health_read_fail_count[idx] += 1
            self._last_error = message
            self._metrics = replace(
                self._metrics,
                health_read_failures=self._metrics.health_read_failures + 1,
                latest_health_read_latency_s=float(latency_s),
            )

    def _read_next_available(self) -> bool:
        servo_id, group_name = self._next_servo_to_read()
        if servo_id is None:
            return False
        self._read_one(servo_id, group_name)
        return True

    def _pop_pending_target(self) -> tuple[dict[int, int], int, float] | None:
        with self._lock:
            target = self._pending_target
            self._pending_target = None
            return target

    def _write_target(self, target: tuple[dict[int, int], int, float]) -> None:
        positions_by_servo_id, move_time_ms, submitted_s = target
        start_s = time.monotonic()
        queue_latency_s = max(0.0, start_s - float(submitted_s))
        write_commands = 0
        write_commands_skipped = 0
        write_failures = 0
        write_deadband_units = max(0, int(self.config.write_deadband_units))
        for servo_id, position in positions_by_servo_id.items():
            target_key = (int(position), int(move_time_ms))
            last_target = self._last_written_target_by_servo.get(int(servo_id))
            if last_target is not None:
                last_position, last_move_time_ms = last_target
                if (
                    int(move_time_ms) == int(last_move_time_ms)
                    and abs(int(position) - int(last_position)) <= write_deadband_units
                ):
                    write_commands_skipped += 1
                    continue
            ok = False
            last_error: Exception | None = None
            for attempt in range(max(1, int(self.config.max_write_attempts))):
                try:
                    self.raw_bus.move_time_write(int(servo_id), int(position), int(move_time_ms))
                    ok = True
                    write_commands += 1
                    self._last_written_target_by_servo[int(servo_id)] = target_key
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt >= max(1, int(self.config.max_write_attempts)) - 1:
                        write_failures += 1
                        self._set_last_error(
                            f"write failed servo_id={servo_id} attempts={attempt + 1}: {exc!r}"
                        )
            if not ok and last_error is not None:
                self._log_error(
                    f"Servo write failed after {self.config.max_write_attempts} attempts: "
                    f"servo_id={servo_id} error={last_error!r}"
                )

        latency_s = time.monotonic() - start_s
        with self._lock:
            self._metrics = replace(
                self._metrics,
                write_commands=self._metrics.write_commands + write_commands,
                write_commands_skipped=self._metrics.write_commands_skipped + write_commands_skipped,
                write_failures=self._metrics.write_failures + write_failures,
                latest_write_queue_latency_s=queue_latency_s,
                latest_write_latency_s=latency_s,
            )

    def _next_servo_to_read(self) -> tuple[int | None, str | None]:
        with self._lock:
            deadline_servo_id = self._next_deadline_servo_to_read_locked(time.monotonic())
            if deadline_servo_id is not None:
                self._retry_queue = [
                    sid for sid in self._retry_queue if int(sid) != int(deadline_servo_id)
                ]
                group = self._group_for_servo.get(deadline_servo_id)
                self._advance_group_offset_after_servo(deadline_servo_id, group)
                self._metrics = replace(
                    self._metrics,
                    cache_deadline_reads=self._metrics.cache_deadline_reads + 1,
                )
                return deadline_servo_id, group.name if group else None
            if self._retry_queue:
                servo_id = self._retry_queue.pop(0)
                group = self._group_for_servo.get(servo_id)
                return servo_id, group.name if group else None

        if not self._schedule:
            return None, None

        for _ in range(len(self._schedule)):
            group_name = self._schedule[self._schedule_index % len(self._schedule)]
            self._schedule_index += 1
            group = self._read_group_by_name[group_name]
            if not group.servo_ids:
                continue
            offset = self._group_offsets[group.name] % len(group.servo_ids)
            self._group_offsets[group.name] = offset + 1
            return int(group.servo_ids[offset]), group.name
        return None, None

    def _next_deadline_servo_to_read_locked(self, now_s: float) -> int | None:
        best_servo_id: int | None = None
        best_score = 0.0
        threshold = float(self._CACHE_DEADLINE_READ_FRACTION)
        for servo_id, idx in self._id_to_index.items():
            last_update_s = float(self._last_update_time_s[idx])
            if not math.isfinite(last_update_s):
                continue
            max_age_s = float(self._max_cache_age_by_servo.get(int(servo_id), self.config.max_cache_age_s))
            if max_age_s <= 0.0:
                continue
            score = max(0.0, (float(now_s) - last_update_s) / max_age_s)
            if score >= threshold and score > best_score:
                best_score = score
                best_servo_id = int(servo_id)
        return best_servo_id

    def _advance_group_offset_after_servo(
        self, servo_id: int, group: ServoReadGroup | None
    ) -> None:
        if group is None or not group.servo_ids:
            return
        try:
            idx = tuple(int(sid) for sid in group.servo_ids).index(int(servo_id))
        except ValueError:
            return
        self._group_offsets[group.name] = idx + 1

    def _read_one(self, servo_id: int, group_name: str | None) -> None:
        start_s = time.monotonic()
        failure_recorded = False
        try:
            position = self.raw_bus.read_position(int(servo_id))
        except Exception as exc:
            position = None
            self._record_read_failure(servo_id, group_name, exc)
            failure_recorded = True

        if position is None:
            if not failure_recorded:
                self._record_read_failure(servo_id, group_name, None)
            return

        now = time.monotonic()
        idx = self._id_to_index[int(servo_id)]
        with self._lock:
            prev_pos = float(self._position_units[idx])
            prev_time = float(self._last_update_time_s[idx])
            self._position_units[idx] = float(position)
            if math.isfinite(prev_pos) and math.isfinite(prev_time) and now > prev_time:
                self._velocity_units_s[idx] = (float(position) - prev_pos) / (now - prev_time)
            self._last_update_time_s[idx] = now
            self._read_fail_count[idx] = 0
            self._retry_attempts.pop(int(servo_id), None)
            self._stale_log_time.pop(int(servo_id), None)
            self._last_read_group = group_name
            self._last_read_servo_id = int(servo_id)
            self._last_error = None
            self._metrics = replace(
                self._metrics,
                read_success=self._metrics.read_success + 1,
                latest_read_latency_s=now - start_s,
            )

    def _record_read_failure(
        self, servo_id: int, group_name: str | None, exc: Exception | None
    ) -> None:
        now = time.monotonic()
        idx = self._id_to_index[int(servo_id)]
        group = self._group_for_servo.get(int(servo_id))
        retry_cache_age_s = (
            float(group.retry_cache_age_s)
            if group and group.retry_cache_age_s is not None
            else float(self.config.retry_cache_age_s)
        )
        max_cache_age_s = (
            float(group.max_cache_age_s)
            if group and group.max_cache_age_s is not None
            else float(self.config.max_cache_age_s)
        )
        with self._lock:
            self._read_fail_count[idx] += 1
            last_update = float(self._last_update_time_s[idx])
            age_s = now - last_update if math.isfinite(last_update) else math.inf
            attempts = int(self._retry_attempts.get(int(servo_id), 0))
            if age_s >= retry_cache_age_s and attempts < max(1, int(self.config.max_read_attempts)):
                if int(servo_id) not in self._retry_queue:
                    self._retry_queue.append(int(servo_id))
                self._retry_attempts[int(servo_id)] = attempts + 1
            message = (
                f"read failed servo_id={servo_id} group={group_name} age_s={age_s:.3f} "
                f"attempts={attempts + 1}"
            )
            if exc is not None:
                message += f" error={exc!r}"
            self._last_error = message
            stale_errors = self._metrics.stale_cache_errors
            if not math.isfinite(age_s) or age_s >= max_cache_age_s:
                stale_errors += 1
            self._metrics = replace(
                self._metrics,
                read_failures=self._metrics.read_failures + 1,
                stale_cache_errors=stale_errors,
            )

        if not math.isfinite(age_s) or age_s >= max_cache_age_s:
            self._log_stale_cache(servo_id, age_s, max_cache_age_s)

    def _log_stale_cache(self, servo_id: int, age_s: float, max_cache_age_s: float) -> None:
        now = time.monotonic()
        last = self._stale_log_time.get(int(servo_id), 0.0)
        if now - last < float(self.config.stale_log_period_s):
            return
        self._stale_log_time[int(servo_id)] = now
        if not math.isfinite(float(age_s)):
            self._log_warning(
                f"Servo cache not initialized yet: servo_id={servo_id} "
                f"max_cache_age_s={max_cache_age_s:.3f}"
            )
        else:
            self._log_error(
                f"Servo cache expired: servo_id={servo_id} age_s={age_s:.3f} "
                f"max_cache_age_s={max_cache_age_s:.3f}"
            )

    def _set_last_error(self, message: str) -> None:
        with self._lock:
            self._last_error = message

    def _log_error(self, message: str) -> None:
        if self.logger is not None:
            self.logger.error(message)
        else:
            print(f"ERROR: {message}", flush=True)

    def _log_warning(self, message: str) -> None:
        if self.logger is not None:
            self.logger.warning(message)
        else:
            print(f"Warning: {message}", flush=True)


class MultiBoardServoIO:
    """Route servo IDs to independent, concurrently running board workers."""

    def __init__(
        self,
        workers_by_board: dict[str, ServoIOWorker],
        *,
        servo_ids: Sequence[int],
    ) -> None:
        if not workers_by_board:
            raise ValueError("MultiBoardServoIO requires at least one board worker")
        self.workers_by_board = dict(workers_by_board)
        self.servo_ids = tuple(int(servo_id) for servo_id in servo_ids)
        if len(set(self.servo_ids)) != len(self.servo_ids):
            raise ValueError("MultiBoardServoIO requires globally unique servo IDs")
        self._worker_by_servo_id: dict[int, ServoIOWorker] = {}
        for worker in self.workers_by_board.values():
            for servo_id in worker.servo_ids:
                sid = int(servo_id)
                if sid in self._worker_by_servo_id:
                    raise ValueError(f"servo id {sid} is assigned to multiple workers")
                self._worker_by_servo_id[sid] = worker
        missing = sorted(set(self.servo_ids) - set(self._worker_by_servo_id))
        extra = sorted(set(self._worker_by_servo_id) - set(self.servo_ids))
        if missing or extra:
            raise ValueError(
                f"MultiBoardServoIO routing mismatch: missing={missing}, extra={extra}"
            )
        self.config = next(iter(self.workers_by_board.values())).config

    def start(self) -> None:
        for worker in self.workers_by_board.values():
            worker.start()

    def stop(self, *, timeout_s: float = 1.0) -> None:
        for worker in self.workers_by_board.values():
            worker.stop(timeout_s=timeout_s)

    def submit_targets_units(
        self,
        positions_by_servo_id: dict[int, int],
        *,
        move_time_ms: int,
    ) -> None:
        targets_by_worker: dict[ServoIOWorker, dict[int, int]] = {}
        for servo_id, position in positions_by_servo_id.items():
            sid = int(servo_id)
            worker = self._worker_by_servo_id.get(sid)
            if worker is None:
                raise KeyError(f"servo id {sid} is not assigned to a board worker")
            targets_by_worker.setdefault(worker, {})[sid] = int(position)
        for worker, targets in targets_by_worker.items():
            worker.submit_targets_units(targets, move_time_ms=int(move_time_ms))

    def get_cached_servo_state(self) -> CachedServoState:
        state_by_id = {}
        last_candidates: list[tuple[float, str | None, int | None]] = []
        errors: list[str] = []
        for board_name, worker in self.workers_by_board.items():
            state = worker.get_cached_servo_state()
            for idx, servo_id in enumerate(state.servo_ids):
                state_by_id[int(servo_id)] = (state, idx)
            finite_updates = state.last_update_time_s[
                np.isfinite(state.last_update_time_s)
            ]
            latest = float(np.max(finite_updates)) if finite_updates.size else -math.inf
            group = (
                f"{board_name}:{state.last_read_group}"
                if state.last_read_group is not None
                else None
            )
            last_candidates.append((latest, group, state.last_read_servo_id))
            if state.last_error:
                errors.append(f"{board_name}: {state.last_error}")

        n = len(self.servo_ids)
        position = np.full(n, np.nan, dtype=np.float32)
        velocity = np.zeros(n, dtype=np.float32)
        age = np.full(n, np.inf, dtype=np.float32)
        fail_count = np.zeros(n, dtype=np.int32)
        update_time = np.full(n, np.nan, dtype=np.float64)
        temperature_c = np.full(n, np.nan, dtype=np.float32)
        voltage_v = np.full(n, np.nan, dtype=np.float32)
        torque_enabled_state = np.full(n, -1, dtype=np.int8)
        temperature_age_s = np.full(n, np.inf, dtype=np.float32)
        voltage_age_s = np.full(n, np.inf, dtype=np.float32)
        torque_enabled_age_s = np.full(n, np.inf, dtype=np.float32)
        health_read_fail_count = np.zeros(n, dtype=np.int32)
        for out_idx, servo_id in enumerate(self.servo_ids):
            state, state_idx = state_by_id[servo_id]
            position[out_idx] = state.position_units[state_idx]
            velocity[out_idx] = state.velocity_units_s[state_idx]
            age[out_idx] = state.position_age_s[state_idx]
            fail_count[out_idx] = state.read_fail_count[state_idx]
            update_time[out_idx] = state.last_update_time_s[state_idx]
            temperature_c[out_idx] = state.temperature_c[state_idx]
            voltage_v[out_idx] = state.voltage_v[state_idx]
            torque_enabled_state[out_idx] = state.torque_enabled_state[state_idx]
            temperature_age_s[out_idx] = state.temperature_age_s[state_idx]
            voltage_age_s[out_idx] = state.voltage_age_s[state_idx]
            torque_enabled_age_s[out_idx] = state.torque_enabled_age_s[state_idx]
            health_read_fail_count[out_idx] = state.health_read_fail_count[state_idx]

        _, last_group, last_servo_id = max(last_candidates, key=lambda item: item[0])
        return CachedServoState(
            servo_ids=self.servo_ids,
            position_units=position,
            velocity_units_s=velocity,
            position_age_s=age,
            read_fail_count=fail_count,
            last_update_time_s=update_time,
            temperature_c=temperature_c,
            voltage_v=voltage_v,
            torque_enabled_state=torque_enabled_state,
            temperature_age_s=temperature_age_s,
            voltage_age_s=voltage_age_s,
            torque_enabled_age_s=torque_enabled_age_s,
            health_read_fail_count=health_read_fail_count,
            last_read_group=last_group,
            last_read_servo_id=last_servo_id,
            last_error="; ".join(errors) or None,
        )

    def get_metrics(self) -> ServoIOMetrics:
        metrics = [worker.get_metrics() for worker in self.workers_by_board.values()]
        return ServoIOMetrics(
            write_targets_submitted=max(m.write_targets_submitted for m in metrics),
            write_targets_replaced=sum(m.write_targets_replaced for m in metrics),
            write_commands=sum(m.write_commands for m in metrics),
            write_commands_skipped=sum(m.write_commands_skipped for m in metrics),
            write_failures=sum(m.write_failures for m in metrics),
            cache_deadline_reads=sum(m.cache_deadline_reads for m in metrics),
            forced_read_after_write=sum(m.forced_read_after_write for m in metrics),
            forced_read_after_write_missed=sum(
                m.forced_read_after_write_missed for m in metrics
            ),
            read_success=sum(m.read_success for m in metrics),
            read_failures=sum(m.read_failures for m in metrics),
            stale_cache_errors=sum(m.stale_cache_errors for m in metrics),
            health_read_success=sum(m.health_read_success for m in metrics),
            health_read_failures=sum(m.health_read_failures for m in metrics),
            unexpected_unload_events=sum(
                m.unexpected_unload_events for m in metrics
            ),
            unload_commands=sum(m.unload_commands for m in metrics),
            unload_failures=sum(m.unload_failures for m in metrics),
            unload_unverified=sum(m.unload_unverified for m in metrics),
            latest_write_queue_latency_s=max(
                m.latest_write_queue_latency_s for m in metrics
            ),
            latest_write_latency_s=max(m.latest_write_latency_s for m in metrics),
            latest_read_latency_s=max(m.latest_read_latency_s for m in metrics),
            latest_health_read_latency_s=max(
                m.latest_health_read_latency_s for m in metrics
            ),
        )

    def unload_servos(self, servo_ids: Sequence[int]) -> dict[str, object]:
        ids_by_worker: dict[ServoIOWorker, list[int]] = {}
        for servo_id in servo_ids:
            sid = int(servo_id)
            worker = self._worker_by_servo_id.get(sid)
            if worker is None:
                raise KeyError(f"servo id {sid} is not assigned to a board worker")
            ids_by_worker.setdefault(worker, []).append(sid)
        reports = [worker.unload_servos(ids) for worker, ids in ids_by_worker.items()]
        list_keys = (
            "attempted_servo_ids",
            "commanded_servo_ids",
            "confirmed_servo_ids",
            "unverified_servo_ids",
            "failed_servo_ids",
            "errors",
        )
        return {
            key: [item for report in reports for item in report.get(key, [])]
            for key in list_keys
        }

    def close(self) -> None:
        for worker in self.workers_by_board.values():
            worker.close()


__all__ = [
    "CachedServoState",
    "ServoIOWorker",
    "ServoIOWorkerConfig",
    "ServoIOMetrics",
    "MultiBoardServoIO",
    "ServoReadGroup",
]
