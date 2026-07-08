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


@dataclass(frozen=True)
class CachedServoState:
    servo_ids: tuple[int, ...]
    position_units: np.ndarray
    velocity_units_s: np.ndarray
    position_age_s: np.ndarray
    read_fail_count: np.ndarray
    last_update_time_s: np.ndarray
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
    forced_read_after_write: int = 0
    forced_read_after_write_missed: int = 0
    read_success: int = 0
    read_failures: int = 0
    stale_cache_errors: int = 0
    latest_write_queue_latency_s: float = 0.0
    latest_write_latency_s: float = 0.0
    latest_read_latency_s: float = 0.0


class ServoIOWorker:
    """Single-owner runtime IO worker for raw HTD/Hiwonder servo buses.

    The worker accepts latest-wins target writes and otherwise polls one servo
    position at a time into a full cache.
    """

    def __init__(
        self,
        raw_bus: RawServoBus,
        config: ServoIOWorkerConfig,
        *,
        logger: Logger | None = None,
    ) -> None:
        self.raw_bus = raw_bus
        self.config = config
        self.logger = logger

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

        n = len(self.servo_ids)
        self._position_units = np.full(n, np.nan, dtype=np.float32)
        self._velocity_units_s = np.zeros(n, dtype=np.float32)
        self._last_update_time_s = np.full(n, np.nan, dtype=np.float64)
        self._read_fail_count = np.zeros(n, dtype=np.int32)

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

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="ServoIOWorker", daemon=True)
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
            last_group = self._last_read_group
            last_read_servo_id = self._last_read_servo_id
            last_error = self._last_error
        age = now - last_update
        age[~np.isfinite(last_update)] = np.inf
        return CachedServoState(
            servo_ids=self.servo_ids,
            position_units=position,
            velocity_units_s=velocity,
            position_age_s=age.astype(np.float32),
            read_fail_count=fail_count,
            last_update_time_s=last_update,
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

            if self._read_next_available():
                continue

            self._wake.wait(timeout=max(0.0, float(self.config.idle_sleep_s)))
            self._wake.clear()

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


__all__ = [
    "CachedServoState",
    "ServoIOWorker",
    "ServoIOWorkerConfig",
    "ServoIOMetrics",
    "ServoReadGroup",
]
