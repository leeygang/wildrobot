import math
import threading
import time

from runtime.wr_runtime.hardware.servo_io_worker import (
    MultiBoardServoIO,
    ServoIOWorker,
    ServoIOWorkerConfig,
    ServoReadGroup,
)


class FakeTransport:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class FakeRawBus:
    def __init__(self, positions, *, write_sleep_s=0.0):
        self.positions = dict(positions)
        self.writes = []
        self.reads = []
        self.transport = FakeTransport()
        self.write_sleep_s = float(write_sleep_s)
        self.temperatures = {int(servo_id): 40 for servo_id in positions}
        self.voltages = {int(servo_id): 11.1 for servo_id in positions}
        self.loaded = {int(servo_id): True for servo_id in positions}
        self.health_reads = []
        self.unloads = []

    def read_position(self, servo_id: int):
        self.reads.append(int(servo_id))
        return self.positions.get(int(servo_id))

    def move_time_write(self, servo_id: int, position: int, time_ms: int):
        if self.write_sleep_s > 0.0:
            time.sleep(self.write_sleep_s)
        self.writes.append((int(servo_id), int(position), int(time_ms)))

    def read_temperature_c(self, servo_id: int):
        self.health_reads.append(("temperature", int(servo_id)))
        return self.temperatures.get(int(servo_id))

    def read_voltage_v(self, servo_id: int):
        self.health_reads.append(("voltage", int(servo_id)))
        return self.voltages.get(int(servo_id))

    def read_loaded(self, servo_id: int):
        self.health_reads.append(("torque_enabled", int(servo_id)))
        return self.loaded.get(int(servo_id))

    def unload(self, servo_id: int):
        self.unloads.append(int(servo_id))
        self.loaded[int(servo_id)] = False


class FakeLogger:
    def __init__(self):
        self.errors = []
        self.warnings = []

    def error(self, message: str):
        self.errors.append(str(message))

    def warning(self, message: str):
        self.warnings.append(str(message))


class ConcurrencyProbe:
    def __init__(self):
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def enter(self):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)

    def exit(self):
        with self.lock:
            self.active -= 1


class ConcurrentFakeRawBus(FakeRawBus):
    def __init__(self, positions, probe):
        super().__init__(positions)
        self.probe = probe

    def move_time_write(self, servo_id: int, position: int, time_ms: int):
        self.probe.enter()
        try:
            time.sleep(0.03)
            super().move_time_write(servo_id, position, time_ms)
        finally:
            self.probe.exit()


def _wait_until(predicate, *, timeout_s=0.5):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.001)
    return False


def test_worker_reads_one_servo_into_cache():
    raw_bus = FakeRawBus({3: 501})
    worker = ServoIOWorker(
        raw_bus,
        ServoIOWorkerConfig(
            read_groups=(ServoReadGroup(name="single", servo_ids=(3,)),),
            idle_sleep_s=0.0001,
        ),
    )

    worker.start()
    try:
        assert _wait_until(lambda: worker.get_metrics().read_success >= 1)
    finally:
        worker.stop()

    state = worker.get_cached_servo_state()
    assert state.servo_ids == (3,)
    assert int(state.position_units[0]) == 501
    assert state.last_read_group == "single"
    assert state.last_read_servo_id == 3
    assert raw_bus.reads


def test_worker_submits_target_to_raw_bus():
    raw_bus = FakeRawBus({3: 501})
    worker = ServoIOWorker(
        raw_bus,
        ServoIOWorkerConfig(
            read_groups=(ServoReadGroup(name="single", servo_ids=(3,)),),
            idle_sleep_s=0.0001,
        ),
    )

    worker.start()
    try:
        worker.submit_targets_units({3: 520}, move_time_ms=20)
        assert _wait_until(lambda: raw_bus.writes == [(3, 520, 20)])
    finally:
        worker.stop()

    metrics = worker.get_metrics()
    assert metrics.write_targets_submitted == 1
    assert metrics.write_commands == 1


def test_worker_polls_temperature_voltage_and_torque_enable_without_blocking_startup():
    raw_bus = FakeRawBus({3: 501})
    raw_bus.temperatures[3] = 52
    raw_bus.voltages[3] = 10.6
    worker = ServoIOWorker(
        raw_bus,
        ServoIOWorkerConfig(
            servo_ids=(3,),
            health_poll_interval_s=0.003,
            idle_sleep_s=0.0001,
        ),
    )

    worker.start()
    try:
        assert _wait_until(lambda: worker.get_metrics().health_read_success >= 3)
    finally:
        worker.stop()

    state = worker.get_cached_servo_state()
    assert state.position_units.tolist() == [501.0]
    assert state.temperature_c.tolist() == [52.0]
    assert state.voltage_v.tolist() == [10.600000381469727]
    assert state.torque_enabled_state.tolist() == [1]
    assert state.health_read_fail_count.tolist() == [0]


def test_health_polling_is_rate_limited_and_yields_to_pending_writes():
    raw_bus = FakeRawBus({1: 501, 2: 502})
    worker = ServoIOWorker(
        raw_bus,
        ServoIOWorkerConfig(
            servo_ids=(1, 2),
            health_poll_interval_s=1.2,
        ),
    )
    worker._last_update_time_s[:] = time.monotonic()
    worker._next_health_read_s = time.monotonic() - 1.0

    worker.submit_targets_units({1: 520}, move_time_ms=20)
    assert worker._health_read_due() is False
    assert worker._pop_pending_target() is not None

    before = time.monotonic()
    assert worker._health_read_due() is True
    worker._read_next_health()

    # Two servos x three fields: at most one transaction every 0.2 seconds,
    # hence each field for each servo is sampled at most once per 1.2 seconds.
    assert worker._next_health_read_s - before >= 0.19
    assert len(raw_bus.health_reads) == 1


def test_worker_counts_unexpected_torque_disable_transition_once():
    raw_bus = FakeRawBus({3: 501})
    raw_bus.loaded[3] = False
    worker = ServoIOWorker(
        raw_bus,
        ServoIOWorkerConfig(servo_ids=(3,)),
    )
    worker._last_written_target_by_servo[3] = (501, 20)

    worker._read_health_one(3, "torque_enabled")
    worker._read_health_one(3, "torque_enabled")

    assert worker.get_cached_servo_state().torque_enabled_state.tolist() == [0]
    assert worker.get_metrics().unexpected_unload_events == 1


def test_initial_unloaded_state_is_not_an_event_before_first_command():
    raw_bus = FakeRawBus({3: 501})
    raw_bus.loaded[3] = False
    worker = ServoIOWorker(raw_bus, ServoIOWorkerConfig(servo_ids=(3,)))

    worker._read_health_one(3, "torque_enabled")

    assert worker.get_cached_servo_state().torque_enabled_state.tolist() == [0]
    assert worker.get_metrics().unexpected_unload_events == 0

    worker._last_written_target_by_servo[3] = (501, 20)
    worker._read_health_one(3, "torque_enabled")
    assert worker.get_metrics().unexpected_unload_events == 1


def test_unload_report_continues_after_command_failure_and_verifies_state():
    class PartiallyFailingRawBus(FakeRawBus):
        def unload(self, servo_id: int):
            if int(servo_id) == 2:
                raise OSError("write failed")
            super().unload(servo_id)

    raw_bus = PartiallyFailingRawBus({1: 501, 2: 502})
    worker = ServoIOWorker(raw_bus, ServoIOWorkerConfig(servo_ids=(1, 2)))

    report = worker.unload_servos([1, 2])

    assert report["attempted_servo_ids"] == [1, 2]
    assert report["confirmed_servo_ids"] == [1]
    assert report["failed_servo_ids"] == [2]
    assert "write failed" in report["errors"][0]
    assert worker.get_metrics().unload_commands == 1
    assert worker.get_metrics().unload_failures == 1


def test_multi_board_io_routes_targets_and_runs_board_writes_concurrently():
    probe = ConcurrencyProbe()
    left_bus = ConcurrentFakeRawBus({1: 501}, probe)
    right_bus = ConcurrentFakeRawBus({2: 502}, probe)
    left = ServoIOWorker(
        left_bus,
        ServoIOWorkerConfig(servo_ids=(1,), idle_sleep_s=0.0001),
    )
    right = ServoIOWorker(
        right_bus,
        ServoIOWorkerConfig(servo_ids=(2,), idle_sleep_s=0.0001),
    )
    multi = MultiBoardServoIO(
        {"left_leg_board": left, "right_leg_board": right},
        servo_ids=(1, 2),
    )
    multi.submit_targets_units({1: 510, 2: 520}, move_time_ms=20)

    multi.start()
    try:
        assert _wait_until(lambda: left_bus.writes and right_bus.writes)
        assert _wait_until(lambda: multi.get_metrics().read_success >= 2)
    finally:
        multi.close()

    assert left_bus.writes == [(1, 510, 20)]
    assert right_bus.writes == [(2, 520, 20)]
    assert probe.max_active == 2
    state = multi.get_cached_servo_state()
    assert state.servo_ids == (1, 2)
    assert state.position_units.tolist() == [501.0, 502.0]
    assert state.last_read_group is not None


def test_worker_skips_unchanged_successful_target():
    raw_bus = FakeRawBus({3: 501})
    worker = ServoIOWorker(
        raw_bus,
        ServoIOWorkerConfig(
            read_groups=(ServoReadGroup(name="single", servo_ids=(3,)),),
            idle_sleep_s=0.0001,
        ),
    )

    worker.start()
    try:
        worker.submit_targets_units({3: 520}, move_time_ms=20)
        assert _wait_until(lambda: raw_bus.writes == [(3, 520, 20)])
        worker.submit_targets_units({3: 520}, move_time_ms=20)
        assert _wait_until(lambda: worker.get_metrics().write_commands_skipped >= 1)
    finally:
        worker.stop()

    metrics = worker.get_metrics()
    assert raw_bus.writes == [(3, 520, 20)]
    assert metrics.write_targets_submitted == 2
    assert metrics.write_commands == 1
    assert metrics.write_commands_skipped == 1


def test_worker_skips_target_within_write_deadband():
    raw_bus = FakeRawBus({3: 501})
    worker = ServoIOWorker(
        raw_bus,
        ServoIOWorkerConfig(
            read_groups=(ServoReadGroup(name="single", servo_ids=(3,)),),
            write_deadband_units=3,
            idle_sleep_s=0.0001,
        ),
    )

    worker.start()
    try:
        worker.submit_targets_units({3: 520}, move_time_ms=20)
        assert _wait_until(lambda: raw_bus.writes == [(3, 520, 20)])
        worker.submit_targets_units({3: 522}, move_time_ms=20)
        assert _wait_until(lambda: worker.get_metrics().write_commands_skipped >= 1)
        worker.submit_targets_units({3: 524}, move_time_ms=20)
        assert _wait_until(lambda: raw_bus.writes == [(3, 520, 20), (3, 524, 20)])
    finally:
        worker.stop()

    metrics = worker.get_metrics()
    assert metrics.write_targets_submitted == 3
    assert metrics.write_commands == 2
    assert metrics.write_commands_skipped == 1


def test_worker_reads_between_continuous_target_writes():
    raw_bus = FakeRawBus({1: 501, 2: 502}, write_sleep_s=0.002)
    worker = ServoIOWorker(
        raw_bus,
        ServoIOWorkerConfig(
            read_groups=(ServoReadGroup(name="legs", servo_ids=(1, 2)),),
            idle_sleep_s=0.0001,
        ),
    )
    stop_spam = threading.Event()

    def spam_targets():
        i = 0
        while not stop_spam.is_set():
            worker.submit_targets_units(
                {1: 520 + (i % 20), 2: 620 + (i % 20)},
                move_time_ms=20,
            )
            i += 1
            time.sleep(0.0001)

    worker.submit_targets_units({1: 520, 2: 620}, move_time_ms=20)
    spam_thread = threading.Thread(target=spam_targets)
    spam_thread.start()
    worker.start()
    try:
        assert _wait_until(lambda: worker.get_metrics().write_commands >= 4)
        assert _wait_until(lambda: worker.get_metrics().read_success >= 1)
    finally:
        stop_spam.set()
        spam_thread.join(timeout=0.5)
        worker.stop()

    metrics = worker.get_metrics()
    assert metrics.forced_read_after_write >= 2
    assert metrics.latest_write_queue_latency_s >= 0.0


def test_worker_reads_minimum_count_after_write_before_next_target():
    raw_bus = FakeRawBus({1: 501, 2: 502, 3: 503})
    worker = ServoIOWorker(
        raw_bus,
        ServoIOWorkerConfig(
            read_groups=(ServoReadGroup(name="legs", servo_ids=(1, 2, 3)),),
            min_reads_after_write=2,
            idle_sleep_s=0.0001,
        ),
    )

    worker.submit_targets_units({1: 520}, move_time_ms=20)
    worker.start()
    try:
        assert _wait_until(lambda: worker.get_metrics().write_commands >= 1)
        assert _wait_until(lambda: worker.get_metrics().forced_read_after_write >= 2)
    finally:
        worker.stop()

    metrics = worker.get_metrics()
    assert metrics.forced_read_after_write >= 2
    assert len(raw_bus.reads) >= 2


def test_worker_prioritizes_servo_near_cache_deadline():
    raw_bus = FakeRawBus({1: 501, 2: 502, 21: 521})
    worker = ServoIOWorker(
        raw_bus,
        ServoIOWorkerConfig(
            read_groups=(
                ServoReadGroup(name="leg", servo_ids=(1, 2), max_cache_age_s=0.16),
                ServoReadGroup(name="arm", servo_ids=(21,), max_cache_age_s=1.25),
            ),
            read_group_schedule=("arm", "leg"),
        ),
    )
    now = time.monotonic()
    worker._last_update_time_s[worker._id_to_index[1]] = now - 0.13
    worker._last_update_time_s[worker._id_to_index[2]] = now - 0.01
    worker._last_update_time_s[worker._id_to_index[21]] = now - 0.01

    servo_id, group_name = worker._next_servo_to_read()

    assert servo_id == 1
    assert group_name == "leg"
    assert worker.get_metrics().cache_deadline_reads == 1


def test_worker_close_closes_transport():
    raw_bus = FakeRawBus({3: 501})
    worker = ServoIOWorker(
        raw_bus,
        ServoIOWorkerConfig(read_groups=(ServoReadGroup(name="single", servo_ids=(3,)),)),
    )

    worker.close()

    assert raw_bus.transport.closed is True


def test_uninitialized_cache_logs_warning_not_error():
    raw_bus = FakeRawBus({3: None})
    logger = FakeLogger()
    worker = ServoIOWorker(
        raw_bus,
        ServoIOWorkerConfig(read_groups=(ServoReadGroup(name="single", servo_ids=(3,)),)),
        logger=logger,
    )

    worker._log_stale_cache(3, math.inf, 0.16)

    assert logger.errors == []
    assert logger.warnings == [
        "Servo cache not initialized yet: servo_id=3 max_cache_age_s=0.160"
    ]


def test_expired_cache_logs_error():
    raw_bus = FakeRawBus({3: None})
    logger = FakeLogger()
    worker = ServoIOWorker(
        raw_bus,
        ServoIOWorkerConfig(read_groups=(ServoReadGroup(name="single", servo_ids=(3,)),)),
        logger=logger,
    )

    worker._log_stale_cache(3, 0.2, 0.16)

    assert logger.warnings == []
    assert logger.errors == [
        "Servo cache expired: servo_id=3 age_s=0.200 max_cache_age_s=0.160"
    ]
