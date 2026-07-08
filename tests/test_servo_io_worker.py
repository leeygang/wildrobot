import math
import threading
import time

from runtime.wr_runtime.hardware.servo_io_worker import (
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

    def read_position(self, servo_id: int):
        self.reads.append(int(servo_id))
        return self.positions.get(int(servo_id))

    def move_time_write(self, servo_id: int, position: int, time_ms: int):
        if self.write_sleep_s > 0.0:
            time.sleep(self.write_sleep_s)
        self.writes.append((int(servo_id), int(position), int(time_ms)))


class FakeLogger:
    def __init__(self):
        self.errors = []
        self.warnings = []

    def error(self, message: str):
        self.errors.append(str(message))

    def warning(self, message: str):
        self.warnings.append(str(message))


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
    assert metrics.forced_read_after_write >= 1
    assert metrics.latest_write_queue_latency_s >= 0.0


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
