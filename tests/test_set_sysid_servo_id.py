from __future__ import annotations

import pytest

from runtime.scripts.set_sysid_servo_id import (
    assign_sysid_servo_id,
    detect_single_servo_id,
    scan_servo_ids,
    set_servo_unit,
)


class FakeBus:
    class Config:
        min_servo_id = 0
        max_servo_id = 120

    def __init__(self, *servo_ids: int) -> None:
        self.config = self.Config()
        self.servo_ids = {int(servo_id) for servo_id in servo_ids}
        self.id_writes: list[tuple[int, int]] = []
        self.position = 500
        self.loaded = False
        self.position_writes: list[tuple[int, int]] = []
        self.load_count = 0
        self.unload_count = 0

    def read_id(self, target_id: int) -> int | None:
        target = int(target_id)
        return target if target in self.servo_ids else None

    def write_id(self, old_id: int, new_id: int) -> None:
        assert int(old_id) in self.servo_ids
        self.id_writes.append((int(old_id), int(new_id)))
        self.servo_ids.remove(int(old_id))
        self.servo_ids.add(int(new_id))

    def read_position(self, servo_id: int) -> int | None:
        return self.position if int(servo_id) in self.servo_ids else None

    def move_time_write(self, servo_id: int, position: int, time_ms: int) -> None:
        assert int(servo_id) in self.servo_ids
        self.position_writes.append((int(position), int(time_ms)))
        self.position = int(position)

    def load(self, servo_id: int) -> None:
        assert int(servo_id) in self.servo_ids
        self.load_count += 1
        self.loaded = True

    def unload(self, servo_id: int) -> None:
        assert int(servo_id) in self.servo_ids
        self.unload_count += 1
        self.loaded = False

    def read_loaded(self, servo_id: int) -> bool | None:
        return self.loaded if int(servo_id) in self.servo_ids else None


def test_scan_servo_ids_returns_every_targeted_response() -> None:
    assert scan_servo_ids(FakeBus(7, 21, 100)) == (7, 21, 100)


def test_detect_single_servo_requires_consistent_repeated_response() -> None:
    bus = FakeBus(7)

    assert detect_single_servo_id(bus, sleep_fn=lambda _seconds: None) == 7


def test_detect_single_servo_rejects_multiple_replies() -> None:
    with pytest.raises(RuntimeError, match="more than one servo is connected"):
        detect_single_servo_id(FakeBus(7, 21), sleep_fn=lambda _seconds: None)


def test_assign_sysid_servo_id_writes_and_verifies_id_100() -> None:
    bus = FakeBus(7)

    result = assign_sysid_servo_id(
        bus,
        new_servo_id=100,
        input_fn=lambda _prompt: "SET 100",
        sleep_fn=lambda _seconds: None,
    )

    assert result == 100
    assert bus.servo_ids == {100}
    assert bus.id_writes == [(7, 100)]


def test_assign_sysid_servo_id_is_noop_when_already_100() -> None:
    bus = FakeBus(100)

    result = assign_sysid_servo_id(
        bus,
        new_servo_id=100,
        sleep_fn=lambda _seconds: None,
    )

    assert result == 100
    assert bus.id_writes == []


def test_assign_sysid_servo_id_supports_requested_nondefault_id() -> None:
    bus = FakeBus(7)

    result = assign_sysid_servo_id(
        bus,
        new_servo_id=42,
        input_fn=lambda _prompt: "SET 42",
        sleep_fn=lambda _seconds: None,
    )

    assert result == 42
    assert bus.servo_ids == {42}
    assert bus.id_writes == [(7, 42)]


def test_assign_prints_current_id_before_confirmation_and_write(capsys) -> None:
    bus = FakeBus(7)

    def confirm(_prompt: str) -> str:
        assert "Current servo ID: 7" in capsys.readouterr().out
        assert bus.id_writes == []
        return "SET 42"

    assign_sysid_servo_id(
        bus,
        new_servo_id=42,
        input_fn=confirm,
        sleep_fn=lambda _seconds: None,
    )


def test_set_servo_unit_reports_moves_verifies_and_unloads(capsys) -> None:
    bus = FakeBus(100)
    bus.position = 400
    sleeps: list[float] = []

    measured = set_servo_unit(
        bus,
        servo_id=100,
        target_unit=500,
        sleep_fn=sleeps.append,
    )

    assert measured == 500
    assert bus.position_writes == [(400, 500), (500, 1200)]
    assert sleeps == [pytest.approx(1.45)]
    assert bus.loaded is False
    assert bus.load_count == 1
    assert bus.unload_count == 1
    output = capsys.readouterr().out
    assert "Current servo unit: 400" in output
    assert "PASS: servo reached unit 500" in output


@pytest.mark.parametrize("target_unit", [-1, 1001])
def test_set_servo_unit_rejects_out_of_range_target(target_unit: int) -> None:
    with pytest.raises(ValueError, match="between 0 and 1000"):
        set_servo_unit(
            FakeBus(100),
            servo_id=100,
            target_unit=target_unit,
            sleep_fn=lambda _seconds: None,
        )


def test_set_servo_unit_is_noop_when_already_at_target() -> None:
    bus = FakeBus(100)

    measured = set_servo_unit(
        bus,
        servo_id=100,
        target_unit=500,
        sleep_fn=lambda _seconds: None,
    )

    assert measured == 500
    assert bus.position_writes == []
    assert bus.loaded is False
    assert bus.load_count == 0
    assert bus.unload_count == 1
