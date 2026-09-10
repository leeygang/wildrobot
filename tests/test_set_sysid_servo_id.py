from __future__ import annotations

import pytest

from runtime.scripts.set_sysid_servo_id import (
    assign_sysid_servo_id,
    detect_single_servo_id,
    scan_servo_ids,
)


class FakeBus:
    class Config:
        min_servo_id = 0
        max_servo_id = 120

    def __init__(self, *servo_ids: int) -> None:
        self.config = self.Config()
        self.servo_ids = {int(servo_id) for servo_id in servo_ids}
        self.id_writes: list[tuple[int, int]] = []

    def read_id(self, target_id: int) -> int | None:
        target = int(target_id)
        return target if target in self.servo_ids else None

    def write_id(self, old_id: int, new_id: int) -> None:
        assert int(old_id) in self.servo_ids
        self.id_writes.append((int(old_id), int(new_id)))
        self.servo_ids.remove(int(old_id))
        self.servo_ids.add(int(new_id))


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
