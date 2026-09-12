#!/usr/bin/env python3
"""Detect one isolated HTD-45H servo and configure its ID or raw position.

The tool scans every valid address individually so it can reject a board with
multiple distinct servo IDs before writing. Raw-position motion is optional and
requires an explicit ``--set-unit`` target.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Callable


_REPO_ROOT = Path(__file__).resolve().parents[2]
_RUNTIME_ROOT = _REPO_ROOT / "runtime"
if str(_RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(_RUNTIME_ROOT))

from wr_runtime.hardware.hiwonder_ttl_bus import (  # noqa: E402
    RawServoBus,
    RawServoBusConfig,
    SerialTransport,
    SerialTransportConfig,
)


SERVO_MIN_UNIT = 0
SERVO_MAX_UNIT = 1000
SERVO_TRAVEL_DEG = 240.0
SET_UNIT_SPEED_DEG_S = 20.0
SET_UNIT_MIN_MOVE_MS = 500
SET_UNIT_SETTLE_S = 0.25
SET_UNIT_TOLERANCE = 5


def scan_servo_ids(bus: RawServoBus) -> tuple[int, ...]:
    """Probe each address and return every distinct responding servo ID."""

    detected: list[int] = []
    for servo_id in range(
        int(bus.config.min_servo_id), int(bus.config.max_servo_id) + 1
    ):
        reported_id = bus.read_id(servo_id)
        if reported_id is None:
            continue
        if int(reported_id) != servo_id:
            raise RuntimeError(
                f"servo address {servo_id} returned inconsistent ID {reported_id}"
            )
        detected.append(servo_id)
    return tuple(detected)


def detect_single_servo_id(
    bus: RawServoBus,
    *,
    confirmation_reads: int = 2,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> int:
    """Return the sole scanned servo ID or fail without writing."""

    detected = scan_servo_ids(bus)
    if len(detected) > 1:
        raise RuntimeError(
            f"more than one servo is connected: IDs={list(detected)}; "
            "disconnect all but the SysID fixture servo"
        )
    if not detected:
        raise RuntimeError(
            "no servo was detected; check board power, port, and wiring"
        )

    current_id = int(detected[0])
    for attempt in range(int(confirmation_reads)):
        if bus.read_id(current_id) != current_id:
            raise RuntimeError(
                f"servo ID {current_id} did not answer repeated verification"
            )
        if attempt + 1 < int(confirmation_reads):
            sleep_fn(0.03)
    return current_id


def assign_sysid_servo_id(
    bus: RawServoBus,
    *,
    new_servo_id: int,
    input_fn: Callable[[str], str] = input,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> int:
    """Detect one servo, set the requested ID, and verify old/new addresses."""

    target_id = int(new_servo_id)
    if target_id < 1 or target_id > 253:
        raise ValueError("new servo ID must be between 1 and 253")

    print("Scanning servo IDs 0-253; this takes about two seconds...")
    current_id = detect_single_servo_id(bus, sleep_fn=sleep_fn)
    print(f"Current servo ID: {current_id}")
    if current_id == target_id:
        print(f"Servo already uses requested ID {target_id}; no write needed.")
        return current_id

    confirmation = input_fn(
        f"Change servo ID {current_id} -> {target_id}? Type SET {target_id}: "
    ).strip()
    if confirmation != f"SET {target_id}":
        raise RuntimeError("servo ID change was not confirmed")

    bus.write_id(current_id, target_id)
    sleep_fn(0.2)

    verified_id = bus.read_id(target_id)
    if verified_id != target_id:
        raise RuntimeError(
            f"ID write was sent but ID {target_id} did not answer verification; "
            "power-cycle the board and scan again before sending any motion command"
        )
    if bus.read_id(current_id) is not None:
        raise RuntimeError(
            f"old servo ID {current_id} still responds; more than one servo may be connected"
        )
    if detect_single_servo_id(bus, sleep_fn=sleep_fn) != target_id:
        raise RuntimeError(f"post-write scan did not return only servo ID {target_id}")

    print(f"PASS: servo ID changed from {current_id} to {target_id} and verified.")
    return target_id


def set_servo_unit(
    bus: RawServoBus,
    *,
    servo_id: int,
    target_unit: int,
    input_fn: Callable[[str], str] = input,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> int:
    """Move one isolated servo, hold it loaded, then unload on confirmation."""

    target = int(target_unit)
    if target < SERVO_MIN_UNIT or target > SERVO_MAX_UNIT:
        raise ValueError(
            f"servo unit must be between {SERVO_MIN_UNIT} and {SERVO_MAX_UNIT}"
        )

    current = bus.read_position(int(servo_id))
    if current is None:
        raise RuntimeError(f"failed to read current unit from servo ID {servo_id}")
    current = int(current)
    print(f"Current servo unit: {current}")

    delta_deg = (
        abs(target - current)
        * SERVO_TRAVEL_DEG
        / float(SERVO_MAX_UNIT - SERVO_MIN_UNIT)
    )
    move_time_ms = max(
        SET_UNIT_MIN_MOVE_MS,
        int(math.ceil(delta_deg / SET_UNIT_SPEED_DEG_S * 1000.0)),
    )
    try:
        bus.move_time_write(int(servo_id), current, SET_UNIT_MIN_MOVE_MS)
        bus.load(int(servo_id))
        if bus.read_loaded(int(servo_id)) is not True:
            raise RuntimeError("servo torque did not enable before position motion")

        if current == target:
            print(f"Servo already uses requested unit {target}; no motion needed.")
        else:
            print(
                f"Moving servo ID {servo_id}: {current} -> {target} units "
                f"over {move_time_ms}ms, then holding position."
            )
            bus.move_time_write(int(servo_id), target, move_time_ms)
            sleep_fn(move_time_ms / 1000.0 + SET_UNIT_SETTLE_S)

        measured = bus.read_position(int(servo_id))
        if measured is None:
            raise RuntimeError("failed to read servo position after motion")
        measured = int(measured)
        if bus.read_loaded(int(servo_id)) is not True:
            raise RuntimeError(
                f"servo torque disabled during motion at unit {measured}"
            )
        error = abs(measured - target)
        if error > SET_UNIT_TOLERANCE:
            raise RuntimeError(
                f"servo did not reach requested unit: measured={measured} "
                f"target={target} error={error} units"
            )
        print(
            f"Servo reached unit {measured} and remains loaded. "
            "Keep the mechanism clear."
        )
        while input_fn("Type y to unload servo torque: ").strip().lower() != "y":
            if bus.read_loaded(int(servo_id)) is not True:
                raise RuntimeError("servo torque disabled while awaiting confirmation")
            print("Servo remains loaded; type y when ready to unload.")
        if bus.read_loaded(int(servo_id)) is not True:
            raise RuntimeError("servo torque disabled while awaiting confirmation")
    finally:
        bus.unload(int(servo_id))
        if bus.read_loaded(int(servo_id)) is not False:
            raise RuntimeError("servo torque did not disable after position motion")

    print(f"PASS: servo reached unit {measured}; torque disabled after confirmation.")
    return measured


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect one isolated HTD-45H servo, set its requested ID, and "
            "optionally move it to a verified raw position."
        )
    )
    parser.add_argument(
        "--board-port",
        "--board_port",
        dest="board_port",
        required=True,
        help="Serial port for the TTL board connected only to the fixture servo.",
    )
    parser.add_argument(
        "--servo-id",
        type=int,
        required=True,
        help="New servo ID to assign (1-253).",
    )
    parser.add_argument(
        "--set-unit",
        type=int,
        default=None,
        metavar="UNIT",
        help="After ID setup, move the isolated servo to raw unit 0-1000.",
    )
    parser.add_argument("--baudrate", type=int, default=115200)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if int(args.baudrate) <= 0:
        raise ValueError("--baudrate must be positive")
    if int(args.servo_id) < 1 or int(args.servo_id) > 253:
        raise ValueError("--servo-id must be between 1 and 253")
    if args.set_unit is not None and not (
        SERVO_MIN_UNIT <= int(args.set_unit) <= SERVO_MAX_UNIT
    ):
        raise ValueError(
            f"--set-unit must be between {SERVO_MIN_UNIT} and {SERVO_MAX_UNIT}"
        )

    print("HTD-45H SysID fixture servo-ID setup")
    print(f"  board_port={args.board_port}")
    print(f"  requested_servo_id={int(args.servo_id)}")
    if args.set_unit is None:
        print("  no position or torque command will be sent")
    else:
        print(f"  requested_servo_unit={int(args.set_unit)}")
        print(
            "  motion enabled: clear the horn/fixture; torque remains enabled "
            "until you type y to unload"
        )

    transport = SerialTransport(
        SerialTransportConfig(
            port=str(args.board_port),
            baudrate=int(args.baudrate),
        )
    )
    bus = RawServoBus(transport, RawServoBusConfig())
    try:
        configured_id = assign_sysid_servo_id(
            bus, new_servo_id=int(args.servo_id)
        )
        if args.set_unit is not None:
            set_servo_unit(
                bus,
                servo_id=configured_id,
                target_unit=int(args.set_unit),
            )
    except Exception as exc:
        print(f"Servo-ID setup failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        transport.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
