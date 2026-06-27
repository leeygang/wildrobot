#!/usr/bin/env python3

from __future__ import annotations

import argparse
from typing import List

from wr_runtime.hardware.hiwonder_board_controller import HiwonderBoardController


def _parse_ids(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _parse_baudrates(text: str) -> List[int]:
    baudrates = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not baudrates:
        raise argparse.ArgumentTypeError("expected at least one baudrate")
    return baudrates


def _test_baudrate(*, port: str, baudrate: int, servo_ids: List[int]) -> bool:
    print("-" * 60)
    print(f"Baudrate: {baudrate}")

    ctrl = HiwonderBoardController(port=port, baudrate=baudrate)
    try:
        v = ctrl.get_battery_voltage()
        print(f"Battery voltage: {v if v is not None else 'N/A'}")

        pos = ctrl.read_servo_positions(servo_ids)
        if not pos:
            print("FAIL: failed to read servo positions (no/invalid response)")
            return False

        returned_ids = [int(sid) for sid, _ in pos]
        returned_set = set(returned_ids)
        requested_set = set(int(sid) for sid in servo_ids)
        missing = sorted(requested_set - returned_set)

        if missing:
            print(f"FAIL: missing {len(missing)}/{len(servo_ids)} requested IDs: {missing}")
        else:
            print("PASS: all requested servo IDs returned")

        print(f"Read {len(pos)} positions:")
        for sid, units in pos:
            print(f"  id={sid}: {units}")

        return not missing
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}")
        return False
    finally:
        ctrl.close()


def main() -> int:
    p = argparse.ArgumentParser(
        description="Test Hiwonder board comms (voltage + position read)"
    )
    p.add_argument("--port", type=str, default="/dev/ttyUSB0")
    p.add_argument("--baudrate", type=int, default=9600)
    p.add_argument(
        "--baudrates",
        type=_parse_baudrates,
        default=None,
        help="Comma-separated baudrates to test, e.g. 9600,115200. Overrides --baudrate.",
    )
    p.add_argument(
        "--test-115200",
        action="store_true",
        help=(
            "Test both --baudrate and 115200, useful for checking whether the "
            "board accepts a faster UART rate."
        ),
    )
    p.add_argument("--servo-ids", type=str, default="1,2,3")
    args = p.parse_args()

    servo_ids = _parse_ids(args.servo_ids)
    baudrates = (
        list(args.baudrates) if args.baudrates is not None else [int(args.baudrate)]
    )
    if args.test_115200 and 115200 not in baudrates:
        baudrates.append(115200)

    print(f"Port: {args.port}")
    print(f"Baudrates: {baudrates}")
    print(f"Servo IDs: {servo_ids}")

    results: List[tuple[int, bool]] = []
    for baudrate in baudrates:
        ok = _test_baudrate(port=args.port, baudrate=int(baudrate), servo_ids=servo_ids)
        results.append((int(baudrate), ok))

    print("-" * 60)
    print("Summary:")
    for baudrate, ok in results:
        print(f"  {baudrate}: {'PASS' if ok else 'FAIL'}")

    return 0 if all(ok for _, ok in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
