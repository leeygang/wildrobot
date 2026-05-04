#!/usr/bin/env python3

from __future__ import annotations

import argparse
from typing import List

from wr_runtime.hardware.hiwonder_board_controller import HiwonderBoardController


def _parse_ids(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description="Test Hiwonder board comms (voltage + position read)")
    p.add_argument("--port", type=str, default="/dev/ttyUSB0")
    p.add_argument("--baudrate", type=int, default=9600)
    p.add_argument("--servo-ids", type=str, default="1,2,3")
    args = p.parse_args()

    servo_ids = _parse_ids(args.servo_ids)
    print(f"Port: {args.port}")
    print(f"Baudrate: {args.baudrate}")
    print(f"Servo IDs: {servo_ids}")

    ctrl = HiwonderBoardController(port=args.port, baudrate=args.baudrate)
    try:
        v = ctrl.get_battery_voltage()
        print(f"Battery voltage: {v if v is not None else 'N/A'}")

        pos = ctrl.read_servo_positions(servo_ids)
        if not pos:
            raise SystemExit("Failed to read servo positions (no/invalid response)")

        returned_ids = [int(sid) for sid, _ in pos]
        returned_set = set(returned_ids)
        requested_set = set(int(sid) for sid in servo_ids)
        missing = sorted(requested_set - returned_set)

        if missing:
            print(f"Warning: missing {len(missing)}/{len(servo_ids)} requested IDs: {missing}")

        print(f"Read {len(pos)} positions:")
        for sid, units in pos:
            print(f"  id={sid}: {units}")

        if missing:
            raise SystemExit("Partial response: not all requested servo IDs returned")
    finally:
        ctrl.close()


if __name__ == "__main__":
    main()
