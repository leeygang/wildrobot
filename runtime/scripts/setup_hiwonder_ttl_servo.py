#!/usr/bin/env python3
"""Interactive setup tool for one Hiwonder/LewanSoul TTL bus servo.

This talks to a raw TTL bus-servo USB debug board, not the LSC controller-board
protocol. Connect exactly one servo before using broadcast ID discovery.
"""

from __future__ import annotations

import argparse
import time
from typing import Iterable, List, Optional

import serial


HEADER = bytes([0x55, 0x55])
SERVO_BROADCAST_ID = 0xFE

CMD_MOVE_TIME_WRITE = 1
CMD_ID_WRITE = 13
CMD_ID_READ = 14
CMD_POS_READ = 28


def _format_bytes(data: Iterable[int]) -> str:
    return " ".join(f"{int(x) & 0xFF:02X}" for x in data)


def _checksum(packet_without_checksum: Iterable[int]) -> int:
    data = list(packet_without_checksum)
    return (~sum(data[2:])) & 0xFF


def _build_packet(servo_id: int, command: int, params: Optional[List[int]] = None) -> bytes:
    params = list(params or [])
    length = 3 + len(params)  # command + params + checksum
    packet = bytearray(HEADER)
    packet.append(int(servo_id) & 0xFF)
    packet.append(length & 0xFF)
    packet.append(int(command) & 0xFF)
    packet.extend(int(x) & 0xFF for x in params)
    packet.append(_checksum(packet))
    return bytes(packet)


def _parse_packets(data: bytes, *, print_bad: bool = False) -> List[bytes]:
    packets: List[bytes] = []
    for i in range(0, max(0, len(data) - 1)):
        if data[i : i + 2] != HEADER:
            continue
        if i + 4 > len(data):
            continue
        length = int(data[i + 3])
        if length < 3:
            continue
        end = i + 4 + (length - 1)
        if end > len(data):
            continue
        packet = data[i:end]
        expected = _checksum(packet[:-1])
        if int(packet[-1]) == expected:
            packets.append(packet)
        elif print_bad:
            print(
                "Ignoring packet with bad checksum: "
                f"{_format_bytes(packet)} expected={expected:02X}"
            )
    return packets


def _write_packet(ser: serial.Serial, packet: bytes, *, print_raw: bool) -> None:
    if print_raw:
        print(f"TX: {_format_bytes(packet)}")
    ser.write(packet)
    ser.flush()


def _read_response(
    ser: serial.Serial,
    *,
    command: int,
    timeout_s: float,
    servo_id: Optional[int] = None,
    min_params: int = 0,
    print_raw: bool = False,
) -> Optional[bytes]:
    deadline = time.perf_counter() + timeout_s
    raw = bytearray()
    while time.perf_counter() < deadline:
        waiting = int(getattr(ser, "in_waiting", 0))
        data = ser.read(waiting or 1)
        if not data:
            continue
        raw.extend(data)
        for packet in _parse_packets(bytes(raw), print_bad=False):
            if servo_id is not None and int(packet[2]) != int(servo_id):
                continue
            if int(packet[4]) != int(command):
                continue
            if len(packet[5:-1]) < min_params:
                continue
            if print_raw:
                print(f"RX_ANY: {_format_bytes(bytes(raw))}")
                print(f"RX: {_format_bytes(packet)}")
            return packet

    if print_raw:
        print(f"RX_ANY: {_format_bytes(bytes(raw)) if raw else '<none>'}")
        _parse_packets(bytes(raw), print_bad=True)
    return None


def _read_servo_id(ser: serial.Serial, *, timeout_s: float, print_raw: bool) -> Optional[int]:
    ser.reset_input_buffer()
    packet = _build_packet(SERVO_BROADCAST_ID, CMD_ID_READ)
    _write_packet(ser, packet, print_raw=print_raw)
    response = _read_response(
        ser,
        command=CMD_ID_READ,
        timeout_s=timeout_s,
        min_params=1,
        print_raw=print_raw,
    )
    if response is None:
        return None
    return int(response[5])


def _write_servo_id(
    ser: serial.Serial,
    *,
    old_id: int,
    new_id: int,
    print_raw: bool,
) -> None:
    ser.reset_input_buffer()
    packet = _build_packet(old_id, CMD_ID_WRITE, [new_id])
    _write_packet(ser, packet, print_raw=print_raw)


def _read_position(
    ser: serial.Serial,
    *,
    servo_id: int,
    timeout_s: float,
    print_raw: bool,
) -> Optional[int]:
    ser.reset_input_buffer()
    packet = _build_packet(servo_id, CMD_POS_READ)
    _write_packet(ser, packet, print_raw=print_raw)
    response = _read_response(
        ser,
        command=CMD_POS_READ,
        timeout_s=timeout_s,
        servo_id=servo_id,
        min_params=2,
        print_raw=print_raw,
    )
    if response is None:
        return None
    params = list(response[5:-1])
    return int(params[0]) | (int(params[1]) << 8)


def _move_to_position(
    ser: serial.Serial,
    *,
    servo_id: int,
    position: int,
    move_time_ms: int,
    print_raw: bool,
) -> None:
    params = [
        position & 0xFF,
        (position >> 8) & 0xFF,
        move_time_ms & 0xFF,
        (move_time_ms >> 8) & 0xFF,
    ]
    packet = _build_packet(servo_id, CMD_MOVE_TIME_WRITE, params)
    _write_packet(ser, packet, print_raw=print_raw)


def _prompt_int(prompt: str, *, minimum: int, maximum: int) -> Optional[int]:
    while True:
        text = input(prompt).strip()
        if text.lower() in {"q", "quit"}:
            return None
        try:
            value = int(text, 0)
        except ValueError:
            print(f"Enter an integer between {minimum} and {maximum}, or q to cancel.")
            continue
        if minimum <= value <= maximum:
            return value
        print(f"Value out of range: expected {minimum}..{maximum}.")


def _confirm(prompt: str) -> bool:
    return input(f"{prompt} Type y then Enter to continue: ").strip().lower() == "y"


def _discover_or_report(
    ser: serial.Serial,
    *,
    timeout_s: float,
    print_raw: bool,
) -> Optional[int]:
    servo_id = _read_servo_id(ser, timeout_s=timeout_s, print_raw=print_raw)
    if servo_id is None:
        print(
            "ERROR: no servo ID response. "
            "Check power, port, and that exactly one servo is connected."
        )
        return None
    print(f"Current servo id: {servo_id}")
    return servo_id


def _print_menu() -> None:
    print()
    print("Options:")
    print("  1) Read servo id")
    print("  2) Update servo id")
    print("  3) Read servo position")
    print("  4) Move servo to position")
    print("  5) Exit")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Interactive one-servo setup utility for a Hiwonder/LewanSoul TTL "
            "bus-servo USB debug board."
        )
    )
    parser.add_argument(
        "--port",
        default="/dev/ttyUSB0",
        help="USB debug board serial port, preferably /dev/serial/by-id/...",
    )
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--timeout-s", type=float, default=0.05)
    parser.add_argument(
        "--move-time-ms",
        type=int,
        default=500,
        help="Servo move duration for option 4.",
    )
    parser.add_argument("--print-raw", action="store_true", help="Print raw TX/RX packets.")
    args = parser.parse_args()

    if int(args.move_time_ms) < 0 or int(args.move_time_ms) > 30000:
        parser.error("--move-time-ms must be between 0 and 30000")

    print("Hiwonder TTL one-servo setup")
    print(f"  port={args.port}")
    print(f"  baudrate={args.baudrate}")
    print(f"  timeout_s={args.timeout_s}")
    print(f"  move_time_ms={args.move_time_ms}")
    print("  requirement: connect exactly one servo to the TTL debug board")

    with serial.Serial(
        port=args.port,
        baudrate=args.baudrate,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=0.001,
    ) as ser:
        time.sleep(0.1)
        cached_id: Optional[int] = None

        while True:
            _print_menu()
            choice = input("Enter choice: ").strip().lower()
            if choice in {"5", "exit", "q", "quit"}:
                return 0

            if choice == "1":
                cached_id = _discover_or_report(
                    ser, timeout_s=float(args.timeout_s), print_raw=bool(args.print_raw)
                )
                continue

            if choice == "2":
                old_id = _discover_or_report(
                    ser, timeout_s=float(args.timeout_s), print_raw=bool(args.print_raw)
                )
                if old_id is None:
                    continue
                new_id = _prompt_int(
                    "New servo id (1-253, q to cancel): ", minimum=1, maximum=253
                )
                if new_id is None:
                    print("Canceled.")
                    continue
                if new_id == old_id:
                    print("Servo already has that id.")
                    cached_id = old_id
                    continue
                if not _confirm(f"Write servo id {old_id} -> {new_id}?"):
                    print("Canceled.")
                    continue
                _write_servo_id(
                    ser,
                    old_id=old_id,
                    new_id=new_id,
                    print_raw=bool(args.print_raw),
                )
                time.sleep(0.1)
                verified_id = _read_servo_id(
                    ser,
                    timeout_s=float(args.timeout_s),
                    print_raw=bool(args.print_raw),
                )
                if verified_id == new_id:
                    print(f"PASS: servo id updated to {new_id}")
                    cached_id = new_id
                else:
                    print(f"WARNING: ID write sent, but verification read returned {verified_id}")
                    cached_id = verified_id
                continue

            if choice == "3":
                if cached_id is None:
                    cached_id = _discover_or_report(
                        ser, timeout_s=float(args.timeout_s), print_raw=bool(args.print_raw)
                    )
                if cached_id is None:
                    continue
                position = _read_position(
                    ser,
                    servo_id=cached_id,
                    timeout_s=float(args.timeout_s),
                    print_raw=bool(args.print_raw),
                )
                if position is None:
                    print(f"ERROR: no position response from servo id={cached_id}")
                else:
                    print(f"Servo id={cached_id} position={position}")
                continue

            if choice == "4":
                if cached_id is None:
                    cached_id = _discover_or_report(
                        ser, timeout_s=float(args.timeout_s), print_raw=bool(args.print_raw)
                    )
                if cached_id is None:
                    continue
                position = _prompt_int(
                    "Target position (0-1000, q to cancel): ", minimum=0, maximum=1000
                )
                if position is None:
                    print("Canceled.")
                    continue
                if not _confirm(
                    f"Move servo id={cached_id} to position={position} over {args.move_time_ms}ms?"
                ):
                    print("Canceled.")
                    continue
                _move_to_position(
                    ser,
                    servo_id=cached_id,
                    position=position,
                    move_time_ms=int(args.move_time_ms),
                    print_raw=bool(args.print_raw),
                )
                print(f"Move command sent: id={cached_id} position={position}")
                continue

            print(f"Unknown choice: {choice!r}")


if __name__ == "__main__":
    raise SystemExit(main())
