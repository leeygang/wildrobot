#!/usr/bin/env python3
"""Probe a Hiwonder/LewanSoul TTL bus servo through a USB debug board.

This uses the raw bus-servo protocol, not the LSC controller-board protocol:

    [0x55][0x55][servo_id][length][command][params...][checksum]

The checksum is bitwise-not of the sum from servo_id through params.
"""

from __future__ import annotations

import argparse
import time
from typing import Iterable, List, Optional, Tuple

import serial


HEADER = bytes([0x55, 0x55])
SERVO_BROADCAST_ID = 0xFE
CMD_ID_READ = 14


def _format_bytes(data: Iterable[int]) -> str:
    return " ".join(f"{int(x) & 0xFF:02X}" for x in data)


def _checksum(packet_without_checksum: Iterable[int]) -> int:
    # Vendor examples subtract the two 0x55 header bytes before inverting.
    # That is equivalent to summing from servo_id onward.
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


def _read_available_until_quiet(
    ser: serial.Serial, *, timeout_s: float, quiet_s: float = 0.02
) -> bytes:
    deadline = time.monotonic() + timeout_s
    quiet_deadline: Optional[float] = None
    chunks = bytearray()
    while time.monotonic() < deadline:
        waiting = int(getattr(ser, "in_waiting", 0))
        data = ser.read(waiting or 1)
        if data:
            chunks.extend(data)
            quiet_deadline = time.monotonic() + quiet_s
            continue
        if quiet_deadline is not None and time.monotonic() >= quiet_deadline:
            break
    return bytes(chunks)


def _read_packet(ser: serial.Serial, *, timeout_s: float) -> Tuple[Optional[bytes], bytes]:
    deadline = time.monotonic() + timeout_s
    raw = bytearray()
    while time.monotonic() < deadline:
        first = ser.read(1)
        if not first:
            continue
        raw.extend(first)
        if first != b"\x55":
            continue
        second = ser.read(1)
        raw.extend(second)
        if second != b"\x55":
            continue

        meta = ser.read(2)
        raw.extend(meta)
        if len(meta) != 2:
            continue
        servo_id = meta[0]
        length = meta[1]
        if length < 3:
            continue
        # Length counts command + params + checksum, not the ID/length bytes.
        body = ser.read(length - 1)
        raw.extend(body)
        if len(body) != length - 1:
            continue

        packet = bytes([0x55, 0x55, servo_id, length]) + body
        expected = _checksum(packet[:-1])
        if packet[-1] != expected:
            continue
        return packet, bytes(raw)
    return None, bytes(raw)


def _read_servo_id(
    ser: serial.Serial,
    *,
    target_id: int,
    timeout_s: float,
    print_raw: bool,
    dump_raw: bool,
) -> Optional[int]:
    ser.reset_input_buffer()
    ser.reset_output_buffer()

    request = _build_packet(target_id, CMD_ID_READ)
    if print_raw:
        print(f"TX: {_format_bytes(request)}")
    ser.write(request)
    ser.flush()
    if dump_raw:
        data = _read_available_until_quiet(ser, timeout_s=timeout_s)
        if data:
            print(f"RX_ANY: {_format_bytes(data)}")
        else:
            print("RX_ANY: <none>")
        for packet in _parse_packets_from_bytes(data):
            if print_raw:
                print(f"RX: {_format_bytes(packet)}")
            servo_id = _servo_id_from_packet(packet, target_id=target_id)
            if servo_id is not None:
                return servo_id
        return None

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        packet, raw = _read_packet(ser, timeout_s=max(0.0, deadline - time.monotonic()))
        if packet is None:
            return None
        if print_raw:
            print(f"RX: {_format_bytes(packet)}")

        servo_id = _servo_id_from_packet(packet, target_id=target_id)
        if servo_id is not None:
            return servo_id

    return None


def _servo_id_from_packet(packet: bytes, *, target_id: int) -> Optional[int]:
    response_id = int(packet[2])
    length = int(packet[3])
    command = int(packet[4])
    params = list(packet[5:-1])

    # Some adapters echo transmitted bytes. Ignore an ID_READ packet with no data.
    if command != CMD_ID_READ or length < 4 or not params:
        return None

    reported_id = int(params[0])
    if target_id != SERVO_BROADCAST_ID and response_id != target_id:
        return None
    return reported_id


def _parse_packets_from_bytes(data: bytes) -> List[bytes]:
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
        if packet[-1] == expected:
            packets.append(packet)
            continue
        print(
            "Ignoring packet with bad checksum: "
            f"{_format_bytes(packet)} expected={expected:02X}"
        )
    return packets


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Read one Hiwonder/LewanSoul raw TTL bus-servo ID through a USB debug board. "
            "Connect only one servo when using the default broadcast query."
        )
    )
    parser.add_argument(
        "--port",
        default="/dev/ttyUSB0",
        help="USB debug board serial port, preferably /dev/serial/by-id/...",
    )
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument(
        "--target-id",
        type=lambda x: int(x, 0),
        default=SERVO_BROADCAST_ID,
        help="ID to query. Default 0xfe broadcasts ID_READ; use only with one servo connected.",
    )
    parser.add_argument("--timeout-s", type=float, default=0.25)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--print-raw", action="store_true", help="Print raw TX/RX packets.")
    parser.add_argument(
        "--dump-raw",
        action="store_true",
        help="Print any bytes received after each request, even if they do not parse.",
    )
    args = parser.parse_args()

    print("Hiwonder TTL bus-servo ID probe")
    print(f"  port={args.port}")
    print(f"  baudrate={args.baudrate}")
    print(f"  target_id=0x{int(args.target_id) & 0xFF:02X}")
    print(f"  retries={args.retries} timeout_s={args.timeout_s}")
    if int(args.target_id) == SERVO_BROADCAST_ID:
        print("  mode=broadcast ID_READ; connect exactly one servo to avoid bus collisions")

    with serial.Serial(
        port=args.port,
        baudrate=args.baudrate,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=min(max(float(args.timeout_s), 0.01), 0.1),
    ) as ser:
        time.sleep(0.1)
        for attempt in range(1, int(args.retries) + 1):
            servo_id = _read_servo_id(
                ser,
                target_id=int(args.target_id),
                timeout_s=float(args.timeout_s),
                print_raw=bool(args.print_raw),
                dump_raw=bool(args.dump_raw),
            )
            if servo_id is not None:
                print(f"PASS: servo id={servo_id}")
                return 0
            print(f"attempt {attempt}/{args.retries}: no valid ID_READ response")
            time.sleep(0.05)

    print("FAIL: no valid ID_READ response")
    print("Check port, servo power, common ground, debug-board mode, and that only one servo is connected.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
