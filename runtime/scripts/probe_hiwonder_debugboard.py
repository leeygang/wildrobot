#!/usr/bin/env python3
"""Probe Hiwonder Debugboard magnetic-encoder servo protocol.

This uses the Debugboard SDK protocol shape:

    [0xff][0xff][id][length][instruction][params...][checksum]

It is intentionally read-only. It pings servo IDs and, on success, reads a few
safe registers so we can confirm whether the servo/debug board works at 1 Mbps.
"""

from __future__ import annotations

import argparse
import time
from typing import Iterable, List, Optional, Tuple

import serial


HEADER = bytes([0xFF, 0xFF])

INST_PING = 0x01
INST_READ = 0x02

REG_ID = 5
REG_BAUD_RATE = 6
REG_PRESENT_POSITION_L = 56
REG_PRESENT_VOLTAGE = 62
REG_PRESENT_TEMPERATURE = 63


def _format_bytes(data: Iterable[int]) -> str:
    return " ".join(f"{int(x) & 0xFF:02X}" for x in data)


def _parse_int_list(text: str) -> List[int]:
    values = [int(x.strip(), 0) for x in text.split(",") if x.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one value")
    return values


def _checksum(packet_without_checksum: Iterable[int]) -> int:
    data = list(packet_without_checksum)
    return (~sum(data[2:])) & 0xFF


def _build_packet(servo_id: int, instruction: int, params: Optional[List[int]] = None) -> bytes:
    params = list(params or [])
    length = len(params) + 2  # instruction + params + checksum
    packet = bytearray(HEADER)
    packet.append(int(servo_id) & 0xFF)
    packet.append(length & 0xFF)
    packet.append(int(instruction) & 0xFF)
    packet.extend(int(x) & 0xFF for x in params)
    packet.append(_checksum(packet))
    return bytes(packet)


def _parse_status_packets(data: bytes, *, print_bad: bool = False) -> List[bytes]:
    packets: List[bytes] = []
    for i in range(0, max(0, len(data) - 1)):
        if data[i : i + 2] != HEADER:
            continue
        if i + 4 > len(data):
            continue
        length = int(data[i + 3])
        if length < 2:
            continue
        end = i + 4 + length
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


def _read_status(
    ser: serial.Serial,
    *,
    servo_id: int,
    timeout_s: float,
    print_raw: bool,
) -> Tuple[Optional[bytes], bytes]:
    deadline = time.perf_counter() + timeout_s
    raw = bytearray()
    while time.perf_counter() < deadline:
        waiting = int(getattr(ser, "in_waiting", 0))
        data = ser.read(waiting or 1)
        if not data:
            continue
        raw.extend(data)
        for packet in _parse_status_packets(bytes(raw), print_bad=False):
            if int(packet[2]) == int(servo_id):
                if print_raw:
                    print(f"RX_ANY: {_format_bytes(bytes(raw))}")
                    print(f"RX: {_format_bytes(packet)}")
                return packet, bytes(raw)
    if print_raw:
        print(f"RX_ANY: {_format_bytes(bytes(raw)) if raw else '<none>'}")
        _parse_status_packets(bytes(raw), print_bad=True)
    return None, bytes(raw)


def _tx_rx(
    ser: serial.Serial,
    *,
    servo_id: int,
    instruction: int,
    params: Optional[List[int]],
    timeout_s: float,
    print_raw: bool,
) -> Optional[bytes]:
    ser.reset_input_buffer()
    packet = _build_packet(servo_id, instruction, params)
    if print_raw:
        print(f"TX: {_format_bytes(packet)}")
    ser.write(packet)
    ser.flush()
    response, _ = _read_status(
        ser,
        servo_id=servo_id,
        timeout_s=timeout_s,
        print_raw=print_raw,
    )
    return response


def _status_error(packet: bytes) -> int:
    return int(packet[4])


def _status_params(packet: bytes) -> List[int]:
    return list(packet[5:-1])


def _ping(
    ser: serial.Serial,
    *,
    servo_id: int,
    timeout_s: float,
    print_raw: bool,
) -> Tuple[bool, Optional[int], Optional[bytes]]:
    response = _tx_rx(
        ser,
        servo_id=servo_id,
        instruction=INST_PING,
        params=None,
        timeout_s=timeout_s,
        print_raw=print_raw,
    )
    if response is None:
        return False, None, None
    return _status_error(response) == 0, _status_error(response), response


def _read_register(
    ser: serial.Serial,
    *,
    servo_id: int,
    address: int,
    length: int,
    timeout_s: float,
    print_raw: bool,
) -> Optional[List[int]]:
    response = _tx_rx(
        ser,
        servo_id=servo_id,
        instruction=INST_READ,
        params=[address, length],
        timeout_s=timeout_s,
        print_raw=print_raw,
    )
    if response is None or _status_error(response) != 0:
        return None
    params = _status_params(response)
    if len(params) < length:
        return None
    return params[:length]


def _u16_le(data: List[int]) -> int:
    return int(data[0]) | (int(data[1]) << 8)


def _signed_mag_to_int(value: int, *, sign_bit: int) -> int:
    if value & (1 << sign_bit):
        return -(value & ~(1 << sign_bit))
    return value


def _probe_one(
    *,
    port: str,
    baudrate: int,
    servo_ids: List[int],
    timeout_s: float,
    print_raw: bool,
    read_registers: bool,
) -> bool:
    print("-" * 60)
    print(f"Baudrate: {baudrate}")
    any_success = False
    with serial.Serial(
        port=port,
        baudrate=baudrate,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=0.001,
    ) as ser:
        time.sleep(0.1)
        for servo_id in servo_ids:
            ok, error, packet = _ping(
                ser,
                servo_id=servo_id,
                timeout_s=timeout_s,
                print_raw=print_raw,
            )
            if not ok:
                error_text = "no response" if packet is None else f"status_error={error}"
                print(f"  id={servo_id}: PING FAIL ({error_text})")
                continue

            any_success = True
            print(f"  id={servo_id}: PING PASS")
            if not read_registers:
                continue

            id_data = _read_register(
                ser,
                servo_id=servo_id,
                address=REG_ID,
                length=1,
                timeout_s=timeout_s,
                print_raw=print_raw,
            )
            baud_data = _read_register(
                ser,
                servo_id=servo_id,
                address=REG_BAUD_RATE,
                length=1,
                timeout_s=timeout_s,
                print_raw=print_raw,
            )
            pos_data = _read_register(
                ser,
                servo_id=servo_id,
                address=REG_PRESENT_POSITION_L,
                length=2,
                timeout_s=timeout_s,
                print_raw=print_raw,
            )
            voltage_data = _read_register(
                ser,
                servo_id=servo_id,
                address=REG_PRESENT_VOLTAGE,
                length=1,
                timeout_s=timeout_s,
                print_raw=print_raw,
            )
            temp_data = _read_register(
                ser,
                servo_id=servo_id,
                address=REG_PRESENT_TEMPERATURE,
                length=1,
                timeout_s=timeout_s,
                print_raw=print_raw,
            )

            print(f"    register_id={id_data[0] if id_data else 'N/A'}")
            print(f"    baud_register={baud_data[0] if baud_data else 'N/A'}")
            if pos_data:
                pos = _signed_mag_to_int(_u16_le(pos_data), sign_bit=15)
                print(f"    present_position={pos}")
            else:
                print("    present_position=N/A")
            print(f"    voltage_raw={voltage_data[0] if voltage_data else 'N/A'}")
            print(f"    temperature_c={temp_data[0] if temp_data else 'N/A'}")
    return any_success


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Probe Hiwonder Debugboard FF-FF servo protocol at one or more baudrates."
    )
    parser.add_argument(
        "--port",
        default="/dev/ttyUSB0",
        help="Debugboard serial port, preferably /dev/serial/by-id/...",
    )
    parser.add_argument(
        "--baudrates",
        type=_parse_int_list,
        default=[1000000, 115200],
        help="Comma-separated baudrates to test. Default: 1000000,115200.",
    )
    parser.add_argument(
        "--servo-ids",
        type=_parse_int_list,
        default=[3, 1],
        help="Comma-separated servo IDs to ping. Default: 3,1.",
    )
    parser.add_argument("--timeout-s", type=float, default=0.05)
    parser.add_argument("--print-raw", action="store_true")
    parser.add_argument(
        "--no-read-registers",
        action="store_true",
        help="Only ping; skip safe register reads after a successful ping.",
    )
    args = parser.parse_args()

    print("Hiwonder Debugboard protocol probe")
    print(f"  port={args.port}")
    print(f"  baudrates={args.baudrates}")
    print(f"  servo_ids={args.servo_ids}")
    print("  protocol=FF FF ID LEN INST PARAMS CHECKSUM")
    print("  mode=read-only; no EEPROM writes and no movement")

    success_by_baud: List[int] = []
    for baudrate in args.baudrates:
        try:
            ok = _probe_one(
                port=args.port,
                baudrate=int(baudrate),
                servo_ids=list(args.servo_ids),
                timeout_s=float(args.timeout_s),
                print_raw=bool(args.print_raw),
                read_registers=not bool(args.no_read_registers),
            )
        except serial.SerialException as exc:
            print("-" * 60)
            print(f"Baudrate: {baudrate}")
            print(f"  ERROR: could not open/use serial port: {exc}")
            ok = False
        if ok:
            success_by_baud.append(int(baudrate))

    print("-" * 60)
    if success_by_baud:
        print(f"PASS: Debugboard protocol responded at baudrates={success_by_baud}")
        return 0
    print("FAIL: no Debugboard protocol response at tested baudrates")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
