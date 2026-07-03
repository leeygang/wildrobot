#!/usr/bin/env python3
"""Timing probe for Hiwonder/LewanSoul raw TTL bus servos.

This script talks directly to a TTL bus-servo USB debug board. It does not use
the LSC controller-board protocol.
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Iterable, List, Optional

import serial


HEADER = bytes([0x55, 0x55])
CMD_MOVE_TIME_WRITE = 1
CMD_POS_READ = 28


def _format_bytes(data: Iterable[int]) -> str:
    return " ".join(f"{int(x) & 0xFF:02X}" for x in data)


def _parse_ids(text: str) -> List[int]:
    ids = [int(x.strip(), 0) for x in text.split(",") if x.strip()]
    if not ids:
        raise argparse.ArgumentTypeError("expected at least one servo id")
    for sid in ids:
        if sid < 0 or sid > 253:
            raise argparse.ArgumentTypeError(f"servo id out of range: {sid}")
    return ids


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


def _read_available_until_quiet(
    ser: serial.Serial, *, timeout_s: float, quiet_s: float = 0.003
) -> bytes:
    deadline = time.perf_counter() + timeout_s
    quiet_deadline: Optional[float] = None
    chunks = bytearray()
    while time.perf_counter() < deadline:
        waiting = int(getattr(ser, "in_waiting", 0))
        data = ser.read(waiting or 1)
        if data:
            chunks.extend(data)
            quiet_deadline = time.perf_counter() + quiet_s
            continue
        if quiet_deadline is not None and time.perf_counter() >= quiet_deadline:
            break
    return bytes(chunks)


def _write_packet(ser: serial.Serial, packet: bytes, *, print_raw: bool) -> None:
    if print_raw:
        print(f"TX: {_format_bytes(packet)}")
    ser.write(packet)
    ser.flush()


def _read_position(
    ser: serial.Serial, servo_id: int, *, timeout_s: float, print_raw: bool
) -> Optional[int]:
    ser.reset_input_buffer()
    packet = _build_packet(servo_id, CMD_POS_READ)
    _write_packet(ser, packet, print_raw=print_raw)
    data = _read_available_until_quiet(ser, timeout_s=timeout_s)
    if print_raw:
        print(f"RX_ANY: {_format_bytes(data) if data else '<none>'}")

    for response in _parse_packets(data, print_bad=print_raw):
        if print_raw:
            print(f"RX: {_format_bytes(response)}")
        if int(response[2]) != int(servo_id):
            continue
        if int(response[4]) != CMD_POS_READ:
            continue
        params = list(response[5:-1])
        if len(params) < 2:
            continue
        return int(params[0]) | (int(params[1]) << 8)
    return None


def _write_position(
    ser: serial.Serial,
    servo_id: int,
    position: int,
    *,
    move_time_ms: int,
    print_raw: bool,
) -> None:
    position = int(position)
    move_time_ms = int(move_time_ms)
    params = [
        position & 0xFF,
        (position >> 8) & 0xFF,
        move_time_ms & 0xFF,
        (move_time_ms >> 8) & 0xFF,
    ]
    _write_packet(
        ser,
        _build_packet(servo_id, CMD_MOVE_TIME_WRITE, params),
        print_raw=print_raw,
    )


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return ordered[idx]


def _format_ms(value_s: float) -> str:
    if value_s != value_s:
        return "nan"
    return f"{value_s * 1000.0:.3f}"


def _print_timing_summary(label: str, samples_s: List[float]) -> None:
    if not samples_s:
        print(f"  {label}: n=0")
        return
    print(
        f"  {label}: n={len(samples_s)} "
        f"avg={_format_ms(statistics.fmean(samples_s))}ms "
        f"p50={_format_ms(_percentile(samples_s, 0.50))}ms "
        f"p95={_format_ms(_percentile(samples_s, 0.95))}ms "
        f"max={_format_ms(max(samples_s))}ms"
    )


def _estimate_wire_ms(*, servo_count: int, mode: str, baudrate: int) -> float:
    write_bytes = 10 * servo_count if mode in ("write", "read-write") else 0
    read_bytes = (6 + 8) * servo_count if mode in ("read", "read-write") else 0
    return (write_bytes + read_bytes) * 10.0 * 1000.0 / float(baudrate)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark raw Hiwonder TTL bus-servo read/write timing."
    )
    parser.add_argument(
        "--port",
        default="/dev/ttyUSB0",
        help="USB debug board serial port, preferably /dev/serial/by-id/...",
    )
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--servo-ids", type=_parse_ids, required=True)
    parser.add_argument(
        "--mode",
        choices=("read", "write", "read-write"),
        default="read",
        help="Default read mode is safe and does not move servos.",
    )
    parser.add_argument("--cycles", type=int, default=200)
    parser.add_argument("--timeout-s", type=float, default=0.05)
    parser.add_argument(
        "--period-s",
        type=float,
        default=0.0,
        help="Optional loop period. Use 0.02 to test a 50 Hz deadline.",
    )
    parser.add_argument("--move-time-ms", type=int, default=20)
    parser.add_argument(
        "--move-amplitude-units",
        type=int,
        default=0,
        help="Optional +/- units around initial position for write modes. Default 0 repeats current position.",
    )
    parser.add_argument("--min-pos", type=int, default=0)
    parser.add_argument("--max-pos", type=int, default=1000)
    parser.add_argument("--print-raw", action="store_true")
    args = parser.parse_args()

    servo_ids = list(args.servo_ids)
    print("Hiwonder TTL bus timing probe")
    print(f"  port={args.port}")
    print(f"  baudrate={args.baudrate}")
    print(f"  servo_ids={servo_ids}")
    print(f"  mode={args.mode} cycles={args.cycles} period_s={args.period_s}")
    print(
        "  estimated_serial_wire_time_ms="
        f"{_estimate_wire_ms(servo_count=len(servo_ids), mode=args.mode, baudrate=args.baudrate):.3f}"
    )

    with serial.Serial(
        port=args.port,
        baudrate=args.baudrate,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=min(max(float(args.timeout_s), 0.001), 0.05),
    ) as ser:
        time.sleep(0.1)

        initial_positions: dict[int, int] = {}
        for sid in servo_ids:
            pos = _read_position(
                ser, int(sid), timeout_s=float(args.timeout_s), print_raw=bool(args.print_raw)
            )
            if pos is None:
                print(f"FAIL: could not read initial position for servo id={sid}")
                return 1
            initial_positions[int(sid)] = int(pos)
        print(f"  initial_positions={initial_positions}")

        total_samples: List[float] = []
        read_samples: List[float] = []
        write_samples: List[float] = []
        read_ok = 0
        read_fail = 0
        write_count = 0
        deadline_misses = 0
        t_start = time.perf_counter()

        for cycle in range(int(args.cycles)):
            loop_start = time.perf_counter()

            if args.mode in ("write", "read-write"):
                write_start = time.perf_counter()
                direction = 1 if cycle % 2 == 0 else -1
                for sid in servo_ids:
                    base = initial_positions[int(sid)]
                    target = base + direction * int(args.move_amplitude_units)
                    target = min(int(args.max_pos), max(int(args.min_pos), int(target)))
                    _write_position(
                        ser,
                        int(sid),
                        target,
                        move_time_ms=int(args.move_time_ms),
                        print_raw=bool(args.print_raw),
                    )
                    write_count += 1
                write_samples.append(time.perf_counter() - write_start)

            if args.mode in ("read", "read-write"):
                read_start = time.perf_counter()
                for sid in servo_ids:
                    pos = _read_position(
                        ser,
                        int(sid),
                        timeout_s=float(args.timeout_s),
                        print_raw=bool(args.print_raw),
                    )
                    if pos is None:
                        read_fail += 1
                    else:
                        read_ok += 1
                read_samples.append(time.perf_counter() - read_start)

            loop_elapsed = time.perf_counter() - loop_start
            total_samples.append(loop_elapsed)
            if float(args.period_s) > 0.0:
                remaining = float(args.period_s) - loop_elapsed
                if remaining > 0.0:
                    time.sleep(remaining)
                else:
                    deadline_misses += 1

        elapsed = time.perf_counter() - t_start

    print("Summary:")
    print(f"  elapsed_s={elapsed:.3f}")
    print(f"  cycles={len(total_samples)} effective_hz={len(total_samples) / elapsed:.3f}")
    if float(args.period_s) > 0.0:
        print(f"  deadline_misses={deadline_misses}/{len(total_samples)} period_s={args.period_s}")
    print(f"  write_commands={write_count}")
    print(f"  read_ok={read_ok} read_fail={read_fail}")
    _print_timing_summary("total_cycle", total_samples)
    _print_timing_summary("write_cycle", write_samples)
    _print_timing_summary("read_cycle", read_samples)
    return 0 if read_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
