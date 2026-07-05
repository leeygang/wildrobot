#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_RUNTIME_ROOT = _REPO_ROOT / "runtime"
if str(_RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(_RUNTIME_ROOT))


def _resolve_board_pin(board_module, pin_name: str):
    name = str(pin_name).strip()
    if name.lower().startswith("board."):
        name = name.split(".", 1)[1]
    upper_name = name.upper()
    if upper_name.startswith("GPIO") and upper_name[4:].isdigit():
        name = f"D{upper_name[4:]}"
    try:
        return getattr(board_module, name)
    except AttributeError as exc:
        raise ValueError(f"Unknown Blinka board pin: {pin_name!r}") from exc


def _parse_csv_ints(value: str) -> list[int]:
    out: list[int] = []
    for part in str(value).split(","):
        text = part.strip()
        if text:
            out.append(int(text, 0))
    if not out:
        raise argparse.ArgumentTypeError("must provide at least one integer")
    return out


def _fmt_bytes(data: bytes | bytearray) -> str:
    return " ".join(f"{int(b):02X}" for b in data)


def _header_summary(data: bytes | bytearray, *, offset: int) -> str:
    if len(data) < offset + 4:
        return f"offset={offset}: n/a"
    b0, b1, channel, seq = [int(x) for x in data[offset : offset + 4]]
    length = ((b1 << 8) | b0) & 0x7FFF
    continuation = bool(b1 & 0x80)
    sane = 4 <= length <= 512 and 0 <= channel <= 5
    return (
        f"offset={offset}: length={length} channel={channel} seq={seq} "
        f"continued={int(continuation)} sane={int(sane)}"
    )


def _sane_headers(data: bytes | bytearray) -> list[str]:
    headers: list[str] = []
    for offset in range(max(0, len(data) - 3)):
        b0, b1, channel, seq = [int(x) for x in data[offset : offset + 4]]
        length = ((b1 << 8) | b0) & 0x7FFF
        continuation = bool(b1 & 0x80)
        if 4 <= length <= 512 and 0 <= channel <= 5:
            headers.append(
                f"offset={offset} length={length} channel={channel} "
                f"seq={seq} continued={int(continuation)}"
            )
    return headers


def _wait_for_int_low(int_pin, *, timeout_s: float) -> float | None:
    start = time.monotonic()
    while time.monotonic() - start < timeout_s:
        if not bool(int_pin.value):
            return time.monotonic() - start
        time.sleep(0.001)
    return None


def _pulse_reset(reset_pin, *, reset_low_s: float, post_reset_s: float) -> None:
    reset_pin.value = True
    time.sleep(0.01)
    reset_pin.value = False
    time.sleep(reset_low_s)
    reset_pin.value = True
    time.sleep(post_reset_s)


def _read_spi_raw(
    bus,
    cs_pin,
    *,
    baudrate: int,
    polarity: int,
    phase: int,
    read_len: int,
    write_value: int,
    cs_setup_s: float,
) -> bytes:
    buf = bytearray(read_len)
    while not bus.try_lock():
        time.sleep(0.001)
    try:
        bus.configure(baudrate=baudrate, polarity=polarity, phase=phase)
        cs_pin.value = False
        if cs_setup_s > 0.0:
            time.sleep(cs_setup_s)
        bus.readinto(buf, write_value=write_value & 0xFF)
        cs_pin.value = True
    finally:
        try:
            cs_pin.value = True
        finally:
            bus.unlock()
    return bytes(buf)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Raw BNO08x SPI probe. This does not use the Adafruit BNO08x driver; "
            "it only checks reset/INT behavior and dumps raw SPI bytes."
        )
    )
    parser.add_argument("--spi-cs-pin", default="D8")
    parser.add_argument("--spi-int-pin", default="D17")
    parser.add_argument("--spi-reset-pin", default="D27")
    parser.add_argument("--baudrates", type=_parse_csv_ints, default=[100000, 400000, 1000000])
    parser.add_argument("--modes", type=_parse_csv_ints, default=[3])
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--read-len", type=int, default=12)
    parser.add_argument("--write-value", type=lambda v: int(v, 0), default=0x00)
    parser.add_argument("--reset-low-s", type=float, default=0.01)
    parser.add_argument("--post-reset-s", type=float, default=0.0)
    parser.add_argument("--int-timeout-s", type=float, default=1.0)
    parser.add_argument("--int-settle-s", type=float, default=0.0)
    parser.add_argument("--cs-setup-s", type=float, default=0.0)
    args = parser.parse_args()

    import board
    import busio
    import digitalio
    from digitalio import Direction, Pull

    bus = busio.SPI(board.SCK, MOSI=board.MOSI, MISO=board.MISO)
    cs_pin = digitalio.DigitalInOut(_resolve_board_pin(board, args.spi_cs_pin))
    int_pin = digitalio.DigitalInOut(_resolve_board_pin(board, args.spi_int_pin))
    reset_pin = digitalio.DigitalInOut(_resolve_board_pin(board, args.spi_reset_pin))

    cs_pin.direction = Direction.OUTPUT
    cs_pin.value = True
    int_pin.direction = Direction.INPUT
    int_pin.pull = Pull.UP
    reset_pin.direction = Direction.OUTPUT
    reset_pin.value = True

    print("BNO08x raw SPI probe")
    print(
        f"  cs={args.spi_cs_pin} int={args.spi_int_pin} reset={args.spi_reset_pin} "
        f"read_len={args.read_len} write_value=0x{args.write_value & 0xFF:02X}"
    )
    print(
        "  Expect after reset: INT should go low, and a sane SHTP header usually "
        "has length 4..512 and channel 0..5."
    )

    try:
        for cycle in range(max(1, int(args.cycles))):
            for baudrate in args.baudrates:
                for mode in args.modes:
                    polarity = 1 if mode in (2, 3) else 0
                    phase = 1 if mode in (1, 3) else 0
                    print("-" * 60)
                    print(
                        f"cycle={cycle + 1} baudrate={baudrate} mode={mode} "
                        f"polarity={polarity} phase={phase}"
                    )
                    print(f"INT before reset: {'LOW' if not int_pin.value else 'HIGH'}")
                    _pulse_reset(
                        reset_pin,
                        reset_low_s=max(0.0, float(args.reset_low_s)),
                        post_reset_s=max(0.0, float(args.post_reset_s)),
                    )
                    print(f"INT after reset pulse: {'LOW' if not int_pin.value else 'HIGH'}")
                    wait_s = _wait_for_int_low(int_pin, timeout_s=max(0.0, float(args.int_timeout_s)))
                    if wait_s is None:
                        print(f"INT did not go LOW within {args.int_timeout_s:.3f}s; skipping SPI read")
                        continue
                    print(f"INT low after {wait_s * 1000.0:.1f} ms")
                    if args.int_settle_s > 0.0:
                        time.sleep(args.int_settle_s)
                    raw = _read_spi_raw(
                        bus,
                        cs_pin,
                        baudrate=baudrate,
                        polarity=polarity,
                        phase=phase,
                        read_len=max(4, int(args.read_len)),
                        write_value=int(args.write_value),
                        cs_setup_s=max(0.0, float(args.cs_setup_s)),
                    )
                    print(f"RX: {_fmt_bytes(raw)}")
                    print(_header_summary(raw, offset=0))
                    print(_header_summary(raw, offset=1))
                    sane_headers = _sane_headers(raw)
                    if sane_headers:
                        print("sane_headers:")
                        for header in sane_headers:
                            print(f"  {header}")
                    else:
                        print("sane_headers: none")
    finally:
        cs_pin.deinit()
        int_pin.deinit()
        reset_pin.deinit()
        try:
            bus.deinit()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
