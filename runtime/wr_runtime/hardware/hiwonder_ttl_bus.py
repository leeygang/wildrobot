from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, Optional, Protocol, Sequence

import serial


HEADER = bytes([0x55, 0x55])
SERVO_BROADCAST_ID = 0xFE

CMD_MOVE_TIME_WRITE = 1
CMD_MOVE_TIME_READ = 2
CMD_MOVE_TIME_WAIT_WRITE = 7
CMD_MOVE_TIME_WAIT_READ = 8
CMD_MOVE_START = 11
CMD_MOVE_STOP = 12
CMD_ID_WRITE = 13
CMD_ID_READ = 14
CMD_POS_READ = 28
CMD_LOAD_OR_UNLOAD_WRITE = 31


def format_bytes(data: Iterable[int]) -> str:
    return " ".join(f"{int(x) & 0xFF:02X}" for x in data)


def checksum(packet_without_checksum: Iterable[int]) -> int:
    """Return HTD/Hiwonder packet checksum.

    Vendor examples invert the sum from servo_id through params, excluding the
    two 0x55 header bytes and the checksum byte itself.
    """

    data = list(packet_without_checksum)
    return (~sum(data[2:])) & 0xFF


def build_packet(servo_id: int, command: int, params: Sequence[int] | None = None) -> bytes:
    params = list(params or [])
    length = 3 + len(params)  # command + params + checksum
    packet = bytearray(HEADER)
    packet.append(int(servo_id) & 0xFF)
    packet.append(length & 0xFF)
    packet.append(int(command) & 0xFF)
    packet.extend(int(x) & 0xFF for x in params)
    packet.append(checksum(packet))
    return bytes(packet)


@dataclass(frozen=True)
class RawServoPacket:
    servo_id: int
    command: int
    params: tuple[int, ...]
    raw: bytes


def parse_packets(data: bytes, *, print_bad: bool = False) -> list[RawServoPacket]:
    packets: list[RawServoPacket] = []
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
        raw = data[i:end]
        expected = checksum(raw[:-1])
        if int(raw[-1]) != expected:
            if print_bad:
                print(
                    "Ignoring packet with bad checksum: "
                    f"{format_bytes(raw)} expected={expected:02X}",
                    flush=True,
                )
            continue
        packets.append(
            RawServoPacket(
                servo_id=int(raw[2]),
                command=int(raw[4]),
                params=tuple(int(x) for x in raw[5:-1]),
                raw=raw,
            )
        )
    return packets


@dataclass(frozen=True)
class SerialTransportConfig:
    port: str
    baudrate: int = 115200
    byte_timeout_s: float = 0.001
    write_timeout_s: float = 0.020
    bytesize: int = serial.EIGHTBITS
    parity: str = serial.PARITY_NONE
    stopbits: int = serial.STOPBITS_ONE
    reset_buffers_on_open: bool = True
    tx_enable_pin: int | None = None
    rx_enable_pin: int | None = None


class PacketTransport(Protocol):
    def write(self, packet: bytes) -> None:
        ...

    def read_available(self, *, deadline_s: float, quiet_s: float = 0.0005) -> bytes:
        ...

    def reset_input_buffer(self) -> None:
        ...

    def reset_output_buffer(self) -> None:
        ...

    def close(self) -> None:
        ...


class SerialTransport:
    """Byte transport for a USB serial TTL bus adapter.

    GPIO half-duplex direction control is intentionally not implemented here
    yet; the USB debug board path does not need it.
    """

    def __init__(self, config: SerialTransportConfig) -> None:
        self.config = config
        self._serial: serial.Serial | None = None

    @property
    def port(self) -> str:
        return self.config.port

    @property
    def baudrate(self) -> int:
        return int(self.config.baudrate)

    def open(self) -> None:
        if self._serial is not None and self._serial.is_open:
            return
        if self.config.tx_enable_pin is not None or self.config.rx_enable_pin is not None:
            raise NotImplementedError("GPIO half-duplex direction pins are not implemented")
        self._serial = serial.Serial(
            port=self.config.port,
            baudrate=int(self.config.baudrate),
            bytesize=self.config.bytesize,
            parity=self.config.parity,
            stopbits=self.config.stopbits,
            timeout=float(self.config.byte_timeout_s),
            write_timeout=float(self.config.write_timeout_s),
        )
        if self.config.reset_buffers_on_open:
            self.reset_input_buffer()
            self.reset_output_buffer()

    def write(self, packet: bytes) -> None:
        self.open()
        assert self._serial is not None
        self._serial.write(packet)
        self._serial.flush()

    def read_available(self, *, deadline_s: float, quiet_s: float = 0.0005) -> bytes:
        self.open()
        assert self._serial is not None
        chunks = bytearray()
        quiet_deadline: float | None = None
        deadline = float(deadline_s)
        while time.monotonic() < deadline:
            waiting = int(getattr(self._serial, "in_waiting", 0) or 0)
            data = self._serial.read(waiting or 1)
            if data:
                chunks.extend(data)
                quiet_deadline = time.monotonic() + float(quiet_s)
                continue
            if quiet_deadline is not None and time.monotonic() >= quiet_deadline:
                break
        return bytes(chunks)

    def reset_input_buffer(self) -> None:
        self.open()
        assert self._serial is not None
        self._serial.reset_input_buffer()

    def reset_output_buffer(self) -> None:
        self.open()
        assert self._serial is not None
        self._serial.reset_output_buffer()

    def close(self) -> None:
        if self._serial is not None:
            self._serial.close()
            self._serial = None


@dataclass(frozen=True)
class RawServoBusConfig:
    response_timeout_s: float = 0.006
    quiet_s: float = 0.0005
    broadcast_id: int = SERVO_BROADCAST_ID
    min_servo_id: int = 0
    max_servo_id: int = 253
    min_position: int = 0
    max_position: int = 1000


class RawServoBus:
    """Typed wrapper around the raw HTD/Hiwonder TTL bus-servo protocol."""

    def __init__(self, transport: PacketTransport, config: RawServoBusConfig | None = None) -> None:
        self.transport = transport
        self.config = config or RawServoBusConfig()

    def read_id(self, target_id: int | None = None) -> int | None:
        target = self.config.broadcast_id if target_id is None else int(target_id)
        packet = self._request_response(target, CMD_ID_READ)
        if packet is None or packet.command != CMD_ID_READ or not packet.params:
            return None
        if target != self.config.broadcast_id and packet.servo_id != target:
            return None
        return int(packet.params[0])

    def write_id(self, old_id: int, new_id: int) -> None:
        self._validate_servo_id(old_id, allow_broadcast=False)
        self._validate_servo_id(new_id, allow_broadcast=False)
        self._write_command(int(old_id), CMD_ID_WRITE, [int(new_id)])

    def move_time_write(self, servo_id: int, position: int, time_ms: int) -> None:
        self._write_position_command(servo_id, CMD_MOVE_TIME_WRITE, position, time_ms)

    def move_time_wait_write(self, servo_id: int, position: int, time_ms: int) -> None:
        self._write_position_command(servo_id, CMD_MOVE_TIME_WAIT_WRITE, position, time_ms)

    def move_start(self, servo_id: int) -> None:
        self._write_command(int(servo_id), CMD_MOVE_START, [])

    def move_stop(self, servo_id: int) -> None:
        self._write_command(int(servo_id), CMD_MOVE_STOP, [])

    def read_position(self, servo_id: int) -> int | None:
        packet = self._request_response(int(servo_id), CMD_POS_READ)
        if packet is None or packet.command != CMD_POS_READ or packet.servo_id != int(servo_id):
            return None
        if len(packet.params) < 2:
            return None
        return int(packet.params[0]) | (int(packet.params[1]) << 8)

    def read_positions(self, servo_ids: Sequence[int]) -> dict[int, int]:
        positions: dict[int, int] = {}
        for servo_id in servo_ids:
            pos = self.read_position(int(servo_id))
            if pos is not None:
                positions[int(servo_id)] = int(pos)
        return positions

    def unload(self, servo_id: int) -> None:
        self._write_command(int(servo_id), CMD_LOAD_OR_UNLOAD_WRITE, [0])

    def _write_position_command(
        self, servo_id: int, command: int, position: int, time_ms: int
    ) -> None:
        self._validate_servo_id(servo_id, allow_broadcast=False)
        position = self._validate_position(position)
        time_ms = max(0, min(30000, int(time_ms)))
        self._write_command(
            int(servo_id),
            int(command),
            [
                position & 0xFF,
                (position >> 8) & 0xFF,
                time_ms & 0xFF,
                (time_ms >> 8) & 0xFF,
            ],
        )

    def _write_command(self, servo_id: int, command: int, params: Sequence[int]) -> None:
        self._validate_servo_id(servo_id, allow_broadcast=True)
        self.transport.write(build_packet(int(servo_id), int(command), params))

    def _request_response(self, servo_id: int, command: int) -> RawServoPacket | None:
        self._validate_servo_id(servo_id, allow_broadcast=True)
        self.transport.reset_input_buffer()
        self.transport.write(build_packet(int(servo_id), int(command), []))
        data = self.transport.read_available(
            deadline_s=time.monotonic() + float(self.config.response_timeout_s),
            quiet_s=float(self.config.quiet_s),
        )
        for packet in parse_packets(data):
            if packet.command != int(command):
                continue
            if servo_id != self.config.broadcast_id and packet.servo_id != int(servo_id):
                continue
            return packet
        return None

    def _validate_servo_id(self, servo_id: int, *, allow_broadcast: bool) -> None:
        sid = int(servo_id)
        if allow_broadcast and sid == int(self.config.broadcast_id):
            return
        if sid < int(self.config.min_servo_id) or sid > int(self.config.max_servo_id):
            raise ValueError(f"servo id out of range: {sid}")

    def _validate_position(self, position: int) -> int:
        pos = int(position)
        if pos < int(self.config.min_position) or pos > int(self.config.max_position):
            raise ValueError(
                f"servo position out of range: {pos} "
                f"[{self.config.min_position}, {self.config.max_position}]"
            )
        return pos


__all__ = [
    "HEADER",
    "SERVO_BROADCAST_ID",
    "CMD_MOVE_TIME_WRITE",
    "CMD_MOVE_TIME_READ",
    "CMD_MOVE_TIME_WAIT_WRITE",
    "CMD_MOVE_TIME_WAIT_READ",
    "CMD_MOVE_START",
    "CMD_MOVE_STOP",
    "CMD_ID_WRITE",
    "CMD_ID_READ",
    "CMD_POS_READ",
    "CMD_LOAD_OR_UNLOAD_WRITE",
    "RawServoPacket",
    "RawServoBusConfig",
    "RawServoBus",
    "SerialTransportConfig",
    "SerialTransport",
    "build_packet",
    "checksum",
    "format_bytes",
    "parse_packets",
]
