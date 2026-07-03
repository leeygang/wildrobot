from runtime.wr_runtime.hardware.hiwonder_ttl_bus import (
    CMD_ID_READ,
    CMD_MOVE_TIME_WRITE,
    CMD_POS_READ,
    SERVO_BROADCAST_ID,
    RawServoBus,
    RawServoBusConfig,
    build_packet,
    parse_packets,
)


class FakeTransport:
    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.writes = []
        self.reset_input_count = 0
        self.reset_output_count = 0
        self.closed = False

    def write(self, packet: bytes) -> None:
        self.writes.append(bytes(packet))

    def read_available(self, *, deadline_s: float, quiet_s: float = 0.0005) -> bytes:
        if not self.responses:
            return b""
        return self.responses.pop(0)

    def reset_input_buffer(self) -> None:
        self.reset_input_count += 1

    def reset_output_buffer(self) -> None:
        self.reset_output_count += 1

    def close(self) -> None:
        self.closed = True


def test_build_packet_matches_validated_id_read_bytes():
    assert build_packet(SERVO_BROADCAST_ID, CMD_ID_READ) == bytes.fromhex("55 55 FE 03 0E F0")
    assert build_packet(1, CMD_ID_READ) == bytes.fromhex("55 55 01 03 0E ED")


def test_parse_id_read_response():
    packet = build_packet(3, CMD_ID_READ, [3])
    assert packet == bytes.fromhex("55 55 03 04 0E 03 E7")

    parsed = parse_packets(packet)
    assert len(parsed) == 1
    assert parsed[0].servo_id == 3
    assert parsed[0].command == CMD_ID_READ
    assert parsed[0].params == (3,)


def test_read_id_uses_broadcast_and_returns_reported_id():
    transport = FakeTransport([build_packet(3, CMD_ID_READ, [3])])
    bus = RawServoBus(transport, RawServoBusConfig(response_timeout_s=0.001))

    assert bus.read_id() == 3
    assert transport.writes == [build_packet(SERVO_BROADCAST_ID, CMD_ID_READ)]
    assert transport.reset_input_count == 1


def test_read_position_returns_little_endian_units():
    transport = FakeTransport([build_packet(3, CMD_POS_READ, [0xF5, 0x01])])
    bus = RawServoBus(transport, RawServoBusConfig(response_timeout_s=0.001))

    assert bus.read_position(3) == 501
    assert transport.writes == [build_packet(3, CMD_POS_READ)]


def test_move_time_write_builds_position_command():
    transport = FakeTransport()
    bus = RawServoBus(transport)

    bus.move_time_write(3, 501, 20)

    assert transport.writes == [
        build_packet(3, CMD_MOVE_TIME_WRITE, [0xF5, 0x01, 0x14, 0x00])
    ]
