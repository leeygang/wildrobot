from runtime.wr_runtime.hardware.hiwonder_ttl_bus import (
    CMD_ANGLE_LIMIT_READ,
    CMD_ID_READ,
    CMD_ID_WRITE,
    CMD_LED_CTRL_READ,
    CMD_LED_ERROR_READ,
    CMD_LOAD_OR_UNLOAD_READ,
    CMD_LOAD_OR_UNLOAD_WRITE,
    CMD_MOVE_TIME_READ,
    CMD_MOVE_TIME_WRITE,
    CMD_OR_MOTOR_MODE_READ,
    CMD_POS_READ,
    CMD_TEMP_MAX_LIMIT_READ,
    CMD_TEMP_READ,
    CMD_VIN_LIMIT_READ,
    CMD_VIN_READ,
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


def test_write_id_builds_targeted_id_command():
    transport = FakeTransport()
    bus = RawServoBus(transport)

    bus.write_id(3, 100)

    assert transport.writes == [build_packet(3, CMD_ID_WRITE, [100])]


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


def test_read_move_time_returns_target_and_duration():
    transport = FakeTransport(
        [build_packet(3, CMD_MOVE_TIME_READ, [0x71, 0x02, 0xD0, 0x05])]
    )
    bus = RawServoBus(transport, RawServoBusConfig(response_timeout_s=0.001))

    assert bus.read_move_time(3) == (625, 1488)
    assert transport.writes == [build_packet(3, CMD_MOVE_TIME_READ)]


def test_read_angle_limits_returns_little_endian_bounds():
    transport = FakeTransport(
        [build_packet(3, CMD_ANGLE_LIMIT_READ, [0x64, 0x00, 0x84, 0x03])]
    )
    bus = RawServoBus(transport, RawServoBusConfig(response_timeout_s=0.001))

    assert bus.read_angle_limits(3) == (100, 900)
    assert transport.writes == [build_packet(3, CMD_ANGLE_LIMIT_READ)]


def test_read_voltage_and_temperature_protection_limits():
    transport = FakeTransport(
        [
            build_packet(3, CMD_VIN_LIMIT_READ, [0x64, 0x19, 0xE0, 0x2E]),
            build_packet(3, CMD_TEMP_MAX_LIMIT_READ, [85]),
        ]
    )
    bus = RawServoBus(transport, RawServoBusConfig(response_timeout_s=0.001))

    assert bus.read_voltage_limits_v(3) == (6.5, 12.0)
    assert bus.read_temperature_limit_c(3) == 85


def test_read_motor_mode_led_and_alarm_configuration():
    transport = FakeTransport(
        [
            build_packet(3, CMD_OR_MOTOR_MODE_READ, [1, 0, 0x18, 0xFC]),
            build_packet(3, CMD_LED_CTRL_READ, [0]),
            build_packet(3, CMD_LED_ERROR_READ, [7]),
        ]
    )
    bus = RawServoBus(transport, RawServoBusConfig(response_timeout_s=0.001))

    assert bus.read_motor_mode(3) == (1, -1000)
    assert bus.read_led_enabled(3) is True
    assert bus.read_alarm_mask(3) == 7


def test_read_temperature_returns_degrees_celsius():
    transport = FakeTransport([build_packet(3, CMD_TEMP_READ, [42])])
    bus = RawServoBus(transport, RawServoBusConfig(response_timeout_s=0.001))

    assert bus.read_temperature_c(3) == 42
    assert transport.writes == [build_packet(3, CMD_TEMP_READ)]


def test_read_voltage_converts_millivolts_to_volts():
    transport = FakeTransport([build_packet(3, CMD_VIN_READ, [0x5C, 0x2B])])
    bus = RawServoBus(transport, RawServoBusConfig(response_timeout_s=0.001))

    assert bus.read_voltage_v(3) == 11.1
    assert transport.writes == [build_packet(3, CMD_VIN_READ)]


def test_load_and_unload_build_torque_enable_commands():
    transport = FakeTransport()
    bus = RawServoBus(transport)

    bus.load(3)
    bus.unload(3)

    assert transport.writes == [
        build_packet(3, CMD_LOAD_OR_UNLOAD_WRITE, [1]),
        build_packet(3, CMD_LOAD_OR_UNLOAD_WRITE, [0]),
    ]


def test_read_loaded_returns_servo_torque_state():
    transport = FakeTransport([build_packet(3, CMD_LOAD_OR_UNLOAD_READ, [1])])
    bus = RawServoBus(transport, RawServoBusConfig(response_timeout_s=0.001))

    assert bus.read_loaded(3) is True
    assert transport.writes == [build_packet(3, CMD_LOAD_OR_UNLOAD_READ)]
