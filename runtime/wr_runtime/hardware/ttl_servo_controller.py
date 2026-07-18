from __future__ import annotations

from typing import List, Optional, Tuple

from .hiwonder_ttl_bus import (
    RawServoBus,
    RawServoBusConfig,
    SerialTransport,
    SerialTransportConfig,
)


class TtlServoController:
    """Small compatibility wrapper over the raw Hiwonder TTL servo bus."""

    def __init__(self, raw_bus: RawServoBus) -> None:
        self.raw_bus = raw_bus

    def move_servos(self, servo_commands: List[Tuple[int, int]], time_ms: int) -> bool:
        for servo_id, position in servo_commands:
            self.raw_bus.move_time_write(int(servo_id), int(position), int(time_ms))
        return True

    def read_servo_positions(self, servo_ids: List[int]) -> Optional[List[Tuple[int, int]]]:
        ids = [int(servo_id) for servo_id in servo_ids]
        positions = self.raw_bus.read_positions(ids)
        if not positions:
            return None
        return [(servo_id, int(positions[servo_id])) for servo_id in ids if servo_id in positions]

    def unload_servos(self, servo_ids: List[int]) -> bool:
        for servo_id in servo_ids:
            self.raw_bus.unload(int(servo_id))
        return True

    def get_battery_voltage(self) -> Optional[float]:
        return None

    def close(self) -> None:
        self.raw_bus.transport.close()


def build_ttl_servo_controller(servo_controller_config) -> TtlServoController:
    controller_type = str(getattr(servo_controller_config, "type", "hiwonder_ttl_bus")).lower()
    if controller_type not in {"hiwonder_ttl_bus", "hiwonder_ttl_debug_board"}:
        raise ValueError(
            f"Unsupported servo_controller.type={getattr(servo_controller_config, 'type', None)!r}. "
            "Use 'hiwonder_ttl_bus' with the USB TTL debug board."
        )

    transport = SerialTransport(
        SerialTransportConfig(
            port=str(servo_controller_config.port),
            baudrate=int(servo_controller_config.baudrate),
        )
    )
    return TtlServoController(RawServoBus(transport, RawServoBusConfig()))
