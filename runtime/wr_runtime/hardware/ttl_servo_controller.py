from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
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

    def probe_servo_id(self, servo_id: int) -> bool:
        return self.raw_bus.read_id(int(servo_id)) == int(servo_id)

    def unload_servos(self, servo_ids: List[int]) -> bool:
        for servo_id in servo_ids:
            self.raw_bus.unload(int(servo_id))
        return True

    def get_battery_voltage(self) -> Optional[float]:
        return None

    def close(self) -> None:
        self.raw_bus.transport.close()


class MultiBoardTtlServoController:
    """Calibration controller that routes globally unique IDs by USB board."""

    def __init__(self, controllers_by_port, servo_ids_by_port) -> None:
        self.controllers_by_port = dict(controllers_by_port)
        self.servo_ids_by_port = {
            str(port): tuple(int(x) for x in servo_ids)
            for port, servo_ids in servo_ids_by_port.items()
        }
        self._port_by_servo_id: dict[int, str] = {}
        for port, servo_ids in self.servo_ids_by_port.items():
            for servo_id in servo_ids:
                if servo_id in self._port_by_servo_id:
                    raise ValueError(f"servo id {servo_id} is assigned to multiple boards")
                self._port_by_servo_id[servo_id] = port
        self._executor = ThreadPoolExecutor(
            max_workers=len(self.controllers_by_port),
            thread_name_prefix="ServoBoardCalibration",
        )

    def _partition(self, servo_ids) -> dict[str, list[int]]:
        by_port: dict[str, list[int]] = {}
        for servo_id in servo_ids:
            sid = int(servo_id)
            port = self._port_by_servo_id.get(sid)
            if port is None:
                raise KeyError(f"servo id {sid} is not assigned to a configured board")
            by_port.setdefault(port, []).append(sid)
        return by_port

    def move_servos(self, servo_commands: List[Tuple[int, int]], time_ms: int) -> bool:
        command_by_id = {int(sid): int(position) for sid, position in servo_commands}
        by_port = self._partition(command_by_id)
        futures = [
            self._executor.submit(
                self.controllers_by_port[port].move_servos,
                [(sid, command_by_id[sid]) for sid in servo_ids],
                int(time_ms),
            )
            for port, servo_ids in by_port.items()
        ]
        results = [bool(future.result()) for future in futures]
        return all(results)

    def read_servo_positions(self, servo_ids: List[int]) -> Optional[List[Tuple[int, int]]]:
        requested = [int(servo_id) for servo_id in servo_ids]
        by_port = self._partition(requested)
        futures = {
            port: self._executor.submit(
                self.controllers_by_port[port].read_servo_positions,
                ids,
            )
            for port, ids in by_port.items()
        }
        positions: dict[int, int] = {}
        for future in futures.values():
            for servo_id, position in future.result() or []:
                positions[int(servo_id)] = int(position)
        result = [(servo_id, positions[servo_id]) for servo_id in requested if servo_id in positions]
        return result or None

    def unload_servos(self, servo_ids: List[int]) -> bool:
        by_port = self._partition(servo_ids)
        futures = [
            self._executor.submit(
                self.controllers_by_port[port].unload_servos,
                ids,
            )
            for port, ids in by_port.items()
        ]
        results = [bool(future.result()) for future in futures]
        return all(results)

    def get_battery_voltage(self) -> Optional[float]:
        for controller in self.controllers_by_port.values():
            voltage = controller.get_battery_voltage()
            if voltage is not None:
                return float(voltage)
        return None

    def close(self) -> None:
        try:
            for controller in self.controllers_by_port.values():
                controller.close()
        finally:
            self._executor.shutdown(wait=True)


def _build_single_ttl_servo_controller(*, port: str, baudrate: int) -> TtlServoController:
    transport = SerialTransport(
        SerialTransportConfig(port=str(port), baudrate=int(baudrate))
    )
    return TtlServoController(RawServoBus(transport, RawServoBusConfig()))


def build_ttl_servo_controller(servo_controller_config):
    controller_type = str(getattr(servo_controller_config, "type", "hiwonder_ttl_bus")).lower()
    if controller_type not in {"hiwonder_ttl_bus", "hiwonder_ttl_debug_board"}:
        raise ValueError(
            f"Unsupported servo_controller.type={getattr(servo_controller_config, 'type', None)!r}. "
            "Use 'hiwonder_ttl_bus' with the USB TTL debug board."
        )

    boards = tuple(getattr(servo_controller_config, "boards", ()) or ())
    if not boards:
        return _build_single_ttl_servo_controller(
            port=str(servo_controller_config.port),
            baudrate=int(servo_controller_config.baudrate),
        )
    controllers_by_port = {}
    try:
        for board in boards:
            controllers_by_port[str(board.port)] = _build_single_ttl_servo_controller(
                port=str(board.port),
                baudrate=int(servo_controller_config.baudrate),
            )
    except Exception:
        for controller in controllers_by_port.values():
            controller.close()
        raise
    if len(controllers_by_port) == 1:
        return next(iter(controllers_by_port.values()))
    return MultiBoardTtlServoController(
        controllers_by_port,
        {str(board.port): tuple(board.servo_ids) for board in boards},
    )
