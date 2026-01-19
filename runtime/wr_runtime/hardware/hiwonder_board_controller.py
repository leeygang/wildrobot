"""Hiwonder Bus Servo Controller board interface.

This module provides low-level serial communication with the Hiwonder
servo controller board using the board-level protocol.

Usage:
    from runtime.configs import WrRuntimeConfig
    from runtime.hardware.hiwonder_board_controller import HiwonderBoardController

    config = WrRuntimeConfig.load("runtime/configs/wr_runtime_config.json")
    controller = HiwonderBoardController(config.hiwonder_controller)

    # Convert policy actions to servo commands and move
    actions = [0.0] * 8  # Policy outputs
    servo_cmds = config.hiwonder_controller.policy_action_to_servo_cmd(actions)
    time_ms = int(config.control.dt * 1000)  # e.g., 20ms at 50Hz
    controller.move_servos(servo_cmds, time_ms)

    # Read servo positions and convert back to policy actions
    servo_ids = [s.id for s in config.hiwonder_controller.servos.values()]
    positions = controller.read_servo_positions(servo_ids)
    actions = config.hiwonder_controller.servo_pos_to_policy_action(positions)
"""

from __future__ import annotations

import serial
import time
from typing import List, Optional, Tuple

from runtime.configs.config import ServoControllerConfig


class HiwonderBoardController:
    """Hiwonder Bus Servo Controller board interface.

    This is a minimal, MJCF-runtime-friendly controller for Hiwonder servo boards.

    Frame format (per Hiwonder protocol):
        [0x55][0x55][Length][Command][...params...]

    Notes:
        - This controller uses board-level commands (not per-servo LX-16A style).
        - Velocity feedback is not available via this protocol.
    """

    CMD_SERVO_MOVE = 0x03
    CMD_GET_BATTERY_VOLTAGE = 0x0F
    CMD_MULT_SERVO_UNLOAD = 0x14
    CMD_MULT_SERVO_POS_READ = 0x15

    HEADER = bytes([0x55, 0x55])

    def __init__(
        self,
        config: Optional[ServoControllerConfig] = None,
        port: Optional[str] = None,
        baudrate: Optional[int] = None,
        timeout: float = 0.5,
    ):
        """Initialize the Hiwonder board controller.

        Args:
            config: ServoControllerConfig with port/baudrate settings.
                If provided, port and baudrate args are ignored.
            port: Serial port (e.g., "/dev/ttyUSB0"). Used if config is None.
            baudrate: Serial baudrate (default: 9600). Used if config is None.
            timeout: Serial read timeout in seconds.
        """
        if config is not None:
            port = config.port
            baudrate = config.baudrate
        else:
            port = port or "/dev/ttyUSB0"
            baudrate = baudrate or 9600

        self.serial = serial.Serial(
            port=port,
            baudrate=baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=timeout,
        )
        time.sleep(0.1)
        self.serial.reset_input_buffer()
        self.serial.reset_output_buffer()

    def close(self) -> None:
        """Close the serial connection."""
        try:
            self.serial.close()
        except Exception:
            pass

    def _send(self, command: int, params: Optional[List[int]] = None) -> None:
        """Send a command packet to the board."""
        if params is None:
            params = []
        length = 2 + len(params)
        pkt = bytearray(self.HEADER)
        pkt.append(length)
        pkt.append(command)
        pkt.extend(params)
        self.serial.write(pkt)
        self.serial.flush()

    def _read_response(self, timeout: Optional[float] = None) -> Optional[List[int]]:
        """Read a response packet from the board."""
        old_timeout = None
        if timeout is not None:
            old_timeout = self.serial.timeout
            self.serial.timeout = timeout

        try:
            # Find header
            while True:
                b = self.serial.read(1)
                if len(b) == 0:
                    return None
                if b == b"\x55":
                    b2 = self.serial.read(1)
                    if b2 == b"\x55":
                        break

            meta = self.serial.read(2)
            if len(meta) != 2:
                return None
            length = meta[0]
            command = meta[1]

            remaining = length - 2
            data = self.serial.read(remaining)
            if len(data) != remaining:
                return None

            return [length, command, *list(data)]
        finally:
            if timeout is not None and old_timeout is not None:
                self.serial.timeout = old_timeout

    def move_servos(self, servo_commands: List[Tuple[int, int]], time_ms: int) -> bool:
        """Move multiple servos to specified positions.

        Args:
            servo_commands: List of (servo_id, position) tuples.
                Position is in range [0, 1000].
            time_ms: Movement time in milliseconds.

        Returns:
            True if command was sent successfully.
        """
        params: List[int] = [len(servo_commands), time_ms & 0xFF, (time_ms >> 8) & 0xFF]
        for servo_id, position in servo_commands:
            params.append(int(servo_id))
            params.append(int(position) & 0xFF)
            params.append((int(position) >> 8) & 0xFF)
        self._send(self.CMD_SERVO_MOVE, params)
        return True

    def unload_servos(self, servo_ids: List[int]) -> bool:
        """Unload (disable torque) for specified servos.

        Args:
            servo_ids: List of servo IDs to unload.

        Returns:
            True if command was sent successfully.
        """
        params = [len(servo_ids)] + [int(i) for i in servo_ids]
        self._send(self.CMD_MULT_SERVO_UNLOAD, params)
        return True

    def get_battery_voltage(self) -> Optional[float]:
        """Read the battery voltage.

        Returns:
            Battery voltage in volts, or None if read failed.
        """
        self._send(self.CMD_GET_BATTERY_VOLTAGE)
        resp = self._read_response(timeout=1.0)
        if not resp or resp[1] != self.CMD_GET_BATTERY_VOLTAGE or resp[0] != 4:
            return None
        mv = resp[2] | (resp[3] << 8)
        return mv / 1000.0

    def read_servo_positions(
        self, servo_ids: List[int]
    ) -> Optional[List[Tuple[int, int]]]:
        """Read current positions of specified servos.

        Args:
            servo_ids: List of servo IDs to read.

        Returns:
            List of (servo_id, position) tuples, or None if read failed.
        """
        params = [len(servo_ids)] + [int(i) for i in servo_ids]
        self._send(self.CMD_MULT_SERVO_POS_READ, params)
        resp = self._read_response(timeout=1.0)
        if not resp or resp[1] != self.CMD_MULT_SERVO_POS_READ:
            return None

        count = resp[2]
        positions: List[Tuple[int, int]] = []
        for i in range(count):
            offset = 3 + i * 3
            if offset + 2 >= len(resp):
                return None
            sid = resp[offset]
            pos = resp[offset + 1] | (resp[offset + 2] << 8)
            positions.append((sid, pos))
        return positions
