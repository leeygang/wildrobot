from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .hiwonder_board_controller import HiwonderBoardController


class HiwonderBoardActuators:
    """Position-controlled actuator interface using the Hiwonder board.

    - Uses radians in/out at the API boundary.
    - Converts to Hiwonder servo units [0..1000] where 500 == 0 rad.
    - Provides position readback; velocity is estimated by finite differences.
    """

    def __init__(
        self,
        joint_names: List[str],
        servo_ids: Dict[str, int],
        joint_offsets_rad: Optional[Dict[str, float]] = None,
        port: str = "/dev/ttyUSB0",
        baudrate: int = 9600,
        default_move_duration_ms: int = 20,
    ):
        self.joint_names = list(joint_names)
        self.servo_ids = dict(servo_ids)
        self.joint_offsets_rad = dict(joint_offsets_rad or {})
        self.controller = HiwonderBoardController(port=port, baudrate=baudrate)
        self.default_move_duration_ms = int(default_move_duration_ms)

        missing = [j for j in self.joint_names if j not in self.servo_ids]
        if missing:
            raise ValueError(f"Missing servo_ids for joints: {missing}")

        self._last_pos: Optional[np.ndarray] = None

    def close(self) -> None:
        self.controller.close()

    def enable(self) -> None:
        # Board protocol doesn't have an explicit global enable; servos are powered externally.
        pass

    def disable(self) -> None:
        servo_ids = [self.servo_ids[j] for j in self.joint_names]
        self.controller.unload_servos(servo_ids)

    def get_battery_voltage(self) -> Optional[float]:
        return self.controller.get_battery_voltage()

    def set_targets_rad(self, targets_rad: np.ndarray, move_duration_ms: Optional[int] = None) -> None:
        targets = np.asarray(targets_rad, dtype=np.float32).reshape(-1)
        if targets.shape[0] != len(self.joint_names):
            raise ValueError(f"Expected {len(self.joint_names)} targets, got {targets.shape}")

        duration = int(move_duration_ms) if move_duration_ms is not None else self.default_move_duration_ms

        cmds: List[Tuple[int, int]] = []
        for joint_name, rad in zip(self.joint_names, targets, strict=True):
            offset = float(self.joint_offsets_rad.get(joint_name, 0.0))
            servo_id = int(self.servo_ids[joint_name])
            units = _radians_to_servo_units(float(rad) + offset)
            cmds.append((servo_id, units))

        self.controller.move_servos(cmds, duration)

    def get_positions_rad(self) -> Optional[np.ndarray]:
        servo_ids = [self.servo_ids[j] for j in self.joint_names]
        resp = self.controller.read_servo_positions(servo_ids)
        if resp is None:
            return None

        by_id = {sid: pos for sid, pos in resp}
        out: List[float] = []
        for joint_name in self.joint_names:
            sid = self.servo_ids[joint_name]
            units = by_id.get(sid)
            if units is None:
                return None
            offset = float(self.joint_offsets_rad.get(joint_name, 0.0))
            out.append(_servo_units_to_radians(int(units)) - offset)

        return np.asarray(out, dtype=np.float32)

    def estimate_velocities_rad_s(self, dt: float) -> np.ndarray:
        pos = self.get_positions_rad()
        if pos is None:
            return np.zeros(len(self.joint_names), dtype=np.float32)

        if self._last_pos is None or dt <= 0:
            self._last_pos = pos
            return np.zeros_like(pos)

        vel = (pos - self._last_pos) / float(dt)
        self._last_pos = pos
        return vel.astype(np.float32)


def _radians_to_servo_units(radians: float) -> int:
    # 240 degrees total range == 4.18879 rad, centered at 500
    units = 500 + (radians * 1000.0 / 4.18879)
    units = int(max(0, min(1000, units)))
    return units


def _servo_units_to_radians(units: int) -> float:
    return (int(units) - 500) * 4.18879 / 1000.0
