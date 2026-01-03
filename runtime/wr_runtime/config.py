from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Dict, Optional


HOME_DIR = os.path.expanduser("~")


@dataclass(frozen=True)
class RuntimeConfig:
    mjcf_path: str
    policy_onnx_path: str

    control_hz: float = 50.0
    action_scale_rad: float = 0.35
    velocity_cmd: float = 0.0

    # Hiwonder board config
    hiwonder_port: str = "/dev/ttyUSB0"
    hiwonder_baudrate: int = 9600
    servo_ids: Dict[str, int] = None  # joint_name -> servo_id
    joint_offsets_rad: Dict[str, float] = None  # joint_name -> offset

    # BNO085 config
    bno085_i2c_address: int = 0x4A
    bno085_upside_down: bool = False

    # Foot switches: Blinka board pin names (strings)
    foot_switch_pins: Dict[str, str] = None  # left_toe/left_heel/right_toe/right_heel


def _parse_int_maybe_hex(val) -> int:
    if isinstance(val, int):
        return val
    if isinstance(val, str):
        return int(val, 0)
    raise TypeError(f"Expected int/str, got {type(val)}")


def load_config(config_path: Optional[str] = None) -> RuntimeConfig:
    """Load runtime configuration.

    Defaults to `~/wildrobot_config.json`.
    """
    if config_path is None:
        config_path = f"{HOME_DIR}/wildrobot_config.json"

    path = Path(config_path)
    data = json.loads(path.read_text())

    hiw = data.get("hiwonder", {})
    bno = data.get("bno085", {})
    foot = data.get("foot_switches", {})

    servo_ids = hiw.get("servo_ids", {})
    joint_offsets = hiw.get("joint_offsets_rad", {})

    i2c_addr = _parse_int_maybe_hex(bno.get("i2c_address", "0x4A"))

    return RuntimeConfig(
        mjcf_path=data["mjcf_path"],
        policy_onnx_path=data["policy_onnx_path"],
        control_hz=float(data.get("control_hz", 50.0)),
        action_scale_rad=float(data.get("action_scale_rad", 0.35)),
        velocity_cmd=float(data.get("velocity_cmd", 0.0)),
        hiwonder_port=hiw.get("port", "/dev/ttyUSB0"),
        hiwonder_baudrate=int(hiw.get("baudrate", 9600)),
        servo_ids={str(k): int(v) for k, v in servo_ids.items()},
        joint_offsets_rad={str(k): float(v) for k, v in joint_offsets.items()},
        bno085_i2c_address=i2c_addr,
        bno085_upside_down=bool(bno.get("upside_down", False)),
        foot_switch_pins={str(k): str(v) for k, v in foot.items()},
    )
