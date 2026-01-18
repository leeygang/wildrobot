"""Runtime configuration for WildRobot hardware deployment.

This module provides a structured configuration class for running the WildRobot
policy on real hardware. Configuration is loaded from a JSON file that specifies:
- Policy and model paths
- Control loop parameters
- Hiwonder servo board settings (port, servo IDs, joint offsets)
- BNO085 IMU settings
- Foot switch GPIO pins

Usage:
    from wr_runtime.config import WildRobotRuntimeConfig

    # Load from default path (~/.wildrobot/config.json)
    config = WildRobotRuntimeConfig.load()

    # Load from specific path
    config = WildRobotRuntimeConfig.load("/path/to/config.json")

    # Access nested config
    print(config.control.hz)  # 50.0
    print(config.hiwonder.servo_ids["left_hip_pitch"])  # 1
    print(config.bno085.i2c_address)  # 0x4A
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


# =============================================================================
# Default paths
# =============================================================================

HOME_DIR = Path(os.path.expanduser("~"))
DEFAULT_CONFIG_DIR = HOME_DIR / ".wildrobot"
DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_DIR / "config.json"


# =============================================================================
# Nested Configuration Dataclasses
# =============================================================================


@dataclass(frozen=True)
class ControlConfig:
    """Control loop configuration.

    Attributes:
        hz: Control loop frequency in Hz (default: 50)
        action_scale_rad: Scale factor for policy actions in radians (default: 0.35)
        velocity_cmd: Initial velocity command for the policy (default: 0.0)
    """

    hz: float = 50.0
    action_scale_rad: float = 0.35
    velocity_cmd: float = 0.0

    @property
    def dt(self) -> float:
        """Control loop period in seconds."""
        return 1.0 / self.hz


@dataclass(frozen=True)
class HiwonderConfig:
    """Hiwonder servo board configuration.

    Attributes:
        port: Serial port for the Hiwonder board (e.g., "/dev/ttyUSB0")
        baudrate: Serial baudrate (default: 9600)
        servo_ids: Mapping from joint name to servo ID (1-8)
         joint_offsets_rad: Calibration offsets per joint in radians (applied before rad→servo units conversion)
         joint_directions: Per-joint sign (+1 or -1) to handle mechanical reversals
     """

    port: str = "/dev/ttyUSB0"
    baudrate: int = 9600
    servo_ids: Dict[str, int] = field(default_factory=dict)
    joint_offsets_rad: Dict[str, float] = field(default_factory=dict)
    joint_directions: Dict[str, float] = field(default_factory=dict)

    # Joint ordering (matches training config actuated_joint_specs)
    JOINT_ORDER = (
        "left_hip_pitch",
        "left_hip_roll",
        "left_knee_pitch",
        "left_ankle_pitch",
        "right_hip_pitch",
        "right_hip_roll",
        "right_knee_pitch",
        "right_ankle_pitch",
    )

    def get_servo_id(self, joint_name: str) -> int:
        """Get servo ID for a joint, raising error if not configured."""
        if joint_name not in self.servo_ids:
            raise KeyError(
                f"Servo ID not configured for joint '{joint_name}'. "
                f"Available joints: {list(self.servo_ids.keys())}"
            )
        return self.servo_ids[joint_name]

    def get_offset_rad(self, joint_name: str) -> float:
        """Get calibration offset for a joint (default: 0.0)."""
        return self.joint_offsets_rad.get(joint_name, 0.0)

    def get_ordered_servo_ids(self) -> list[int]:
        """Get servo IDs in canonical joint order."""
        return [self.get_servo_id(name) for name in self.JOINT_ORDER]


@dataclass(frozen=True)
class BNO085Config:
    """BNO085 IMU configuration.

    The BNO085 is a 9-DOF IMU with onboard sensor fusion that provides:
    - Orientation quaternion (framequat)
    - Angular velocity (gyro)
    - Linear acceleration (accelerometer)

    Attributes:
        i2c_address: I2C address (default: 0x4A, alternative: 0x4B)
        upside_down: Whether the IMU is mounted upside-down (flips orientation)
    """

    i2c_address: int = 0x4A
    upside_down: bool = False


@dataclass(frozen=True)
class FootSwitchConfig:
    """Foot switch GPIO configuration.

    Each foot has two switches (toe and heel) for contact detection.
    Pin names are Blinka board designations (e.g., "D5", "D13").

    Attributes:
        left_toe: GPIO pin name for left toe switch
        left_heel: GPIO pin name for left heel switch
        right_toe: GPIO pin name for right toe switch
        right_heel: GPIO pin name for right heel switch
    """

    left_toe: str = "D5"
    left_heel: str = "D6"
    right_toe: str = "D13"
    right_heel: str = "D19"

    def get_all_pins(self) -> Dict[str, str]:
        """Get all foot switch pins as a dict."""
        return {
            "left_toe": self.left_toe,
            "left_heel": self.left_heel,
            "right_toe": self.right_toe,
            "right_heel": self.right_heel,
        }

    def get_ordered_pins(self) -> tuple[str, str, str, str]:
        """Get pins in order: (left_toe, left_heel, right_toe, right_heel)."""
        return (self.left_toe, self.left_heel, self.right_toe, self.right_heel)


# =============================================================================
# Main Configuration Class
# =============================================================================


@dataclass(frozen=True)
class WildRobotRuntimeConfig:
    """Complete runtime configuration for WildRobot hardware deployment.

    This class provides a structured way to load and access all configuration
    needed to run a trained policy on the physical robot.

    Attributes:
        mjcf_path: Path to MuJoCo XML model file
        policy_onnx_path: Path to exported ONNX policy file
        control: Control loop settings (hz, action_scale, velocity_cmd)
        hiwonder: Hiwonder servo board settings
        bno085: BNO085 IMU settings
        foot_switches: Foot switch GPIO pin settings

    Example:
        config = WildRobotRuntimeConfig.load("config.json")

        # Access paths
        model_path = config.mjcf_path
        policy_path = config.policy_onnx_path

        # Access control settings
        dt = config.control.dt  # 0.02 for 50 Hz

        # Access hardware settings
        servo_id = config.hiwonder.get_servo_id("left_hip_pitch")
        imu_addr = config.bno085.i2c_address
        toe_pin = config.foot_switches.left_toe
    """

    mjcf_path: str
    policy_onnx_path: str
    control: ControlConfig
    hiwonder: HiwonderConfig
    bno085: BNO085Config
    foot_switches: FootSwitchConfig

    # Store config directory for resolving relative paths
    _config_dir: Path = field(default=Path("."), repr=False, compare=False)

    def resolve_path(self, path: str) -> Path:
        """Resolve a path relative to the config file directory.

        Args:
            path: Path string (may be relative or absolute)

        Returns:
            Resolved absolute Path
        """
        p = Path(path)
        if p.is_absolute():
            return p
        return (self._config_dir / p).resolve()

    @property
    def mjcf_resolved_path(self) -> Path:
        """Get the resolved MJCF model path."""
        return self.resolve_path(self.mjcf_path)

    @property
    def policy_resolved_path(self) -> Path:
        """Get the resolved policy ONNX path."""
        return self.resolve_path(self.policy_onnx_path)

    @classmethod
    def load(cls, config_path: Optional[str | Path] = None) -> WildRobotRuntimeConfig:
        """Load runtime configuration from JSON file.

        Args:
            config_path: Path to JSON config file. If None, searches for:
                1. ~/.wildrobot/config.json
                2. ~/wildrobot_config.json (legacy)

        Returns:
            WildRobotRuntimeConfig instance

        Raises:
            FileNotFoundError: If config file not found
            json.JSONDecodeError: If config file is not valid JSON
            KeyError: If required fields are missing
        """
        # Resolve config path
        if config_path is None:
            if DEFAULT_CONFIG_PATH.exists():
                config_path = DEFAULT_CONFIG_PATH
            elif (HOME_DIR / "wildrobot_config.json").exists():
                config_path = HOME_DIR / "wildrobot_config.json"
            else:
                raise FileNotFoundError(
                    f"Config file not found. Searched:\n"
                    f"  - {DEFAULT_CONFIG_PATH}\n"
                    f"  - {HOME_DIR / 'wildrobot_config.json'}\n"
                    f"Please create a config file or specify path explicitly."
                )

        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        config_dir = path.parent.resolve()

        # Load JSON
        data = json.loads(path.read_text())

        # Parse nested configs
        control = cls._parse_control_config(data)
        hiwonder = cls._parse_hiwonder_config(data)
        bno085 = cls._parse_bno085_config(data)
        foot_switches = cls._parse_foot_switch_config(data)

        return cls(
            mjcf_path=data["mjcf_path"],
            policy_onnx_path=data["policy_onnx_path"],
            control=control,
            hiwonder=hiwonder,
            bno085=bno085,
            foot_switches=foot_switches,
            _config_dir=config_dir,
        )

    @staticmethod
    def _parse_control_config(data: dict) -> ControlConfig:
        """Parse control configuration from JSON data."""
        return ControlConfig(
            hz=float(data.get("control_hz", 50.0)),
            action_scale_rad=float(data.get("action_scale_rad", 0.35)),
            velocity_cmd=float(data.get("velocity_cmd", 0.0)),
        )

    @staticmethod
    def _parse_hiwonder_config(data: dict) -> HiwonderConfig:
        """Parse Hiwonder configuration from JSON data."""
        hiw = data.get("hiwonder", {})

        servo_ids = {str(k): int(v) for k, v in hiw.get("servo_ids", {}).items()}
        joint_offsets = {
            str(k): float(v) for k, v in hiw.get("joint_offsets_rad", {}).items()
        }
        joint_directions = {str(k): float(v) for k, v in hiw.get("joint_directions", {}).items()}
        if joint_directions:
            invalid = {
                k: v for k, v in joint_directions.items() if abs(abs(v) - 1.0) > 1e-3
            }
            if invalid:
                raise ValueError(
                    "joint_directions must be +/-1.0 (tolerance 1e-3); invalid entries: "
                    + ", ".join(f"{k}={v}" for k, v in invalid.items())
                )

        return HiwonderConfig(
            port=hiw.get("port", "/dev/ttyUSB0"),
            baudrate=int(hiw.get("baudrate", 9600)),
            servo_ids=servo_ids,
            joint_offsets_rad=joint_offsets,
            joint_directions=joint_directions,
        )

    @staticmethod
    def _parse_bno085_config(data: dict) -> BNO085Config:
        """Parse BNO085 configuration from JSON data."""
        bno = data.get("bno085", {})

        # Parse I2C address (may be hex string like "0x4A" or int)
        i2c_addr = bno.get("i2c_address", 0x4A)
        if isinstance(i2c_addr, str):
            i2c_addr = int(i2c_addr, 0)  # Handles "0x4A" format

        return BNO085Config(
            i2c_address=i2c_addr,
            upside_down=bool(bno.get("upside_down", False)),
        )

    @staticmethod
    def _parse_foot_switch_config(data: dict) -> FootSwitchConfig:
        """Parse foot switch configuration from JSON data."""
        foot = data.get("foot_switches", {})

        return FootSwitchConfig(
            left_toe=str(foot.get("left_toe", "D5")),
            left_heel=str(foot.get("left_heel", "D6")),
            right_toe=str(foot.get("right_toe", "D13")),
            right_heel=str(foot.get("right_heel", "D19")),
        )

    def to_dict(self) -> dict:
        """Convert config back to a dictionary (for serialization)."""
        return {
            "mjcf_path": self.mjcf_path,
            "policy_onnx_path": self.policy_onnx_path,
            "control_hz": self.control.hz,
            "action_scale_rad": self.control.action_scale_rad,
            "velocity_cmd": self.control.velocity_cmd,
            "hiwonder": {
                "port": self.hiwonder.port,
                "baudrate": self.hiwonder.baudrate,
                "servo_ids": self.hiwonder.servo_ids,
                "joint_offsets_rad": self.hiwonder.joint_offsets_rad,
                "joint_directions": self.hiwonder.joint_directions,
            },
            "bno085": {
                "i2c_address": hex(self.bno085.i2c_address),
                "upside_down": self.bno085.upside_down,
            },
            "foot_switches": self.foot_switches.get_all_pins(),
        }

    def save(self, config_path: str | Path) -> None:
        """Save configuration to JSON file.

        Args:
            config_path: Path to save the config file
        """
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))


# =============================================================================
# Legacy Compatibility
# =============================================================================

# Alias for backward compatibility
RuntimeConfig = WildRobotRuntimeConfig


def load_config(config_path: Optional[str] = None) -> WildRobotRuntimeConfig:
    """Load runtime configuration (legacy function).

    Deprecated: Use WildRobotRuntimeConfig.load() instead.

    Args:
        config_path: Path to config file (optional)

    Returns:
        WildRobotRuntimeConfig instance
    """
    return WildRobotRuntimeConfig.load(config_path)
