"""Runtime configuration for WildRobot hardware deployment.

This module provides a structured configuration class for running the WildRobot
policy on real hardware. Configuration is loaded from a JSON file that specifies:
- Policy and model paths
- Control loop parameters
- Hiwonder servo board settings (port, servo IDs, joint offsets)
- BNO085 IMU settings
- Foot switch GPIO pins

Usage:
    from runtime.configs import WrRuntimeConfig

    # Load from default path (~/.wildrobot/config.json)
    config = WrRuntimeConfig.load()

    # Load from specific path
    config = WrRuntimeConfig.load("/path/to/config.json")

    # Access nested config
    print(config.control.hz)  # 50.0
    print(config.hiwonder_controller.get_servo_id("left_hip_pitch"))  # 1
    print(config.bno085.i2c_address)  # 0x4A
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


import yaml


# =============================================================================
# Default paths
# =============================================================================

HOME_DIR = Path(os.path.expanduser("~"))
DEFAULT_CONFIG_DIR = HOME_DIR / ".wildrobot"
DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_DIR / "config.json"
DEFAULT_ROBOT_CONFIG_PATH = Path("assets/robot_config.yaml")


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
class ServoConfig:
    """Configuration for a single servo.

    Attributes:
        id: Servo ID (1-254)
        offset: Calibration offset in servo units (default: 0)
        direction: Hardware direction correction (+1 or -1, default: +1)
        rad_range: Joint range in radians (min, max) from robot_config.yaml
        max_velocity: Maximum joint velocity in rad/s from robot_config.yaml
        mirror_sign: +1.0 or -1.0 for action direction correction from robot_config.yaml
    """

    id: int
    offset: int = 0
    direction: int = 1
    rad_range: Tuple[float, float] = (0.0, 0.0)
    max_velocity: float = 10.0
    mirror_sign: float = 1.0

    # Servo conversion constants
    # Hiwonder HTD-45H: 240° range = 4.1887902 rad, units [0, 1000], center at 500
    UNITS_MIN: int = 0
    UNITS_MAX: int = 1000
    UNITS_CENTER: int = 500
    RANGE_RAD: float = 4.1887902047863905  # 240 degrees in radians
    UNITS_PER_RAD: float = 1000.0 / 4.1887902047863905  # ~238.73

    def rad_to_units(self, target_rad: float) -> int:
        """Convert MuJoCo radians to servo units.

        Applies hardware calibration:
        - direction: corrects for servo installation orientation (+1 or -1)
        - offset: corrects for neutral position alignment (in servo units)

        Args:
            target_rad: Joint target in MuJoCo radians (mirror_sign already applied
                        by policy_contract if coming from policy output)

        Returns:
            Servo position in units [0, 1000]
        """
        units = self.UNITS_CENTER + self.direction * target_rad * self.UNITS_PER_RAD + self.offset
        return int(max(self.UNITS_MIN, min(self.UNITS_MAX, round(units))))

    def rad_to_units_for_calibrate(
        self,
        target_rad: float,
        *,
        direction: int,
        offset: int,
    ) -> int:
        """Convert MuJoCo radians to servo units with overridden calibration values.

        Used during calibration when testing different direction/offset values.

        Args:
            target_rad: Joint target in MuJoCo radians
            direction: Direction to use (+1 or -1)
            offset: Offset to use (in servo units)

        Returns:
            Servo position in units [0, 1000]
        """
        units = self.UNITS_CENTER + direction * target_rad * self.UNITS_PER_RAD + offset
        return int(max(self.UNITS_MIN, min(self.UNITS_MAX, round(units))))

    def units_to_rad(self, units: int) -> float:
        """Convert servo units to MuJoCo radians.

        Inverse of rad_to_units. Applies hardware calibration:
        - direction: corrects for servo installation orientation
        - offset: corrects for neutral position alignment

        Args:
            units: Servo position in units [0, 1000]

        Returns:
            Joint position in MuJoCo radians
        """
        return self.direction * (units - self.UNITS_CENTER - self.offset) / self.UNITS_PER_RAD

    def units_to_rad_for_calibrate(
        self,
        units: int,
        *,
        direction: int,
        offset: int,
    ) -> float:
        """Convert servo units to MuJoCo radians with overridden calibration values.

        Used during calibration when testing different direction/offset values.

        Args:
            units: Servo position in units [0, 1000]
            direction: Direction to use (+1 or -1)
            offset: Offset to use (in servo units)

        Returns:
            Joint position in MuJoCo radians
        """
        return direction * (units - self.UNITS_CENTER - offset) / self.UNITS_PER_RAD

    @property
    def effective_sign(self) -> float:
        """Sign applied to ctrl_rad<->servo conversion (hardware direction only)."""
        return float(self.direction)

    @property
    def ctrl_center(self) -> float:
        """Center of joint range in radians."""
        return (self.rad_range[0] + self.rad_range[1]) / 2

    @property
    def ctrl_span(self) -> float:
        """Half-span of joint range in radians."""
        return (self.rad_range[1] - self.rad_range[0]) / 2


@dataclass(frozen=True)
class HiwonderControllerConfig:
    """Hiwonder servo controller board configuration.

    Attributes:
        port: Serial port for the Hiwonder board (e.g., "/dev/ttyUSB0")
        baudrate: Serial baudrate (default: 9600)
        servos: Mapping from joint name to ServoConfig
    """

    port: str = "/dev/ttyUSB0"
    baudrate: int = 9600
    servos: Dict[str, ServoConfig] = field(default_factory=dict)

    # HTD-45H servo constants
    SERVO_MIN: int = 0
    SERVO_MAX: int = 1000
    SERVO_CENTER: int = 500  # Our zero point (physical 120 degrees)
    DEG_TO_SERVO: float = 1000.0 / 240.0  # ~4.1667 servo units per degree

    def get_servo(self, joint_name: str) -> ServoConfig:
        """Get servo config for a joint, raising error if not configured."""
        if joint_name not in self.servos:
            raise KeyError(
                f"Servo not configured for joint '{joint_name}'. "
                f"Available joints: {list(self.servos.keys())}"
            )
        return self.servos[joint_name]

    def get_servo_id(self, joint_name: str) -> int:
        """Get servo ID for a joint, raising error if not configured."""
        return self.get_servo(joint_name).id

    def get_offset(self, joint_name: str) -> int:
        """Get calibration offset for a joint (default: 0)."""
        if joint_name not in self.servos:
            return 0
        return self.servos[joint_name].offset

    @property
    def joint_names(self) -> List[str]:
        """Get list of joint names in order."""
        return list(self.servos.keys())

    def _policy_action_to_servo_pos(self, action: float, servo: ServoConfig) -> int:
        """Convert single policy action [-1, 1] to servo position [0, 1000].

        Conversion chain:
            policy_action [-1, 1]
            → corrected = action * mirror_sign
            → ctrl_rad = corrected * ctrl_span + ctrl_center
            → degrees = ctrl_rad * (180/π)
            → servo_pos = 500 + direction * degrees * DEG_TO_SERVO + offset
        """
        action_clipped = max(-1.0, min(1.0, action))
        corrected = action_clipped * servo.mirror_sign
        ctrl_rad = corrected * servo.ctrl_span + servo.ctrl_center
        degrees = math.degrees(ctrl_rad)
        servo_pos = self.SERVO_CENTER + servo.direction * degrees * self.DEG_TO_SERVO + servo.offset
        return max(self.SERVO_MIN, min(self.SERVO_MAX, int(round(servo_pos))))

    def _servo_pos_to_policy_action(self, servo_pos: int, servo: ServoConfig) -> float:
        """Convert servo position [0, 1000] to policy action [-1, 1].

        Inverse of _policy_action_to_servo_pos.
        """
        degrees = (servo_pos - servo.offset - self.SERVO_CENTER) / self.DEG_TO_SERVO
        ctrl_rad = math.radians(degrees) / servo.direction
        corrected = (ctrl_rad - servo.ctrl_center) / servo.ctrl_span
        action = corrected / servo.mirror_sign
        return max(-1.0, min(1.0, action))

    def policy_action_to_servo_cmd(
        self, actions: List[float]
    ) -> List[Tuple[int, int]]:
        """Convert policy actions to servo commands.

        Args:
            actions: List of policy actions in joint order (from robot_config.yaml)

        Returns:
            List of (servo_id, servo_position) tuples
        """
        joint_names = self.joint_names
        if len(actions) != len(joint_names):
            raise ValueError(
                f"Expected {len(joint_names)} actions, got {len(actions)}"
            )

        commands = []
        for i, joint_name in enumerate(joint_names):
            servo = self.servos[joint_name]
            servo_pos = self._policy_action_to_servo_pos(actions[i], servo)
            commands.append((servo.id, servo_pos))
        return commands

    def servo_pos_to_policy_action(
        self, positions: List[Tuple[int, int]]
    ) -> List[float]:
        """Convert servo positions to policy actions.

        Args:
            positions: List of (servo_id, servo_position) tuples from read_positions()

        Returns:
            List of policy actions in joint order
        """
        # Build servo_id -> position mapping
        pos_by_id = {servo_id: pos for servo_id, pos in positions}

        # Convert in joint order
        actions = []
        for joint_name in self.joint_names:
            servo = self.servos[joint_name]
            if servo.id not in pos_by_id:
                raise ValueError(f"No position for servo ID {servo.id} ({joint_name})")
            servo_pos = pos_by_id[servo.id]
            action = self._servo_pos_to_policy_action(servo_pos, servo)
            actions.append(action)
        return actions


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
class WrRuntimeConfig:
    """Complete runtime configuration for WildRobot hardware deployment.

    This class provides a structured way to load and access all configuration
    needed to run a trained policy on the physical robot.

    Attributes:
        mjcf_path: Path to MuJoCo XML model file
        policy_onnx_path: Path to exported ONNX policy file
        control: Control loop settings (hz, action_scale, velocity_cmd)
        hiwonder_controller: Hiwonder servo controller settings
        bno085: BNO085 IMU settings
        foot_switches: Foot switch GPIO pin settings

    Example:
        config = WrRuntimeConfig.load("config.json")

        # Access paths
        model_path = config.mjcf_path
        policy_path = config.policy_onnx_path

        # Access control settings
        dt = config.control.dt  # 0.02 for 50 Hz

        # Access hardware settings
        servo_id = config.hiwonder_controller.get_servo_id("left_hip_pitch")
        imu_addr = config.bno085.i2c_address
        toe_pin = config.foot_switches.left_toe
    """

    mjcf_path: str
    policy_onnx_path: str
    control: ControlConfig
    hiwonder_controller: HiwonderControllerConfig
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
    def load(
        cls,
        config_path: Optional[str | Path] = None,
        robot_config_path: Optional[str | Path] = None,
    ) -> WrRuntimeConfig:
        """Load runtime configuration from JSON file.

        Args:
            config_path: Path to JSON config file. If None, searches for:
                1. ~/.wildrobot/config.json
                2. ~/wildrobot_config.json (legacy)
            robot_config_path: Path to robot_config.yaml for joint specs.
                If None, uses assets/robot_config.yaml relative to config_path.

        Returns:
            WrRuntimeConfig instance

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

        # Resolve robot_config_path
        if robot_config_path is None:
            # Try relative to config dir, then default path
            robot_config_path = config_dir / ".." / "assets" / "robot_config.yaml"
            if not robot_config_path.exists():
                robot_config_path = DEFAULT_ROBOT_CONFIG_PATH
        robot_config_path = Path(robot_config_path)

        # Load robot config for joint specs
        joint_specs = {}
        if robot_config_path.exists():
            with open(robot_config_path, "r") as f:
                robot_config = yaml.safe_load(f)
            for joint in robot_config.get("actuated_joint_specs", []):
                name = joint.get("name")
                if name:
                    joint_specs[name] = {
                        "rad_range": tuple(joint.get("range", [0.0, 0.0])),
                        "max_velocity": float(joint.get("max_velocity", 10.0)),
                        "mirror_sign": float(joint.get("mirror_sign", 1.0)),
                    }

        # Parse nested configs
        control = cls._parse_control_config(data)
        hiwonder = cls._parse_hiwonder_config(data, joint_specs)
        bno085 = cls._parse_bno085_config(data)
        foot_switches = cls._parse_foot_switch_config(data)

        return cls(
            mjcf_path=data["mjcf_path"],
            policy_onnx_path=data["policy_onnx_path"],
            control=control,
            hiwonder_controller=hiwonder,
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
    def _parse_hiwonder_config(
        data: dict, joint_specs: Dict[str, dict]
    ) -> HiwonderControllerConfig:
        """Parse Hiwonder controller configuration from JSON data.

        Args:
            data: JSON config data
            joint_specs: Joint specs from robot_config.yaml with rad_range and max_velocity
        """
        hiw = data.get("Hiwonder_controller", {})

        # Parse servos - each servo has "id", "offset", and optional "direction" fields
        # Merge with joint_specs for rad_range, max_velocity, mirror_sign
        servos = {}
        for joint_name, servo_data in hiw.get("servos", {}).items():
            joint_spec = joint_specs.get(joint_name, {})
            direction = int(servo_data.get("direction", 1))
            if direction not in (-1, 1):
                raise ValueError(
                    f"Invalid direction for servo '{joint_name}': {direction} (expected +1 or -1)"
                )
            servos[str(joint_name)] = ServoConfig(
                id=int(servo_data["id"]),
                offset=int(servo_data.get("offset", 0)),
                direction=direction,
                rad_range=joint_spec.get("rad_range", (0.0, 0.0)),
                max_velocity=joint_spec.get("max_velocity", 10.0),
                mirror_sign=joint_spec.get("mirror_sign", 1.0),
            )

        return HiwonderControllerConfig(
            port=hiw.get("port", "/dev/ttyUSB0"),
            baudrate=int(hiw.get("baudrate", 9600)),
            servos=servos,
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
        # Build servos dict in new format
        servos_dict = {}
        for joint_name, servo in self.hiwonder_controller.servos.items():
            servos_dict[joint_name] = {
                "id": servo.id,
                "offset": servo.offset,
                "direction": servo.direction,
            }

        return {
            "mjcf_path": self.mjcf_path,
            "policy_onnx_path": self.policy_onnx_path,
            "control_hz": self.control.hz,
            "action_scale_rad": self.control.action_scale_rad,
            "velocity_cmd": self.control.velocity_cmd,
            "Hiwonder_controller": {
                "port": self.hiwonder_controller.port,
                "baudrate": self.hiwonder_controller.baudrate,
                "servos": servos_dict,
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
RuntimeConfig = WrRuntimeConfig
WildRobotRuntimeConfig = WrRuntimeConfig  # Legacy alias


def load_config(config_path: Optional[str] = None) -> WrRuntimeConfig:
    """Load runtime configuration (legacy function).

    Deprecated: Use WrRuntimeConfig.load() instead.

    Args:
        config_path: Path to config file (optional)

    Returns:
        WrRuntimeConfig instance
    """
    return WrRuntimeConfig.load(config_path)
