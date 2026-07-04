"""Runtime configuration for WildRobot hardware deployment.

This module provides a structured configuration class for running the WildRobot
policy on real hardware. Configuration is loaded from a JSON file that specifies:
- Policy and model paths
- Control loop parameters
- Servo controller settings (port, servo IDs, joint offsets)
- BNO085 IMU settings
- Foot switch GPIO pins

Usage:
    from configs import WrRuntimeConfig

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

# Default robot_config path (joint ranges, mirror signs, etc.)
# Current assets layout is v2.
DEFAULT_ROBOT_CONFIG_PATH = Path("assets/v2/mujoco_robot_config.json")


# =============================================================================
# Nested Configuration Dataclasses
# =============================================================================


@dataclass(frozen=True)
class ControlConfig:
    """Control loop configuration.

    Attributes:
        hz: Control loop frequency in Hz (default: 50)
        action_scale_rad: Scale factor for policy actions in radians (default: 0.35)
        velocity_cmd: Initial forward-speed command for the policy in m/s (default: 0.0)
        yaw_rate_cmd: Initial yaw-rate command for the policy in rad/s (default: 0.0)
    """

    hz: float = 50.0
    action_scale_rad: float = 0.35
    velocity_cmd: float = 0.0
    yaw_rate_cmd: float = 0.0

    @property
    def dt(self) -> float:
        """Control loop period in seconds."""
        return 1.0 / self.hz


@dataclass(frozen=True)
class ServoConfig:
    """Configuration for a single servo.

    Attributes:
        id: Servo ID (1-254)
        servo_offset_unit: Calibration offset in servo units (default: 0)
        motor_unit_direction: Hardware motor direction correction (+1 or -1, default: +1)
        joint_angle_at_zero_unit_deg: MuJoCo angle (deg) that maps to the servo-unit center
            (servo_unit == 500) when offset==0.
        rad_range: Joint range in radians (min, max) from mujoco_robot_config.json
        max_velocity: Maximum joint velocity in rad/s from mujoco_robot_config.json
    """

    id: int
    servo_offset_unit: int = 0
    motor_unit_direction: float = 1.0
    joint_angle_at_zero_unit_deg: float = 0.0
    rad_range: Tuple[float, float] = (0.0, 0.0)
    max_velocity: float = 10.0

    # Servo conversion constants
    # Hiwonder HTD-45H: 240° range = 4.1887902 rad, units [0, 1000], center at 500
    UNITS_MIN: int = 0
    UNITS_MAX: int = 1000
    UNITS_CENTER: int = 500
    RANGE_RAD: float = 4.1887902047863905  # 240 degrees in radians
    UNITS_PER_RAD: float = 1000.0 / 4.1887902047863905  # ~238.73

    @property
    def offset_unit(self) -> int:
        return int(self.servo_offset_unit)

    @property
    def offset(self) -> int:
        return int(self.servo_offset_unit)

    @property
    def servo_offset(self) -> int:
        return int(self.servo_offset_unit)

    @property
    def motor_sign(self) -> float:
        return float(self.motor_unit_direction)

    @property
    def servo_unit_direction(self) -> float:
        return float(self.motor_unit_direction)

    @property
    def motor_center_mujoco_deg(self) -> float:
        return float(self.joint_angle_at_zero_unit_deg)

    @property
    def center_rad(self) -> float:
        return math.radians(float(self.joint_angle_at_zero_unit_deg))

    def joint_target_rad_to_elect_unit(self, target_rad: float) -> int:
        """Convert MuJoCo radians to servo units.

        Applies hardware calibration:
        - motor_unit_direction: corrects for servo installation orientation (+1 or -1)
        - offset: corrects for neutral position alignment (in servo units)

        Args:
            target_rad: Joint target in MuJoCo radians

        Returns:
            Servo position in units [0, 1000]
        """
        delta = float(target_rad) - self.center_rad
        units = self.UNITS_CENTER + self.servo_offset_unit + self.motor_unit_direction * (delta * self.UNITS_PER_RAD)
        return int(max(self.UNITS_MIN, min(self.UNITS_MAX, round(units))))

    def joint_target_rad_to_elect_unit_for_calibrate(
        self,
        target_rad: float,
        *,
        motor_sign: int,
        offset: int,
    ) -> int:
        """Convert MuJoCo radians to servo units with overridden calibration values.

        Used during calibration when testing different direction/offset values.

        Args:
            target_rad: Joint target in MuJoCo radians
            motor_sign: Motor sign to use (+1 or -1)
            offset: Offset to use (in servo units)

        Returns:
            Servo position in units [0, 1000]
        """
        delta = float(target_rad) - self.center_rad
        units = self.UNITS_CENTER + offset + motor_sign * (delta * self.UNITS_PER_RAD)
        return int(max(self.UNITS_MIN, min(self.UNITS_MAX, round(units))))

    def servo_elect_units_to_joint_target_rad(self, units: int) -> float:
        """Convert servo units to MuJoCo radians.

        Inverse of joint_target_rad_to_elect_unit. Applies hardware calibration:
        - motor_unit_direction: corrects for servo installation orientation
        - offset: corrects for neutral position alignment

        Args:
            units: Servo position in units [0, 1000]

        Returns:
            Joint position in MuJoCo radians
        """
        delta_units = float(units) - self.UNITS_CENTER - self.servo_offset_unit
        return self.center_rad + self.motor_unit_direction * (delta_units / self.UNITS_PER_RAD)

    def servo_elect_units_to_joint_target_rad_for_calibrate(
        self,
        units: int,
        *,
        motor_sign: int,
        offset: int,
    ) -> float:
        """Convert servo units to MuJoCo radians with overridden calibration values.

        Used during calibration when testing different direction/offset values.

        Args:
            units: Servo position in units [0, 1000]
            motor_sign: Motor sign to use (+1 or -1)
            offset: Offset to use (in servo units)

        Returns:
            Joint position in MuJoCo radians
        """
        delta_units = float(units) - self.UNITS_CENTER - offset
        return self.center_rad + motor_sign * (delta_units / self.UNITS_PER_RAD)

    @property
    def effective_sign(self) -> float:
        """Sign applied to ctrl_rad<->servo conversion (hardware direction only)."""
        return float(self.motor_unit_direction)

    @property
    def ctrl_center(self) -> float:
        """Center of joint range in radians."""
        return (self.rad_range[0] + self.rad_range[1]) / 2

    @property
    def ctrl_span(self) -> float:
        """Half-span of joint range in radians."""
        return (self.rad_range[1] - self.rad_range[0]) / 2


@dataclass(frozen=True)
class ServoSpec:
    """Minimal servo specification used by canonical configs and legacy tests."""

    id: int
    servo_offset_unit: int = 0
    motor_unit_direction: float = 1.0
    joint_angle_at_zero_unit_deg: float = 0.0

    @property
    def offset_unit(self) -> int:
        return int(self.servo_offset_unit)

    @property
    def motor_sign(self) -> float:
        return float(self.motor_unit_direction)

    @property
    def servo_unit_direction(self) -> float:
        return float(self.motor_unit_direction)

    @property
    def motor_center_mujoco_deg(self) -> float:
        return float(self.joint_angle_at_zero_unit_deg)

    def to_servo_config(self, joint_spec: Optional[dict] = None) -> ServoConfig:
        joint_spec = joint_spec or {}
        return ServoConfig(
            id=int(self.id),
            servo_offset_unit=int(self.servo_offset_unit),
            motor_unit_direction=float(self.motor_unit_direction),
            joint_angle_at_zero_unit_deg=float(self.joint_angle_at_zero_unit_deg),
            rad_range=joint_spec.get("rad_range", (0.0, 0.0)),
            max_velocity=joint_spec.get("max_velocity", 10.0),
        )


@dataclass(frozen=True)
class ServoControllerConfig:
    """Servo controller board configuration (Hiwonder-compatible).

    Attributes:
        type: Controller type (e.g., "hiwonder_ttl_bus")
        port: Serial port for the USB TTL debug board.
        baudrate: Serial baudrate (default: 115200)
        servos: Mapping from joint name to ServoConfig
        default_move_time_ms: Optional default move time for commands
    """

    type: str = "hiwonder_ttl_bus"
    port: str = "/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0"
    baudrate: int = 115200
    servos: Dict[str, ServoConfig] = field(default_factory=dict)
    default_move_time_ms: Optional[int] = None

    # HTD-45H servo constants
    SERVO_MIN: int = 0
    SERVO_MAX: int = 1000
    SERVO_CENTER: int = 500  # Our zero point (physical 120 degrees)
    DEG_TO_SERVO: float = 1000.0 / 240.0  # ~4.1667 servo units per degree

    def __post_init__(self) -> None:
        if not self.servos:
            return
        if all(isinstance(s, ServoConfig) for s in self.servos.values()):
            return

        normalized: Dict[str, ServoConfig] = {}
        for name, servo in self.servos.items():
            if isinstance(servo, ServoConfig):
                normalized[name] = servo
            elif isinstance(servo, ServoSpec):
                normalized[name] = servo.to_servo_config()
            else:
                raise TypeError(
                    f"servo_controller.servos['{name}'] must be ServoConfig or ServoSpec, got {type(servo)!r}"
                )

        object.__setattr__(self, "servos", normalized)

    @property
    def servo_ids(self) -> Dict[str, int]:
        return {k: v.id for k, v in self.servos.items()}

    @property
    def joint_offset_units(self) -> Dict[str, int]:
        return {k: int(v.offset_unit) for k, v in self.servos.items()}

    @property
    def joint_servo_offset_units(self) -> Dict[str, int]:
        return {k: int(v.servo_offset_unit) for k, v in self.servos.items()}

    @property
    def joint_motor_signs(self) -> Dict[str, float]:
        return {k: float(v.motor_unit_direction) for k, v in self.servos.items()}

    @property
    def joint_motor_unit_directions(self) -> Dict[str, float]:
        return {k: float(v.motor_unit_direction) for k, v in self.servos.items()}

    @property
    def joint_servo_unit_directions(self) -> Dict[str, float]:
        return {k: float(v.motor_unit_direction) for k, v in self.servos.items()}

    @property
    def joint_motor_center_mujoco_deg(self) -> Dict[str, float]:
        return {k: float(v.joint_angle_at_zero_unit_deg) for k, v in self.servos.items()}

    @property
    def joint_angle_at_zero_unit_deg(self) -> Dict[str, float]:
        return {k: float(v.joint_angle_at_zero_unit_deg) for k, v in self.servos.items()}

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
        return self.servos[joint_name].servo_offset_unit

    @property
    def joint_names(self) -> List[str]:
        """Get list of joint names in order."""
        return list(self.servos.keys())

@dataclass(frozen=True)
class BNO085Config:
    """BNO085 IMU configuration (runtime-facing fields)."""

    i2c_address: int = 0x4A
    upside_down: bool = False
    suppress_debug: bool = True
    axis_map: Optional[list[str]] = None
    i2c_frequency_hz: int = 100_000
    init_retries: int = 3
    sampling_hz: Optional[int] = None
    enable_rotation_vector: bool = True


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


@dataclass(frozen=True)
class ServoReadScheduleConfig:
    """Optional runtime servo read schedule.

    ``mode="full"`` preserves the old behavior: read all policy servos every
    control step. ``mode="staggered"`` reads an initial full state, then reads
    one configured joint group per step while returning the full cached state.
    """

    mode: str = "full"
    groups: List[List[str]] = field(default_factory=list)
    max_cache_age_s: Dict[str, float] = field(
        default_factory=lambda: {
            "leg": 0.12,
            "arm": 0.25,
            "wrist": 0.50,
            "default": 0.25,
        }
    )

    @property
    def enabled(self) -> bool:
        return self.mode == "staggered"


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
        servo_controller: Servo controller settings
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
    servo_controller: ServoControllerConfig
    servo_read_schedule: ServoReadScheduleConfig
    bno085: BNO085Config
    foot_switches: FootSwitchConfig
    realism_profile_path: Optional[str] = None

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
            robot_config_path: Path to robot_config.(json|yaml) for joint specs.
                If None, uses a best-effort search (prefers assets/v2/mujoco_robot_config.json).

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
        # Priority:
        #   1) explicit argument `robot_config_path`
        #   2) JSON key `robot_config_path` (resolved relative to config file dir)
        #   3) repo-relative assets/v2/mujoco_robot_config.json
        #   4) cwd-relative assets/v2/mujoco_robot_config.json
        if robot_config_path is None:
            json_robot_cfg = data.get("robot_config_path")
            if isinstance(json_robot_cfg, str) and json_robot_cfg.strip():
                robot_config_path = (config_dir / json_robot_cfg).resolve()

        candidate_paths: List[Path] = []
        if robot_config_path is not None:
            candidate_paths.append(Path(robot_config_path))

        repo_root = Path(__file__).resolve().parents[2]
        candidate_paths.extend(
            [
                (repo_root / "assets" / "v2" / "mujoco_robot_config.json"),
                Path(DEFAULT_ROBOT_CONFIG_PATH),
            ]
        )

        resolved_robot_cfg: Optional[Path] = None
        for p in candidate_paths:
            try:
                pp = p.expanduser()
            except Exception:
                pp = p
            if pp.exists():
                resolved_robot_cfg = pp
                break
        robot_config_path = resolved_robot_cfg if resolved_robot_cfg is not None else Path(candidate_paths[0])

        # Load robot config for joint specs
        joint_specs = {}
        if robot_config_path.exists():
            with open(robot_config_path, "r") as f:
                robot_config = yaml.safe_load(f)
            range_unit = str(robot_config.get("joint_range_unit", "rad")).lower()
            if range_unit not in {"rad", "deg"}:
                raise ValueError(
                    "mujoco_robot_config.json 'joint_range_unit' must be 'rad' or 'deg'"
                )
            for joint in robot_config.get("actuated_joint_specs", []):
                name = joint.get("name")
                if name:
                    range_vals = joint.get("range", [0.0, 0.0])
                    if not isinstance(range_vals, list) or len(range_vals) != 2:
                        raise ValueError(f"Invalid range for joint '{name}': {range_vals}")
                    range_min = float(range_vals[0])
                    range_max = float(range_vals[1])
                    if range_unit == "deg":
                        range_min = math.radians(range_min)
                        range_max = math.radians(range_max)
                    joint_specs[name] = {
                        "rad_range": (range_min, range_max),
                        "max_velocity": float(joint.get("max_velocity", 10.0)),
                    }

        # Parse nested configs
        control = cls._parse_control_config(data)
        servo_controller = cls._parse_servo_controller_config(data, joint_specs)
        servo_read_schedule = cls._parse_servo_read_schedule_config(data)
        bno085 = cls._parse_bno085_config(data)
        foot_switches = cls._parse_foot_switch_config(data)

        return cls(
            mjcf_path=data["mjcf_path"],
            policy_onnx_path=data["policy_onnx_path"],
            control=control,
            servo_controller=servo_controller,
            servo_read_schedule=servo_read_schedule,
            bno085=bno085,
            foot_switches=foot_switches,
            realism_profile_path=(
                str(data["realism_profile_path"])
                if "realism_profile_path" in data and data["realism_profile_path"] is not None
                else None
            ),
            _config_dir=config_dir,
        )

    @staticmethod
    def _parse_control_config(data: dict) -> ControlConfig:
        """Parse control configuration from JSON data."""
        return ControlConfig(
            hz=float(data.get("control_hz", 50.0)),
            action_scale_rad=float(data.get("action_scale_rad", 0.35)),
            velocity_cmd=float(data.get("velocity_cmd", 0.0)),
            yaw_rate_cmd=float(data.get("yaw_rate_cmd", 0.0)),
        )

    @staticmethod
    def _parse_servo_controller_config(
        data: dict, joint_specs: Dict[str, dict]
    ) -> ServoControllerConfig:
        """Parse servo controller configuration from the canonical 'servo_controller' block."""

        def _parse_block(block: dict, key_path: str) -> ServoControllerConfig:
            servos: Dict[str, ServoConfig] = {}
            for joint_name, servo_data in block.get("servos", {}).items():
                joint_spec = joint_specs.get(joint_name, {})
                direction_values = {
                    key: float(servo_data[key])
                    for key in ("motor_unit_direction", "servo_unit_direction", "motor_sign")
                    if key in servo_data
                }
                if direction_values:
                    chosen_direction = next(iter(direction_values.values()))
                    for key, value in direction_values.items():
                        if abs(float(value) - float(chosen_direction)) > 1e-9:
                            raise ValueError(
                                f"{key_path}.{joint_name}.{key}={value} conflicts with servo direction value {chosen_direction}"
                            )
                    direction_raw = chosen_direction
                else:
                    direction_raw = 1.0
                motor_unit_direction = float(direction_raw)
                WrRuntimeConfig._validate_motor_signs({joint_name: motor_unit_direction})

                center_deg_raw = servo_data.get(
                    "joint_angle_at_zero_unit_deg",
                    servo_data.get("motor_center_mujoco_deg", 0.0),
                )
                joint_angle_at_zero_unit_deg = float(center_deg_raw)
                rad_range = joint_spec.get("rad_range", (0.0, 0.0))
                WrRuntimeConfig._validate_motor_center_mujoco_deg(
                    motor_center_mujoco_deg=joint_angle_at_zero_unit_deg,
                    rad_range=rad_range,
                    joint_name=joint_name,
                    key_path=key_path,
                )

                servo_offset_unit = servo_data.get("servo_offset_unit")
                offset_unit = servo_data.get("offset_unit")
                legacy_offset = servo_data.get("offset")
                legacy_offset_rad = servo_data.get("offset_rad")
                if servo_offset_unit is not None:
                    chosen_offset = int(servo_offset_unit)
                    if offset_unit is not None and int(offset_unit) != chosen_offset:
                        raise ValueError(
                            f"{key_path}.{joint_name}.servo_offset_unit={chosen_offset} conflicts with offset_unit={offset_unit}"
                        )
                    if legacy_offset is not None and int(legacy_offset) != chosen_offset:
                        raise ValueError(
                            f"{key_path}.{joint_name}.servo_offset_unit={chosen_offset} conflicts with offset={legacy_offset}"
                        )
                    if legacy_offset_rad is not None and int(legacy_offset_rad) != chosen_offset:
                        raise ValueError(
                            f"{key_path}.{joint_name}.servo_offset_unit={chosen_offset} conflicts with offset_rad={legacy_offset_rad}"
                        )
                elif offset_unit is not None:
                    chosen_offset = int(offset_unit)
                    if legacy_offset is not None and int(legacy_offset) != chosen_offset:
                        raise ValueError(
                            f"{key_path}.{joint_name}.offset_unit={chosen_offset} conflicts with offset={legacy_offset}"
                        )
                    if legacy_offset_rad is not None and int(legacy_offset_rad) != chosen_offset:
                        raise ValueError(
                            f"{key_path}.{joint_name}.offset_unit={chosen_offset} conflicts with offset_rad={legacy_offset_rad}"
                        )
                elif legacy_offset is not None:
                    chosen_offset = int(legacy_offset)
                elif legacy_offset_rad is not None:
                    chosen_offset = int(legacy_offset_rad)
                    print(
                        f"WARNING: {key_path}.{joint_name}.offset_rad found; treating value as servo units (naming bug). Please rename to servo_offset_unit."
                    )
                else:
                    chosen_offset = 0

                servos[str(joint_name)] = ServoConfig(
                    id=int(servo_data["id"]),
                    servo_offset_unit=int(chosen_offset),
                    motor_unit_direction=motor_unit_direction,
                    joint_angle_at_zero_unit_deg=joint_angle_at_zero_unit_deg,
                    rad_range=rad_range,
                    max_velocity=joint_spec.get("max_velocity", 10.0),
                )

            return ServoControllerConfig(
                type=str(block.get("type", "hiwonder_ttl_bus")),
                port=str(block.get("port", "/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0")),
                baudrate=int(block.get("baudrate", 115200)),
                servos=servos,
                default_move_time_ms=int(block["default_move_time_ms"]) if "default_move_time_ms" in block else None,
            )

        canonical_block = data.get("servo_controller")
        legacy_keys = [k for k in data.keys() if str(k).lower() in {"hiwonder", "hiwonder_controller"}]

        if legacy_keys and canonical_block is not None:
            raise ValueError(
                "Conflicting servo controller definitions: found 'servo_controller' and legacy blocks "
                + ", ".join(repr(k) for k in legacy_keys)
                + ". Remove legacy blocks and use only 'servo_controller'."
            )

        if canonical_block is None:
            if legacy_keys:
                raise KeyError(
                    "Servo controller config missing 'servo_controller'. Legacy blocks "
                    + ", ".join(repr(k) for k in legacy_keys)
                    + " are no longer supported; use 'servo_controller' with fields {type, port, baudrate, servos}."
                )
            raise KeyError(
                "Servo controller config missing. Provide 'servo_controller' with fields {type, port, baudrate, servos}."
            )

        return _parse_block(canonical_block, "servo_controller.servos")

    @staticmethod
    def _parse_servo_read_schedule_config(data: dict) -> ServoReadScheduleConfig:
        raw = data.get("servo_read_schedule")
        if raw is None:
            return ServoReadScheduleConfig()
        if not isinstance(raw, dict):
            raise ValueError("servo_read_schedule must be an object")

        groups_raw = raw.get("groups", [])
        if groups_raw is None:
            groups_raw = []
        if not isinstance(groups_raw, list):
            raise ValueError("servo_read_schedule.groups must be a list of joint-name lists")

        groups: List[List[str]] = []
        for i, group in enumerate(groups_raw):
            if not isinstance(group, list) or not group:
                raise ValueError(
                    f"servo_read_schedule.groups[{i}] must be a non-empty list"
                )
            names = [str(name) for name in group]
            if any(not name.strip() for name in names):
                raise ValueError(
                    f"servo_read_schedule.groups[{i}] contains an empty joint name"
                )
            groups.append(names)

        mode = str(raw.get("mode", "staggered" if groups else "full")).lower()
        if mode not in {"full", "staggered"}:
            raise ValueError("servo_read_schedule.mode must be 'full' or 'staggered'")
        if mode == "staggered" and not groups:
            raise ValueError("servo_read_schedule.groups is required for staggered mode")

        default_limits = ServoReadScheduleConfig().max_cache_age_s
        limits_raw = raw.get("max_cache_age_s", {})
        if limits_raw is None:
            limits_raw = {}
        if not isinstance(limits_raw, dict):
            raise ValueError("servo_read_schedule.max_cache_age_s must be an object")
        limits = dict(default_limits)
        for key, value in limits_raw.items():
            value_f = float(value)
            if value_f <= 0.0:
                raise ValueError(
                    f"servo_read_schedule.max_cache_age_s.{key} must be positive"
                )
            limits[str(key)] = value_f

        return ServoReadScheduleConfig(mode=mode, groups=groups, max_cache_age_s=limits)

    @staticmethod
    def _validate_motor_signs(motor_signs: Dict[str, float]) -> None:
        invalid = {k: v for k, v in motor_signs.items() if abs(abs(v) - 1.0) > 1e-3}
        if invalid:
            raise ValueError(
                "servo_controller.servos.motor_unit_direction must be +/-1.0 (tolerance 1e-3); invalid entries: "
                + ", ".join(f"{k}={v}" for k, v in invalid.items())
            )

    @staticmethod
    def _validate_motor_center_mujoco_deg(
        *,
        motor_center_mujoco_deg: float,
        rad_range: Tuple[float, float],
        joint_name: str,
        key_path: str,
    ) -> None:
        range_min, range_max = float(rad_range[0]), float(rad_range[1])
        if abs(range_min) < 1e-12 and abs(range_max) < 1e-12:
            return
        if range_min > range_max:
            range_min, range_max = range_max, range_min
        center_rad = math.radians(float(motor_center_mujoco_deg))
        if not (range_min - 1e-9 <= center_rad <= range_max + 1e-9):
            print(
                "WARNING: "
                f"{key_path}.{joint_name}.joint_angle_at_zero_unit_deg={motor_center_mujoco_deg} maps to {center_rad:.6f} rad, "
                f"outside joint range [{range_min:.6f}, {range_max:.6f}] rad",
                flush=True,
            )

    @staticmethod
    def _parse_bno085_config(data: dict) -> BNO085Config:
        """Parse BNO085 configuration from JSON data."""
        bno = data.get("bno085", {})

        # Parse I2C address (may be hex string like "0x4A" or int)
        i2c_addr = bno.get("i2c_address", 0x4A)
        if isinstance(i2c_addr, str):
            i2c_addr = int(i2c_addr, 0)  # Handles "0x4A" format

        axis_map_raw = bno.get("axis_map")
        axis_map = None
        if axis_map_raw is not None:
            if not isinstance(axis_map_raw, list) or len(axis_map_raw) != 3:
                raise ValueError("bno085.axis_map must be a list of 3 strings like ['+X','-Y','-Z']")
            normalized: list[str] = []
            allowed = {"X", "Y", "Z"}
            seen: set[str] = set()
            for entry in axis_map_raw:
                s = str(entry).strip().upper()
                if len(s) != 2 or s[0] not in {"+", "-"} or s[1] not in allowed:
                    raise ValueError(
                        f"Invalid bno085.axis_map entry: {entry!r} (expected entries like '+X', '-Y', '+Z')"
                    )
                if s[1] in seen:
                    raise ValueError(f"Duplicate axis in bno085.axis_map: {axis_map_raw}")
                seen.add(s[1])
                normalized.append(s)
            det = WrRuntimeConfig._axis_map_det(normalized)
            if det < 0:
                raise ValueError(
                    "bno085.axis_map must be right-handed with determinant +1 "
                    f"(got det=-1 for {axis_map_raw}); flip two axes, not one"
                )
            axis_map = normalized

        return BNO085Config(
            i2c_address=i2c_addr,
            upside_down=bool(bno.get("upside_down", False)),
            suppress_debug=bool(bno.get("suppress_debug", True)),
            axis_map=axis_map,
            i2c_frequency_hz=int(bno.get("i2c_frequency_hz", 100_000)),
            init_retries=int(bno.get("init_retries", 3)),
            sampling_hz=(
                int(bno["sampling_hz"]) if bno.get("sampling_hz") is not None else None
            ),
            enable_rotation_vector=bool(bno.get("enable_rotation_vector", True)),
        )

    @staticmethod
    def _axis_map_det(axis_map: list[str]) -> int:
        axis_index = {"X": 0, "Y": 1, "Z": 2}
        perm = [axis_index[entry[1]] for entry in axis_map]
        inversions = 0
        for i in range(3):
            for j in range(i + 1, 3):
                if perm[i] > perm[j]:
                    inversions += 1
        sign = -1 if inversions % 2 else 1
        for entry in axis_map:
            if entry[0] == "-":
                sign *= -1
        return sign

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
        servos_dict = {
            joint_name: {
                "id": servo.id,
                "servo_offset_unit": servo.servo_offset_unit,
                "motor_unit_direction": servo.motor_unit_direction,
                "joint_angle_at_zero_unit_deg": servo.joint_angle_at_zero_unit_deg,
            }
            for joint_name, servo in self.servo_controller.servos.items()
        }

        out = {
            "mjcf_path": self.mjcf_path,
            "policy_onnx_path": self.policy_onnx_path,
            "control_hz": self.control.hz,
            "action_scale_rad": self.control.action_scale_rad,
            "velocity_cmd": self.control.velocity_cmd,
            "yaw_rate_cmd": self.control.yaw_rate_cmd,
            "servo_controller": {
                "type": self.servo_controller.type,
                "port": self.servo_controller.port,
                "baudrate": self.servo_controller.baudrate,
                "servos": servos_dict,
                **(
                    {"default_move_time_ms": self.servo_controller.default_move_time_ms}
                    if self.servo_controller.default_move_time_ms is not None
                    else {}
                ),
            },
            "bno085": {
                "i2c_address": hex(self.bno085.i2c_address),
                "upside_down": self.bno085.upside_down,
                "suppress_debug": self.bno085.suppress_debug,
                "i2c_frequency_hz": int(self.bno085.i2c_frequency_hz),
                "init_retries": int(self.bno085.init_retries),
                **(
                    {"sampling_hz": int(self.bno085.sampling_hz)}
                    if self.bno085.sampling_hz is not None
                    else {}
                ),
                "enable_rotation_vector": bool(self.bno085.enable_rotation_vector),
                **({"axis_map": self.bno085.axis_map} if self.bno085.axis_map is not None else {}),
            },
            "foot_switches": self.foot_switches.get_all_pins(),
            **(
                {
                    "servo_read_schedule": {
                        "mode": self.servo_read_schedule.mode,
                        "groups": self.servo_read_schedule.groups,
                        "max_cache_age_s": self.servo_read_schedule.max_cache_age_s,
                    }
                }
                if self.servo_read_schedule.enabled or self.servo_read_schedule.groups
                else {}
            ),
            **(
                {"realism_profile_path": self.realism_profile_path}
                if self.realism_profile_path is not None
                else {}
            ),
        }
        return out

    def save(self, config_path: str | Path) -> None:
        """Save configuration to JSON file.

        Args:
            config_path: Path to save the config file
        """
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @property
    def hiwonder_controller(self) -> ServoControllerConfig:
        """Legacy attribute alias for servo_controller."""
        return self.servo_controller


# =============================================================================
# Legacy Compatibility
# =============================================================================

# Legacy alias (previous class name)
HiwonderControllerConfig = ServoControllerConfig

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
