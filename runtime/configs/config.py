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
        motor_sign: Hardware motor sign correction (+1 or -1, default: +1)
        motor_center_mujoco_deg: MuJoCo angle (deg) that maps to the servo-unit center
            (servo_unit == 500) when offset==0.
        rad_range: Joint range in radians (min, max) from mujoco_robot_config.json
        max_velocity: Maximum joint velocity in rad/s from mujoco_robot_config.json
        policy_action_sign: +1.0 or -1.0 for action direction correction from mujoco_robot_config.json
    """

    id: int
    offset: int = 0
    motor_sign: float = 1.0
    motor_center_mujoco_deg: float = 0.0
    rad_range: Tuple[float, float] = (0.0, 0.0)
    max_velocity: float = 10.0
    policy_action_sign: float = 1.0

    # Servo conversion constants
    # Hiwonder HTD-45H: 240° range = 4.1887902 rad, units [0, 1000], center at 500
    UNITS_MIN: int = 0
    UNITS_MAX: int = 1000
    UNITS_CENTER: int = 500
    RANGE_RAD: float = 4.1887902047863905  # 240 degrees in radians
    UNITS_PER_RAD: float = 1000.0 / 4.1887902047863905  # ~238.73

    @property
    def offset_unit(self) -> int:
        return int(self.offset)

    @property
    def center_rad(self) -> float:
        return math.radians(float(self.motor_center_mujoco_deg))

    def joint_target_rad_to_elect_unit(self, target_rad: float) -> int:
        """Convert MuJoCo radians to servo units.

        Applies hardware calibration:
        - motor_sign: corrects for servo installation orientation (+1 or -1)
        - offset: corrects for neutral position alignment (in servo units)

        Args:
            target_rad: Joint target in MuJoCo radians (policy_action_sign already applied
                        by policy_contract if coming from policy output)

        Returns:
            Servo position in units [0, 1000]
        """
        delta = float(target_rad) - self.center_rad
        units = self.UNITS_CENTER + self.offset + self.motor_sign * (delta * self.UNITS_PER_RAD)
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
        - motor_sign: corrects for servo installation orientation
        - offset: corrects for neutral position alignment

        Args:
            units: Servo position in units [0, 1000]

        Returns:
            Joint position in MuJoCo radians
        """
        delta_units = float(units) - self.UNITS_CENTER - self.offset
        return self.center_rad + self.motor_sign * (delta_units / self.UNITS_PER_RAD)

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
        """Sign applied to ctrl_rad<->servo conversion (hardware motor_sign only)."""
        return float(self.motor_sign)

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
    offset_unit: int = 0
    motor_sign: float = 1.0
    motor_center_mujoco_deg: float = 0.0

    def to_servo_config(self, joint_spec: Optional[dict] = None) -> ServoConfig:
        joint_spec = joint_spec or {}
        return ServoConfig(
            id=int(self.id),
            offset=int(self.offset_unit),
            motor_sign=float(self.motor_sign),
            motor_center_mujoco_deg=float(self.motor_center_mujoco_deg),
            rad_range=joint_spec.get("rad_range", (0.0, 0.0)),
            max_velocity=joint_spec.get("max_velocity", 10.0),
            policy_action_sign=joint_spec.get("policy_action_sign", 1.0),
        )


@dataclass(frozen=True)
class ServoControllerConfig:
    """Servo controller board configuration (Hiwonder-compatible).

    Attributes:
        type: Controller type (e.g., "hiwonder")
        port: Serial port for the Hiwonder board (e.g., "/dev/ttyUSB0")
        baudrate: Serial baudrate (default: 9600)
        servos: Mapping from joint name to ServoConfig
        default_move_time_ms: Optional default move time for commands
    """

    type: str = "hiwonder"
    port: str = "/dev/ttyUSB0"
    baudrate: int = 9600
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
    def joint_motor_signs(self) -> Dict[str, float]:
        return {k: float(v.motor_sign) for k, v in self.servos.items()}

    @property
    def joint_motor_center_mujoco_deg(self) -> Dict[str, float]:
        return {k: float(v.motor_center_mujoco_deg) for k, v in self.servos.items()}

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
            → corrected = action * policy_action_sign
            → ctrl_rad = corrected * ctrl_span + ctrl_center
            → servo_pos = servo.joint_target_rad_to_elect_unit(ctrl_rad)  (applies motor_center_mujoco_deg, motor_sign, offset)
        """
        action_clipped = max(-1.0, min(1.0, action))
        corrected = action_clipped * servo.policy_action_sign
        ctrl_rad = corrected * servo.ctrl_span + servo.ctrl_center
        servo_pos = servo.joint_target_rad_to_elect_unit(ctrl_rad)
        return max(self.SERVO_MIN, min(self.SERVO_MAX, int(round(servo_pos))))

    def _servo_pos_to_policy_action(
        self, servo_pos: int, servo: ServoConfig, *, joint_name: str
    ) -> float:
        """Convert servo position [0, 1000] to policy action [-1, 1].

        Inverse of _policy_action_to_servo_pos.
        """
        ctrl_rad = servo.servo_elect_units_to_joint_target_rad(servo_pos)
        if abs(float(servo.ctrl_span)) < 1e-9:
            raise ValueError(
                "Cannot invert servo_pos->policy_action with degenerate joint range "
                f"(ctrl_span={float(servo.ctrl_span)}). Joint '{joint_name}' likely missing range in mujoco_robot_config.json."
            )
        corrected = (ctrl_rad - servo.ctrl_center) / servo.ctrl_span
        action = corrected / servo.policy_action_sign
        return max(-1.0, min(1.0, action))

    def policy_action_to_servo_cmd(
        self, actions: List[float]
    ) -> List[Tuple[int, int]]:
        """Convert policy actions to servo commands.

        Args:
            actions: List of policy actions in joint order (from mujoco_robot_config.json)

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
            action = self._servo_pos_to_policy_action(servo_pos, servo, joint_name=joint_name)
            actions.append(action)
        return actions


@dataclass(frozen=True)
class BNO085Config:
    """BNO085 IMU configuration (runtime-facing fields)."""

    i2c_address: int = 0x4A
    upside_down: bool = False
    suppress_debug: bool = True
    axis_map: Optional[list[str]] = None
    i2c_frequency_hz: int = 100_000
    init_retries: int = 3


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
                        "policy_action_sign": float(joint.get("policy_action_sign", 1.0)),
                    }

        # Parse nested configs
        control = cls._parse_control_config(data)
        servo_controller = cls._parse_servo_controller_config(data, joint_specs)
        bno085 = cls._parse_bno085_config(data)
        foot_switches = cls._parse_foot_switch_config(data)

        return cls(
            mjcf_path=data["mjcf_path"],
            policy_onnx_path=data["policy_onnx_path"],
            control=control,
            servo_controller=servo_controller,
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
    def _parse_servo_controller_config(
        data: dict, joint_specs: Dict[str, dict]
    ) -> ServoControllerConfig:
        """Parse servo controller configuration from the canonical 'servo_controller' block."""

        def _parse_block(block: dict, key_path: str) -> ServoControllerConfig:
            servos: Dict[str, ServoConfig] = {}
            for joint_name, servo_data in block.get("servos", {}).items():
                joint_spec = joint_specs.get(joint_name, {})
                motor_sign_val = float(servo_data.get("motor_sign", 1.0))
                WrRuntimeConfig._validate_motor_signs({joint_name: motor_sign_val})

                motor_center_mujoco_deg = float(servo_data.get("motor_center_mujoco_deg", 0.0))
                rad_range = joint_spec.get("rad_range", (0.0, 0.0))
                WrRuntimeConfig._validate_motor_center_mujoco_deg(
                    motor_center_mujoco_deg=motor_center_mujoco_deg,
                    rad_range=rad_range,
                    joint_name=joint_name,
                    key_path=key_path,
                )

                offset_unit = servo_data.get("offset_unit")
                legacy_offset = servo_data.get("offset")
                legacy_offset_rad = servo_data.get("offset_rad")
                if offset_unit is not None:
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
                        f"WARNING: {key_path}.{joint_name}.offset_rad found; treating value as servo units (naming bug). Please rename to offset_unit."
                    )
                else:
                    chosen_offset = 0

                servos[str(joint_name)] = ServoConfig(
                    id=int(servo_data["id"]),
                    offset=int(chosen_offset),
                    motor_sign=motor_sign_val,
                    motor_center_mujoco_deg=motor_center_mujoco_deg,
                    rad_range=rad_range,
                    max_velocity=joint_spec.get("max_velocity", 10.0),
                    policy_action_sign=joint_spec.get("policy_action_sign", 1.0),
                )

            return ServoControllerConfig(
                type=str(block.get("type", "hiwonder")),
                port=str(block.get("port", "/dev/ttyUSB0")),
                baudrate=int(block.get("baudrate", 9600)),
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
    def _validate_motor_signs(motor_signs: Dict[str, float]) -> None:
        invalid = {k: v for k, v in motor_signs.items() if abs(abs(v) - 1.0) > 1e-3}
        if invalid:
            raise ValueError(
                "servo_controller.servos.motor_sign must be +/-1.0 (tolerance 1e-3); invalid entries: "
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
                f"{key_path}.{joint_name}.motor_center_mujoco_deg={motor_center_mujoco_deg} maps to {center_rad:.6f} rad, "
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
                raise ValueError("bno085.axis_map must be a list of 3 strings like ['+X','-Y','+Z']")
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
            axis_map = normalized

        return BNO085Config(
            i2c_address=i2c_addr,
            upside_down=bool(bno.get("upside_down", False)),
            suppress_debug=bool(bno.get("suppress_debug", True)),
            axis_map=axis_map,
            i2c_frequency_hz=int(bno.get("i2c_frequency_hz", 100_000)),
            init_retries=int(bno.get("init_retries", 3)),
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
        servos_dict = {
            joint_name: {
                "id": servo.id,
                "offset_unit": servo.offset_unit,
                "motor_sign": servo.motor_sign,
                "motor_center_mujoco_deg": servo.motor_center_mujoco_deg,
            }
            for joint_name, servo in self.servo_controller.servos.items()
        }

        out = {
            "mjcf_path": self.mjcf_path,
            "policy_onnx_path": self.policy_onnx_path,
            "control_hz": self.control.hz,
            "action_scale_rad": self.control.action_scale_rad,
            "velocity_cmd": self.control.velocity_cmd,
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
                **({"axis_map": self.bno085.axis_map} if self.bno085.axis_map is not None else {}),
            },
            "foot_switches": self.foot_switches.get_all_pins(),
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
