"""WildRobot runtime configuration package.

Provides structured configuration classes for hardware deployment.
"""

from configs.config import (
    DEFAULT_HARDWARE_CONFIG_PATH,
    SERVO_BOARD_NAMES,
    BNO085Config,
    ControlConfig,
    FootSwitchConfig,
    HiwonderControllerConfig,
    RuntimeConfig,
    ServoBoardConfig,
    ServoConfig,
    ServoControllerConfig,
    ServoReadScheduleConfig,
    ServoSpec,
    WildRobotRuntimeConfig,
    WrRuntimeConfig,
    load_config,
    servo_board_name_for_joint,
)

__all__ = [
    "DEFAULT_HARDWARE_CONFIG_PATH",
    "SERVO_BOARD_NAMES",
    "BNO085Config",
    "ControlConfig",
    "FootSwitchConfig",
    "HiwonderControllerConfig",  # Legacy alias
    "RuntimeConfig",
    "ServoBoardConfig",
    "ServoConfig",
    "ServoControllerConfig",
    "ServoReadScheduleConfig",
    "ServoSpec",
    "WildRobotRuntimeConfig",  # Legacy alias
    "WrRuntimeConfig",
    "load_config",
    "servo_board_name_for_joint",
]
