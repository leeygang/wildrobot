"""WildRobot runtime configuration package.

Provides structured configuration classes for hardware deployment.
"""

from runtime.configs.config import (
    BNO085Config,
    ControlConfig,
    FootSwitchConfig,
    ServoControllerConfig,
    ServoConfig,
    ServoSpec,
    load_config,
    RuntimeConfig,
    WildRobotRuntimeConfig,
    WrRuntimeConfig,
    HiwonderControllerConfig,
)

__all__ = [
    "WrRuntimeConfig",
    "WildRobotRuntimeConfig",  # Legacy alias
    "RuntimeConfig",
    "ControlConfig",
    "ServoControllerConfig",
    "HiwonderControllerConfig",  # Legacy alias
    "ServoConfig",
    "ServoSpec",
    "BNO085Config",
    "FootSwitchConfig",
    "load_config",
]
