"""WildRobot runtime configuration package.

Provides structured configuration classes for hardware deployment.
"""

from runtime.configs.config import (
    BNO085Config,
    ControlConfig,
    FootSwitchConfig,
    HiwonderControllerConfig,
    ServoConfig,
    load_config,
    RuntimeConfig,
    WildRobotRuntimeConfig,
    WrRuntimeConfig,
)

__all__ = [
    "WrRuntimeConfig",
    "WildRobotRuntimeConfig",  # Legacy alias
    "RuntimeConfig",
    "ControlConfig",
    "HiwonderControllerConfig",
    "ServoConfig",
    "BNO085Config",
    "FootSwitchConfig",
    "load_config",
]
