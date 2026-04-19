"""WildRobot runtime package.

The v0.19.x ``LocomotionLogRecord`` / ``RobotReplayState`` /
``required_replay_log_fields`` log-schema exports were removed at
v0.20.1 along with the v1/v2 walking references.  v0.20.2 will land
a v3-aligned hardware log schema.
"""

from configs import WildRobotRuntimeConfig, load_config

__all__ = [
    "WildRobotRuntimeConfig",
    "load_config",
]
