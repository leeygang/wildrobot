"""Shared runtime/training logging contracts for locomotion."""

from .locomotion_log_schema import (
    LocomotionLogRecord,
    RobotReplayState,
    required_replay_log_fields,
)

__all__ = [
    "LocomotionLogRecord",
    "RobotReplayState",
    "required_replay_log_fields",
]
