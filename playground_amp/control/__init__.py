"""Control Abstraction Layer (CAL) module.

This module provides the Control Abstraction Layer for MuJoCo control,
decoupling high-level flows (training, testing, sim2real) from MuJoCo primitives.

Key Features:
- Normalized action space [-1, 1] at policy level
- Automatic joint symmetry handling
- Clear action/ctrl naming convention
- Sim2Real ready with ControlCommand trajectory support

Usage:
    from playground_amp.control import (
        ControlAbstractionLayer,
        JointSpec,
        ActuatorSpec,
        ControlCommand,
        ActuatorType,
    )

    # Create CAL from MuJoCo model and config
    cal = ControlAbstractionLayer(mj_model, actuated_joints_config)

    # Policy inference
    policy_action = policy(obs)  # [-1, 1]
    ctrl = cal.policy_action_to_ctrl(policy_action)
    data = data.replace(ctrl=ctrl)

    # State observation
    joint_pos = cal.get_joint_positions(data.qpos, normalize=True)

Version: v0.11.0
"""

from playground_amp.control.cal import ControlAbstractionLayer
from playground_amp.control.specs import ActuatorSpec, ControlCommand, JointSpec
from playground_amp.control.types import ActuatorType

__all__ = [
    # Main class
    "ControlAbstractionLayer",
    # Dataclasses
    "JointSpec",
    "ActuatorSpec",
    "ControlCommand",
    # Enums
    "ActuatorType",
]
