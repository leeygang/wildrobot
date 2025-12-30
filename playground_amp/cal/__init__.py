"""Control Abstraction Layer (CAL) module.

This module provides the Control Abstraction Layer for MuJoCo control,
decoupling high-level flows (training, testing, sim2real) from MuJoCo primitives.

Key Features:
- Normalized action space [-1, 1] at policy level
- Automatic joint symmetry handling
- Clear action/ctrl naming convention
- Sim2Real ready with ControlCommand trajectory support
- Extended state access (foot, root)

Usage:
    from playground_amp.cal import (
        ControlAbstractionLayer,
        JointSpec,
        ActuatorSpec,
        ControlCommand,
        ActuatorType,
        CoordinateFrame,
        Pose3D,
        Velocity3D,
        FootSpec,
        RootSpec,
    )

    # Create CAL from MuJoCo model and config
    cal = ControlAbstractionLayer(mj_model, actuated_joints_config)

    # Policy inference
    policy_action = policy(obs)  # [-1, 1]
    ctrl = cal.policy_action_to_ctrl(policy_action)
    data = data.replace(ctrl=ctrl)

    # State observation
    joint_pos = cal.get_joint_positions(data.qpos, normalize=True)

    # Extended state (after init_extended_state)
    foot_contacts = cal.get_foot_contacts(data, normalize=True)
    root_pose = cal.get_root_pose(data, frame=CoordinateFrame.WORLD)

Version: v0.11.0
"""

from playground_amp.cal.cal import ControlAbstractionLayer
from playground_amp.cal.specs import (
    ActuatorSpec,
    ControlCommand,
    FootSpec,
    JointSpec,
    Pose3D,
    RootSpec,
    Velocity3D,
)
from playground_amp.cal.types import ActuatorType, CoordinateFrame

__all__ = [
    # Main class
    "ControlAbstractionLayer",
    # Dataclasses - Core
    "JointSpec",
    "ActuatorSpec",
    "ControlCommand",
    # Dataclasses - 3D Types
    "Pose3D",
    "Velocity3D",
    # Dataclasses - Extended State Specs
    "FootSpec",
    "RootSpec",
    # Enums
    "ActuatorType",
    "CoordinateFrame",
]
