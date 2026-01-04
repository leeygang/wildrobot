from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from policy_contract.spec import PolicySpec, validate_runtime_compat

from ..config import WildRobotRuntimeConfig
from ..utils.mjcf import MjcfModelInfo


@dataclass(frozen=True)
class RuntimeInterfaceInfo:
    obs_dim: int
    action_dim: int
    actuator_names: List[str]


def validate_runtime_interface(
    *,
    cfg: WildRobotRuntimeConfig,
    mjcf_info: MjcfModelInfo,
    spec: PolicySpec,
    onnx_obs_dim: Optional[int],
    onnx_action_dim: Optional[int],
) -> RuntimeInterfaceInfo:
    """Fail-fast runtime validator for policy contract + hardware config."""
    validate_runtime_compat(
        spec=spec,
        mjcf_actuator_names=mjcf_info.actuator_names,
        onnx_obs_dim=onnx_obs_dim,
        onnx_action_dim=onnx_action_dim,
    )

    missing_servo_ids = [
        name for name in spec.robot.actuator_names if name not in (cfg.hiwonder.servo_ids or {})
    ]
    if missing_servo_ids:
        raise ValueError(
            "Runtime config missing servo_ids for actuators: "
            + ", ".join(missing_servo_ids)
            + ". Update your runtime JSON under hiwonder.servo_ids."
        )

    return RuntimeInterfaceInfo(
        obs_dim=spec.model.obs_dim,
        action_dim=spec.model.action_dim,
        actuator_names=list(spec.robot.actuator_names),
    )
