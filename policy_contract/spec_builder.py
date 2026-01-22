from __future__ import annotations

from typing import Any, Dict, List, Optional

from policy_contract.spec import (
    ActionSpec,
    JointSpec,
    ModelSpec,
    ObservationSpec,
    ObsFieldSpec,
    PolicySpec,
    RobotSpec,
    validate_spec,
)


def build_policy_spec(
    *,
    robot_name: str,
    actuated_joint_specs: List[Dict[str, Any]],
    action_filter_alpha: float,
    home_ctrl_rad: Optional[List[float]] = None,
    provenance: Optional[Dict[str, Any]] = None,
    contract_name: str = "wildrobot_policy",
    contract_version: str = "1.0.0",
    spec_version: int = 1,
    layout_id: str = "wr_obs_v1",
    model_input_name: str = "observation",
    model_output_name: str = "action",
) -> PolicySpec:
    """Build a PolicySpec from data available in training/export pipelines.

    This function is intentionally framework-agnostic (no JAX/MuJoCo deps) so it can
    be used consistently by:
      - training startup (to define the actor contract)
      - export (to write policy_spec.json)
      - eval tooling (to ensure parity with runtime)
    """
    if not isinstance(actuated_joint_specs, list) or not actuated_joint_specs:
        raise ValueError("actuated_joint_specs must be a non-empty list[dict]")

    joints: Dict[str, JointSpec] = {}
    actuator_names: List[str] = []
    for item in actuated_joint_specs:
        if not isinstance(item, dict):
            raise ValueError(f"Invalid joint spec entry: {item!r}")
        name = str(item.get("name"))
        if not name:
            raise ValueError(f"Joint spec missing name: {item!r}")
        rng = item.get("range") or [0.0, 0.0]
        if not isinstance(rng, list) or len(rng) != 2:
            raise ValueError(f"Invalid range for joint '{name}': {rng!r}")
        actuator_names.append(name)
        joints[name] = JointSpec(
            range_min_rad=float(rng[0]),
            range_max_rad=float(rng[1]),
            mirror_sign=float(item.get("mirror_sign", 1.0)),
            max_velocity_rad_s=float(item.get("max_velocity", 10.0)),
        )

    action_dim = len(actuator_names)
    layout = _build_obs_layout(action_dim=action_dim, layout_id=layout_id)

    alpha = float(action_filter_alpha)
    postprocess_id = "lowpass_v1" if alpha > 0.0 else "none"
    postprocess_params: Dict[str, Any] = {}
    if postprocess_id == "lowpass_v1":
        postprocess_params["alpha"] = alpha

    spec = PolicySpec(
        contract_name=contract_name,
        contract_version=contract_version,
        spec_version=int(spec_version),
        model=ModelSpec(
            format="onnx",
            input_name=model_input_name,
            output_name=model_output_name,
            dtype="float32",
            obs_dim=sum(field.size for field in layout),
            action_dim=action_dim,
        ),
        robot=RobotSpec(
            robot_name=str(robot_name),
            actuator_names=actuator_names,
            joints=joints,
            home_ctrl_rad=list(home_ctrl_rad) if home_ctrl_rad is not None else None,
        ),
        observation=ObservationSpec(
            dtype="float32",
            layout_id=layout_id,
            layout=layout,
        ),
        action=ActionSpec(
            dtype="float32",
            bounds={"min": -1.0, "max": 1.0},
            postprocess_id=postprocess_id,
            postprocess_params=postprocess_params,
            mapping_id="pos_target_rad_v1",
        ),
        provenance=provenance,
    )
    validate_spec(spec)
    return spec


def _build_obs_layout(*, action_dim: int, layout_id: str) -> List[ObsFieldSpec]:
    if layout_id != "wr_obs_v1":
        raise ValueError(f"Unsupported layout_id: {layout_id!r}")
    return [
        ObsFieldSpec(name="gravity_local", size=3, frame="local", units="unit_vector"),
        ObsFieldSpec(name="angvel_heading_local", size=3, frame="heading_local", units="rad_s"),
        ObsFieldSpec(name="joint_pos_normalized", size=action_dim, units="normalized_-1_1"),
        ObsFieldSpec(name="joint_vel_normalized", size=action_dim, units="normalized_-1_1"),
        ObsFieldSpec(name="foot_switches", size=4, units="bool_as_float"),
        ObsFieldSpec(name="prev_action", size=action_dim, units="normalized_-1_1"),
        ObsFieldSpec(name="velocity_cmd", size=1, units="m_s"),
        ObsFieldSpec(name="padding", size=1, units="unused"),
    ]

