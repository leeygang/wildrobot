from __future__ import annotations

from typing import Dict

from policy_contract.spec import (
    ActionSpec,
    JointSpec,
    ModelSpec,
    ObservationSpec,
    ObsFieldSpec,
    PolicySpec,
    RobotSpec,
)


def build_policy_spec(config, robot_config) -> PolicySpec:
    action_dim = robot_config.action_dim

    layout = [
        ObsFieldSpec(name="gravity_local", size=3, frame="local", units="unit_vector"),
        ObsFieldSpec(name="angvel_heading_local", size=3, frame="heading_local", units="rad_s"),
        ObsFieldSpec(name="joint_pos_normalized", size=action_dim, units="normalized_-1_1"),
        ObsFieldSpec(name="joint_vel_normalized", size=action_dim, units="normalized_-1_1"),
        ObsFieldSpec(name="foot_switches", size=4, units="bool_as_float"),
        ObsFieldSpec(name="prev_action", size=action_dim, units="normalized_-1_1"),
        ObsFieldSpec(name="velocity_cmd", size=1, units="m_s"),
        ObsFieldSpec(name="padding", size=1, units="unused"),
    ]

    joints: Dict[str, JointSpec] = {}
    for item in robot_config.actuated_joints:
        name = str(item.get("name"))
        rng = item.get("range") or [0.0, 0.0]
        if len(rng) != 2:
            raise ValueError(f"Invalid range for joint '{name}': {rng}")
        joints[name] = JointSpec(
            range_min_rad=float(rng[0]),
            range_max_rad=float(rng[1]),
            mirror_sign=float(item.get("mirror_sign", 1.0)),
            max_velocity_rad_s=float(item.get("max_velocity", 10.0)),
        )

    postprocess_id = "lowpass_v1" if config.env.action_filter_alpha > 0.0 else "none"
    postprocess_params = {}
    if postprocess_id == "lowpass_v1":
        postprocess_params["alpha"] = float(config.env.action_filter_alpha)

    return PolicySpec(
        contract_name="wildrobot_policy",
        contract_version="1.0.0",
        spec_version=1,
        model=ModelSpec(
            format="onnx",
            input_name="observation",
            output_name="action",
            dtype="float32",
            obs_dim=sum(field.size for field in layout),
            action_dim=action_dim,
        ),
        robot=RobotSpec(
            robot_name=robot_config.robot_name,
            actuator_names=list(robot_config.actuator_names),
            joints=joints,
        ),
        observation=ObservationSpec(
            dtype="float32",
            layout_id="wr_obs_v1",
            layout=layout,
        ),
        action=ActionSpec(
            dtype="float32",
            bounds={"min": -1.0, "max": 1.0},
            postprocess_id=postprocess_id,
            postprocess_params=postprocess_params,
            mapping_id="pos_target_rad_v1",
        ),
        provenance=None,
    )
