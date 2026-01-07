from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import yaml

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


def _load_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"YAML did not parse to dict: {path}")
    return data


def _get_env_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    env = cfg.get("env")
    if not isinstance(env, dict):
        raise ValueError("Missing or invalid 'env' in training config")
    return env


def _get_robot_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(cfg, dict):
        raise ValueError("robot_config.yaml must be a dict")
    return cfg


def _build_obs_layout(action_dim: int) -> List[ObsFieldSpec]:
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


def _build_joints(robot_cfg: Dict[str, Any]) -> Dict[str, JointSpec]:
    specs = robot_cfg.get("actuated_joint_specs")
    if not isinstance(specs, list) or not specs:
        raise ValueError("robot_config.yaml missing or invalid 'actuated_joint_specs'")

    joints: Dict[str, JointSpec] = {}
    for item in specs:
        if not isinstance(item, dict):
            raise ValueError(f"Invalid joint spec entry: {item!r}")
        name = str(item.get("name"))
        rng = item.get("range") or [0.0, 0.0]
        if not isinstance(rng, list) or len(rng) != 2:
            raise ValueError(f"Invalid range for joint '{name}': {rng}")
        joints[name] = JointSpec(
            range_min_rad=float(rng[0]),
            range_max_rad=float(rng[1]),
            mirror_sign=float(item.get("mirror_sign", 1.0)),
            max_velocity_rad_s=float(item.get("max_velocity", 10.0)),
        )
    return joints


def generate_policy_spec(*, config_path: Path, robot_config_path: Path) -> PolicySpec:
    cfg = _load_yaml(config_path)
    env = _get_env_config(cfg)
    robot_cfg = _get_robot_config(_load_yaml(robot_config_path))

    action_filter_alpha = float(env.get("action_filter_alpha", 0.0))

    joints = _build_joints(robot_cfg)
    actuator_names = list(joints.keys())
    action_dim = len(actuator_names)

    layout = _build_obs_layout(action_dim)

    postprocess_id = "lowpass_v1" if action_filter_alpha > 0.0 else "none"
    postprocess_params: Dict[str, Any] = {}
    if postprocess_id == "lowpass_v1":
        postprocess_params["alpha"] = action_filter_alpha

    spec = PolicySpec(
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
            robot_name=str(robot_cfg.get("robot_name", "wildrobot")),
            actuator_names=actuator_names,
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
        provenance={
            "training_config": str(config_path),
            "robot_config": str(robot_config_path),
        },
    )

    validate_spec(spec)
    return spec


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate policy_spec.json from training config")
    parser.add_argument(
        "--config",
        type=str,
        default="training/configs/ppo_walking.yaml",
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--robot-config",
        type=str,
        default="assets/robot_config.yaml",
        help="Path to robot_config.yaml",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="policy_spec.json",
        help="Output path for policy_spec.json",
    )
    args = parser.parse_args()

    spec = generate_policy_spec(
        config_path=Path(args.config),
        robot_config_path=Path(args.robot_config),
    )

    out_path = Path(args.output)
    out_path.write_text(json.dumps(spec.to_json_dict(), indent=2))
    print(f"Wrote policy spec to {out_path}")


if __name__ == "__main__":
    main()
