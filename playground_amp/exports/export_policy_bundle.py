#!/usr/bin/env python3
"""Export a policy bundle: policy.onnx + policy_spec.json + checksums.

Usage:
  uv run python playground_amp/exports/export_policy_bundle.py \
    --checkpoint playground_amp/checkpoints/policy_latest.pkl \
    --config playground_amp/configs/ppo_walking.yaml \
    --output-dir playground_amp/checkpoints/wildrobot_policy_bundle
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
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

from playground_amp.exports.export_onnx import export_checkpoint_to_onnx, get_checkpoint_dims


def export_policy_bundle(
    *,
    checkpoint_path: Path,
    config_path: Path,
    output_dir: Path,
    robot_config_path: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = output_dir / "policy.onnx"
    export_checkpoint_to_onnx(checkpoint_path=checkpoint_path, output_path=onnx_path)

    spec = _build_policy_spec(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        robot_config_path=robot_config_path,
    )

    obs_dim, action_dim = get_checkpoint_dims(checkpoint_path)
    if spec.model.obs_dim != obs_dim:
        raise ValueError(
            f"spec.model.obs_dim={spec.model.obs_dim} != checkpoint.obs_dim={obs_dim} "
            f"(checkpoint={checkpoint_path})"
        )
    if spec.model.action_dim != action_dim:
        raise ValueError(
            f"spec.model.action_dim={spec.model.action_dim} != checkpoint.action_dim={action_dim} "
            f"(checkpoint={checkpoint_path})"
        )

    spec_path = output_dir / "policy_spec.json"
    spec_path.write_text(json.dumps(spec.to_json_dict(), indent=2))

    robot_snapshot_path = output_dir / "robot_config.yaml"
    if robot_config_path.exists():
        robot_snapshot_path.write_text(robot_config_path.read_text())

    checksums = _build_checksums([onnx_path, spec_path, robot_snapshot_path])
    (output_dir / "checksums.json").write_text(json.dumps(checksums, indent=2))


def _build_policy_spec(
    *,
    checkpoint_path: Path,
    config_path: Path,
    robot_config_path: Path,
) -> PolicySpec:
    config = _load_yaml(config_path)
    env = _require_dict(config, "env", context="config")
    robot_cfg = _load_yaml(robot_config_path)

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
            "created_at": datetime.now(timezone.utc).isoformat(),
            "training_config": str(config_path),
            "source_checkpoint": {"format": "pkl", "path": str(checkpoint_path)},
            "robot_config": str(robot_config_path),
        },
    )
    validate_spec(spec)
    return spec


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


def _build_checksums(paths: List[Path]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for path in paths:
        if not path.exists():
            continue
        out[path.name] = _sha256(path)
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"YAML did not parse to dict: {path}")
    return data


def _require_dict(data: Dict[str, Any], key: str, *, context: str) -> Dict[str, Any]:
    if key not in data or not isinstance(data[key], dict):
        raise ValueError(f"Missing or invalid '{context}.{key}'")
    return data[key]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export WildRobot policy bundle")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to PPO checkpoint (.pkl)")
    parser.add_argument("--config", type=str, required=True, help="Path to training config (.yaml)")
    parser.add_argument("--output-dir", type=str, required=True, help="Output bundle directory")
    parser.add_argument(
        "--robot-config",
        type=str,
        default="assets/robot_config.yaml",
        help="Path to robot_config.yaml",
    )
    args = parser.parse_args()

    export_policy_bundle(
        checkpoint_path=Path(args.checkpoint),
        config_path=Path(args.config),
        output_dir=Path(args.output_dir),
        robot_config_path=Path(args.robot_config),
    )


if __name__ == "__main__":
    main()
