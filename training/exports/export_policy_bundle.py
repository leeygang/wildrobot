#!/usr/bin/env python3
"""Export a policy bundle: policy.onnx + policy_spec.json + checksums.

Usage:
  uv run python training/exports/export_policy_bundle.py \
    --checkpoint training/checkpoints/policy_latest.pkl \
    --config training/configs/ppo_walking.yaml \
    --output-dir training/checkpoints/wildrobot_policy_bundle
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import mujoco
import yaml

from policy_contract.spec import JointSpec, PolicySpec, validate_spec
from policy_contract.spec_builder import build_policy_spec

from training.exports.export_onnx import export_checkpoint_to_onnx, get_checkpoint_dims
from training.configs.asset_paths import resolve_env_asset_paths


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

    mjcf_snapshot_path = _export_mjcf_snapshot(
        output_dir=output_dir, config_path=config_path
    )

    _export_runtime_config(output_dir=output_dir)

    checksums = _build_checksums(
        [
            onnx_path,
            spec_path,
            robot_snapshot_path,
            mjcf_snapshot_path,
            output_dir / "wildrobot_config.json",
        ]
    )
    (output_dir / "checksums.json").write_text(json.dumps(checksums, indent=2))


def _export_runtime_config(
    *,
    output_dir: Path,
) -> None:
    """Generate a runtime config JSON colocated with the exported bundle.

    Source of truth for hardware settings is `runtime/configs/wr_runtime_config.json`.
    The generated config patches:
      - `policy_onnx_path` -> `./policy.onnx`
      - `mjcf_path` -> `./wildrobot.xml` (bundle-local snapshot)
    """
    project_root = Path(__file__).parent.parent.parent
    base_path = project_root / "runtime" / "configs" / "wr_runtime_config.json"

    if not base_path.exists():
        raise FileNotFoundError(f"Runtime config base not found: {base_path}")

    data = json.loads(base_path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Runtime config base is not a JSON object: {base_path}")

    data["policy_onnx_path"] = "./policy.onnx"
    data["mjcf_path"] = "./wildrobot.xml"

    out_path = output_dir / "wildrobot_config.json"
    out_path.write_text(json.dumps(data, indent=2) + "\n")


def _export_mjcf_snapshot(*, output_dir: Path, config_path: Path) -> Path:
    """Snapshot the runtime MJCF into the bundle for self-contained validation.

    Runtime uses MJCF primarily for actuator order validation; this snapshot allows
    `wildrobot-validate-bundle` to run using only files inside the bundle folder.
    """
    src = _resolve_mjcf_path(config_path)
    dst = output_dir / "wildrobot.xml"
    dst.write_text(src.read_text())
    return dst


def _resolve_mjcf_path(config_path: Path) -> Path:
    config = _load_yaml(config_path)
    env = _require_dict(config, "env", context="config")
    resolved = resolve_env_asset_paths(env)
    src = Path(resolved.mjcf_path)
    if not src.is_absolute():
        src = (Path(__file__).parent.parent.parent / src).resolve()
    if not src.exists():
        raise FileNotFoundError(f"MJCF snapshot source not found: {src}")
    return src


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
    home_ctrl_rad = _get_home_ctrl_from_mjcf(config_path, actuator_names=actuator_names)
    home_ctrl_rad = _clamp_home_ctrl(home_ctrl_rad, joints, actuator_names)

    actuated_joint_specs = robot_cfg.get("actuated_joint_specs")
    if not isinstance(actuated_joint_specs, list) or not actuated_joint_specs:
        raise ValueError("robot_config.yaml missing or invalid 'actuated_joint_specs'")

    return build_policy_spec(
        robot_name=str(robot_cfg.get("robot_name", "wildrobot")),
        actuated_joint_specs=actuated_joint_specs,
        action_filter_alpha=action_filter_alpha,
        home_ctrl_rad=home_ctrl_rad,
        provenance={
            "created_at": datetime.now(timezone.utc).isoformat(),
            "training_config": str(config_path),
            "source_checkpoint": {"format": "pkl", "path": str(checkpoint_path)},
            "robot_config": str(robot_config_path),
        },
    )


def _get_home_ctrl_from_mjcf(config_path: Path, actuator_names: list[str]) -> list[float]:
    config = _load_yaml(config_path)
    env = _require_dict(config, "env", context="config")
    resolved = resolve_env_asset_paths(env)
    xml_path = Path(resolved.model_path)
    if not xml_path.is_absolute():
        xml_path = (Path(__file__).parent.parent.parent / xml_path).resolve()

    mj_model = mujoco.MjModel.from_xml_path(str(xml_path))
    mj_data = mujoco.MjData(mj_model)

    key_id = 0
    if mj_model.nkey > 0:
        try:
            key_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "home")
        except Exception:
            key_id = 0
        if key_id < 0:
            key_id = 0
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, key_id)
    else:
        mujoco.mj_resetData(mj_model, mj_data)

    home_ctrl = []
    for name in actuator_names:
        act_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if act_id < 0:
            raise ValueError(f"Actuator '{name}' not found in MJCF for home pose")
        trn_type = mj_model.actuator_trntype[act_id]
        if trn_type != mujoco.mjtTrn.mjTRN_JOINT:
            raise ValueError(
                f"Actuator '{name}' does not target a joint (trntype={int(trn_type)})"
            )
        joint_id = int(mj_model.actuator_trnid[act_id][0])
        qpos_adr = int(mj_model.jnt_qposadr[joint_id])
        home_ctrl.append(float(mj_data.qpos[qpos_adr]))

    return home_ctrl


def _clamp_home_ctrl(
    home_ctrl: list[float],
    joints: Dict[str, JointSpec],
    actuator_names: list[str],
) -> list[float]:
    clipped = []
    clipped_any = False
    for name, value in zip(actuator_names, home_ctrl):
        joint = joints[name]
        clamped = min(max(float(value), joint.range_min_rad), joint.range_max_rad)
        if clamped != float(value):
            clipped_any = True
        clipped.append(clamped)
    if clipped_any:
        print("Warning: home_ctrl_rad clamped to joint limits for export bundle")
    return clipped


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
        default=None,
        help="Path to robot_config.yaml (defaults to env.robot_config_path from config)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    env_section = _require_dict(_load_yaml(config_path), "env", context="config")
    resolved = resolve_env_asset_paths(env_section)
    robot_config_default = Path(resolved.robot_config_path)
    if not robot_config_default.is_absolute():
        robot_config_default = (Path(__file__).parent.parent.parent / robot_config_default).resolve()
    if args.robot_config:
        robot_config_path = Path(args.robot_config)
        if not robot_config_path.is_absolute():
            robot_config_path = (Path(__file__).parent.parent.parent / robot_config_path).resolve()
    else:
        robot_config_path = robot_config_default

    export_policy_bundle(
        checkpoint_path=Path(args.checkpoint),
        config_path=config_path,
        output_dir=Path(args.output_dir),
        robot_config_path=robot_config_path,
    )


if __name__ == "__main__":
    main()
