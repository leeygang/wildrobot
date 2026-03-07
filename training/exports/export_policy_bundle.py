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
import math
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import mujoco
import yaml

# Add project root to import path (exports/ -> training/ -> project_root/)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

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
    checkpoint_snapshot_path = _export_checkpoint_snapshot(
        checkpoint_path=checkpoint_path, output_dir=output_dir
    )

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

    robot_snapshot_path = output_dir / "mujoco_robot_config.json"
    if robot_config_path.exists():
        robot_snapshot_path.write_text(robot_config_path.read_text())

    mjcf_snapshot_path = _export_mjcf_snapshot(
        output_dir=output_dir,
        config_path=config_path,
        actuator_names=spec.robot.actuator_names,
    )

    # Fail-fast: ensure the bundle is self-consistent for the hardware runtime.
    # (This is the same interface check `wildrobot-validate-bundle` performs.)
    mjcf_actuator_names = _load_mjcf_actuator_names(mjcf_snapshot_path)
    from policy_contract.spec import validate_runtime_compat

    validate_runtime_compat(
        spec=spec,
        mjcf_actuator_names=mjcf_actuator_names,
        onnx_obs_dim=obs_dim,
        onnx_action_dim=action_dim,
    )

    _export_runtime_config(output_dir=output_dir)

    checksums = _build_checksums(
        [
            onnx_path,
            checkpoint_snapshot_path,
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

    Source of truth for hardware settings is `runtime/configs/runtime_config_template.json`.
    The generated config patches:
      - `policy_onnx_path` -> `./policy.onnx`
      - `mjcf_path` -> `./wildrobot.xml` (bundle-local snapshot)
    """
    project_root = Path(__file__).parent.parent.parent
    base_path = project_root / "runtime" / "configs" / "runtime_config_template.json"

    if not base_path.exists():
        raise FileNotFoundError(f"Runtime config base not found: {base_path}")

    data = json.loads(base_path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Runtime config base is not a JSON object: {base_path}")

    data["policy_onnx_path"] = "./policy.onnx"
    data["mjcf_path"] = "./wildrobot.xml"

    out_path = output_dir / "wildrobot_config.json"
    out_path.write_text(json.dumps(data, indent=2) + "\n")


def _export_mjcf_snapshot(
    *,
    output_dir: Path,
    config_path: Path,
    actuator_names: list[str] | None = None,
) -> Path:
    """Snapshot the runtime MJCF into the bundle for self-contained validation.

    Runtime uses MJCF primarily for actuator order validation; this snapshot allows
    `wildrobot-validate-bundle` to run using only files inside the bundle folder.
    """
    src = _resolve_mjcf_path(config_path)
    dst = output_dir / "wildrobot.xml"
    text = src.read_text()
    if actuator_names is not None:
        text = _rewrite_mjcf_actuator_order(text, actuator_names)
    dst.write_text(text)
    return dst


def _load_mjcf_actuator_names(xml_path: Path) -> list[str]:
    """Return MJCF actuator names in runtime order.

    Runtime treats `<actuator><position name=...>` element ordering as the action order.
    This avoids needing MuJoCo to parse the XML (and avoids mesh/asset resolution).
    """
    import xml.etree.ElementTree as ET

    root = ET.parse(xml_path).getroot()
    actuator = root.find("actuator")
    if actuator is None:
        raise ValueError(f"MJCF missing <actuator>: {xml_path}")

    names: list[str] = []
    for pos_act in actuator.findall("position"):
        name = pos_act.get("name")
        if name:
            names.append(str(name))

    if not names:
        raise ValueError(f"No <actuator><position name=...> found in: {xml_path}")
    return names


_ACTUATOR_NAME_RE = re.compile(r"\bname=\"([^\"]+)\"")


def _rewrite_mjcf_actuator_order(xml_text: str, actuator_names: list[str]) -> str:
    """Rewrite the <actuator> block to match `actuator_names` ordering.

    Why: runtime requires `policy_spec.robot.actuator_names` to match the MJCF actuator order.
    MuJoCo actuator order is defined by the order of children inside the <actuator> element.

    This rewrite preserves the file text outside the <actuator> block and keeps each actuator
    line intact (no XML reformatting) under the assumption actuators are one-per-line (as in
    our generated MJCFs).
    """
    if not actuator_names:
        raise ValueError("actuator_names must be a non-empty list")

    lines = xml_text.splitlines(True)  # keep line endings
    start_idx = None
    end_idx = None
    for i, line in enumerate(lines):
        if start_idx is None and "<actuator" in line and line.strip().startswith("<actuator"):
            start_idx = i
            continue
        if start_idx is not None and line.strip().startswith("</actuator"):
            end_idx = i
            break

    if start_idx is None or end_idx is None or end_idx <= start_idx:
        raise ValueError("MJCF missing <actuator> block")

    inner = lines[start_idx + 1 : end_idx]
    by_name: dict[str, str] = {}
    passthrough: list[str] = []

    for line in inner:
        m = _ACTUATOR_NAME_RE.search(line)
        if m and line.strip().startswith("<") and line.strip().endswith("/>"):
            name = m.group(1)
            if name in by_name:
                raise ValueError(f"Duplicate actuator name in MJCF: {name}")
            by_name[name] = line
        else:
            passthrough.append(line)

    missing = [name for name in actuator_names if name not in by_name]
    if missing:
        raise ValueError(f"MJCF actuator block missing names required by spec: {missing}")

    # Rebuild inner actuator lines: desired order first, then any remaining named actuators.
    ordered: list[str] = [by_name[name] for name in actuator_names]
    remaining = [
        line
        for name, line in by_name.items()
        if name not in set(actuator_names)
    ]

    new_inner = ordered + remaining
    # Keep any non-actuator lines (whitespace/comments) at the end.
    new_inner.extend(passthrough)

    return "".join(lines[: start_idx + 1] + new_inner + lines[end_idx:])


def _export_checkpoint_snapshot(*, checkpoint_path: Path, output_dir: Path) -> Path:
    """Copy source checkpoint into the bundle for traceability/re-export."""
    src = checkpoint_path
    if not src.exists():
        raise FileNotFoundError(f"Checkpoint source not found: {src}")
    dst = output_dir / "checkpoint.pkl"
    shutil.copy2(src, dst)
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

    actuated_joint_specs = _normalize_actuated_joint_specs_to_rad(robot_cfg)
    if not isinstance(actuated_joint_specs, list) or not actuated_joint_specs:
        raise ValueError("mujoco_robot_config.json missing or invalid 'actuated_joint_specs'")

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
    clipped_details: list[tuple[str, float, float, float, float]] = []
    for name, value in zip(actuator_names, home_ctrl):
        joint = joints[name]
        original = float(value)
        clamped = min(max(original, joint.range_min_rad), joint.range_max_rad)
        if not math.isclose(clamped, original, rel_tol=0.0, abs_tol=1e-12):
            clipped_details.append(
                (
                    name,
                    original,
                    clamped,
                    float(joint.range_min_rad),
                    float(joint.range_max_rad),
                )
            )
        clipped.append(clamped)
    if clipped_details:
        print("Warning: home_ctrl_rad clamped to joint limits for export bundle")
        print(
            f"  clamped_actuators: {len(clipped_details)}/{len(actuator_names)} "
            "(source: MJCF home keyframe or keyframe[0] fallback)"
        )
        for name, original, clamped, range_min, range_max in clipped_details:
            print(
                f"  - {name}: {original:.6f} -> {clamped:.6f} rad "
                f"(limits [{range_min:.6f}, {range_max:.6f}])"
            )
        print(
            "  hint: align MJCF home pose with mujoco_robot_config joint ranges "
            "to avoid export-time clamping"
        )
    return clipped


def _build_joints(robot_cfg: Dict[str, Any]) -> Dict[str, JointSpec]:
    specs = _normalize_actuated_joint_specs_to_rad(robot_cfg)
    if not isinstance(specs, list) or not specs:
        raise ValueError("mujoco_robot_config.json missing or invalid 'actuated_joint_specs'")

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
            policy_action_sign=float(item.get("policy_action_sign", 1.0)),
            max_velocity_rad_s=float(item.get("max_velocity", 10.0)),
        )
    return joints


def _normalize_actuated_joint_specs_to_rad(robot_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    specs = robot_cfg.get("actuated_joint_specs")
    if not isinstance(specs, list):
        return []

    range_unit = str(robot_cfg.get("joint_range_unit", "rad")).lower()
    if range_unit not in {"rad", "deg"}:
        raise ValueError("mujoco_robot_config.json 'joint_range_unit' must be 'rad' or 'deg'")

    normalized_specs: List[Dict[str, Any]] = []
    for item in specs:
        if not isinstance(item, dict):
            raise ValueError(f"Invalid joint spec entry: {item!r}")
        normalized_item = dict(item)
        rng = normalized_item.get("range") or [0.0, 0.0]
        if not isinstance(rng, list) or len(rng) != 2:
            raise ValueError(f"Invalid range for joint '{item.get('name')}': {rng}")
        range_min = float(rng[0])
        range_max = float(rng[1])
        if range_unit == "deg":
            range_min = math.radians(range_min)
            range_max = math.radians(range_max)
        normalized_item["range"] = [range_min, range_max]
        normalized_specs.append(normalized_item)

    return normalized_specs


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
        help="Path to mujoco_robot_config.json (defaults to env.robot_config_path from config)",
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
