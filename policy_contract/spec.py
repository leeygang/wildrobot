from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union


SUPPORTED_LAYOUT_IDS = {"wr_obs_v1"}
SUPPORTED_MAPPING_IDS = {"pos_target_rad_v1"}
SUPPORTED_POSTPROCESS_IDS = {"none", "lowpass_v1"}


@dataclass(frozen=True)
class ModelSpec:
    format: str
    input_name: str
    output_name: str
    dtype: str
    obs_dim: int
    action_dim: int


@dataclass(frozen=True)
class JointSpec:
    range_min_rad: float
    range_max_rad: float
    mirror_sign: float
    max_velocity_rad_s: float


@dataclass(frozen=True)
class RobotSpec:
    robot_name: str
    actuator_names: List[str]
    joints: Dict[str, JointSpec]
    home_ctrl_rad: Optional[List[float]] = None


@dataclass(frozen=True)
class ObsFieldSpec:
    name: str
    size: int
    frame: Optional[str] = None
    units: Optional[str] = None


@dataclass(frozen=True)
class ObservationSpec:
    dtype: str
    layout_id: str
    layout: List[ObsFieldSpec] = field(default_factory=list)


@dataclass(frozen=True)
class ActionSpec:
    dtype: str
    bounds: Dict[str, float]
    postprocess_id: str
    postprocess_params: Dict[str, Any]
    mapping_id: str


@dataclass(frozen=True)
class PolicySpec:
    contract_name: str
    contract_version: str
    spec_version: int
    model: ModelSpec
    robot: RobotSpec
    observation: ObservationSpec
    action: ActionSpec
    provenance: Optional[Dict[str, Any]] = None

    @classmethod
    def from_json(cls, path_or_str: str | Path) -> PolicySpec:
        data = _load_json(path_or_str)
        return _policy_spec_from_dict(data)

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "contract_name": self.contract_name,
            "contract_version": self.contract_version,
            "spec_version": self.spec_version,
            "model": {
                "format": self.model.format,
                "input_name": self.model.input_name,
                "output_name": self.model.output_name,
                "dtype": self.model.dtype,
                "obs_dim": self.model.obs_dim,
                "action_dim": self.model.action_dim,
            },
            "robot": {
                "robot_name": self.robot.robot_name,
                "actuator_names": list(self.robot.actuator_names),
                "joints": {
                    name: {
                        "range_min_rad": joint.range_min_rad,
                        "range_max_rad": joint.range_max_rad,
                        "mirror_sign": joint.mirror_sign,
                        "max_velocity_rad_s": joint.max_velocity_rad_s,
                    }
                    for name, joint in self.robot.joints.items()
                },
                **(
                    {"home_ctrl_rad": list(self.robot.home_ctrl_rad)}
                    if self.robot.home_ctrl_rad is not None
                    else {}
                ),
            },
            "observation": {
                "dtype": self.observation.dtype,
                "layout_id": self.observation.layout_id,
                "layout": [
                    {
                        "name": field.name,
                        "size": field.size,
                        "frame": field.frame,
                        "units": field.units,
                    }
                    for field in self.observation.layout
                ],
            },
            "action": {
                "dtype": self.action.dtype,
                "bounds": dict(self.action.bounds),
                "postprocess_id": self.action.postprocess_id,
                "postprocess_params": dict(self.action.postprocess_params),
                "mapping_id": self.action.mapping_id,
            },
            "provenance": self.provenance,
        }


def canonical_json_dumps(value: Any) -> str:
    """Serialize JSON in a stable, hashable form."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def policy_spec_hash(spec: Union[PolicySpec, Dict[str, Any]]) -> str:
    """Stable fingerprint for a PolicySpec (used to prevent resume drift)."""
    spec_dict = spec.to_json_dict() if isinstance(spec, PolicySpec) else spec
    payload = canonical_json_dumps(spec_dict).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class PolicyBundle:
    spec: PolicySpec
    model_path: Path
    spec_path: Path

    @classmethod
    def load(cls, path: str | Path) -> PolicyBundle:
        path = Path(path)
        if path.is_dir():
            spec_path = path / "policy_spec.json"
            model_path = path / "policy.onnx"
        else:
            if path.suffix.lower() != ".json":
                raise ValueError(f"Expected directory or policy_spec.json path, got: {path}")
            spec_path = path
            model_path = path.parent / "policy.onnx"

        if not spec_path.exists():
            raise FileNotFoundError(f"policy_spec.json not found: {spec_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"policy.onnx not found: {model_path}")

        spec = PolicySpec.from_json(spec_path)
        return cls(spec=spec, model_path=model_path, spec_path=spec_path)


def validate_spec(spec: PolicySpec) -> None:
    if not spec.contract_name:
        raise ValueError("contract_name is required")
    if not spec.contract_version:
        raise ValueError("contract_version is required")
    if spec.spec_version < 1:
        raise ValueError(f"spec_version must be >= 1, got {spec.spec_version}")

    _validate_model(spec.model)
    _validate_robot(spec.robot)
    _validate_observation(spec.observation, spec.model)
    _validate_action(spec.action, spec.model, spec.robot)


def validate_runtime_compat(
    *,
    spec: PolicySpec,
    mjcf_actuator_names: Iterable[str],
    onnx_obs_dim: Optional[int] = None,
    onnx_action_dim: Optional[int] = None,
) -> None:
    validate_spec(spec)

    mjcf_names = list(mjcf_actuator_names)
    if mjcf_names != list(spec.robot.actuator_names):
        _raise_list_mismatch(
            "spec.robot.actuator_names",
            list(spec.robot.actuator_names),
            "mjcf.actuator_names",
            mjcf_names,
        )

    if onnx_obs_dim is not None and int(onnx_obs_dim) != int(spec.model.obs_dim):
        raise ValueError(
            f"ONNX obs_dim mismatch: onnx_obs_dim={onnx_obs_dim} != spec.model.obs_dim={spec.model.obs_dim}"
        )
    if onnx_action_dim is not None and int(onnx_action_dim) != int(spec.model.action_dim):
        raise ValueError(
            f"ONNX action_dim mismatch: onnx_action_dim={onnx_action_dim} != spec.model.action_dim={spec.model.action_dim}"
        )


def _validate_model(model: ModelSpec) -> None:
    for name, value in {
        "model.format": model.format,
        "model.input_name": model.input_name,
        "model.output_name": model.output_name,
        "model.dtype": model.dtype,
    }.items():
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a non-empty string")

    if not isinstance(model.obs_dim, int) or model.obs_dim <= 0:
        raise ValueError(f"model.obs_dim must be a positive int, got {model.obs_dim!r}")
    if not isinstance(model.action_dim, int) or model.action_dim <= 0:
        raise ValueError(f"model.action_dim must be a positive int, got {model.action_dim!r}")


def _validate_robot(robot: RobotSpec) -> None:
    if not isinstance(robot.robot_name, str) or not robot.robot_name:
        raise ValueError("robot.robot_name must be a non-empty string")

    if not isinstance(robot.actuator_names, list) or not robot.actuator_names:
        raise ValueError("robot.actuator_names must be a non-empty list")
    for idx, name in enumerate(robot.actuator_names):
        if not isinstance(name, str) or not name:
            raise ValueError(f"robot.actuator_names[{idx}] must be a non-empty string")

    if not isinstance(robot.joints, dict):
        raise ValueError("robot.joints must be a dict of joint specs")

    missing = [name for name in robot.actuator_names if name not in robot.joints]
    if missing:
        raise ValueError(f"robot.joints missing specs for: {missing}")

    for name, joint in robot.joints.items():
        if not isinstance(joint, JointSpec):
            raise ValueError(f"robot.joints['{name}'] must be a JointSpec")
        if joint.range_min_rad >= joint.range_max_rad:
            raise ValueError(
                f"robot.joints['{name}'] invalid range: {joint.range_min_rad} >= {joint.range_max_rad}"
            )
        if joint.range_min_rad < -2.094395102 or joint.range_max_rad > 2.094395102:
            raise ValueError(
                "robot.joints['{name}'] range must be within [-2.094395102, 2.094395102] rad, "
                f"got [{joint.range_min_rad}, {joint.range_max_rad}]"
            )
        if joint.max_velocity_rad_s <= 0:
            raise ValueError(
                f"robot.joints['{name}'] max_velocity_rad_s must be > 0, got {joint.max_velocity_rad_s}"
            )

    if robot.home_ctrl_rad is not None:
        if not isinstance(robot.home_ctrl_rad, list):
            raise ValueError("robot.home_ctrl_rad must be a list if provided")
        if len(robot.home_ctrl_rad) != len(robot.actuator_names):
            raise ValueError(
                "robot.home_ctrl_rad length must match actuator_names: "
                f"{len(robot.home_ctrl_rad)} != {len(robot.actuator_names)}"
            )
        for idx, value in enumerate(robot.home_ctrl_rad):
            if not isinstance(value, (int, float)):
                raise ValueError(f"robot.home_ctrl_rad[{idx}] must be a number")
            name = robot.actuator_names[idx]
            joint = robot.joints[name]
            if value < joint.range_min_rad or value > joint.range_max_rad:
                raise ValueError(
                    f"robot.home_ctrl_rad[{idx}]={value} out of range "
                    f"[{joint.range_min_rad}, {joint.range_max_rad}] for joint '{name}'"
                )


def _validate_observation(obs: ObservationSpec, model: ModelSpec) -> None:
    if obs.layout_id not in SUPPORTED_LAYOUT_IDS:
        raise ValueError(f"observation.layout_id unsupported: {obs.layout_id}")
    if not isinstance(obs.dtype, str) or not obs.dtype:
        raise ValueError("observation.dtype must be a non-empty string")
    if not obs.layout:
        raise ValueError("observation.layout must be a non-empty list")

    if obs.layout_id == "wr_obs_v1":
        expected = [
            ("gravity_local", 3),
            ("angvel_heading_local", 3),
            ("joint_pos_normalized", int(model.action_dim)),
            ("joint_vel_normalized", int(model.action_dim)),
            ("foot_switches", 4),
            ("prev_action", int(model.action_dim)),
            ("velocity_cmd", 1),
            ("padding", 1),
        ]
        got = [(field.name, int(field.size)) for field in obs.layout]
        if got != expected:
            raise ValueError(
                "observation.layout mismatch for layout_id='wr_obs_v1':\n"
                f"  expected={expected}\n"
                f"  got={got}"
            )

    total = 0
    for idx, field in enumerate(obs.layout):
        if not isinstance(field, ObsFieldSpec):
            raise ValueError(f"observation.layout[{idx}] must be an ObsFieldSpec")
        if not field.name:
            raise ValueError(f"observation.layout[{idx}].name must be a non-empty string")
        if not isinstance(field.size, int) or field.size <= 0:
            raise ValueError(f"observation.layout[{idx}].size must be a positive int")
        total += int(field.size)

    if total != model.obs_dim:
        raise ValueError(f"obs_dim mismatch: model.obs_dim={model.obs_dim} != layout_sum={total}")


def _validate_action(action: ActionSpec, model: ModelSpec, robot: RobotSpec) -> None:
    if not isinstance(action.dtype, str) or not action.dtype:
        raise ValueError("action.dtype must be a non-empty string")
    if action.mapping_id not in SUPPORTED_MAPPING_IDS:
        raise ValueError(f"action.mapping_id unsupported: {action.mapping_id}")
    if action.postprocess_id not in SUPPORTED_POSTPROCESS_IDS:
        raise ValueError(f"action.postprocess_id unsupported: {action.postprocess_id}")

    bounds = action.bounds or {}
    if "min" not in bounds or "max" not in bounds:
        raise ValueError("action.bounds must include 'min' and 'max'")
    if not isinstance(bounds["min"], (int, float)) or not isinstance(bounds["max"], (int, float)):
        raise ValueError("action.bounds min/max must be numbers")
    if float(bounds["min"]) >= float(bounds["max"]):
        raise ValueError(f"action.bounds invalid: min={bounds['min']} >= max={bounds['max']}")

    if model.action_dim != len(robot.actuator_names):
        raise ValueError(
            f"action_dim mismatch: model.action_dim={model.action_dim} != "
            f"len(robot.actuator_names)={len(robot.actuator_names)}"
        )


def _policy_spec_from_dict(data: Dict[str, Any]) -> PolicySpec:
    contract_name = _require_str(data, "contract_name")
    contract_version = _require_str(data, "contract_version")
    spec_version = _require_int(data, "spec_version")

    model = _parse_model(_require_dict(data, "model"))
    robot = _parse_robot(_require_dict(data, "robot"))
    observation = _parse_observation(_require_dict(data, "observation"))
    action = _parse_action(_require_dict(data, "action"))
    provenance = data.get("provenance")
    if provenance is not None and not isinstance(provenance, dict):
        raise ValueError("provenance must be a dict if provided")

    return PolicySpec(
        contract_name=contract_name,
        contract_version=contract_version,
        spec_version=spec_version,
        model=model,
        robot=robot,
        observation=observation,
        action=action,
        provenance=provenance,
    )


def _parse_model(data: Dict[str, Any]) -> ModelSpec:
    return ModelSpec(
        format=_require_str(data, "format", context="model"),
        input_name=_require_str(data, "input_name", context="model"),
        output_name=_require_str(data, "output_name", context="model"),
        dtype=_require_str(data, "dtype", context="model"),
        obs_dim=_require_int(data, "obs_dim", context="model"),
        action_dim=_require_int(data, "action_dim", context="model"),
    )


def _parse_robot(data: Dict[str, Any]) -> RobotSpec:
    actuator_names = _require_list(data, "actuator_names", context="robot")
    joints_raw = _require_dict(data, "joints", context="robot")
    joints: Dict[str, JointSpec] = {}
    for name, spec in joints_raw.items():
        if not isinstance(spec, dict):
            raise ValueError(f"robot.joints['{name}'] must be a dict")
        joints[name] = JointSpec(
            range_min_rad=_require_float(spec, "range_min_rad", context=f"robot.joints['{name}']"),
            range_max_rad=_require_float(spec, "range_max_rad", context=f"robot.joints['{name}']"),
            mirror_sign=_require_float(spec, "mirror_sign", context=f"robot.joints['{name}']"),
            max_velocity_rad_s=_require_float(spec, "max_velocity_rad_s", context=f"robot.joints['{name}']"),
        )

    return RobotSpec(
        robot_name=_require_str(data, "robot_name", context="robot"),
        actuator_names=[_require_str({"_": name}, "_", context="robot.actuator_names") for name in actuator_names],
        joints=joints,
        home_ctrl_rad=_parse_optional_float_list(data, "home_ctrl_rad", context="robot"),
    )


def _parse_optional_float_list(
    data: Dict[str, Any], key: str, *, context: str
) -> Optional[List[float]]:
    if key not in data:
        return None
    value = data[key]
    if not isinstance(value, list):
        raise ValueError(f"{context}.{key} must be a list if provided")
    out = []
    for idx, item in enumerate(value):
        if not isinstance(item, (int, float)):
            raise ValueError(f"{context}.{key}[{idx}] must be a number")
        out.append(float(item))
    return out


def _parse_observation(data: Dict[str, Any]) -> ObservationSpec:
    layout_raw = _require_list(data, "layout", context="observation")
    layout: List[ObsFieldSpec] = []
    for idx, item in enumerate(layout_raw):
        if not isinstance(item, dict):
            raise ValueError(f"observation.layout[{idx}] must be a dict")
        layout.append(
            ObsFieldSpec(
                name=_require_str(item, "name", context=f"observation.layout[{idx}]"),
                size=_require_int(item, "size", context=f"observation.layout[{idx}]"),
                frame=item.get("frame"),
                units=item.get("units"),
            )
        )
    return ObservationSpec(
        dtype=_require_str(data, "dtype", context="observation"),
        layout_id=_require_str(data, "layout_id", context="observation"),
        layout=layout,
    )


def _parse_action(data: Dict[str, Any]) -> ActionSpec:
    bounds = _require_dict(data, "bounds", context="action")
    postprocess_params = data.get("postprocess_params", {})
    if postprocess_params is None:
        postprocess_params = {}
    if not isinstance(postprocess_params, dict):
        raise ValueError("action.postprocess_params must be a dict if provided")
    return ActionSpec(
        dtype=_require_str(data, "dtype", context="action"),
        bounds={
            "min": _require_float(bounds, "min", context="action.bounds"),
            "max": _require_float(bounds, "max", context="action.bounds"),
        },
        postprocess_id=_require_str(data, "postprocess_id", context="action"),
        postprocess_params=postprocess_params,
        mapping_id=_require_str(data, "mapping_id", context="action"),
    )


def _load_json(path_or_str: str | Path) -> Dict[str, Any]:
    if isinstance(path_or_str, Path):
        path = path_or_str.expanduser()
        return _json_safe_load(path.read_text(), source=str(path))

    text = str(path_or_str)
    if text.lstrip().startswith(("{", "[")):
        return _json_safe_load(text, source="<string>")

    try:
        path = Path(text).expanduser()
        if path.exists():
            return _json_safe_load(path.read_text(), source=str(path))
    except OSError:
        # Treat as JSON string if it is not a valid filesystem path.
        return _json_safe_load(text, source="<string>")

    return _json_safe_load(text, source="<string>")


def _json_safe_load(text: str, source: str) -> Dict[str, Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON ({source}): {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"policy_spec.json must be a dict, got {type(data).__name__}")
    return data


def _require_str(data: Dict[str, Any], key: str, *, context: str = "") -> str:
    val = _require_key(data, key, context=context)
    if not isinstance(val, str) or not val:
        name = _ctx_name(context, key)
        raise ValueError(f"{name} must be a non-empty string")
    return val


def _require_int(data: Dict[str, Any], key: str, *, context: str = "") -> int:
    val = _require_key(data, key, context=context)
    if not isinstance(val, int):
        name = _ctx_name(context, key)
        raise ValueError(f"{name} must be an int")
    return val


def _require_float(data: Dict[str, Any], key: str, *, context: str = "") -> float:
    val = _require_key(data, key, context=context)
    if not isinstance(val, (int, float)):
        name = _ctx_name(context, key)
        raise ValueError(f"{name} must be a number")
    return float(val)


def _require_list(data: Dict[str, Any], key: str, *, context: str = "") -> List[Any]:
    val = _require_key(data, key, context=context)
    if not isinstance(val, list):
        name = _ctx_name(context, key)
        raise ValueError(f"{name} must be a list")
    return val


def _require_dict(data: Dict[str, Any], key: str, *, context: str = "") -> Dict[str, Any]:
    val = _require_key(data, key, context=context)
    if not isinstance(val, dict):
        name = _ctx_name(context, key)
        raise ValueError(f"{name} must be a dict")
    return val


def _require_key(data: Dict[str, Any], key: str, context: str) -> Any:
    if key not in data:
        name = _ctx_name(context, key)
        raise ValueError(f"Missing required field: {name}")
    return data[key]


def _ctx_name(context: str, key: str) -> str:
    return f"{context}.{key}" if context else key


def _raise_list_mismatch(a_name: str, a: List[str], b_name: str, b: List[str]) -> None:
    diffs: List[str] = []
    max_len = max(len(a), len(b))
    for i in range(max_len):
        ai = a[i] if i < len(a) else "<missing>"
        bi = b[i] if i < len(b) else "<missing>"
        if ai != bi:
            diffs.append(f"[{i}] {a_name}='{ai}' vs {b_name}='{bi}'")
        if len(diffs) >= 8:
            break

    msg = "\n".join(
        [
            f"Interface mismatch: {a_name} != {b_name}",
            f"{a_name}: {a}",
            f"{b_name}: {b}",
            "First diffs:",
            *diffs,
        ]
    )
    raise ValueError(msg)
