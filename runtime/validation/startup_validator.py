from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence

from ..config import RuntimeConfig
from ..utils.mjcf import MjcfModelInfo


class _PolicyInfo(Protocol):
    obs_dim: Optional[int]
    action_dim: Optional[int]


class _PolicyLike(Protocol):
    info: _PolicyInfo


@dataclass(frozen=True)
class RobotConfigInfo:
    action_dim: int
    actuator_names: List[str]
    obs_dim: int


def validate_runtime_interface(
    *,
    cfg: RuntimeConfig,
    mjcf_info: MjcfModelInfo,
    policy: _PolicyLike,
    robot_config_path: Optional[str | Path] = None,
) -> RobotConfigInfo:
    """Fail-fast startup validator.

    Validates that:
    - `assets/robot_config.yaml` actuator_names/action_dim/obs_dim match the ONNX model
    - MJCF-derived actuator order matches `robot_config.yaml` actuator_names
    - Runtime config has servo_ids for all actuators

    Raises:
        ValueError: on any mismatch.
        FileNotFoundError: if robot_config.yaml cannot be located.
    """

    robot_cfg_path = _resolve_robot_config_path(cfg, robot_config_path)
    robot_cfg = _load_robot_config_yaml(robot_cfg_path)

    expected_action_dim = robot_cfg.action_dim
    expected_actuator_names = robot_cfg.actuator_names
    expected_obs_dim = robot_cfg.obs_dim

    mjcf_actuator_names = list(mjcf_info.actuator_names)

    _assert_equal_int("action_dim(robot_config)", expected_action_dim, "len(mjcf_actuators)", len(mjcf_actuator_names))
    _assert_equal_list("actuator_names(robot_config)", expected_actuator_names, "actuator_names(mjcf)", mjcf_actuator_names)

    if policy.info.obs_dim is None:
        raise ValueError(
            "ONNX model input obs_dim could not be determined from the model signature. "
            "Expected a static last-dimension shape like [N, obs_dim]."
        )
    if policy.info.action_dim is None:
        raise ValueError(
            "ONNX model output action_dim could not be determined from the model signature. "
            "Expected a static last-dimension shape like [N, action_dim]."
        )

    _assert_equal_int("obs_dim(robot_config)", expected_obs_dim, "obs_dim(onnx)", int(policy.info.obs_dim))
    _assert_equal_int("action_dim(robot_config)", expected_action_dim, "action_dim(onnx)", int(policy.info.action_dim))

    missing_servo_ids = [name for name in expected_actuator_names if name not in (cfg.servo_ids or {})]
    if missing_servo_ids:
        raise ValueError(
            "Runtime config missing servo_ids for actuators: "
            + ", ".join(missing_servo_ids)
            + ". Update your runtime JSON under hiwonder.servo_ids."
        )

    return robot_cfg


def _resolve_robot_config_path(cfg: RuntimeConfig, robot_config_path: Optional[str | Path]) -> Path:
    if robot_config_path is not None:
        path = Path(robot_config_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"robot_config.yaml not found at: {path}")
        return path

    # Common layout: assets/wildrobot.xml and assets/robot_config.yaml in same directory.
    mjcf_path = Path(cfg.mjcf_path).expanduser()
    candidate = mjcf_path.with_name("robot_config.yaml")
    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        "Could not locate robot_config.yaml. "
        f"Tried: {candidate}. "
        "Pass an explicit robot_config_path to validate_runtime_interface()."
    )


def _load_robot_config_yaml(path: Path) -> RobotConfigInfo:
    data = _yaml_safe_load(path)

    specs = data.get("actuated_joint_specs")
    if not isinstance(specs, list) or not specs:
        raise ValueError(f"robot_config.yaml missing or invalid 'actuated_joint_specs': {path}")

    actuator_names: List[str] = []
    for item in specs:
        if not isinstance(item, dict) or "name" not in item:
            raise ValueError(f"Invalid actuator spec entry in {path}: {item!r}")
        actuator_names.append(str(item["name"]))

    action_dim = len(actuator_names)

    # Prefer the explicit breakdown if present (single source of truth in this repo).
    breakdown = data.get("observation_breakdown")
    if isinstance(breakdown, dict) and breakdown:
        obs_dim = int(sum(int(v) for v in breakdown.values()))
    else:
        # Match training.envs.wildrobot_env.ObsLayout
        fixed = 3 + 3 + 3 + 4 + 1 + 1
        obs_dim = fixed + 3 * action_dim

    return RobotConfigInfo(action_dim=action_dim, actuator_names=actuator_names, obs_dim=obs_dim)


def _yaml_safe_load(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "PyYAML is required for startup validation. "
            "Install it in the runtime environment (e.g. 'pip install pyyaml')."
        ) from exc

    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"robot_config.yaml did not parse to a dict: {path}")
    return data


def _assert_equal_int(a_name: str, a: int, b_name: str, b: int) -> None:
    if int(a) != int(b):
        raise ValueError(f"Interface mismatch: {a_name}={a} != {b_name}={b}")


def _assert_equal_list(a_name: str, a: Sequence[str], b_name: str, b: Sequence[str]) -> None:
    if list(a) == list(b):
        return

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
            f"{a_name}: {list(a)}",
            f"{b_name}: {list(b)}",
            "First diffs:",
            *diffs,
        ]
    )
    raise ValueError(msg)
