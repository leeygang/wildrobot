from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from policy_contract.spec import PolicyBundle, validate_runtime_compat, validate_spec

_REPO_ROOT = Path(__file__).resolve().parents[3]
_RUNTIME_ROOT = _REPO_ROOT / "runtime"
for _p in (str(_REPO_ROOT), str(_RUNTIME_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from runtime.configs import WildRobotRuntimeConfig
from wr_runtime.inference.onnx_policy import OnnxPolicy
from wr_runtime.utils.mjcf import load_mjcf_model_info


def _print_pass(message: str) -> None:
    print(f"✅ [pass] {message}")


def _print_fail(message: str) -> None:
    print(f"❌ [fail] {message}")


def validate_bundle(
    *,
    bundle_path: Path,
    mjcf_path: Optional[Path],
) -> None:
    bundle = PolicyBundle.load(bundle_path)
    _print_pass(f"found {bundle.spec_path.name}")
    _print_pass(f"found {bundle.model_path.name}")

    policy = OnnxPolicy(str(bundle.model_path), input_name=bundle.spec.model.input_name, output_name=bundle.spec.model.output_name)

    validate_spec(bundle.spec)
    _print_pass("policy_spec.json schema validation")

    if policy.info.obs_dim is None or policy.info.action_dim is None:
        raise ValueError("ONNX model does not expose static obs/action dims")
    if int(policy.info.obs_dim) != bundle.spec.model.obs_dim:
        raise ValueError(
            f"ONNX obs_dim mismatch: onnx={policy.info.obs_dim} != spec={bundle.spec.model.obs_dim}"
        )
    if int(policy.info.action_dim) != bundle.spec.model.action_dim:
        raise ValueError(
            f"ONNX action_dim mismatch: onnx={policy.info.action_dim} != spec={bundle.spec.model.action_dim}"
        )
    _print_pass(f"onnx dims match spec (obs_dim={bundle.spec.model.obs_dim}, action_dim={bundle.spec.model.action_dim})")

    if mjcf_path is None:
        raise ValueError("MJCF path is required for actuator order validation")

    mjcf_info = load_mjcf_model_info(mjcf_path)
    _print_pass(f"found MJCF ({mjcf_path.name})")
    _print_pass(f"loaded actuator order (n={len(mjcf_info.actuator_names)})")
    validate_runtime_compat(
        spec=bundle.spec,
        mjcf_actuator_names=mjcf_info.actuator_names,
        onnx_obs_dim=policy.info.obs_dim,
        onnx_action_dim=policy.info.action_dim,
    )
    _print_pass("MJCF actuator_names match spec.robot.actuator_names")


def _default_mjcf_from_bundle(bundle_path: Path) -> Optional[Path]:
    """Best-effort MJCF path discovery from a colocated runtime config in the bundle."""
    cfg_path = bundle_path / "wildrobot_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Bundle missing wildrobot_config.json: {cfg_path}")
    _print_pass(f"found {cfg_path.name}")
    cfg = WildRobotRuntimeConfig.load(cfg_path)
    mjcf_path = cfg.mjcf_resolved_path
    if not mjcf_path.exists():
        raise FileNotFoundError(f"MJCF not found at bundle-resolved path: {mjcf_path}")
    return mjcf_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a policy bundle against MJCF and ONNX dims.")
    parser.add_argument("--bundle", type=str, required=True, help="Bundle directory (contains policy_spec.json + policy.onnx)")
    args = parser.parse_args()

    bundle_path = Path(args.bundle)
    print(f"Bundle: {bundle_path}")

    step = "startup"
    try:
        step = "locate wildrobot_config.json"
        mjcf_path = _default_mjcf_from_bundle(bundle_path)

        step = "validate policy bundle"
        validate_bundle(bundle_path=bundle_path, mjcf_path=mjcf_path)

        _print_pass("Bundle validation OK")
    except Exception as exc:
        _print_fail(f"{step}: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
