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

from wr_runtime.inference.onnx_policy import OnnxPolicy
from wr_runtime.utils.mjcf import load_mjcf_model_info
from wr_runtime.deployment_bundle import (
    DeploymentBundle,
    is_deployment_bundle,
)


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
    """Return the policy bundle's model snapshot."""
    direct_mjcf = bundle_path / "wildrobot.xml"
    if direct_mjcf.exists():
        return direct_mjcf
    raise FileNotFoundError(f"Bundle missing MJCF snapshot: {direct_mjcf}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a policy bundle against MJCF and ONNX dims.")
    parser.add_argument("--bundle", type=str, required=True, help="Bundle directory (contains policy_spec.json + policy.onnx)")
    args = parser.parse_args()

    bundle_path = Path(args.bundle)
    print(f"Bundle: {bundle_path}")

    step = "startup"
    try:
        if is_deployment_bundle(bundle_path):
            step = "load deployment bundle"
            deployment = DeploymentBundle.load(bundle_path)
            mjcf_path = deployment.mjcf_path
            for role in ("standing", "walking"):
                step = f"validate {role} policy bundle"
                validate_bundle(
                    bundle_path=deployment.policy_dir(role),
                    mjcf_path=mjcf_path,
                )
        else:
            step = "locate hardware configuration"
            mjcf_path = _default_mjcf_from_bundle(bundle_path)
            step = "validate policy bundle"
            validate_bundle(bundle_path=bundle_path, mjcf_path=mjcf_path)

        _print_pass("Bundle validation OK")
    except Exception as exc:
        _print_fail(f"{step}: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
