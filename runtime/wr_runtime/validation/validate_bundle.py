from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from policy_contract.spec import PolicyBundle, validate_runtime_compat, validate_spec

from ..config import WildRobotRuntimeConfig
from ..inference.onnx_policy import OnnxPolicy
from ..utils.mjcf import load_mjcf_model_info


def validate_bundle(
    *,
    bundle_path: Path,
    mjcf_path: Optional[Path],
) -> None:
    bundle = PolicyBundle.load(bundle_path)
    policy = OnnxPolicy(str(bundle.model_path), input_name=bundle.spec.model.input_name, output_name=bundle.spec.model.output_name)

    if mjcf_path is None:
        validate_spec(bundle.spec)
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
        return

    mjcf_info = load_mjcf_model_info(mjcf_path)
    validate_runtime_compat(
        spec=bundle.spec,
        mjcf_actuator_names=mjcf_info.actuator_names,
        onnx_obs_dim=policy.info.obs_dim,
        onnx_action_dim=policy.info.action_dim,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a policy bundle against MJCF and ONNX dims.")
    parser.add_argument("--bundle", type=str, help="Bundle directory (contains policy_spec.json + policy.onnx)")
    parser.add_argument("--mjcf", type=str, default=None, help="Path to MJCF for actuator order validation")
    parser.add_argument("--config", type=str, default=None, help="Runtime JSON config (optional)")
    args = parser.parse_args()

    bundle_path: Optional[Path] = Path(args.bundle) if args.bundle else None
    mjcf_path: Optional[Path] = Path(args.mjcf) if args.mjcf else None

    if args.config:
        cfg = WildRobotRuntimeConfig.load(args.config)
        if bundle_path is None:
            bundle_path = cfg.policy_resolved_path.parent
        if mjcf_path is None:
            mjcf_path = cfg.mjcf_resolved_path

    if bundle_path is None:
        raise ValueError("Provide --bundle or --config")

    validate_bundle(bundle_path=bundle_path, mjcf_path=mjcf_path)
    print("Bundle validation OK")


if __name__ == "__main__":
    main()
