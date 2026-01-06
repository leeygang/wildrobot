#!/usr/bin/env python3
"""Static pose round-trip check for a policy bundle.

This verifies that converting a default control pose to policy action and back
to control targets is numerically stable within a small tolerance.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Add project root to path (scripts/ -> wr_runtime/ -> runtime/ -> project_root/)
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from policy_contract.calib import NumpyCalibOps
from policy_contract.spec import PolicyBundle, validate_spec


def main() -> None:
    parser = argparse.ArgumentParser(description="Check policy bundle pose round-trip")
    parser.add_argument("--bundle", type=str, required=True, help="Path to policy bundle directory")
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-4,
        help="Max allowed absolute error for round-trip ctrl (radians)",
    )
    args = parser.parse_args()

    bundle = PolicyBundle.load(Path(args.bundle))
    validate_spec(bundle.spec)

    if bundle.spec.robot.home_ctrl_rad is None:
        raise ValueError(
            "policy_spec.json missing robot.home_ctrl_rad; re-export bundle with a "
            "canonical home pose"
        )

    ctrl_default = np.asarray(bundle.spec.robot.home_ctrl_rad, dtype=np.float32)
    action = NumpyCalibOps.ctrl_to_policy_action(spec=bundle.spec, ctrl_rad=ctrl_default)
    ctrl_roundtrip = NumpyCalibOps.action_to_ctrl(spec=bundle.spec, action=action)

    err = np.max(np.abs(ctrl_roundtrip - ctrl_default))
    print(f"Round-trip max error: {err:.6e} rad (tol={args.tol})")
    if err > args.tol:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
