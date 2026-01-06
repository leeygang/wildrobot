#!/usr/bin/env python3
"""Export a runtime-ready policy bundle (onnx + policy_spec + checksums).

Usage:
  uv run python playground_amp/scripts/export_policy_bundle.py \
    --checkpoint playground_amp/checkpoints/ppo_standing_v00115_20260103_181636-7jg4si5s/checkpoint_580_76021760.pkl \
    --config playground_amp/configs/ppo_standing.yaml \
    --output-dir playground_amp/checkpoints/wildrobot_policy_bundle
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path (scripts/ -> playground_amp/ -> project_root/)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from policy_contract.spec import PolicyBundle, validate_spec

from playground_amp.exports.export_policy_bundle import export_policy_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a WildRobot policy bundle")
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

    bundle = PolicyBundle.load(Path(args.output_dir))
    validate_spec(bundle.spec)
    print(f"Bundle export OK: {args.output_dir}")


if __name__ == "__main__":
    main()
