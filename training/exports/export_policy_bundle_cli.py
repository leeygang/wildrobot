#!/usr/bin/env python3
"""Export a runtime-ready policy bundle (onnx + policy_spec + checksums).

Usage:
  uv run python training/exports/export_policy_bundle_cli.py \
    --checkpoint training/checkpoints/ppo_standing/.../checkpoint_580.pkl \
    --config training/configs/ppo_standing.yaml \
    --asset v1 \
    --bundle-name standing_v0.12.2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import yaml

# Add project root to path (exports/ -> training/ -> project_root/)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from policy_contract.spec import PolicyBundle, validate_spec

from training.exports.export_policy_bundle import export_policy_bundle


def _resolve_repo_relative(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def _load_env_section(config_path: Path) -> dict:
    data = yaml.safe_load(config_path.read_text())
    if not isinstance(data, dict):
        return {}
    env = data.get("env", {})
    if not isinstance(env, dict):
        return {}
    from training.configs.asset_paths import resolve_env_asset_paths

    resolved = resolve_env_asset_paths(env)
    env_out = dict(env)
    env_out.setdefault("assets_root", resolved.assets_root)
    env_out.setdefault("scene_xml_path", resolved.scene_xml_path)
    env_out.setdefault("robot_config_path", resolved.robot_config_path)
    env_out.setdefault("mjcf_path", resolved.mjcf_path)
    env_out.setdefault("model_path", resolved.model_path)
    return env_out


def _resolve_robot_config_path(
    *,
    robot_config_arg: str | None,
    asset: str | None,
    config_path: Path,
) -> Path:
    env = _load_env_section(config_path)
    config_assets_root = _resolve_repo_relative(
        Path(env.get("assets_root", "assets"))
    )

    if robot_config_arg and asset:
        raise SystemExit("--robot-config and --asset are mutually exclusive; pass only one.")

    if robot_config_arg:
        robot_path = Path(robot_config_arg)
    elif asset:
        asset_root = _resolve_repo_relative(Path(f"assets/{asset}"))
        if asset_root != config_assets_root:
            raise SystemExit(
                f"--asset={asset} mismatch: config env.assets_root={config_assets_root} "
                "must match the selected asset to avoid mixed MJCF/robot_config. "
                "Update the config or remove --asset."
            )
        robot_path = Path(f"assets/{asset}/robot_config.yaml")
    else:
        default_path = env.get("robot_config_path") or f"{env.get('assets_root', 'assets')}/robot_config.yaml"
        robot_path = Path(default_path)

    return _resolve_repo_relative(robot_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a WildRobot policy bundle")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to PPO checkpoint (.pkl)")
    parser.add_argument("--config", type=str, required=True, help="Path to training config (.yaml)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output bundle directory (overrides --bundle-name/--output-root)",
    )
    parser.add_argument(
        "--bundle-name",
        type=str,
        default=None,
        help="Bundle folder name (e.g., standing_v0.12.1).",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="training/checkpoints",
        help="Root directory for bundles when using --bundle-name",
    )
    parser.add_argument(
        "--asset",
        type=str,
        choices=["v1", "v2"],
        default=None,
        help="Asset variant (sets robot_config to assets/<asset>/robot_config.yaml and validates assets_root)",
    )
    parser.add_argument(
        "--robot-config",
        type=str,
        default=None,
        help="Path to robot_config.yaml (mutually exclusive with --asset; otherwise config-derived)",
    )
    args = parser.parse_args()

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        if not args.bundle_name:
            raise SystemExit("Provide --output-dir or --bundle-name.")
        output_dir = Path(args.output_root) / args.bundle_name

    robot_config_path = _resolve_robot_config_path(
        robot_config_arg=args.robot_config,
        asset=args.asset,
        config_path=Path(args.config),
    )

    export_policy_bundle(
        checkpoint_path=Path(args.checkpoint),
        config_path=Path(args.config),
        output_dir=output_dir,
        robot_config_path=robot_config_path,
    )

    bundle = PolicyBundle.load(output_dir)
    validate_spec(bundle.spec)
    print(f"Bundle export OK: {output_dir}")


if __name__ == "__main__":
    main()
