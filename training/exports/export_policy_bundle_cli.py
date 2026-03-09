#!/usr/bin/env python3
"""Export a runtime-ready policy bundle (onnx + policy_spec + checksums).

Usage:
  uv run python training/exports/export_policy_bundle_cli.py \
    --checkpoint training/checkpoints/ppo_standing/.../checkpoint_580.pkl \
    --training-config training/configs/ppo_standing.yaml \
    --asset v1 \
        --bundle-path training/checkpoints/standing_v0.12.2
"""

from __future__ import annotations

import argparse
import shutil
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
        robot_path = Path(f"assets/{asset}/mujoco_robot_config.json")
    else:
        default_path = env.get("robot_config_path") or f"{env.get('assets_root', 'assets')}/mujoco_robot_config.json"
        robot_path = Path(default_path)

    return _resolve_repo_relative(robot_path)


def _resolve_training_config_path(
    *,
    checkpoint_path: Path,
    training_config_arg: str | None,
) -> Path:
    if training_config_arg:
        config_path = _resolve_repo_relative(Path(training_config_arg))
        if not config_path.exists():
            raise SystemExit(f"--training-config not found: {config_path}")
        return config_path

    run_dir = checkpoint_path.parent if checkpoint_path.is_file() else checkpoint_path
    direct_candidates = [
        run_dir / "training_config.yaml",
        run_dir / "training_config.yml",
        run_dir / "config.yaml",
        run_dir / "config.yml",
    ]
    candidates = [p for p in direct_candidates if p.exists()]

    if not candidates:
        yaml_files = sorted(run_dir.glob("*.yaml")) + sorted(run_dir.glob("*.yml"))
        if len(yaml_files) == 1:
            candidates = [yaml_files[0]]

    if not candidates:
        config_dir = project_root / "training" / "configs"
        run_name = run_dir.name
        matches = [
            p
            for p in config_dir.glob("*.yaml")
            if run_name.startswith(p.stem)
        ]
        if matches:
            matches.sort(key=lambda p: len(p.stem), reverse=True)
            top_len = len(matches[0].stem)
            top = [m for m in matches if len(m.stem) == top_len]
            if len(top) == 1:
                candidates = [top[0]]

    unique_candidates = list(dict.fromkeys(candidates))
    if len(unique_candidates) == 1:
        return unique_candidates[0].resolve()

    if len(unique_candidates) > 1:
        raise SystemExit(
            "Could not uniquely identify training config from checkpoint context. "
            f"Candidates: {', '.join(str(p) for p in unique_candidates)}. "
            "Please pass --training-config."
        )

    raise SystemExit(
        "Could not identify training config from checkpoint context. "
        "Please pass --training-config."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a WildRobot policy bundle")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to PPO checkpoint (.pkl)")
    parser.add_argument(
        "--training-config",
        type=str,
        default=None,
        help="Path to training config (.yaml). Required if auto-detection fails.",
    )
    parser.add_argument(
        "--bundle-path",
        type=str,
        required=True,
        help="Output bundle directory path, including folder name (e.g., training/checkpoints/standing_v0.12.1).",
    )
    parser.add_argument(
        "--asset",
        type=str,
        choices=["v1", "v2"],
        default=None,
        help="Asset variant (sets robot_config to assets/<asset>/mujoco_robot_config.json and validates assets_root)",
    )
    parser.add_argument(
        "--robot-config",
        type=str,
        default=None,
        help="Path to mujoco_robot_config.json (mutually exclusive with --asset; otherwise config-derived)",
    )
    args = parser.parse_args()

    checkpoint_path = _resolve_repo_relative(Path(args.checkpoint))
    config_path = _resolve_training_config_path(
        checkpoint_path=checkpoint_path,
        training_config_arg=args.training_config,
    )
    output_dir = Path(args.bundle_path)

    robot_config_path = _resolve_robot_config_path(
        robot_config_arg=args.robot_config,
        asset=args.asset,
        config_path=config_path,
    )

    export_policy_bundle(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        output_dir=output_dir,
        robot_config_path=robot_config_path,
    )
    shutil.copy2(config_path, output_dir / "training_config.yaml")

    bundle = PolicyBundle.load(output_dir)
    validate_spec(bundle.spec)
    print(f"Bundle export OK: {output_dir}")


if __name__ == "__main__":
    main()
