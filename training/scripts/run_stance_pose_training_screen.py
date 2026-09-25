#!/usr/bin/env python3
"""Prepare or run the matched WR stance-pose training screen.

All arms start from the same actor checkpoint with fresh critic and optimizer
state.  Generated configs differ only in experiment metadata and the three
geometry-linked fields: canonical roll pose, reference half-width, and the
ToddlerBot-normalized close-feet threshold.
"""

from __future__ import annotations

import argparse
import copy
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from training.configs.training_config import load_training_config_from_dict
from training.eval.verify_walking_stance_geometry import (
    analyze_stance_candidate,
    load_stance_inputs,
)


DEFAULT_BASE_CONFIG = (
    PROJECT_ROOT
    / "training/configs/ppo_walking_v0210_tb11_reference_geometry_com_lever.yaml"
)
DEFAULT_INIT_POLICY = (
    PROJECT_ROOT
    / "training/checkpoints/ppo_walking_v0210_tb9_sysid_actuator_adaptation/"
    "ppo_walking_v0210_tb9_sysid_actuator_adaptation_"
    "v0210-tb9_20260913_085055-zklwsbm9/checkpoint_20_409600.pkl"
)

ARM_SPECS: dict[str, dict[str, Any]] = {
    "control": {
        "version": "0.21.0-tb12a",
        "roll_offset_rad": 0.0300,
        "stance_half_width_m": 0.0781973944718895,
        "close_feet_threshold_m": 0.1268065856300911,
    },
    "moderate": {
        "version": "0.21.0-tb12b",
        "roll_offset_rad": 0.0500,
        "stance_half_width_m": 0.07053951283620806,
        "close_feet_threshold_m": 0.1143883991938509,
    },
    "narrow": {
        "version": "0.21.0-tb12c",
        "roll_offset_rad": 0.0635,
        "stance_half_width_m": 0.06537447970301243,
        "close_feet_threshold_m": 0.1060126697886688,
    },
}


def build_arm_config(
    base: dict[str, Any],
    *,
    arm_name: str,
    seed: int,
    iterations: int,
) -> dict[str, Any]:
    """Return one complete, independently loadable training configuration."""
    spec = ARM_SPECS[arm_name]
    config = copy.deepcopy(base)
    config["version"] = spec["version"]
    config["version_name"] = (
        f"v0.21.0 matched stance-pose screen: {arm_name}"
    )
    config["seed"] = int(seed)

    roll = float(spec["roll_offset_rad"])
    config["env"]["home_joint_offsets_rad"] = {
        "left_hip_roll": roll,
        "right_hip_roll": -roll,
        "left_ankle_roll": -roll,
        "right_ankle_roll": roll,
    }
    config["env"]["loc_ref_default_stance_width_m"] = float(
        spec["stance_half_width_m"]
    )
    config["env"]["close_feet_threshold"] = float(
        spec["close_feet_threshold_m"]
    )

    config["ppo"]["iterations"] = int(iterations)
    config["ppo"]["log_interval"] = 1
    config["ppo"]["eval"]["interval"] = 5
    config["ppo"]["eval"]["post_training_top_k"] = max(1, iterations // 5)
    config["checkpoints"]["interval"] = 5
    config["checkpoints"]["dir"] = (
        f"training/checkpoints/ppo_walking_v0210_tb12_pose_ab_{arm_name}"
    )
    inherited_tags = [
        tag
        for tag in config["wandb"].get("tags", [])
        if tag not in {"tb11", "diagnostic_410k"}
    ]
    config["wandb"]["tags"] = inherited_tags + [
        "tb12_pose_ab",
        f"pose_{arm_name}",
        f"seed_{seed}",
        f"iterations_{iterations}",
    ]
    return config


def _load_base_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("env"), dict):
        raise ValueError(f"invalid training config: {path}")
    return payload


def _write_config(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )


def _validate_geometry(
    config_path: Path,
    expected_half_width_m: float,
) -> dict[str, Any]:
    model, robot_config, home_qpos, home_foot_rotations, close_threshold = (
        load_stance_inputs(config_path)
    )
    result = analyze_stance_candidate(
        model=model,
        robot_config=robot_config,
        home_qpos=home_qpos,
        home_foot_rotations=home_foot_rotations,
        offset_rad=0.0,
        close_feet_threshold_m=close_threshold,
        max_support_torque_ratio=0.8,
        max_foot_orientation_delta_deg=1.0,
        max_sole_height_delta_m=0.002,
    )
    if not result["passed"]:
        failures = [name for name, passed in result["gates"].items() if not passed]
        raise RuntimeError(
            f"generated config failed geometry gates: {', '.join(failures)}"
        )
    actual_half_width_m = float(result["foot_center_separation_m"]) / 2.0
    if abs(actual_half_width_m - expected_half_width_m) > 1e-4:
        raise RuntimeError(
            "generated pose does not realize its reference half-width: "
            f"expected={expected_half_width_m:.6f}, "
            f"actual={actual_half_width_m:.6f}"
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare or sequentially run matched control/moderate/narrow "
            "stance-pose training arms."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--init-policy", type=Path, default=DEFAULT_INIT_POLICY)
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=tuple(ARM_SPECS),
        default=tuple(ARM_SPECS),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--config-output-dir", type=Path, default=None)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Write and validate configs, but do not start training.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_config_path = args.base_config.resolve()
    init_policy = args.init_policy.resolve()
    if not base_config_path.is_file():
        raise FileNotFoundError(f"base config not found: {base_config_path}")
    if not init_policy.is_file():
        raise FileNotFoundError(f"initial policy not found: {init_policy}")
    if args.iterations <= 0 or args.iterations % 5 != 0:
        raise ValueError("iterations must be a positive multiple of 5")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("seeds must be non-empty and unique")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = args.config_output_dir
    if output_dir is None:
        output_dir = PROJECT_ROOT / "training/configs/auto" / f"pose_ab_{stamp}"
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    base = _load_base_config(base_config_path)
    jobs = []
    for seed in args.seeds:
        for arm_name in args.arms:
            config = build_arm_config(
                base,
                arm_name=arm_name,
                seed=int(seed),
                iterations=int(args.iterations),
            )
            parsed = load_training_config_from_dict(config)
            if str(parsed.env.loc_ref_residual_base) != "home":
                raise ValueError("generated pose-screen config must retain home base")
            config_path = output_dir / f"{arm_name}_seed{seed}.yaml"
            _write_config(config_path, config)
            geometry = _validate_geometry(
                config_path,
                float(ARM_SPECS[arm_name]["stance_half_width_m"]),
            )
            command = [
                sys.executable,
                str(PROJECT_ROOT / "training/train.py"),
                "--config",
                str(config_path),
                "--init-policy",
                str(init_policy),
            ]
            jobs.append(
                {
                    "arm": arm_name,
                    "seed": int(seed),
                    "config": str(config_path),
                    "foot_center_separation_m": float(
                        geometry["foot_center_separation_m"]
                    ),
                    "inner_foot_clearance_m": float(
                        geometry["inner_foot_clearance_m"]
                    ),
                    "command": command,
                }
            )

    manifest = {
        "base_config": str(base_config_path),
        "init_policy": str(init_policy),
        "iterations": int(args.iterations),
        "jobs": jobs,
        "controlled_variables": [
            "initial actor checkpoint",
            "PPO and network configuration",
            "reward weights",
            "actuator model and domain randomization",
            "gait timing and command distribution",
        ],
        "arm_variables": [
            "home_joint_offsets_rad",
            "loc_ref_default_stance_width_m",
            "close_feet_threshold",
        ],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("Matched stance-pose training screen", flush=True)
    print(f"  base_config={base_config_path}", flush=True)
    print(f"  init_policy={init_policy}", flush=True)
    print(f"  iterations={args.iterations}", flush=True)
    print(f"  configs={output_dir}", flush=True)
    for job in jobs:
        spec = ARM_SPECS[str(job["arm"])]
        print(
            f"  {job['arm']} seed={job['seed']}: "
            f"roll={spec['roll_offset_rad']:.4f}rad "
            f"half_width={spec['stance_half_width_m']:.4f}m",
            flush=True,
        )
        print(f"    {shlex.join(job['command'])}", flush=True)
    print(f"Wrote {manifest_path}", flush=True)

    if args.prepare_only:
        print("Prepare-only complete; no training was started.", flush=True)
        return 0

    for index, job in enumerate(jobs, start=1):
        print(
            f"\n[{index}/{len(jobs)}] Training {job['arm']} seed={job['seed']}",
            flush=True,
        )
        subprocess.run(job["command"], cwd=PROJECT_ROOT, check=True)
    print("Matched stance-pose training screen complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
