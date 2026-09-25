#!/usr/bin/env python3
"""Prepare or run a 2x2 WR morphology length/mass training experiment.

The experiment separates two causes that a pose-only test cannot distinguish:

* geometry: uniformly scale every robot length while preserving link mass;
* mass: uniformly scale link mass while preserving geometry;
* interaction: apply both changes together.

Actuator dynamics, torque limits, policy inputs, rewards, commands, and random
seeds otherwise remain matched.  Geometry cases use Froude-scaled gait timing
and dimensionally scaled reference/reward lengths.  The generated morphology
is an idealized causal counterfactual, not a manufacturable CAD proposal.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import mujoco
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from training.configs.training_config import load_training_config_from_dict
from training.eval.audit_reference_feasibility import (
    build_reference_feasibility_report,
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
SOURCE_SCENE = PROJECT_ROOT / "assets/v2/scene_flat_terrain.xml"
SOURCE_ROBOT = PROJECT_ROOT / "assets/v2/wildrobot.xml"
SOURCE_KEYFRAMES = PROJECT_ROOT / "assets/v2/keyframes.xml"
SOURCE_MESHES = PROJECT_ROOT / "assets/v2/assets"

# Source-of-truth morphology values:
# WR ZMPWalkConfig.upper_leg_m + lower_leg_m = 0.193 + 0.180 m.
# ToddlerBot robot.yml hip_to_ankle_pitch_z = 0.2115 m.
WR_REFERENCE_LEG_LENGTH_M = 0.373
TB_REFERENCE_LEG_LENGTH_M = 0.2115
# Compiled ToddlerBot 2xm scene_mjx.xml body mass (2026-09-25 audit).
TB_REFERENCE_MASS_KG = 3.76887188


@dataclass(frozen=True)
class FactorialArm:
    name: str
    version: str
    geometry_scale: float
    mass_scale: float


def _format_values(values: Iterable[float]) -> str:
    return " ".join(f"{float(value):.12g}" for value in values)


def _scale_attribute(element: ET.Element, name: str, factor: float) -> None:
    raw = element.get(name)
    if raw is None:
        return
    element.set(name, _format_values(float(value) * factor for value in raw.split()))


def _write_scaled_robot_xml(
    output_path: Path,
    *,
    geometry_scale: float,
    mass_scale: float,
) -> None:
    tree = ET.parse(SOURCE_ROBOT)
    root = tree.getroot()
    root.set("model", f"WildRobot_g{geometry_scale:.6f}_m{mass_scale:.6f}")

    for element in root.iter():
        if element.tag in {"body", "geom", "site", "joint", "camera"}:
            _scale_attribute(element, "pos", geometry_scale)
        if element.tag in {"geom", "site"}:
            _scale_attribute(element, "size", geometry_scale)
            _scale_attribute(element, "fromto", geometry_scale)
        if element.tag == "mesh":
            raw_scale = element.get("scale", "1 1 1")
            values = [float(value) for value in raw_scale.split()]
            if len(values) == 1:
                values *= 3
            if len(values) != 3:
                raise ValueError(f"invalid mesh scale: {raw_scale!r}")
            element.set(
                "scale",
                _format_values(value * geometry_scale for value in values),
            )
        if element.tag == "inertial":
            _scale_attribute(element, "pos", geometry_scale)
            if "mass" not in element.attrib:
                raise ValueError("every WR body must have an explicit inertial mass")
            element.set(
                "mass",
                f"{float(element.attrib['mass']) * mass_scale:.12g}",
            )
            inertia_factor = mass_scale * geometry_scale * geometry_scale
            if "fullinertia" in element.attrib:
                _scale_attribute(element, "fullinertia", inertia_factor)
            elif "diaginertia" in element.attrib:
                _scale_attribute(element, "diaginertia", inertia_factor)
            else:
                raise ValueError("every WR inertial must specify inertia explicitly")

    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="unicode")


def _write_scaled_keyframes(output_path: Path, *, geometry_scale: float) -> None:
    tree = ET.parse(SOURCE_KEYFRAMES)
    for key in tree.getroot().iter("key"):
        qpos = [float(value) for value in key.attrib["qpos"].split()]
        if len(qpos) < 7:
            raise ValueError("WR keyframes must begin with a free-joint pose")
        qpos[:3] = [value * geometry_scale for value in qpos[:3]]
        key.set("qpos", _format_values(qpos))
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="unicode")


def write_scaled_assets(
    output_dir: Path,
    *,
    geometry_scale: float,
    mass_scale: float,
) -> Path:
    """Write one self-contained scaled MJCF asset directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(SOURCE_MESHES, output_dir / "assets", dirs_exist_ok=True)
    _write_scaled_robot_xml(
        output_dir / "wildrobot.xml",
        geometry_scale=geometry_scale,
        mass_scale=mass_scale,
    )
    _write_scaled_keyframes(
        output_dir / "keyframes.xml",
        geometry_scale=geometry_scale,
    )
    scene_tree = ET.parse(SOURCE_SCENE)
    ET.indent(scene_tree, space="  ")
    scene_path = output_dir / "scene_flat_terrain.xml"
    scene_tree.write(scene_path, encoding="unicode")
    return scene_path


def _named_id(model: mujoco.MjModel, object_type: Any, name: str) -> int:
    object_id = int(mujoco.mj_name2id(model, object_type, name))
    if object_id < 0:
        raise ValueError(f"MuJoCo object not found: {name}")
    return object_id


def inspect_model(scene_path: Path) -> dict[str, Any]:
    """Return physical measurements used to validate a generated model."""
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    home_id = _named_id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, home_id)
    mujoco.mj_forward(model, data)

    root_id = _named_id(model, mujoco.mjtObj.mjOBJ_BODY, "waist")
    joint_ids = [
        _named_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in ("left_hip_roll", "left_knee_pitch", "left_ankle_pitch")
    ]
    anchors = [np.asarray(data.xanchor[joint_id]).copy() for joint_id in joint_ids]
    leg_chain_length = float(
        np.linalg.norm(anchors[1] - anchors[0])
        + np.linalg.norm(anchors[2] - anchors[1])
    )
    lateral_axis = np.asarray(data.xmat[root_id]).reshape(3, 3)[:, 1]
    root_position = np.asarray(data.xpos[root_id])
    hip_offsets = []
    for side in ("left", "right"):
        joint_id = _named_id(
            model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_hip_roll"
        )
        hip_offsets.append(
            abs(float(np.dot(np.asarray(data.xanchor[joint_id]) - root_position, lateral_axis)))
        )
    foot_positions = []
    for side in ("left", "right"):
        body_id = _named_id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_foot")
        foot_positions.append(np.asarray(data.xpos[body_id]).copy())
    foot_separation = abs(float(np.dot(foot_positions[0] - foot_positions[1], lateral_axis)))
    force_limits = np.max(np.abs(np.asarray(model.actuator_forcerange)), axis=1)
    return {
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        "nbody": int(model.nbody),
        "total_mass_kg": float(model.body_subtreemass[root_id]),
        "home_root_height_m": float(data.qpos[2]),
        "leg_joint_chain_length_m": leg_chain_length,
        "hip_lateral_offset_m": float(np.mean(hip_offsets)),
        "home_foot_center_separation_m": foot_separation,
        "actuator_force_limit_min_nm": float(np.min(force_limits)),
        "actuator_force_limit_max_nm": float(np.max(force_limits)),
        "body_mass": np.asarray(model.body_mass[1:]).tolist(),
        "body_inertia": np.asarray(model.body_inertia[1:]).tolist(),
    }


def validate_scaled_model(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    geometry_scale: float,
    mass_scale: float,
) -> None:
    for key in ("nq", "nv", "nu", "nbody"):
        if candidate[key] != baseline[key]:
            raise RuntimeError(f"scaled model changed topology field {key}")
    scalar_scales = {
        "total_mass_kg": mass_scale,
        "home_root_height_m": geometry_scale,
        "leg_joint_chain_length_m": geometry_scale,
        "hip_lateral_offset_m": geometry_scale,
        "home_foot_center_separation_m": geometry_scale,
    }
    for key, expected_scale in scalar_scales.items():
        expected = float(baseline[key]) * expected_scale
        if not math.isclose(float(candidate[key]), expected, rel_tol=2e-5, abs_tol=2e-6):
            raise RuntimeError(
                f"scaled model {key} mismatch: expected={expected:.9g}, "
                f"actual={candidate[key]:.9g}"
            )
    if not np.allclose(
        candidate["body_mass"],
        np.asarray(baseline["body_mass"]) * mass_scale,
        rtol=2e-5,
        atol=1e-8,
    ):
        raise RuntimeError("scaled model body masses do not match mass_scale")
    expected_inertia = (
        np.asarray(baseline["body_inertia"])
        * mass_scale
        * geometry_scale
        * geometry_scale
    )
    if not np.allclose(
        candidate["body_inertia"], expected_inertia, rtol=3e-5, atol=1e-10
    ):
        raise RuntimeError("scaled model inertias do not match m*s^2")
    for key in ("actuator_force_limit_min_nm", "actuator_force_limit_max_nm"):
        if not math.isclose(candidate[key], baseline[key], abs_tol=1e-9):
            raise RuntimeError("scaled model changed the HTD actuator force limit")


def factorial_arms(*, geometry_scale: float, mass_scale: float) -> tuple[FactorialArm, ...]:
    return (
        FactorialArm("baseline", "0.21.0-morph-a", 1.0, 1.0),
        FactorialArm("dimension_only", "0.21.0-morph-b", geometry_scale, 1.0),
        FactorialArm("mass_only", "0.21.0-morph-c", 1.0, mass_scale),
        FactorialArm(
            "dimension_and_mass",
            "0.21.0-morph-d",
            geometry_scale,
            mass_scale,
        ),
    )


def _scale_config_value(section: dict[str, Any], key: str, factor: float) -> None:
    if key in section:
        section[key] = float(section[key]) * factor


def _config_path_value(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def build_arm_config(
    base: dict[str, Any],
    *,
    arm: FactorialArm,
    scene_path: Path,
    seed: int,
    iterations: int,
) -> dict[str, Any]:
    config = copy.deepcopy(base)
    config["version"] = arm.version
    config["version_name"] = f"v0.21.0 morphology factorial: {arm.name}"
    config["seed"] = int(seed)
    env = config["env"]
    geometry_scale = float(arm.geometry_scale)
    mass_scale = float(arm.mass_scale)
    scene_text = _config_path_value(scene_path)
    env["assets_root"] = _config_path_value(scene_path.parent)
    env["scene_xml_path"] = scene_text
    env["model_path"] = scene_text
    env["mjcf_path"] = _config_path_value(scene_path.with_name("wildrobot.xml"))
    env["robot_config_path"] = _config_path_value(
        PROJECT_ROOT / "assets/v2/mujoco_robot_config.json"
    )
    env["loc_ref_morphology_length_scale"] = geometry_scale

    for key in ("target_height", "min_height", "max_height"):
        _scale_config_value(env, key, geometry_scale)
    for key in (
        "loc_ref_default_stance_width_m",
        "loc_ref_hip_lateral_offset_m",
        "close_feet_threshold",
        "min_feet_y_dist",
        "max_feet_y_dist",
        "loc_ref_walking_pelvis_height_m",
        "loc_ref_nominal_lateral_foot_offset_m",
        "loc_ref_min_step_length_m",
        "loc_ref_max_step_length_m",
        "loc_ref_max_lateral_step_m",
        "loc_ref_swing_height_m",
        "loc_ref_support_margin_m",
        "loc_ref_walking_crouch_extra_m",
        "loc_ref_max_swing_x_delta_m",
        "loc_ref_max_swing_z_delta_m",
        "reset_foot_stagger_range_m",
    ):
        value = env.get(key)
        if isinstance(value, list):
            env[key] = [float(item) * geometry_scale for item in value]
        else:
            _scale_config_value(env, key, geometry_scale)
    env["single_support_com_lateral_normalization_m"] = 0.10 * geometry_scale
    for key in ("contact_threshold_force", "foot_switch_threshold"):
        _scale_config_value(env, key, mass_scale)

    rewards = config["reward_weights"]
    _scale_config_value(rewards, "feet_phase_swing_height", geometry_scale)
    if "feet_phase_alpha" in rewards:
        rewards["feet_phase_alpha"] = float(rewards["feet_phase_alpha"]) / (
            geometry_scale * geometry_scale
        )

    config["ppo"]["iterations"] = int(iterations)
    config["ppo"]["log_interval"] = 1
    config["ppo"]["eval"]["interval"] = 5
    config["ppo"]["eval"]["post_training_top_k"] = min(5, iterations)
    # Preserve every state so the exact final policy can be evaluated. The
    # prior pose screen saved only the reward-best state in a five-step window.
    config["checkpoints"]["interval"] = 1
    config["checkpoints"]["dir"] = (
        f"training/checkpoints/ppo_walking_v0210_morphology_{arm.name}"
    )
    inherited_tags = [
        tag
        for tag in config["wandb"].get("tags", [])
        if tag not in {"tb11", "diagnostic_410k"}
    ]
    config["wandb"]["tags"] = inherited_tags + [
        "morphology_size_factorial",
        arm.name,
        f"geometry_scale_{geometry_scale:.6f}",
        f"mass_scale_{mass_scale:.6f}",
        f"seed_{seed}",
        f"iterations_{iterations}",
    ]
    return config


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )


def _audit_commands(
    config: dict[str, Any], *, geometry_scale: float
) -> list[tuple[float, float, float]]:
    env = config["env"]
    commands = [tuple(float(value) for value in env["eval_velocity_cmd"])]
    commands.extend(
        tuple(float(value) for value in probe)
        for probe in env.get("eval_velocity_cmd_probes", [])
    )
    primary = commands[0]
    if geometry_scale < 1.0:
        # The standard policy gate remains at the same absolute deployment
        # speed.  This additional mechanics-only command compares equal
        # Froude number without introducing an unmatched reference bin into
        # training or post-training evaluation.
        commands.append(
            (
                primary[0] * math.sqrt(geometry_scale),
                primary[1],
                primary[2],
            )
        )
    return list(dict.fromkeys(commands))


def _audit_metrics(report: dict[str, Any]) -> list[dict[str, Any]]:
    metrics = []
    for command in report["commands"]:
        support = command["support"]
        metrics.append(
            {
                "command": command["command"],
                "max_support_ratio_p95": max(
                    float(support[side]["quasi_static_support_ratio"]["p95"])
                    for side in ("left", "right")
                ),
                "max_realized_lateral_lever_mean_m": max(
                    float(support[side]["realized_lateral_lever_abs_m"]["mean"])
                    for side in ("left", "right")
                ),
            }
        )
    return metrics


def _factorial_effects(jobs: list[dict[str, Any]]) -> dict[str, float]:
    primary_by_arm: dict[str, float] = {}
    for job in jobs:
        arm = str(job["arm"])
        if arm in primary_by_arm:
            continue
        primary_by_arm[arm] = float(
            job["reference_metrics"][0]["max_support_ratio_p95"]
        )
    required = {"baseline", "dimension_only", "mass_only", "dimension_and_mass"}
    if set(primary_by_arm) != required:
        return {}
    baseline = primary_by_arm["baseline"]
    dimension = primary_by_arm["dimension_only"]
    mass = primary_by_arm["mass_only"]
    combined = primary_by_arm["dimension_and_mass"]
    return {
        "baseline_max_support_ratio_p95": baseline,
        "dimension_effect_absolute": dimension - baseline,
        "dimension_effect_relative": dimension / baseline - 1.0,
        "mass_effect_absolute": mass - baseline,
        "mass_effect_relative": mass / baseline - 1.0,
        "combined_effect_absolute": combined - baseline,
        "combined_effect_relative": combined / baseline - 1.0,
        "interaction_absolute": combined - dimension - mass + baseline,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare or run the WR length/mass morphology factorial",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--init-policy", type=Path, default=DEFAULT_INIT_POLICY)
    parser.add_argument(
        "--target-leg-length-m", type=float, default=TB_REFERENCE_LEG_LENGTH_M
    )
    parser.add_argument("--target-mass-kg", type=float, default=TB_REFERENCE_MASS_KG)
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=("baseline", "dimension_only", "mass_only", "dimension_and_mass"),
        default=("baseline", "dimension_only", "mass_only", "dimension_and_mass"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--prepare-only", action="store_true")
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
    if not math.isfinite(args.target_leg_length_m) or args.target_leg_length_m <= 0:
        raise ValueError("target-leg-length-m must be finite and positive")
    if not math.isfinite(args.target_mass_kg) or args.target_mass_kg <= 0:
        raise ValueError("target-mass-kg must be finite and positive")

    baseline_model = inspect_model(SOURCE_SCENE)
    geometry_scale = float(args.target_leg_length_m) / WR_REFERENCE_LEG_LENGTH_M
    mass_scale = float(args.target_mass_kg) / float(baseline_model["total_mass_kg"])
    if geometry_scale >= 1.0:
        raise ValueError("this experiment requires a target leg shorter than WR")
    arms_by_name = {
        arm.name: arm
        for arm in factorial_arms(
            geometry_scale=geometry_scale,
            mass_scale=mass_scale,
        )
    }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = PROJECT_ROOT / "training/configs/auto" / f"morphology_{stamp}"
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    if not isinstance(base, dict) or not isinstance(base.get("env"), dict):
        raise ValueError(f"invalid training config: {base_config_path}")

    jobs: list[dict[str, Any]] = []
    model_reports: dict[str, Any] = {}
    for arm_name in args.arms:
        arm = arms_by_name[arm_name]
        asset_dir = output_dir / "assets" / arm.name
        scene_path = write_scaled_assets(
            asset_dir,
            geometry_scale=arm.geometry_scale,
            mass_scale=arm.mass_scale,
        )
        model_report = inspect_model(scene_path)
        validate_scaled_model(
            baseline_model,
            model_report,
            geometry_scale=arm.geometry_scale,
            mass_scale=arm.mass_scale,
        )
        model_reports[arm.name] = {
            key: value
            for key, value in model_report.items()
            if key not in {"body_mass", "body_inertia"}
        }
        for seed in args.seeds:
            config = build_arm_config(
                base,
                arm=arm,
                scene_path=scene_path,
                seed=seed,
                iterations=args.iterations,
            )
            load_training_config_from_dict(config)
            config_path = output_dir / "configs" / f"{arm.name}_seed{seed}.yaml"
            _write_yaml(config_path, config)
            audit = build_reference_feasibility_report(
                config_path=config_path,
                commands=_audit_commands(
                    config, geometry_scale=arm.geometry_scale
                ),
                stable_start_s=2.0,
                max_support_ratio=0.8,
                max_realization_gap_m=0.01 * arm.geometry_scale,
            )
            audit_path = output_dir / "audits" / f"{arm.name}_seed{seed}.json"
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            audit_path.write_text(
                json.dumps(audit, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            reference_metrics = _audit_metrics(audit)
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
                    "arm": arm.name,
                    "seed": seed,
                    "geometry_scale": arm.geometry_scale,
                    "mass_scale": arm.mass_scale,
                    "config": str(config_path),
                    "scene": str(scene_path),
                    "reference_audit": str(audit_path),
                    "reference_audit_passed": bool(audit["passed"]),
                    "reference_metrics": reference_metrics,
                    "command": command,
                }
            )

    mechanics_factorial = _factorial_effects(jobs)
    manifest = {
        "question": {
            "dimension_effect": "dimension_only - baseline, controlling mass",
            "mass_effect": "mass_only - baseline, controlling dimensions",
            "combined_effect": "dimension_and_mass - baseline",
            "interaction": (
                "(dimension_and_mass - mass_only) - "
                "(dimension_only - baseline)"
            ),
        },
        "interpretation_limit": (
            "Uniformly scaled WR geometry with unchanged HTD actuators is a causal "
            "simulation counterfactual, not a packaging-valid CAD design."
        ),
        "sources": {
            "wr_reference_leg_length_m": WR_REFERENCE_LEG_LENGTH_M,
            "tb_reference_leg_length_m": TB_REFERENCE_LEG_LENGTH_M,
            "wr_model_mass_kg": baseline_model["total_mass_kg"],
            "tb_model_mass_kg": TB_REFERENCE_MASS_KG,
        },
        "geometry_scale": geometry_scale,
        "mass_scale": mass_scale,
        "base_config": str(base_config_path),
        "init_policy": str(init_policy),
        "iterations": args.iterations,
        "model_reports": model_reports,
        "mechanics_factorial": mechanics_factorial,
        "jobs": jobs,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("WR morphology length/mass factorial")
    print(
        f"  length: WR={WR_REFERENCE_LEG_LENGTH_M:.4f}m "
        f"target={args.target_leg_length_m:.4f}m scale={geometry_scale:.6f}"
    )
    print(
        f"  mass: WR={baseline_model['total_mass_kg']:.6f}kg "
        f"target={args.target_mass_kg:.6f}kg scale={mass_scale:.6f}"
    )
    print("  actuator dynamics and 4.413 Nm model limit remain unchanged")
    for job in jobs:
        report = model_reports[job["arm"]]
        primary_ratio = job["reference_metrics"][0]["max_support_ratio_p95"]
        print(
            f"  {job['arm']}: mass={report['total_mass_kg']:.3f}kg "
            f"leg_chain={report['leg_joint_chain_length_m']:.3f}m "
            f"support_ratio_p95={primary_ratio:.3f} "
            f"reference_audit={'PASS' if job['reference_audit_passed'] else 'FAIL'}"
        )
    if mechanics_factorial:
        print(
            "  primary mechanics effects: "
            f"dimension={mechanics_factorial['dimension_effect_relative']:+.1%} "
            f"mass={mechanics_factorial['mass_effect_relative']:+.1%} "
            f"combined={mechanics_factorial['combined_effect_relative']:+.1%}"
        )
    print(f"  manifest={manifest_path}")

    if args.prepare_only:
        print("Prepared only; no training was started.")
        for job in jobs:
            print("  " + " ".join(job["command"]))
        return 0

    for index, job in enumerate(jobs, start=1):
        print(
            f"\n[{index}/{len(jobs)}] training {job['arm']} seed={job['seed']}",
            flush=True,
        )
        completed = subprocess.run(job["command"], cwd=PROJECT_ROOT, check=False)
        if completed.returncode != 0:
            print(
                f"Training stopped: {job['arm']} seed={job['seed']} "
                f"exited with {completed.returncode}",
                file=sys.stderr,
            )
            return int(completed.returncode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
