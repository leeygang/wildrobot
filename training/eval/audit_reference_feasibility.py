#!/usr/bin/env python3
"""Audit planned versus MJCF-realized lateral walking geometry.

The ZMP planner needs two independent lateral dimensions:

* ``default_stance_width_m`` is the per-side footstep target.
* ``hip_lateral_offset_m`` is the root-to-hip-roll offset used by leg IK.

Treating them as the same number can produce a plausible planner trajectory
whose fixed-base FK realization has a materially different support lever.  The
report below makes that mismatch explicit before another PPO run is launched.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from assets.robot_config import RobotConfig
from control.zmp.zmp_walk import ZMPWalkGenerator
from training.configs.training_config import load_training_config
from training.configs.zmp_reference import zmp_walk_config_from_env
from training.utils.ctrl_order import CtrlOrderMapper


def _resolve_project_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved.resolve()


def _named_id(model: mujoco.MjModel, object_type: Any, name: str) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise ValueError(f"MuJoCo object not found: {name}")
    return int(object_id)


def _distribution(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("reference audit received an empty sample set")
    return {
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _command_key(command: Sequence[float]) -> tuple[float, float, float]:
    values = tuple(float(value) for value in command)
    if len(values) != 3 or not np.all(np.isfinite(values)):
        raise ValueError(
            f"velocity command must contain three finite values: {command}"
        )
    return values


def _configured_commands(training_config: Any) -> list[tuple[float, float, float]]:
    commands = [_command_key(training_config.env.eval_velocity_cmd)]
    commands.extend(
        _command_key(command)
        for command in training_config.env.eval_velocity_cmd_probes
    )
    return list(
        dict.fromkeys(
            command for command in commands if np.linalg.norm(command) > 0
        )
    )


def _reference_coupling(training_config: Any) -> dict[str, Any]:
    residual_base = str(training_config.env.loc_ref_residual_base)
    pose_anchor = str(training_config.env.loc_ref_penalty_pose_anchor)
    pose_weight = float(training_config.reward_weights.penalty_pose)
    ref_q_weight = float(training_config.reward_weights.ref_q_track)
    roll_base = bool(training_config.env.loc_ref_walking_base_from_ref_init_roll)
    direct = (
        residual_base in {"q_ref", "ref_init"}
        or roll_base
        or (pose_anchor == "q_ref" and pose_weight != 0.0)
        or ref_q_weight != 0.0
    )
    return {
        "residual_base": residual_base,
        "penalty_pose_anchor": pose_anchor,
        "penalty_pose_weight": pose_weight,
        "ref_q_track_weight": ref_q_weight,
        "walking_base_from_ref_init_roll": roll_base,
        "critic_imitation_refs": bool(training_config.env.critic_imitation_refs),
        "joint_reference_directly_coupled_to_actor_or_reward": direct,
        "joint_reference_is_critic_only": bool(
            training_config.env.critic_imitation_refs and not direct
        ),
    }


def _load_model_inputs(training_config: Any) -> dict[str, Any]:
    scene_path = _resolve_project_path(training_config.env.scene_xml_path)
    robot_config_path = _resolve_project_path(
        training_config.env.robot_config_path
    )
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    robot_config = RobotConfig.from_file(robot_config_path)
    data = mujoco.MjData(model)
    home_key = _named_id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, home_key)
    mujoco.mj_forward(model, data)

    root_body_id = _named_id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        str(robot_config.floating_base_body),
    )
    root_lateral = np.asarray(data.xmat[root_body_id]).reshape(3, 3)[:, 1]
    root_position = np.asarray(data.xpos[root_body_id])
    hip_offsets = {}
    for side in ("left", "right"):
        joint_id = _named_id(
            model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_hip_roll"
        )
        relative = np.asarray(data.xanchor[joint_id]) - root_position
        hip_offsets[side] = abs(float(np.dot(relative, root_lateral)))

    actuator_names = [
        str(item["name"]) for item in robot_config.actuated_joints
    ]
    mapper = CtrlOrderMapper(model, actuator_names)
    actuator_qpos = np.asarray(
        [model.jnt_qposadr[model.actuator_trnid[k, 0]] for k in range(model.nu)],
        dtype=np.int32,
    )
    policy_to_qpos = actuator_qpos[mapper.policy_to_mj_order]
    foot_geom_ids = {
        "left": [
            _named_id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in robot_config.feet_left_geoms
        ],
        "right": [
            _named_id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in robot_config.feet_right_geoms
        ],
    }
    return {
        "model": model,
        "data": data,
        "home_qpos": data.qpos.copy(),
        "root_body_id": root_body_id,
        "hip_offsets": hip_offsets,
        "policy_to_qpos": policy_to_qpos,
        "foot_geom_ids": foot_geom_ids,
        "actuator_names": actuator_names,
    }


def _realize_reference(
    *,
    trajectory: Any,
    model_inputs: dict[str, Any],
) -> dict[str, np.ndarray]:
    model = model_inputs["model"]
    data = model_inputs["data"]
    home_qpos = model_inputs["home_qpos"]
    policy_to_qpos = model_inputs["policy_to_qpos"]
    root_body_id = model_inputs["root_body_id"]
    foot_geom_ids = model_inputs["foot_geom_ids"]

    n_steps = int(trajectory.n_steps)
    whole_body_com = np.zeros((n_steps, 3), dtype=np.float64)
    foot_centers = {
        "left": np.zeros((n_steps, 3), dtype=np.float64),
        "right": np.zeros((n_steps, 3), dtype=np.float64),
    }
    lateral_axes = np.zeros((n_steps, 3), dtype=np.float64)
    for index in range(n_steps):
        data.qpos[:] = home_qpos
        data.qpos[policy_to_qpos] = trajectory.q_ref[index]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        whole_body_com[index] = data.subtree_com[root_body_id]
        lateral_axes[index] = np.asarray(data.xmat[root_body_id]).reshape(3, 3)[:, 1]
        for side in ("left", "right"):
            foot_centers[side][index] = np.mean(
                data.geom_xpos[foot_geom_ids[side]], axis=0
            )

    return {
        "whole_body_com": whole_body_com,
        "left_foot": foot_centers["left"],
        "right_foot": foot_centers["right"],
        "lateral_axes": lateral_axes,
    }


def _support_report(
    *,
    side: str,
    support_mask: np.ndarray,
    planned_foot_position: np.ndarray,
    realized: dict[str, np.ndarray],
    robot_weight_n: float,
    torque_limit_nm: float,
    max_realization_gap_m: float,
    max_support_ratio: float,
) -> dict[str, Any]:
    if not np.any(support_mask):
        raise ValueError(f"reference contains no {side}-only support samples")
    planned_signed = -np.asarray(planned_foot_position[support_mask, 1])
    realized_delta = (
        realized["whole_body_com"][support_mask]
        - realized[f"{side}_foot"][support_mask]
    )
    realized_signed = np.sum(
        realized_delta * realized["lateral_axes"][support_mask], axis=1
    )
    planned_abs = np.abs(planned_signed)
    realized_abs = np.abs(realized_signed)
    gap_abs = np.abs(realized_abs - planned_abs)
    ratio = realized_abs * robot_weight_n / torque_limit_nm
    gates = {
        "realization_gap_p95": float(np.percentile(gap_abs, 95))
        <= max_realization_gap_m,
        "quasi_static_support_ratio_p95": float(np.percentile(ratio, 95))
        <= max_support_ratio,
    }
    return {
        "sample_count": int(np.sum(support_mask)),
        "planned_lateral_lever_abs_m": _distribution(planned_abs),
        "realized_lateral_lever_abs_m": _distribution(realized_abs),
        "realization_gap_abs_m": _distribution(gap_abs),
        "quasi_static_support_ratio": _distribution(ratio),
        "gates": gates,
        "passed": all(gates.values()),
    }


def build_reference_feasibility_report(
    *,
    config_path: Path,
    commands: Sequence[Sequence[float]] | None = None,
    stable_start_s: float = 2.0,
    max_support_ratio: float = 0.8,
    max_realization_gap_m: float = 0.01,
) -> dict[str, Any]:
    """Build a deterministic planner/IK/FK consistency report."""
    if stable_start_s < 0.0:
        raise ValueError("stable_start_s must be non-negative")
    if max_support_ratio <= 0.0:
        raise ValueError("max_support_ratio must be positive")
    if max_realization_gap_m < 0.0:
        raise ValueError("max_realization_gap_m must be non-negative")

    training_config = load_training_config(config_path)
    command_list = (
        [_command_key(command) for command in commands]
        if commands is not None
        else _configured_commands(training_config)
    )
    if any(np.linalg.norm(command) <= 0.0 for command in command_list):
        raise ValueError("reference audit commands must be non-zero walking commands")
    if not command_list:
        raise ValueError("reference audit requires at least one non-zero command")

    zmp_config = zmp_walk_config_from_env(
        training_config.env,
        offline_library_path=training_config.env.loc_ref_offline_library_path,
    )
    model_inputs = _load_model_inputs(training_config)
    model = model_inputs["model"]
    root_body_id = model_inputs["root_body_id"]
    robot_weight_n = float(model.body_subtreemass[root_body_id]) * float(
        np.linalg.norm(model.opt.gravity)
    )
    configured_limit = training_config.env.actuator_force_limit_nm
    if configured_limit is None:
        torque_limits = {}
        for side in ("left", "right"):
            actuator_id = _named_id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{side}_hip_roll"
            )
            torque_limits[side] = float(
                np.max(np.abs(model.actuator_forcerange[actuator_id]))
            )
    else:
        torque_limits = {side: float(configured_limit) for side in ("left", "right")}

    measured_hip_offset = float(np.mean(list(model_inputs["hip_offsets"].values())))
    hip_offset_gap = abs(measured_hip_offset - zmp_config.hip_lateral_offset_m)
    close_feet_threshold = float(training_config.env.close_feet_threshold)
    command_reports = []
    generator = ZMPWalkGenerator(
        config=zmp_config,
        scene_xml_path=training_config.env.scene_xml_path,
        robot_config_path=training_config.env.robot_config_path,
    )
    for command in command_list:
        trajectory = generator.generate(*command)
        elapsed = np.arange(trajectory.n_steps, dtype=np.float64) * trajectory.dt
        stable_mask = elapsed >= stable_start_s
        contacts = np.asarray(trajectory.contact_mask) > 0.5
        realized = _realize_reference(
            trajectory=trajectory,
            model_inputs=model_inputs,
        )
        planned_separation = np.abs(
            np.asarray(trajectory.left_foot_pos[:, 1])
            - np.asarray(trajectory.right_foot_pos[:, 1])
        )[stable_mask]
        realized_separation = np.abs(
            np.sum(
                (realized["left_foot"] - realized["right_foot"])
                * realized["lateral_axes"],
                axis=1,
            )
        )[stable_mask]
        support = {}
        for side, contact_index, foot_position in (
            ("left", 0, trajectory.left_foot_pos),
            ("right", 1, trajectory.right_foot_pos),
        ):
            support_mask = (
                stable_mask
                & contacts[:, contact_index]
                & ~contacts[:, 1 - contact_index]
            )
            support[side] = _support_report(
                side=side,
                support_mask=support_mask,
                planned_foot_position=np.asarray(foot_position),
                realized=realized,
                robot_weight_n=robot_weight_n,
                torque_limit_nm=torque_limits[side],
                max_realization_gap_m=max_realization_gap_m,
                max_support_ratio=max_support_ratio,
            )
        gates = {
            "planned_foot_separation": float(np.min(planned_separation))
            >= close_feet_threshold,
            "realized_foot_separation": float(np.min(realized_separation))
            >= close_feet_threshold,
            "left_support": support["left"]["passed"],
            "right_support": support["right"]["passed"],
        }
        command_reports.append(
            {
                "command": list(command),
                "n_steps": int(trajectory.n_steps),
                "stable_start_s": float(stable_start_s),
                "planned_foot_separation_m": _distribution(planned_separation),
                "realized_foot_separation_m": _distribution(realized_separation),
                "support": support,
                "gates": gates,
                "passed": all(gates.values()),
            }
        )

    morphology_gates = {
        "hip_lateral_offset_matches_mjcf": hip_offset_gap
        <= max_realization_gap_m,
    }
    return {
        "config": str(config_path.resolve()),
        "thresholds": {
            "close_feet_threshold_m": close_feet_threshold,
            "max_support_ratio": float(max_support_ratio),
            "max_realization_gap_m": float(max_realization_gap_m),
        },
        "morphology": {
            "configured_hip_lateral_offset_m": float(
                zmp_config.hip_lateral_offset_m
            ),
            "mjcf_hip_lateral_offset_m": measured_hip_offset,
            "mjcf_hip_lateral_offset_by_side_m": model_inputs["hip_offsets"],
            "hip_lateral_offset_gap_m": hip_offset_gap,
            "configured_stance_half_width_m": float(
                zmp_config.default_stance_width_m
            ),
            "robot_weight_n": robot_weight_n,
            "hip_roll_torque_limit_nm": torque_limits,
            "gates": morphology_gates,
        },
        "reference_coupling": _reference_coupling(training_config),
        "commands": command_reports,
        "passed": all(morphology_gates.values())
        and all(command["passed"] for command in command_reports),
    }


def _print_report(report: dict[str, Any]) -> None:
    morphology = report["morphology"]
    print("Reference feasibility audit")
    print(f"  config: {report['config']}")
    print(
        "  hip lateral offset: "
        f"configured={morphology['configured_hip_lateral_offset_m']:.4f}m "
        f"MJCF={morphology['mjcf_hip_lateral_offset_m']:.4f}m "
        f"gap={morphology['hip_lateral_offset_gap_m'] * 1000:.1f}mm"
    )
    coupling = report["reference_coupling"]
    print(
        "  joint-reference coupling: "
        + (
            "actor/reward"
            if coupling["joint_reference_directly_coupled_to_actor_or_reward"]
            else "critic-only"
        )
    )
    print(
        "  cmd(vx,vy,wz) side planned_mm realized_mm gap_p95_mm "
        "support_ratio_p95 result"
    )
    for command_report in report["commands"]:
        command_text = ",".join(f"{value:+.3f}" for value in command_report["command"])
        for side in ("left", "right"):
            support = command_report["support"][side]
            print(
                f"  ({command_text}) {side:5s} "
                f"{support['planned_lateral_lever_abs_m']['mean'] * 1000:10.1f} "
                f"{support['realized_lateral_lever_abs_m']['mean'] * 1000:11.1f} "
                f"{support['realization_gap_abs_m']['p95'] * 1000:10.1f} "
                f"{support['quasi_static_support_ratio']['p95']:17.3f} "
                f"{'PASS' if support['passed'] else 'FAIL'}"
            )
        planned_mean = command_report["planned_foot_separation_m"]["mean"]
        realized_mean = command_report["realized_foot_separation_m"]["mean"]
        print(
            "    foot separation mean: "
            f"planned={planned_mean * 1000:.1f}mm "
            f"realized={realized_mean * 1000:.1f}mm"
        )
    print(f"Result: {'PASS' if report['passed'] else 'FAIL'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit ZMP planner versus MJCF-realized lateral geometry",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--velocity-cmd",
        nargs=3,
        type=float,
        action="append",
        metavar=("VX", "VY", "WZ"),
        help="Command to audit; repeatable. Defaults to config primary and probes.",
    )
    parser.add_argument("--stable-start-s", type=float, default=2.0)
    parser.add_argument("--max-support-ratio", type=float, default=0.8)
    parser.add_argument("--max-realization-gap-m", type=float, default=0.01)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = _resolve_project_path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    report = build_reference_feasibility_report(
        config_path=config_path,
        commands=args.velocity_cmd,
        stable_start_s=float(args.stable_start_s),
        max_support_ratio=float(args.max_support_ratio),
        max_realization_gap_m=float(args.max_realization_gap_m),
    )
    _print_report(report)
    if args.output is not None:
        output_path = _resolve_project_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {output_path}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
