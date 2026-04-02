#!/usr/bin/env python3
"""Smoke-test walking reference + IK nominal target path (v0.19.2)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from assets.robot_config import load_robot_config
from control.adapters import reference_to_nominal_joint_targets
from control.kinematics import LegIkConfig
from control.references import WalkingRefV1Config, WalkingRefV1Input, WalkingRefV1State, step_reference


@dataclass(frozen=True)
class SmokeSummary:
    steps: int
    stance_switches: int
    mid_swing_clearance_min_m: float
    touchdown_xy_error_max_m: float
    touchdown_z_abs_max_m: float
    min_joint_limit_margin_rad: float
    ik_unreachable_frac: float
    phase_progression_ok: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "stance_switches": self.stance_switches,
            "mid_swing_clearance_min_m": self.mid_swing_clearance_min_m,
            "touchdown_xy_error_max_m": self.touchdown_xy_error_max_m,
            "touchdown_z_abs_max_m": self.touchdown_z_abs_max_m,
            "min_joint_limit_margin_rad": self.min_joint_limit_margin_rad,
            "ik_unreachable_frac": self.ik_unreachable_frac,
            "phase_progression_ok": self.phase_progression_ok,
        }


def _min_joint_margin_rad(q_ref: tuple[float, ...], limits: dict[str, tuple[float, float]], names: list[str]) -> float:
    margins = []
    for value, name in zip(q_ref, names):
        lo, hi = limits.get(name, (-3.14, 3.14))
        margins.append(min(float(value) - lo, hi - float(value)))
    return float(min(margins)) if margins else 0.0


def _active_locomotion_joint_names(names: list[str]) -> list[str]:
    keywords = ("hip", "knee", "ankle", "waist")
    filtered = [name for name in names if any(k in name for k in keywords)]
    return filtered if filtered else names


def run_smoke(
    *,
    robot_config_path: Path,
    steps: int,
    dt_s: float,
    forward_speed_mps: float,
) -> SmokeSummary:
    robot_cfg = load_robot_config(robot_config_path)
    ref_cfg = WalkingRefV1Config()
    ik_cfg = LegIkConfig()
    state = WalkingRefV1State()

    stance_switches = 0
    prev_stance = state.stance_foot_id
    unreachable_count = 0
    mid_swing_clearance_min = 1e9
    touchdown_xy_error_max = 0.0
    touchdown_z_abs_max = 0.0
    min_margin = 1e9
    phase_ok = True
    prev_phase = 0.0

    active_names = _active_locomotion_joint_names(list(robot_cfg.actuator_names))
    for _ in range(steps):
        out = step_reference(
            config=ref_cfg,
            state=state,
            inputs=WalkingRefV1Input(forward_speed_mps=forward_speed_mps),
            dt_s=dt_s,
        )
        state = out.next_state
        ref = out.reference

        phase = max(0.0, min(1.0, state.phase_time_s / ref_cfg.step_time_s))
        if phase + 1e-6 < prev_phase and state.stance_foot_id == prev_stance:
            phase_ok = False
        prev_phase = phase

        if state.stance_foot_id != prev_stance:
            stance_switches += 1
            prev_stance = state.stance_foot_id
            prev_phase = phase

        swing_z = float(ref.desired_swing_foot_position[2])
        if 0.35 <= phase <= 0.65:
            mid_swing_clearance_min = min(mid_swing_clearance_min, swing_z)
        if phase >= 0.95:
            touchdown_z_abs_max = max(touchdown_z_abs_max, abs(swing_z))
            dx = float(ref.desired_swing_foot_position[0] - ref.desired_next_foothold_stance_frame[0])
            dy = float(ref.desired_swing_foot_position[1] - ref.desired_next_foothold_stance_frame[1])
            touchdown_xy_error_max = max(touchdown_xy_error_max, (dx * dx + dy * dy) ** 0.5)
        nominal = reference_to_nominal_joint_targets(
            reference=ref,
            robot_cfg=robot_cfg,
            leg_ik_config=ik_cfg,
        )
        if not nominal.left_leg_reachable or not nominal.right_leg_reachable:
            unreachable_count += 1
        min_margin = min(
            min_margin,
            _min_joint_margin_rad(
                nominal.q_ref,
                robot_cfg.joint_limits,
                active_names,
            ),
        )

    return SmokeSummary(
        steps=steps,
        stance_switches=stance_switches,
        mid_swing_clearance_min_m=float(mid_swing_clearance_min if mid_swing_clearance_min < 1e8 else 0.0),
        touchdown_xy_error_max_m=float(touchdown_xy_error_max),
        touchdown_z_abs_max_m=float(touchdown_z_abs_max),
        min_joint_limit_margin_rad=float(min_margin),
        ik_unreachable_frac=float(unreachable_count / max(1, steps)),
        phase_progression_ok=phase_ok,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reference+IK smoke checks.")
    parser.add_argument("--robot-config", default="assets/v2/mujoco_robot_config.json")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--dt-s", type=float, default=0.02)
    parser.add_argument("--forward-speed-mps", type=float, default=0.12)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    summary = run_smoke(
        robot_config_path=Path(args.robot_config),
        steps=int(args.steps),
        dt_s=float(args.dt_s),
        forward_speed_mps=float(args.forward_speed_mps),
    )
    payload = summary.to_dict()
    print(json.dumps(payload, indent=2))
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
