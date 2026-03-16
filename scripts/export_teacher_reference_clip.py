#!/usr/bin/env python3
"""Export deterministic bootstrap teacher reference clip for v0.16.0."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.configs.training_config import load_robot_config
from training.reference_motion.loader import save_reference_motion_clip
from training.reference_motion.retarget import (
    JointLimits,
    generate_bootstrap_nominal_walk_clip,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robot-config",
        type=str,
        default="assets/v2/mujoco_robot_config.json",
        help="Path to WildRobot robot config JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="training/reference_motion",
        help="Directory to write reference clip files.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="wildrobot_walk_forward_v001",
        help="Reference clip base name (without extension).",
    )
    parser.add_argument("--frame-count", type=int, default=72)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--nominal-speed", type=float, default=0.10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    robot = load_robot_config(args.robot_config)
    actuator_names = list(robot.actuator_names)
    lower = np.asarray([joint["range"][0] for joint in robot.actuated_joints], dtype=np.float32)
    upper = np.asarray([joint["range"][1] for joint in robot.actuated_joints], dtype=np.float32)

    clip = generate_bootstrap_nominal_walk_clip(
        actuator_names=actuator_names,
        joint_limits=JointLimits(lower=lower, upper=upper),
        frame_count=args.frame_count,
        dt=args.dt,
        nominal_forward_velocity_mps=args.nominal_speed,
        name=args.name,
    )

    output_dir = Path(args.output_dir)
    npz_path = output_dir / f"{args.name}.npz"
    json_path = output_dir / f"{args.name}.json"
    save_reference_motion_clip(clip, npz_path=npz_path, metadata_path=json_path)

    print("Exported bootstrap teacher reference clip:")
    print(f"  npz: {npz_path}")
    print(f"  json: {json_path}")
    print(f"  frames: {clip.frame_count}, dt: {clip.dt}, actuators: {clip.actuator_dim}")


if __name__ == "__main__":
    main()
