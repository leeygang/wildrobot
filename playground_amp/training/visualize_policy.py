#!/usr/bin/env python3
"""Visualize trained WildRobot policy in MuJoCo viewer.

This script loads a trained checkpoint and runs the policy in real-time
using MuJoCo's native viewer.

Usage:
    # On macOS, use mjpython (required for viewer):
    uv run mjpython playground_amp/training/visualize_policy.py

    # Or use headless mode to record video without display:
    uv run python playground_amp/training/visualize_policy.py --headless --record output.mp4

    # Visualize specific checkpoint
    mjpython playground_amp/training/visualize_policy.py --checkpoint path/to/checkpoint.pkl

    # Visualize with different config
    mjpython playground_amp/training/visualize_policy.py --config playground_amp/configs/ppo_walking.yaml

    # With velocity command (for walking)
    mjpython playground_amp/training/visualize_policy.py --velocity-cmd 0.5

Controls (viewer mode):
    - Space: Pause/Resume
    - Backspace: Reset environment
    - Escape: Exit
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np

# Add project root to path (training/ -> playground_amp/ -> project_root/)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from assets.robot_config import get_robot_config
from policy_contract.numpy.action import postprocess_action
from policy_contract.calib import NumpyCalibOps
from policy_contract.numpy.obs import build_observation
from policy_contract.numpy.signals import Signals
from policy_contract.numpy.state import PolicyState
from policy_contract.spec import (
    ActionSpec,
    JointSpec,
    ModelSpec,
    ObservationSpec,
    ObsFieldSpec,
    PolicySpec,
    RobotSpec,
)
from playground_amp.cal.cal import ControlAbstractionLayer
from playground_amp.cal.specs import CoordinateFrame, Pose3D
from playground_amp.configs.training_config import (
    load_robot_config,
    load_training_config,
)
from playground_amp.training.ppo_core import create_networks, sample_actions


# Default paths (relative to project_root)
DEFAULT_CONFIG_PATH = project_root / "playground_amp" / "configs" / "ppo_walking.yaml"
DEFAULT_CHECKPOINT_PATH = (
    project_root / "playground_amp" / "checkpoints" / "final_ppo_policy.pkl"
)
DEFAULT_ROBOT_CONFIG_PATH = project_root / "assets" / "robot_config.yaml"


@dataclass(frozen=True)
class PushSchedule:
    start_step: int
    end_step: int
    force_xy: np.ndarray  # shape (2,)


def _read_sensor(mj_model: mujoco.MjModel, mj_data: mujoco.MjData, name: str, fallback: np.ndarray) -> np.ndarray:
    sensor_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if sensor_id < 0:
        return np.asarray(fallback, dtype=np.float32)
    adr = int(mj_model.sensor_adr[sensor_id])
    dim = int(mj_model.sensor_dim[sensor_id])
    return np.asarray(mj_data.sensordata[adr : adr + dim], dtype=np.float32)


def _build_policy_spec(training_cfg, robot_cfg, action_filter_alpha: float) -> PolicySpec:
    action_dim = robot_cfg.action_dim
    layout = [
        ObsFieldSpec(name="gravity_local", size=3, frame="local", units="unit_vector"),
        ObsFieldSpec(name="angvel_heading_local", size=3, frame="heading_local", units="rad_s"),
        ObsFieldSpec(name="joint_pos_normalized", size=action_dim, units="normalized_-1_1"),
        ObsFieldSpec(name="joint_vel_normalized", size=action_dim, units="normalized_-1_1"),
        ObsFieldSpec(name="foot_switches", size=4, units="bool_as_float"),
        ObsFieldSpec(name="prev_action", size=action_dim, units="normalized_-1_1"),
        ObsFieldSpec(name="velocity_cmd", size=1, units="m_s"),
        ObsFieldSpec(name="padding", size=1, units="unused"),
    ]

    joints = {}
    for item in robot_cfg.actuated_joints:
        name = str(item.get("name"))
        rng = item.get("range") or [0.0, 0.0]
        joints[name] = JointSpec(
            range_min_rad=float(rng[0]),
            range_max_rad=float(rng[1]),
            mirror_sign=float(item.get("mirror_sign", 1.0)),
            max_velocity_rad_s=float(item.get("max_velocity", 10.0)),
        )

    postprocess_id = "lowpass_v1" if action_filter_alpha > 0.0 else "none"
    postprocess_params = {}
    if postprocess_id == "lowpass_v1":
        postprocess_params["alpha"] = float(action_filter_alpha)

    return PolicySpec(
        contract_name="wildrobot_policy",
        contract_version="1.0.0",
        spec_version=1,
        model=ModelSpec(
            format="onnx",
            input_name="observation",
            output_name="action",
            dtype="float32",
            obs_dim=sum(field.size for field in layout),
            action_dim=action_dim,
        ),
        robot=RobotSpec(
            robot_name=robot_cfg.robot_name,
            actuator_names=list(robot_cfg.actuator_names),
            joints=joints,
        ),
        observation=ObservationSpec(
            dtype="float32",
            layout_id="wr_obs_v1",
            layout=layout,
        ),
        action=ActionSpec(
            dtype="float32",
            bounds={"min": -1.0, "max": 1.0},
            postprocess_id=postprocess_id,
            postprocess_params=postprocess_params,
            mapping_id="pos_target_rad_v1",
        ),
        provenance={
            "training_config": str(training_cfg.config_path) if hasattr(training_cfg, "config_path") else "<runtime>",
        },
    )


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize trained WildRobot policy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file (.pkl)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use deterministic policy (no sampling noise)",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy (with sampling noise)",
    )
    parser.add_argument(
        "--velocity-cmd",
        type=float,
        default=None,
        help="Fixed velocity command (default: random from config range)",
    )
    parser.add_argument(
        "--fixed-velocity",
        type=float,
        default=None,
        help="Alias for --velocity-cmd (fixed command for every episode)",
    )
    parser.add_argument(
        "--no-reset-noise",
        action="store_true",
        help="Disable joint noise on reset (less faithful, more repeatable)",
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Disable push disturbances (overrides config push_enabled)",
    )
    parser.add_argument(
        "--action-filter-alpha",
        type=float,
        default=None,
        help="Override env.action_filter_alpha from the config (0 disables filtering)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Deterministic, fixed-velocity, no-noise demo mode",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier (1.0 = real-time)",
    )
    parser.add_argument(
        "--record",
        type=str,
        default=None,
        help="Path to save video recording (e.g., output.mp4)",
    )
    parser.add_argument(
        "--record-duration",
        type=float,
        default=10.0,
        help="Duration of recording in seconds",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without viewer (for recording or batch evaluation)",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=None,
        help="Number of episodes to run (headless mode). Default: 1 for headless, unlimited for viewer",
    )
    return parser.parse_args()


def resolve_checkpoint_path(checkpoint_arg: str | None) -> Path | None:
    """Resolve checkpoint path, trying multiple locations.

    Tries:
    1. Absolute path as-is
    2. Relative to current working directory
    3. Relative to project root
    4. Default checkpoint path

    Args:
        checkpoint_arg: Path from command line or None

    Returns:
        Resolved Path if found, None otherwise
    """
    if checkpoint_arg is None:
        # Use default
        if DEFAULT_CHECKPOINT_PATH.exists():
            return DEFAULT_CHECKPOINT_PATH
        return None

    path = Path(checkpoint_arg)

    # Try as-is (handles absolute paths and correct relative paths)
    if path.exists():
        return path.resolve()

    # Try relative to project root
    project_relative = project_root / path
    if project_relative.exists():
        return project_relative.resolve()

    # Try relative to project root with playground_amp prefix stripped/added
    if str(path).startswith("playground_amp/"):
        without_prefix = project_root / Path(
            str(path).replace("playground_amp/", "", 1)
        )
        if without_prefix.exists():
            return without_prefix.resolve()
    else:
        with_prefix = project_root / "playground_amp" / path
        if with_prefix.exists():
            return with_prefix.resolve()

    return None


def list_available_checkpoints() -> list[Path]:
    """Find all available checkpoint files."""
    checkpoints = []

    # Search in common locations
    search_paths = [
        project_root / "playground_amp" / "checkpoints",
        project_root / "checkpoints",
        Path.cwd() / "playground_amp" / "checkpoints",
        Path.cwd() / "checkpoints",
    ]

    for search_path in search_paths:
        if search_path.exists():
            checkpoints.extend(search_path.glob("**/*.pkl"))

    # Deduplicate by resolving to absolute paths
    seen = set()
    unique = []
    for p in checkpoints:
        resolved = p.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(p)

    return sorted(unique, key=lambda p: p.stat().st_mtime, reverse=True)


def main():
    """Main visualization loop."""
    args = parse_args()

    # Determine deterministic mode
    deterministic = not args.stochastic
    if args.demo:
        deterministic = True

    # Load robot config
    if DEFAULT_ROBOT_CONFIG_PATH.exists():
        robot_cfg = load_robot_config(DEFAULT_ROBOT_CONFIG_PATH)
        print(f"Loaded robot config: {robot_cfg.robot_name}")
    else:
        print(f"Error: Robot config not found at {DEFAULT_ROBOT_CONFIG_PATH}")
        print("Run 'cd assets && python post_process.py' to generate it.")
        return 1

    # Load training config
    config_path = Path(args.config) if args.config else DEFAULT_CONFIG_PATH
    print(f"Loading config from: {config_path}")
    training_cfg = load_training_config(config_path)

    # Resolve checkpoint path (tries multiple locations)
    checkpoint_path = resolve_checkpoint_path(args.checkpoint)

    if checkpoint_path is None:
        print(f"Error: Checkpoint not found: {args.checkpoint or 'default'}")
        print("\nSearched locations:")
        if args.checkpoint:
            print(f"  - {args.checkpoint} (as provided)")
            print(f"  - {project_root / args.checkpoint}")
        else:
            print(f"  - {DEFAULT_CHECKPOINT_PATH}")

        available = list_available_checkpoints()
        if available:
            print(f"\nAvailable checkpoints ({len(available)} found):")
            for p in available[:10]:  # Show top 10 most recent
                try:
                    mtime = p.stat().st_mtime
                    from datetime import datetime

                    mtime_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
                except:
                    mtime_str = "unknown"
                print(f"  - {p} ({mtime_str})")
            if len(available) > 10:
                print(f"  ... and {len(available) - 10} more")
        else:
            print("\nNo checkpoints found in common locations.")
        return 1

    print(f"Loading checkpoint from: {checkpoint_path}")

    with open(checkpoint_path, "rb") as f:
        checkpoint = pickle.load(f)

    print(f"Checkpoint mode: {checkpoint.get('mode', 'unknown')}")

    # Load MuJoCo model for native simulation (FAST on CPU)
    model_path = project_root / training_cfg.env.model_path
    print(f"Loading model from: {model_path}")

    mj_model = mujoco.MjModel.from_xml_path(str(model_path))
    mj_data = mujoco.MjData(mj_model)

    # Create Control Abstraction Layer (CAL) for action/observation transforms
    # This is CRITICAL for training/inference parity (v0.11.0+)
    cal = ControlAbstractionLayer(mj_model, robot_cfg)
    print(f"  CAL initialized with {cal.num_actuators} actuators")

    # Action filtering configuration (match training unless overridden)
    action_filter_alpha = (
        float(args.action_filter_alpha)
        if args.action_filter_alpha is not None
        else float(training_cfg.env.action_filter_alpha)
    )

    policy_spec = _build_policy_spec(training_cfg, robot_cfg, action_filter_alpha)

    # Get dimensions from model/config
    obs_dim = policy_spec.model.obs_dim
    action_dim = policy_spec.model.action_dim

    print(f"Native MuJoCo: obs_dim={obs_dim}, action_dim={action_dim}")

    # Create PPO networks with same architecture as training
    policy_hidden = tuple(training_cfg.networks.actor.hidden_sizes)
    value_hidden = tuple(training_cfg.networks.critic.hidden_sizes)

    ppo_network = create_networks(
        obs_dim=obs_dim,
        action_dim=action_dim,
        policy_hidden_dims=policy_hidden,
        value_hidden_dims=value_hidden,
    )

    # Extract policy params from checkpoint
    # Brax expects policy_params = {'params': {...}} format
    policy_params = checkpoint["policy_params"]

    processor_params = checkpoint.get("processor_params", ())
    if processor_params is None:
        processor_params = ()

    # JIT compile the action sampling function (this is fast, ~1ms)
    @jax.jit
    def get_action(obs, rng):
        """Get action from policy."""
        obs_batch = obs[None, ...]  # Add batch dimension
        action, _, _ = sample_actions(
            processor_params, policy_params, ppo_network, obs_batch, rng, deterministic
        )
        return action[0]  # Remove batch dimension

    # Warmup JIT compilation for policy network
    print("Warming up policy network JIT compilation...")
    dummy_obs = jnp.zeros(obs_dim)
    rng = jax.random.PRNGKey(42)
    rng, warmup_rng = jax.random.split(rng)
    _ = get_action(dummy_obs, warmup_rng)
    print("JIT warmup complete.")

    # Physics substeps per control step
    ctrl_dt = training_cfg.env.ctrl_dt
    sim_dt = mj_model.opt.timestep
    n_substeps = int(ctrl_dt / sim_dt)
    print(f"Control dt: {ctrl_dt}s, Sim dt: {sim_dt}s, Substeps: {n_substeps}")

    def get_forward_velocity(mj_data, prev_root_pos, prev_root_quat, dt):
        """Get forward velocity in heading-local frame (matches training).

        Uses Pose3D finite-difference for consistency with observation computation.
        Returns x-component (forward direction in heading-local frame).
        """
        if prev_root_pos is None or prev_root_quat is None:
            # Fallback to qvel-based velocity for first frame (training parity)
            return float(
                cal.get_root_velocity(
                    mj_data, frame=CoordinateFrame.HEADING_LOCAL
                ).linear[0]
            )

        curr_pose = Pose3D.from_numpy(
            position=mj_data.qpos[0:3].copy(),
            orientation=mj_data.qpos[3:7].copy(),
            frame=CoordinateFrame.WORLD,
        )
        prev_pose = Pose3D.from_numpy(
            position=prev_root_pos,
            orientation=prev_root_quat,
            frame=CoordinateFrame.WORLD,
        )
        linvel = curr_pose.linvel_fd(prev_pose, dt, frame=CoordinateFrame.HEADING_LOCAL)
        return float(linvel[0])  # x = forward in heading-local

    # Note: Joint addresses now handled by CAL (get_joint_positions/velocities)

    # Track previous action (training parity).
    # In training, the initial observation's action slice is the default pose action,
    # not zeros. Using zeros can destabilize early steps due to action-filter history
    # and observation mismatch.
    default_ctrl = np.array(cal.get_ctrl_for_default_pose(), dtype=np.float32)
    default_policy_action = np.array(
        NumpyCalibOps.ctrl_to_policy_action(spec=policy_spec, ctrl_rad=default_ctrl),
        dtype=np.float32,
    )
    prev_action = default_policy_action.copy()

    def get_observation(mj_data, velocity_cmd, prev_action_in):
        """Compute observation via policy_contract (hardware-aligned)."""
        quat = _read_sensor(
            mj_model, mj_data, robot_cfg.orientation_sensor, fallback=mj_data.qpos[3:7]
        )
        gyro = _read_sensor(
            mj_model, mj_data, robot_cfg.root_gyro_sensor, fallback=mj_data.qvel[3:6]
        )

        joint_pos = mj_data.qpos[joint_qpos_idx]
        joint_vel = mj_data.qvel[joint_qvel_idx]
        foot_switches = np.zeros((4,), dtype=np.float32)

        signals = Signals(
            quat_xyzw=np.asarray(quat, dtype=np.float32),
            gyro_rad_s=np.asarray(gyro, dtype=np.float32),
            joint_pos_rad=np.asarray(joint_pos, dtype=np.float32),
            joint_vel_rad_s=np.asarray(joint_vel, dtype=np.float32),
            foot_switches=np.asarray(foot_switches, dtype=np.float32),
        )

        obs = build_observation(
            spec=policy_spec,
            state=PolicyState(prev_action=np.asarray(prev_action_in, dtype=np.float32)),
            signals=signals,
            velocity_cmd=np.array(velocity_cmd, dtype=np.float32),
        )

        return obs

    use_action_filter = action_filter_alpha > 0
    print(f"Action filter: enabled={use_action_filter}, alpha={action_filter_alpha}")

    def apply_action(mj_data, action, prev_action_for_filter):
        """Apply action to MuJoCo model via policy_contract semantics."""
        action_np = np.array(action)
        state = PolicyState(prev_action=np.asarray(prev_action_for_filter, dtype=np.float32))
        filtered_action, _ = postprocess_action(
            spec=policy_spec,
            state=state,
            action_raw=action_np,
        )
        ctrl = NumpyCalibOps.action_to_ctrl(spec=policy_spec, action=filtered_action)
        mj_data.ctrl[:] = np.array(ctrl)
        return filtered_action

    def step_physics(mj_model, mj_data, n_substeps):
        """Step physics simulation for one control step."""
        for _ in range(n_substeps):
            mujoco.mj_step(mj_model, mj_data)

    rng_env = np.random.default_rng()

    def sample_velocity_cmd():
        return float(
            rng_env.uniform(
                training_cfg.env.min_velocity, training_cfg.env.max_velocity
            )
        )

    def reset_robot(mj_model, mj_data, apply_noise=True):
        """Reset robot to initial state (training parity)."""
        if mj_model.nkey > 0:
            mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
        else:
            mujoco.mj_resetData(mj_model, mj_data)
        qpos = (
            mj_model.key_qpos[0].copy() if mj_model.nkey > 0 else mj_model.qpos0.copy()
        )
        qvel = np.zeros(mj_model.nv, dtype=np.float32)

        if apply_noise:
            joint_noise = rng_env.uniform(-0.05, 0.05, size=action_dim)
            qpos[7 : 7 + action_dim] = qpos[7 : 7 + action_dim] + joint_noise

        mj_data.qpos[:] = qpos
        mj_data.qvel[:] = qvel
        if hasattr(mj_data, "xfrc_applied"):
            mj_data.xfrc_applied[:] = 0.0
        # Use CAL to get default pose ctrl (physical joint angles from keyframe)
        mj_data.ctrl[:] = np.array(cal.get_ctrl_for_default_pose())
        mujoco.mj_forward(mj_model, mj_data)

    def check_termination(mj_data, step_count):
        """Check if episode should terminate (training parity)."""
        height = mj_data.qpos[2]
        height_too_low = height < training_cfg.env.min_height
        height_too_high = height > training_cfg.env.max_height
        # Use Pose3D factory pattern for Euler angle extraction
        pose = Pose3D.from_numpy(
            position=mj_data.qpos[0:3].copy(),
            orientation=mj_data.qpos[3:7].copy(),
            frame=CoordinateFrame.WORLD,
        )
        roll, pitch, _ = pose.euler_angles()
        pitch_fail = abs(float(pitch)) > training_cfg.env.max_pitch
        roll_fail = abs(float(roll)) > training_cfg.env.max_roll
        terminated = height_too_low or height_too_high or pitch_fail or roll_fail
        truncated = (
            step_count >= training_cfg.env.max_episode_steps
        ) and not terminated
        return terminated or truncated

    def get_yaw(mj_data):
        pose = Pose3D.from_numpy(
            position=mj_data.qpos[0:3].copy(),
            orientation=mj_data.qpos[3:7].copy(),
            frame=CoordinateFrame.WORLD,
        )
        _, _, yaw = pose.euler_angles()
        return float(yaw)

    def wrap_angle(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    fixed_velocity = (
        args.fixed_velocity if args.fixed_velocity is not None else args.velocity_cmd
    )
    if args.demo and fixed_velocity is None:
        fixed_velocity = (
            training_cfg.env.min_velocity + training_cfg.env.max_velocity
        ) / 2

    if fixed_velocity is not None:
        velocity_cmd = fixed_velocity
        print(f"Fixed velocity command: {velocity_cmd:.2f} m/s")
    else:
        velocity_cmd = sample_velocity_cmd()
        print(f"Sampled velocity command: {velocity_cmd:.2f} m/s")

    apply_reset_noise = not args.no_reset_noise and not args.demo
    reset_robot(mj_model, mj_data, apply_noise=apply_reset_noise)
    prev_action = default_policy_action.copy()
    episode_start_pos = mj_data.qpos[0:3].copy()
    episode_start_yaw = get_yaw(mj_data)
    left_hip_pitch_idx, right_hip_pitch_idx = cal._robot_config.get_hip_pitch_indices()
    left_knee_pitch_idx, right_knee_pitch_idx = (
        cal._robot_config.get_knee_pitch_indices()
    )
    left_hip_roll_idx = cal._robot_config.get_actuator_index("left_hip_roll")
    right_hip_roll_idx = cal._robot_config.get_actuator_index("right_hip_roll")
    left_ankle_pitch_idx = cal._robot_config.get_actuator_index("left_ankle_pitch")
    right_ankle_pitch_idx = cal._robot_config.get_actuator_index("right_ankle_pitch")

    # Control timing
    ctrl_dt = training_cfg.env.ctrl_dt

    # Determine mode
    headless = args.headless
    is_macos = sys.platform == "darwin"

    push_enabled = bool(training_cfg.env.push_enabled) and not args.no_push
    push_body_id = -1
    if push_enabled:
        push_body_id = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_BODY, training_cfg.env.push_body
        )
        if push_body_id < 0:
            raise ValueError(
                f"Push body '{training_cfg.env.push_body}' not found in model."
            )

    def sample_push_schedule() -> PushSchedule:
        if not push_enabled:
            return PushSchedule(
                start_step=0,
                end_step=0,
                force_xy=np.zeros((2,), dtype=np.float32),
            )
        start_step = int(
            rng_env.integers(
                training_cfg.env.push_start_step_min,
                training_cfg.env.push_start_step_max + 1,
            )
        )
        force_mag = float(
            rng_env.uniform(
                training_cfg.env.push_force_min, training_cfg.env.push_force_max
            )
        )
        angle = float(rng_env.uniform(0.0, 2.0 * np.pi))
        force_xy = np.array(
            [force_mag * np.cos(angle), force_mag * np.sin(angle)], dtype=np.float32
        )
        end_step = start_step + int(training_cfg.env.push_duration_steps)
        return PushSchedule(start_step=start_step, end_step=end_step, force_xy=force_xy)

    push_schedule = sample_push_schedule()

    def apply_push(step_count: int) -> bool:
        if not push_enabled or push_body_id < 0:
            return False
        if not hasattr(mj_data, "xfrc_applied"):
            return False
        active = (step_count >= push_schedule.start_step) and (
            step_count < push_schedule.end_step
        )
        mj_data.xfrc_applied[:] = 0.0
        if active:
            fx, fy = float(push_schedule.force_xy[0]), float(push_schedule.force_xy[1])
            mj_data.xfrc_applied[push_body_id, :] = np.array(
                [fx, fy, 0.0, 0.0, 0.0, 0.0], dtype=np.float32
            )
        return active

    # On macOS, we'll try to launch the viewer and catch the error if mjpython isn't being used
    # This is more reliable than trying to detect mjpython in advance

    print(f"\n{'=' * 60}")
    print("WildRobot Policy Visualization")
    print(f"{'=' * 60}")
    print(f"  Mode: {'Deterministic' if deterministic else 'Stochastic'}")
    print(f"  Render: {'Headless' if headless else 'Interactive viewer'}")
    print(
        f"  Velocity range: [{training_cfg.env.min_velocity:.2f}, {training_cfg.env.max_velocity:.2f}] m/s"
    )
    if args.velocity_cmd is not None:
        print(f"  Fixed velocity cmd: {args.velocity_cmd:.2f} m/s")
    print(f"  Control dt: {ctrl_dt}s ({1/ctrl_dt:.0f} Hz)")
    try:
        left_toe, left_heel, right_toe, right_heel = (
            cal._robot_config.get_foot_geom_names()
        )
        print("  Foot geom params:")
        for geom_name in (left_toe, left_heel, right_toe, right_heel):
            geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            if geom_id < 0:
                print(f"    {geom_name}: not found")
                continue
            geom_pos = mj_model.geom_pos[geom_id]
            geom_size = mj_model.geom_size[geom_id]
            geom_friction = mj_model.geom_friction[geom_id]
            print(
                f"    {geom_name}: pos=({geom_pos[0]:.4f}, {geom_pos[1]:.4f}, {geom_pos[2]:.4f}), "
                f"size=({geom_size[0]:.4f}, {geom_size[1]:.4f}, {geom_size[2]:.4f}), "
                f"friction=({geom_friction[0]:.3f}, {geom_friction[1]:.3f}, {geom_friction[2]:.3f})"
            )
    except Exception as exc:
        print(f"  Foot geom params: unavailable ({exc})")
    if not headless:
        print(f"  Playback speed: {args.speed}x")
    print(f"{'=' * 60}")
    if not headless:
        print("\nControls:")
        print("  Space     - Pause/Resume")
        print("  Backspace - Reset")
        print("  Escape    - Exit")
        print(f"{'=' * 60}\n")

    # Recording setup
    frames = []
    record_steps = int(args.record_duration / ctrl_dt) if args.record else 0
    if args.record:
        print(
            f"Recording {args.record_duration}s ({record_steps} steps) to {args.record}"
        )

    # Determine number of episodes
    max_episodes = args.num_episodes
    if max_episodes is None and headless and not args.record:
        max_episodes = 1  # Default to 1 episode in headless mode without recording

    step_count = 0
    episode_count = 0
    prev_root_pos = None
    prev_root_quat = None

    if headless:
        # Headless mode - run without viewer
        # Try to create renderer with appropriate backend
        renderer = None
        if args.record:
            import os

            # Try different rendering backends for headless mode
            original_backend = os.environ.get("MUJOCO_GL", None)

            backends_to_try = ["osmesa", "egl", "glfw"]
            for backend in backends_to_try:
                try:
                    os.environ["MUJOCO_GL"] = backend
                    # Force reimport of mujoco to pick up new backend
                    import importlib

                    import mujoco as mj_reimport

                    importlib.reload(mj_reimport)
                    renderer = mj_reimport.Renderer(mj_model, 640, 480)
                    print(f"  Using {backend} backend for rendering")
                    break
                except Exception as e:
                    if backend == backends_to_try[-1]:
                        # Last backend failed, try without setting env var
                        if original_backend:
                            os.environ["MUJOCO_GL"] = original_backend
                        else:
                            os.environ.pop("MUJOCO_GL", None)
                        print(f"  Warning: Could not initialize renderer ({e})")
                        print(
                            f"  Video recording disabled. Install osmesa or run on Linux for headless rendering."
                        )
                    continue

        print("\nRunning in headless mode (native MuJoCo - fast)...")

        max_steps = (
            record_steps if args.record else 10000
        )  # Limit steps if not recording
        while step_count < max_steps:
            # Get observation from native MuJoCo state
            obs = get_observation(mj_data, velocity_cmd, prev_action)
            obs_jax = jnp.asarray(obs)

            # Get action from policy
            rng, action_rng = jax.random.split(rng)
            action = get_action(obs_jax, action_rng)

            last_root_pos = mj_data.qpos[0:3].copy()
            last_root_quat = mj_data.qpos[3:7].copy()

            # Apply action with filtering and step physics (native MuJoCo - FAST)
            filtered_action = apply_action(mj_data, action, prev_action)
            push_active = apply_push(step_count)
            step_physics(mj_model, mj_data, n_substeps)

            # Update previous action for next observation (use filtered action)
            prev_action = filtered_action
            prev_root_pos = last_root_pos
            prev_root_quat = last_root_quat

            step_count += 1

            # Debug: Print progress every 50 steps
            if step_count % 50 == 0:
                forward_vel = get_forward_velocity(
                    mj_data, prev_root_pos, prev_root_quat, ctrl_dt
                )  # heading-local forward velocity
                root_vel = cal.get_root_velocity(
                    mj_data, frame=CoordinateFrame.HEADING_LOCAL
                )
                lateral_vel = float(root_vel.linear[1])
                height = mj_data.qpos[2]
                robot_pos = mj_data.qpos[0:3]
                drift_xy = robot_pos[0:2] - episode_start_pos[0:2]
                drift_norm = float(np.linalg.norm(drift_xy))
                yaw_delta = wrap_angle(get_yaw(mj_data) - episode_start_yaw)
                action_bias = (
                    float(action[left_hip_pitch_idx] - action[right_hip_pitch_idx]),
                    float(action[left_hip_roll_idx] - action[right_hip_roll_idx]),
                    float(action[left_knee_pitch_idx] - action[right_knee_pitch_idx]),
                    float(action[left_ankle_pitch_idx] - action[right_ankle_pitch_idx]),
                )
                contacts = cal.get_geom_based_foot_contacts(mj_data, normalize=False)
                contacts = np.asarray(contacts)
                push_str = ""
                if push_enabled:
                    push_str = (
                        f", push={'1' if push_active else '0'}"
                        f", push_f=({float(push_schedule.force_xy[0]):+.2f}, {float(push_schedule.force_xy[1]):+.2f})"
                        f", push_win=[{push_schedule.start_step},{push_schedule.end_step})"
                    )
                print(
                    "  Step "
                    f"{step_count}: vel={forward_vel:.2f}m/s, "
                    f"lat_vel={lateral_vel:.3f}m/s, "
                    f"height={height:.3f}m, "
                    f"pos=({robot_pos[0]:.3f}, {robot_pos[1]:.3f}), "
                    f"drift=({drift_xy[0]:.3f}, {drift_xy[1]:.3f})m "
                    f"(|d|={drift_norm:.3f}m), "
                    f"yaw_d={yaw_delta:.3f}rad, "
                    f"act_d=({action_bias[0]:+.3f}, {action_bias[1]:+.3f}, "
                    f"{action_bias[2]:+.3f}, {action_bias[3]:+.3f}), "
                    f"foot_f=({contacts[0]:.1f}, {contacts[1]:.1f}, "
                    f"{contacts[2]:.1f}, {contacts[3]:.1f})"
                    f"{push_str}"
                )

            # Recording
            if renderer and step_count <= record_steps:
                renderer.update_scene(mj_data)
                frame = renderer.render()
                frames.append(frame)

            # Check for episode end (robot fell or max steps)
            done = check_termination(mj_data, step_count)
            if done:
                episode_count += 1
                forward_vel = get_forward_velocity(
                    mj_data, prev_root_pos, prev_root_quat, ctrl_dt
                )  # heading-local forward velocity
                height = mj_data.qpos[2]
                print(
                    f"Episode {episode_count} ended at step {step_count}: vel={forward_vel:.2f}m/s, height={height:.2f}m"
                )

                # Check if we've reached max episodes
                if max_episodes and episode_count >= max_episodes:
                    break

                # Reset environment
                if fixed_velocity is None:
                    velocity_cmd = sample_velocity_cmd()
                reset_robot(mj_model, mj_data, apply_noise=apply_reset_noise)
                push_schedule = sample_push_schedule()
                prev_action = default_policy_action.copy()
                prev_root_pos = None
                prev_root_quat = None
                episode_start_pos = mj_data.qpos[0:3].copy()
                episode_start_yaw = get_yaw(mj_data)
                step_count = 0

        if renderer:
            renderer.close()

        # Save video
        if args.record and frames:
            import imageio

            print(f"Saving video to {args.record}...")
            imageio.mimsave(args.record, frames, fps=int(1 / ctrl_dt))
            print(f"Done! Saved {len(frames)} frames.")

    else:
        # Interactive viewer mode (native MuJoCo - fast)
        # On macOS with mjpython, we need to use a different approach
        # The passive viewer runs in a separate thread, but we control the simulation
        try:
            with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
                # Set viewer options
                viewer.cam.distance = 2.5
                viewer.cam.elevation = -15
                viewer.cam.azimuth = 135  # View from behind-right
                viewer.cam.lookat[:] = [0, 0, 0.4]

                print(
                    f"Viewer started (native MuJoCo - fast). is_running={viewer.is_running()}"
                )

                # Main simulation loop
                while viewer.is_running():
                    step_start = time.time()

                    # Get observation from native MuJoCo state
                    obs = get_observation(mj_data, velocity_cmd, prev_action)
                    obs_jax = jnp.asarray(obs)

                    # Get action from policy
                    rng, action_rng = jax.random.split(rng)
                    action = get_action(obs_jax, action_rng)

                    last_root_pos = mj_data.qpos[0:3].copy()
                    last_root_quat = mj_data.qpos[3:7].copy()

                    # Apply action with filtering and step physics (native MuJoCo - FAST)
                    filtered_action = apply_action(mj_data, action, prev_action)
                    push_active = apply_push(step_count)
                    step_physics(mj_model, mj_data, n_substeps)

                    # Update previous action for next observation (use filtered action)
                    prev_action = filtered_action
                    prev_root_pos = last_root_pos
                    prev_root_quat = last_root_quat

                    step_count += 1

                    # Camera tracking: follow the robot
                    robot_pos = mj_data.qpos[0:3]  # [x, y, z] position
                    viewer.cam.lookat[:] = [robot_pos[0], robot_pos[1], 0.4]

                    # Debug: Print progress every 50 steps
                    if step_count % 50 == 0:
                        forward_vel = get_forward_velocity(
                            mj_data, prev_root_pos, prev_root_quat, ctrl_dt
                        )  # heading-local forward velocity
                        root_vel = cal.get_root_velocity(
                            mj_data, frame=CoordinateFrame.HEADING_LOCAL
                        )
                        lateral_vel = float(root_vel.linear[1])
                        height = mj_data.qpos[2]
                        drift_xy = robot_pos[0:2] - episode_start_pos[0:2]
                        drift_norm = float(np.linalg.norm(drift_xy))
                        yaw_delta = wrap_angle(get_yaw(mj_data) - episode_start_yaw)
                        action_bias = (
                            float(
                                action[left_hip_pitch_idx] - action[right_hip_pitch_idx]
                            ),
                            float(
                                action[left_hip_roll_idx] - action[right_hip_roll_idx]
                            ),
                            float(
                                action[left_knee_pitch_idx]
                                - action[right_knee_pitch_idx]
                            ),
                            float(
                                action[left_ankle_pitch_idx]
                                - action[right_ankle_pitch_idx]
                            ),
                        )
                        contacts = cal.get_geom_based_foot_contacts(
                            mj_data, normalize=False
                        )
                        contacts = np.asarray(contacts)
                        push_str = ""
                        if push_enabled:
                            push_str = (
                                f", push={'1' if push_active else '0'}"
                                f", push_f=({float(push_schedule.force_xy[0]):+.2f}, {float(push_schedule.force_xy[1]):+.2f})"
                                f", push_win=[{push_schedule.start_step},{push_schedule.end_step})"
                            )
                        print(
                            "  Step "
                            f"{step_count}: vel={forward_vel:.2f}m/s, "
                            f"lat_vel={lateral_vel:.3f}m/s, "
                            f"height={height:.3f}m, "
                            f"pos=({robot_pos[0]:.3f}, {robot_pos[1]:.3f}), "
                            f"drift=({drift_xy[0]:.3f}, {drift_xy[1]:.3f})m "
                            f"(|d|={drift_norm:.3f}m), "
                            f"yaw_d={yaw_delta:.3f}rad, "
                            f"act_d=({action_bias[0]:+.3f}, {action_bias[1]:+.3f}, "
                            f"{action_bias[2]:+.3f}, {action_bias[3]:+.3f}), "
                            f"foot_f=({contacts[0]:.1f}, {contacts[1]:.1f}, "
                            f"{contacts[2]:.1f}, {contacts[3]:.1f})"
                            f"{push_str}"
                        )

                    # Recording (if enabled with viewer)
                    if args.record and step_count <= record_steps:
                        renderer = mujoco.Renderer(mj_model, 640, 480)
                        renderer.update_scene(mj_data)
                        frame = renderer.render()
                        frames.append(frame)
                        renderer.close()

                        if step_count == record_steps:
                            import imageio

                            print(f"Saving video to {args.record}...")
                            imageio.mimsave(args.record, frames, fps=int(1 / ctrl_dt))
                            print("Done!")

                    # Check for episode end (robot fell or max steps)
                    done = check_termination(mj_data, step_count)
                    if done:
                        episode_count += 1
                        forward_vel = get_forward_velocity(
                            mj_data, prev_root_pos, prev_root_quat, ctrl_dt
                        )  # heading-local forward velocity
                        height = mj_data.qpos[2]
                        print(
                            f"Episode {episode_count} ended at step {step_count}: vel={forward_vel:.2f}m/s, height={height:.2f}m"
                        )

                        # Check if we've reached max episodes
                        if max_episodes and episode_count >= max_episodes:
                            break

                        # Reset environment
                        if fixed_velocity is None:
                            velocity_cmd = sample_velocity_cmd()
                        reset_robot(mj_model, mj_data, apply_noise=apply_reset_noise)
                        push_schedule = sample_push_schedule()
                        prev_action = default_policy_action.copy()
                        prev_root_pos = None
                        prev_root_quat = None
                        episode_start_pos = mj_data.qpos[0:3].copy()
                        episode_start_yaw = get_yaw(mj_data)
                        step_count = 0

                    # Sync viewer
                    viewer.sync()

                    # Timing for real-time playback
                    elapsed = time.time() - step_start
                    sleep_time = (ctrl_dt / args.speed) - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

        except RuntimeError as e:
            if "mjpython" in str(e) and is_macos:
                print(f"\n⚠️  macOS requires mjpython for the viewer.")
                print(
                    "   Run with: uv run mjpython playground_amp/training/visualize_policy.py ..."
                )
                print(
                    "   Or use --headless mode: uv run python ... --headless --num-episodes 1"
                )
                return 1
            raise

    print(f"\nVisualization ended. {episode_count} episodes completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
