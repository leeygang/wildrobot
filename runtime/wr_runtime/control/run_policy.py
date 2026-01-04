from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from policy_contract.numpy.action import postprocess_action
from policy_contract.numpy.calib import action_to_ctrl, normalize_joint_pos, normalize_joint_vel
from policy_contract.numpy.obs import build_observation_from_components
from policy_contract.numpy.state import PolicyState
from policy_contract.spec import PolicyBundle, validate_spec

from ..config import load_config
from ..hardware.bno085 import BNO085IMU
from ..hardware.foot_switches import FootSwitches
from ..hardware.hiwonder_actuators import HiwonderBoardActuators
from ..inference.onnx_policy import OnnxPolicy
from ..utils.mjcf import load_mjcf_model_info
from ..utils.frames import angvel_heading_local, gravity_local_from_quat
from ..validation.startup_validator import validate_runtime_interface


def main() -> None:
    parser = argparse.ArgumentParser(description="Run WildRobot ONNX policy on hardware")
    parser.add_argument("--config", type=str, default=None, help="Path to runtime JSON (default: ~/wildrobot_config.json)")
    parser.add_argument("--log-path", type=str, default=None, help="Optional .npz path to save replay logs on exit")
    parser.add_argument(
        "--log-steps",
        type=int,
        default=None,
        help="Optional max steps to log before auto-stopping (default: run until Ctrl+C)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    mjcf_info = load_mjcf_model_info(Path(cfg.mjcf_resolved_path))
    joint_names = mjcf_info.actuator_names

    bundle = PolicyBundle.load(Path(cfg.policy_resolved_path).parent)
    spec = bundle.spec
    validate_spec(spec)
    if spec.observation.linvel_mode != "zero":
        raise ValueError(
            "Runtime requires linvel_mode='zero' until a hardware estimator is available. "
            f"Found: {spec.observation.linvel_mode}"
        )

    policy = OnnxPolicy(
        str(bundle.model_path),
        input_name=spec.model.input_name,
        output_name=spec.model.output_name,
    )
    print(f"ONNX policy input='{policy.info.input_name}' output='{policy.info.output_name}'")
    if policy.info.obs_dim is not None:
        print(f"ONNX policy obs_dim={policy.info.obs_dim}")
    if policy.info.action_dim is not None:
        print(f"ONNX policy action_dim={policy.info.action_dim}")

    # Fail-fast interface checks (before touching hardware)
    robot_cfg = validate_runtime_interface(
        cfg=cfg,
        mjcf_info=mjcf_info,
        spec=spec,
        onnx_obs_dim=policy.info.obs_dim,
        onnx_action_dim=policy.info.action_dim,
    )
    print(
        "Startup validation OK: "
        f"obs_dim={robot_cfg.obs_dim} action_dim={robot_cfg.action_dim} "
        f"actuators={robot_cfg.actuator_names}"
    )

    actuators = HiwonderBoardActuators(
        joint_names=joint_names,
        servo_ids=cfg.hiwonder.servo_ids,
        joint_offsets_rad=cfg.hiwonder.joint_offsets_rad,
        port=cfg.hiwonder.port,
        baudrate=cfg.hiwonder.baudrate,
        default_move_duration_ms=int(1000.0 / cfg.control.hz),
    )

    imu = BNO085IMU(
        i2c_address=cfg.bno085.i2c_address,
        upside_down=cfg.bno085.upside_down,
        sampling_hz=int(cfg.control.hz),
    )

    foot = FootSwitches(cfg.foot_switches.get_all_pins())

    state = PolicyState.init(spec)
    last_time = time.time()

    log_path = Path(args.log_path).expanduser() if args.log_path else None
    log_steps = int(args.log_steps) if args.log_steps is not None else None

    log_quat: list[np.ndarray] = []
    log_gyro: list[np.ndarray] = []
    log_joint_pos: list[np.ndarray] = []
    log_joint_vel: list[np.ndarray] = []
    log_foot: list[np.ndarray] = []
    log_vel_cmd: list[np.ndarray] = []

    print(f"Running control loop at {cfg.control.hz} Hz with {len(joint_names)} actuators")
    print("Ctrl+C to stop")

    try:
        while True:
            loop_start = time.time()
            dt = loop_start - last_time
            last_time = loop_start

            # Sensors
            imu_sample = imu.read()
            gravity = gravity_local_from_quat(imu_sample.quat_xyzw)
            angvel = angvel_heading_local(imu_sample.gyro_rad_s, imu_sample.quat_xyzw)
            linvel = np.zeros(3, dtype=np.float32)  # estimator TBD

            # Actuators feedback
            joint_pos = actuators.get_positions_rad()
            if joint_pos is None:
                joint_pos = np.zeros(len(joint_names), dtype=np.float32)
            joint_vel = actuators.estimate_velocities_rad_s(dt)

            foot_sample = foot.read()
            foot_switches = np.array(foot_sample.switches, dtype=np.float32)

            joint_pos_norm = normalize_joint_pos(spec=spec, joint_pos_rad=joint_pos)
            joint_vel_norm = normalize_joint_vel(spec=spec, joint_vel_rad_s=joint_vel)

            obs = build_observation_from_components(
                spec=spec,
                gravity_local=gravity,
                angvel_heading_local=angvel,
                linvel_heading_local=linvel,
                joint_pos_normalized=joint_pos_norm,
                joint_vel_normalized=joint_vel_norm,
                foot_switches=foot_switches,
                prev_action=state.prev_action,
                velocity_cmd=np.array([cfg.control.velocity_cmd], dtype=np.float32),
            )

            action_raw = policy.predict(obs)
            action_post, state = postprocess_action(spec=spec, state=state, action_raw=action_raw)
            ctrl_targets = action_to_ctrl(spec=spec, action=action_post)
            actuators.set_targets_rad(ctrl_targets)

            if log_path is not None:
                log_quat.append(np.asarray(imu_sample.quat_xyzw, dtype=np.float32))
                log_gyro.append(np.asarray(imu_sample.gyro_rad_s, dtype=np.float32))
                log_joint_pos.append(np.asarray(joint_pos, dtype=np.float32))
                log_joint_vel.append(np.asarray(joint_vel, dtype=np.float32))
                log_foot.append(np.asarray(foot_switches, dtype=np.float32))
                log_vel_cmd.append(np.asarray([cfg.control.velocity_cmd], dtype=np.float32))
                if log_steps is not None and len(log_quat) >= log_steps:
                    break

            # Timing
            period = 1.0 / cfg.control.hz
            elapsed = time.time() - loop_start
            if elapsed < period:
                time.sleep(period - elapsed)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        if log_path is not None and log_quat:
            np.savez(
                log_path,
                quat_xyzw=np.stack(log_quat, axis=0),
                gyro_rad_s=np.stack(log_gyro, axis=0),
                joint_pos_rad=np.stack(log_joint_pos, axis=0),
                joint_vel_rad_s=np.stack(log_joint_vel, axis=0),
                foot_switches=np.stack(log_foot, axis=0),
                velocity_cmd=np.stack(log_vel_cmd, axis=0),
            )
            print(f"Saved replay log to {log_path}")
        try:
            foot.close()
        finally:
            try:
                imu.close()
            finally:
                actuators.disable()
                actuators.close()


if __name__ == "__main__":
    main()
