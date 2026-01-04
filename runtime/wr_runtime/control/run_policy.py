from __future__ import annotations

import argparse
import math
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
from ..validation.startup_validator import validate_runtime_interface


def _normalize_quat_xyzw(quat_xyzw: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_xyzw, dtype=np.float32).reshape(4)
    n = float(np.linalg.norm(quat))
    if n <= 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return (quat / n).astype(np.float32)


def _quat_conjugate(quat_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = [float(v) for v in quat_xyzw]
    return np.array([-x, -y, -z, w], dtype=np.float32)


def _rotate_vec_by_quat(quat_xyzw: np.ndarray, vec: np.ndarray) -> np.ndarray:
    x, y, z, w = [float(v) for v in quat_xyzw]
    vx, vy, vz = [float(v) for v in vec]

    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)

    vpx = vx + w * tx + (y * tz - z * ty)
    vpy = vy + w * ty + (z * tx - x * tz)
    vpz = vz + w * tz + (x * ty - y * tx)
    return np.array([vpx, vpy, vpz], dtype=np.float32)


def _quat_xyzw_to_gravity_local(quat_xyzw: np.ndarray) -> np.ndarray:
    quat = _normalize_quat_xyzw(quat_xyzw)
    quat_inv = _quat_conjugate(quat)
    gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    return _rotate_vec_by_quat(quat_inv, gravity_world)


def _yaw_from_quat_xyzw(quat_xyzw: np.ndarray) -> float:
    x, y, z, w = [float(v) for v in quat_xyzw]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _world_to_heading_local(vec_world: np.ndarray, yaw: float) -> np.ndarray:
    vx, vy, vz = [float(v) for v in vec_world]
    c = math.cos(-yaw)
    s = math.sin(-yaw)
    hx = c * vx - s * vy
    hy = s * vx + c * vy
    return np.array([hx, hy, vz], dtype=np.float32)


def _angvel_heading_local(gyro_body: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    quat = _normalize_quat_xyzw(quat_xyzw)
    gyro_body = np.asarray(gyro_body, dtype=np.float32).reshape(3)
    gyro_world = _rotate_vec_by_quat(quat, gyro_body)
    yaw = _yaw_from_quat_xyzw(quat)
    return _world_to_heading_local(gyro_world, yaw)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run WildRobot ONNX policy on hardware")
    parser.add_argument("--config", type=str, default=None, help="Path to runtime JSON (default: ~/wildrobot_config.json)")
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

    print(f"Running control loop at {cfg.control.hz} Hz with {len(joint_names)} actuators")
    print("Ctrl+C to stop")

    try:
        while True:
            loop_start = time.time()
            dt = loop_start - last_time
            last_time = loop_start

            # Sensors
            imu_sample = imu.read()
            gravity = _quat_xyzw_to_gravity_local(imu_sample.quat_xyzw)
            angvel = _angvel_heading_local(imu_sample.gyro_rad_s, imu_sample.quat_xyzw)
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

            # Timing
            period = 1.0 / cfg.control.hz
            elapsed = time.time() - loop_start
            if elapsed < period:
                time.sleep(period - elapsed)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
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
