from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from ..config import load_config
from ..inference.onnx_policy import OnnxPolicy
from ..utils.mjcf import load_mjcf_model_info
from ..hardware.hiwonder_actuators import HiwonderBoardActuators, HiwonderJointConfig
from ..hardware.bno085 import BNO085IMU
from ..hardware.foot_switches import FootSwitches
from ..validation.startup_validator import validate_runtime_interface


def _quat_xyzw_to_gravity_local(quat_xyzw: np.ndarray) -> np.ndarray:
    """Compute gravity vector in body(local) frame.

    Returns a unit-ish vector pointing in the direction of gravity expressed in body frame.
    """
    x, y, z, w = [float(v) for v in quat_xyzw]
    # Normalize
    n = (x * x + y * y + z * z + w * w) ** 0.5
    if n <= 1e-8:
        return np.array([0.0, 0.0, -1.0], dtype=np.float32)
    x, y, z, w = x / n, y / n, z / n, w / n

    # Rotate world gravity (0,0,-1) into body frame: v_body = q^{-1} * v_world * q
    # Using quaternion-vector rotation formula.
    vx, vy, vz = 0.0, 0.0, -1.0

    # q^{-1}
    ix, iy, iz, iw = -x, -y, -z, w

    # t = 2 * cross(q_vec, v)
    tx = 2.0 * (iy * vz - iz * vy)
    ty = 2.0 * (iz * vx - ix * vz)
    tz = 2.0 * (ix * vy - iy * vx)

    # v' = v + w*t + cross(q_vec, t)
    vpx = vx + w * tx + (y * tz - z * ty)
    vpy = vy + w * ty + (z * tx - x * tz)
    vpz = vz + w * tz + (x * ty - y * tx)

    # Now apply inverse? We already used q_vec with original q, which gives v_rot = q * v * q^{-1}.
    # For v_body from world using q_body_world (as sensor returns), this is typically correct.
    return np.array([vpx, vpy, vpz], dtype=np.float32)


def build_observation(
    *,
    gravity: np.ndarray,
    angvel: np.ndarray,
    linvel: np.ndarray,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    foot_switches: np.ndarray,
    prev_action: np.ndarray,
    velocity_cmd: float,
) -> np.ndarray:
    # Layout must match training.envs.wildrobot_env.ObsLayout
    obs = np.concatenate(
        [
            gravity.astype(np.float32).reshape(3),
            angvel.astype(np.float32).reshape(3),
            linvel.astype(np.float32).reshape(3),
            joint_pos.astype(np.float32).reshape(-1),
            joint_vel.astype(np.float32).reshape(-1),
            foot_switches.astype(np.float32).reshape(4),
            prev_action.astype(np.float32).reshape(-1),
            np.array([velocity_cmd], dtype=np.float32),
            np.array([0.0], dtype=np.float32),  # padding
        ]
    )
    return obs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run WildRobot ONNX policy on hardware")
    parser.add_argument("--config", type=str, default=None, help="Path to runtime JSON (default: ~/wildrobot_config.json)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    mjcf_info = load_mjcf_model_info(Path(cfg.mjcf_path))
    joint_names = mjcf_info.actuator_names

    policy = OnnxPolicy(cfg.policy_onnx_path)
    print(f"ONNX policy input='{policy.info.input_name}' output='{policy.info.output_name}'")
    if policy.info.obs_dim is not None:
        print(f"ONNX policy obs_dim={policy.info.obs_dim}")
    if policy.info.action_dim is not None:
        print(f"ONNX policy action_dim={policy.info.action_dim}")

    # Fail-fast interface checks (before touching hardware)
    robot_cfg = validate_runtime_interface(cfg=cfg, mjcf_info=mjcf_info, policy=policy)
    print(
        "Startup validation OK: "
        f"obs_dim={robot_cfg.obs_dim} action_dim={robot_cfg.action_dim} "
        f"actuators={robot_cfg.actuator_names}"
    )

    actuators = HiwonderBoardActuators(
        joint_names=joint_names,
        cfg=HiwonderJointConfig(servo_ids=cfg.servo_ids, joint_offsets_rad=cfg.joint_offsets_rad),
        port=cfg.hiwonder_port,
        baudrate=cfg.hiwonder_baudrate,
        default_move_duration_ms=int(1000.0 / cfg.control_hz),
    )

    imu = BNO085IMU(
        i2c_address=cfg.bno085_i2c_address,
        upside_down=cfg.bno085_upside_down,
        sampling_hz=int(cfg.control_hz),
    )

    foot = FootSwitches(cfg.foot_switch_pins)

    prev_action = np.zeros(len(joint_names), dtype=np.float32)
    last_time = time.time()

    print(f"Running control loop at {cfg.control_hz} Hz with {len(joint_names)} actuators")
    print("Ctrl+C to stop")

    try:
        while True:
            loop_start = time.time()
            dt = loop_start - last_time
            last_time = loop_start

            # Sensors
            imu_sample = imu.read()
            gravity = _quat_xyzw_to_gravity_local(imu_sample.quat_xyzw)
            angvel = imu_sample.gyro_rad_s
            linvel = np.zeros(3, dtype=np.float32)  # no estimator yet

            # Actuators feedback
            joint_pos = actuators.get_positions_rad()
            if joint_pos is None:
                joint_pos = np.zeros(len(joint_names), dtype=np.float32)
            joint_vel = actuators.estimate_velocities_rad_s(dt)

            foot_sample = foot.read()
            foot_switches = np.array(foot_sample.switches, dtype=np.float32)

            obs = build_observation(
                gravity=gravity,
                angvel=angvel,
                linvel=linvel,
                joint_pos=joint_pos,
                joint_vel=joint_vel,
                foot_switches=foot_switches,
                prev_action=prev_action,
                velocity_cmd=cfg.velocity_cmd,
            )

            action = policy.predict(obs)
            if action.shape[0] != len(joint_names):
                raise ValueError(
                    f"Policy returned action_dim={action.shape[0]} but MJCF has {len(joint_names)} actuators ({joint_names})"
                )

            # Basic safety clamp
            action = np.clip(action, -1.0, 1.0)

            # Interpret action as delta around zero pose (configurable later)
            targets = cfg.action_scale_rad * action
            actuators.set_targets_rad(targets)

            prev_action = action

            # Timing
            period = 1.0 / cfg.control_hz
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
