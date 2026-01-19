from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from policy_contract.numpy.action import postprocess_action
from policy_contract.calib import NumpyCalibOps
from policy_contract.numpy.obs import build_observation
from policy_contract.numpy.signals import Signals
from policy_contract.numpy.state import PolicyState
from policy_contract.spec import PolicyBundle, validate_spec
from policy_contract.numpy.frames import gravity_local_from_quat, normalize_quat_xyzw

from runtime.configs import load_config
from ..hardware.actuators import HiwonderBoardActuators
from ..hardware.bno085 import BNO085IMU
from ..hardware.foot_switches import FootSwitches
from ..hardware.imu import Imu
from ..hardware.robot_io import HardwareRobotIO
from ..inference.onnx_policy import OnnxPolicy
from ..utils.mjcf import load_mjcf_model_info
from ..validation.startup_validator import validate_runtime_interface


def _imu_sanity_check(
    imu: Imu,
    *,
    samples: int,
    gravity_z_tol: float,
    gyro_norm_tol: float,
    sleep_s: float,
) -> None:
    """Fail fast if IMU outputs look wrong before commanding motors.

    Checks:
      - quat norms are close to 1
      - gravity_local z is near -1 when upright
      - gyro norm is small when still
    """

    quats = []
    gyro_norms: list[float] = []
    gravities = []
    for _ in range(max(1, int(samples))):
        s = imu.read()
        if not getattr(s, "valid", True):
            raise RuntimeError("IMU sanity check failed: sample marked invalid (valid=False)")
        q = normalize_quat_xyzw(np.asarray(s.quat_xyzw, dtype=np.float32))
        quats.append(q)
        g = gravity_local_from_quat(q)
        gravities.append(g)
        gyro = np.asarray(s.gyro_rad_s, dtype=np.float32)
        gyro_norms.append(float(np.linalg.norm(gyro)))
        time.sleep(max(0.0, float(sleep_s)))

    quats_arr = np.stack(quats, axis=0)
    norms = np.linalg.norm(quats_arr, axis=1)
    norm_dev = float(np.max(np.abs(norms - 1.0)))

    g_local_mean = np.mean(np.stack(gravities, axis=0), axis=0)
    g_z_err = abs(float(g_local_mean[2]) + 1.0)

    gyro_mean_norm = float(np.mean(gyro_norms))

    if norm_dev > 0.05:
        raise RuntimeError(f"IMU sanity check failed: quat norm deviation {norm_dev:.3f} > 0.05")
    if g_z_err > gravity_z_tol:
        raise RuntimeError(
            f"IMU sanity check failed: gravity z error {g_z_err:.3f} > {gravity_z_tol:.3f} (sensor upright?)"
        )
    if gyro_mean_norm > gyro_norm_tol:
        raise RuntimeError(
            f"IMU sanity check failed: mean gyro norm {gyro_mean_norm:.3f} rad/s > {gyro_norm_tol:.3f} (sensor should be still)"
        )


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
    parser.add_argument(
        "--skip-imu-check",
        action="store_true",
        help="Skip IMU sanity check before enabling control loop (not recommended)",
    )
    parser.add_argument(
        "--imu-check-samples",
        type=int,
        default=20,
        help="Number of IMU samples to average for sanity check",
    )
    parser.add_argument(
        "--imu-gravity-z-tol",
        type=float,
        default=0.25,
        help="Allowed error for gravity_local z vs -1 (abs error)",
    )
    parser.add_argument(
        "--imu-gyro-norm-tol",
        type=float,
        default=1.0,
        help="Allowed mean gyro norm (rad/s) during sanity check",
    )
    parser.add_argument(
        "--imu-check-sleep-s",
        type=float,
        default=0.02,
        help="Sleep between IMU samples during sanity check",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    mjcf_info = load_mjcf_model_info(Path(cfg.mjcf_resolved_path))
    joint_names = mjcf_info.actuator_names

    bundle = PolicyBundle.load(Path(cfg.policy_resolved_path).parent)
    spec = bundle.spec
    validate_spec(spec)

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

    control_dt = 1.0 / cfg.control.hz
    imu = BNO085IMU(
        i2c_address=cfg.bno085.i2c_address,
        upside_down=cfg.bno085.upside_down,
        axis_map=cfg.bno085.axis_map,
        suppress_debug=cfg.bno085.suppress_debug,
        i2c_frequency_hz=cfg.bno085.i2c_frequency_hz,
        init_retries=cfg.bno085.init_retries,
        sampling_hz=int(cfg.control.hz),
    )

    print(
        "IMU config: "
        f"addr=0x{cfg.bno085.i2c_address:02X} upside_down={cfg.bno085.upside_down} "
        f"axis_map={cfg.bno085.axis_map if cfg.bno085.axis_map is not None else ['+X','+Y','+Z']} "
        f"freq={cfg.bno085.i2c_frequency_hz}Hz retries={cfg.bno085.init_retries}"
    )

    if args.skip_imu_check:
        print("Skipping IMU sanity check (requested).")
    else:
        print("Running IMU sanity check (no motors commanded)...")
        _imu_sanity_check(
            imu,
            samples=args.imu_check_samples,
            gravity_z_tol=args.imu_gravity_z_tol,
            gyro_norm_tol=args.imu_gyro_norm_tol,
            sleep_s=args.imu_check_sleep_s,
        )
        print("IMU sanity check passed.")

    move_time_ms = (
        cfg.servo_controller.default_move_time_ms
        if cfg.servo_controller.default_move_time_ms is not None
        else int(control_dt * 1000.0)
    )

    actuators = HiwonderBoardActuators(
        actuator_names=spec.robot.actuator_names,
        servo_ids=cfg.servo_controller.servo_ids,
        joint_offset_units=cfg.servo_controller.joint_offset_units,
        joint_directions=cfg.servo_controller.joint_directions,
        port=cfg.servo_controller.port,
        baudrate=cfg.servo_controller.baudrate,
        default_move_time_ms=move_time_ms,
    )

    foot = FootSwitches(cfg.foot_switches.get_all_pins())

    robot_io = HardwareRobotIO(
        actuator_names=spec.robot.actuator_names,
        control_dt=control_dt,
        actuators=actuators,
        imu=imu,
        foot_switches=foot,
    )

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

            try:
                signals = robot_io.read()

                obs = build_observation(
                    spec=spec,
                    state=state,
                    signals=signals,
                    velocity_cmd=np.array([cfg.control.velocity_cmd], dtype=np.float32),
                )

                action_raw = policy.predict(obs)
                action_post, state = postprocess_action(spec=spec, state=state, action_raw=action_raw)
                ctrl_targets = NumpyCalibOps.action_to_ctrl(spec=spec, action=action_post)
                robot_io.write_ctrl(ctrl_targets)
            except Exception as exc:  # noqa: BLE001
                print(f"Runtime error in control loop: {exc}")
                actuators.disable()
                raise

            if log_path is not None:
                log_quat.append(np.asarray(signals.quat_xyzw, dtype=np.float32))
                log_gyro.append(np.asarray(signals.gyro_rad_s, dtype=np.float32))
                log_joint_pos.append(np.asarray(signals.joint_pos_rad, dtype=np.float32))
                log_joint_vel.append(np.asarray(signals.joint_vel_rad_s, dtype=np.float32))
                log_foot.append(np.asarray(signals.foot_switches, dtype=np.float32))
                log_vel_cmd.append(np.asarray([cfg.control.velocity_cmd], dtype=np.float32))
                if log_steps is not None and len(log_quat) >= log_steps:
                    break

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
        robot_io.close()


if __name__ == "__main__":
    main()
