"""``wildrobot-run-policy`` — deterministic control loop for the latest bundle.

Loads a policy bundle (``policy.onnx`` + ``policy_spec.json`` +
``runtime_policy_config.json``) and runs the v8 home-base-residual control loop
at ``control_hz``.  Supports a hardware-free ``--dry-run`` mode (mock IO) for
smoke tests and safe validation on a developer machine.

Examples
--------
Dry run (no hardware), 5 steps, straight walk::

    uv run --project runtime wildrobot-run-policy \
      --bundle /tmp/wr_runtime_smoke9_bundle_check \
      --dry-run --max-steps 5 --velocity-cmd 0.13,0.0,0.0

Hardware run (on the robot), forward walk for 500 control steps::

    uv run --project runtime wildrobot-run-policy \
      --bundle /path/to/bundle \
      --runtime-config ~/.wildrobot/config.json \
      --max-steps 500 --velocity-cmd 0.13,0.0,0.0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from policy_contract.spec import PolicyBundle, validate_spec

from wr_runtime.inference.onnx_policy import OnnxPolicy
from wr_runtime.control.mock_robot_io import MockRobotIO
from wr_runtime.control.policy_runner import RuntimePolicyRunner
from wr_runtime.control.runtime_policy_config import RuntimePolicyConfig


_LEG_LOG_JOINTS = (
    ("LHP", "left_hip_pitch"),
    ("LHR", "left_hip_roll"),
    ("LK", "left_knee_pitch"),
    ("LAP", "left_ankle_pitch"),
    ("LAR", "left_ankle_roll"),
    ("RHP", "right_hip_pitch"),
    ("RHR", "right_hip_roll"),
    ("RK", "right_knee_pitch"),
    ("RAP", "right_ankle_pitch"),
    ("RAR", "right_ankle_roll"),
)

_FOOT_SWITCH_LABELS = ("left_toe", "left_heel", "right_toe", "right_heel")


def _parse_velocity_cmd(text: Optional[str], default: List[float]) -> np.ndarray:
    if text is None:
        return np.asarray(default, dtype=np.float32).reshape(3)
    parts = [p.strip() for p in str(text).split(",") if p.strip() != ""]
    if len(parts) == 1:
        return np.array([float(parts[0]), 0.0, 0.0], dtype=np.float32)
    if len(parts) == 3:
        return np.array([float(p) for p in parts], dtype=np.float32)
    raise SystemExit(
        f"--velocity-cmd must be 'vx' or 'vx,vy,wz'; got {text!r}"
    )


def _build_hardware_robot_io(
    *, runtime_config_path: Path, actuator_names: List[str], control_dt: float
):
    """Construct the real hardware RobotIO from the runtime config.

    Imported lazily (GPIO / serial / I2C backends are Linux-only), and wires the
    concrete hardware classes (HiwonderBoardActuators / BNO085IMU / FootSwitches)
    directly — there are no ``*.from_config`` factories.
    """
    from configs import WrRuntimeConfig
    from wr_runtime.hardware.actuators import HiwonderBoardActuators
    from wr_runtime.hardware.bno085 import BNO085IMU
    from wr_runtime.hardware.foot_switches import FootSwitches
    from wr_runtime.hardware.robot_io import HardwareRobotIO

    cfg = WrRuntimeConfig.load(runtime_config_path)
    sc = cfg.servo_controller

    # Fail fast with an actionable message if the runtime config does not cover
    # every actuator in the policy spec (the actuator constructor would
    # otherwise raise a bare KeyError mid-init).  This catches stale configs
    # missing newer joints (e.g. left/right_ankle_roll on the v8 21-actuator
    # spec) before any hardware is touched.
    missing = [n for n in actuator_names if n not in sc.servo_ids]
    if missing:
        raise SystemExit(
            "Runtime config is missing servo entries for "
            f"{missing} (required by the policy spec's actuator_names). "
            f"Add them under servo_controller.servos in {runtime_config_path}."
        )

    # HardwareRobotIO.write_ctrl calls set_targets_rad(move_time_ms=None), so a
    # default move time is required; fall back to one control period.
    default_move_time_ms = sc.default_move_time_ms
    if default_move_time_ms is None:
        default_move_time_ms = max(1, int(round(control_dt * 1000.0)))

    actuators = HiwonderBoardActuators(
        actuator_names=actuator_names,
        servo_ids=sc.servo_ids,
        port=sc.port,
        baudrate=sc.baudrate,
        default_move_time_ms=default_move_time_ms,
        joint_servo_offset_units=sc.joint_servo_offset_units,
        joint_motor_unit_directions=sc.joint_motor_unit_directions,
        joint_angle_at_zero_unit_deg=sc.joint_angle_at_zero_unit_deg,
    )
    imu = BNO085IMU(
        i2c_address=cfg.bno085.i2c_address,
        upside_down=cfg.bno085.upside_down,
        sampling_hz=max(1, int(round(1.0 / control_dt))),
        axis_map=cfg.bno085.axis_map,
        suppress_debug=cfg.bno085.suppress_debug,
        i2c_frequency_hz=cfg.bno085.i2c_frequency_hz,
        init_retries=cfg.bno085.init_retries,
    )
    foot_switches = FootSwitches(pins=cfg.foot_switches.get_all_pins())
    return HardwareRobotIO(
        actuator_names=actuator_names,
        control_dt=control_dt,
        actuators=actuators,
        imu=imu,
        foot_switches=foot_switches,
    )


def _actuator_indices(
    actuator_names: Optional[List[str]], joint_names: tuple[str, ...]
) -> List[int]:
    if not actuator_names:
        return []
    by_name = {name: idx for idx, name in enumerate(actuator_names)}
    return [by_name[name] for name in joint_names if name in by_name]


def _format_leg_targets_deg(
    target_q_rad: np.ndarray, actuator_names: Optional[List[str]]
) -> str:
    return _format_leg_values_deg(target_q_rad, actuator_names)


def _format_leg_values_deg(
    values_rad: np.ndarray, actuator_names: Optional[List[str]]
) -> str:
    values = np.asarray(values_rad, dtype=np.float32).reshape(-1)
    if not actuator_names:
        return ""

    by_name = {name: idx for idx, name in enumerate(actuator_names)}
    parts = []
    for label, name in _LEG_LOG_JOINTS:
        idx = by_name.get(name)
        if idx is not None and idx < values.size:
            parts.append(f"{label}={float(np.rad2deg(values[idx])):+.1f}")
    return " ".join(parts)


def _format_foot_switches(info: dict) -> str:
    signals = info.get("signals")
    if signals is None:
        return ""
    switches = np.asarray(signals.foot_switches, dtype=np.float32).reshape(-1)
    if switches.size != 4:
        return ""
    values = [int(round(float(v))) for v in switches]
    return f"fs=[LT={values[0]},LH={values[1]},RT={values[2]},RH={values[3]}]"


def _format_rad_deg(value_rad: float) -> str:
    return f"{float(np.rad2deg(value_rad)):+.1f}"


def _run_hardware_preflight(
    *,
    robot_io,
    actuator_names: List[str],
    home_q_rad: np.ndarray,
    joint_min_rad: np.ndarray,
    joint_max_rad: np.ndarray,
    imu_startup_timeout_s: float,
    require_all_footswitches: bool,
    home_tolerance_deg: float,
) -> None:
    """Print and validate hardware state before the policy writes commands."""
    errors: List[str] = []
    warnings: List[str] = []
    print("Hardware preflight:", flush=True)

    _preflight_servos(
        robot_io=robot_io,
        actuator_names=actuator_names,
        home_q_rad=home_q_rad,
        joint_min_rad=joint_min_rad,
        joint_max_rad=joint_max_rad,
        home_tolerance_deg=home_tolerance_deg,
        errors=errors,
        warnings=warnings,
    )
    _preflight_imu(
        robot_io=robot_io,
        imu_startup_timeout_s=imu_startup_timeout_s,
        errors=errors,
    )
    _preflight_footswitches(
        robot_io=robot_io,
        require_all_footswitches=require_all_footswitches,
        errors=errors,
    )

    for warning in warnings:
        print(f"  WARNING: {warning}", flush=True)
    if errors:
        print("Hardware preflight FAILED:", flush=True)
        for error in errors:
            print(f"  ERROR: {error}", flush=True)
        raise SystemExit(
            "Hardware preflight failed; fix errors above before running policy."
        )
    print("Hardware preflight OK.", flush=True)


def _preflight_servos(
    *,
    robot_io,
    actuator_names: List[str],
    home_q_rad: np.ndarray,
    joint_min_rad: np.ndarray,
    joint_max_rad: np.ndarray,
    home_tolerance_deg: float,
    errors: List[str],
    warnings: List[str],
) -> None:
    actuators = robot_io.actuators
    port = getattr(actuators, "port", "unknown")
    baudrate = getattr(actuators, "baudrate", "unknown")
    controller = getattr(actuators, "controller", None)
    voltage = None
    if controller is not None and hasattr(controller, "get_battery_voltage"):
        try:
            voltage = controller.get_battery_voltage()
        except Exception as exc:
            warnings.append(f"servo board voltage read failed: {exc!r}")

    voltage_text = "unknown" if voltage is None else f"{float(voltage):.2f}V"
    print(
        f"  Servo board: port={port} baud={baudrate} voltage={voltage_text}",
        flush=True,
    )

    try:
        positions = actuators.get_positions_rad()
    except Exception as exc:
        positions = None
        errors.append(f"servo position read raised {exc!r}")

    if positions is None:
        last_error = getattr(actuators, "_last_error", None)
        suffix = f": {last_error!r}" if last_error is not None else ""
        errors.append(f"servo position read failed{suffix}")
        return

    pos = np.asarray(positions, dtype=np.float32).reshape(-1)
    expected_n = len(actuator_names)
    if pos.size != expected_n:
        errors.append(f"servo position count {pos.size} != actuator count {expected_n}")
        return

    ids = list(getattr(actuators, "servo_ids_list", []))
    if len(ids) != expected_n:
        ids = [None] * expected_n

    finite = np.isfinite(pos)
    limit_tol = np.deg2rad(5.0)
    home_tol = np.deg2rad(max(0.0, float(home_tolerance_deg)))
    max_home_err = 0.0
    print("  Servos:", flush=True)
    for idx, name in enumerate(actuator_names):
        sid = ids[idx]
        sid_text = "?" if sid is None else str(int(sid))
        q = float(pos[idx])
        home = float(home_q_rad[idx])
        qmin = float(joint_min_rad[idx])
        qmax = float(joint_max_rad[idx])
        home_err = q - home
        max_home_err = max(max_home_err, abs(home_err))
        status = "OK"
        if not bool(finite[idx]):
            status = "ERROR"
            errors.append(f"{name} servo id={sid_text} readback is non-finite")
        elif q < qmin - limit_tol or q > qmax + limit_tol:
            status = "ERROR"
            errors.append(
                f"{name} servo id={sid_text} readback {_format_rad_deg(q)}deg "
                f"is outside policy range [{_format_rad_deg(qmin)}, {_format_rad_deg(qmax)}]deg"
            )
        elif abs(home_err) > home_tol:
            status = "WARN"
        print(
            "    "
            f"{name:<20} id={sid_text:>3} "
            f"pos={_format_rad_deg(q)}deg "
            f"home={_format_rad_deg(home)}deg "
            f"err={_format_rad_deg(home_err)}deg "
            f"range=[{_format_rad_deg(qmin)}, {_format_rad_deg(qmax)}]deg "
            f"{status}",
            flush=True,
        )

    if max_home_err > home_tol:
        warnings.append(
            f"max servo home error {float(np.rad2deg(max_home_err)):.1f}deg "
            f"> tolerance {float(home_tolerance_deg):.1f}deg"
        )


def _preflight_imu(*, robot_io, imu_startup_timeout_s: float, errors: List[str]) -> None:
    try:
        if hasattr(robot_io, "wait_for_valid_imu_sample"):
            robot_io.wait_for_valid_imu_sample(timeout_s=float(imu_startup_timeout_s))
        sample = getattr(robot_io, "_last_valid_imu_sample", None)
        if sample is None:
            sample = robot_io.imu.read()
    except Exception as exc:
        errors.append(f"IMU valid sample unavailable: {exc}")
        print(f"  IMU: ERROR {exc}", flush=True)
        return

    valid = bool(getattr(sample, "valid", True))
    quat = np.asarray(getattr(sample, "quat_xyzw", []), dtype=np.float32).reshape(-1)
    gyro = np.asarray(getattr(sample, "gyro_rad_s", []), dtype=np.float32).reshape(-1)
    quat_norm = float(np.linalg.norm(quat)) if quat.size == 4 else float("nan")
    imu = robot_io.imu
    diag = getattr(imu, "diag", None)
    diag_text = f" diag={diag}" if diag else ""
    print(
        "  IMU: "
        f"valid={valid} "
        f"quat={np.round(quat, 4).tolist()} "
        f"quat_norm={quat_norm:.3f} "
        f"gyro={np.round(gyro, 4).tolist()} "
        f"errors={getattr(imu, 'error_count', 0)} "
        f"last_error={getattr(imu, 'last_error', None)}"
        f"{diag_text}",
        flush=True,
    )
    if not valid:
        errors.append("IMU sample is invalid")
    if (
        quat.size != 4
        or not np.all(np.isfinite(quat))
        or not (0.9 <= quat_norm <= 1.1)
    ):
        errors.append(
            f"IMU quaternion is invalid: quat={quat.tolist()} norm={quat_norm:.3f}"
        )
    if gyro.size != 3 or not np.all(np.isfinite(gyro)):
        errors.append(f"IMU gyro is invalid: gyro={gyro.tolist()}")


def _preflight_footswitches(
    *,
    robot_io,
    require_all_footswitches: bool,
    errors: List[str],
) -> None:
    try:
        sample = robot_io.foot_switches.read()
    except Exception as exc:
        errors.append(f"footswitch read failed: {exc!r}")
        print(f"  Footswitches: ERROR {exc!r}", flush=True)
        return

    switches = np.asarray(sample.switches, dtype=np.float32).reshape(-1)
    if switches.size != len(_FOOT_SWITCH_LABELS) or not np.all(np.isfinite(switches)):
        errors.append(f"footswitch sample invalid: {switches.tolist()}")
        print(f"  Footswitches: ERROR values={switches.tolist()}", flush=True)
        return

    values = [int(round(float(v))) for v in switches]
    states = ", ".join(
        f"{name}={value}" for name, value in zip(_FOOT_SWITCH_LABELS, values)
    )
    print(f"  Footswitches: {states} (1=pressed, 0=open)", flush=True)
    if require_all_footswitches:
        open_names = [
            name for name, value in zip(_FOOT_SWITCH_LABELS, values) if value == 0
        ]
        if open_names:
            errors.append(
                "footswitches open at walk start: "
                f"{open_names}; use --allow-unpressed-footswitch for suspended tests"
            )


def run_policy_loop(
    *,
    runner: RuntimePolicyRunner,
    max_steps: int,
    velocity_cmd: np.ndarray,
    log_steps: int,
    ctrl_dt: float,
    realtime: bool,
    actuator_names: Optional[List[str]] = None,
) -> List[dict]:
    """Run the control loop for ``max_steps`` iterations; return per-log infos."""
    logs: List[dict] = []
    leg_indices = _actuator_indices(
        actuator_names, tuple(name for _, name in _LEG_LOG_JOINTS)
    )
    for step in range(int(max_steps)):
        t0 = time.monotonic()
        info = runner.step(velocity_cmd)
        if log_steps > 0 and (step % log_steps == 0 or step == max_steps - 1):
            applied = info["applied_action"]
            target = info["target_q_rad"]
            leg_applied_max = (
                float(np.max(np.abs(applied[leg_indices]))) if leg_indices else None
            )
            leg_summary = _format_leg_targets_deg(target, actuator_names)
            observed = np.asarray(info["signals"].joint_pos_rad, dtype=np.float32)
            observed_leg_summary = _format_leg_values_deg(observed, actuator_names)
            leg_err_max = (
                float(np.max(np.abs(target[leg_indices] - observed[leg_indices])))
                if leg_indices and observed.size >= target.size
                else None
            )
            foot_summary = _format_foot_switches(info)
            extra_parts = []
            if leg_applied_max is not None:
                extra_parts.append(f"leg|applied|max={leg_applied_max:.3f}")
            if leg_summary:
                extra_parts.append(f"leg_deg={leg_summary}")
            if observed_leg_summary:
                extra_parts.append(f"obs_leg_deg={observed_leg_summary}")
            if leg_err_max is not None:
                extra_parts.append(
                    f"leg_err|max_deg={float(np.rad2deg(leg_err_max)):.1f}"
                )
            if foot_summary:
                extra_parts.append(foot_summary)
            extra = " " + " ".join(extra_parts) if extra_parts else ""
            print(
                f"[step {step:5d}] idx={info['step_idx']:5d} "
                f"|applied|max={float(np.max(np.abs(applied))):.3f} "
                f"target[0:3]={np.round(target[:3], 4).tolist()}"
                f"{extra}",
                flush=True,
            )
            logs.append(info)
        if realtime:
            elapsed = time.monotonic() - t0
            remaining = ctrl_dt - elapsed
            if remaining > 0:
                time.sleep(remaining)
    return logs


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a WildRobot policy bundle (latest v8 home-residual contract)."
    )
    parser.add_argument(
        "--bundle", type=str, required=True,
        help="Bundle directory (policy.onnx + policy_spec.json + runtime_policy_config.json)",
    )
    parser.add_argument(
        "--runtime-config", type=str, default=None,
        help="Hardware runtime config JSON (servo IDs/calibration). Required unless --dry-run.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run with mock IO (no hardware): exercises the full loop for smoke tests.",
    )
    parser.add_argument("--max-steps", type=int, default=500, help="Number of control steps.")
    parser.add_argument("--log-steps", type=int, default=20, help="Log every N steps (0=off).")
    parser.add_argument(
        "--velocity-cmd", type=str, default=None,
        help="Command 'vx' or 'vx,vy,wz' (default: bundle default_velocity_cmd).",
    )
    parser.add_argument(
        "--no-realtime", action="store_true",
        help="Do not sleep to maintain control_hz (default: realtime ON for hardware, OFF for --dry-run).",
    )
    parser.add_argument(
        "--imu-startup-timeout-s",
        type=float,
        default=3.0,
        help="Hardware only: wait this long for the first valid IMU sample before starting control.",
    )
    parser.add_argument(
        "--allow-unpressed-footswitch",
        action="store_true",
        help="Hardware preflight only: do not fail if one or more footswitches are open.",
    )
    parser.add_argument(
        "--preflight-home-tolerance-deg",
        type=float,
        default=25.0,
        help="Warn when any servo starts this many degrees away from policy home.",
    )
    parser.add_argument(
        "--skip-hardware-preflight",
        action="store_true",
        help="Skip hardware preflight checks before the policy loop.",
    )
    args = parser.parse_args(argv)

    bundle_path = Path(args.bundle)
    bundle = PolicyBundle.load(bundle_path)
    validate_spec(bundle.spec)
    runtime_cfg_path = bundle_path / "runtime_policy_config.json"
    runtime_config = RuntimePolicyConfig.from_json(runtime_cfg_path)

    velocity_cmd = _parse_velocity_cmd(
        args.velocity_cmd, runtime_config.default_velocity_cmd
    )

    policy = OnnxPolicy(
        str(bundle.model_path),
        input_name=bundle.spec.model.input_name,
        output_name=bundle.spec.model.output_name,
        expected_obs_dim=int(bundle.spec.model.obs_dim),
        expected_action_dim=int(bundle.spec.model.action_dim),
    )
    # Fail fast on ONNX/spec dim mismatch BEFORE the first control tick (mirrors
    # wildrobot-validate-bundle; the run command must not silently broadcast a
    # wrong-sized action into a full joint target).
    if policy.info.obs_dim is not None and int(policy.info.obs_dim) != int(
        bundle.spec.model.obs_dim
    ):
        raise SystemExit(
            f"ONNX obs_dim {policy.info.obs_dim} != spec {bundle.spec.model.obs_dim}"
        )
    if policy.info.action_dim is not None and int(policy.info.action_dim) != int(
        bundle.spec.model.action_dim
    ):
        raise SystemExit(
            f"ONNX action_dim {policy.info.action_dim} != spec "
            f"{bundle.spec.model.action_dim}"
        )

    actuator_names = list(bundle.spec.robot.actuator_names)
    ctrl_dt = float(runtime_config.ctrl_dt)

    if args.dry_run:
        home = (
            np.asarray(bundle.spec.robot.home_ctrl_rad, dtype=np.float32)
            if bundle.spec.robot.home_ctrl_rad is not None
            else None
        )
        robot_io = MockRobotIO(
            actuator_names=actuator_names, control_dt=ctrl_dt, home_q_rad=home
        )
        realtime = False  # dry-run is a smoke test; never sleep
    else:
        if args.runtime_config is None:
            raise SystemExit("--runtime-config is required unless --dry-run is set.")
        if not args.skip_hardware_preflight and bundle.spec.robot.home_ctrl_rad is None:
            raise SystemExit(
                "policy_spec.robot.home_ctrl_rad is required for hardware preflight"
            )
        robot_io = _build_hardware_robot_io(
            runtime_config_path=Path(args.runtime_config),
            actuator_names=actuator_names,
            control_dt=ctrl_dt,
        )
        if not args.skip_hardware_preflight:
            home = np.asarray(bundle.spec.robot.home_ctrl_rad, dtype=np.float32)
            joint_min = np.asarray(
                [
                    float(bundle.spec.robot.joints[name].range_min_rad)
                    for name in actuator_names
                ],
                dtype=np.float32,
            )
            joint_max = np.asarray(
                [
                    float(bundle.spec.robot.joints[name].range_max_rad)
                    for name in actuator_names
                ],
                dtype=np.float32,
            )
            try:
                _run_hardware_preflight(
                    robot_io=robot_io,
                    actuator_names=actuator_names,
                    home_q_rad=home,
                    joint_min_rad=joint_min,
                    joint_max_rad=joint_max,
                    imu_startup_timeout_s=float(args.imu_startup_timeout_s),
                    require_all_footswitches=not bool(
                        args.allow_unpressed_footswitch
                    ),
                    home_tolerance_deg=float(args.preflight_home_tolerance_deg),
                )
            except BaseException:
                try:
                    robot_io.close()
                finally:
                    raise
        elif hasattr(robot_io, "wait_for_valid_imu_sample"):
            print(
                "Skipping hardware preflight; waiting for first valid IMU sample "
                f"(timeout {float(args.imu_startup_timeout_s):.1f}s)...",
                flush=True,
            )
            robot_io.wait_for_valid_imu_sample(timeout_s=float(args.imu_startup_timeout_s))
        realtime = not args.no_realtime

    runner = RuntimePolicyRunner(
        spec=bundle.spec,
        runtime_config=runtime_config,
        policy=policy,
        robot_io=robot_io,
    )

    print(
        f"Running bundle {bundle_path} | layout={bundle.spec.observation.layout_id} "
        f"| residual_base={runtime_config.loc_ref_residual_base} "
        f"| control_hz={runtime_config.control_hz:.1f} "
        f"| cmd={velocity_cmd.tolist()} | dry_run={args.dry_run}",
        flush=True,
    )
    try:
        run_policy_loop(
            runner=runner,
            max_steps=args.max_steps,
            velocity_cmd=velocity_cmd,
            log_steps=args.log_steps,
            ctrl_dt=ctrl_dt,
            realtime=realtime,
            actuator_names=actuator_names,
        )
    finally:
        try:
            robot_io.close()
        except Exception as exc:  # pragma: no cover - best-effort cleanup
            print(f"Warning: robot_io.close() failed: {exc}", flush=True)
    print("Run complete.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
