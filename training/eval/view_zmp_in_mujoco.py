#!/usr/bin/env python3
"""MuJoCo viewer for ZMP reference library (v0.20.0-B verification).

Replays reference trajectories from the offline library on the WildRobot
MuJoCo model.  The viewer sets joint position targets (ctrl) from the
library's q_ref and lets the PD servos track them, showing whether the
reference is physically trackable.

All output is logged to a timestamped file in /tmp/zmp_viewer_logs/.

Usage:
  # Interactive viewer at cmd=0.15
  uv run mjpython training/eval/view_zmp_in_mujoco.py --vx 0.15

  # Fixed-base: pelvis pinned at the standing pose, legs track via PD.
  # Use this for tracking-RMSE / saturation diagnostics ONLY — the
  # pelvis is held in place, so foot world positions do not advance,
  # and stride length must be measured in free-floating (or kinematic).
  uv run mjpython training/eval/view_zmp_in_mujoco.py --fixed-base --vx 0.10

  # Headless recording
  uv run mjpython training/eval/view_zmp_in_mujoco.py --vx 0.15 \
      --headless --horizon 100 --print-every 5

  # Custom library path
  uv run mjpython training/eval/view_zmp_in_mujoco.py \
      --library-path /tmp/zmp_ref_library --vx 0.10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from control.references.reference_library import ReferenceLibrary
from training.utils.ctrl_order import CtrlOrderMapper


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------

class Logger:
    """Dual-output logger: prints to console and writes to a log file."""

    def __init__(self, log_dir: str = "/tmp/zmp_viewer_logs") -> None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.file_path = log_path / f"zmp_viewer_{timestamp}.log"
        self._file = open(self.file_path, "w")
        self.log(f"Log file: {self.file_path}")

    def log(self, msg: str) -> None:
        print(msg)
        self._file.write(msg + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model():
    """Load WildRobot MuJoCo model."""
    model_path = Path("assets/v2/scene_flat_terrain.xml")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    return model, data


def _load_policy_spec():
    """Load policy spec and extract actuator names."""
    spec_path = Path("assets/v2/mujoco_robot_config.json")
    with open(spec_path) as f:
        spec = json.load(f)
    return [j["name"] for j in spec["actuated_joint_specs"]]


def _init_standing(model, data):
    """Initialize robot to standing keyframe."""
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)


# ---------------------------------------------------------------------------
# Main viewer dispatch
# ---------------------------------------------------------------------------

def run_viewer(
    lib: ReferenceLibrary,
    vx: float,
    logger: Logger,
    headless: bool = False,
    kinematic: bool = False,
    fixed_base: bool = False,
    horizon: int = 200,
    print_every: int = 10,
    sim_dt_factor: int = 1,
) -> None:
    model, data = _load_model()
    actuator_names = _load_policy_spec()
    mapper = CtrlOrderMapper(model, actuator_names)

    traj = lib.lookup(vx)
    n_steps = traj.n_steps
    if kinematic:
        mode_str = "kinematic"
    elif fixed_base:
        mode_str = "fixed-base"
    else:
        mode_str = "free-floating"
    logger.log(f"Trajectory: vx={traj.command_vx:.3f}, n_steps={n_steps}, "
               f"duration={n_steps * traj.dt:.2f}s, "
               f"gait_cycle={traj.cycle_time:.3f}s")
    horizon = min(horizon, n_steps)
    logger.log(f"Viewer: horizon={horizon}, mode={mode_str}")

    _init_standing(model, data)

    # Build qpos index mapping: policy order -> qpos index
    act_to_qpos = []
    for i in range(model.nu):
        joint_id = model.actuator_trnid[i, 0]
        act_to_qpos.append(model.jnt_qposadr[joint_id])
    act_to_qpos = np.array(act_to_qpos)

    # Policy order -> MuJoCo actuator order permutation
    policy_to_mj = mapper._perm_np

    ref_dt = traj.dt

    if kinematic:
        _run_kinematic(model, data, traj, vx, horizon, print_every,
                       ref_dt, policy_to_mj, act_to_qpos, headless, logger)
    else:
        _run_physics(model, data, traj, mapper, horizon, print_every,
                     ref_dt, sim_dt_factor, headless, logger,
                     act_to_qpos=act_to_qpos, policy_to_mj=policy_to_mj,
                     fixed_base=fixed_base)


# ---------------------------------------------------------------------------
# Kinematic replay
# ---------------------------------------------------------------------------

def _run_kinematic(model, data, traj, vx, horizon, print_every,
                   ref_dt, policy_to_mj, act_to_qpos, headless, logger):
    """Kinematic replay: set qpos directly, no physics."""
    import time as time_mod

    n_steps = traj.n_steps

    def set_pose(step_idx):
        idx = min(step_idx, n_steps - 1)
        q_ref = traj.q_ref[idx]

        mj_ctrl_order = np.zeros(len(policy_to_mj))
        mj_ctrl_order[policy_to_mj] = q_ref
        for act_i in range(len(act_to_qpos)):
            data.qpos[act_to_qpos[act_i]] = mj_ctrl_order[act_i]

        # Use the trajectory's monotonic pelvis x (matches IK frame).
        data.qpos[0] = float(traj.pelvis_pos[idx, 0]) if traj.pelvis_pos is not None else step_idx * ref_dt * vx
        data.qpos[1] = float(traj.pelvis_pos[idx, 1]) if traj.pelvis_pos is not None else 0.0
        data.qpos[2] = float(traj.pelvis_pos[idx, 2]) if traj.pelvis_pos is not None else 0.45
        data.qpos[3:7] = [1, 0, 0, 0]
        mujoco.mj_forward(model, data)

    if headless:
        logger.log(f"\n{'step':>5} {'phase':>5} {'root_x':>7}")
        for i in range(horizon):
            set_pose(i)
            if i % print_every == 0:
                logger.log(f"{i:5d} {traj.phase[i]:5.2f} {data.qpos[0]:+7.4f}")
        logger.log("Kinematic replay complete.")
    else:
        step_idx = [0]
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.distance = 1.5
            viewer.cam.elevation = -20
            viewer.cam.lookat[:] = [0, 0, 0.4]

            while viewer.is_running() and step_idx[0] < horizon:
                set_pose(step_idx[0])
                viewer.cam.lookat[0] = data.qpos[0]
                viewer.sync()
                step_idx[0] += 1
                time_mod.sleep(ref_dt)

            logger.log(f"Kinematic replay done: {step_idx[0]} steps")


# ---------------------------------------------------------------------------
# Physics replay
# ---------------------------------------------------------------------------

_LEG_JOINT_NAMES = (
    "L_hip_pitch", "R_hip_pitch", "L_hip_roll", "R_hip_roll",
    "L_knee_pitch", "R_knee_pitch", "L_ankle_pitch", "R_ankle_pitch",
)


def _run_physics(model, data, traj, mapper, horizon, print_every,
                 ref_dt, sim_dt_factor, headless, logger,
                 act_to_qpos=None, policy_to_mj=None, fixed_base=False):
    """Physics replay: set ctrl, step physics, observe response.

    Captures v0.20.0-C strict-pass diagnostics:
      - per-joint tracking RMSE (8 leg joints, qpos vs commanded q_ref)
      - touchdown events (0->1 contact_mask transitions per foot)
      - per-side realized step length and command ratio
    """
    import time as time_mod

    n_steps = traj.n_steps
    physics_dt = model.opt.timestep
    physics_steps_per_ref = max(1, int(round(ref_dt / physics_dt)))

    _init_standing(model, data)

    root_qpos_init = data.qpos[:7].copy()

    # Startup ramp
    startup_steps = 50
    standing_q = np.zeros(19, dtype=np.float32)
    first_q = traj.q_ref[0]

    logger.log(f"Startup ramp: {startup_steps} steps blending to walking posture")
    for i in range(startup_steps):
        alpha = (i + 1) / startup_steps
        blended = standing_q * (1 - alpha) + first_q * alpha
        mapper.set_all_ctrl(data, blended)
        for _ in range(physics_steps_per_ref):
            mujoco.mj_step(model, data)

    root_z_after_ramp = data.qpos[2]
    logger.log(f"After ramp: root_z={root_z_after_ramp:.4f}")

    # ---- Strict-pass instrumentation (RMSE / saturation / touchdowns) ----
    n_leg = 8
    leg_qpos_log = np.zeros((horizon, n_leg), dtype=np.float64)
    leg_qref_log = np.zeros((horizon, n_leg), dtype=np.float64)
    leg_sat_log = np.zeros((horizon, n_leg), dtype=bool)
    log_count = [0]

    # Pull leg-joint limits from the MJCF in the same policy order as q_ref.
    leg_joint_names = list(_LEG_JOINT_NAMES)
    name_map = {
        "L_hip_pitch": "left_hip_pitch", "R_hip_pitch": "right_hip_pitch",
        "L_hip_roll":  "left_hip_roll",  "R_hip_roll":  "right_hip_roll",
        "L_knee_pitch": "left_knee_pitch", "R_knee_pitch": "right_knee_pitch",
        "L_ankle_pitch": "left_ankle_pitch", "R_ankle_pitch": "right_ankle_pitch",
    }
    leg_jnt_limits = np.zeros((n_leg, 2), dtype=np.float64)
    for j, short in enumerate(leg_joint_names):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name_map[short])
        leg_jnt_limits[j] = model.jnt_range[jid]
    sat_margin = 0.01  # rad ≈ 0.6° from limit counts as saturated

    left_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_foot")
    right_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_foot")

    left_touchdown_x = []   # foot world x at each L 0->1 transition
    right_touchdown_x = []
    prev_contact = traj.contact_mask[0].astype(np.float32).copy()

    step_log = []

    def step_fn(step_idx: int) -> dict:
        # Trajectory plays linearly through the multi-cycle plan; clamp at
        # the end so we hold the last frame rather than wrap to cycle 0.
        idx = min(step_idx, n_steps - 1)
        q_ref = traj.q_ref[idx]
        mapper.set_all_ctrl(data, q_ref)

        for _ in range(physics_steps_per_ref * sim_dt_factor):
            mujoco.mj_step(model, data)

        if fixed_base:
            # Pin pelvis at the standing pose (no synthetic forward
            # injection — earlier code teleported root_x by command_vx
            # each step, which made any "speed tracking" reading
            # tautological).  After this, root_x stays at the initial
            # value and only RMSE / saturation are valid metrics from
            # this mode; stride length must come from free-floating.
            data.qpos[:7] = root_qpos_init
            data.qvel[:6] = 0
            mujoco.mj_forward(model, data)

        # Capture leg qpos in policy order for RMSE + saturation
        if act_to_qpos is not None and policy_to_mj is not None and log_count[0] < horizon:
            mj_qpos = data.qpos[act_to_qpos]
            policy_qpos = mj_qpos[policy_to_mj]
            leg_qpos_log[log_count[0]] = policy_qpos[:n_leg]
            leg_qref_log[log_count[0]] = q_ref[:n_leg]
            leg_sat_log[log_count[0]] = (
                (policy_qpos[:n_leg] <= leg_jnt_limits[:, 0] + sat_margin) |
                (policy_qpos[:n_leg] >= leg_jnt_limits[:, 1] - sat_margin)
            )
            log_count[0] += 1

        # Touchdown detection on q_ref's contact mask (0->1 per foot)
        cur_contact = traj.contact_mask[idx]
        if cur_contact[0] > 0.5 and prev_contact[0] < 0.5:
            left_touchdown_x.append(float(data.xpos[left_foot_id, 0]))
        if cur_contact[1] > 0.5 and prev_contact[1] < 0.5:
            right_touchdown_x.append(float(data.xpos[right_foot_id, 0]))
        prev_contact[:] = cur_contact

        root_pos = data.qpos[:3].copy()
        qw, qx, qy, qz = data.qpos[3:7]
        pitch = np.arctan2(2 * (qw * qy - qz * qx),
                           1 - 2 * (qx**2 + qy**2))
        roll = np.arctan2(2 * (qw * qx + qy * qz),
                          1 - 2 * (qx**2 + qy**2))

        return {
            "step": step_idx,
            "phase": float(traj.phase[idx]),
            "root_x": root_pos[0],
            "root_y": root_pos[1],
            "root_z": root_pos[2],
            "pitch": pitch,
            "roll": roll,
            "stance": int(traj.stance_foot_id[idx]),
            "contact_l": float(traj.contact_mask[idx, 0]),
            "contact_r": float(traj.contact_mask[idx, 1]),
        }

    if headless:
        logger.log(f"\n{'step':>5} {'phase':>5} {'root_z':>7} {'pitch':>7} "
                   f"{'roll':>7} {'root_x':>7} {'stance':>6}")
        logger.log("-" * 55)

        for i in range(horizon):
            info = step_fn(i)
            step_log.append(info)

            if i % print_every == 0 or i == horizon - 1:
                st = "L" if info["stance"] == 0 else "R"
                logger.log(f"{i:5d} {info['phase']:5.2f} {info['root_z']:7.4f} "
                           f"{info['pitch']:+7.4f} {info['roll']:+7.4f} "
                           f"{info['root_x']:+7.4f} {st:>6}")

            if not fixed_base and (abs(info["pitch"]) > 0.8 or info["root_z"] < 0.15):
                logger.log(f"\n  TERMINATED at step {i}: "
                           f"pitch={info['pitch']:.3f} root_z={info['root_z']:.3f}")
                break

        if step_log:
            pitches = [s["pitch"] for s in step_log]
            rolls = [s["roll"] for s in step_log]
            heights = [s["root_z"] for s in step_log]
            x_final = step_log[-1]["root_x"]
            survived = len(step_log)
            logger.log(f"\nSummary ({survived}/{horizon} steps):")
            logger.log(f"  pitch: mean={np.mean(pitches):+.4f} "
                       f"p95={np.percentile(np.abs(pitches), 95):.4f}")
            logger.log(f"  roll:  mean={np.mean(rolls):+.4f} "
                       f"p95={np.percentile(np.abs(rolls), 95):.4f}")
            logger.log(f"  height: mean={np.mean(heights):.4f} "
                       f"min={np.min(heights):.4f}")
            logger.log(f"  forward: x={x_final:+.4f}m in {survived} steps")

        # ---- v0.20.0-C strict-pass diagnostics ----
        n_logged = log_count[0]
        if n_logged > 0:
            err = leg_qpos_log[:n_logged] - leg_qref_log[:n_logged]
            rmse = np.sqrt(np.mean(err ** 2, axis=0))
            mae = np.mean(np.abs(err), axis=0)
            sat_frac = leg_sat_log[:n_logged].mean(axis=0)
            logger.log(f"\nLeg-joint tracking ({n_logged} steps, rad):")
            logger.log(f"  {'joint':<14} {'rmse':>7} {'mae':>7} "
                       f"{'max_abs':>8} {'sat%':>6}")
            for j in range(n_leg):
                logger.log(f"  {_LEG_JOINT_NAMES[j]:<14} "
                           f"{rmse[j]:7.4f} {mae[j]:7.4f} "
                           f"{np.max(np.abs(err[:, j])):8.4f} "
                           f"{100.0 * sat_frac[j]:6.1f}")
            any_sat = leg_sat_log[:n_logged].any(axis=1).mean()
            logger.log(f"  overall rmse  : {np.sqrt(np.mean(err ** 2)):.4f} rad")
            logger.log(f"  any-joint saturation: {100.0 * any_sat:.1f}% of steps")

        # Touchdowns: per-side stride = consecutive same-foot touchdowns;
        # step length = stride/2.  In fixed-base mode the pelvis is
        # pinned, so foot world positions reflect IK only — strides
        # measured here are not meaningful.  Use free-floating mode to
        # measure realized stride.
        sl_cmd = abs(traj.command_vx) * traj.cycle_time / 2.0
        stride_cmd = 2.0 * sl_cmd
        if fixed_base:
            logger.log(f"\nTouchdowns (count only — strides not meaningful "
                       f"in fixed-base):")
            logger.log(f"  left  count={len(left_touchdown_x):2d}  "
                       f"right count={len(right_touchdown_x):2d}")
        else:
            logger.log(f"\nTouchdowns (commanded step={sl_cmd:.4f} m, "
                       f"stride={stride_cmd:.4f} m):")
            for name, xs in (("left",  left_touchdown_x),
                             ("right", right_touchdown_x)):
                if len(xs) >= 2:
                    strides = np.diff(np.array(xs))
                    step_len = strides / 2.0
                    ratio = (np.mean(step_len) / sl_cmd
                             if sl_cmd > 1e-6 else float('nan'))
                    logger.log(f"  {name:<5} touchdowns={len(xs):2d}  "
                               f"stride={np.mean(strides):+.4f}m "
                               f"(min={np.min(strides):+.4f})  "
                               f"step={np.mean(step_len):+.4f}m  "
                               f"realized/cmd={ratio:+.2f}")
                else:
                    logger.log(f"  {name:<5} touchdowns={len(xs):2d} "
                               f"(need ≥ 2 for stride)")
    else:
        step_idx = [0]

        def controller(model, data):
            info = step_fn(step_idx[0])
            step_log.append(info)
            step_idx[0] += 1
            if step_idx[0] % print_every == 0:
                st = "L" if info["stance"] == 0 else "R"
                logger.log(f"step={info['step']:5d} z={info['root_z']:.3f} "
                           f"pitch={info['pitch']:+.3f} x={info['root_x']:+.3f} {st}")

        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.distance = 1.5
            viewer.cam.elevation = -20
            viewer.cam.lookat[:] = [0, 0, 0.4]

            while viewer.is_running():
                controller(model, data)
                viewer.sync()
                time.sleep(ref_dt * sim_dt_factor)

        # Write summary after viewer closes
        if step_log:
            survived = len(step_log)
            pitches = [s["pitch"] for s in step_log]
            heights = [s["root_z"] for s in step_log]
            x_final = step_log[-1]["root_x"]
            logger.log(f"\nSession summary ({survived} steps):")
            logger.log(f"  pitch: mean={np.mean(pitches):+.4f} "
                       f"p95={np.percentile(np.abs(pitches), 95):.4f}")
            logger.log(f"  height: mean={np.mean(heights):.4f} "
                       f"min={np.min(heights):.4f}")
            logger.log(f"  forward: x={x_final:+.4f}m")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MuJoCo viewer for ZMP reference library")
    parser.add_argument("--library-path", type=str, default=None,
                        help="Path to reference library (default: generate fresh)")
    parser.add_argument("--vx", type=float, default=0.15,
                        help="Forward speed command")
    parser.add_argument("--headless", action="store_true",
                        help="Run headless with text output")
    parser.add_argument("--kinematic", action="store_true",
                        help="Kinematic replay (set qpos directly, no physics). "
                             "Shows the intended reference motion without balance issues.")
    parser.add_argument("--fixed-base", action="store_true",
                        help="Fixed-base physics: root held upright, legs track via PD. "
                             "Shows servo tracking of the reference without balance issues.")
    parser.add_argument("--horizon", type=int, default=200,
                        help="Number of reference steps to run")
    parser.add_argument("--print-every", type=int, default=10,
                        help="Print diagnostics every N steps")
    parser.add_argument("--sim-dt-factor", type=int, default=1,
                        help="Physics substeps per reference step")
    args = parser.parse_args()

    logger = Logger()

    if args.library_path:
        lib = ReferenceLibrary.load(args.library_path)
    else:
        logger.log("No library path given, generating fresh...")
        from control.zmp.zmp_walk import ZMPWalkGenerator
        gen = ZMPWalkGenerator()
        lib = gen.build_library()

    logger.log(lib.summary())
    logger.log("")
    run_viewer(
        lib, args.vx,
        logger=logger,
        headless=args.headless,
        kinematic=args.kinematic,
        fixed_base=args.fixed_base,
        horizon=args.horizon,
        print_every=args.print_every,
        sim_dt_factor=args.sim_dt_factor,
    )

    logger.log(f"\nLog saved to: {logger.file_path}")
    logger.close()


if __name__ == "__main__":
    main()
