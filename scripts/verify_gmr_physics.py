#!/usr/bin/env python3
"""Verify GMR retargeting by playing motion in MuJoCo physics.

This script helps diagnose why Tier 0+ extraction fails by showing:
1. How well the robot tracks the GMR joint targets
2. How much harness support is needed
3. What causes the robot to fall/fail

Usage:
    cd ~/projects/wildrobot
    uv run python scripts/verify_gmr_physics.py \
        --motion playground_amp/data/gmr/walking_medium01.pkl \
        --render
"""

import argparse
import pickle
import sys
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from playground_amp.envs.wildrobot_env import get_assets


def load_mujoco_model(model_path: str) -> mujoco.MjModel:
    """Load MuJoCo model with assets."""
    model_path = Path(model_path)
    mj_model = mujoco.MjModel.from_xml_string(
        model_path.read_text(), assets=get_assets(model_path.parent)
    )
    return mj_model


def resample_motion(motion_data, target_fps: float):
    """Resample motion to target FPS."""
    source_fps = motion_data["fps"]
    num_frames = motion_data["num_frames"]
    duration = motion_data["duration_sec"]

    new_num_frames = int(duration * target_fps)
    t_original = np.linspace(0, duration, num_frames)
    t_new = np.linspace(0, duration, new_num_frames)

    def interp_array(arr):
        if arr is None:
            return None
        result = np.zeros((new_num_frames, arr.shape[1]), dtype=np.float32)
        for i in range(arr.shape[1]):
            result[:, i] = np.interp(t_new, t_original, arr[:, i])
        return result

    def interp_quat(quats):
        result = interp_array(quats)
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        return result / np.clip(norms, 1e-8, None)

    return {
        "fps": target_fps,
        "root_pos": interp_array(motion_data["root_pos"]),
        "root_rot": interp_quat(motion_data["root_rot"]),
        "dof_pos": interp_array(motion_data["dof_pos"]),
        "num_frames": new_num_frames,
        "duration_sec": duration,
    }


def run_physics_test(
    motion_path: str,
    model_path: str = "assets/scene_flat_terrain.xml",
    render: bool = False,
    harness_mode: str = "capped",  # "none", "capped", "full"
    harness_cap: float = 0.15,  # fraction of mg
):
    """Run physics test on GMR motion.
    
    Args:
        motion_path: Path to GMR motion pickle
        model_path: Path to MuJoCo scene XML
        render: Whether to render the simulation
        harness_mode: "none" (no support), "capped" (limited support), "full" (full support)
        harness_cap: Harness force cap as fraction of mg (for "capped" mode)
    """
    # Load motion
    print(f"Loading motion: {motion_path}")
    with open(motion_path, "rb") as f:
        motion = pickle.load(f)
    
    print(f"  Source: {motion.get('source_file', 'unknown')}")
    print(f"  FPS: {motion['fps']:.1f}")
    print(f"  Frames: {motion['num_frames']}")
    print(f"  Duration: {motion['duration_sec']:.2f}s")
    
    # Resample to 50Hz if needed
    if abs(motion["fps"] - 50.0) > 0.1:
        print(f"  Resampling from {motion['fps']:.1f} to 50 Hz...")
        motion = resample_motion(motion, 50.0)
    
    # Load model
    print(f"\nLoading model: {model_path}")
    mj_model = load_mujoco_model(model_path)
    mj_data = mujoco.MjData(mj_model)
    
    # Physics parameters
    sim_dt = 0.002  # 500 Hz
    ctrl_dt = 0.02  # 50 Hz
    n_substeps = int(ctrl_dt / sim_dt)
    mj_model.opt.timestep = sim_dt
    
    # Robot parameters
    robot_mass = 3.383  # kg
    gravity = 9.81
    mg = robot_mass * gravity
    
    # Harness parameters
    stab_z_kp = 60.0
    stab_z_kd = 10.0
    stab_xy_kp = 50.0
    stab_xy_kd = 10.0
    stab_z_cap = harness_cap * mg
    stab_xy_cap = 0.10 * mg
    
    print(f"\nHarness mode: {harness_mode}")
    if harness_mode == "capped":
        print(f"  Z cap: {stab_z_cap:.1f} N ({harness_cap*100:.0f}% mg)")
        print(f"  XY cap: {stab_xy_cap:.1f} N (10% mg)")
    
    # Get foot geom IDs for contact detection
    left_toe_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "left_toe")
    left_heel_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "left_heel")
    right_toe_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "right_toe")
    right_heel_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "right_heel")
    foot_geom_ids = [left_toe_id, left_heel_id, right_toe_id, right_heel_id]
    
    # Initialize from home keyframe
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
    mj_data.qvel[:] = 0
    mujoco.mj_forward(mj_model, mj_data)
    
    # Reference data
    ref_dof_pos = motion["dof_pos"]
    ref_root_pos = motion["root_pos"]
    target_height = ref_root_pos[:, 2].mean()
    ref_xy_start = ref_root_pos[0, :2].copy()
    sim_xy_start = mj_data.qpos[0:2].copy()
    
    n_joints = 8
    num_frames = motion["num_frames"]
    
    # Metrics storage
    tracking_errors = []
    harness_forces = []
    contact_forces = []
    heights = []
    pitches = []
    rolls = []
    
    print(f"\nRunning physics simulation ({num_frames} frames)...")
    
    # Setup viewer if rendering
    viewer = None
    if render:
        viewer = mujoco.viewer.launch_passive(mj_model, mj_data)
    
    try:
        for i in range(num_frames):
            # Set control targets
            mj_data.ctrl[:n_joints] = ref_dof_pos[i]
            
            # Compute orientation
            w, x, y, z = mj_data.qpos[3:7]
            sinp = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
            curr_pitch = np.arcsin(sinp)
            sinr_cosp = 2.0 * (w * x + y * z)
            cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
            curr_roll = np.arctan2(sinr_cosp, cosr_cosp)
            
            pitches.append(np.degrees(abs(curr_pitch)))
            rolls.append(np.degrees(abs(curr_roll)))
            heights.append(mj_data.qpos[2])
            
            # Apply harness forces
            stab_force_z = 0.0
            stab_force_xy = np.zeros(2)
            
            if harness_mode != "none":
                # Only apply if upright
                if abs(curr_pitch) < 0.436 and abs(curr_roll) < 0.436:
                    # Z stabilization
                    height_error = target_height - mj_data.qpos[2]
                    stab_force_z = stab_z_kp * height_error - stab_z_kd * mj_data.qvel[2]
                    
                    # XY stabilization
                    ref_xy_delta = ref_root_pos[i, :2] - ref_xy_start
                    target_xy = sim_xy_start + ref_xy_delta
                    xy_error = target_xy - mj_data.qpos[0:2]
                    stab_force_xy = stab_xy_kp * xy_error - stab_xy_kd * mj_data.qvel[0:2]
                    
                    # Apply caps (if not full mode)
                    if harness_mode == "capped":
                        stab_force_z = np.clip(stab_force_z, -stab_z_cap, stab_z_cap)
                        stab_force_xy = np.clip(stab_force_xy, -stab_xy_cap, stab_xy_cap)
            
            mj_data.qfrc_applied[0:2] = stab_force_xy
            mj_data.qfrc_applied[2] = stab_force_z
            mj_data.qfrc_applied[3:6] = 0.0
            
            total_harness = np.sqrt(stab_force_z**2 + np.sum(stab_force_xy**2))
            harness_forces.append(total_harness)
            
            # Step physics
            for _ in range(n_substeps):
                mujoco.mj_step(mj_model, mj_data)
            
            # Measure tracking error
            actual_joints = mj_data.qpos[7:7+n_joints]
            target_joints = ref_dof_pos[i]
            error = np.abs(actual_joints - target_joints)
            tracking_errors.append(error)
            
            # Measure contact forces
            total_fn = 0.0
            wrench = np.zeros(6)
            for contact_idx in range(mj_data.ncon):
                contact = mj_data.contact[contact_idx]
                geom1, geom2 = contact.geom1, contact.geom2
                
                foot_geom = None
                if geom1 == 0 and geom2 in foot_geom_ids:
                    foot_geom = geom2
                elif geom2 == 0 and geom1 in foot_geom_ids:
                    foot_geom = geom1
                
                if foot_geom is not None:
                    mujoco.mj_contactForce(mj_model, mj_data, contact_idx, wrench)
                    fn = wrench[0]
                    if fn > 0:
                        total_fn += fn
            
            contact_forces.append(total_fn)
            
            # Update viewer
            if viewer is not None:
                viewer.sync()
            
            # Check for catastrophic failure
            if abs(curr_pitch) > 1.0 or abs(curr_roll) > 1.0:
                print(f"  [FAIL] Robot fell at frame {i} (pitch={np.degrees(curr_pitch):.1f}°, roll={np.degrees(curr_roll):.1f}°)")
                break
            
            if mj_data.qpos[2] < 0.1:
                print(f"  [FAIL] Robot collapsed at frame {i} (height={mj_data.qpos[2]:.3f}m)")
                break
    
    finally:
        if viewer is not None:
            viewer.close()
    
    # Print results
    frames_run = len(tracking_errors)
    print(f"\n{'='*60}")
    print("PHYSICS VERIFICATION RESULTS")
    print(f"{'='*60}")
    print(f"Frames simulated: {frames_run}/{num_frames} ({100*frames_run/num_frames:.0f}%)")
    
    tracking_errors = np.array(tracking_errors)
    harness_forces = np.array(harness_forces)
    contact_forces = np.array(contact_forces)
    
    print(f"\n[Joint Tracking]")
    print(f"  Mean error: {np.degrees(tracking_errors.mean()):.2f}° ({tracking_errors.mean():.4f} rad)")
    print(f"  Max error:  {np.degrees(tracking_errors.max()):.2f}° ({tracking_errors.max():.4f} rad)")
    
    print(f"\n[Load Support]")
    load_ratio = contact_forces / mg
    print(f"  Mean contact force: {contact_forces.mean():.1f} N ({100*load_ratio.mean():.0f}% mg)")
    print(f"  Min contact force:  {contact_forces.min():.1f} N ({100*load_ratio.min():.0f}% mg)")
    
    print(f"\n[Harness Reliance]")
    harness_ratio = harness_forces / mg
    print(f"  Mean harness force: {harness_forces.mean():.1f} N ({100*harness_ratio.mean():.0f}% mg)")
    print(f"  Max harness force:  {harness_forces.max():.1f} N ({100*harness_ratio.max():.0f}% mg)")
    print(f"  p95 harness force:  {np.percentile(harness_forces, 95):.1f} N ({100*np.percentile(harness_ratio, 95):.0f}% mg)")
    
    print(f"\n[Orientation]")
    print(f"  Mean pitch: {np.mean(pitches):.1f}°, max: {np.max(pitches):.1f}°")
    print(f"  Mean roll:  {np.mean(rolls):.1f}°, max: {np.max(rolls):.1f}°")
    
    print(f"\n[Height]")
    print(f"  Target: {target_height:.3f}m")
    print(f"  Mean:   {np.mean(heights):.3f}m")
    print(f"  Range:  [{np.min(heights):.3f}, {np.max(heights):.3f}]m")
    
    # Tier 0+ assessment
    print(f"\n[Tier 0+ Assessment]")
    meets_load = load_ratio.mean() >= 0.90
    meets_harness = np.percentile(harness_ratio, 95) <= 0.10
    meets_orientation = np.max(pitches) <= 20 and np.max(rolls) <= 20
    
    print(f"  Load support >= 90%:    {'✓' if meets_load else '✗'} ({100*load_ratio.mean():.0f}%)")
    print(f"  p95 harness <= 10%:     {'✓' if meets_harness else '✗'} ({100*np.percentile(harness_ratio, 95):.0f}%)")
    print(f"  Orientation <= 20°:     {'✓' if meets_orientation else '✗'} (pitch={np.max(pitches):.0f}°, roll={np.max(rolls):.0f}°)")
    
    if meets_load and meets_harness and meets_orientation:
        print(f"\n  [bold green]✓ PASSES Tier 0+ gates[/bold green]")
    else:
        print(f"\n  ✗ FAILS Tier 0+ gates")


def main():
    parser = argparse.ArgumentParser(description="Verify GMR physics tracking")
    parser.add_argument(
        "--motion",
        type=str,
        default="playground_amp/data/gmr/walking_medium01.pkl",
        help="Path to GMR motion pickle",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="assets/scene_flat_terrain.xml",
        help="Path to MuJoCo scene XML",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the simulation in MuJoCo viewer",
    )
    parser.add_argument(
        "--harness",
        type=str,
        default="capped",
        choices=["none", "capped", "full"],
        help="Harness mode: none, capped (default), or full",
    )
    parser.add_argument(
        "--harness-cap",
        type=float,
        default=0.15,
        help="Harness force cap as fraction of mg (default: 0.15)",
    )
    
    args = parser.parse_args()
    
    run_physics_test(
        motion_path=args.motion,
        model_path=args.model,
        render=args.render,
        harness_mode=args.harness,
        harness_cap=args.harness_cap,
    )


if __name__ == "__main__":
    main()
