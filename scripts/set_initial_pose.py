#!/usr/bin/env python3
"""Interactive MuJoCo viewer for setting WildRobot initial pose.

Usage:
    python scripts/set_initial_pose.py [--pause]

    --pause: Start with simulation paused (only works on Linux or with mjpython on macOS)

    On macOS, if launch_passive fails, the script will use blocking mode.
    Alternatively, install mjpython: pip install mujoco[mjpython]
    Then run: mjpython scripts/set_initial_pose.py --pause

Controls:
    - Mouse drag: Rotate view
    - Scroll: Zoom
    - Double-click body: Select and drag to move
    - Ctrl+drag: Pan view
    - Space: Pause/unpause simulation
    - Backspace: Reset to home position
    - P: Print current qpos (copy this for keyframe) [passive mode only]
    - S: Save current qpos to file [passive mode only]
    - Q/Esc: Quit

After positioning the robot:
    1. Press 'P' to print the qpos (or wait for viewer to close in blocking mode)
    2. Copy the qpos values to scene_flat_terrain.xml keyframe
"""

import argparse
import platform
import sys
from pathlib import Path

import mujoco
import mujoco.viewer


def print_qpos(model: mujoco.MjModel, data: mujoco.MjData, prefix: str = "") -> str:
    """Print current qpos in keyframe format and return as string."""
    qpos_str = " ".join([f"{x:.6f}" for x in data.qpos])

    print(f"\n{prefix}=== Current qpos ({model.nq} values) ===")
    keyframe_line = f'<key name="home" qpos="{qpos_str}" />'
    print(keyframe_line)
    print()

    # Also print as Python array for reference
    print("Python array:")
    print(f"qpos = [{', '.join([f'{x:.6f}' for x in data.qpos])}]")
    print()

    # Print joint-by-joint breakdown
    print("Joint breakdown:")
    print(f"  Position (xyz): {data.qpos[0:3]}")
    print(f"  Orientation (wxyz quat): {data.qpos[3:7]}")
    for i in range(model.njnt):
        if model.jnt(i).type[0] == 0:  # Skip free joint
            continue
        qpos_adr = model.jnt_qposadr[i]
        print(f"  {model.jnt(i).name}: {data.qpos[qpos_adr]:.6f}")

    return keyframe_line


def save_qpos(
    model: mujoco.MjModel, data: mujoco.MjData, saved_poses: list, project_root: Path
) -> None:
    """Save current qpos to file."""
    qpos_str = " ".join([f"{x:.6f}" for x in data.qpos])
    saved_poses.append(data.qpos.copy())

    output_file = project_root / "data" / "saved_poses.txt"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "a") as f:
        f.write(f'<key name="pose_{len(saved_poses)}" qpos="{qpos_str}" />\n')

    print(f"\n✓ Saved pose #{len(saved_poses)} to {output_file}")


def run_passive_mode(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    project_root: Path,
    start_paused: bool = False,
) -> None:
    """Run viewer in passive (non-blocking) mode with keyboard callbacks."""
    saved_poses = []
    paused = [start_paused]  # Use list to allow mutation in nested function

    def key_callback(keycode):
        """Handle keyboard input."""
        if keycode == ord("P") or keycode == ord("p"):
            print_qpos(model, data)
        elif keycode == ord("S") or keycode == ord("s"):
            save_qpos(model, data, saved_poses, project_root)
        elif keycode == ord(" "):  # Spacebar
            paused[0] = not paused[0]
            print(f"Simulation {'PAUSED' if paused[0] else 'RUNNING'}")

    print("\nLaunching MuJoCo viewer (passive mode)...")
    if start_paused:
        print("Starting PAUSED - press SPACE to unpause")
    print("Position the robot, then press 'P' to print qpos\n")

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        # Set initial camera view
        viewer.cam.azimuth = 160
        viewer.cam.elevation = -20
        viewer.cam.distance = 2.5
        viewer.cam.lookat[:] = [0, 0, 0.5]

        while viewer.is_running():
            if not paused[0]:
                mujoco.mj_step(model, data)
            viewer.sync()

    print("\nViewer closed.")
    if saved_poses:
        print(f"Saved {len(saved_poses)} poses to data/saved_poses.txt")


def run_blocking_mode(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Run viewer in blocking mode (simpler, works on all platforms)."""
    print("\nLaunching MuJoCo viewer (blocking mode)...")
    print("\nInstructions:")
    print("  1. Pause simulation with SPACEBAR")
    print("  2. Double-click on robot parts to select and drag them")
    print("  3. Use Ctrl+Right-click to apply forces")
    print("  4. When done, close the viewer window")
    print("  5. The final qpos will be printed automatically")
    print()

    # Launch blocking viewer
    mujoco.viewer.launch(model, data)

    # Print qpos after viewer closes
    print_qpos(model, data, prefix="[FINAL] ")


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Interactive MuJoCo viewer for setting WildRobot initial pose"
    )
    parser.add_argument(
        "--pause",
        action="store_true",
        help="Start with simulation paused (only works on Linux or with mjpython)",
    )
    args = parser.parse_args()

    # Find model path
    project_root = Path(__file__).parent.parent
    scene_path = project_root / "assets" / "scene_flat_terrain.xml"

    if not scene_path.exists():
        scene_path = project_root / "assets" / "scene.xml"

    print(f"Loading model from: {scene_path}")

    # Load model
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)

    # Print model info
    print("\n=== Model Info ===")
    print(f"nq (qpos size): {model.nq}")
    print(f"nv (qvel size): {model.nv}")
    print(f"nu (actuators): {model.nu}")
    print(f"njnt (joints): {model.njnt}")

    print("\n=== Joints ===")
    for i in range(model.njnt):
        jnt = model.jnt(i)
        qpos_adr = model.jnt_qposadr[i]
        print(f"  {i}: {jnt.name} (type={jnt.type[0]}, qposadr={qpos_adr})")

    print(f"\n=== Current qpos (from keyframe) ===")
    print(f"  {list(data.qpos)}")

    print("=" * 50)

    # Detect platform and choose viewer mode
    is_macos = platform.system() == "Darwin"

    # Check if running under mjpython (has special attribute)
    running_under_mjpython = hasattr(sys, "_mujoco_mjpython")

    if is_macos and not running_under_mjpython:
        # Try passive mode first, fall back to blocking if it fails
        try:
            run_passive_mode(model, data, project_root, start_paused=args.pause)
        except RuntimeError as e:
            if "mjpython" in str(e):
                print("\n" + "=" * 50)
                print("Note: For full functionality on macOS, run with mjpython:")
                print("  mjpython scripts/set_initial_pose.py --pause")
                print("=" * 50)
                print("\nFalling back to blocking mode...")
                print("(--pause flag not supported in blocking mode)")
                run_blocking_mode(model, data)
            else:
                raise
    else:
        # Linux or running under mjpython - use passive mode
        run_passive_mode(model, data, project_root, start_paused=args.pause)


if __name__ == "__main__":
    main()
