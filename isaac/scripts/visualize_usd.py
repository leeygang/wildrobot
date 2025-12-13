#!/usr/bin/env python3
"""
Simple script to visualize a USD file in Isaac Sim.
"""
import argparse
import sys
import os

# Parse command line arguments FIRST (before any isaaclab imports)
parser = argparse.ArgumentParser(description="Visualize a USD file in Isaac Sim")
parser.add_argument("usd_path", nargs="?", type=str, help="Path to the USD file to visualize")
parser.add_argument("--usd", dest="usd_opt", type=str, help="Path to the USD file to visualize (optional flag)")
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
parser.add_argument("--paused", action="store_true", help="Start paused (no physics stepping)")
parser.add_argument("--height", type=float, default=0.48, help="Initial Z height for articulation root (meters)")
args_cli = parser.parse_args()
usd_path_cli = args_cli.usd_opt or args_cli.usd_path
if not usd_path_cli:
    parser.error("USD path is required. Provide positional arg or --usd")

# Capture original working directory to resolve relative USD path correctly
ORIG_CWD = os.getcwd()

# Launch the simulator BEFORE importing isaaclab modules
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows after app launch."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils import configclass
from pxr import Usd, UsdPhysics, UsdGeom, Gf
import omni.usd


@configclass
class ViewerSceneCfg(InteractiveSceneCfg):
    """Configuration for the viewer scene."""

    # Scene without pre-declared robot; we'll import USD manually and then
    # create the articulation once the prim exists under env namespace.


def main():
    """Main function."""
    # Print info
    print("\n" + "=" * 80)
    print(f"Visualizing USD: {usd_path_cli}")
    print("=" * 80 + "\n")

    # Create simulation context
    # Ensure gravity is always a real-valued vector. If paused, disable gravity with (0,0,0),
    # otherwise use standard Earth gravity so SimulationContext can create the gravity tensor.
    sim_cfg = SimulationCfg(dt=0.01, render_interval=1, gravity=(0.0, 0.0, 0.0) if args_cli.paused else (0.0, 0.0, -9.81))
    sim = SimulationContext(sim_cfg)

    # Design scene manually (ground + lights) before creating InteractiveScene
    print("[INFO] Setting up scene...")
    sim_utils.spawn_ground_plane(prim_path="/World/defaultGroundPlane", cfg=sim_utils.GroundPlaneCfg())
    sim_utils.spawn_light(
        prim_path="/World/Light",
        cfg=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.9, 0.9, 0.9)),
    )

    # Set camera view
    sim.set_camera_view(eye=(3.5, 3.5, 3.5), target=(0.0, 0.0, 0.5))

    # Create scene (without robot yet)
    scene_cfg = ViewerSceneCfg(num_envs=1, env_spacing=5.0)
    scene = InteractiveScene(scene_cfg)

    # Import USD under the env namespace, then create articulation
    # Resolve environment namespace robustly across IsaacLab versions
    env_ns_attr = getattr(scene, "env_ns", None)
    if callable(env_ns_attr):
        env_ns = env_ns_attr(0)
    elif isinstance(env_ns_attr, str):
        env_ns = env_ns_attr
    else:
        # Fallback to a common default
        env_ns = "/World/envs/env_0"
    target_root = f"{env_ns}/Robot"
    print(f"[INFO] Importing USD under: {target_root}")
    stage = omni.usd.get_context().get_stage()
    # Ensure target root prim exists
    target_prim = stage.GetPrimAtPath(target_root)
    if not target_prim:
        target_prim = stage.DefinePrim(target_root)
    # Ensure the import root has identity transform to avoid huge world offsets
    xform_api = UsdGeom.XformCommonAPI(target_prim)
    xform_api.SetTranslate((0.0, 0.0, 0.0))
    xform_api.SetRotate((0.0, 0.0, 0.0))
    xform_api.SetScale((1.0, 1.0, 1.0))
    # Add a reference to the USD file using original CWD for relative paths
    if os.path.isabs(usd_path_cli):
        usd_abs = usd_path_cli
    else:
        usd_abs = os.path.abspath(os.path.join(ORIG_CWD, usd_path_cli))
    target_prim.GetReferences().AddReference(usd_abs)
    # Introspection: dump xform ops on target import root BEFORE height
    xformable_target = UsdGeom.Xformable(target_prim)
    print("[DEBUG] Target root xformOpOrder (pre-height):", xformable_target.GetXformOpOrderAttr().Get())
    ops_pre = xformable_target.GetOrderedXformOps()
    print("[DEBUG] Target root authored ops (pre-height):", [op.GetOpType() for op in ops_pre])
    # Manual height only
    # Apply manual height
    height_to_apply = float(args_cli.height)
    print(f"[INFO] Height to apply: {height_to_apply:.4f} m")
    # Apply initial height at the import root to lift the entire hierarchy.
    # Some referenced USDs author an incompatible xform stack; force a clean stack and reset.
    # Reset the xform stack explicitly and author a fresh translate op
    try:
        xformable_target.SetXformOpOrder([])
        # Author identity rotate/scale followed by translate height
        rot_op = xformable_target.AddXformOp(UsdGeom.XformOp.TypeOrient)
        rot_op.Set(Gf.Quatd(1.0, 0.0, 0.0, 0.0))
        scale_op = xformable_target.AddXformOp(UsdGeom.XformOp.TypeScale)
        scale_op.Set(Gf.Vec3d(1.0, 1.0, 1.0))
        trans_op = xformable_target.AddXformOp(UsdGeom.XformOp.TypeTranslate)
        trans_op.Set(Gf.Vec3d(0.0, 0.0, height_to_apply))
    except Exception as e:
        print(f"[WARN] Failed to clear/author xform ops: {e}")
    # Introspection: dump xform ops and computed transform AFTER height
    print("[DEBUG] Target root xformOpOrder (post-height):", xformable_target.GetXformOpOrderAttr().Get())
    ops_post = xformable_target.GetOrderedXformOps()
    print("[DEBUG] Target root authored ops (post-height):", [op.GetOpType() for op in ops_post])
    world_xf_target = xformable_target.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    t_target = world_xf_target.ExtractTranslation()
    print("[DEBUG] Target root world translation (post-height):", (t_target[0], t_target[1], t_target[2]))

    # Create articulation pointing to waist root now that prim exists
    # Auto-detect articulation root under the imported USD
    # Common roots observed: /WildRobotDev/waist, /wildrobot/waist/waist
    # After import under {ENV}/Robot, these become children of target_root.
    # Probe likely paths and pick the first that exists with ArticulationRootAPI.
    candidate_paths = [
        f"{target_root}/waist/waist",
        f"{target_root}/waist",
        f"{target_root}/worldBody",
    ]
    sel_root = None
    for p in candidate_paths:
        prim = stage.GetPrimAtPath(p)
        if prim and UsdPhysics.ArticulationRootAPI(prim):
            sel_root = p
            break
    if sel_root is None:
        # Fallback: search children under target_root for any articulation root
        tr_children = [str(c.GetPath()) for c in stage.GetPrimAtPath(target_root).GetChildren()]
        for c in tr_children:
            prim = stage.GetPrimAtPath(c)
            if prim and UsdPhysics.ArticulationRootAPI(prim):
                sel_root = c
                break
    if sel_root is None:
        raise RuntimeError("No articulation root found under import root; please inspect USD hierarchy.")

    waist_root = sel_root
    robot_cfg = ArticulationCfg(
        prim_path=waist_root,
        # Do not override orientation; use authored USD pose
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        actuators={
            "joints": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=0.0,
                damping=0.0,
            ),
        },
    )
    # Register articulation directly on scene
    scene._articulations["robot"] = Articulation(robot_cfg)

    # Ensure visibility on root and selected articulation
    UsdGeom.Imageable(target_prim).MakeVisible()
    waist_prim = stage.GetPrimAtPath(waist_root)
    if waist_prim:
        UsdGeom.Imageable(waist_prim).MakeVisible()
        # Introspection: dump articulation (waist) xform state
        waist_xformable = UsdGeom.Xformable(waist_prim)
        print("[DEBUG] Articulation root xformOpOrder:", waist_xformable.GetXformOpOrderAttr().Get())
        ops_waist = waist_xformable.GetOrderedXformOps()
        print("[DEBUG] Articulation root authored ops:", [op.GetOpType() for op in ops_waist])
        world_xf_waist = waist_xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        t_waist = world_xf_waist.ExtractTranslation()
        print("[DEBUG] Articulation root world translation (post-height):", (t_waist[0], t_waist[1], t_waist[2]))

    # Play the simulator to activate physics
    sim.reset()
    # No auto-height or fallback logic
    # Render once to ensure stage updates are visible
    sim.render()

    # Dynamically frame camera using world transform of articulation root
    if waist_prim and UsdGeom.Xformable(waist_prim):
        xformable = UsdGeom.Xformable(waist_prim)
        world_xf = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        # Extract translation from transform matrix
        translate = world_xf.ExtractTranslation()
        center = (translate[0], translate[1], translate[2])
        # Use a fixed radius for framing if sizes are unknown
        radius = 1.0
        eye = (center[0] + radius * 2.0, center[1] + radius * 1.5, center[2] + radius * 1.5)
        target = center
        sim.set_camera_view(eye=eye, target=target)

    # Verify imported prims and articulation roots under the target
    target_root = f"{env_ns}/Robot"
    print(f"[VERIFY] Target import root: {target_root}")
    # Obtain current stage from USD
    stage = omni.usd.get_context().get_stage()
    # List children under target root using USD APIs
    target_prim = stage.GetPrimAtPath(target_root)
    child_prims = list(target_prim.GetChildren()) if target_prim else []
    children = [str(p.GetPath()) for p in child_prims]
    print(f"[VERIFY] Children under {target_root}: {children}")
    # Find articulation roots
    articulation_roots = []
    for child in children:
        prim = stage.GetPrimAtPath(child)
        if prim and UsdPhysics.ArticulationRootAPI(prim):
            articulation_roots.append(child)
        # Also check nested waist path if present
        waist_path = f"{child}/waist"
        prim_waist = stage.GetPrimAtPath(waist_path)
        if prim_waist and UsdPhysics.ArticulationRootAPI(prim_waist):
            articulation_roots.append(waist_path)
        waist_waist_path = f"{child}/waist/waist"
        prim_waist_waist = stage.GetPrimAtPath(waist_waist_path)
        if prim_waist_waist and UsdPhysics.ArticulationRootAPI(prim_waist_waist):
            articulation_roots.append(waist_waist_path)
    print(f"[VERIFY] Articulation roots found: {articulation_roots}")
    print(f"[VERIFY] Selected articulation root: {waist_root}")

    # Now we can access robot info after reset
    robot = scene.articulations["robot"]
    print("\n" + "=" * 80)
    print("Robot Information:")
    print(f"  Number of bodies: {robot.num_bodies}")
    print(f"  Number of joints: {robot.num_joints}")
    print(f"  Joint names: {robot.joint_names}")
    print(f"  Body names: {robot.body_names}")
    print(f"\nActuators:")
    for actuator_name, actuator in robot.actuators.items():
        print(f"  - {actuator_name}: {len(actuator.joint_names)} joints")
        print(f"    Joints: {actuator.joint_names}")
    print("=" * 80)

    # Simulation loop
    print("\n" + "=" * 80)
    print("Controls:")
    print("  - Close window or press ESC to exit")
    print("  - Use mouse to rotate/pan/zoom camera")
    print("=" * 80 + "\n")

    # Simulation loop
    sim_time = 0.0
    count = 0
    while simulation_app.is_running():
        # If paused, don't advance physics; just render
        if not args_cli.paused:
            # Step simulation without periodic forced resets
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim.get_physics_dt())
        else:
            sim.render()

        # Print status periodically
        if count % 100 == 0:
            print(f"[INFO] Simulation time: {sim_time:.2f}s", end="\r")

        if not args_cli.paused:
            sim_time += sim.get_physics_dt()
            count += 1


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
