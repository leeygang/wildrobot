#!/usr/bin/env python3
"""
Inspect a USD file to list articulation roots, rigid bodies, joints, and notable prim paths.

Usage:
  ./isaaclab.sh -p /home/leeygang/projects/wildrobot/scripts/inspect_usd.py /path/to/file.usd

Notes:
  - This script uses Isaac Sim's Python (via isaaclab.sh) to access pxr APIs reliably.
  - It does not launch a simulation; it only reads the USD stage.
"""

import argparse

parser = argparse.ArgumentParser(description="Inspect articulation and physics info in a USD file")
parser.add_argument("usd_path", type=str, help="Path to a USD/USDA/USDC file")
parser.add_argument("--out", "--log", dest="out_path", type=str, default=None,
                    help="Optional path to write the inspection output (e.g., /tmp/usd_inspect.log)")
args = parser.parse_args()

# Import pxr from Isaac Sim Python environment
# Ensure Isaac Sim app is launched so pxr modules are available
try:
    from isaaclab.app import AppLauncher  # type: ignore
except Exception:
    AppLauncher = None

if AppLauncher is not None:
    _app = AppLauncher({"headless": True}).app

from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema  # type: ignore

def main():
    # Prepare logger (writes to stdout and optional file)
    log_file = None
    def log(msg: str = ""):
        print(msg)
        if log_file:
            try:
                log_file.write(msg + "\n")
                log_file.flush()
            except Exception:
                pass

    if args.out_path:
        try:
            log_file = open(args.out_path, "w", encoding="utf-8")
        except Exception as e:
            print(f"[WARN] Failed to open log file '{args.out_path}': {e}")
            log_file = None
    log("\n" + "=" * 80)
    log(f"Inspecting USD: {args.usd_path}")
    log("=" * 80 + "\n")

    stage = Usd.Stage.Open(args.usd_path)
    if stage is None:
        log(f"[ERROR] Failed to open USD: {args.usd_path}")
        return

    # Collect articulation roots
    articulation_paths = []
    for prim in stage.Traverse():
        art_api = UsdPhysics.ArticulationRootAPI(prim)
        if art_api and art_api.GetPrim().IsValid():
            articulation_paths.append(str(prim.GetPath()))

    # Collect joints (common USD Physics joint schemas)
    joint_paths = []
    joint_types = (
        UsdPhysics.RevoluteJoint,
        UsdPhysics.PrismaticJoint,
        UsdPhysics.FixedJoint,
        UsdPhysics.DistanceJoint,
        UsdPhysics.DriveAPI,  # drive API applied to joints
    )
    for prim in stage.Traverse():
        for JT in joint_types:
            try:
                api = JT(prim)
            except Exception:
                api = None
            if api and getattr(api, "GetPrim", lambda: None)():
                if api.GetPrim().IsValid():
                    joint_paths.append(str(prim.GetPath()))
                    break

    # Collect rigid bodies
    rigid_body_paths = []
    for prim in stage.Traverse():
        rb = UsdPhysics.RigidBodyAPI.Apply(prim) if False else UsdPhysics.RigidBodyAPI(prim)
        if rb and rb.GetPrim().IsValid():
            rigid_body_paths.append(str(prim.GetPath()))

    # Collect interesting names like worldBody/waist/Robot
    interesting = []
    for prim in stage.Traverse():
        p = str(prim.GetPath())
        if any(k in p for k in ["worldBody", "waist", "Robot", "wildrobot"]):
            interesting.append((p, prim.GetTypeName()))

    log("Articulation Roots:")
    if articulation_paths:
        for p in articulation_paths:
            log(f"  - {p}")
    else:
        log("  (none found)")

    log("\nJoints:")
    if joint_paths:
        for p in joint_paths[:100]:
            log(f"  - {p}")
        if len(joint_paths) > 100:
            log(f"  ... ({len(joint_paths)-100} more)")
    else:
        log("  (none found)")

    log("\nRigid Bodies:")
    if rigid_body_paths:
        for p in rigid_body_paths[:100]:
            log(f"  - {p}")
        if len(rigid_body_paths) > 100:
            log(f"  ... ({len(rigid_body_paths)-100} more)")
    else:
        log("  (none found)")

    log("\nNotable Prims (worldBody/waist/Robot/wildrobot):")
    if interesting:
        for p, t in interesting:
            log(f"  - {p} (Type={t})")
    else:
        log("  (none found)")

    log("\nDone.")

    # Close Isaac Sim app cleanly if we launched it
    try:
        if AppLauncher is not None:
            _app.close()
    except Exception:
        pass

    if log_file:
        try:
            log_file.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
