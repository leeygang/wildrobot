#!/usr/bin/env python3
"""
Validate an aggregate USD: check referenced files' prim existence, articulation roots,
and visibility. Prints a report and exits non-zero on critical issues.

Usage:
    ./isaaclab.sh -p scripts/validate_usd.py --usd /path/to/wildrobot.usd --root /WildRobotDev [--headless]
"""

import argparse
import sys

# Parse args before launching Isaac app
ap = argparse.ArgumentParser()
ap.add_argument("--usd", required=False)
ap.add_argument("--root", default="/WildRobotDev")
ap.add_argument("--headless", action="store_true")
args_pre = ap.parse_args([]) if False else None  # placeholder to keep structure

# Launch Isaac Sim app to ensure pxr is available
def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--usd", required=True)
        parser.add_argument("--root", default="/WildRobotDev")
        parser.add_argument("--headless", action="store_true")
        return parser.parse_args()

args = parse_args()
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from pxr import Usd, UsdGeom, UsdPhysics


def find_articulation_roots(stage, root_path):
    roots = []
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim:
        return roots
    for child in root_prim.GetChildren():
        if UsdPhysics.ArticulationRootAPI(child):
            roots.append(str(child.GetPath()))
        # Nested waist pattern
        waist = stage.GetPrimAtPath(f"{child.GetPath()}/waist")
        if waist and UsdPhysics.ArticulationRootAPI(waist):
            roots.append(str(waist.GetPath()))
        waist2 = stage.GetPrimAtPath(f"{child.GetPath()}/waist/waist")
        if waist2 and UsdPhysics.ArticulationRootAPI(waist2):
            roots.append(str(waist2.GetPath()))
    return roots


def main():
    # args already parsed and app launched above

    stage = Usd.Stage.Open(args.usd)
    if not stage:
        print(f"✗ Could not open USD: {args.usd}")
        sys.exit(2)

    print(f"Validating USD: {args.usd}")
    print(f"Root: {args.root}")

    root_prim = stage.GetPrimAtPath(args.root)
    if not root_prim:
        print(f"✗ Root prim not found: {args.root}")
        # Help by listing top-level prims
        pseudo = stage.GetPseudoRoot()
        tops = [str(p.GetPath()) for p in pseudo.GetChildren()]
        print("Top-level prims:")
        for t in tops:
            print(f"  - {t}")
        sys.exit(2)

    # Check presence of visuals prim and list children as a proxy for composition
    visuals_prim = stage.GetPrimAtPath(f"{args.root}/visuals")
    if visuals_prim:
        v_children = [str(c.GetPath()) for c in visuals_prim.GetChildren()]
        print(f"Visuals prim found: {visuals_prim.GetPath()} | children: {len(v_children)}")
        for p in v_children:
            print(f"  - {p}")
    else:
        print("Visuals prim not found at 'root/visuals' — continuing")

    # Articulation roots
    art_roots = find_articulation_roots(stage, args.root)
    print(f"Articulation roots: {art_roots}")
    if not art_roots:
        print("✗ No articulation roots found under the provided root")
        sys.exit(2)

    # Visibility check on key prims
    img = UsdGeom.Imageable(root_prim)
    vis = img.ComputeVisibility()
    print(f"Root visibility: {vis}")

    # Basic transform sanity
    xfa = UsdGeom.XformCommonAPI(root_prim)
    # No direct getters; rely on authored ops presence
    authored = UsdGeom.Xformable(root_prim).GetXformOpOrderAttr().Get()
    print(f"Authored xform ops on root: {authored}")

    print("✓ Validation complete")

    # Close app
    simulation_app.close()


if __name__ == "__main__":
    main()
