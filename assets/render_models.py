#!/usr/bin/env python3
"""Render MuJoCo scene files and print home keyframe pose."""

from __future__ import annotations

import argparse
import math
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import mujoco.viewer


def _default_scene_file() -> Path:
    return Path(__file__).resolve().parent / "v2" / "scene_flat_terrain.xml"


def _resolve_scene_file(scene_file: str | Path) -> Path:
    candidate = Path(scene_file)
    if candidate.exists():
        return candidate.resolve()
    repo_relative = Path(__file__).resolve().parent.parent / candidate
    return repo_relative.resolve()


def _find_root_qpos_addr(model: mujoco.MjModel) -> int:
    """Match training convention: root height = qpos[root_qpos_addr + 2]."""
    root_joint_name = "waist_freejoint"
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, root_joint_name)
    if joint_id >= 0:
        return int(model.jnt_qposadr[joint_id])

    # Fallback: first free joint in model
    for jid in range(model.njnt):
        if int(model.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_FREE):
            return int(model.jnt_qposadr[jid])

    # Final fallback for floating-base robots where root is at qpos[0:7]
    return 0


def _get_training_height(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    qpos_addr = _find_root_qpos_addr(model)
    return float(data.qpos[qpos_addr + 2])


def _ctrl_from_qpos(model: mujoco.MjModel, qpos) -> "list[float]":
    """Build the position-actuator ctrl vector that holds the given qpos.

    For every actuator, look up the joint it drives and read that joint's
    qpos slot.  Mirrors the pattern in ``assets/resettle_keyframes.py`` and
    ``assets/derive_walk_ready_home.py``: without this, the viewer would
    step physics with ``ctrl=0`` (drive every joint to angle 0), and the
    robot would visibly extend its legs to a straight-leg pose regardless
    of the initial keyframe — masking the actual home pose on screen.
    """
    out: list[float] = []
    for aid in range(model.nu):
        jid = int(model.actuator_trnid[aid, 0])
        qaddr = int(model.jnt_qposadr[jid])
        out.append(float(qpos[qaddr]))
    return out


def _format_qpos(qpos: list[float] | tuple[float, ...] | object) -> str:
    return " ".join(f"{float(value):.8g}" for value in qpos)


def _find_home_key_id(model: mujoco.MjModel) -> int:
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id >= 0:
        return int(key_id)
    if model.nkey > 0:
        return 0
    return -1


def _read_home_qpos_from_keyframes_xml(keyframes_xml: Path) -> list[float] | None:
    if not keyframes_xml.exists():
        return None
    root = ET.parse(keyframes_xml).getroot()
    key = root.find('./keyframe/key[@name="home"]')
    if key is None:
        return None
    qpos_str = key.attrib.get("qpos", "")
    if not qpos_str.strip():
        return None
    return [float(x) for x in qpos_str.split()]


def _load_home_qpos(model: mujoco.MjModel, scene_path: Path) -> tuple[str, list[float]] | None:
    """Return (source, qpos) for the home keyframe.

    Prefer keyframes embedded in the loaded MuJoCo model; fall back to a sibling
    keyframes.xml next to the scene/variant if present.
    """
    key_id = _find_home_key_id(model)
    if key_id >= 0:
        return ("model.key_qpos", [float(x) for x in model.key_qpos[key_id]])

    # Typical layout: assets/v2/scene_flat_terrain.xml + assets/v2/keyframes.xml
    fallback = scene_path.with_name("keyframes.xml")
    qpos = _read_home_qpos_from_keyframes_xml(fallback)
    if qpos is None:
        return None
    if len(qpos) != int(model.nq):
        return None
    return (str(fallback), qpos)


def _quat_wxyz_to_yaw_deg(w: float, x: float, y: float, z: float) -> float:
    """Yaw (Z) from quaternion in MuJoCo freejoint qpos order (w,x,y,z)."""
    # Standard yaw extraction (assuming right-handed, Z-up).
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.degrees(math.atan2(siny_cosp, cosy_cosp))


def _print_home_joint_table(model: mujoco.MjModel, home_qpos: list[float]) -> None:
    """Print hinge joints: name, home deg, range deg, with left/right pairs adjacent."""
    root_qpos_addr = _find_root_qpos_addr(model)
    height_m = float(home_qpos[root_qpos_addr + 2])
    qw, qx, qy, qz = (float(home_qpos[root_qpos_addr + i]) for i in range(3, 7))
    yaw_deg = _quat_wxyz_to_yaw_deg(qw, qx, qy, qz)

    rad2deg = 180.0 / math.pi
    hinge: dict[str, tuple[float, float, float]] = {}
    for jid in range(model.njnt):
        if int(model.jnt_type[jid]) != int(mujoco.mjtJoint.mjJNT_HINGE):
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if not name:
            continue
        adr = int(model.jnt_qposadr[jid])
        home_deg = float(home_qpos[adr]) * rad2deg
        rmin, rmax = map(float, model.jnt_range[jid])
        hinge[name] = (home_deg, rmin * rad2deg, rmax * rad2deg)

    ordered: list[str] = []
    seen: set[str] = set()
    for name in sorted(hinge):
        if name in seen:
            continue
        if name.startswith("left_"):
            pair = "right_" + name[len("left_") :]
            if pair in hinge:
                ordered.append(name)
                ordered.append(pair)
                seen.add(name)
                seen.add(pair)
                continue
        if name.startswith("right_"):
            pair = "left_" + name[len("right_") :]
            if pair in hinge:
                continue
        ordered.append(name)
        seen.add(name)

    print("[render_models] home joint table:")
    print(f"[render_models] home height_m={height_m:.6f} init_yaw_deg={yaw_deg:.6f}")
    print("joint_name | home (deg) | range (deg)")
    for name in ordered:
        home_deg, rmin_deg, rmax_deg = hinge[name]
        print(f"{name} | {home_deg:.6f} | {rmin_deg:.6f}..{rmax_deg:.6f}")


def _print_collision_capsules(
    model: mujoco.MjModel, data: mujoco.MjData
) -> None:
    """Print dimensions + world endpoints for every collision-class capsule.

    For each capsule, also list visual mesh geoms attached to the same body
    so a mis-placed primitive proxy (e.g. a finger capsule that doesn't sit
    inside its visual finger mesh) is immediately visible by comparing the
    capsule's world endpoints to the visual mesh's world center.
    """
    # Index visual mesh geoms by body for the alignment hint.
    visual_meshes_by_body: dict[int, list[tuple[int, str, list[float]]]] = {}
    for gid in range(model.ngeom):
        if int(model.geom_type[gid]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            continue
        if int(model.geom_contype[gid]) != 0 or int(model.geom_conaffinity[gid]) != 0:
            continue  # collision-participating meshes aren't "visual"
        bid = int(model.geom_bodyid[gid])
        mesh_id = int(model.geom_dataid[gid])
        mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id) or "?"
        visual_meshes_by_body.setdefault(bid, []).append(
            (gid, mesh_name, data.geom_xpos[gid].tolist())
        )

    print("[render_models] collision capsules (--show-capsule):")
    any_found = False
    for gid in range(model.ngeom):
        if int(model.geom_type[gid]) != int(mujoco.mjtGeom.mjGEOM_CAPSULE):
            continue
        if int(model.geom_contype[gid]) == 0 and int(model.geom_conaffinity[gid]) == 0:
            continue
        any_found = True
        bid = int(model.geom_bodyid[gid])
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or "?"
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        radius = float(model.geom_size[gid, 0])
        half_len = float(model.geom_size[gid, 1])
        # Capsule axis is the geom's local +Z = third column of geom_xmat.
        center = data.geom_xpos[gid]
        z_axis = data.geom_xmat[gid].reshape(3, 3)[:, 2]
        cap0 = center - half_len * z_axis
        cap1 = center + half_len * z_axis
        # Local (body-frame) pos/quat the MJCF wrote, for cross-check
        # against `fromto=` values in the XML.
        lpos = model.geom_pos[gid]
        lquat = model.geom_quat[gid]
        print(f"  gid={gid} body={body!r} name={gname!r}")
        print(
            f"    radius={radius:.5f}  half_len={half_len:.5f}  "
            f"end_to_end={2 * (half_len + radius):.5f}"
        )
        print(
            f"    world center = ({center[0]:+.4f}, {center[1]:+.4f}, {center[2]:+.4f})"
        )
        print(f"    world cap0   = ({cap0[0]:+.4f}, {cap0[1]:+.4f}, {cap0[2]:+.4f})")
        print(f"    world cap1   = ({cap1[0]:+.4f}, {cap1[1]:+.4f}, {cap1[2]:+.4f})")
        print(
            f"    body-frame pos  = ({lpos[0]:+.4f}, {lpos[1]:+.4f}, {lpos[2]:+.4f})"
        )
        print(
            f"    body-frame quat = ({lquat[0]:+.4f}, {lquat[1]:+.4f}, "
            f"{lquat[2]:+.4f}, {lquat[3]:+.4f})"
        )
        for vid, vname, vpos in visual_meshes_by_body.get(bid, []):
            print(
                f"    visual mesh on same body: gid={vid} mesh={vname!r} "
                f"world_pos=({vpos[0]:+.4f}, {vpos[1]:+.4f}, {vpos[2]:+.4f})"
            )
    if not any_found:
        print("  (no collision capsules found)")


def _print_home_from_model(
    model: mujoco.MjModel, data: mujoco.MjData, scene_path: Path
) -> None:
    loaded = _load_home_qpos(model, scene_path)
    if loaded is None:
        print(f"[render_models] no home keyframe found (model or {scene_path.with_name('keyframes.xml')})")
        return
    source, qpos = loaded
    key_name = "home"
    print("[render_models] home frame from model:")
    print(f"[render_models] measured height (training convention qpos[root+2]): {_get_training_height(model, data):.6g}")
    print(f'<key name="{key_name}" qpos="{_format_qpos(qpos)}" />')
    print(f"[render_models] home qpos source: {source}")
    _print_home_joint_table(model, qpos)
    print("[render_models] press H in viewer to print current pose as home keyframe")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render MuJoCo scene and print home keyframe.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        # Suppress the auto-added -h/--help so we can register a single
        # help action with -h / --h / --help as explicit aliases.
        # Without --h being explicit, it would still work today via
        # argparse prefix matching, but a future ``--height`` /
        # ``--hide`` flag would make ``--h`` silently ambiguous.
        add_help=False,
    )
    parser.add_argument(
        "-h", "--h", "--help",
        action="help",
        default=argparse.SUPPRESS,
        help="Show this help message and exit.",
    )
    parser.add_argument(
        "--scene-file",
        "--scne-file",
        dest="scene_file",
        default=str(_default_scene_file()),
        help="Path to scene XML (default: assets/v2/scene_flat_terrain.xml).",
    )
    parser.add_argument(
        "--init",
        dest="init",
        choices=("home", "mjcf"),
        default="home",
        help=(
            "Initial pose for the viewer. "
            "'home' resets MjData to the `home` keyframe in keyframes.xml "
            "(locomotion-ready pose used as PPO's nominal action); "
            "'mjcf' uses the MJCF default qpos0 (model's straight-leg/zero pose) "
            "so you can compare against the unmodified asset."
        ),
    )
    parser.add_argument(
        "--show-capsule",
        "--show_capsule",
        dest="show_capsule",
        action="store_true",
        help=(
            "Print dimensions + world-frame endpoints of every collision-class "
            "capsule geom, alongside any visual mesh geoms on the same body, "
            "so a mis-placed primitive proxy (e.g. a finger capsule that "
            "doesn't sit inside the visual finger mesh) is immediately visible."
        ),
    )
    args = parser.parse_args()

    scene_path = _resolve_scene_file(args.scene_file)
    if not scene_path.exists():
        raise FileNotFoundError(f"scene file not found: {scene_path}")
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    if args.init == "home":
        home_key_id = _find_home_key_id(model)
        if home_key_id >= 0:
            mujoco.mj_resetDataKeyframe(model, data, home_key_id)
        else:
            print("[render_models] no home keyframe found; falling back to MJCF qpos0")
            mujoco.mj_resetData(model, data)
    else:
        mujoco.mj_resetData(model, data)
    # Pin position-actuator ctrl to the initial qpos so the per-frame
    # ``mj_step`` in the viewer loop below holds the chosen init pose
    # under PD control.  Without this, ctrl stays at zero and the
    # actuators drive every joint to angle 0 — the home keyframe pose
    # is visible for one frame and then physics extends the legs to a
    # straight-leg ctrl=0 settled state, regardless of the keyframe.
    data.ctrl[:] = _ctrl_from_qpos(model, data.qpos)
    mujoco.mj_forward(model, data)
    print(f"[render_models] viewer initial pose: --init={args.init}")
    _print_home_from_model(model, data, scene_path)
    if args.show_capsule:
        _print_collision_capsules(model, data)

    def on_key(keycode: int) -> None:
        if keycode == ord("H"):
            print("[render_models] current pose as home frame:")
            print(f"[render_models] measured height (training convention qpos[root+2]): {_get_training_height(model, data):.6g}")
            print(f'<key name="home" qpos="{_format_qpos(data.qpos)}" />')

    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.002)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
