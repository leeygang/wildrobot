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
    parser = argparse.ArgumentParser(description="Render MuJoCo scene and print home keyframe.")
    parser.add_argument(
        "--scene-file",
        "--scne-file",
        dest="scene_file",
        default=str(_default_scene_file()),
        help="Path to scene XML (default: assets/v2/scene_flat_terrain.xml).",
    )
    args = parser.parse_args()

    scene_path = _resolve_scene_file(args.scene_file)
    if not scene_path.exists():
        raise FileNotFoundError(f"scene file not found: {scene_path}")
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    _print_home_from_model(model, data, scene_path)

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
