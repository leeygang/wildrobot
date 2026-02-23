#!/usr/bin/env python3
"""Render MuJoCo scene files and print home keyframe pose."""

from __future__ import annotations

import argparse
import time
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


def _format_qpos(qpos: list[float] | tuple[float, ...] | object) -> str:
    return " ".join(f"{float(value):.8g}" for value in qpos)


def _find_home_key_id(model: mujoco.MjModel) -> int:
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id >= 0:
        return int(key_id)
    if model.nkey > 0:
        return 0
    return -1


def _print_home_from_model(model: mujoco.MjModel, scene_path: Path) -> None:
    key_id = _find_home_key_id(model)
    if key_id < 0:
        print(f"[render_models] no keyframe found in scene: {scene_path}")
        return
    key_name = "home" if key_id == mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home") else "key_0"
    qpos = model.key_qpos[key_id]
    print("[render_models] home frame from model:")
    print(f'<key name="{key_name}" qpos="{_format_qpos(qpos)}" />')
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
    _print_home_from_model(model, scene_path)

    def on_key(keycode: int) -> None:
        if keycode == ord("H"):
            print("[render_models] current pose as home frame:")
            print(f'<key name="home" qpos="{_format_qpos(data.qpos)}" />')

    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.002)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
