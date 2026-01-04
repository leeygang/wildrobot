#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class Thresholds:
    pos_tol: float
    quat_tol: float
    friction_tol: float
    size_tol: float
    mass_rel_tol: float
    inertia_rel_tol: float
    mesh_pos_abs_max: float
    contact_penetration_tol: float


def _repo_root() -> Path:
    # assets/validate_model.py -> repo root
    return Path(__file__).resolve().parents[1]


def _pair_name(name: str) -> Optional[str]:
    if name.startswith("left_"):
        return "right_" + name[len("left_") :]
    if name.startswith("right_"):
        return "left_" + name[len("right_") :]
    return None


def _rel_diff(a: float, b: float) -> float:
    denom = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / denom


def _max_rel_diff_vec(a, b) -> float:
    import numpy as np

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-12)
    return float(np.max(np.abs(a - b) / denom))


def _load_robot_config(robot_config_yaml: Path):
    sys.path.insert(0, str(_repo_root()))
    from assets.robot_config import clear_robot_config_cache, load_robot_config

    clear_robot_config_cache()
    return load_robot_config(robot_config_yaml)


def _load_mujoco_model(xml_path: Path):
    import mujoco

    return mujoco.MjModel.from_xml_path(str(xml_path))


def _infer_robot_xml_from_scene(scene_xml: Path) -> Path:
    import xml.etree.ElementTree as ET

    root = ET.parse(scene_xml).getroot()
    includes = [inc.get("file") for inc in root.findall("include") if inc.get("file")]
    if not includes:
        raise ValueError(f"No <include file=...> found in scene xml: {scene_xml}")

    # Heuristic: prefer the first include that isn't keyframes.xml.
    for inc in includes:
        if Path(inc).name != "keyframes.xml":
            return (scene_xml.parent / inc).resolve()

    # Fallback: first include.
    return (scene_xml.parent / includes[0]).resolve()


def _validate_robot_config_matches_scene(
    *,
    scene_xml: Path,
    robot_config_yaml: Path,
    robot_xml: Path,
) -> None:
    import yaml

    raw = yaml.safe_load(robot_config_yaml.read_text())
    generated_from = raw.get("generated_from")
    if generated_from is None:
        raise ValueError(f"{robot_config_yaml} missing 'generated_from'")
    if Path(generated_from).name != robot_xml.name:
        raise ValueError(
            "robot_config.yaml does not match scene include: "
            f"generated_from={generated_from} but scene includes {robot_xml.name} "
            f"(scene={scene_xml})"
        )


def _validate_toe_heel(
    model,
    robot_config,
    thresholds: Thresholds,
) -> None:
    import mujoco
    import numpy as np

    left_foot = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, robot_config.left_foot_body
    )
    right_foot = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, robot_config.right_foot_body
    )
    if left_foot < 0 or right_foot < 0:
        raise ValueError(
            f"Foot bodies not found: left='{robot_config.left_foot_body}', right='{robot_config.right_foot_body}'"
        )

    left_toe, left_heel, right_toe, right_heel = robot_config.get_foot_geom_names()

    def geom_info(name: str):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid < 0:
            raise ValueError(f"Foot geom not found: {name}")
        return {
            "gid": gid,
            "body": int(model.geom_bodyid[gid]),
            "pos": model.geom_pos[gid].copy(),
            "quat": model.geom_quat[gid].copy(),
            "friction": model.geom_friction[gid].copy(),
            "size": model.geom_size[gid].copy(),
            "type": int(model.geom_type[gid]),
            "dataid": int(model.geom_dataid[gid]),
        }

    lt = geom_info(left_toe)
    lh = geom_info(left_heel)
    rt = geom_info(right_toe)
    rh = geom_info(right_heel)

    if lt["body"] != left_foot or lh["body"] != left_foot:
        raise ValueError(
            f"Left toe/heel geoms not under left foot body: {left_toe}->{lt['body']} {left_heel}->{lh['body']}"
        )
    if rt["body"] != right_foot or rh["body"] != right_foot:
        raise ValueError(
            f"Right toe/heel geoms not under right foot body: {right_toe}->{rt['body']} {right_heel}->{rh['body']}"
        )

    if not np.allclose(lt["pos"], rt["pos"], atol=thresholds.pos_tol, rtol=0.0):
        raise ValueError(
            f"Toe geom_pos mismatch: {left_toe} {lt['pos']} vs {right_toe} {rt['pos']}"
        )
    if not np.allclose(lh["pos"], rh["pos"], atol=thresholds.pos_tol, rtol=0.0):
        raise ValueError(
            f"Heel geom_pos mismatch: {left_heel} {lh['pos']} vs {right_heel} {rh['pos']}"
        )

    if not np.allclose(lt["quat"], rt["quat"], atol=thresholds.quat_tol, rtol=0.0):
        raise ValueError(
            f"Toe geom_quat mismatch: {left_toe} {lt['quat']} vs {right_toe} {rt['quat']}"
        )
    if not np.allclose(lh["quat"], rh["quat"], atol=thresholds.quat_tol, rtol=0.0):
        raise ValueError(
            f"Heel geom_quat mismatch: {left_heel} {lh['quat']} vs {right_heel} {rh['quat']}"
        )

    if not np.allclose(lt["size"], rt["size"], atol=thresholds.size_tol, rtol=0.0):
        raise ValueError(
            f"Toe geom_size mismatch: {left_toe} {lt['size']} vs {right_toe} {rt['size']}"
        )
    if not np.allclose(lh["size"], rh["size"], atol=thresholds.size_tol, rtol=0.0):
        raise ValueError(
            f"Heel geom_size mismatch: {left_heel} {lh['size']} vs {right_heel} {rh['size']}"
        )

    if not np.allclose(
        lt["friction"], rt["friction"], atol=thresholds.friction_tol, rtol=0.0
    ):
        raise ValueError(
            f"Toe geom_friction mismatch: {left_toe} {lt['friction']} vs {right_toe} {rt['friction']}"
        )
    if not np.allclose(
        lh["friction"], rh["friction"], atol=thresholds.friction_tol, rtol=0.0
    ):
        raise ValueError(
            f"Heel geom_friction mismatch: {left_heel} {lh['friction']} vs {right_heel} {rh['friction']}"
        )

    # Mesh-local origin sanity check (historically the main failure mode).
    mesh_pos = getattr(model, "mesh_pos", None)
    if mesh_pos is not None:
        for label, info in (
            (left_toe, lt),
            (left_heel, lh),
            (right_toe, rt),
            (right_heel, rh),
        ):
            if info["type"] != int(mujoco.mjtGeom.mjGEOM_MESH):
                continue
            mesh_id = info["dataid"]
            if mesh_id < 0:
                continue
            mp = mesh_pos[mesh_id]
            if float(np.max(np.abs(mp))) > thresholds.mesh_pos_abs_max:
                raise ValueError(f"{label} mesh_pos too large (bad mesh origin?): {mp}")


def _validate_mass_inertia_symmetry(model, thresholds: Thresholds) -> None:
    import mujoco
    import numpy as np

    # Pair left_* and right_* bodies by name and compare mass + inertia.
    for body_id in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if not name or not name.startswith("left_"):
            continue
        pair = _pair_name(name)
        if not pair:
            continue
        pair_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, pair)
        if pair_id < 0:
            continue
        m_left = float(model.body_mass[body_id])
        m_right = float(model.body_mass[pair_id])
        if _rel_diff(m_left, m_right) > thresholds.mass_rel_tol:
            raise ValueError(f"Body mass mismatch: {name}={m_left} vs {pair}={m_right}")

        # body_inertia is (3,) principal moments in body frame
        i_left = model.body_inertia[body_id]
        i_right = model.body_inertia[pair_id]
        if _max_rel_diff_vec(i_left, i_right) > thresholds.inertia_rel_tol:
            raise ValueError(
                f"Body inertia mismatch: {name}={i_left} vs {pair}={i_right}"
            )


def _validate_left_right_named_collision_geoms(model, thresholds: Thresholds) -> None:
    """Validate left_* / right_* named collision geoms have matching parameters.

    We intentionally do NOT enforce mirrored positions/orientations here, since
    those depend on body frame conventions. Instead we enforce that physical
    contact-relevant parameters match (type/size/friction/solref/solimp/condim).
    """
    import mujoco
    import numpy as np

    def geom_id(name: str) -> int:
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)

    # Collect all named geoms with left_/right_ prefix.
    left_names = []
    right_names = []
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        if not name:
            continue
        if name.startswith("left_"):
            left_names.append(name)
        elif name.startswith("right_"):
            right_names.append(name)

    # Only validate collision geoms (group=3 in this project).
    def is_collision(gid: int) -> bool:
        return int(model.geom_group[gid]) == 3

    checked = 0
    for name in left_names:
        pair = _pair_name(name)
        if pair is None:
            continue
        gid_l = geom_id(name)
        gid_r = geom_id(pair)
        if gid_l < 0 or gid_r < 0:
            continue
        if not (is_collision(gid_l) and is_collision(gid_r)):
            continue

        checked += 1
        # Contact-relevant parameters.
        if int(model.geom_type[gid_l]) != int(model.geom_type[gid_r]):
            raise ValueError(f"Collision geom type mismatch: {name} vs {pair}")
        if not np.allclose(
            model.geom_size[gid_l],
            model.geom_size[gid_r],
            atol=thresholds.size_tol,
            rtol=0.0,
        ):
            raise ValueError(
                f"Collision geom size mismatch: {name}={model.geom_size[gid_l]} vs {pair}={model.geom_size[gid_r]}"
            )
        if not np.allclose(
            model.geom_friction[gid_l],
            model.geom_friction[gid_r],
            atol=thresholds.friction_tol,
            rtol=0.0,
        ):
            raise ValueError(
                f"Collision geom friction mismatch: {name}={model.geom_friction[gid_l]} vs {pair}={model.geom_friction[gid_r]}"
            )
        if int(model.geom_condim[gid_l]) != int(model.geom_condim[gid_r]):
            raise ValueError(
                f"Collision geom condim mismatch: {name}={int(model.geom_condim[gid_l])} vs {pair}={int(model.geom_condim[gid_r])}"
            )
        if not np.allclose(
            model.geom_solref[gid_l], model.geom_solref[gid_r], atol=1e-6, rtol=0.0
        ):
            raise ValueError(
                f"Collision geom solref mismatch: {name}={model.geom_solref[gid_l]} vs {pair}={model.geom_solref[gid_r]}"
            )
        if not np.allclose(
            model.geom_solimp[gid_l], model.geom_solimp[gid_r], atol=1e-6, rtol=0.0
        ):
            raise ValueError(
                f"Collision geom solimp mismatch: {name}={model.geom_solimp[gid_l]} vs {pair}={model.geom_solimp[gid_r]}"
            )

    print(f"  - left/right named collision geom parity ({checked} pairs)")


def _validate_default_pose_contacts(
    scene_xml: Path,
    robot_config,
    thresholds: Thresholds,
) -> None:
    # Optional: requires a scene with ground. Validate that toe/heel world heights
    # are similar on left/right, and penetration isn't extreme.
    import mujoco
    import numpy as np

    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)

    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    left_toe, left_heel, right_toe, right_heel = robot_config.get_foot_geom_names()
    geom_ids = {
        name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in (left_toe, left_heel, right_toe, right_heel)
    }
    if any(gid < 0 for gid in geom_ids.values()):
        return

    # Compare world geom positions.
    lt = data.geom_xpos[geom_ids[left_toe]].copy()
    lh = data.geom_xpos[geom_ids[left_heel]].copy()
    rt = data.geom_xpos[geom_ids[right_toe]].copy()
    rh = data.geom_xpos[geom_ids[right_heel]].copy()

    if not np.allclose(lt[2], rt[2], atol=thresholds.pos_tol, rtol=0.0):
        raise ValueError(
            f"Default pose toe height mismatch: {left_toe} z={lt[2]} vs {right_toe} z={rt[2]}"
        )
    if not np.allclose(lh[2], rh[2], atol=thresholds.pos_tol, rtol=0.0):
        raise ValueError(
            f"Default pose heel height mismatch: {left_heel} z={lh[2]} vs {right_heel} z={rh[2]}"
        )

    # Contact penetration check (best-effort): reject very deep contacts.
    max_pen = 0.0
    for i in range(data.ncon):
        c = data.contact[i]
        # dist < 0 means penetration
        max_pen = min(max_pen, float(c.dist))
    if max_pen < -thresholds.contact_penetration_tol:
        raise ValueError(
            f"Excessive penetration at reset: min contact.dist={max_pen:.6f}m"
        )


def validate_model(
    robot_config_yaml: Path,
    scene_xml: Path,
    robot_xml: Optional[Path],
    thresholds: Thresholds,
) -> None:
    if robot_xml is None:
        robot_xml = _infer_robot_xml_from_scene(scene_xml)

    _validate_robot_config_matches_scene(
        scene_xml=scene_xml,
        robot_config_yaml=robot_config_yaml,
        robot_xml=robot_xml,
    )

    # Compile the scene (this matches training and will include the robot).
    model = _load_mujoco_model(scene_xml)
    robot_config = _load_robot_config(robot_config_yaml)

    _validate_toe_heel(model, robot_config, thresholds)
    _validate_mass_inertia_symmetry(model, thresholds)
    _validate_left_right_named_collision_geoms(model, thresholds)

    _validate_default_pose_contacts(scene_xml, robot_config, thresholds)

    # Joint range consistency: robot_config.yaml must match compiled model joints.
    import mujoco
    import numpy as np

    joint_tol = 1e-3
    for spec in robot_config.actuated_joints:
        joint_name = spec["name"]
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid < 0:
            raise ValueError(f"Actuated joint not found in model: {joint_name}")
        rmin, rmax = float(model.jnt_range[jid, 0]), float(model.jnt_range[jid, 1])
        cmin, cmax = float(spec["range"][0]), float(spec["range"][1])
        if not np.allclose([rmin, rmax], [cmin, cmax], atol=joint_tol, rtol=0.0):
            raise ValueError(
                f"Joint range mismatch for {joint_name}: model=({rmin:.6f},{rmax:.6f}) "
                f"config=({cmin:.6f},{cmax:.6f})"
            )

    print("Checks passed:")
    print("  - scene include ↔ robot_config.generated_from")
    print("  - toe/heel collision symmetry + mesh_pos sanity")
    print("  - left/right body mass + inertia symmetry")
    print("  - left/right named collision geom parity")
    print("  - default reset contact sanity")
    print("  - actuated joint range parity")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate model symmetry and asset/config consistency.",
        epilog=(
            "Examples:\n"
            "  uv run python assets/validate_model.py\n"
            "  uv run python assets/validate_model.py --scene-xml assets/scene_flat_terrain.xml\n"
            "  uv run python assets/validate_model.py --robot-config assets/robot_config.yaml\n"
            "  uv run python assets/validate_model.py --robot-xml assets/wildrobot.xml\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--robot-config",
        type=Path,
        default=Path("assets/robot_config.yaml"),
        help="Path to generated robot_config.yaml.",
    )
    parser.add_argument(
        "--scene-xml",
        type=Path,
        default=Path("assets/scene_flat_terrain.xml"),
        help="Scene XML used for training (must include the robot XML).",
    )
    parser.add_argument(
        "--robot-xml",
        type=Path,
        default=None,
        help="Optional override. When omitted, inferred from --scene-xml <include file=...>.",
    )

    parser.add_argument("--pos-tol", type=float, default=1e-4)
    parser.add_argument("--quat-tol", type=float, default=1e-4)
    parser.add_argument("--friction-tol", type=float, default=1e-4)
    parser.add_argument("--size-tol", type=float, default=1e-4)
    parser.add_argument("--mass-rel-tol", type=float, default=0.05)
    parser.add_argument("--inertia-rel-tol", type=float, default=0.10)
    parser.add_argument("--mesh-pos-abs-max", type=float, default=0.10)
    parser.add_argument("--contact-penetration-tol", type=float, default=0.005)
    args = parser.parse_args()

    thresholds = Thresholds(
        pos_tol=args.pos_tol,
        quat_tol=args.quat_tol,
        friction_tol=args.friction_tol,
        size_tol=args.size_tol,
        mass_rel_tol=args.mass_rel_tol,
        inertia_rel_tol=args.inertia_rel_tol,
        mesh_pos_abs_max=args.mesh_pos_abs_max,
        contact_penetration_tol=args.contact_penetration_tol,
    )

    if args.robot_xml is not None and not args.robot_xml.exists():
        raise FileNotFoundError(args.robot_xml)
    if not args.robot_config.exists():
        raise FileNotFoundError(args.robot_config)

    if not args.scene_xml.exists():
        raise FileNotFoundError(args.scene_xml)

    validate_model(
        robot_config_yaml=args.robot_config,
        scene_xml=args.scene_xml,
        robot_xml=args.robot_xml,
        thresholds=thresholds,
    )
    print("Model validation OK")
    print(f"  robot_config={args.robot_config}")
    print(f"  scene_xml={args.scene_xml}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
