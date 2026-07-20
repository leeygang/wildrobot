from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import mujoco

from policy_contract.spec import JointSpec, PolicySpec
from policy_contract.spec_builder import build_policy_spec


_PROJECT_ROOT = Path(__file__).parent.parent


def resolve_repo_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if not path.is_absolute():
        path = (_PROJECT_ROOT / path).resolve()
    return path


def get_home_ctrl_from_mj_model(
    *,
    mj_model: mujoco.MjModel,
    actuator_names: List[str],
) -> List[float]:
    """Extract actuator-space home joint positions from keyframe or qpos0."""
    mj_data = mujoco.MjData(mj_model)

    key_id = 0
    if mj_model.nkey > 0:
        try:
            key_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "home")
        except Exception:
            key_id = 0
        if key_id < 0:
            key_id = 0
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, key_id)
    else:
        mujoco.mj_resetData(mj_model, mj_data)

    home_ctrl = []
    for name in actuator_names:
        act_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if act_id < 0:
            raise ValueError(f"Actuator '{name}' not found in MJCF for home pose")
        trn_type = mj_model.actuator_trntype[act_id]
        if trn_type != mujoco.mjtTrn.mjTRN_JOINT:
            raise ValueError(
                f"Actuator '{name}' does not target a joint (trntype={int(trn_type)})"
            )
        joint_id = int(mj_model.actuator_trnid[act_id][0])
        qpos_adr = int(mj_model.jnt_qposadr[joint_id])
        home_ctrl.append(float(mj_data.qpos[qpos_adr]))
    return home_ctrl


def get_home_ctrl_from_model_path(
    *,
    model_path: str | Path,
    actuator_names: List[str],
) -> List[float]:
    mj_model = mujoco.MjModel.from_xml_path(str(resolve_repo_path(model_path)))
    return get_home_ctrl_from_mj_model(mj_model=mj_model, actuator_names=actuator_names)


def clamp_home_ctrl(
    *,
    home_ctrl: List[float],
    actuated_joint_specs: List[Dict[str, Any]],
    actuator_names: List[str],
) -> List[float]:
    joints: Dict[str, JointSpec] = {}
    for item in actuated_joint_specs:
        joints[str(item["name"])] = JointSpec(
            range_min_rad=float(item["range"][0]),
            range_max_rad=float(item["range"][1]),
            max_velocity_rad_s=float(item.get("max_velocity", 10.0)),
        )

    clipped = []
    for name, value in zip(actuator_names, home_ctrl):
        joint = joints[name]
        original = float(value)
        clamped = min(max(original, joint.range_min_rad), joint.range_max_rad)
        if not math.isclose(clamped, original, rel_tol=0.0, abs_tol=1e-12):
            pass
        clipped.append(clamped)
    return clipped


def policy_excluded_actuator_names(training_cfg: Any) -> List[str]:
    names = getattr(training_cfg.env, "policy_excluded_actuator_names", ()) or ()
    return [str(name) for name in names]


def _policy_actuated_joint_specs(*, training_cfg: Any, robot_cfg: Any) -> List[Dict[str, Any]]:
    excluded = set(policy_excluded_actuator_names(training_cfg))
    specs = list(robot_cfg.actuated_joints)
    if not excluded:
        return specs
    known = {str(item["name"]) for item in specs}
    unknown = sorted(excluded - known)
    if unknown:
        raise ValueError(
            "env.policy_excluded_actuator_names contains unknown actuators: "
            f"{unknown}"
        )
    active = [item for item in specs if str(item["name"]) not in excluded]
    if not active:
        raise ValueError("env.policy_excluded_actuator_names excludes every actuator")
    return active


def _runtime_fixed_home_metadata(
    *,
    training_cfg: Any,
    robot_cfg: Any,
) -> Optional[Dict[str, Any]]:
    excluded = set(policy_excluded_actuator_names(training_cfg))
    if not excluded:
        return None

    full_specs = list(robot_cfg.actuated_joints)
    full_names = [str(item["name"]) for item in full_specs]
    active_names = [name for name in full_names if name not in excluded]
    fixed_names = [name for name in full_names if name in excluded]
    full_home = get_home_ctrl_from_model_path(
        model_path=training_cfg.env.model_path,
        actuator_names=full_names,
    )
    full_home = clamp_home_ctrl(
        home_ctrl=full_home,
        actuated_joint_specs=full_specs,
        actuator_names=full_names,
    )
    home_by_name = dict(zip(full_names, full_home))
    spec_by_name = {str(item["name"]): item for item in full_specs}
    return {
        "full_actuator_names": full_names,
        "active_actuator_names": active_names,
        "fixed_actuator_names": fixed_names,
        "fixed_home_ctrl_rad": [float(home_by_name[name]) for name in fixed_names],
        "fixed_joint_ranges_rad": {
            name: [
                float(spec_by_name[name]["range"][0]),
                float(spec_by_name[name]["range"][1]),
            ]
            for name in fixed_names
        },
        "source": "env.policy_excluded_actuator_names",
    }


def maybe_get_home_ctrl_from_training_config(
    *,
    training_cfg: Any,
    robot_cfg: Any,
) -> Optional[List[float]]:
    """Populate ``policy_spec.robot.home_ctrl_rad`` when the runtime
    needs it.  Two triggers:

    1. ``action_mapping_id`` is a home-centered mapping — the action
       calibration ops use home_ctrl as the centering reference.
    2. ``loc_ref_residual_base == "home"`` (smoke8) — V6EvalAdapter
       needs the home pose to compose target_q under home base.  Without
       this, native-MuJoCo eval of a smoke8 checkpoint silently falls
       back to q_ref base, applying the smoke7 control contract.
    """
    mapping_id = str(getattr(training_cfg.env, "action_mapping_id", "pos_target_rad_v1"))
    residual_base = str(
        getattr(training_cfg.env, "loc_ref_residual_base", "q_ref")
    ).lower()
    home_mapping_ids = {"pos_target_home_v1", "pos_target_home_025_v1"}
    if mapping_id not in home_mapping_ids and residual_base != "home":
        return None

    actuated_joint_specs = _policy_actuated_joint_specs(
        training_cfg=training_cfg,
        robot_cfg=robot_cfg,
    )
    actuator_names = [str(item["name"]) for item in actuated_joint_specs]
    home_ctrl = get_home_ctrl_from_model_path(
        model_path=training_cfg.env.model_path,
        actuator_names=actuator_names,
    )
    return clamp_home_ctrl(
        home_ctrl=home_ctrl,
        actuated_joint_specs=actuated_joint_specs,
        actuator_names=actuator_names,
    )


def build_policy_spec_from_training_config(
    *,
    training_cfg: Any,
    robot_cfg: Any,
    action_filter_alpha: Optional[float] = None,
    provenance: Optional[Dict[str, Any]] = None,
) -> PolicySpec:
    policy_joint_specs = _policy_actuated_joint_specs(
        training_cfg=training_cfg,
        robot_cfg=robot_cfg,
    )
    spec_provenance = dict(provenance or {})
    fixed_home_metadata = _runtime_fixed_home_metadata(
        training_cfg=training_cfg,
        robot_cfg=robot_cfg,
    )
    if fixed_home_metadata is not None:
        spec_provenance["runtime_fixed_home"] = fixed_home_metadata

    return build_policy_spec(
        robot_name=robot_cfg.robot_name,
        actuated_joint_specs=policy_joint_specs,
        action_filter_alpha=float(
            training_cfg.env.action_filter_alpha
            if action_filter_alpha is None
            else action_filter_alpha
        ),
        layout_id=str(training_cfg.env.actor_obs_layout_id),
        mapping_id=str(training_cfg.env.action_mapping_id),
        home_ctrl_rad=maybe_get_home_ctrl_from_training_config(
            training_cfg=training_cfg,
            robot_cfg=robot_cfg,
        ),
        provenance=spec_provenance if spec_provenance else None,
    )
