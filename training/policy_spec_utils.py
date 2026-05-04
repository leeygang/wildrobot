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
            policy_action_sign=float(item.get("policy_action_sign", 1.0)),
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


def maybe_get_home_ctrl_from_training_config(
    *,
    training_cfg: Any,
    robot_cfg: Any,
) -> Optional[List[float]]:
    mapping_id = str(getattr(training_cfg.env, "action_mapping_id", "pos_target_rad_v1"))
    if mapping_id != "pos_target_home_v1":
        return None

    actuator_names = [str(item["name"]) for item in robot_cfg.actuated_joints]
    home_ctrl = get_home_ctrl_from_model_path(
        model_path=training_cfg.env.model_path,
        actuator_names=actuator_names,
    )
    return clamp_home_ctrl(
        home_ctrl=home_ctrl,
        actuated_joint_specs=robot_cfg.actuated_joints,
        actuator_names=actuator_names,
    )


def build_policy_spec_from_training_config(
    *,
    training_cfg: Any,
    robot_cfg: Any,
    action_filter_alpha: Optional[float] = None,
    provenance: Optional[Dict[str, Any]] = None,
) -> PolicySpec:
    return build_policy_spec(
        robot_name=robot_cfg.robot_name,
        actuated_joint_specs=robot_cfg.actuated_joints,
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
        provenance=provenance,
    )
