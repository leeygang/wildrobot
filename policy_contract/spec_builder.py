from __future__ import annotations

from typing import Any, Dict, List, Optional

from policy_contract.spec import (
    PROPRIO_HISTORY_FRAMES,
    ActionSpec,
    JointSpec,
    ModelSpec,
    ObservationSpec,
    ObsFieldSpec,
    PolicySpec,
    RobotSpec,
    validate_spec,
)


def build_policy_spec(
    *,
    robot_name: str,
    actuated_joint_specs: List[Dict[str, Any]],
    action_filter_alpha: float,
    home_ctrl_rad: Optional[List[float]] = None,
    provenance: Optional[Dict[str, Any]] = None,
    contract_name: str = "wildrobot_policy",
    contract_version: str = "1.0.0",
    spec_version: int = 2,
    layout_id: str = "wr_obs_v1",
    mapping_id: str = "pos_target_rad_v1",
    model_input_name: str = "observation",
    model_output_name: str = "action",
) -> PolicySpec:
    """Build a PolicySpec from data available in training/export pipelines.

    This function is intentionally framework-agnostic (no JAX/MuJoCo deps) so it can
    be used consistently by:
      - training startup (to define the actor contract)
      - export (to write policy_spec.json)
      - eval tooling (to ensure parity with runtime)
    """
    if not isinstance(actuated_joint_specs, list) or not actuated_joint_specs:
        raise ValueError("actuated_joint_specs must be a non-empty list[dict]")

    joints: Dict[str, JointSpec] = {}
    actuator_names: List[str] = []
    for item in actuated_joint_specs:
        if not isinstance(item, dict):
            raise ValueError(f"Invalid joint spec entry: {item!r}")
        name = str(item.get("name"))
        if not name:
            raise ValueError(f"Joint spec missing name: {item!r}")
        rng = item.get("range") or [0.0, 0.0]
        if not isinstance(rng, list) or len(rng) != 2:
            raise ValueError(f"Invalid range for joint '{name}': {rng!r}")
        actuator_names.append(name)
        joints[name] = JointSpec(
            range_min_rad=float(rng[0]),
            range_max_rad=float(rng[1]),
            policy_action_sign=float(item.get("policy_action_sign", 1.0)),
            max_velocity_rad_s=float(item.get("max_velocity", 10.0)),
        )

    action_dim = len(actuator_names)
    layout = _build_obs_layout(action_dim=action_dim, layout_id=layout_id)

    alpha = float(action_filter_alpha)
    postprocess_id = "lowpass_v1" if alpha > 0.0 else "none"
    postprocess_params: Dict[str, Any] = {}
    if postprocess_id == "lowpass_v1":
        postprocess_params["alpha"] = alpha

    spec = PolicySpec(
        contract_name=contract_name,
        contract_version=contract_version,
        spec_version=int(spec_version),
        model=ModelSpec(
            format="onnx",
            input_name=model_input_name,
            output_name=model_output_name,
            dtype="float32",
            obs_dim=sum(field.size for field in layout),
            action_dim=action_dim,
        ),
        robot=RobotSpec(
            robot_name=str(robot_name),
            actuator_names=actuator_names,
            joints=joints,
            home_ctrl_rad=list(home_ctrl_rad) if home_ctrl_rad is not None else None,
        ),
        observation=ObservationSpec(
            dtype="float32",
            layout_id=layout_id,
            layout=layout,
        ),
        action=ActionSpec(
            dtype="float32",
            bounds={"min": -1.0, "max": 1.0},
            postprocess_id=postprocess_id,
            postprocess_params=postprocess_params,
            mapping_id=mapping_id,
        ),
        provenance=provenance,
    )
    validate_spec(spec)
    return spec


def _build_obs_layout(*, action_dim: int, layout_id: str) -> List[ObsFieldSpec]:
    if layout_id == "wr_obs_v1":
        return [
            ObsFieldSpec(name="gravity_local", size=3, frame="local", units="unit_vector"),
            ObsFieldSpec(name="angvel_heading_local", size=3, frame="heading_local", units="rad_s"),
            ObsFieldSpec(name="joint_pos_normalized", size=action_dim, units="normalized_-1_1"),
            ObsFieldSpec(name="joint_vel_normalized", size=action_dim, units="normalized_-1_1"),
            ObsFieldSpec(name="foot_switches", size=4, units="bool_as_float"),
            ObsFieldSpec(name="prev_action", size=action_dim, units="normalized_-1_1"),
            ObsFieldSpec(name="velocity_cmd", size=1, units="m_s"),
            ObsFieldSpec(name="padding", size=1, units="unused"),
        ]
    if layout_id == "wr_obs_v2":
        return [
            ObsFieldSpec(name="gravity_local", size=3, frame="local", units="unit_vector"),
            ObsFieldSpec(name="angvel_heading_local", size=3, frame="heading_local", units="rad_s"),
            ObsFieldSpec(name="joint_pos_normalized", size=action_dim, units="normalized_-1_1"),
            ObsFieldSpec(name="joint_vel_normalized", size=action_dim, units="normalized_-1_1"),
            ObsFieldSpec(name="foot_switches", size=4, units="bool_as_float"),
            ObsFieldSpec(name="prev_action", size=action_dim, units="normalized_-1_1"),
            ObsFieldSpec(name="velocity_cmd", size=1, units="m_s"),
            ObsFieldSpec(name="capture_point_error", size=2, frame="heading_local", units="m"),
            ObsFieldSpec(name="padding", size=1, units="unused"),
        ]
    if layout_id == "wr_obs_v3":
        return [
            ObsFieldSpec(name="gravity_local", size=3, frame="local", units="unit_vector"),
            ObsFieldSpec(name="angvel_heading_local", size=3, frame="heading_local", units="rad_s"),
            ObsFieldSpec(name="joint_pos_normalized", size=action_dim, units="normalized_-1_1"),
            ObsFieldSpec(name="joint_vel_normalized", size=action_dim, units="normalized_-1_1"),
            ObsFieldSpec(name="foot_switches", size=4, units="bool_as_float"),
            ObsFieldSpec(name="prev_action", size=action_dim, units="normalized_-1_1"),
            ObsFieldSpec(name="velocity_cmd", size=1, units="m_s"),
            ObsFieldSpec(name="gait_clock", size=4, units="sin_cos_phase"),
            ObsFieldSpec(name="padding", size=1, units="unused"),
        ]
    if layout_id == "wr_obs_v4":
        return [
            ObsFieldSpec(name="gravity_local", size=3, frame="local", units="unit_vector"),
            ObsFieldSpec(name="angvel_heading_local", size=3, frame="heading_local", units="rad_s"),
            ObsFieldSpec(name="joint_pos_normalized", size=action_dim, units="normalized_-1_1"),
            ObsFieldSpec(name="joint_vel_normalized", size=action_dim, units="normalized_-1_1"),
            ObsFieldSpec(name="foot_switches", size=4, units="bool_as_float"),
            ObsFieldSpec(name="prev_action", size=action_dim, units="normalized_-1_1"),
            ObsFieldSpec(name="velocity_cmd", size=1, units="m_s"),
            ObsFieldSpec(name="loc_ref_phase_sin_cos", size=2, units="sin_cos_phase"),
            ObsFieldSpec(name="loc_ref_stance_foot", size=1, units="foot_id_float"),
            ObsFieldSpec(name="loc_ref_next_foothold", size=2, units="m_stance_frame"),
            ObsFieldSpec(name="loc_ref_swing_pos", size=3, units="m_stance_frame"),
            ObsFieldSpec(name="loc_ref_swing_vel", size=3, units="mps_stance_frame"),
            ObsFieldSpec(name="loc_ref_pelvis_targets", size=3, units="m_rad_rad"),
            ObsFieldSpec(name="loc_ref_history", size=4, units="recent_phase_sin_cos"),
            ObsFieldSpec(name="padding", size=1, units="unused"),
        ]
    if layout_id == "wr_obs_v7_phase_proprio":
        # smoke11 (post-home-migration de-hybridization).  Mirrors
        # ToddlerBot's default ``use_phase_signal=True`` actor obs
        # (walk.gin :: num_single_obs = 84 = 2(phase) + 3(cmd) +
        # 30(motor_pos) + 30(motor_vel) + 12(last_act) + 3(ang_vel) +
        # 4(quat)).  WR substitutes:
        #   - ``gravity_local`` (3) for TB's ``torso_quat`` (4) — same
        #     information; gravity-in-body is the WR convention.
        #   - per-joint normalized pos/vel for TB's delta-from-default
        #     scaled pos/vel — same role, different normalization.
        #   - 4-dim ``foot_switches`` for the WR contact contract.
        # The only remaining reference-derived channel is the 2-dim
        # ``loc_ref_phase_sin_cos`` gait clock, kept because TB does the
        # same and the actor needs an internal periodic time signal.
        # ``proprio_history`` is the WR signal-engineering addition
        # (lin_vel is privileged-only on WR, so the actor estimates
        # base velocity from short-horizon kinematic stacking).
        proprio_bundle = 3 + 4 + 3 * action_dim
        return [
            ObsFieldSpec(name="gravity_local", size=3, frame="local", units="unit_vector"),
            ObsFieldSpec(name="angvel_heading_local", size=3, frame="heading_local", units="rad_s"),
            ObsFieldSpec(name="joint_pos_normalized", size=action_dim, units="normalized_-1_1"),
            ObsFieldSpec(name="joint_vel_normalized", size=action_dim, units="normalized_-1_1"),
            ObsFieldSpec(name="foot_switches", size=4, units="bool_as_float"),
            ObsFieldSpec(name="prev_action", size=action_dim, units="normalized_-1_1"),
            ObsFieldSpec(name="velocity_cmd", size=1, units="m_s"),
            ObsFieldSpec(name="loc_ref_phase_sin_cos", size=2, units="sin_cos_phase"),
            ObsFieldSpec(
                name="proprio_history",
                size=PROPRIO_HISTORY_FRAMES * proprio_bundle,
                units="stacked_proprio_oldest_to_newest",
            ),
            ObsFieldSpec(name="padding", size=1, units="unused"),
        ]
    if layout_id == "wr_obs_v6_offline_ref_history":
        # v0.20.1 — the active locomotion contract.  Channels:
        #   - v4 base (gravity, gyro, joint pos/vel, foot switches,
        #     prev_action, velocity_cmd, gait phase / stance / foothold
        #     / swing / pelvis / phase-history)
        #   - offline reference window: q_ref, pelvis pos/vel, per-
        #     foot pos/vel, contact mask
        #   - proprio_history: PROPRIO_HISTORY_FRAMES past bundles of
        #     gyro + foot_switches + joint_pos_norm + joint_vel_norm +
        #     prev_action so the actor can estimate base velocity from
        #     short-horizon kinematics (lin_vel is privileged-only).
        # ToddlerBot's c_frame_stack=15 is the larger reference;
        # PROPRIO_HISTORY_FRAMES=3 is the minimum that gives finite-
        # diff velocity from joint pos and one integration of gyro.
        # Reference-channel history is NOT restacked here (it already
        # lives in loc_ref_history / loc_ref_phase_sin_cos / etc.).
        # Channel order MUST match policy_contract/{jax,numpy}/obs.py
        # ``build_observation_from_components`` so obs_dim ==
        # concat-len.  See v0201_env_wiring.md §5.2.
        proprio_bundle = 3 + 4 + 3 * action_dim
        return [
            ObsFieldSpec(name="gravity_local", size=3, frame="local", units="unit_vector"),
            ObsFieldSpec(name="angvel_heading_local", size=3, frame="heading_local", units="rad_s"),
            ObsFieldSpec(name="joint_pos_normalized", size=action_dim, units="normalized_-1_1"),
            ObsFieldSpec(name="joint_vel_normalized", size=action_dim, units="normalized_-1_1"),
            ObsFieldSpec(name="foot_switches", size=4, units="bool_as_float"),
            ObsFieldSpec(name="prev_action", size=action_dim, units="normalized_-1_1"),
            ObsFieldSpec(name="velocity_cmd", size=1, units="m_s"),
            ObsFieldSpec(name="loc_ref_phase_sin_cos", size=2, units="sin_cos_phase"),
            ObsFieldSpec(name="loc_ref_stance_foot", size=1, units="foot_id_float"),
            ObsFieldSpec(name="loc_ref_next_foothold", size=2, units="m_stance_frame"),
            ObsFieldSpec(name="loc_ref_swing_pos", size=3, units="m_stance_frame"),
            ObsFieldSpec(name="loc_ref_swing_vel", size=3, units="mps_stance_frame"),
            ObsFieldSpec(name="loc_ref_pelvis_targets", size=3, units="m_rad_rad"),
            ObsFieldSpec(name="loc_ref_history", size=4, units="recent_phase_sin_cos"),
            ObsFieldSpec(name="loc_ref_q_ref", size=action_dim, units="rad"),
            ObsFieldSpec(name="loc_ref_pelvis_pos", size=3, units="m_world"),
            ObsFieldSpec(name="loc_ref_pelvis_vel", size=3, units="mps_world"),
            ObsFieldSpec(name="loc_ref_left_foot_pos", size=3, units="m_world"),
            ObsFieldSpec(name="loc_ref_right_foot_pos", size=3, units="m_world"),
            ObsFieldSpec(name="loc_ref_left_foot_vel", size=3, units="mps_world"),
            ObsFieldSpec(name="loc_ref_right_foot_vel", size=3, units="mps_world"),
            ObsFieldSpec(name="loc_ref_contact_mask", size=2, units="bool_as_float"),
            ObsFieldSpec(
                name="proprio_history",
                size=PROPRIO_HISTORY_FRAMES * proprio_bundle,
                units="stacked_proprio_oldest_to_newest",
            ),
            ObsFieldSpec(name="padding", size=1, units="unused"),
        ]
    raise ValueError(f"Unsupported layout_id: {layout_id!r}")
