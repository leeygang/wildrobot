"""Build the ``runtime_policy_config.json`` bundle metadata.

The hardware runtime (``runtime/wr_runtime/control``) is a standalone package
that must NOT import the training-side ``control`` / JAX stack.  But the latest
``wr_obs_v8_cmd3d`` home-base-residual contract needs a handful of training
quantities that are NOT encoded in ``policy_spec.json``:

  - the action path: ``loc_ref_residual_base`` (home), optional walking-only
    base offsets, per-joint residual scales
    (``loc_ref_residual_scale_per_joint`` + scalar fallback), action delay,
    and action filtering
  - the gait clock: ``loc_ref_phase_sin_cos`` per offline step index, which
    requires the offline ZMP reference library to generate.

This module runs at EXPORT time (training side, has ``control`` + JAX) and
freezes those quantities into ``runtime_policy_config.json`` so the runtime
can reconstruct the exact training control contract from bundle-local files
only.

Important: the gait phase is a normalized clock that is **bin-independent**
across the whole (vx, vy, wz) library (verified at build time — see
``build_reference_phase_table``).  We therefore store a single shared
``phase_sin`` / ``phase_cos`` array plus per-bin ``n_steps`` (for the
absorbing-boundary clamp) and the ``cmd_keys`` matrix (for nearest-bin
selection), mirroring ``training/eval/v6_eval_adapter.py`` exactly.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np

from policy_contract.spec import PolicySpec

# Tolerance for the bin-independence check on the normalized gait phase.
_PHASE_BIN_TOL = 1e-4


def _env_get(env: Dict[str, Any], key: str, default: Any) -> Any:
    val = env.get(key, default)
    return default if val is None else val


def build_reference_phase_table(env: Dict[str, Any]) -> Dict[str, Any]:
    """Build the bundled gait-phase clock from the offline ZMP library.

    Mirrors ``V6EvalAdapter._init_offline_service`` (1D + 3D paths) so the
    runtime sees the same per-bin command keys and step counts the training
    env used.  Returns a JSON-serializable dict:

      - ``gait_n_cycle``: frames per gait cycle (detected from the phase
        reset; informational)
      - ``n_steps``: global max steps across bins (step-index advance bound)
      - ``cmd_keys``: ``[[vx, vy, wz], ...]`` per bin (nearest-bin selection)
      - ``per_bin_n_steps``: ``[n_i, ...]`` per bin (absorbing-boundary clamp)
      - ``phase_sin`` / ``phase_cos``: shared length-``n_steps`` clock arrays
      - ``primary_q_ref0``: frame-zero joint reference for the canonical
        straight-walk bin, used to freeze a generator-native roll action base

    Raises if the per-bin phase clocks diverge beyond ``_PHASE_BIN_TOL`` —
    that would mean a single shared clock is unsafe and the schema must move
    to per-bin storage.
    """
    offline_path = _env_get(env, "loc_ref_offline_library_path", None)
    offline_vx = float(_env_get(env, "loc_ref_offline_command_vx", 0.20))
    cmd_conditioned = bool(_env_get(env, "loc_ref_command_conditioned", False))
    axes_3d = bool(_env_get(env, "loc_ref_command_axes_3d", False))

    trajectories = _build_trajectories(
        offline_path=offline_path,
        offline_vx=offline_vx,
        cmd_conditioned=cmd_conditioned,
        axes_3d=axes_3d,
        env=env,
    )

    cmd_keys: List[List[float]] = []
    per_bin_n_steps: List[int] = []
    phases: List[np.ndarray] = []
    for traj in trajectories:
        phase = np.asarray(traj.phase, dtype=np.float32).reshape(-1)
        cmd_keys.append(
            [
                float(traj.command_vx),
                float(traj.command_vy),
                float(traj.command_yaw_rate),
            ]
        )
        per_bin_n_steps.append(int(phase.shape[0]))
        phases.append(phase)

    anchor = np.asarray([offline_vx, 0.0, 0.0], dtype=np.float64)
    primary_idx = int(
        np.argmin(
            np.linalg.norm(np.asarray(cmd_keys, dtype=np.float64) - anchor, axis=1)
        )
    )
    primary_q_ref0 = np.asarray(
        trajectories[primary_idx].q_ref[0], dtype=np.float32
    ).reshape(-1)

    # Canonical (longest) phase; verify all multi-step bins agree on the
    # overlap so a single shared clock is provably exact.
    canonical = max(phases, key=lambda p: p.shape[0])
    for phase in phases:
        if phase.shape[0] <= 1:
            continue  # static (0,0,0) bin — phase is a single 0.0 frame
        overlap = min(phase.shape[0], canonical.shape[0])
        diff = float(np.max(np.abs(phase[:overlap] - canonical[:overlap])))
        if diff > _PHASE_BIN_TOL:
            raise ValueError(
                "Gait phase is NOT bin-independent (max overlap diff "
                f"{diff:.3e} > {_PHASE_BIN_TOL}); the runtime_policy_config "
                "shared-clock assumption is unsafe.  Switch the schema to "
                "per-bin phase storage."
            )

    phase_sin = np.sin(2.0 * np.pi * canonical).astype(np.float32)
    phase_cos = np.cos(2.0 * np.pi * canonical).astype(np.float32)
    gait_n_cycle = _detect_n_cycle(canonical)

    return {
        "gait_n_cycle": int(gait_n_cycle),
        "n_steps": int(canonical.shape[0]),
        "cmd_keys": cmd_keys,
        "per_bin_n_steps": per_bin_n_steps,
        "phase_sin": [float(v) for v in phase_sin],
        "phase_cos": [float(v) for v in phase_cos],
        "primary_q_ref0": [float(v) for v in primary_q_ref0],
    }


def _build_trajectories(
    *,
    offline_path: Any,
    offline_vx: float,
    cmd_conditioned: bool,
    axes_3d: bool,
    env: Dict[str, Any],
) -> List[Any]:
    """Mirror V6EvalAdapter._init_offline_service trajectory construction."""
    from training.configs.zmp_reference import zmp_walk_config_from_env

    zmp_config = zmp_walk_config_from_env(
        env,
        offline_library_path=offline_path,
    )
    if offline_path:
        from control.references.reference_library import ReferenceLibrary

        lib = ReferenceLibrary.load(offline_path)
        if cmd_conditioned and axes_3d:
            return list(lib._entries.values())
        return [lib.lookup(offline_vx)]

    if cmd_conditioned and axes_3d:
        from control.zmp.zmp_walk import ZMPWalkGenerator
        vy_grid = list(_env_get(env, "loc_ref_offline_command_vy_grid", ())) or [0.0]
        wz_grid = list(
            _env_get(env, "loc_ref_offline_command_yaw_rate_grid", ())
        ) or [0.0]
        min_vx = float(env["min_velocity"])
        max_vx = float(env["max_velocity"])
        interval = float(_env_get(env, "loc_ref_command_grid_interval", 0.05))
        if interval <= 0.0:
            raise ValueError(
                f"loc_ref_command_grid_interval must be positive; got {interval!r}"
            )
        arange_vals = np.arange(min_vx, max_vx + 1e-6, interval, dtype=np.float64)
        vx_grid = sorted(
            {round(float(v), 6) for v in arange_vals} | {round(offline_vx, 6)}
        )
        lib = ZMPWalkGenerator(
            config=zmp_config,
            scene_xml_path=_env_get(env, "scene_xml_path", None),
            robot_config_path=_env_get(env, "robot_config_path", None),
        ).build_library_for_3d_values(
            vx_values=vx_grid,
            vy_values=vy_grid,
            yaw_rate_values=wz_grid,
        )
        return list(lib._entries.values())

    from control.zmp.zmp_walk import ZMPWalkGenerator
    lib = ZMPWalkGenerator(
        config=zmp_config,
        scene_xml_path=_env_get(env, "scene_xml_path", None),
        robot_config_path=_env_get(env, "robot_config_path", None),
    ).build_library_for_vx_values([offline_vx])
    return [lib.lookup(offline_vx)]


def _detect_n_cycle(phase: np.ndarray) -> int:
    """Frames per gait cycle = index of the first phase reset + 1."""
    if phase.shape[0] <= 1:
        return int(phase.shape[0])
    resets = np.where(np.diff(phase) < 0.0)[0]
    if resets.size == 0:
        return int(phase.shape[0])
    return int(resets[0]) + 1


def build_runtime_metadata(
    *,
    env: Dict[str, Any],
    spec: PolicySpec,
    reference: Dict[str, Any],
) -> Dict[str, Any]:
    """Assemble the runtime_policy_config dict (pure; no library build).

    ``reference`` is the output of ``build_reference_phase_table`` (injected so
    this function stays fast + unit-testable without the ZMP stack).
    """
    ctrl_dt = float(_env_get(env, "ctrl_dt", 0.02))
    if ctrl_dt <= 0.0:
        raise ValueError(f"ctrl_dt must be positive; got {ctrl_dt!r}")

    actuator_names = list(spec.robot.actuator_names)
    per_joint = dict(_env_get(env, "loc_ref_residual_scale_per_joint", {}) or {})
    scalar_scale = float(_env_get(env, "loc_ref_residual_scale", 0.18))
    residual_scale_per_actuator = [
        float(per_joint.get(name, scalar_scale)) for name in actuator_names
    ]
    base_offsets = dict(
        _env_get(env, "loc_ref_walking_joint_offsets_rad", {}) or {}
    )
    residual_base = str(_env_get(env, "loc_ref_residual_base", "q_ref"))
    if base_offsets and residual_base != "home":
        raise ValueError(
            "env.loc_ref_walking_joint_offsets_rad requires "
            "loc_ref_residual_base='home'"
        )
    unknown_base_offsets = sorted(set(base_offsets) - set(actuator_names))
    if unknown_base_offsets:
        raise ValueError(
            "env.loc_ref_walking_joint_offsets_rad contains unknown actuators: "
            f"{unknown_base_offsets}"
        )
    residual_base_offset_per_actuator = [
        float(base_offsets.get(name, 0.0)) for name in actuator_names
    ]
    use_reference_roll_base = bool(
        _env_get(env, "loc_ref_walking_base_from_ref_init_roll", False)
    )
    if use_reference_roll_base:
        from training.configs.zmp_reference import reference_roll_base_offsets

        if residual_base != "home":
            raise ValueError(
                "env.loc_ref_walking_base_from_ref_init_roll requires "
                "loc_ref_residual_base='home'"
            )
        if _env_get(env, "loc_ref_default_stance_width_m", None) is None:
            raise ValueError(
                "env.loc_ref_walking_base_from_ref_init_roll requires "
                "loc_ref_default_stance_width_m"
            )
        home_q = spec.robot.home_ctrl_rad
        if home_q is None:
            raise ValueError(
                "policy_spec.robot.home_ctrl_rad is required for the "
                "frame-zero roll walking base"
            )
        if "primary_q_ref0" not in reference:
            raise ValueError(
                "reference.primary_q_ref0 is required for the frame-zero "
                "roll walking base"
            )
        residual_base_offset_per_actuator = reference_roll_base_offsets(
            actuator_names=actuator_names,
            home_q_rad=np.asarray(home_q, dtype=np.float32),
            ref_init_q_rad=np.asarray(reference["primary_q_ref0"], dtype=np.float32),
            enabled=True,
            explicit_offsets=base_offsets,
        ).astype(float).tolist()
    if not np.all(np.isfinite(residual_base_offset_per_actuator)):
        raise ValueError(
            "env.loc_ref_walking_joint_offsets_rad values must be finite"
        )

    default_cmd = _as_three_vec(_env_get(env, "eval_velocity_cmd", [0.0, 0.0, 0.0]))

    return {
        "schema_version": 1,
        "actor_obs_layout_id": str(_env_get(env, "actor_obs_layout_id", "")),
        "action_mapping_id": str(_env_get(env, "action_mapping_id", "")),
        "ctrl_dt": ctrl_dt,
        "control_hz": 1.0 / ctrl_dt,
        "action_delay_steps": int(_env_get(env, "action_delay_steps", 0)),
        "action_filter_alpha": float(_env_get(env, "action_filter_alpha", 0.0)),
        "loc_ref_residual_base": residual_base,
        "loc_ref_residual_mode": str(
            _env_get(env, "loc_ref_residual_mode", "absolute")
        ),
        "loc_ref_residual_scale": scalar_scale,
        "loc_ref_residual_scale_per_joint": {k: float(v) for k, v in per_joint.items()},
        # Resolved into actuator order so the runtime can apply it without
        # re-deriving the name->index map.
        "residual_scale_per_actuator": residual_scale_per_actuator,
        "loc_ref_walking_joint_offsets_rad": {
            k: float(v) for k, v in base_offsets.items()
        },
        "loc_ref_default_stance_width_m": _env_get(
            env, "loc_ref_default_stance_width_m", None
        ),
        "loc_ref_walking_base_from_ref_init_roll": use_reference_roll_base,
        "residual_base_offset_per_actuator": residual_base_offset_per_actuator,
        "loc_ref_command_conditioned": bool(
            _env_get(env, "loc_ref_command_conditioned", False)
        ),
        "loc_ref_command_axes_3d": bool(
            _env_get(env, "loc_ref_command_axes_3d", False)
        ),
        "default_velocity_cmd": default_cmd,
        "reference": reference,
    }


def _as_three_vec(value: Any) -> List[float]:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == 1:
        return [float(arr[0]), 0.0, 0.0]
    if arr.size == 3:
        return [float(arr[0]), float(arr[1]), float(arr[2])]
    raise ValueError(f"velocity command must be scalar or length-3; got size {arr.size}")


def build_runtime_policy_config(
    *,
    env: Dict[str, Any],
    spec: PolicySpec,
) -> Dict[str, Any]:
    """Full build: reference phase table (heavy) + metadata assembly."""
    if spec.observation.layout_id in (
        "wr_obs_v1", "wr_obs_v9_standing", "wr_obs_v10_standing_recovery"
    ):
        ctrl_dt = float(_env_get(env, "ctrl_dt", 0.02))
        if ctrl_dt <= 0.0:
            raise ValueError(f"ctrl_dt must be positive; got {ctrl_dt!r}")
        metadata = {
            "schema_version": 1,
            "policy_role": "standing",
            "actor_obs_layout_id": spec.observation.layout_id,
            "action_mapping_id": spec.action.mapping_id,
            "ctrl_dt": ctrl_dt,
            "control_hz": 1.0 / ctrl_dt,
            "action_delay_steps": int(_env_get(env, "action_delay_steps", 0)),
            "action_filter_alpha": float(_env_get(env, "action_filter_alpha", 0.0)),
            "default_velocity_cmd": [0.0, 0.0, 0.0],
        }
        if spec.observation.layout_id == "wr_obs_v10_standing_recovery":
            metadata["recovery"] = {
                "enabled": bool(_env_get(env, "standing_recovery_enabled", False)),
                "trigger_angle_rad": float(
                    _env_get(env, "standing_recovery_trigger_angle_rad", 0.0872665)
                ),
                "lookahead_s": float(
                    _env_get(env, "standing_recovery_lookahead_s", 0.25)
                ),
                "com_height_m": float(_env_get(env, "target_height", 0.46)),
                "capture_gain": float(
                    _env_get(env, "standing_recovery_capture_gain", 1.0)
                ),
                "max_step_m": float(
                    _env_get(env, "standing_recovery_max_step_m", 0.10)
                ),
                "swing_duration_steps": int(
                    _env_get(env, "standing_recovery_swing_duration_steps", 20)
                ),
                "settle_min_steps": int(
                    _env_get(env, "standing_recovery_settle_min_steps", 25)
                ),
                "settle_max_steps": int(
                    _env_get(env, "standing_recovery_settle_max_steps", 75)
                ),
                "settle_angle_rad": float(
                    _env_get(env, "standing_recovery_settle_angle_rad", 0.0523599)
                ),
                "settle_rate_rad_s": float(
                    _env_get(env, "standing_recovery_settle_rate_rad_s", 0.10)
                ),
            }
        return metadata
    reference = build_reference_phase_table(env)
    return build_runtime_metadata(env=env, spec=spec, reference=reference)
