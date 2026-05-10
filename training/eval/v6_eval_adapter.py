"""V6 eval adapter — native-MuJoCo runtime for the v0.20.1 wr_obs_v6
observation contract.

The v0.20.1 PPO recipe uses the
``wr_obs_v6_offline_ref_history`` observation layout, which carries:

  - the v4 base channels (gravity, gyro, joint pos/vel, foot switches,
    prev_action, velocity_cmd, gait phase / stance / foothold / swing /
    pelvis / phase-history)
  - the offline reference window (q_ref, pelvis pos/vel, per-foot pos/vel,
    contact mask)
  - a flat past-proprio stack of ``PROPRIO_HISTORY_FRAMES`` frames of
    ``(gyro, foot_switches, joint_pos_norm, joint_vel_norm, prev_action)``

and consumes a residual action: ``q_target = clip(q_ref + clip(action) *
scale_per_joint, joint_min, joint_max)``.

This adapter mirrors the env's per-step semantics
(``training/envs/wildrobot_env.py:_compute_proprio_bundle``,
``_v4_compat_channels_from_window``, ``_compose_target_q_from_residual``,
``_to_mj_ctrl``, ``_roll_proprio_history``) for the native-MuJoCo viewer
path, so visualize_policy.py can run a v6 checkpoint without re-implementing
each piece inline.

Why not drive WildRobotEnv directly: the native viewer's interactive
controls (pause / reset / camera) and fast native-MuJoCo stepping are worth
preserving.  This adapter keeps the viewer architecture intact and treats
v6 as a contract-aware eval shim — see walking_training.md Appendix B
(post-Plan-A note).

Single source of truth for the schema is the env; if the contract changes,
update both env and adapter together (``test_v6_eval_adapter_parity`` is the
guard).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import mujoco
import numpy as np

from policy_contract.calib import NumpyCalibOps
from policy_contract.numpy.obs import build_observation
from policy_contract.numpy.state import PolicyState
from policy_contract.spec import PROPRIO_HISTORY_FRAMES, PolicySpec
from training.utils.ctrl_order import CtrlOrderMapper

V6_LAYOUT_ID = "wr_obs_v6_offline_ref_history"


@dataclass
class V6AdapterState:
    """Per-episode mutable state held by V6EvalAdapter."""
    proprio_history: np.ndarray  # (PROPRIO_HISTORY_FRAMES, bundle_size); oldest first
    prev_phase_sin_cos: Optional[np.ndarray]  # (2,) or None at reset
    step_idx: int  # offline-trajectory frame index, clamped at n_steps - 1


class V6EvalAdapter:
    """Native-MuJoCo eval adapter for ``wr_obs_v6_offline_ref_history``.

    Lifecycle (per control step in the viewer loop):

        1. ``obs = adapter.compute_obs(mj_data, prev_action, velocity_cmd)``
           — uses the *pre-roll* history (the buffer at the start of this
           step, holding the past PROPRIO_HISTORY_FRAMES bundles, oldest
           first) and the offline reference window at ``step_idx``.
        2. ``action = policy(obs)`` — caller's responsibility.
        3. ``adapter.apply_action(mj_data, action)`` — composes
           ``q_target = clip(q_ref + clip(action) * scale, joint_range)``
           and writes ``mj_data.ctrl`` in MJ actuator order.
        4. ``mj_step(...)`` ×n_substeps — caller's responsibility.
        5. ``adapter.post_step()`` — rolls the proprio buffer (drop oldest,
           append the bundle that produced *this* step's obs) and advances
           ``step_idx`` (clamped at end-of-trajectory).

    Reset: call ``adapter.reset()`` after ``mj_resetData`` /
    ``mj_resetDataKeyframe`` to zero the buffers and rewind the trajectory
    index.  Pair with ``prev_action = np.zeros(action_dim)`` (the residual
    contract starts with no residual; env mirrors this in
    ``_make_initial_state``).
    """

    def __init__(
        self,
        *,
        training_cfg,
        mj_model: mujoco.MjModel,
        policy_spec: PolicySpec,
        signals_adapter,
        action_dim: int,
    ) -> None:
        if policy_spec.observation.layout_id != V6_LAYOUT_ID:
            raise ValueError(
                f"V6EvalAdapter expects layout_id={V6_LAYOUT_ID!r}, "
                f"got {policy_spec.observation.layout_id!r}"
            )
        self._cfg = training_cfg
        self._mj_model = mj_model
        self._policy_spec = policy_spec
        self._signals_adapter = signals_adapter
        self._action_dim = int(action_dim)

        self._bundle_size = 3 + 4 + 3 * self._action_dim  # gyro + sw + 3*N
        self._init_offline_service()
        self._init_residual_scale()
        self._init_joint_ranges()
        self._init_ctrl_mapper()

        self._state = V6AdapterState(
            proprio_history=np.zeros(
                (PROPRIO_HISTORY_FRAMES, self._bundle_size), dtype=np.float32
            ),
            prev_phase_sin_cos=None,
            step_idx=0,
        )
        # Bundle produced at the most recent compute_obs() call; rolled in
        # post_step().  Initialised to zeros so the first post_step() before
        # any compute_obs() is a no-op.
        self._pending_bundle: Optional[np.ndarray] = None
        self._pending_phase_sin_cos: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ init

    def _init_offline_service(self) -> None:
        from control.references.runtime_reference_service import (
            RuntimeReferenceService,
        )
        offline_path = getattr(self._cfg.env, "loc_ref_offline_library_path", None)
        offline_vx = float(getattr(self._cfg.env, "loc_ref_offline_command_vx", 0.20))
        if offline_path:
            from control.references.reference_library import ReferenceLibrary
            lib = ReferenceLibrary.load(offline_path)
        else:
            from control.zmp.zmp_walk import ZMPWalkGenerator
            lib = ZMPWalkGenerator().build_library_for_vx_values([offline_vx])
        traj = lib.lookup(offline_vx)
        self._service = RuntimeReferenceService(traj, n_anchor=2)
        self._n_steps = int(self._service.n_steps)

    def _init_residual_scale(self) -> None:
        """Mirror env's _init_residual_scale: per-joint scale array in
        policy actuator order."""
        per_joint = dict(
            getattr(self._cfg.env, "loc_ref_residual_scale_per_joint", {}) or {}
        )
        scalar = float(getattr(self._cfg.env, "loc_ref_residual_scale", 0.18))
        self._scale_per_joint = np.array(
            [
                float(per_joint.get(name, scalar))
                for name in self._policy_spec.robot.actuator_names
            ],
            dtype=np.float32,
        )

    def _init_joint_ranges(self) -> None:
        """Joint position min/max in policy actuator order."""
        actuator_names = list(self._policy_spec.robot.actuator_names)
        joint_qpos_addrs: list[int] = []
        joint_ranges: list[tuple[float, float]] = []
        for name in actuator_names:
            act_id = mujoco.mj_name2id(
                self._mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
            )
            joint_id = int(self._mj_model.actuator_trnid[act_id][0])
            joint_qpos_addrs.append(int(self._mj_model.jnt_qposadr[joint_id]))
            jrange = self._mj_model.jnt_range[joint_id]
            joint_ranges.append((float(jrange[0]), float(jrange[1])))
        self._joint_qpos_addrs = np.asarray(joint_qpos_addrs, dtype=np.int32)
        self._joint_min = np.array([r[0] for r in joint_ranges], dtype=np.float32)
        self._joint_max = np.array([r[1] for r in joint_ranges], dtype=np.float32)

    def _init_ctrl_mapper(self) -> None:
        self._ctrl_mapper = CtrlOrderMapper(
            self._mj_model, list(self._policy_spec.robot.actuator_names)
        )

    # ------------------------------------------------------------ public API

    @property
    def n_steps(self) -> int:
        return self._n_steps

    @property
    def step_idx(self) -> int:
        return int(self._state.step_idx)

    def reset(self) -> None:
        """Re-zero per-episode state.  Caller should set
        ``prev_action = np.zeros(action_dim)`` to match env's
        residual-contract reset."""
        self._state = V6AdapterState(
            proprio_history=np.zeros(
                (PROPRIO_HISTORY_FRAMES, self._bundle_size), dtype=np.float32
            ),
            prev_phase_sin_cos=None,
            step_idx=0,
        )
        self._pending_bundle = None
        self._pending_phase_sin_cos = None

    def compute_obs(
        self,
        mj_data: mujoco.MjData,
        prev_action: np.ndarray,
        velocity_cmd: float,
    ) -> np.ndarray:
        """Build the v6 observation.  Stores the current proprio bundle and
        phase for ``post_step()`` to roll into the history."""
        signals = self._signals_adapter.read(mj_data)
        win = self._service.lookup_np(self._state.step_idx)

        # v4-compat derived channels (mirror env's
        # _v4_compat_channels_from_window).
        phase_sin_cos = np.array(
            [float(win.phase_sin), float(win.phase_cos)], dtype=np.float32
        )
        is_left_stance = int(win.stance_foot_id) == 0
        if is_left_stance:
            next_foothold_xyz = win.right_foot_pos
            swing_pos_xyz = win.right_foot_pos
            swing_vel_xyz = win.right_foot_vel
        else:
            next_foothold_xyz = win.left_foot_pos
            swing_pos_xyz = win.left_foot_pos
            swing_vel_xyz = win.left_foot_vel
        next_foothold_xy = np.asarray(next_foothold_xyz[:2], dtype=np.float32)
        swing_pos = np.asarray(swing_pos_xyz[:3], dtype=np.float32)
        swing_vel = np.asarray(swing_vel_xyz[:3], dtype=np.float32)
        pelvis_targets = np.array(
            [float(win.pelvis_pos[2]), 0.0, 0.0], dtype=np.float32
        )
        if self._state.prev_phase_sin_cos is None:
            history_4 = np.tile(phase_sin_cos, 2).astype(np.float32)
        else:
            history_4 = np.concatenate(
                [phase_sin_cos, self._state.prev_phase_sin_cos[:2]]
            ).astype(np.float32)

        # Proprio bundle for THIS step's signals + the action that produced
        # this state.  Stored as _pending_bundle and rolled in post_step().
        joint_pos_norm = NumpyCalibOps.normalize_joint_pos(
            spec=self._policy_spec, joint_pos_rad=signals.joint_pos_rad
        ).astype(np.float32)
        joint_vel_norm = NumpyCalibOps.normalize_joint_vel(
            spec=self._policy_spec, joint_vel_rad_s=signals.joint_vel_rad_s
        ).astype(np.float32)
        bundle = np.concatenate(
            [
                np.asarray(signals.gyro_rad_s, dtype=np.float32),
                np.asarray(signals.foot_switches, dtype=np.float32),
                joint_pos_norm,
                joint_vel_norm,
                np.asarray(prev_action, dtype=np.float32),
            ]
        )
        self._pending_bundle = bundle
        self._pending_phase_sin_cos = phase_sin_cos

        # Build the obs with the PRE-roll history (env contract).
        flat_history = self._state.proprio_history.reshape(-1)
        policy_state = PolicyState(prev_action=np.asarray(prev_action, dtype=np.float32))
        return build_observation(
            spec=self._policy_spec,
            state=policy_state,
            signals=signals,
            velocity_cmd=np.array(velocity_cmd, dtype=np.float32),
            loc_ref_phase_sin_cos=phase_sin_cos,
            loc_ref_stance_foot=np.array(
                [float(win.stance_foot_id)], dtype=np.float32
            ),
            loc_ref_next_foothold=next_foothold_xy,
            loc_ref_swing_pos=swing_pos,
            loc_ref_swing_vel=swing_vel,
            loc_ref_pelvis_targets=pelvis_targets,
            loc_ref_history=history_4,
            loc_ref_q_ref=np.asarray(win.q_ref, dtype=np.float32),
            loc_ref_pelvis_pos=np.asarray(win.pelvis_pos, dtype=np.float32),
            loc_ref_pelvis_vel=np.asarray(win.pelvis_vel, dtype=np.float32),
            loc_ref_left_foot_pos=np.asarray(win.left_foot_pos, dtype=np.float32),
            loc_ref_right_foot_pos=np.asarray(win.right_foot_pos, dtype=np.float32),
            loc_ref_left_foot_vel=np.asarray(win.left_foot_vel, dtype=np.float32),
            loc_ref_right_foot_vel=np.asarray(win.right_foot_vel, dtype=np.float32),
            loc_ref_contact_mask=np.asarray(win.contact_mask, dtype=np.float32),
            proprio_history=flat_history,
        )

    def apply_action(
        self,
        mj_data: mujoco.MjData,
        action: np.ndarray,
    ) -> np.ndarray:
        """Compose target_q via the residual contract and write
        ``mj_data.ctrl`` in MJ order.

        Returns ``target_q`` (rad, policy actuator order) for diagnostics.
        """
        win = self._service.lookup_np(self._state.step_idx)
        q_ref = np.asarray(win.q_ref, dtype=np.float32)
        clipped = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        residual = clipped * self._scale_per_joint
        target_q = np.clip(
            q_ref + residual, self._joint_min, self._joint_max
        ).astype(np.float32)
        ctrl_mj = self._ctrl_mapper.to_mj_np(target_q)
        mj_data.ctrl[:] = np.asarray(ctrl_mj, dtype=np.float32)
        return target_q

    def post_step(self) -> None:
        """Roll proprio_history (drop oldest, append the bundle that
        produced *this* step's obs) and advance the trajectory index.

        No-op if compute_obs() hasn't been called yet (defensive — the
        viewer may sync once before the first policy step).
        """
        if self._pending_bundle is not None:
            history = self._state.proprio_history
            new_history = np.concatenate(
                [history[1:], self._pending_bundle[None, :]], axis=0
            )
            self._state.proprio_history = new_history.astype(np.float32)
        if self._pending_phase_sin_cos is not None:
            self._state.prev_phase_sin_cos = self._pending_phase_sin_cos.copy()
        # Clamp at end-of-trajectory (matches env's
        # RuntimeReferenceService.lookup_np semantics: trajectory ends
        # are absorbing boundary frames).
        self._state.step_idx = min(self._state.step_idx + 1, self._n_steps - 1)
        self._pending_bundle = None
        self._pending_phase_sin_cos = None
