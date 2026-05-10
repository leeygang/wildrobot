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
    ``(gyro, foot_switches, joint_pos_norm, joint_vel_norm, applied_action)``

and consumes a residual action with optional 1-step delay + filter:

    filtered  = lowpass(prev_filtered, raw_action)        # alpha=0 ⇒ no-op
    applied   = prev_filtered if delay else filtered      # 1-step delay
    target_q  = clip(q_ref[step_idx + 1] + applied * scale, joint_range)

Notice three subtleties that earlier adapter revisions got wrong:

1. ``applied`` (not ``raw_action``) is the value that lands in the obs's
   ``prev_action`` slot AND in the rolled proprio bundle's prev_action
   slot.  The bundle's other channels come from POST-physics signals.
2. The control reference advances BEFORE the lookup: at env step N the
   target uses ``q_ref[step_idx + 1]``, not ``q_ref[step_idx]``.
3. The proprio history obs uses the PRE-roll buffer (zeros at step 1),
   and the POST-physics bundle is rolled in *after* the obs is built so
   it shows up at step 2.

This adapter mirrors env's per-step semantics exactly:

  - ``training/envs/wildrobot_env.py:_compute_proprio_bundle``
  - ``training/envs/wildrobot_env.py:_v4_compat_channels_from_window``
  - ``training/envs/wildrobot_env.py:_compose_target_q_from_residual``
  - ``training/envs/wildrobot_env.py:_to_mj_ctrl``
  - ``training/envs/wildrobot_env.py:_roll_proprio_history``
  - ``training/envs/wildrobot_env.py.step`` lines 1682-1727 (delay/filter)

so visualize_policy.py can run a v6 checkpoint without re-implementing
each piece inline.

Lifecycle (per visualizer iteration; mirrors env.step end-to-end):

    adapter.reset()                           # once at episode start
    obs = adapter.compute_obs(mj_data, velocity_cmd)  # uses step_idx
    action_raw = policy(obs)
    adapter.apply_action(mj_data, action_raw) # advances step_idx,
                                              # writes ctrl
    step_physics(mj_model, mj_data, n_substeps)
    adapter.post_physics(mj_data)             # builds bundle from POST
                                              # signals, rolls history
    # next iteration: compute_obs reads the NEW step_idx + rolled history

Single source of truth for the schema is the env; if the contract changes,
update both env and adapter together (``test_v6_eval_adapter.py`` is the
guard).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import mujoco
import numpy as np

from policy_contract.calib import NumpyCalibOps
from policy_contract.numpy.action import postprocess_action
from policy_contract.numpy.obs import build_observation
from policy_contract.numpy.state import PolicyState
from policy_contract.spec import PROPRIO_HISTORY_FRAMES, PolicySpec
from training.utils.ctrl_order import CtrlOrderMapper

V6_LAYOUT_ID = "wr_obs_v6_offline_ref_history"


@dataclass
class V6AdapterState:
    """Per-episode mutable state held by V6EvalAdapter."""
    proprio_history: np.ndarray  # (PROPRIO_HISTORY_FRAMES, bundle_size); oldest first
    loc_ref_history: Optional[np.ndarray]  # (4,) rolled phase_sin_cos; None at reset
    step_idx: int  # offline-trajectory frame for the CURRENT obs (next ctrl uses step_idx + 1)
    pending_action: np.ndarray  # (action_dim,) last-filtered value; also "applied" when delay enabled
    last_applied_action: np.ndarray  # (action_dim,) action that was applied to physics this step


class V6EvalAdapter:
    """Native-MuJoCo eval adapter for ``wr_obs_v6_offline_ref_history``.

    See module docstring for the lifecycle and parity rationale.
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

        # Action delay flag: env supports {0, 1}; v0.20.1 smoke uses 1.
        delay_steps = int(getattr(self._cfg.env, "action_delay_steps", 0))
        if delay_steps not in (0, 1):
            raise ValueError(
                f"V6EvalAdapter supports action_delay_steps in {{0, 1}}, "
                f"got {delay_steps}"
            )
        self._action_delay_enabled = (delay_steps == 1)

        self._init_offline_service()
        self._init_residual_scale()
        self._init_joint_ranges()
        self._init_ctrl_mapper()
        self.reset()

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
        actuator_names = list(self._policy_spec.robot.actuator_names)
        joint_ranges: list[tuple[float, float]] = []
        for name in actuator_names:
            act_id = mujoco.mj_name2id(
                self._mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
            )
            joint_id = int(self._mj_model.actuator_trnid[act_id][0])
            jrange = self._mj_model.jnt_range[joint_id]
            joint_ranges.append((float(jrange[0]), float(jrange[1])))
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
        """Re-zero per-episode state.  Mirrors env's ``_make_initial_state``:
        proprio_history zero-filled, pending_action and last_applied_action
        zero (so iter-1 with action=0 keeps target_q == q_ref exactly)."""
        self._state = V6AdapterState(
            proprio_history=np.zeros(
                (PROPRIO_HISTORY_FRAMES, self._bundle_size), dtype=np.float32
            ),
            loc_ref_history=None,
            step_idx=0,
            pending_action=np.zeros(self._action_dim, dtype=np.float32),
            last_applied_action=np.zeros(self._action_dim, dtype=np.float32),
        )

    def compute_obs(
        self,
        mj_data: mujoco.MjData,
        velocity_cmd: float,
    ) -> np.ndarray:
        """Build the v6 obs at the current ``step_idx``.

        Mirrors env's ``_get_obs`` call from ``step()``:
          - obs's prev_action slot = ``last_applied_action`` (the action
            that was applied THIS step, written by ``apply_action`` and
            picked up by the bundle / current proprio prev_action slot).
          - proprio_history slot is the PRE-roll buffer (zeros at iter 1
            because ``post_physics`` of iter 1 hasn't run yet).
          - v4_compat["history"] is computed from the previous
            ``loc_ref_history``, then stored as the new
            ``loc_ref_history`` for the next obs (env's wr.loc_ref_history).
        """
        signals = self._signals_adapter.read(mj_data)
        win = self._service.lookup_np(self._state.step_idx)

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

        # v4_compat history: env's _v4_compat_channels_from_window uses
        # tile(phase_sc, 2) when prev_history is None (reset path) and
        # concat([phase_sc, prev_history[:2]]) otherwise.  After this
        # step's compute_obs, store v4_compat["history"] as the NEW
        # loc_ref_history for the next compute_obs.
        prev_loc_ref_history = self._state.loc_ref_history
        if prev_loc_ref_history is None:
            history_4 = np.tile(phase_sin_cos, 2).astype(np.float32)
        else:
            history_4 = np.concatenate(
                [phase_sin_cos, prev_loc_ref_history[:2]]
            ).astype(np.float32)

        # PolicyState.prev_action for the obs's prev_action slot is the
        # action APPLIED this step (env: PolicyState(prev_action=action)
        # where action == applied_action).  At reset, last_applied_action
        # is zeros.
        policy_state = PolicyState(
            prev_action=np.asarray(self._state.last_applied_action, dtype=np.float32)
        )
        obs = build_observation(
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
            proprio_history=self._state.proprio_history.reshape(-1),
        )

        # Update for next compute_obs.
        self._state.loc_ref_history = history_4
        return obs

    def apply_action(
        self,
        mj_data: mujoco.MjData,
        raw_action: np.ndarray,
    ) -> np.ndarray:
        """Filter, delay, advance step_idx, compose target_q, write ctrl.

        Mirrors env.step lines 1682-1727 exactly:

            policy_state = PolicyState(prev_action=pending_action)
            filtered, _ = postprocess_action(spec, policy_state, raw_action)
            applied = pending_action if delay else filtered
            next_step_idx = clip(step_idx + 1, n - 1)
            q_ref = service.lookup_np(next_step_idx).q_ref
            target_q = clip(q_ref + clip(applied) * scale, joint_range)
            ctrl_mj = ctrl_mapper.to_mj_np(target_q)
            mj_data.ctrl[:] = ctrl_mj
            new pending_action = filtered
            new last_applied_action = applied

        Returns ``applied_action`` for callers that want it; the adapter
        also stores it on ``self._state.last_applied_action`` so the next
        ``compute_obs`` and ``post_physics`` use the correct value.
        """
        raw = np.asarray(raw_action, dtype=np.float32)

        # Filter (alpha=0 ⇒ no-op; lowpass_v1 otherwise).  Filter input
        # is pending_action (= last filtered), NOT last_applied_action,
        # per env line 1713.
        filtered, _ = postprocess_action(
            spec=self._policy_spec,
            state=PolicyState(prev_action=self._state.pending_action),
            action_raw=raw,
        )
        filtered = np.asarray(filtered, dtype=np.float32)

        # 1-step delay: applied = previous filtered (= pending_action).
        # No delay: applied = current filtered.
        if self._action_delay_enabled:
            applied = self._state.pending_action.copy()
        else:
            applied = filtered.copy()

        # Advance step_idx.  Env clamps at n_steps - 1 (lookup beyond is
        # an absorbing-boundary frame).
        self._state.step_idx = min(self._state.step_idx + 1, self._n_steps - 1)

        win = self._service.lookup_np(self._state.step_idx)
        q_ref = np.asarray(win.q_ref, dtype=np.float32)
        clipped = np.clip(applied, -1.0, 1.0)
        residual = clipped * self._scale_per_joint
        target_q = np.clip(
            q_ref + residual, self._joint_min, self._joint_max
        ).astype(np.float32)
        ctrl_mj = self._ctrl_mapper.to_mj_np(target_q)
        mj_data.ctrl[:] = np.asarray(ctrl_mj, dtype=np.float32)

        # Save for next iteration.
        self._state.pending_action = filtered
        self._state.last_applied_action = applied
        return applied

    def post_physics(self, mj_data: mujoco.MjData) -> None:
        """Read POST-physics signals, build the per-step bundle, roll the
        proprio_history.

        Mirrors env.step lines 1893-1908: bundle uses POST-step signals
        and the action APPLIED this step (``last_applied_action``).  The
        rolled buffer becomes the "past" for the NEXT compute_obs.
        """
        signals_post = self._signals_adapter.read(mj_data)
        joint_pos_norm = NumpyCalibOps.normalize_joint_pos(
            spec=self._policy_spec, joint_pos_rad=signals_post.joint_pos_rad
        ).astype(np.float32)
        joint_vel_norm = NumpyCalibOps.normalize_joint_vel(
            spec=self._policy_spec, joint_vel_rad_s=signals_post.joint_vel_rad_s
        ).astype(np.float32)
        bundle = np.concatenate(
            [
                np.asarray(signals_post.gyro_rad_s, dtype=np.float32),
                np.asarray(signals_post.foot_switches, dtype=np.float32),
                joint_pos_norm,
                joint_vel_norm,
                np.asarray(self._state.last_applied_action, dtype=np.float32),
            ]
        )
        history = self._state.proprio_history
        self._state.proprio_history = np.concatenate(
            [history[1:], bundle[None, :]], axis=0
        ).astype(np.float32)
