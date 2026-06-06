"""Online control state machine for the latest WildRobot walking contract.

This is the hardware-side mirror of ``training/eval/v6_eval_adapter.py`` for the
``wr_obs_v8_cmd3d`` home-base-residual policy, without any MuJoCo / JAX /
``control`` dependency.  It maintains the four pieces of online state the v8
policy needs:

  - gait phase / reference step index
  - ``prev_action`` (the APPLIED action, not the raw output)
  - the 15-frame proprio history (rolled oldest -> newest)
  - the pending filtered action for the 1-step action delay

and composes the control target as a residual around the home pose:

    filtered = postprocess(raw)                      # action_filter_alpha=0 -> identity
    applied  = prev_filtered if delay else filtered  # action_delay_steps=1
    delta    = clip(applied, -1, 1) * residual_scale_per_joint
    target_q = clip(home_q_rad + delta, joint_min, joint_max)

This is NOT the generic ``pos_target_rad_v1`` midpoint mapping
(``NumpyCalibOps.action_to_ctrl``).  The smoke9 env commands residuals around
``home_q_rad`` with per-joint scales; using the midpoint mapping would be
behaviorally wrong even though ``policy_spec.json`` validation passes.

Per-iteration order (mirrors env.step / V6EvalAdapter; see ``step`` docstring)
reproduces the env's 1-iteration proprio-history lag exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from policy_contract.calib import NumpyCalibOps
from policy_contract.numpy.action import postprocess_action
from policy_contract.numpy.obs import build_observation
from policy_contract.numpy.signals import Signals
from policy_contract.numpy.state import PolicyState
from policy_contract.spec import PROPRIO_HISTORY_FRAMES, PolicySpec

from .reference_phase import ReferencePhaseService
from .runtime_policy_config import RuntimePolicyConfig

_SUPPORTED_LAYOUT = "wr_obs_v8_cmd3d"
_SUPPORTED_RESIDUAL_BASE = "home"


@dataclass
class RunnerState:
    proprio_history: np.ndarray   # (PROPRIO_HISTORY_FRAMES, bundle_size)
    pending_history: np.ndarray   # (PROPRIO_HISTORY_FRAMES, bundle_size)
    step_idx: int
    pending_action: np.ndarray    # last filtered action (filter state + delay buffer)
    last_applied_action: np.ndarray
    first_step: bool


class RuntimePolicyRunner:
    def __init__(
        self,
        *,
        spec: PolicySpec,
        runtime_config: RuntimePolicyConfig,
        policy,
        robot_io,
    ) -> None:
        layout = spec.observation.layout_id
        if layout != _SUPPORTED_LAYOUT:
            raise ValueError(
                f"RuntimePolicyRunner supports only layout_id={_SUPPORTED_LAYOUT!r}; "
                f"got {layout!r}.  Older layouts need their own runtime path."
            )
        base = runtime_config.loc_ref_residual_base
        if base != _SUPPORTED_RESIDUAL_BASE:
            raise ValueError(
                f"RuntimePolicyRunner supports only loc_ref_residual_base="
                f"{_SUPPORTED_RESIDUAL_BASE!r} (q_ref/ref_init need bundled q_ref "
                f"arrays, not yet supported); got {base!r}."
            )

        self._spec = spec
        self._cfg = runtime_config
        self._policy = policy
        self._robot_io = robot_io
        self._action_dim = int(spec.model.action_dim)
        self._bundle_size = 3 + 4 + 3 * self._action_dim

        delay_steps = int(runtime_config.action_delay_steps)
        if delay_steps not in (0, 1):
            raise ValueError(
                f"action_delay_steps must be 0 or 1; got {delay_steps}"
            )
        self._action_delay_enabled = delay_steps == 1

        self._phase_service = ReferencePhaseService(runtime_config.reference)

        self._init_joint_ranges()
        self._init_home_q_rad()
        self._init_residual_scale()
        self.reset()

    # ----------------------------------------------------------------- init

    def _init_joint_ranges(self) -> None:
        mins, maxs = [], []
        for name in self._spec.robot.actuator_names:
            joint = self._spec.robot.joints[name]
            mins.append(float(joint.range_min_rad))
            maxs.append(float(joint.range_max_rad))
        self._joint_min = np.asarray(mins, dtype=np.float32)
        self._joint_max = np.asarray(maxs, dtype=np.float32)

    def _init_home_q_rad(self) -> None:
        home = self._spec.robot.home_ctrl_rad
        if home is None:
            raise ValueError(
                "policy_spec.robot.home_ctrl_rad is required for the home-base "
                "residual contract but is missing from the bundle."
            )
        if len(home) != self._action_dim:
            raise ValueError(
                f"home_ctrl_rad length {len(home)} != action_dim {self._action_dim}"
            )
        self._home_q_rad = np.clip(
            np.asarray(home, dtype=np.float32), self._joint_min, self._joint_max
        ).astype(np.float32)

    def _init_residual_scale(self) -> None:
        per_actuator = self._cfg.residual_scale_per_actuator
        if per_actuator and len(per_actuator) == self._action_dim:
            self._residual_scale = np.asarray(per_actuator, dtype=np.float32)
            return
        # Fall back to per-joint dict + scalar default, keyed by actuator name.
        per_joint = self._cfg.loc_ref_residual_scale_per_joint
        scalar = float(self._cfg.loc_ref_residual_scale)
        self._residual_scale = np.asarray(
            [
                float(per_joint.get(name, scalar))
                for name in self._spec.robot.actuator_names
            ],
            dtype=np.float32,
        )

    # --------------------------------------------------------------- public

    @property
    def home_q_rad(self) -> np.ndarray:
        return self._home_q_rad.copy()

    @property
    def residual_scale_per_actuator(self) -> np.ndarray:
        return self._residual_scale.copy()

    @property
    def step_idx(self) -> int:
        return int(self._state.step_idx)

    def reset(self) -> None:
        """Re-zero per-episode state (mirrors env ``_make_initial_state``).

        proprio_history + pending_history zero-filled, pending/last_applied zero
        so iter-1 with action=0 yields ``target_q == home_q_rad`` exactly.
        """
        zeros_hist = np.zeros(
            (PROPRIO_HISTORY_FRAMES, self._bundle_size), dtype=np.float32
        )
        self._state = RunnerState(
            proprio_history=zeros_hist.copy(),
            pending_history=zeros_hist.copy(),
            step_idx=0,
            pending_action=np.zeros(self._action_dim, dtype=np.float32),
            last_applied_action=np.zeros(self._action_dim, dtype=np.float32),
            first_step=True,
        )

    def build_obs(self, signals: Signals, velocity_cmd: np.ndarray) -> np.ndarray:
        """Build the v8 obs at the current ``step_idx``.

        Uses the PRE-roll proprio_history and ``last_applied`` for the
        prev_action slot, then promotes ``pending_history`` into
        ``proprio_history`` for the next obs (mirrors V6EvalAdapter.compute_obs).
        """
        cmd = _as_three_vec(velocity_cmd)
        bin_idx = self._phase_service.select_bin(cmd)
        phase = self._phase_service.phase_sin_cos(
            bin_idx=bin_idx, step_idx=self._state.step_idx
        )
        policy_state = PolicyState(
            prev_action=np.asarray(self._state.last_applied_action, dtype=np.float32)
        )
        obs = build_observation(
            spec=self._spec,
            state=policy_state,
            signals=signals,
            velocity_cmd=cmd,
            velocity_cmd_lateral_yaw=cmd[1:],
            loc_ref_phase_sin_cos=phase,
            proprio_history=self._state.proprio_history.reshape(-1),
        )
        # Promote AFTER reading proprio_history (env 1-iteration lag).
        self._state.proprio_history = self._state.pending_history
        return obs

    def compose_and_apply(self, raw_action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Filter, delay, advance step_idx, compose home-base residual target.

        Returns ``(target_q_rad, applied_action)`` in actuator order.
        """
        raw = np.asarray(raw_action, dtype=np.float32).reshape(-1)
        # Guard the broadcast site: a wrong-sized raw action (e.g. length-1)
        # would otherwise NumPy-broadcast through the filter / residual scale
        # into a full joint target.  Fail loudly instead.
        if raw.shape[0] != self._action_dim:
            raise ValueError(
                f"raw_action has {raw.shape[0]} elements, expected "
                f"{self._action_dim} (action_dim)"
            )
        filtered, _ = postprocess_action(
            spec=self._spec,
            state=PolicyState(prev_action=self._state.pending_action),
            action_raw=raw,
        )
        filtered = np.asarray(filtered, dtype=np.float32)

        if self._action_delay_enabled:
            applied = self._state.pending_action.copy()
        else:
            applied = filtered.copy()

        self._state.step_idx = min(
            self._state.step_idx + 1, self._phase_service.n_steps - 1
        )

        residual = np.clip(applied, -1.0, 1.0) * self._residual_scale
        target_q = np.clip(
            self._home_q_rad + residual, self._joint_min, self._joint_max
        ).astype(np.float32)

        self._state.pending_action = filtered
        self._state.last_applied_action = applied
        return target_q, applied

    def roll_history(self, signals: Signals) -> None:
        """Build the per-step proprio bundle from POST-physics signals and the
        action APPLIED this step, then roll it into ``pending_history``.

        Mirrors V6EvalAdapter.post_physics: bundle =
        ``[gyro, foot_switches, joint_pos_norm, joint_vel_norm, last_applied]``.
        """
        joint_pos_norm = NumpyCalibOps.normalize_joint_pos(
            spec=self._spec, joint_pos_rad=signals.joint_pos_rad
        ).astype(np.float32)
        joint_vel_norm = NumpyCalibOps.normalize_joint_vel(
            spec=self._spec, joint_vel_rad_s=signals.joint_vel_rad_s
        ).astype(np.float32)
        bundle = np.concatenate(
            [
                np.asarray(signals.gyro_rad_s, dtype=np.float32).reshape(3),
                np.asarray(signals.foot_switches, dtype=np.float32).reshape(4),
                joint_pos_norm,
                joint_vel_norm,
                np.asarray(self._state.last_applied_action, dtype=np.float32),
            ]
        )
        self._state.pending_history = np.concatenate(
            [self._state.proprio_history[1:], bundle[None, :]], axis=0
        ).astype(np.float32)

    def step(self, velocity_cmd: np.ndarray) -> dict:
        """Run one control iteration against ``robot_io``.

        Order (reproduces the env proprio-history lag exactly — see module
        docstring):

            signals = robot_io.read()              # = post-physics of prev ctrl
            roll_history(signals)                  # post_physics(t-1), skipped at t=1
            obs = build_obs(signals, cmd)          # compute_obs(t), promotes pending
            raw = policy.predict(obs)
            target_q, applied = compose_and_apply(raw)
            robot_io.write_ctrl(target_q)
        """
        signals = self._robot_io.read()
        if not self._state.first_step:
            self.roll_history(signals)
        self._state.first_step = False

        obs = self.build_obs(signals, velocity_cmd)
        raw = np.asarray(self._policy.predict(obs), dtype=np.float32).reshape(-1)
        target_q, applied = self.compose_and_apply(raw)
        self._robot_io.write_ctrl(target_q)

        return {
            "step_idx": int(self._state.step_idx),
            "obs": obs,
            "raw_action": raw,
            "applied_action": applied,
            "target_q_rad": target_q,
            "signals": signals,
        }


def _as_three_vec(velocity_cmd: np.ndarray) -> np.ndarray:
    cmd = np.asarray(velocity_cmd, dtype=np.float32).reshape(-1)
    if cmd.size == 1:
        return np.array([float(cmd[0]), 0.0, 0.0], dtype=np.float32)
    if cmd.size == 3:
        return cmd.astype(np.float32)
    raise ValueError(
        f"velocity_cmd must be scalar or length-3 (vx, vy, wz); got size {cmd.size}"
    )
