"""Runtime runner for the legacy standing policy contract.

The walking runner is intentionally specific to ``wr_obs_v8_cmd3d`` and its
phase/reference history.  Standing policies use ``wr_obs_v1`` or the
foot-switch-free actor contract ``wr_obs_v9_standing``.

This runner also supports the ToddlerBot-style active-action subset used by the
home stabilizer: the policy can control a subset of actuators while runtime
expands the command back to the hardware actuator vector and holds excluded
servos at their home targets.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Mapping

import numpy as np

from policy_contract.calib import NumpyCalibOps
from policy_contract.numpy.action import postprocess_action
from policy_contract.numpy.obs import build_observation
from policy_contract.numpy.signals import Signals
from policy_contract.numpy.state import PolicyState
from policy_contract.spec import PolicySpec
from wr_runtime.control.runtime_policy_config import StandingRuntimePolicyConfig


_SUPPORTED_LAYOUTS = {"wr_obs_v1", "wr_obs_v9_standing"}


@dataclass
class StandingRunnerState:
    step_idx: int
    pending_action: np.ndarray
    last_applied_action: np.ndarray


class StandingPolicyRunner:
    """Run a supported standing policy against RobotIO."""

    def __init__(
        self,
        *,
        spec: PolicySpec,
        policy,
        robot_io,
        runtime_config: StandingRuntimePolicyConfig | None = None,
        fixed_home_targets_rad: Mapping[str, float] | None = None,
        zero_cmd_hold_home_deadzone: float | None = None,
    ) -> None:
        layout = spec.observation.layout_id
        if layout not in _SUPPORTED_LAYOUTS:
            raise ValueError(
                f"StandingPolicyRunner supports layout_id in {sorted(_SUPPORTED_LAYOUTS)!r}; "
                f"got {layout!r}."
            )
        if spec.robot.home_ctrl_rad is None:
            raise ValueError("standing runtime requires policy_spec.robot.home_ctrl_rad")

        self._spec = spec
        self._policy = policy
        self._robot_io = robot_io
        self._runtime_config = runtime_config
        self._zero_cmd_hold_home_deadzone = zero_cmd_hold_home_deadzone
        self._action_dim = int(spec.model.action_dim)
        self._policy_actuator_names = list(spec.robot.actuator_names)
        self._hardware_actuator_names = list(
            getattr(robot_io, "actuator_names", self._policy_actuator_names)
        )
        self._fixed_home_targets_rad = {
            str(name): float(value)
            for name, value in (fixed_home_targets_rad or {}).items()
        }
        delay_steps = 0 if runtime_config is None else int(runtime_config.action_delay_steps)
        if delay_steps not in (0, 1):
            raise ValueError(f"standing action_delay_steps must be 0 or 1; got {delay_steps}")
        self._action_delay_enabled = delay_steps == 1
        if runtime_config is not None:
            if runtime_config.actor_obs_layout_id != layout:
                raise ValueError(
                    "standing runtime config layout mismatch: "
                    f"{runtime_config.actor_obs_layout_id!r} != {layout!r}"
                )
            if runtime_config.action_mapping_id != spec.action.mapping_id:
                raise ValueError(
                    "standing runtime config action mapping mismatch: "
                    f"{runtime_config.action_mapping_id!r} != {spec.action.mapping_id!r}"
                )
            spec_alpha = float(spec.action.postprocess_params.get("alpha", 0.0))
            if abs(float(runtime_config.action_filter_alpha) - spec_alpha) > 1e-9:
                raise ValueError(
                    "standing runtime config action filter mismatch: "
                    f"{runtime_config.action_filter_alpha} != {spec_alpha}"
                )

        self._init_joint_ranges()
        self._init_home_q_rad()
        self._init_hardware_mapping()
        self.reset()

    @property
    def spec(self) -> PolicySpec:
        return self._spec

    @property
    def home_q_rad(self) -> np.ndarray:
        return self._home_q_rad.copy()

    @property
    def hardware_home_q_rad(self) -> np.ndarray:
        return self._hardware_home_q_rad.copy()

    @property
    def hardware_actuator_names(self) -> list[str]:
        return list(self._hardware_actuator_names)

    @property
    def fixed_home_actuator_names(self) -> list[str]:
        return list(self._fixed_home_targets_rad.keys())

    @property
    def step_idx(self) -> int:
        return int(self._state.step_idx)

    def reset(self) -> None:
        zeros = np.zeros(self._action_dim, dtype=np.float32)
        self._state = StandingRunnerState(
            step_idx=0,
            pending_action=zeros.copy(),
            last_applied_action=zeros.copy(),
        )

    def _init_joint_ranges(self) -> None:
        mins: list[float] = []
        maxs: list[float] = []
        for name in self._policy_actuator_names:
            joint = self._spec.robot.joints[name]
            mins.append(float(joint.range_min_rad))
            maxs.append(float(joint.range_max_rad))
        self._joint_min = np.asarray(mins, dtype=np.float32)
        self._joint_max = np.asarray(maxs, dtype=np.float32)

    def _init_home_q_rad(self) -> None:
        home = np.asarray(self._spec.robot.home_ctrl_rad, dtype=np.float32).reshape(-1)
        if home.size != self._action_dim:
            raise ValueError(
                f"home_ctrl_rad length {home.size} != action_dim {self._action_dim}"
            )
        self._home_q_rad = np.clip(home, self._joint_min, self._joint_max).astype(
            np.float32
        )

    def _init_hardware_mapping(self) -> None:
        hw_by_name = {name: idx for idx, name in enumerate(self._hardware_actuator_names)}
        missing_active = [
            name for name in self._policy_actuator_names if name not in hw_by_name
        ]
        if missing_active:
            raise ValueError(
                f"hardware RobotIO is missing standing policy actuators: {missing_active}"
            )
        missing_fixed = [
            name for name in self._fixed_home_targets_rad if name not in hw_by_name
        ]
        if missing_fixed:
            raise ValueError(
                f"hardware RobotIO is missing fixed-home actuators: {missing_fixed}"
            )

        self._active_to_hw = np.asarray(
            [hw_by_name[name] for name in self._policy_actuator_names],
            dtype=np.intp,
        )
        active_set = set(self._policy_actuator_names)
        fixed_set = set(self._fixed_home_targets_rad)
        unmapped = [
            name
            for name in self._hardware_actuator_names
            if name not in active_set and name not in fixed_set
        ]
        if unmapped:
            raise ValueError(
                "standing runtime needs every hardware actuator to be policy-active "
                f"or fixed-home; unmapped={unmapped}"
            )

        hardware_home = np.zeros(len(self._hardware_actuator_names), dtype=np.float32)
        for name, idx in hw_by_name.items():
            if name in active_set:
                active_idx = self._policy_actuator_names.index(name)
                hardware_home[idx] = self._home_q_rad[active_idx]
            else:
                hardware_home[idx] = np.float32(self._fixed_home_targets_rad[name])
        self._hardware_home_q_rad = hardware_home

    def _slice_active_signals(self, signals: Signals) -> Signals:
        joint_pos = np.asarray(signals.joint_pos_rad, dtype=np.float32).reshape(-1)
        joint_vel = np.asarray(signals.joint_vel_rad_s, dtype=np.float32).reshape(-1)
        if joint_pos.size != len(self._hardware_actuator_names):
            raise ValueError(
                f"RobotIO joint_pos size {joint_pos.size} != hardware actuator count "
                f"{len(self._hardware_actuator_names)}"
            )
        if joint_vel.size != len(self._hardware_actuator_names):
            raise ValueError(
                f"RobotIO joint_vel size {joint_vel.size} != hardware actuator count "
                f"{len(self._hardware_actuator_names)}"
            )
        return Signals(
            quat_xyzw=np.asarray(signals.quat_xyzw, dtype=np.float32).reshape(4),
            gyro_rad_s=np.asarray(signals.gyro_rad_s, dtype=np.float32).reshape(3),
            joint_pos_rad=joint_pos[self._active_to_hw].astype(np.float32),
            joint_vel_rad_s=joint_vel[self._active_to_hw].astype(np.float32),
            foot_switches=np.asarray(signals.foot_switches, dtype=np.float32).reshape(4),
            timestamp_s=signals.timestamp_s,
        )

    def build_obs(self, signals: Signals, velocity_cmd: np.ndarray) -> np.ndarray:
        return build_observation(
            spec=self._spec,
            state=PolicyState(prev_action=self._state.last_applied_action),
            signals=signals,
            velocity_cmd=_velocity_cmd_scalar(velocity_cmd),
        )

    def compose_and_apply(
        self, raw_action: np.ndarray, *, action_scale: float = 1.0
    ) -> tuple[np.ndarray, np.ndarray]:
        raw = np.asarray(raw_action, dtype=np.float32).reshape(-1)
        if raw.size != self._action_dim:
            raise ValueError(
                f"raw_action has {raw.size} elements, expected {self._action_dim}"
            )
        scaled_raw = raw * np.float32(_sanitize_action_scale(action_scale))
        applied, new_state = postprocess_action(
            spec=self._spec,
            state=PolicyState(prev_action=self._state.pending_action),
            action_raw=scaled_raw,
        )
        filtered = np.asarray(applied, dtype=np.float32).reshape(self._action_dim)
        if self._action_delay_enabled:
            applied = self._state.pending_action.copy()
        else:
            applied = filtered
        target_q = NumpyCalibOps.action_to_ctrl(spec=self._spec, action=applied)
        target_q = np.clip(target_q, self._joint_min, self._joint_max).astype(np.float32)
        self._state.pending_action = np.asarray(new_state.prev_action, dtype=np.float32)
        self._state.last_applied_action = applied.copy()
        self._state.step_idx += 1
        return target_q, applied

    def hold_home_step(self) -> tuple[np.ndarray, np.ndarray]:
        zeros = np.zeros(self._action_dim, dtype=np.float32)
        self._state.pending_action = zeros.copy()
        self._state.last_applied_action = zeros.copy()
        self._state.step_idx += 1
        return self._home_q_rad.copy(), zeros

    def _expand_to_hardware_ctrl(self, active_target_q: np.ndarray) -> np.ndarray:
        full = self._hardware_home_q_rad.copy()
        full[self._active_to_hw] = np.asarray(active_target_q, dtype=np.float32).reshape(
            self._action_dim
        )
        return full.astype(np.float32)

    def step(
        self,
        velocity_cmd: np.ndarray,
        *,
        force_home_hold: bool = False,
        home_hold_mode: str = "home_hold",
        action_scale: float = 1.0,
    ) -> dict:
        step_t0 = time.monotonic()
        read_t0 = time.monotonic()
        full_signals = self._robot_io.read()
        read_s = time.monotonic() - read_t0

        obs_t0 = time.monotonic()
        signals = self._slice_active_signals(full_signals)
        obs = self.build_obs(signals, velocity_cmd)
        obs_s = time.monotonic() - obs_t0

        policy_t0 = time.monotonic()
        if force_home_hold:
            raw = np.zeros(self._action_dim, dtype=np.float32)
            control_mode = str(home_hold_mode)
        elif self._should_hold_home_for_cmd(velocity_cmd):
            raw = np.zeros(self._action_dim, dtype=np.float32)
            control_mode = "zero_cmd_hold_home"
        else:
            raw = np.asarray(self._policy.predict(obs), dtype=np.float32).reshape(-1)
            control_mode = "policy"
        policy_s = time.monotonic() - policy_t0

        compose_t0 = time.monotonic()
        if force_home_hold or control_mode == "zero_cmd_hold_home":
            target_q, applied = self.hold_home_step()
        else:
            target_q, applied = self.compose_and_apply(raw, action_scale=action_scale)
        hardware_target_q = self._expand_to_hardware_ctrl(target_q)
        compose_s = time.monotonic() - compose_t0

        write_t0 = time.monotonic()
        self._robot_io.write_ctrl(hardware_target_q)
        write_s = time.monotonic() - write_t0

        timing_s = {
            "read": read_s,
            "obs": obs_s,
            "policy": policy_s,
            "compose": compose_s,
            "write": write_s,
            "step": time.monotonic() - step_t0,
        }
        io_timing_s = getattr(self._robot_io, "last_timing_s", None)
        if isinstance(io_timing_s, dict):
            for key, value in io_timing_s.items():
                try:
                    timing_s[f"io_{key}"] = float(value)
                except (TypeError, ValueError):
                    pass
        servo_metrics = getattr(self._robot_io, "last_servo_metrics", None)
        if not isinstance(servo_metrics, dict):
            servo_metrics = {}

        return {
            "step_idx": int(self._state.step_idx),
            "obs": obs,
            "raw_action": raw,
            "applied_action": applied,
            "target_q_rad": target_q,
            "hardware_target_q_rad": hardware_target_q,
            "signals": signals,
            "hardware_signals": full_signals,
            "control_mode": control_mode,
            "timing_s": timing_s,
            "servo_metrics": dict(servo_metrics),
            "obs_debug": {"velocity_cmd": _as_three_vec(velocity_cmd)},
            "action_scale": _sanitize_action_scale(action_scale),
        }

    def _should_hold_home_for_cmd(self, velocity_cmd: np.ndarray) -> bool:
        threshold = self._zero_cmd_hold_home_deadzone
        if threshold is None:
            return False
        cmd = _as_three_vec(velocity_cmd)
        return bool(float(np.max(np.abs(cmd))) <= float(threshold))


def _as_three_vec(velocity_cmd: np.ndarray) -> np.ndarray:
    cmd = np.asarray(velocity_cmd, dtype=np.float32).reshape(-1)
    if cmd.size == 1:
        return np.array([float(cmd[0]), 0.0, 0.0], dtype=np.float32)
    if cmd.size == 3:
        return cmd.astype(np.float32)
    raise ValueError(
        f"velocity_cmd must be scalar or length-3 (vx, vy, wz); got size {cmd.size}"
    )


def _velocity_cmd_scalar(velocity_cmd: np.ndarray) -> np.ndarray:
    cmd = _as_three_vec(velocity_cmd)
    return np.asarray([cmd[0]], dtype=np.float32)


def _sanitize_action_scale(value: float) -> float:
    try:
        scale = float(value)
    except (TypeError, ValueError):
        return 1.0
    if not np.isfinite(scale):
        return 1.0
    return min(1.0, max(0.0, scale))
