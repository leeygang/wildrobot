"""Unified walking controller with nominal-only and residual-PPO modes.

This is the top-level orchestrator for v0.19.4 (nominal-only) and v0.19.5
(residual PPO) deployment.  It wraps the walking reference generator and IK
adapter into a single step() interface.

Modes:
    nominal_only:  ref → IK → q_ref → ctrl  (no neural network needed)
    residual_ppo:  ref → IK → q_ref → q_ref + delta_q → ctrl
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Tuple

import numpy as np

from control.locomotion.nominal_ik_adapter import (
    NominalIkConfig,
    NominalIkResult,
    compute_nominal_q_ref,
)
from control.references.walking_ref_v2 import (
    WalkingRefV2Config,
    WalkingRefV2Input,
    WalkingRefV2Mode,
    WalkingRefV2State,
    WalkingReferenceV2Output,
    step_reference_v2,
)


class ControllerMode(str, Enum):
    NOMINAL_ONLY = "nominal_only"
    RESIDUAL_PPO = "residual_ppo"


@dataclass(frozen=True)
class WalkingControllerConfig:
    """Configuration for the walking controller."""

    mode: ControllerMode = ControllerMode.NOMINAL_ONLY
    ref_config: WalkingRefV2Config = field(default_factory=WalkingRefV2Config)
    ik_config: NominalIkConfig = field(default_factory=NominalIkConfig)
    residual_scale: float = 0.18
    dt_s: float = 0.02


@dataclass
class WalkingControllerState:
    """Mutable state for the walking controller."""

    ref_state: WalkingRefV2State = field(default_factory=WalkingRefV2State)
    phase_history: np.ndarray = field(
        default_factory=lambda: np.zeros(4, dtype=np.float32)
    )
    prev_q_ref: np.ndarray | None = None
    last_ref_output: WalkingReferenceV2Output | None = None

    def reset(self, initial_mode: WalkingRefV2Mode = WalkingRefV2Mode.SUPPORT_STABILIZE) -> None:
        """Reset controller state for a new episode."""
        self.ref_state = WalkingRefV2State(mode_id=int(initial_mode))
        self.phase_history = np.zeros(4, dtype=np.float32)
        self.prev_q_ref = None
        self.last_ref_output = None


@dataclass(frozen=True)
class WalkingControllerOutput:
    """Output from one controller step."""

    q_target: np.ndarray       # Final joint targets (after residual if applicable)
    q_ref: np.ndarray          # Nominal IK reference (before residual)
    ref_output: WalkingReferenceV2Output  # Walking reference output
    ik_result: NominalIkResult  # IK adapter output
    mode: ControllerMode


class WalkingController:
    """Unified walking controller for nominal-only and residual-PPO modes.

    Usage::

        # v0.19.4: nominal-only deployment
        cfg = WalkingControllerConfig(mode=ControllerMode.NOMINAL_ONLY)
        ctrl = WalkingController(cfg)
        out = ctrl.step(forward_speed_mps=0.15, ...)
        servo_targets = out.q_target  # send directly to servos

        # v0.19.5: with PPO residual
        cfg = WalkingControllerConfig(mode=ControllerMode.RESIDUAL_PPO)
        ctrl = WalkingController(cfg)
        out = ctrl.step(forward_speed_mps=0.15, ...)
        # ... build obs, run ONNX policy to get delta_q ...
        q_target = ctrl.compose_residual(out.q_ref, delta_q)
    """

    def __init__(self, config: WalkingControllerConfig) -> None:
        self._config = config
        self._state = WalkingControllerState()

    @property
    def config(self) -> WalkingControllerConfig:
        return self._config

    @property
    def state(self) -> WalkingControllerState:
        return self._state

    @property
    def mode(self) -> ControllerMode:
        return self._config.mode

    def reset(
        self,
        initial_mode: WalkingRefV2Mode = WalkingRefV2Mode.SUPPORT_STABILIZE,
    ) -> None:
        """Reset the controller for a new episode."""
        self._state.reset(initial_mode)

    def step(
        self,
        *,
        forward_speed_mps: float,
        com_position_stance_frame: Tuple[float, float] = (0.0, 0.0),
        com_velocity_stance_frame: Tuple[float, float] = (0.0, 0.0),
        root_pitch_rad: float = 0.0,
        root_pitch_rate_rad_s: float = 0.0,
        root_height_m: float | None = None,
        left_foot_loaded: bool = True,
        right_foot_loaded: bool = False,
        home_pose: np.ndarray | None = None,
        idx: Dict[str, int] | None = None,
        delta_q: np.ndarray | None = None,
    ) -> WalkingControllerOutput:
        """Run one controller step.

        Args:
            forward_speed_mps: Commanded forward speed.
            com_position_stance_frame: COM (x,y) in stance-foot frame.
            com_velocity_stance_frame: COM velocity (vx,vy) in stance-foot frame.
            root_pitch_rad: Current root pitch.
            root_pitch_rate_rad_s: Current root pitch rate.
            root_height_m: Current root height (for startup transition).
            left_foot_loaded: Left foot contact flag.
            right_foot_loaded: Right foot contact flag.
            home_pose: (9,) home joint positions.
            idx: Joint name → index map.
            delta_q: (9,) PPO residual (only used in RESIDUAL_PPO mode).

        Returns:
            WalkingControllerOutput with q_target, q_ref, and diagnostics.
        """
        cfg = self._config
        state = self._state

        # Step walking reference
        ref_input = WalkingRefV2Input(
            forward_speed_mps=forward_speed_mps,
            com_position_stance_frame=com_position_stance_frame,
            com_velocity_stance_frame=com_velocity_stance_frame,
            root_pitch_rad=root_pitch_rad,
            root_pitch_rate_rad_s=root_pitch_rate_rad_s,
            left_foot_loaded=left_foot_loaded,
            right_foot_loaded=right_foot_loaded,
        )
        ref_output = step_reference_v2(
            config=cfg.ref_config,
            state=state.ref_state,
            inputs=ref_input,
            dt_s=cfg.dt_s,
            root_height_m=root_height_m,
        )

        # Update reference state
        state.ref_state = ref_output.next_state
        state.last_ref_output = ref_output

        # Update phase history
        ref = ref_output.reference
        state.phase_history = np.array(
            [
                state.phase_history[2],
                state.phase_history[3],
                ref.gait_phase_sin,
                ref.gait_phase_cos,
            ],
            dtype=np.float32,
        )

        # Compute nominal q_ref via IK, threading COM trajectory from ref
        ik_result = compute_nominal_q_ref(
            config=cfg.ik_config,
            pelvis_height_m=ref.desired_pelvis_height_m,
            pelvis_roll_rad=ref.desired_pelvis_roll_rad,
            pelvis_pitch_rad=ref.desired_pelvis_pitch_rad,
            stance_foot_id=ref.stance_foot_id,
            swing_pos=ref.desired_swing_foot_position,
            com_x_planned=ref_output.com_x_planned,
            mode_id=ref_output.hybrid_mode_id,
            home_pose=home_pose,
            idx=idx,
        )
        q_ref = ik_result.q_ref.copy()
        state.prev_q_ref = q_ref.copy()

        # Compose final targets
        if cfg.mode == ControllerMode.RESIDUAL_PPO and delta_q is not None:
            q_target = self.compose_residual(q_ref, delta_q)
        else:
            q_target = q_ref.copy()

        return WalkingControllerOutput(
            q_target=q_target,
            q_ref=q_ref,
            ref_output=ref_output,
            ik_result=ik_result,
            mode=cfg.mode,
        )

    def compose_residual(
        self,
        q_ref: np.ndarray,
        delta_q: np.ndarray,
    ) -> np.ndarray:
        """Compose q_target = clip(q_ref + scale * delta_q, min, max).

        Args:
            q_ref: (N,) nominal joint targets from IK.
            delta_q: (N,) PPO residual output (typically in [-1, 1]).

        Returns:
            (N,) clipped joint targets.
        """
        cfg = self._config.ik_config
        range_min = np.array(cfg.joint_range_min[: len(q_ref)], dtype=np.float32)
        range_max = np.array(cfg.joint_range_max[: len(q_ref)], dtype=np.float32)
        half_spans = 0.5 * (range_max - range_min)

        scaled_delta = np.clip(delta_q, -1.0, 1.0) * self._config.residual_scale * half_spans
        q_target = np.clip(q_ref + scaled_delta, range_min, range_max)
        return q_target

    def get_reference_features(self) -> Dict[str, np.ndarray]:
        """Get current reference state as obs features for wr_obs_v4.

        Returns dict with arrays matching the loc_ref observation slots.
        """
        state = self._state
        if state.last_ref_output is None:
            raise RuntimeError("Must call step() before get_reference_features()")

        ref = state.last_ref_output.reference
        return {
            "phase_sin_cos": np.array(
                [ref.gait_phase_sin, ref.gait_phase_cos], dtype=np.float32
            ),
            "stance_foot": np.array([float(ref.stance_foot_id)], dtype=np.float32),
            "next_foothold": np.array(
                ref.desired_next_foothold_stance_frame, dtype=np.float32
            ),
            "swing_pos": np.array(ref.desired_swing_foot_position, dtype=np.float32),
            "swing_vel": np.array(ref.desired_swing_foot_velocity, dtype=np.float32),
            "pelvis_targets": np.array(
                [
                    ref.desired_pelvis_height_m,
                    ref.desired_pelvis_roll_rad,
                    ref.desired_pelvis_pitch_rad,
                ],
                dtype=np.float32,
            ),
            "phase_history": state.phase_history.copy(),
        }
