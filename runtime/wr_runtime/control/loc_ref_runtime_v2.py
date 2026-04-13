"""Walking reference v2 runtime wrapper (NumPy).

Wraps walking_ref_v2's pure-Python step_reference_v2() for on-robot use,
following the same pattern as loc_ref_runtime.py (which wraps v1).

Produces RuntimeLocRefFeatures compatible with wr_obs_v4 observation layout.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from control.references.walking_ref_v2 import (
    WalkingRefV2Config,
    WalkingRefV2Input,
    WalkingRefV2Mode,
    WalkingRefV2State,
    step_reference_v2,
)


@dataclass(frozen=True)
class RuntimeLocRefV2Features:
    """Observation features from walking reference v2 for wr_obs_v4."""

    phase_sin_cos: np.ndarray   # (2,) sin/cos of gait phase
    stance_foot: np.ndarray     # (1,) 0=left, 1=right
    next_foothold: np.ndarray   # (2,) (x, y) in stance frame
    swing_pos: np.ndarray       # (3,) (x, y, z) swing foot position
    swing_vel: np.ndarray       # (3,) (x, y, z) swing foot velocity
    pelvis_targets: np.ndarray  # (3,) (height, roll, pitch)
    history: np.ndarray         # (4,) two previous phase sin/cos pairs
    support_health: float
    mode_id: int


class RuntimeLocRefV2Builder:
    """Runtime walking reference v2 feature builder for wr_obs_v4.

    Usage::

        builder = RuntimeLocRefV2Builder(default_dt_s=0.02)
        features = builder.step(
            forward_speed_mps=0.15,
            root_pitch_rad=imu_pitch,
            root_pitch_rate_rad_s=imu_pitch_rate,
            left_foot_loaded=left_switch,
            right_foot_loaded=right_switch,
        )
        obs[38:40] = features.phase_sin_cos
        obs[40:41] = features.stance_foot
        ...
    """

    def __init__(
        self,
        *,
        default_dt_s: float,
        config: WalkingRefV2Config | None = None,
        initial_mode: WalkingRefV2Mode = WalkingRefV2Mode.SUPPORT_STABILIZE,
    ) -> None:
        if default_dt_s <= 0.0:
            raise ValueError("default_dt_s must be > 0")
        self._default_dt_s = float(default_dt_s)
        self._config = config if config is not None else WalkingRefV2Config()
        self._state = WalkingRefV2State(mode_id=int(initial_mode))
        self._history = np.zeros(4, dtype=np.float32)

    @property
    def state(self) -> WalkingRefV2State:
        return self._state

    @property
    def config(self) -> WalkingRefV2Config:
        return self._config

    def reset(
        self,
        initial_mode: WalkingRefV2Mode = WalkingRefV2Mode.SUPPORT_STABILIZE,
    ) -> None:
        """Reset the reference to initial state."""
        self._state = WalkingRefV2State(mode_id=int(initial_mode))
        self._history = np.zeros(4, dtype=np.float32)

    def step(
        self,
        *,
        forward_speed_mps: float,
        dt_s: float | None = None,
        com_position_stance_frame: tuple[float, float] | None = None,
        com_velocity_stance_frame: tuple[float, float] | None = None,
        root_pitch_rad: float = 0.0,
        root_pitch_rate_rad_s: float = 0.0,
        root_height_m: float | None = None,
        left_foot_loaded: bool = True,
        right_foot_loaded: bool = False,
    ) -> RuntimeLocRefV2Features:
        """Advance the walking reference and return obs features.

        Args:
            forward_speed_mps: Commanded forward speed.
            dt_s: Timestep override (defaults to default_dt_s).
            com_position_stance_frame: COM (x,y) in stance-foot frame.
            com_velocity_stance_frame: COM velocity (vx,vy) in stance-foot frame.
            root_pitch_rad: Current root pitch (from IMU).
            root_pitch_rate_rad_s: Current root pitch rate (from IMU).
            root_height_m: Current root height (optional, for startup).
            left_foot_loaded: Left foot contact flag.
            right_foot_loaded: Right foot contact flag.

        Returns:
            RuntimeLocRefV2Features for building wr_obs_v4 observation.
        """
        dt = float(dt_s) if dt_s is not None and dt_s > 0.0 else self._default_dt_s

        if com_position_stance_frame is None:
            com_position_stance_frame = (0.0, 0.0)
        if com_velocity_stance_frame is None:
            com_velocity_stance_frame = (float(forward_speed_mps), 0.0)

        ref_input = WalkingRefV2Input(
            forward_speed_mps=forward_speed_mps,
            com_position_stance_frame=com_position_stance_frame,
            com_velocity_stance_frame=com_velocity_stance_frame,
            root_pitch_rad=root_pitch_rad,
            root_pitch_rate_rad_s=root_pitch_rate_rad_s,
            left_foot_loaded=left_foot_loaded,
            right_foot_loaded=right_foot_loaded,
        )

        out = step_reference_v2(
            config=self._config,
            state=self._state,
            inputs=ref_input,
            dt_s=dt,
            root_height_m=root_height_m,
        )

        self._state = out.next_state
        ref = out.reference

        phase_sin_cos = np.array(
            [ref.gait_phase_sin, ref.gait_phase_cos],
            dtype=np.float32,
        )

        # Update phase history (rolling window of last 2 phases)
        self._history = np.array(
            [self._history[2], self._history[3], phase_sin_cos[0], phase_sin_cos[1]],
            dtype=np.float32,
        )

        return RuntimeLocRefV2Features(
            phase_sin_cos=phase_sin_cos,
            stance_foot=np.array([float(ref.stance_foot_id)], dtype=np.float32),
            next_foothold=np.array(
                ref.desired_next_foothold_stance_frame, dtype=np.float32
            ),
            swing_pos=np.array(ref.desired_swing_foot_position, dtype=np.float32),
            swing_vel=np.array(ref.desired_swing_foot_velocity, dtype=np.float32),
            pelvis_targets=np.array(
                [
                    ref.desired_pelvis_height_m,
                    ref.desired_pelvis_roll_rad,
                    ref.desired_pelvis_pitch_rad,
                ],
                dtype=np.float32,
            ),
            history=self._history.copy(),
            support_health=out.support_health,
            mode_id=out.hybrid_mode_id,
        )
