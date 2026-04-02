from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from control.kinematics.leg_ik import LegIkConfig
from control.references.walking_ref_v1 import (
    WalkingRefV1Config,
    WalkingRefV1Input,
    WalkingRefV1State,
    step_reference,
)


@dataclass(frozen=True)
class RuntimeLocRefFeatures:
    phase_sin_cos: np.ndarray
    stance_foot: np.ndarray
    next_foothold: np.ndarray
    swing_pos: np.ndarray
    swing_vel: np.ndarray
    pelvis_targets: np.ndarray
    history: np.ndarray


class RuntimeLocRefBuilder:
    """Runtime/replay locomotion-reference feature builder for wr_obs_v4."""

    def __init__(
        self,
        *,
        default_dt_s: float,
        actuator_names: tuple[str, ...] | list[str] | None = None,
        config: WalkingRefV1Config | None = None,
        leg_cfg: LegIkConfig | None = None,
    ) -> None:
        if default_dt_s <= 0.0:
            raise ValueError("default_dt_s must be > 0")
        self._default_dt_s = float(default_dt_s)
        self._config = config if config is not None else WalkingRefV1Config()
        self._leg_cfg = leg_cfg if leg_cfg is not None else LegIkConfig()
        self._state = WalkingRefV1State()
        self._history = np.zeros((4,), dtype=np.float32)
        self._pseudo_root_xy = np.zeros((2,), dtype=np.float32)
        idx = {str(name): i for i, name in enumerate(tuple(actuator_names or ()))}
        self._idx_lhp = idx.get("left_hip_pitch", -1)
        self._idx_lkp = idx.get("left_knee_pitch", -1)
        self._idx_rhp = idx.get("right_hip_pitch", -1)
        self._idx_rkp = idx.get("right_knee_pitch", -1)

    def _estimate_root_velocity_heading_local(
        self,
        *,
        joint_pos_rad: np.ndarray | None,
        joint_vel_rad_s: np.ndarray | None,
    ) -> np.ndarray:
        if (
            joint_pos_rad is None
            or joint_vel_rad_s is None
            or self._idx_lhp < 0
            or self._idx_lkp < 0
            or self._idx_rhp < 0
            or self._idx_rkp < 0
        ):
            return np.zeros((2,), dtype=np.float32)
        q = np.asarray(joint_pos_rad, dtype=np.float32)
        qd = np.asarray(joint_vel_rad_s, dtype=np.float32)
        if q.ndim != 1 or qd.ndim != 1:
            return np.zeros((2,), dtype=np.float32)
        if q.shape[0] <= max(self._idx_lhp, self._idx_lkp, self._idx_rhp, self._idx_rkp):
            return np.zeros((2,), dtype=np.float32)
        if qd.shape[0] <= max(self._idx_lhp, self._idx_lkp, self._idx_rhp, self._idx_rkp):
            return np.zeros((2,), dtype=np.float32)

        use_left = self._state.stance_foot_id == 0
        if use_left:
            h = float(q[self._idx_lhp])
            k = float(q[self._idx_lkp])
            hd = float(qd[self._idx_lhp])
            kd = float(qd[self._idx_lkp])
        else:
            h = float(q[self._idx_rhp])
            k = float(q[self._idx_rkp])
            hd = float(qd[self._idx_rhp])
            kd = float(qd[self._idx_rkp])
        l1 = float(self._leg_cfg.upper_leg_length_m)
        l2 = float(self._leg_cfg.lower_leg_length_m)
        foot_xdot_rel = l1 * np.cos(h) * hd + l2 * np.cos(h + k) * (hd + kd)
        root_vx = -foot_xdot_rel
        return np.asarray([root_vx, 0.0], dtype=np.float32)

    def step(
        self,
        *,
        forward_speed_mps: float,
        dt_s: float | None = None,
        joint_pos_rad: np.ndarray | None = None,
        joint_vel_rad_s: np.ndarray | None = None,
        com_position_stance_frame: tuple[float, float] | None = None,
        com_velocity_stance_frame: tuple[float, float] | None = None,
    ) -> RuntimeLocRefFeatures:
        dt = float(dt_s) if (dt_s is not None and dt_s > 0.0) else self._default_dt_s
        est_vel_xy = self._estimate_root_velocity_heading_local(
            joint_pos_rad=joint_pos_rad,
            joint_vel_rad_s=joint_vel_rad_s,
        )
        if com_position_stance_frame is None:
            com_position_stance_frame = (
                float(self._pseudo_root_xy[0]),
                float(self._pseudo_root_xy[1]),
            )
        if com_velocity_stance_frame is None:
            if np.linalg.norm(est_vel_xy) > 1e-6:
                com_velocity_stance_frame = (float(est_vel_xy[0]), float(est_vel_xy[1]))
            else:
                com_velocity_stance_frame = (float(forward_speed_mps), 0.0)
        vel_xy = np.asarray(com_velocity_stance_frame, dtype=np.float32).reshape(2)
        self._pseudo_root_xy = self._pseudo_root_xy + vel_xy * np.float32(dt)

        out = step_reference(
            config=self._config,
            state=self._state,
            inputs=WalkingRefV1Input(
                forward_speed_mps=float(forward_speed_mps),
                com_position_stance_frame=(
                    float(com_position_stance_frame[0]),
                    float(com_position_stance_frame[1]),
                ),
                com_velocity_stance_frame=(
                    float(com_velocity_stance_frame[0]),
                    float(com_velocity_stance_frame[1]),
                ),
            ),
            dt_s=dt,
        )
        self._state = out.next_state
        ref = out.reference

        phase_sin_cos = np.asarray(
            [ref.gait_phase_sin, ref.gait_phase_cos],
            dtype=np.float32,
        )
        self._history = np.asarray(
            [self._history[2], self._history[3], phase_sin_cos[0], phase_sin_cos[1]],
            dtype=np.float32,
        )

        return RuntimeLocRefFeatures(
            phase_sin_cos=phase_sin_cos,
            stance_foot=np.asarray([float(ref.stance_foot_id)], dtype=np.float32),
            next_foothold=np.asarray(ref.desired_next_foothold_stance_frame, dtype=np.float32),
            swing_pos=np.asarray(ref.desired_swing_foot_position, dtype=np.float32),
            swing_vel=np.asarray(ref.desired_swing_foot_velocity, dtype=np.float32),
            pelvis_targets=np.asarray(
                [
                    ref.desired_pelvis_height_m,
                    ref.desired_pelvis_roll_rad,
                    ref.desired_pelvis_pitch_rad,
                ],
                dtype=np.float32,
            ),
            history=self._history.copy(),
        )
