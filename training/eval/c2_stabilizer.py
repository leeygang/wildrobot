"""v0.20.0-C2 validation-only stabilizer harness.

WARNING: validation tooling only.  This module MUST NOT be imported
from anything under ``runtime/``.  The contract in
``training/docs/reference_design.md`` v0.20.0-C explicitly forbids
re-using this code in the deployed runtime stack — the runtime
controller is PPO (v0.20.1), not this harness.

Three bounded feedback channels, each with a hard clip:

  - Torso pitch PD: K_p * pitch + K_d * pitch_rate -> ankle pitch
    offset on the stance side(s).  Hard clip: ±0.10 rad.
  - Torso roll  PD: same form -> hip roll offset on stance side(s).
    Hard clip: ±0.05 rad.
  - Capture-point swing nudge: (x_cp - x_swing) * gain -> swing-leg
    hip pitch offset, snapshotted at the foot-lift transition and
    held until the next touchdown.  Hard clip: ±0.03 m on the
    swing-x equivalent (converted to hip pitch via the leg length).

Allowed inputs (and only these):
  (a) measured MuJoCo foot/floor contacts via ``data.contact``
      geom-pair lookup, identical to the C1 touchdown detector.
  (b) time-derived gait phase ``(sin, cos)(2*pi * t / cycle_time)``.

Forbidden inputs:
  - ``traj.stance_foot_id`` / ``traj.contact_mask`` / ``traj.phase``
  - any FSM-derived annotation
  - the stored reference's per-step contact schedule
  - any reference-based foothold preview

The "stance side" used for the torso PD channels is determined from
the measured foot/floor contacts only.  The capture-point nudge
identifies "foot lift" from a measured 1->0 contact transition on
that foot.

Per-step output: a modified copy of ``q_ref`` (length 19) with the
allowed offsets applied to the leg actuator slots (indices 0..7 in
policy order: L/R hip pitch, hip roll, knee, ankle).  The harness
returns its clip-saturation flags so the closeout artifact can
report harness-clip % per channel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


# Policy joint indices for legs (matches mujoco_robot_config.json
# actuated_joint_specs ordering and the IK output in zmp_walk.py).
_L_HIP_PITCH, _R_HIP_PITCH = 0, 1
_L_HIP_ROLL, _R_HIP_ROLL = 2, 3
_L_KNEE, _R_KNEE = 4, 5
_L_ANK_PITCH, _R_ANK_PITCH = 6, 7


@dataclass
class C2StabilizerConfig:
    """Gains and hard clips for the C2 validation harness.

    Hard clips MUST match the contract in reference_design.md.
    Gains may be tuned freely as long as the resulting offsets stay
    inside the hard clips for the whole replay.
    """

    # Torso pitch PD.  Default gains tuned conservatively — the
    # closeout artifact must show clip-saturation < 25 % per channel.
    # Sign of the ankle correction is empirical (FK frames make
    # closed-form sign analysis fragile); see ``apply_pitch_sign`` /
    # ``apply_roll_sign`` to flip without touching control logic.
    pitch_kp: float = 0.25       # rad of ankle offset per rad of pitch error
    pitch_kd: float = 0.03       # rad / (rad/s)
    pitch_clip: float = 0.10     # HARD clip per contract
    apply_pitch_sign: float = -1.0   # empirically determined

    # Torso roll PD
    roll_kp: float = 0.20
    roll_kd: float = 0.025
    roll_clip: float = 0.05      # HARD clip per contract
    apply_roll_sign: float = +1.0    # empirically determined

    # Capture-point swing nudge.  Gain bumped from 0.30 → 0.7 after
    # the closeout 6000177 retro showed the CP channel was rarely
    # using its ±0.03 m clip budget (clip-sat 0-3 %).  Bumped further
    # to 1.5 in a probe but pitch_PD started saturating because the
    # CP nudge moves swing further from foot-flat IK and increases
    # ankle demand — settled on 0.7 as the largest gain that does
    # not push pitch_PD past hard-fail in the seed=0 vx=0.15 probe.
    # Clip is unchanged (the contract bound, not the gain).
    cp_gain: float = 0.7         # dimensionless; nudge = gain * (x_cp - x_swing)
    cp_clip: float = 0.03        # HARD clip per contract (m of swing-x)

    # Geometry constants (used only for capture-point and swing-x → hip-pitch
    # conversion; not for reading any reference annotation).
    com_height_m: float = 0.473  # for w = sqrt(g / h_com)
    leg_length_m: float = 0.413  # upper + lower leg, for Δhip ≈ Δfoot_x / leg
    g: float = 9.81

    # Initialized from trajectory metadata at construction time.
    cycle_time_s: float = 0.64


class C2Stabilizer:
    """Bounded validation-only feedback harness for v0.20.0-C2.

    Usage::

        stab = C2Stabilizer(model, cfg)
        stab.reset()                        # call once per episode
        for step in episode:
            q_ref = traj.q_ref[idx]
            q_mod, info = stab.step(model, data, q_ref, t_now)
            mapper.set_all_ctrl(data, q_mod)
            mj_step(...)
    """

    def __init__(self, model, cfg: C2StabilizerConfig | None = None) -> None:
        import mujoco  # local import to avoid a hard dep at module load

        self.cfg = cfg or C2StabilizerConfig()
        self.model = model

        # Identify foot bodies and their collision geoms.
        self._left_foot_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "left_foot")
        self._right_foot_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "right_foot")
        self._floor_geom_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

        def _foot_geoms(body_id: int) -> set:
            return {g for g in range(model.ngeom)
                    if model.geom_bodyid[g] == body_id
                    and int(model.geom_contype[g]) != 0}

        self._left_foot_geoms = _foot_geoms(self._left_foot_id)
        self._right_foot_geoms = _foot_geoms(self._right_foot_id)

        # Episode-local state (reset() initializes).
        self._prev_in_contact = [True, True]
        self._held_swing_offset = [0.0, 0.0]   # swing-x meters, per side
        self._swing_active = [False, False]    # currently in swing for nudge

        # Counters for harness-clip-saturation reporting.
        self._n_steps = 0
        self._clipped_pitch = 0
        self._clipped_roll = 0
        self._clipped_cp = 0

    # ----- public API ----------------------------------------------------

    def reset(self, data) -> None:
        """Reset episode-local state from the current physics state.

        Initializes contact state from real MuJoCo contacts so the
        first step does not see a phantom 1->0 transition.
        """
        self._prev_in_contact = [
            self._foot_floor_in_contact(data, self._left_foot_geoms),
            self._foot_floor_in_contact(data, self._right_foot_geoms),
        ]
        self._held_swing_offset = [0.0, 0.0]
        self._swing_active = [False, False]
        self._n_steps = 0
        self._clipped_pitch = 0
        self._clipped_roll = 0
        self._clipped_cp = 0

    def step(self, model, data, q_ref: np.ndarray,
             t_now: float) -> Tuple[np.ndarray, Dict]:
        """Compute and apply the bounded stabilizer offsets.

        Returns ``(q_modified, info)``.  ``q_modified`` is a fresh
        ndarray (length 19); the original ``q_ref`` is not mutated.
        ``info`` carries per-step quantities used by the closeout
        artifact (clip flags, raw + clipped offsets, contact state).
        """
        cfg = self.cfg
        q_mod = np.asarray(q_ref, dtype=np.float32).copy()

        # --- 1. Read torso state (allowed inputs only) ---
        pitch, roll = _torso_pitch_roll(data.qpos[3:7])
        # qvel[3:6] is base angular velocity in body frame for a free
        # joint; qvel[3]=ωx (roll rate), qvel[4]=ωy (pitch rate).
        roll_rate = float(data.qvel[3])
        pitch_rate = float(data.qvel[4])

        # --- 2. Read measured contacts (allowed) ---
        l_in = self._foot_floor_in_contact(data, self._left_foot_geoms)
        r_in = self._foot_floor_in_contact(data, self._right_foot_geoms)

        # --- 3. Time-derived gait phase (allowed; not from traj) ---
        # Computed but kept for completeness — current channels do
        # not condition on phase, only on contact state.  Documented
        # as available so future iterations stay within the contract.
        _phase_sin = np.sin(2.0 * np.pi * t_now / cfg.cycle_time_s)
        _phase_cos = np.cos(2.0 * np.pi * t_now / cfg.cycle_time_s)

        # --- 4. Torso pitch PD → ankle pitch offset on stance side ---
        pitch_raw = cfg.apply_pitch_sign * \
            -(cfg.pitch_kp * pitch + cfg.pitch_kd * pitch_rate)
        pitch_off = float(np.clip(pitch_raw, -cfg.pitch_clip, +cfg.pitch_clip))
        if pitch_off != pitch_raw:
            self._clipped_pitch += 1
        if l_in:
            q_mod[_L_ANK_PITCH] = q_mod[_L_ANK_PITCH] + pitch_off
        if r_in:
            q_mod[_R_ANK_PITCH] = q_mod[_R_ANK_PITCH] + pitch_off

        # --- 5. Torso roll PD → hip roll offset on stance side ---
        roll_raw = cfg.apply_roll_sign * \
            -(cfg.roll_kp * roll + cfg.roll_kd * roll_rate)
        roll_off = float(np.clip(roll_raw, -cfg.roll_clip, +cfg.roll_clip))
        if roll_off != roll_raw:
            self._clipped_roll += 1
        # WildRobot sign convention (zmp_walk.py): q[L_hip_roll] = -hip_r,
        # q[R_hip_roll] = +hip_r.  Apply offset with matching signs so
        # positive roll_off rotates the pelvis correction consistently.
        if l_in:
            q_mod[_L_HIP_ROLL] = q_mod[_L_HIP_ROLL] + (-roll_off)
        if r_in:
            q_mod[_R_HIP_ROLL] = q_mod[_R_HIP_ROLL] + (+roll_off)

        # --- 6. Capture-point swing nudge (snapshot at foot lift) ---
        cp_clipped_this_step = False
        for side, is_in, foot_id, hip_idx, sign in (
            (0, l_in, self._left_foot_id,  _L_HIP_PITCH, +1.0),
            (1, r_in, self._right_foot_id, _R_HIP_PITCH, -1.0),
        ):
            # Foot just lifted: snapshot a new held offset.
            if (not is_in) and self._prev_in_contact[side]:
                pelvis_x = float(data.qpos[0])
                pelvis_vx = float(data.qvel[0])
                w = np.sqrt(cfg.g / cfg.com_height_m)
                x_cp = pelvis_x + pelvis_vx / w
                x_swing = float(data.xpos[foot_id, 0])
                nudge_raw = cfg.cp_gain * (x_cp - x_swing)
                nudge = float(np.clip(nudge_raw, -cfg.cp_clip, +cfg.cp_clip))
                if nudge != nudge_raw:
                    cp_clipped_this_step = True
                self._held_swing_offset[side] = nudge
                self._swing_active[side] = True
            # Foot just landed: clear the held offset.
            if is_in and not self._prev_in_contact[side]:
                self._held_swing_offset[side] = 0.0
                self._swing_active[side] = False

            # Apply the held offset while in swing, converted from
            # swing-x meters to hip-pitch radians via leg length.
            # Sign per WildRobot convention: positive q[L_hip_pitch]
            # = L foot forward; positive q[R_hip_pitch] = R foot
            # backward (so we negate for the right side).
            if self._swing_active[side]:
                d_hip = self._held_swing_offset[side] / cfg.leg_length_m
                q_mod[hip_idx] = q_mod[hip_idx] + sign * d_hip

        if cp_clipped_this_step:
            self._clipped_cp += 1

        # Update prev-contact for next step.
        self._prev_in_contact[0] = l_in
        self._prev_in_contact[1] = r_in
        self._n_steps += 1

        info = {
            "pitch_off": pitch_off,
            "roll_off": roll_off,
            "swing_off_l": self._held_swing_offset[0],
            "swing_off_r": self._held_swing_offset[1],
            "stance_l": l_in,
            "stance_r": r_in,
            "phase_sin": float(_phase_sin),
            "phase_cos": float(_phase_cos),
        }
        return q_mod, info

    def clip_saturation(self) -> Dict[str, float]:
        """Per-channel fraction of steps where the channel was clipped."""
        n = max(1, self._n_steps)
        return {
            "pitch": self._clipped_pitch / n,
            "roll":  self._clipped_roll / n,
            "cp":    self._clipped_cp / n,
            "any":   max(self._clipped_pitch, self._clipped_roll,
                         self._clipped_cp) / n,
        }

    # ----- internals -----------------------------------------------------

    def _foot_floor_in_contact(self, data, foot_geom_set: set) -> bool:
        """True if any contact pair links a foot collision geom and the floor."""
        for c_idx in range(data.ncon):
            c = data.contact[c_idx]
            g1, g2 = int(c.geom1), int(c.geom2)
            if (g1 == self._floor_geom_id and g2 in foot_geom_set) or \
               (g2 == self._floor_geom_id and g1 in foot_geom_set):
                return True
        return False


def _torso_pitch_roll(quat_wxyz) -> Tuple[float, float]:
    """Extract pitch and roll from a (w, x, y, z) quaternion."""
    qw, qx, qy, qz = float(quat_wxyz[0]), float(quat_wxyz[1]), \
        float(quat_wxyz[2]), float(quat_wxyz[3])
    pitch = float(np.arctan2(2.0 * (qw * qy - qz * qx),
                             1.0 - 2.0 * (qx * qx + qy * qy)))
    roll = float(np.arctan2(2.0 * (qw * qx + qy * qz),
                            1.0 - 2.0 * (qx * qx + qy * qy)))
    return pitch, roll
