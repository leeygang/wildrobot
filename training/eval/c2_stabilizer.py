"""v0.20.0-C2 validation-only stabilizer harness.

WARNING: validation tooling only.  This module MUST NOT be imported
from anything under ``runtime/``.  The contract in
``training/docs/reference_design.md`` v0.20.0-C explicitly forbids
re-using this code in the deployed runtime stack — the runtime
controller is PPO (v0.20.1), not this harness.

Three bounded feedback channels, each with a hard clip:

  - Torso pitch PD: K_p * pitch + K_d * pitch_rate -> ankle pitch
    offset on the stance side(s).  Hard clip: ±0.10 rad.
  - Torso roll  PD: same form -> **ankle_roll** offset on stance
    side(s).  Hard clip: ±0.05 rad.  This was hip_roll pre-merge;
    the v20 ankle_roll merge added a dedicated ankle_roll DOF and
    the design call (Q1) is to use the local foot-flat actuator
    (TB-style) now that one exists, with identical servo power
    (htd45hServo, kp=21.1, ±4 Nm) to the ankle_pitch joint that
    already carries the pitch PD.
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

Per-step output: a modified copy of ``q_ref`` (length =
``model.nu``, e.g. 21 for the post-merge WildRobot) with the
allowed offsets applied to the leg actuator slots resolved by name
from ``mujoco_robot_config.json``.  The harness returns its
clip-saturation flags so the closeout artifact can report
harness-clip % per channel.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


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

    # Torso roll PD (post-merge: targets ankle_roll, not hip_roll).
    # The L/R sign mirror is automatically picked up from each
    # ankle_roll's ``policy_action_sign`` in the robot config — no
    # hard-coded mirror needed at this level.  ``apply_roll_sign``
    # is the unified scalar to flip the entire channel if the
    # closeout shows saturation in the wrong direction.
    roll_kp: float = 0.20
    roll_kd: float = 0.025
    roll_clip: float = 0.05      # HARD clip per contract
    apply_roll_sign: float = +1.0    # empirically determined

    # Capture-point swing nudge.  Magnitude bumped from 0.30 → 0.7
    # after the closeout 6000177 retro (CP channel was rarely using
    # its ±0.03 m clip budget).  Bumped further to 1.5 in a probe
    # but pitch_PD started saturating because the CP nudge moves
    # swing further from foot-flat IK and increases ankle demand —
    # settled on 0.7 as the largest magnitude that does not push
    # pitch_PD past hard-fail in the seed=0 vx=0.15 probe.
    #
    # Phase 9A C2 closeout (2026-05-08): sign FLIPPED 0.7 → -0.7.
    # Empirical sign-tuning sweep at vx=0.15 seed=0 after the v20
    # ankle_roll merge + CAD upper_leg mate fix shows the CP nudge
    # at +0.7 produces backward motion (-0.015 m / step at vx=0.15);
    # at -0.7 it produces forward motion (+0.010 m / step) with
    # 200/200 survival.  Magnitude unchanged (still |0.7|, the
    # pitch_PD-non-saturating ceiling).  Mechanism: the v20 merge
    # changed the swing-foot xpos reference frame (post-process
    # collision-primitive geometry shifted the foot-body x extent),
    # so the same x_cp - x_swing computation now produces the
    # opposite sense relative to the body's intended forward dir.
    # Sign flip is the empirical fix; mechanism diagnosis deferred.
    cp_gain: float = -0.7        # dimensionless; nudge = gain * (x_cp - x_swing)
    cp_clip: float = 0.03        # HARD clip per contract (m of swing-x)

    # Geometry constants (used only for capture-point and swing-x → hip-pitch
    # conversion; not for reading any reference annotation).
    #
    # com_height_m: was 0.473 pre-merge.  Reconciled to the live home
    # keyframe pelvis_z (0.4714) after the v20 ankle_roll merge tightened
    # waist spawn z 0.50 → 0.48 m.  ``parity_report.json`` reports 0.458
    # under the legacy 19-DOF prior fingerprint and is stale; this value
    # is the measured one off ``assets/v2/keyframes.xml`` `home`.
    com_height_m: float = 0.4714
    leg_length_m: float = 0.413  # upper + lower leg, for Δhip ≈ Δfoot_x / leg
    g: float = 9.81

    # Initialized from trajectory metadata at construction time.
    cycle_time_s: float = 0.64

    # Robot config providing actuator order + per-joint policy_action_sign.
    # Defaults to the same path used by ZMPWalkGenerator so the C2 harness
    # always resolves slots against the same source of truth as the
    # planner that produced the q_ref it consumes.
    robot_config_path: str = "assets/v2/mujoco_robot_config.json"


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

    # Joint names the harness writes to.  Resolved by name against
    # ``mujoco_robot_config.json`` so the harness adapts automatically
    # if the actuator list grows / shrinks.
    _CHANNEL_JOINT_NAMES = (
        "left_ankle_pitch", "right_ankle_pitch",   # pitch PD target
        "left_ankle_roll",  "right_ankle_roll",    # roll PD target (post-merge)
        "left_hip_pitch",   "right_hip_pitch",     # CP nudge target
    )

    def __init__(self, model, cfg: C2StabilizerConfig | None = None) -> None:
        import mujoco  # local import to avoid a hard dep at module load

        self.cfg = cfg or C2StabilizerConfig()
        self.model = model

        # Resolve actuator slots + per-joint policy_action_sign from
        # the robot config.  Failing here means the harness was paired
        # with a model that doesn't expose the required leg joints —
        # that's a contract break, not a tuning error.
        config_path = Path(self.cfg.robot_config_path)
        with open(config_path) as f:
            spec = json.load(f)
        actuator_specs = spec["actuated_joint_specs"]
        actuator_names = [j["name"] for j in actuator_specs]
        missing = [n for n in self._CHANNEL_JOINT_NAMES if n not in actuator_names]
        if missing:
            raise RuntimeError(
                f"C2 stabilizer requires {self._CHANNEL_JOINT_NAMES}; "
                f"missing from {config_path}: {missing}"
            )
        self._n_actuators = len(actuator_names)
        self._slot: Dict[str, int] = {
            n: actuator_names.index(n) for n in self._CHANNEL_JOINT_NAMES
        }
        # policy_action_sign is +1 / -1 per joint and encodes the L/R
        # axis mirror in the MJCF.  Multiplying the unified per-channel
        # offset by this sign produces a physically consistent
        # correction across L/R without per-side hard-coded sign flips.
        self._sign: Dict[str, float] = {
            n: float(actuator_specs[self._slot[n]]["policy_action_sign"])
            for n in self._CHANNEL_JOINT_NAMES
        }

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

        # Per-channel clip counters AND a per-step "any clipped" mask.
        # The doc contract reports "any harness output pinned at its
        # clip" as the union of channel-clipped step indices, NOT
        # max(channel_counts)/n.  Track both so per-channel and
        # aggregate metrics are correct independently.
        self._n_steps = 0
        self._clipped_pitch = 0
        self._clipped_roll = 0
        self._clipped_cp = 0
        self._clipped_any = 0

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
        self._clipped_any = 0

    def step(self, model, data, q_ref: np.ndarray,
             t_now: float) -> Tuple[np.ndarray, Dict]:
        """Compute and apply the bounded stabilizer offsets.

        Returns ``(q_modified, info)``.  ``q_modified`` is a fresh
        ndarray (length = ``model.nu``); the original ``q_ref`` is
        not mutated.  ``info`` carries per-step quantities used by
        the closeout artifact (clip flags, raw + clipped offsets,
        contact state).
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

        # Per-step "any channel clipped" flag — accumulated across
        # the three channels below so the aggregate metric is the
        # union of step indices, not the max per-channel count.
        any_clipped_this_step = False

        # --- 4. Torso pitch PD → ankle_pitch offset on stance side ---
        pitch_raw = cfg.apply_pitch_sign * \
            -(cfg.pitch_kp * pitch + cfg.pitch_kd * pitch_rate)
        pitch_off = float(np.clip(pitch_raw, -cfg.pitch_clip, +cfg.pitch_clip))
        if pitch_off != pitch_raw:
            self._clipped_pitch += 1
            any_clipped_this_step = True
        if l_in:
            slot = self._slot["left_ankle_pitch"]
            q_mod[slot] = q_mod[slot] + pitch_off * self._sign["left_ankle_pitch"]
        if r_in:
            slot = self._slot["right_ankle_pitch"]
            q_mod[slot] = q_mod[slot] + pitch_off * self._sign["right_ankle_pitch"]

        # --- 5. Torso roll PD → ankle_roll offset on stance side ---
        # Post-merge target (was hip_roll pre-merge): the v20 merge added
        # a dedicated ankle_roll DOF.  Per the Q1 design call, route the
        # roll PD through the local foot-flat actuator now that one
        # exists.  L/R sign mirror is read from policy_action_sign in
        # the robot config (left_ankle_roll = +1, right_ankle_roll = -1
        # for the v2 model — opposite of hip_roll's convention).
        roll_raw = cfg.apply_roll_sign * \
            -(cfg.roll_kp * roll + cfg.roll_kd * roll_rate)
        roll_off = float(np.clip(roll_raw, -cfg.roll_clip, +cfg.roll_clip))
        if roll_off != roll_raw:
            self._clipped_roll += 1
            any_clipped_this_step = True
        if l_in:
            slot = self._slot["left_ankle_roll"]
            q_mod[slot] = q_mod[slot] + roll_off * self._sign["left_ankle_roll"]
        if r_in:
            slot = self._slot["right_ankle_roll"]
            q_mod[slot] = q_mod[slot] + roll_off * self._sign["right_ankle_roll"]

        # --- 6. Capture-point swing nudge (snapshot at foot lift) ---
        cp_clipped_this_step = False
        for side, is_in, foot_id, hip_name in (
            (0, l_in, self._left_foot_id,  "left_hip_pitch"),
            (1, r_in, self._right_foot_id, "right_hip_pitch"),
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
            # L/R sign mirror is read from policy_action_sign so the
            # forward/backward foot-x convention is consistent without
            # a hard-coded per-side flip.
            if self._swing_active[side]:
                d_hip = self._held_swing_offset[side] / cfg.leg_length_m
                slot = self._slot[hip_name]
                q_mod[slot] = q_mod[slot] + d_hip * self._sign[hip_name]

        if cp_clipped_this_step:
            self._clipped_cp += 1
            any_clipped_this_step = True

        if any_clipped_this_step:
            self._clipped_any += 1

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
        """Per-channel fraction of steps where the channel was clipped,
        plus the aggregate ``any`` = fraction of steps where at least
        one of the three channels clipped (the contract metric in
        reference_design.md, NOT max(channel_counts) / n)."""
        n = max(1, self._n_steps)
        return {
            "pitch": self._clipped_pitch / n,
            "roll":  self._clipped_roll / n,
            "cp":    self._clipped_cp / n,
            "any":   self._clipped_any / n,
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
