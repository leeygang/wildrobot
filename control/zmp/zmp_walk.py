"""ZMP-based walking trajectory generator for WildRobot v2.

Generates offline walking reference trajectories by:
  1. Planning footsteps from velocity command
  2. Computing optimal COM trajectory via ZMP preview control
  3. Generating swing-foot trajectories (triangle lift)
  4. Solving IK for WildRobot's 4-DOF legs
  5. Packaging as a ``ReferenceTrajectory`` for the offline library

Reference: ToddlerBot ``toddlerbot.algorithms.zmp_walk``
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from control.references.reference_library import (
    ReferenceLibrary,
    ReferenceLibraryMeta,
    ReferenceTrajectory,
)
from control.zmp.zmp_planner import ZMPPlanner


@dataclass
class ZMPWalkConfig:
    """Configuration for ZMP walking trajectory generation."""

    # Robot morphology — calibrated against MJCF FK at commit 50d0993
    # (round-7 geometry refit).  The previous values (0.21 / 0.21 /
    # 0.06) were nominal CAD measurements but did not match the
    # actual MJCF joint placements:
    #   - the MJCF "waist" body's hip joint sits 0.034 m BELOW the
    #     pelvis frame origin (the IK's pelvis_x reference)
    #   - hip-to-knee distance is 0.193 m (not 0.21)
    #   - knee-to-ankle distance is 0.180 m (not 0.21)
    #   - foot collision boxes sit 0.061 m below the foot body frame
    # Lumping the pelvis-to-hip vertical (0.034 m) into ankle_to_ground
    # keeps the 2-link IK valid while making the flat-foot constraint
    # land q_ref's stance feet on the floor (FK gate test).  Without
    # this fix the prior placed stance feet ~14 mm above the floor at
    # nominal stance frames, which caused the body to drift backward
    # in physics across all C2 closeout sweeps (rounds 1-6).
    upper_leg_m: float = 0.193
    lower_leg_m: float = 0.180
    hip_lateral_offset_m: float = 0.0536
    ankle_to_ground_m: float = 0.061   # foot body origin to foot collision-bottom (from MJCF FK)
    pelvis_to_hip_m: float = 0.034     # pelvis frame origin to hip joint, vertical (round-7 refit)

    # Joint limits (radians) — used to compute safe COM height and step length
    ankle_dorsiflexion_limit_rad: float = 0.698   # 40°
    hip_extension_limit_rad: float = 0.5236        # 30° backward extension
    knee_pitch_max_rad: float = 2.094             # 120° (was 80° pre-2026-04-25)

    # Gait timing — longer cycle gives bigger steps at same speed.
    # Phase 6 (2026-04-26): adopted ToddlerBot's 0.72 s cycle (was 0.64 s)
    # to close the load-bearing size-normalised parity gaps the
    # architecture phases left in place:
    #   - cadence_froude_norm: 0.666 → 0.591 (TB 0.472; gate 1.20·TB = 0.566)
    #   - step_length_per_leg: 0.128 → 0.145 (TB 0.256; gate 0.85·TB = 0.218)
    # cycle_time directly drives both: a longer cycle produces
    # proportionally longer steps at the same vx (``half_cycle = cycle/2``
    # in ``_generate_at_step_length``) and proportionally lower cadence
    # (``touchdown_rate ≈ 2/cycle``).  This is local scalar tuning, not
    # an architectural change — the planner pipeline, swing-z envelope,
    # IK chain, and asset/reward semantics chosen in Phases 1-4 are
    # unchanged.  Per ``training/docs/reference_architecture_comparison.md``
    # § "Architectural alignment moves toward ToddlerBot": adopting TB's
    # value is the explicit Item 1 recommendation.
    # Phase 9D-revisited (2026-05-09): scale cycle_time to WR's leg
    # pendulum frequency instead of inheriting TB's 0.72 s.  The Phase 9A
    # cadence_norm FAIL at 1.252× TB was a direct consequence of forcing
    # WR's longer leg to step at TB's clock rate.  Froude-similar scaling:
    #   cycle_time_WR = cycle_time_TB × √(L_WR / L_TB)
    #                 = 0.72 × √(0.373 / 0.2115)
    #                 ≈ 0.96 s.
    # Companion operating-point shift: vx 0.265 → 0.20 m/s preserves
    # step_length_per_leg ≈ 0.256 (TB's value) under the longer cycle.
    # Empirically closes 2 of 3 normalised P1A FAILs (cadence_norm
    # 1.252× → 0.934×; swing_step_per_clr 1.185× → 0.825×) and brings
    # absolute swing_z_step_max from 1.59× → 1.109× TB.  See CHANGELOG
    # entry `v0.20.1-phase9D-cycle-time-scaling`.
    cycle_time_s: float = 0.96
    single_double_ratio: float = 2.0
    dt_s: float = 0.02

    # Gait geometry.  foot_step_height was 0.08 before round 7
    # (~19 % of the OLD 0.42 m leg length).  After the geometry
    # refit (real leg length = 0.193 + 0.180 = 0.373 m), the same
    # 0.08 m peak lift required the knee to bend 92° at swing
    # apex — over the (then) 80° knee limit, triggering knee
    # saturation in fixed-base / C1 / C2.  Reduced to 0.05 m
    # (~13 % of the actual leg length) which kept the swing-peak
    # knee inside the (old) limit.  ToddlerBot uses 24 % of leg
    # length and has a higher knee limit.
    #
    # 2026-04-25: knee_pitch_max_rad raised from 80° → 120°.  The
    # previous 80°-saturation rationale no longer binds.
    #
    # Phase 8 attempt-1 (2026-05-05): tried bumping 0.04 → 0.05 and
    # 0.04 → 0.045 to close the `swing_clearance_per_com_height`
    # normalised P1A gate (0.143 → 0.158/0.175 vs gate 0.149).
    # Both regressed survival at vx=0.15 even though the saturation
    # gate stayed clean: 200→74 steps at h=0.05 and 200→104 at
    # h=0.045 (pitch termination from the deeper-bend swing
    # destabilising stance, not the saturation mechanism the
    # original Phase 8 warning anticipated).
    # Reverted; gated on Phase 12A.
    #
    # Phase 8 attempt-2 / Phase 12A pair (2026-05-08): with the
    # smoothstepped plantarflex schedule (`plantarflex_ramp_band_m
    # = 0.005`, see below), the deeper-bend swing no longer
    # destabilises stance.  Survival at vx=0.15 recovers from
    # 200→104 (Phase 8 alone) back to 200/200, and the
    # `swing_clearance_per_com_height` gate closes (0.78× TB →
    # ~0.91× TB at h=0.045).  Net cost: small vx=0.10 left_hip_roll
    # uptick (0.091 → 0.273), out of the in-scope load-bearing band.
    foot_step_height_m: float = 0.045
    default_stance_width_m: float = 0.0536  # = hip_lateral_offset_m
    min_walking_speed_mps: float = 0.06     # below this, use standing

    # Phase 12A (2026-05-08): soften the plantarflex up-cross /
    # down-cross transition.  ankle_pitch is now blended between the
    # flat-foot constraint (-(hip+knee)) and the swing-neutral
    # plantarflex (-0.15 rad) via a smoothstep ramp keyed off the
    # commanded foot z.  ``plantarflex_ramp_band_m`` is the half-width
    # of the smoothstep band centred on
    # ``swing_foot_z_floor_clearance_m`` (0.025 m).
    #
    # Default 0.005 m spans z=0.020 to z=0.030 — narrow enough that
    # the boundary frames (z<0.020 at lift-off / touchdown) stay
    # exactly flat-foot, but the actual up-cross / down-cross
    # transition is now a 1-2 frame ramp instead of a 1-frame step
    # at h=0.04 (up_delta=0.013 m/frame).  The wider 0.020 m band
    # tried first regressed survival at vx=0.15 (200→92 pitch term)
    # because it bled plantarflex into the early-swing boundary
    # frames where the body needs the flat-foot constraint to keep
    # the stance side balanced; the 0.005 m band preserves that
    # boundary behaviour while still smoothing the actual transition.
    # The pre-Phase-12A binary behaviour is recovered by setting this
    # to 0.
    plantarflex_ramp_band_m: float = 0.005

    # ZMP planner costs
    zmp_cost_Q: float = 1.0
    zmp_cost_R: float = 0.1

    # Trajectory generation: plan long horizon and replay linearly
    # (matches ToddlerBot ``zmp_walk``: 22 s of monotonic forward
    # walking).  Under the round-9 LQR path the COM trajectory is
    # built by forward-integrating the closed-loop preview-control
    # feedback from a from-rest initial condition; foot placement
    # is uniform with cycle 0's right swing as a half-step (matches
    # ToddlerBot's footstep schedule).  No quintic warm-up is used
    # (the LQR's preview generates a non-zero acceleration at t=0
    # by itself; the previous ``n_warmup_cycles`` knob is dead under
    # this path and was removed).
    total_plan_time_s: float = 22.0

    # Limits
    max_step_length_m: float = 0.10
    min_reach_margin_m: float = 0.01

    @property
    def com_height_m(self) -> float:
        """Compute pelvis (= COM) height at safe walking knee bend.

        ``com_height = pelvis_to_hip + leg_reach(knee_safe) + ankle_to_ground``

        Where ``leg_reach`` is the hip-to-foot-body distance at the
        chosen knee bend (2-link chain).  Round-7 refit added the
        ``pelvis_to_hip_m`` term explicitly because the MJCF "waist"
        body's hip joint sits ~34 mm below the pelvis frame origin
        (the reference point the IK uses for ``pelvis_x``); without
        it, the IK underestimates body height for stance and
        OVERestimates it for swing, which the FK gate
        (``tests/test_v0200c_geometry.py``) catches.
        """
        hip_margin_rad = np.radians(20.0)
        knee_budget = self.ankle_dorsiflexion_limit_rad - hip_margin_rad
        knee_safe = max(np.radians(10.0), min(knee_budget, self.knee_pitch_max_rad))
        l1, l2 = self.upper_leg_m, self.lower_leg_m
        reach = np.sqrt(l1**2 + l2**2 + 2 * l1 * l2 * np.cos(knee_safe))
        # The IK solver clamps target distance to (l1 + l2 - min_margin).
        # If our requested ``reach`` would exceed that cap, the IK will
        # bend the knee deeper than ``knee_safe`` and deliver only
        # ``max_reach`` of leg drop.  Reflect that here so the
        # downstream ``traj.pelvis_pos[i, 2]`` and IK target produce
        # consistent geometry — without this, the FK foot bottom
        # sits ~``min_reach_margin`` above the floor at every stance
        # frame (round-7 retro).
        max_reach = l1 + l2 - self.min_reach_margin_m
        effective_reach = min(reach, max_reach)
        return effective_reach + self.ankle_to_ground_m + self.pelvis_to_hip_m

    @property
    def safe_max_step_length_m(self) -> float:
        """Max step length that keeps all joints within limits.

        Two constraints:
        1. Ankle: hip_pitch + knee ≤ ankle_dorsiflexion_limit
        2. Hip extension: foot behind COM → hip extends backward ≤ hip_extension_limit

        The tighter constraint wins.
        """
        hip_to_ankle = self.com_height_m - self.ankle_to_ground_m - self.pelvis_to_hip_m

        # Constraint 1: ankle budget
        l1, l2 = self.upper_leg_m, self.lower_leg_m
        cos_knee = (l1**2 + l2**2 - hip_to_ankle**2) / (2 * l1 * l2)
        cos_knee = np.clip(cos_knee, -1.0, 1.0)
        knee_at_height = np.pi - np.arccos(cos_knee)
        max_hip_ankle = self.ankle_dorsiflexion_limit_rad - knee_at_height
        max_hip_ankle = max(0.01, max_hip_ankle)
        half_step_ankle = hip_to_ankle * np.tan(max_hip_ankle)

        # Constraint 2: hip extension limit (5° backward)
        half_step_hip = hip_to_ankle * np.tan(self.hip_extension_limit_rad)

        # Take the tighter constraint.
        # The foot-to-COM distance during stance exceeds half the step
        # length because the LIPM COM accelerates during stance.
        # Use a 0.7 factor (empirically, max foot-behind-COM ≈ 0.7 × step).
        max_foot_excursion = min(half_step_ankle, half_step_hip)
        safe_step = max_foot_excursion / 0.7
        return min(self.max_step_length_m, safe_step)

    @property
    def double_support_s(self) -> float:
        return self.cycle_time_s / 2.0 / (self.single_double_ratio + 1.0)

    @property
    def single_support_s(self) -> float:
        return self.single_double_ratio * self.double_support_s

    @property
    def steps_per_cycle(self) -> int:
        return int(round(self.cycle_time_s / self.dt_s))


_SWING_PLANTARFLEX_RAD = -0.15  # neutral plantarflex held during full swing


def _smoothstep(x: float) -> float:
    """Cubic Hermite smoothstep: 0 at x≤0, 1 at x≥1, smooth in between."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return float(x * x * (3.0 - 2.0 * x))


def _solve_sagittal_ik(
    target_x: float,
    target_z: float,
    l1: float,
    l2: float,
    min_margin: float = 0.01,
    ankle_limit_rad: float = 0.698,
    plantarflex_strength: float = 0.0,
) -> Tuple[float, float, float, bool]:
    """Analytical 2-link sagittal IK.

    Returns (hip_pitch, knee_pitch, ankle_pitch, reachable).

    The ankle target is a smooth blend between two limits:
      * **flat-foot**: ``ankle_pitch = -(hip_pitch + knee_pitch)`` — keeps
        the foot parallel to the ground.  Used during stance and at
        boundary frames of swing (lift-off + touchdown) where the
        commanded foot z is too small to absorb the plantarflex drop.
      * **swing-neutral**: ``ankle_pitch = _SWING_PLANTARFLEX_RAD``
        (-0.15 rad) — small plantarflex for toe clearance during the
        airborne portion of swing, freeing the knee budget for foot
        lift.

    Caller passes ``plantarflex_strength ∈ [0, 1]`` to specify the
    blend weight (0 = flat-foot, 1 = swing-neutral).  Phase 12A
    (2026-05-08) replaced the previous binary ``is_swing`` flag with
    this continuous parameter so the ankle target ramps smoothly
    across the swing-z up-cross / down-cross instead of stepping
    abruptly in one frame.  The binary version produced a
    discrete ~0.15 rad ankle jump synchronous with the discrete
    contact-mask flip, which the Phase 8 experiment (2026-05-05)
    surfaced as the load-bearing destabilising mechanism for the
    deeper-bend swing at h=0.045/0.05.
    """
    max_reach = l1 + l2 - min_margin
    dist = np.sqrt(target_x**2 + target_z**2)
    reachable = dist <= max_reach

    if dist > max_reach:
        scale = max_reach / max(dist, 1e-8)
        target_x *= scale
        target_z *= scale
        dist = max_reach

    if dist < abs(l1 - l2) + min_margin:
        dist = abs(l1 - l2) + min_margin

    cos_knee = (l1**2 + l2**2 - dist**2) / (2.0 * l1 * l2)
    cos_knee = np.clip(cos_knee, -1.0, 1.0)
    knee_pitch = np.pi - np.arccos(cos_knee)

    alpha = np.arctan2(-target_x, -target_z)
    cos_beta = (l1**2 + dist**2 - l2**2) / (2.0 * l1 * dist)
    cos_beta = np.clip(cos_beta, -1.0, 1.0)
    beta = np.arccos(cos_beta)
    hip_pitch = alpha - beta

    p = float(np.clip(plantarflex_strength, 0.0, 1.0))
    ankle_flat = -(hip_pitch + knee_pitch)
    ankle_blended = (1.0 - p) * ankle_flat + p * _SWING_PLANTARFLEX_RAD
    ankle_pitch = np.clip(ankle_blended, -ankle_limit_rad, ankle_limit_rad)

    return hip_pitch, knee_pitch, ankle_pitch, reachable


class ZMPWalkGenerator:
    """Generate walking trajectories for WildRobot v2 using ZMP preview control."""

    # Foot collision-geom pairs used to synthesise foot-center sites
    # for the FK replay (WR's MJCF doesn't define explicit
    # ``left_foot_center`` / ``right_foot_center`` sites the way TB's
    # does, so we compute the centre as the midpoint of the
    # heel + toe collision boxes — same convention the parity tool
    # already uses, see ``tools/reference_geometry_parity.py::_wr_foot_center_pos``).
    _FOOT_CENTER_GEOMS = (
        ("left_foot_center", "left_heel", "left_toe"),
        ("right_foot_center", "right_heel", "right_toe"),
    )
    _DEFAULT_SCENE_XML = "assets/v2/scene_flat_terrain.xml"
    _DEFAULT_ROBOT_CONFIG = "assets/v2/mujoco_robot_config.json"

    # Sagittal IK + hip-roll IK fill these 8 leg joints in PolicySpec
    # actuator order; all other actuators (waist, arms, ankle_roll) keep
    # the default-zero q_ref slot.  ankle_roll was added by the v20
    # ankle_roll merge; the ZMP family does not plan a non-zero roll
    # target, so it remains at 0 and the policy's residual is the only
    # control signal on that joint.
    _LEG_JOINT_NAMES = (
        "left_hip_pitch", "left_hip_roll", "left_knee_pitch", "left_ankle_pitch",
        "right_hip_pitch", "right_hip_roll", "right_knee_pitch", "right_ankle_pitch",
    )

    def __init__(self, config: ZMPWalkConfig | None = None) -> None:
        self.cfg = config or ZMPWalkConfig()
        self.planner = ZMPPlanner()
        # FK-replay assets are loaded lazily and cached.  Avoids paying
        # the MJCF parse cost when ``generate(0.0)`` is called repeatedly
        # for standing references.
        self._fk_assets: Optional[dict] = None
        self._actuator_layout: Optional[dict] = None

    # ------------------------------------------------------------------
    # Phase 3: TB-style realized-reference enrichment
    # ------------------------------------------------------------------
    # Mirrors ``ZMPWalk.mujoco_replay`` from
    # ``toddlerbot/algorithms/zmp_walk.py:153-212``: after the IK
    # produces ``q_ref``, run fixed-base FK frame-by-frame to harvest
    # realized body / site quantities the parity tool, RSI init, and
    # debugging consumers can read directly without re-FKing.

    def _load_actuator_layout(self) -> dict:
        """Resolve PolicySpec actuator order + leg-joint slots from the robot config.

        Returns a dict with:
          * ``n_joints`` — total actuator count (= ``q_ref.shape[1]``)
          * ``actuator_names`` — full list in PolicySpec order
          * ``leg_idx`` — name -> slot index for the 8 leg joints
          * ``leg_clip_min`` / ``leg_clip_max`` — length-``n_joints``
            arrays with leg slots set to the joint's MJCF range and all
            other slots at ``±inf`` (so a single ``np.clip`` over the
            full ``q_ref`` only constrains the IK-driven leg slots).

        Joint ranges are read from ``mujoco_robot_config.json`` (degrees)
        and converted to radians.  Cached after first call.
        """
        if self._actuator_layout is not None:
            return self._actuator_layout

        config_path = Path(self._DEFAULT_ROBOT_CONFIG)
        with open(config_path) as f:
            spec = json.load(f)
        actuator_specs = spec["actuated_joint_specs"]
        actuator_names: List[str] = [j["name"] for j in actuator_specs]

        missing = [n for n in self._LEG_JOINT_NAMES if n not in actuator_names]
        if missing:
            raise RuntimeError(
                f"mujoco_robot_config.json missing leg joints required by ZMP IK: {missing}"
            )
        leg_idx = {n: actuator_names.index(n) for n in self._LEG_JOINT_NAMES}

        n_joints = len(actuator_names)
        leg_clip_min = np.full(n_joints, -np.inf, dtype=np.float32)
        leg_clip_max = np.full(n_joints, np.inf, dtype=np.float32)
        for joint_name in self._LEG_JOINT_NAMES:
            slot = leg_idx[joint_name]
            rng_deg = actuator_specs[slot]["range"]
            leg_clip_min[slot] = float(np.deg2rad(rng_deg[0]))
            leg_clip_max[slot] = float(np.deg2rad(rng_deg[1]))

        self._actuator_layout = {
            "actuator_names": actuator_names,
            "n_joints": n_joints,
            "leg_idx": leg_idx,
            "leg_clip_min": leg_clip_min,
            "leg_clip_max": leg_clip_max,
        }
        return self._actuator_layout

    def _load_fk_assets(self) -> dict:
        """Lazily load and cache the MJCF + ctrl mapping for FK replay."""
        if self._fk_assets is not None:
            return self._fk_assets
        # Local imports to keep the module's top-level deps minimal for
        # callers that never trigger FK replay (e.g. unit tests that
        # construct a stub trajectory).
        import mujoco  # noqa: WPS433  -- pulled in only when FK is needed
        from training.utils.ctrl_order import CtrlOrderMapper  # noqa: WPS433

        scene = Path(self._DEFAULT_SCENE_XML)
        config_path = Path(self._DEFAULT_ROBOT_CONFIG)
        model = mujoco.MjModel.from_xml_path(str(scene))
        data = mujoco.MjData(model)
        with open(config_path) as f:
            spec = json.load(f)
        actuator_names = [j["name"] for j in spec["actuated_joint_specs"]]
        mapper = CtrlOrderMapper(model, actuator_names)
        act_to_qpos = np.array(
            [model.jnt_qposadr[model.actuator_trnid[k, 0]] for k in range(model.nu)]
        )
        # Anchor at the home keyframe so the body stays at its nominal
        # pose (TB-style fixed-base FK semantics — matches
        # ``MuJoCoSim(robot, fixed_base=True)`` in TB's mujoco_replay).
        home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if home_id < 0:
            raise RuntimeError("WR scene_flat_terrain.xml is missing the 'home' keyframe")
        mujoco.mj_resetDataKeyframe(model, data, home_id)
        home_qpos = data.qpos.copy()
        body_names = tuple(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or f"body_{b}"
            for b in range(model.nbody)
        )
        # Foot-center "sites" synthesised from heel/toe geom midpoints.
        site_names = tuple(name for name, _, _ in self._FOOT_CENTER_GEOMS)
        site_geom_pairs = []
        for _name, geom_a, geom_b in self._FOOT_CENTER_GEOMS:
            ga = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_a)
            gb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_b)
            if ga < 0 or gb < 0:
                raise RuntimeError(
                    f"Missing foot-center geom pair: {geom_a} / {geom_b}"
                )
            site_geom_pairs.append((ga, gb))
        self._fk_assets = {
            "model": model,
            "data": data,
            "mapper": mapper,
            "act_to_qpos": act_to_qpos,
            "home_qpos": home_qpos,
            "body_names": body_names,
            "site_names": site_names,
            "site_geom_pairs": site_geom_pairs,
        }
        return self._fk_assets

    def _fixed_base_fk_replay(
        self, q_ref: np.ndarray, dt: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple[str, ...], Tuple[str, ...]]:
        """Replay ``q_ref`` frame-by-frame in fixed-base FK.

        Returns ``(body_pos, body_quat, body_lin_vel, body_ang_vel,
        site_pos, body_names, site_names)``.

        Body position / orientation come straight from MuJoCo
        (``data.xpos`` / ``data.xquat``).  Velocities are finite-diff
        of the position arrays (G2 finite-diff convention).  Angular
        velocity is computed from a quaternion delta; the small-angle
        approximation ``ang_vel = 2 · (q1·conj(q0))[xyz] / dt`` is
        used because per-step rotations are <<1 rad at ``dt=0.02``.
        """
        import mujoco  # noqa: WPS433

        assets = self._load_fk_assets()
        model = assets["model"]
        data = assets["data"]
        mapper = assets["mapper"]
        act_to_qpos = assets["act_to_qpos"]
        home_qpos = assets["home_qpos"]
        body_names = assets["body_names"]
        site_names = assets["site_names"]
        site_geom_pairs = assets["site_geom_pairs"]

        n_steps = int(q_ref.shape[0])
        n_bodies = int(model.nbody)
        n_sites = len(site_names)
        body_pos = np.zeros((n_steps, n_bodies, 3), dtype=np.float32)
        body_quat = np.zeros((n_steps, n_bodies, 4), dtype=np.float32)
        site_pos = np.zeros((n_steps, n_sites, 3), dtype=np.float32)

        policy_to_mj = mapper.policy_to_mj_order
        for i in range(n_steps):
            data.qpos[:] = home_qpos
            mj_q = np.zeros(len(policy_to_mj), dtype=np.float64)
            mj_q[policy_to_mj] = q_ref[i]
            for ai in range(model.nu):
                data.qpos[act_to_qpos[ai]] = mj_q[ai]
            mujoco.mj_forward(model, data)
            body_pos[i] = data.xpos.astype(np.float32)
            body_quat[i] = data.xquat.astype(np.float32)
            for s_idx, (ga, gb) in enumerate(site_geom_pairs):
                site_pos[i, s_idx] = (
                    0.5 * (data.geom_xpos[ga] + data.geom_xpos[gb])
                ).astype(np.float32)

        body_lin_vel = self._fdiff(body_pos, dt)
        body_ang_vel = self._quat_finite_diff_ang_vel(body_quat, dt)
        return (
            body_pos, body_quat, body_lin_vel, body_ang_vel, site_pos,
            body_names, site_names,
        )

    @staticmethod
    def _fdiff(arr: np.ndarray, dt: float) -> np.ndarray:
        out = np.zeros_like(arr)
        if arr.shape[0] >= 2 and dt > 0.0:
            out[:-1] = (arr[1:] - arr[:-1]) / dt
            out[-1] = out[-2]
        return out

    @staticmethod
    def _quat_finite_diff_ang_vel(q_wxyz: np.ndarray, dt: float) -> np.ndarray:
        """Per-body angular velocity from quaternion finite-diff.

        Uses ``ang_vel = 2 · (q1 · conj(q0))[xyz] / dt`` (small-angle
        approximation valid for the per-step rotations at ``dt=0.02``
        in this planner).  Sign-aligns ``q1`` against ``q0`` to handle
        the double-cover ambiguity before the delta multiplication.
        """
        n_steps = q_wxyz.shape[0]
        ang_vel = np.zeros((n_steps,) + q_wxyz.shape[1:-1] + (3,), dtype=np.float32)
        if n_steps < 2 or dt <= 0.0:
            return ang_vel
        q0 = q_wxyz[:-1]
        q1 = q_wxyz[1:]
        # Double-cover sign alignment per body.
        dot = np.sum(q0 * q1, axis=-1, keepdims=True)
        q1 = np.where(dot < 0.0, -q1, q1)
        # delta = q1 · conj(q0); wxyz convention.
        w0, x0, y0, z0 = q0[..., 0], q0[..., 1], q0[..., 2], q0[..., 3]
        w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        # conj(q0) = [w0, -x0, -y0, -z0]
        dx = w1 * (-x0) + x1 * w0 + y1 * (-z0) - z1 * (-y0)
        dy = w1 * (-y0) - x1 * (-z0) + y1 * w0 + z1 * (-x0)
        dz = w1 * (-z0) + x1 * (-y0) - y1 * (-x0) + z1 * w0
        # Small-angle: rotation vector ≈ 2 · [dx, dy, dz] / dt.
        ang = 2.0 * np.stack([dx, dy, dz], axis=-1) / dt
        ang_vel[:-1] = ang.astype(np.float32)
        ang_vel[-1] = ang_vel[-2]
        return ang_vel

    def generate(self, command_vx: float) -> ReferenceTrajectory:
        """Generate one gait-cycle trajectory for the given forward speed.

        At low speeds, the cycle time is lengthened to maintain a minimum
        step length (normal-sized steps at lower cadence, not tiny shuffles
        at normal cadence).  Joint limits are enforced via a generate-
        validate-reduce loop.

        Parameters
        ----------
        command_vx : forward speed in m/s (0 = standing)

        Returns
        -------
        ReferenceTrajectory with all arrays populated and within joint limits.
        """
        cfg = self.cfg

        if abs(command_vx) < 1e-4:
            return self._generate_standing()

        if abs(command_vx) < cfg.min_walking_speed_mps:
            return self._generate_standing()

        # Compute step length at the normal cadence
        abs_vx = abs(command_vx)
        half_cycle = cfg.cycle_time_s / 2.0
        step_length = min(abs_vx * half_cycle, cfg.safe_max_step_length_m)
        cycle_time = cfg.cycle_time_s

        if command_vx < 0:
            step_length = -step_length

        # Generate-validate-reduce loop
        for attempt in range(5):
            traj = self._generate_at_step_length(
                command_vx, step_length, cycle_time_override=cycle_time)
            violations = self._count_joint_violations(traj)
            if violations == 0:
                break
            step_length *= 0.8

        return traj

    def _count_joint_violations(self, traj: ReferenceTrajectory) -> int:
        """Count (joint, timestep) pairs where any leg joint exceeds its limit.

        Uses the name-resolved leg-slot bounds from
        ``_load_actuator_layout`` (length-``n_joints`` arrays where
        non-leg columns are ``±inf`` and so never trigger), which means
        this check now covers the full {hip_pitch, hip_roll, knee,
        ankle_pitch} × L/R set in PolicySpec actuator order even though
        legs sit at non-contiguous indices post the v20 ankle_roll
        merge.  Pre-merge the loop hard-coded columns 0..7 and silently
        stopped checking the right leg once the actuator layout
        re-ordered.
        """
        layout = self._load_actuator_layout()
        lo = layout["leg_clip_min"] - 0.005   # length-n_joints
        hi = layout["leg_clip_max"] + 0.005
        violations = (traj.q_ref < lo) | (traj.q_ref > hi)
        return int(np.sum(violations))

    @staticmethod
    def _quintic_coeffs(T: float, p0: float, v0: float, a0: float,
                        pT: float, vT: float, aT: float) -> np.ndarray:
        """Quintic polynomial through ``(p0, v0, a0)`` at t=0 and
        ``(pT, vT, aT)`` at t=T.

        Returns ``[a0, a1, a2, a3, a4, a5]`` for
        ``p(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5``.
        Used for the cycle-0 from-rest blend in
        ``_generate_at_step_length``.
        """
        a0_c = p0
        a1_c = v0
        a2_c = a0 / 2.0
        # Solve the 3x3 system for (a3, a4, a5).
        T2 = T * T
        T3 = T2 * T
        T4 = T3 * T
        T5 = T4 * T
        M = np.array([
            [T3,        T4,         T5],
            [3 * T2,    4 * T3,     5 * T4],
            [6 * T,     12 * T2,    20 * T3],
        ], dtype=np.float64)
        rhs = np.array([
            pT - (a0_c + a1_c * T + a2_c * T2),
            vT - (a1_c + 2 * a2_c * T),
            aT - (2 * a2_c),
        ], dtype=np.float64)
        a3_c, a4_c, a5_c = np.linalg.solve(M, rhs)
        return np.array([a0_c, a1_c, a2_c, a3_c, a4_c, a5_c],
                        dtype=np.float64)

    @staticmethod
    def _quintic_eval(coeffs: np.ndarray, t: float) -> float:
        a0_c, a1_c, a2_c, a3_c, a4_c, a5_c = coeffs
        t2 = t * t
        return float(a0_c + a1_c * t + a2_c * t2
                     + a3_c * t2 * t + a4_c * t2 * t2 + a5_c * t2 * t2 * t)

    def _generate_at_step_length(
        self, command_vx: float, step_length: float,
        cycle_time_override: float | None = None,
    ) -> ReferenceTrajectory:
        """Generate a multi-cycle monotonic walking trajectory.

        Matches ToddlerBot's approach (``toddlerbot.algorithms.zmp_walk``)
        more directly as of round 9: the **sagittal COM** is computed
        by the full Kajita preview-control LQR
        (``control.zmp.zmp_planner.ZMPPlanner``) with a from-rest
        initial condition.  The LQR's preview of the entire ZMP
        schedule generates a non-zero acceleration at t=0 even with
        the COM at rest, eliminating the from-rest startup transient
        problem that the analytical-LIPM + cycle-0-quintic approach
        could not solve (rounds 7-8 retro: vx=0.15 still fell forward
        in ~1 s).

        Both x and y COM come from the **forward-integrated** LQR:
        the LIPM state is propagated by Euler at ``dt`` under the
        closed-loop feedback ``planner.get_optim_com_acc(t, x)``.
        This matches ToddlerBot's ``zmp_walk.update_step`` exactly.

        With the same Qy=I, R=0.1·I that ToddlerBot uses, the
        per-cycle lateral COM swing is small (sub-mm in cycle 0).
        Over the full 22-s plan, however, forward-Euler integration
        of the LQR feedback drifts laterally; on the v0.15 m/s
        trajectory we measure ``com_y`` in
        ``[-1.6 mm, +19.8 mm]`` end-to-end (round-9 reviewer).  The
        drift is small enough not to change foot placement (which
        is fixed up front, not LQR-driven), but it should not be
        described as "1 mm peak-to-peak" — earlier docs that did
        were based on a too-short standalone probe.

        Round-9b retro: the previous "0.25·lat cosine" override
        demanded a lateral body sway ToddlerBot's design does not
        require, and the original "LQR-y overshoots" justification
        was based on a probe done while a view-mutation bug in the
        planner port was active (commit ``5faae74`` fixed it).

        Foot positions advance monotonically:
          - Left feet at world x = 0, 2·sl, 4·sl, ...
          - Right feet at world x = 0 (cycle 0 start, from rest),
            then sl, 3·sl, 5·sl, ... after the first half-step.

        The cycle-0 right swing is a half-step (0 → sl) because the
        right foot has not yet placed; cycles 1+ are full 2·sl
        swings.  This matches ToddlerBot's footstep schedule.
        """
        cfg = self.cfg
        cycle_time = cycle_time_override or cfg.cycle_time_s

        ds = cycle_time / 2.0 / (cfg.single_double_ratio + 1.0)
        ss = cfg.single_double_ratio * ds
        T_half = ds + ss  # half-cycle time
        T_cycle = 2.0 * T_half
        sl = step_length
        lat = cfg.default_stance_width_m
        dt = cfg.dt_s

        # WR swing-foot floor-clearance compensation.
        # During swing the IK plantarflexes the ankle by 0.15 rad for
        # toe clearance, which drops the foot's TOE corner ~9 mm below
        # the foot body and the foot collision box's geometry adds
        # another ~9 mm of swing-side vertical extent on the rotated
        # toe (round-7 FK gate caught this without compensation).
        # The compensation rides the same envelope as the swing arc so
        # it vanishes at lift-off and touchdown (boundary continuity
        # with the adjacent stance frames); ``swing_ankle_lift_threshold_m``
        # then keeps the IK ankle flat at boundary frames where the
        # commanded z is too small to absorb the plantarflex drop.
        swing_foot_z_floor_clearance_m = 0.025
        swing_ankle_lift_threshold_m = swing_foot_z_floor_clearance_m

        n_half = int(round(T_half / dt))
        n_cycle = 2 * n_half
        n_ds = int(round(ds / dt))

        # Phase 7: TB-style single-peaked triangle swing-z envelope.
        # Mirrors ``toddlerbot/algorithms/zmp_walk.py:483-491`` over the
        # WR swing window (n_half - n_ds frames per phase).  Peak is
        # ``foot_step_height_m + swing_foot_z_floor_clearance_m`` so the
        # H1 hardware exemption stays explicit; the envelope hits exactly
        # zero at lift-off (i_swing=0) and touchdown (i_swing=swing_window-1)
        # so the Phase 1 plantarflex gating keeps boundary frames flat-ankle.
        swing_window = max(2, n_half - n_ds)
        swing_half = swing_window // 2
        up_delta = (
            (cfg.foot_step_height_m + swing_foot_z_floor_clearance_m)
            / max(1, swing_half - 1)
        )
        swing_z_table = up_delta * np.concatenate(
            (
                np.arange(swing_half, dtype=np.float64),
                np.arange(swing_window - swing_half - 1, -1, -1, dtype=np.float64),
            )
        )

        n_cycles = max(1, int(np.ceil(cfg.total_plan_time_s / cycle_time)))
        n_total = n_cycles * n_cycle

        # --- Build the ZMP schedule (matches ToddlerBot's pattern) ---
        # Footsteps alternate left/right starting at the origin:
        #   step 0: left  at (0,         +lat)
        #   step 1: right at (sl,        -lat)
        #   step 2: left  at (2·sl,      +lat)
        #   step 3: right at (3·sl,      -lat)
        # Phases alternate DS, SS, DS, SS ... starting with DS.  The
        # ZMP is at the previous stance foot during DS, transitions to
        # the next stance foot at SS start, and stays there during SS.
        # Following ToddlerBot we duplicate each footstep ZMP so the
        # LQR sees a piecewise-constant ZMP within each phase.
        n_steps = 2 * n_cycles + 1  # +1 for the trailing duplicate
        footsteps: list[np.ndarray] = []
        for i in range(n_steps):
            x = i * sl
            y = lat if (i % 2 == 0) else -lat
            footsteps.append(np.array([x, y], dtype=np.float64))

        # time_steps: [0, ds, ds+ss, 2ds+ss, ...] matching ToddlerBot
        time_list = [0.0, ds] + [ss, ds] * (len(footsteps) - 1)
        zmp_times = np.cumsum(time_list).astype(np.float64)
        zmp_times = zmp_times[:len(footsteps) * 2 - 1]
        # desired_zmps: each footstep duplicated (DS-onset + SS-onset)
        zmps_d = [step for step in footsteps for _ in range(2)]
        zmps_d = zmps_d[:len(zmp_times)]

        # --- Solve the LQR ---
        x0_lqr = np.array([0.0, 0.0, 0.0, 0.0])  # COM at rest
        self.planner.plan(
            zmp_times, zmps_d, x0_lqr, cfg.com_height_m,
            Qy=np.eye(2) * cfg.zmp_cost_Q,
            R=np.eye(2) * cfg.zmp_cost_R,
        )

        # --- Forward-integrate the closed-loop LQR (ToddlerBot style) ---
        # See ``toddlerbot/algorithms/zmp_walk.py`` ``update_step``: the
        # COM trajectory is the closed-loop simulation of LIPM under
        # the LQR's state feedback ``get_optim_com_acc(t, x)``, NOT
        # the analytical ``get_nominal_com(t)``.  Both methods would
        # be equivalent in continuous time, but at our 0.02 s control
        # dt the analytical solution depends on which segment ``t``
        # falls in (per-segment ``(a, b)`` parameters), while the
        # state-feedback path is robust to that segmenting because it
        # only queries ``k2(t)`` and ``K @ x``.  Use the same path.
        n_lqr = int(np.ceil((zmp_times[-1] - zmp_times[0]) / dt))
        com_lqr = np.zeros((n_lqr, 2), dtype=np.float64)
        x_state = x0_lqr.copy()
        u = self.planner.get_optim_com_acc(0.0, x_state)
        com_lqr[0] = x_state[:2]
        for j in range(1, n_lqr):
            t_j = j * dt
            xd = np.hstack([x_state[2:], u])
            x_state = x_state + xd * dt
            u = self.planner.get_optim_com_acc(t_j, x_state)
            com_lqr[j] = x_state[:2]

        com_world = np.zeros((n_total, 2), dtype=np.float64)
        left_world = np.zeros((n_total, 3), dtype=np.float64)
        right_world = np.zeros((n_total, 3), dtype=np.float64)
        contact_out = np.zeros((n_total, 2), dtype=np.float64)
        stance_out = np.zeros(n_total, dtype=np.float64)

        for k in range(n_cycles):
            cycle_offset = 2 * k * sl

            # Right's start position in this cycle's phase A.
            # Cycle 0: from rest (right at world x=0, half-step
            # swing 0→sl).  Cycle k≥1: steady state (right at
            # (2k-1)·sl from previous cycle's phase B, full 2·sl
            # swing).
            if k == 0:
                right_a_start = 0.0
                right_a_swing = sl
            else:
                right_a_start = (2 * k - 1) * sl
                right_a_swing = 2.0 * sl

            # --- Phase A: left stance (ZMP at world x=cycle_offset) ---
            for i in range(n_half):
                t = i * dt
                idx = k * n_cycle + i

                # Both x and y COM from the forward-integrated LQR
                # (ToddlerBot's design — small lateral COM swing is
                # correct; the lateral support transfer comes from
                # which foot is in stance, not from a large COM-y).
                t_global = k * T_cycle + t
                lqr_idx = min(int(round(t_global / dt)), n_lqr - 1)
                com_world[idx, 0] = com_lqr[lqr_idx, 0]
                com_world[idx, 1] = com_lqr[lqr_idx, 1]

                left_world[idx] = [cycle_offset, lat, 0]
                if i < n_ds:
                    right_world[idx] = [right_a_start, -lat, 0]
                    contact_out[idx] = [1, 1]
                else:
                    i_swing = i - n_ds
                    frac = i_swing / max(1, swing_window - 1)
                    frac = min(frac, 1.0)
                    swing_z = float(swing_z_table[min(i_swing, swing_window - 1)])
                    right_world[idx] = [
                        right_a_start + right_a_swing * frac, -lat, swing_z,
                    ]
                    contact_out[idx] = [1, 0]
                stance_out[idx] = 0  # left

            # --- Phase B: right stance (ZMP at world x=cycle_offset+sl) ---
            # Left always swings 2·sl (from cycle_offset to cycle_offset+2·sl).
            for i in range(n_half):
                t = i * dt
                idx = k * n_cycle + n_half + i

                t_global = k * T_cycle + T_half + t
                lqr_idx = min(int(round(t_global / dt)), n_lqr - 1)
                com_world[idx, 0] = com_lqr[lqr_idx, 0]
                com_world[idx, 1] = com_lqr[lqr_idx, 1]

                right_world[idx] = [cycle_offset + sl, -lat, 0]
                if i < n_ds:
                    left_world[idx] = [cycle_offset, lat, 0]
                    contact_out[idx] = [1, 1]
                else:
                    i_swing = i - n_ds
                    frac = i_swing / max(1, swing_window - 1)
                    frac = min(frac, 1.0)
                    swing_z = float(swing_z_table[min(i_swing, swing_window - 1)])
                    left_world[idx] = [
                        cycle_offset + 2 * sl * frac, lat, swing_z,
                    ]
                    contact_out[idx] = [0, 1]
                stance_out[idx] = 1  # right

        # --- Solve IK ---
        # Resolve leg-joint slot indices from the PolicySpec actuator
        # order; non-leg slots (waist, arms, ankle_roll) stay at the
        # default-zero q_ref initialization.  The v20 ankle_roll merge
        # added two actuators that the ZMP family does not plan for —
        # they are intentionally left at 0 here.
        layout = self._load_actuator_layout()
        n_joints = layout["n_joints"]
        leg_idx = layout["leg_idx"]
        n = n_total
        q_ref = np.zeros((n, n_joints), dtype=np.float32)

        for i in range(n):
            pelvis_x = com_world[i, 0]
            com_y = com_world[i, 1]

            for side, foot_world_i in [
                ("left", left_world[i]),
                ("right", right_world[i]),
            ]:
                lat = cfg.hip_lateral_offset_m
                hip_y = com_y + (lat if side == "left" else -lat)

                # Determine if this leg is in swing (not in contact).
                # Plantarflex is gated on the commanded foot z (not just
                # the contact mask) so the boundary frames where the
                # swing-z envelope returns to zero stay flat-ankle and
                # don't dip the foot below the floor.  The contact mask
                # the env reads is unchanged.
                #
                # Phase 12A (2026-05-08): replaced the previous binary
                # `is_swing_for_ankle` flag (which produced a discrete
                # ~0.15 rad ankle jump within one frame at the swing-z
                # threshold crossing) with a continuous smoothstep
                # blend.  ``plantarflex_strength`` ramps smoothly from
                # 0 (flat-foot) to 1 (full plantarflex) across a
                # ±half-band centred on the threshold, then is forced
                # to 0 during stance regardless of foot z.  Setting
                # ``cfg.plantarflex_ramp_band_m = 0`` recovers the
                # pre-Phase-12A binary behaviour.
                contact_idx = 0 if side == "left" else 1
                is_swing = contact_out[i, contact_idx] < 0.5
                if not is_swing:
                    plantarflex_strength = 0.0
                elif cfg.plantarflex_ramp_band_m <= 0.0:
                    plantarflex_strength = 1.0 if (
                        foot_world_i[2] > swing_ankle_lift_threshold_m
                    ) else 0.0
                else:
                    band_lo = swing_ankle_lift_threshold_m - cfg.plantarflex_ramp_band_m
                    band_hi = swing_ankle_lift_threshold_m + cfg.plantarflex_ramp_band_m
                    t = (foot_world_i[2] - band_lo) / max(
                        band_hi - band_lo, 1e-9
                    )
                    plantarflex_strength = _smoothstep(t)

                # Sagittal IK: foot position relative to HIP joint.
                # The hip joint sits ``pelvis_to_hip_m`` below the
                # pelvis frame in the MJCF, so the IK's chain length
                # excludes that offset.
                foot_rel_x = foot_world_i[0] - pelvis_x
                foot_z_above_ground = foot_world_i[2]
                hip_to_foot_z = -(cfg.com_height_m - cfg.pelvis_to_hip_m
                                  - cfg.ankle_to_ground_m
                                  - foot_z_above_ground)

                hip_p, knee_p, ank_p, _ = _solve_sagittal_ik(
                    foot_rel_x, hip_to_foot_z,
                    cfg.upper_leg_m, cfg.lower_leg_m,
                    cfg.min_reach_margin_m,
                    plantarflex_strength=plantarflex_strength,
                )

                # Hip roll: foot lateral offset from hip
                foot_rel_y = foot_world_i[1] - hip_y
                hip_r = np.arctan2(foot_rel_y, -hip_to_foot_z)
                hip_r = np.clip(hip_r, -0.15, 0.15)

                # Apply WildRobot sign conventions (from
                # nominal_ik_adapter.py): the hip-pitch and hip-roll
                # axes are mirrored across L/R in the MJCF, so the
                # left side negates both; knee + ankle_pitch share
                # sign across L/R.
                if side == "left":
                    q_ref[i, leg_idx["left_hip_pitch"]] = -hip_p
                    q_ref[i, leg_idx["left_hip_roll"]] = -hip_r
                    q_ref[i, leg_idx["left_knee_pitch"]] = knee_p
                    q_ref[i, leg_idx["left_ankle_pitch"]] = ank_p
                else:
                    q_ref[i, leg_idx["right_hip_pitch"]] = hip_p
                    q_ref[i, leg_idx["right_hip_roll"]] = hip_r
                    q_ref[i, leg_idx["right_knee_pitch"]] = knee_p
                    q_ref[i, leg_idx["right_ankle_pitch"]] = ank_p

        # Safety clip — catches any residual IK rounding.  Joint ranges
        # are read from ``mujoco_robot_config.json`` (radians) by
        # ``_load_actuator_layout``; non-leg slots use ``±inf`` so this
        # single clip is a no-op outside the leg slots.
        np.clip(q_ref, layout["leg_clip_min"], layout["leg_clip_max"], out=q_ref)

        # Foot positions stored COM-relative (bounded across cycles).
        left_out = left_world.copy()
        right_out = right_world.copy()
        left_out[:, 0] -= com_world[:, 0]
        left_out[:, 1] -= com_world[:, 1]
        right_out[:, 0] -= com_world[:, 0]
        right_out[:, 1] -= com_world[:, 1]

        # Phase: normalized within one gait cycle, repeating each cycle.
        phase = np.tile(
            np.linspace(0.0, 1.0, n_cycle, endpoint=False, dtype=np.float32),
            n_cycles,
        )

        # Pelvis world pose: monotonically advances with COM.
        pelvis_pos = np.zeros((n, 3), dtype=np.float32)
        pelvis_pos[:, 0] = com_world[:, 0].astype(np.float32)
        pelvis_pos[:, 1] = com_world[:, 1].astype(np.float32)
        pelvis_pos[:, 2] = cfg.com_height_m
        pelvis_rpy = np.zeros((n, 3), dtype=np.float32)

        com_pos = pelvis_pos.copy()
        foot_rpy = np.zeros((n, 3), dtype=np.float32)

        # Phase 3: TB-style realized FK enrichment (mujoco_replay).
        body_pos, body_quat, body_lin_vel, body_ang_vel, site_pos, body_names, site_names = (
            self._fixed_base_fk_replay(q_ref, cfg.dt_s)
        )

        return ReferenceTrajectory(
            command_vx=command_vx,
            dt=cfg.dt_s,
            cycle_time=cycle_time,  # gait period (per-cycle), not total duration
            q_ref=q_ref,
            phase=phase,
            pelvis_pos=pelvis_pos,
            pelvis_rpy=pelvis_rpy,
            com_pos=com_pos,
            left_foot_pos=left_out.astype(np.float32),
            left_foot_rpy=foot_rpy.copy(),
            right_foot_pos=right_out.astype(np.float32),
            right_foot_rpy=foot_rpy.copy(),
            stance_foot_id=stance_out.astype(np.float32),
            contact_mask=contact_out.astype(np.float32),
            body_pos=body_pos,
            body_quat=body_quat,
            body_lin_vel=body_lin_vel,
            body_ang_vel=body_ang_vel,
            site_pos=site_pos,
            body_names=body_names,
            site_names=site_names,
            generator_version="zmp_v0.20.0_multicycle",
        )

    def _generate_standing(self) -> ReferenceTrajectory:
        """Generate a single-frame standing posture."""
        cfg = self.cfg
        n = 1
        n_joints = self._load_actuator_layout()["n_joints"]

        # Standing: straight legs, both feet on ground.  All actuator
        # slots stay at zero — this is the legacy convention from
        # before the v20 ankle_roll merge and is preserved here so
        # the standing keyframe behavior is unchanged for a 21-wide
        # q_ref (waist, arms, ankle_roll all at zero).
        q_ref = np.zeros((n, n_joints), dtype=np.float32)
        phase = np.array([0.0], dtype=np.float32)
        stance = np.array([0.0], dtype=np.float32)  # left stance
        contact = np.ones((n, 2), dtype=np.float32)

        lat = cfg.hip_lateral_offset_m
        pelvis_pos = np.array([[0.0, 0.0, cfg.com_height_m]], dtype=np.float32)
        pelvis_rpy = np.zeros((n, 3), dtype=np.float32)
        com_pos = pelvis_pos.copy()
        left_foot = np.array([[0.0, lat, 0.0]], dtype=np.float32)
        right_foot = np.array([[0.0, -lat, 0.0]], dtype=np.float32)
        foot_rpy = np.zeros((n, 3), dtype=np.float32)

        # Phase 3: TB-style realized FK enrichment for the standing
        # frame as well, so consumers always get a populated body / site
        # path regardless of whether the trajectory is walk or standing.
        body_pos, body_quat, body_lin_vel, body_ang_vel, site_pos, body_names, site_names = (
            self._fixed_base_fk_replay(q_ref, cfg.dt_s)
        )

        return ReferenceTrajectory(
            command_vx=0.0,
            dt=cfg.dt_s,
            cycle_time=cfg.dt_s,
            q_ref=q_ref,
            phase=phase,
            pelvis_pos=pelvis_pos,
            pelvis_rpy=pelvis_rpy,
            com_pos=com_pos,
            left_foot_pos=left_foot,
            left_foot_rpy=foot_rpy.copy(),
            right_foot_pos=right_foot,
            right_foot_rpy=foot_rpy.copy(),
            stance_foot_id=stance,
            contact_mask=contact,
            body_pos=body_pos,
            body_quat=body_quat,
            body_lin_vel=body_lin_vel,
            body_ang_vel=body_ang_vel,
            site_pos=site_pos,
            body_names=body_names,
            site_names=site_names,
            generator_version="zmp_v0.20.0",
        )

    def _plan_footsteps(
        self,
        step_length: float,
        n_cycles: int,
        ds: float | None = None,
        ss: float | None = None,
    ) -> Tuple[List[np.ndarray], np.ndarray, List[np.ndarray]]:
        """Plan footstep positions and timing.

        Returns (footsteps, time_steps, zmp_refs).
        """
        cfg = self.cfg
        lat = cfg.default_stance_width_m
        if ds is None:
            ds = cfg.double_support_s
        if ss is None:
            ss = cfg.single_support_s

        footsteps: List[np.ndarray] = []
        # Start: left foot at origin+lat, right at origin-lat
        left_x = 0.0
        right_x = 0.0

        for i in range(n_cycles):
            # Left step
            left_x = i * 2 * step_length
            footsteps.append(np.array([left_x, lat], dtype=np.float64))
            # Right step
            right_x = (i * 2 + 1) * step_length
            footsteps.append(np.array([right_x, -lat], dtype=np.float64))

        # Build timing: alternating DS/SS phases
        time_list = [0.0, ds]
        for _ in range(len(footsteps) - 1):
            time_list.extend([ss, ds])
        time_steps = np.cumsum(time_list)

        # ZMP refs: one per transition (same count as time_steps)
        zmp_refs: List[np.ndarray] = []
        for step in footsteps:
            zmp_refs.append(step.copy())  # DS: under previous stance
            zmp_refs.append(step.copy())  # SS: under current stance

        # Trim to match time_steps length
        zmp_refs = zmp_refs[:len(time_steps)]

        return footsteps, time_steps, zmp_refs

    def _compute_foot_trajectories(
        self,
        footsteps: List[np.ndarray],
        time_steps: np.ndarray,
        n_total: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute per-timestep foot positions, stance IDs, and contact masks.

        Returns (left_foot_pos, right_foot_pos, stance_foot_id, contact_mask).
        """
        cfg = self.cfg
        lat = cfg.default_stance_width_m

        left_pos = np.zeros((n_total, 3), dtype=np.float64)
        right_pos = np.zeros((n_total, 3), dtype=np.float64)
        stance_ids = np.zeros(n_total, dtype=np.float64)
        contacts = np.ones((n_total, 2), dtype=np.float64)

        # Initialize feet at first footstep positions
        left_pos[:, 1] = lat
        right_pos[:, 1] = -lat

        # Current foot positions
        cur_left = np.array([0.0, lat, 0.0], dtype=np.float64)
        cur_right = np.array([0.0, -lat, 0.0], dtype=np.float64)

        step_idx = 0
        for phase_idx in range(len(time_steps) - 1):
            t_start = time_steps[phase_idx]
            t_end = time_steps[phase_idx + 1]
            i_start = int(round(t_start / cfg.dt_s))
            i_end = min(int(round(t_end / cfg.dt_s)), n_total)

            is_double = (phase_idx % 2 == 0)

            if is_double:
                # Double support: feet stay put
                for i in range(i_start, i_end):
                    left_pos[i] = cur_left.copy()
                    right_pos[i] = cur_right.copy()
                    contacts[i] = [1.0, 1.0]
                    # Stance is the foot that was just placed
                    stance_ids[i] = float(step_idx % 2)
            else:
                # Single support: one foot swings
                fs = footsteps[step_idx]
                swing_is_right = (step_idx % 2 == 1)
                step_idx += 1

                if swing_is_right:
                    # Right foot swings to next footstep
                    next_footstep = footsteps[min(step_idx, len(footsteps) - 1)]
                    start_pos = cur_right.copy()
                    end_pos = np.array([next_footstep[0], -lat, 0.0])

                    n_swing = max(1, i_end - i_start)
                    for i in range(i_start, i_end):
                        frac = (i - i_start) / n_swing
                        right_pos[i, 0] = start_pos[0] + (end_pos[0] - start_pos[0]) * frac
                        right_pos[i, 1] = -lat
                        right_pos[i, 2] = cfg.foot_step_height_m * np.sin(np.pi * frac)
                        left_pos[i] = cur_left.copy()
                        contacts[i] = [1.0, 0.0]
                        stance_ids[i] = 0.0  # left stance

                    cur_right = end_pos.copy()
                else:
                    # Left foot swings
                    next_footstep = footsteps[min(step_idx, len(footsteps) - 1)]
                    start_pos = cur_left.copy()
                    end_pos = np.array([next_footstep[0], lat, 0.0])

                    n_swing = max(1, i_end - i_start)
                    for i in range(i_start, i_end):
                        frac = (i - i_start) / n_swing
                        left_pos[i, 0] = start_pos[0] + (end_pos[0] - start_pos[0]) * frac
                        left_pos[i, 1] = lat
                        left_pos[i, 2] = cfg.foot_step_height_m * np.sin(np.pi * frac)
                        right_pos[i] = cur_right.copy()
                        contacts[i] = [0.0, 1.0]
                        stance_ids[i] = 1.0  # right stance

                    cur_left = end_pos.copy()

        return left_pos, right_pos, stance_ids, contacts

    def build_library(
        self,
        command_range_vx: Tuple[float, float] = (0.0, 0.25),
        interval: float = 0.05,
    ) -> ReferenceLibrary:
        """Build a complete reference library across command bins.

        Parameters
        ----------
        command_range_vx : (min_vx, max_vx)
        interval : command bin spacing in m/s
        """
        vx_values = np.arange(
            command_range_vx[0],
            command_range_vx[1] + interval * 0.5,
            interval,
        )
        return self._build_library_from_vx_values(
            vx_values=vx_values,
            command_range_vx=command_range_vx,
            interval=float(interval),
        )

    def build_library_for_vx_values(
        self,
        vx_values: Sequence[float],
        *,
        interval: float | None = None,
    ) -> ReferenceLibrary:
        """Build a deterministic library for an explicit set of ``vx`` bins.

        This lifecycle path mirrors ToddlerBot's pre-baked lookup-table usage:
        parity/debug tools can predefine the exact command bins once, persist the
        resulting asset, and reuse it across repeated runs.
        """
        if not vx_values:
            raise ValueError("build_library_for_vx_values requires at least one vx value.")
        vx_sorted = sorted({round(float(v), 4) for v in vx_values})
        range_vx = (float(vx_sorted[0]), float(vx_sorted[-1]))
        if interval is None:
            if len(vx_sorted) >= 2:
                diffs = np.diff(np.asarray(vx_sorted, dtype=np.float64))
                interval = float(np.min(diffs))
            else:
                interval = 0.0
        return self._build_library_from_vx_values(
            vx_values=vx_sorted,
            command_range_vx=range_vx,
            interval=float(interval),
        )

    def _build_library_from_vx_values(
        self,
        *,
        vx_values: Sequence[float],
        command_range_vx: Tuple[float, float],
        interval: float,
    ) -> ReferenceLibrary:
        trajectories = []
        for vx in vx_values:
            vx_rounded = round(float(vx), 4)
            print(f"  Generating vx={vx_rounded:+.3f} m/s ...", end="", flush=True)
            traj = self.generate(vx_rounded)
            issues = traj.validate()
            if issues:
                print(f" ISSUES: {issues}")
                traj.is_valid = False
                traj.validation_notes = "; ".join(issues)
            else:
                print(f" OK (steps={traj.n_steps})")
            trajectories.append(traj)

        meta = ReferenceLibraryMeta(
            generator="zmp_walk",
            generator_version="0.20.0",
            robot="wildrobot_v2",
            dt=self.cfg.dt_s,
            cycle_time=self.cfg.cycle_time_s,
            n_joints=self._load_actuator_layout()["n_joints"],
            command_range_vx=command_range_vx,
            command_interval=interval,
        )
        return ReferenceLibrary(trajectories, meta)
