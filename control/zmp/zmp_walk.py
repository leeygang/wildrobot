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

from dataclasses import dataclass
from typing import List, Tuple

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
    knee_pitch_max_rad: float = 1.396             # 80°

    # Gait timing — longer cycle gives bigger steps at same speed
    # ToddlerBot uses 0.72s; WildRobot uses 0.64s (similar proportion)
    cycle_time_s: float = 0.64
    single_double_ratio: float = 2.0
    dt_s: float = 0.02

    # Gait geometry.  foot_step_height was 0.08 before round 7
    # (~19 % of the OLD 0.42 m leg length).  After the geometry
    # refit (real leg length = 0.193 + 0.180 = 0.373 m), the same
    # 0.08 m peak lift required the knee to bend 92° at swing
    # apex — over the 80° knee limit, triggering knee saturation
    # in fixed-base / C1 / C2.  Reduced to 0.05 m (~13 % of the
    # actual leg length) which keeps the swing-peak knee inside
    # the limit.  ToddlerBot uses 24 % of leg length but also has
    # a higher knee limit; the saturation gate (< 5 % in fixed-
    # base) was the binding constraint here.
    foot_step_height_m: float = 0.04
    default_stance_width_m: float = 0.0536  # = hip_lateral_offset_m
    min_walking_speed_mps: float = 0.06     # below this, use standing

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


def _solve_sagittal_ik(
    target_x: float,
    target_z: float,
    l1: float,
    l2: float,
    min_margin: float = 0.01,
    ankle_limit_rad: float = 0.698,
    is_swing: bool = False,
) -> Tuple[float, float, float, bool]:
    """Analytical 2-link sagittal IK.

    Returns (hip_pitch, knee_pitch, ankle_pitch, reachable).

    During stance (is_swing=False): ankle keeps the foot flat on the
    ground, clamped to ±ankle_limit_rad.

    During swing (is_swing=True): ankle stays at a neutral position
    (slight plantarflexion for toe clearance) rather than tracking
    the flat-foot constraint.  This allows the knee to bend freely
    for foot lift without exhausting the ankle budget.
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

    if is_swing:
        # During swing: ankle at neutral (slight plantarflexion).
        # The foot is in the air — no need to stay flat.
        ankle_pitch = np.clip(-0.15, -ankle_limit_rad, ankle_limit_rad)
    else:
        # During stance: keep foot flat, clamped to joint limits
        ankle_flat = -(hip_pitch + knee_pitch)
        ankle_pitch = np.clip(ankle_flat, -ankle_limit_rad, ankle_limit_rad)

    return hip_pitch, knee_pitch, ankle_pitch, reachable


class ZMPWalkGenerator:
    """Generate walking trajectories for WildRobot v2 using ZMP preview control."""

    def __init__(self, config: ZMPWalkConfig | None = None) -> None:
        self.cfg = config or ZMPWalkConfig()
        self.planner = ZMPPlanner()

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
        """Count timesteps where any leg joint exceeds its limit."""
        limits = np.array([
            [-0.5236, 1.5708],   # left_hip_pitch
            [-1.5708, 0.5236],   # right_hip_pitch
            [-1.571, 0.175],   # left_hip_roll
            [-0.175, 1.571],   # right_hip_roll
            [0.0, 1.396],      # left_knee_pitch
            [0.0, 1.396],      # right_knee_pitch
            [-0.698, 0.785],   # left_ankle_pitch
            [-0.698, 0.785],   # right_ankle_pitch
        ])
        count = 0
        for j in range(8):
            count += int(np.sum(
                (traj.q_ref[:, j] < limits[j, 0] - 0.005) |
                (traj.q_ref[:, j] > limits[j, 1] + 0.005)
            ))
        return count

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

        # Swing-foot z offset: during swing the IK plantarflexes the
        # ankle by 0.15 rad for toe clearance, which drops the foot's
        # TOE corner ~9 mm below the foot body and the foot collision
        # box's geometry adds another ~9 mm of swing-side vertical
        # extent on the rotated toe.  Without compensation the q_ref
        # placed swing feet below the floor at lift-off and
        # touchdown frames (round-7 FK gate caught this).
        swing_foot_z_floor_clearance_m = 0.025

        n_half = int(round(T_half / dt))
        n_cycle = 2 * n_half
        n_ds = int(round(ds / dt))

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
                    frac = (i - n_ds) / max(1, n_half - n_ds - 1)
                    frac = min(frac, 1.0)
                    right_world[idx] = [
                        right_a_start + right_a_swing * frac, -lat,
                        (cfg.foot_step_height_m * np.sin(np.pi * frac)
                         + swing_foot_z_floor_clearance_m),
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
                    frac = (i - n_ds) / max(1, n_half - n_ds - 1)
                    frac = min(frac, 1.0)
                    left_world[idx] = [
                        cycle_offset + 2 * sl * frac, lat,
                        (cfg.foot_step_height_m * np.sin(np.pi * frac)
                         + swing_foot_z_floor_clearance_m),
                    ]
                    contact_out[idx] = [0, 1]
                stance_out[idx] = 1  # right

        # --- Solve IK ---
        n = n_total
        n_joints = 19
        q_ref = np.zeros((n, n_joints), dtype=np.float32)

        for i in range(n):
            pelvis_x = com_world[i, 0]
            com_y = com_world[i, 1]

            for side, foot_world_i, indices in [
                ("left", left_world[i], [0, 2, 4, 6]),
                ("right", right_world[i], [1, 3, 5, 7]),
            ]:
                lat = cfg.hip_lateral_offset_m
                hip_y = com_y + (lat if side == "left" else -lat)

                # Determine if this leg is in swing (not in contact)
                contact_idx = 0 if side == "left" else 1
                is_swing = contact_out[i, contact_idx] < 0.5

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
                    is_swing=is_swing,
                )

                # Hip roll: foot lateral offset from hip
                foot_rel_y = foot_world_i[1] - hip_y
                hip_r = np.arctan2(foot_rel_y, -hip_to_foot_z)
                hip_r = np.clip(hip_r, -0.15, 0.15)

                # Apply WildRobot sign conventions (from nominal_ik_adapter.py):
                if side == "left":
                    q_ref[i, indices[0]] = -hip_p
                    q_ref[i, indices[1]] = -hip_r
                else:
                    q_ref[i, indices[0]] = hip_p
                    q_ref[i, indices[1]] = hip_r
                q_ref[i, indices[2]] = knee_p
                q_ref[i, indices[3]] = ank_p

        # Safety clip — catches any residual IK rounding.
        joint_limits_rad = np.array([
            [-0.5236, 1.5708], [-1.5708, 0.5236],
            [-1.571, 0.175], [-0.175, 1.571],
            [0.0, 1.396], [0.0, 1.396],
            [-0.698, 0.785], [-0.698, 0.785],
        ], dtype=np.float32)
        for j in range(8):
            q_ref[:, j] = np.clip(q_ref[:, j], joint_limits_rad[j, 0], joint_limits_rad[j, 1])

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
            generator_version="zmp_v0.20.0_multicycle",
        )

    def _generate_standing(self) -> ReferenceTrajectory:
        """Generate a single-frame standing posture."""
        cfg = self.cfg
        n = 1
        n_joints = 19

        # Standing: straight legs, both feet on ground
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
            n_joints=19,
            command_range_vx=command_range_vx,
            command_interval=interval,
        )

        return ReferenceLibrary(trajectories, meta)
