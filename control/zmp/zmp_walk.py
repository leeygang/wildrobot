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

    # Robot morphology
    upper_leg_m: float = 0.21
    lower_leg_m: float = 0.21
    hip_lateral_offset_m: float = 0.0536
    ankle_to_ground_m: float = 0.06

    # Joint limits (radians) — used to compute safe COM height and step length
    ankle_dorsiflexion_limit_rad: float = 0.698   # 40°
    hip_extension_limit_rad: float = 0.087        # 5° backward extension
    knee_pitch_max_rad: float = 1.396             # 80°

    # Gait timing
    cycle_time_s: float = 0.50
    single_double_ratio: float = 2.0
    dt_s: float = 0.02

    # Gait geometry
    foot_step_height_m: float = 0.025
    default_stance_width_m: float = 0.0536  # = hip_lateral_offset_m
    min_walking_speed_mps: float = 0.08     # below this, use standing (no tiny shuffles)

    # ZMP planner costs
    zmp_cost_Q: float = 1.0
    zmp_cost_R: float = 0.1

    # Trajectory generation: plan extra cycles to ensure enough steps
    n_plan_extra_cycles: int = 2  # extra cycles beyond needed output
    n_output_cycles: int = 1  # cycles to keep in library entry

    # Limits
    max_step_length_m: float = 0.10
    min_reach_margin_m: float = 0.01

    @property
    def com_height_m(self) -> float:
        """Compute COM height from joint limits.

        The flat-foot IK constraint ``ankle = -(hip + knee)`` means the
        ankle dorsiflexion grows with knee bend.  Given:
          - ankle limit: ``ankle_dorsiflexion_limit_rad``
          - worst-case hip during walking: ~15° (foot half a step ahead)
        The maximum safe knee bend is:
          ``knee_max = ankle_limit - hip_margin``
        The COM height is derived from the leg reach at that knee bend,
        with a small margin to keep the ankle away from hard saturation.
        """
        # Reserve ankle budget for hip pitch excursion during walking.
        # During stance, hip pitch can reach ~15° when the foot is half
        # a step ahead of COM.  Keep 5° additional margin from the limit.
        hip_margin_rad = np.radians(20.0)  # 15° typical + 5° margin
        knee_budget = self.ankle_dorsiflexion_limit_rad - hip_margin_rad
        knee_safe = max(np.radians(10.0), min(knee_budget, self.knee_pitch_max_rad))

        # Leg reach at this knee bend (foot directly under hip)
        # Law of cosines: reach² = l1² + l2² + 2·l1·l2·cos(knee)
        # (knee=0 → straight → reach=l1+l2, knee=π → folded → reach=0)
        l1, l2 = self.upper_leg_m, self.lower_leg_m
        reach = np.sqrt(l1**2 + l2**2 + 2 * l1 * l2 * np.cos(knee_safe))

        return reach + self.ankle_to_ground_m

    @property
    def safe_max_step_length_m(self) -> float:
        """Max step length that keeps all joints within limits.

        Two constraints:
        1. Ankle: hip_pitch + knee ≤ ankle_dorsiflexion_limit
        2. Hip extension: foot behind COM → hip extends backward ≤ hip_extension_limit

        The tighter constraint wins.
        """
        hip_to_ankle = self.com_height_m - self.ankle_to_ground_m

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
) -> Tuple[float, float, float, bool]:
    """Analytical 2-link sagittal IK.

    Returns (hip_pitch, knee_pitch, ankle_pitch, reachable).
    Ankle pitch is clamped to ±ankle_limit_rad rather than forced to
    keep the foot perfectly flat.  This avoids saturating the ankle
    joint on robots with limited dorsiflexion range.
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

    # Ankle: ideally -(hip+knee) for flat foot, but clamped to joint limits
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
            [-0.087, 1.484],   # left_hip_pitch
            [-1.484, 0.087],   # right_hip_pitch
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

    def _generate_at_step_length(
        self, command_vx: float, step_length: float,
        cycle_time_override: float | None = None,
    ) -> ReferenceTrajectory:
        """Generate trajectory at a specific step length (no limit reduction)."""
        cfg = self.cfg
        cycle_time = cycle_time_override or cfg.cycle_time_s

        # Plan footsteps with timing derived from the (possibly adjusted) cycle_time
        ds = cycle_time / 2.0 / (cfg.single_double_ratio + 1.0)
        ss = cfg.single_double_ratio * ds

        total_cycles = cfg.n_output_cycles + cfg.n_plan_extra_cycles
        footsteps, time_steps, zmp_refs = self._plan_footsteps(
            step_length, total_cycles, ds=ds, ss=ss
        )

        # Run ZMP planner
        total_time = time_steps[-1]
        n_total = int(np.ceil(total_time / cfg.dt_s))
        x0 = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)

        self.planner.plan(
            time_steps, zmp_refs, x0, cfg.com_height_m,
            Qy=np.eye(2) * cfg.zmp_cost_Q,
            R=np.eye(2) * cfg.zmp_cost_R,
        )

        # Forward-integrate COM trajectory using the planner's optimal
        # feedback law.  This gives better numerical behavior than the
        # planner's analytical nominal trajectory (which can diverge
        # due to ExpPlusPPoly accumulation errors).
        com_traj = np.zeros((n_total, 4), dtype=np.float64)
        com_traj[0] = x0
        for i in range(1, n_total):
            t = i * cfg.dt_s
            u = self.planner.get_optim_com_acc(t, com_traj[i - 1])
            com_traj[i, :2] = com_traj[i - 1, :2] + com_traj[i - 1, 2:] * cfg.dt_s
            com_traj[i, 2:] = com_traj[i - 1, 2:] + u * cfg.dt_s

        # Compute foot trajectories
        left_foot, right_foot, stance_ids, contacts = self._compute_foot_trajectories(
            footsteps, time_steps, n_total
        )

        # Extract one gait cycle that is truly periodic.
        # A full cycle = left step + right step. Find consecutive points
        # where the gait repeats: stance transitions from L→R, which
        # marks the same phase point each cycle.
        stance_lr_transitions = []
        for i in range(1, n_total):
            # Detect L→R stance transition (stance goes 0→1)
            if stance_ids[i] > 0.5 and stance_ids[i - 1] < 0.5:
                stance_lr_transitions.append(i)

        # Need at least 3 transitions to skip warmup and extract one cycle
        if len(stance_lr_transitions) >= 3:
            start_step = stance_lr_transitions[1]
            end_step = stance_lr_transitions[2]
        elif len(stance_lr_transitions) >= 2:
            start_step = stance_lr_transitions[0]
            end_step = stance_lr_transitions[1]
        else:
            # Fallback: use fixed extraction
            steps_per_cycle = int(round(cycle_time / cfg.dt_s))
            out_steps = steps_per_cycle * cfg.n_output_cycles
            start_step = min(steps_per_cycle, n_total - out_steps)
            start_step = max(start_step, 0)
            end_step = min(start_step + out_steps, n_total)

        actual_out = end_step - start_step
        sl = slice(start_step, end_step)

        # Keep world-frame copies for IK computation.
        # Use integrated COM (not planner's analytical nominal, which diverges).
        com_world = com_traj[sl, :2].copy()  # [n, 2] integrated COM x,y
        left_world = left_foot[sl].copy()    # [n, 3] foot xyz in world
        right_world = right_foot[sl].copy()
        stance_out = stance_ids[sl].copy()
        contact_out = contacts[sl].copy()

        # Solve IK from world-frame positions (before COM-relative conversion).
        # IK target = foot position relative to hip joint.
        # Hip joint is at (COM_x, COM_y ± hip_lateral_offset, COM_height).
        n = actual_out
        n_joints = 19
        q_ref = np.zeros((n, n_joints), dtype=np.float32)

        for i in range(n):
            com_x = com_world[i, 0]
            com_y = com_world[i, 1]

            for side, foot_world_i, indices in [
                ("left", left_world[i], [0, 2, 4, 6]),
                ("right", right_world[i], [1, 3, 5, 7]),
            ]:
                lat = cfg.hip_lateral_offset_m
                hip_y = com_y + (lat if side == "left" else -lat)

                # Sagittal IK: foot position relative to hip in x-z plane
                foot_rel_x = foot_world_i[0] - com_x
                foot_z_above_ground = foot_world_i[2]
                hip_to_foot_z = -(cfg.com_height_m - cfg.ankle_to_ground_m
                                  - foot_z_above_ground)

                hip_p, knee_p, ank_p, _ = _solve_sagittal_ik(
                    foot_rel_x, hip_to_foot_z,
                    cfg.upper_leg_m, cfg.lower_leg_m,
                    cfg.min_reach_margin_m,
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
            [-0.087, 1.484], [-1.484, 0.087],
            [-1.571, 0.175], [-0.175, 1.571],
            [0.0, 1.396], [0.0, 1.396],
            [-0.698, 0.785], [-0.698, 0.785],
        ], dtype=np.float32)
        for j in range(8):
            q_ref[:, j] = np.clip(q_ref[:, j], joint_limits_rad[j, 0], joint_limits_rad[j, 1])

        # Now convert foot positions to COM-relative for periodic storage
        left_out = left_world.copy()
        right_out = right_world.copy()
        for i in range(n):
            left_out[i, 0] -= com_world[i, 0]
            left_out[i, 1] -= com_world[i, 1]
            right_out[i, 0] -= com_world[i, 0]
            right_out[i, 1] -= com_world[i, 1]

        # COM in the de-trended frame
        com_detrended = np.zeros((n, 4), dtype=np.float64)
        stride_x = com_world[-1, 0] - com_world[0, 0]
        stride_y = com_world[-1, 1] - com_world[0, 1]
        for i in range(n):
            frac = i / n
            com_detrended[i, 0] = com_world[i, 0] - com_world[0, 0] - frac * stride_x
            com_detrended[i, 1] = com_world[i, 1] - com_world[0, 1] - frac * stride_y

        # Build phase array
        phase = np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float32)

        # Pelvis position: de-trended COM oscillation + height
        pelvis_pos = np.zeros((n, 3), dtype=np.float32)
        pelvis_pos[:, 0] = com_detrended[:, 0].astype(np.float32)
        pelvis_pos[:, 1] = com_detrended[:, 1].astype(np.float32)
        pelvis_pos[:, 2] = cfg.com_height_m
        pelvis_rpy = np.zeros((n, 3), dtype=np.float32)

        # COM position: same as pelvis for LIPM
        com_pos = pelvis_pos.copy()

        foot_rpy = np.zeros((n, 3), dtype=np.float32)

        return ReferenceTrajectory(
            command_vx=command_vx,
            dt=cfg.dt_s,
            cycle_time=actual_out * cfg.dt_s,
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
            generator_version="zmp_v0.20.0",
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
