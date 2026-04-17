"""Offline reference library schema for v0.20.0 prior-reference pivot.

This module defines the command-indexed reference library that replaces
the runtime walking FSM.  The library stores pre-computed reference
trajectories generated offline (by ZMP preview control, ALIP, or any
future planner) and exposes a lookup/preview interface consumed by:

  - the training env (via runtime reference service)
  - the bounded-residual PPO policy (via preview window observations)
  - viewers and evaluation tools (via direct trajectory access)

Design notes
------------
- Schema is ZMP-first but ALIP-compatible: no field assumes a specific
  reduced-order model.
- Trajectories are stored per gait cycle and meant to be looped.
- The library is the single source of locomotion semantics for v0.20.x.
  Runtime layers must NOT reintroduce gait FSM logic.

Reference implementation target
--------------------------------
- Algorithm family : Kajita et al. ZMP preview control (LIPM / cart-table)
- Public reference : ToddlerBot ``toddlerbot.algorithms.zmp_planner`` and
  ``toddlerbot.algorithms.zmp_walk`` (github.com/hshi74/toddlerbot)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Library entry: one gait-cycle trajectory for a single command bin
# ---------------------------------------------------------------------------

@dataclass
class ReferenceTrajectory:
    """One gait cycle of reference data for a single command bin.

    All per-timestep arrays have shape ``[n_steps, ...]`` where
    ``n_steps = int(round(cycle_time / dt))``.  The trajectory is
    periodic: ``trajectory[0]`` follows ``trajectory[-1]`` for
    continuous walking.

    For a **stop** command (vx == 0), the trajectory is a single-frame
    standing posture (``n_steps == 1``) that repeats indefinitely.
    """

    # -- Command key ----------------------------------------------------------
    command_vx: float
    command_vy: float = 0.0
    command_yaw_rate: float = 0.0

    # -- Timing ---------------------------------------------------------------
    dt: float = 0.02
    cycle_time: float = 0.50

    # -- Per-timestep nominal targets (arrays set after init) -----------------
    # Joint-space targets: shape [n_steps, n_joints]
    q_ref: Optional[np.ndarray] = field(default=None, repr=False)
    dq_ref: Optional[np.ndarray] = field(default=None, repr=False)

    # -- Per-timestep task-space preview --------------------------------------
    # Pelvis pose: shape [n_steps, 3] (x, y, z in world or stance frame)
    pelvis_pos: Optional[np.ndarray] = field(default=None, repr=False)
    # Pelvis orientation: shape [n_steps, 3] (roll, pitch, yaw)
    pelvis_rpy: Optional[np.ndarray] = field(default=None, repr=False)
    # COM position: shape [n_steps, 3]
    com_pos: Optional[np.ndarray] = field(default=None, repr=False)
    # Left foot pose: shape [n_steps, 3] position
    left_foot_pos: Optional[np.ndarray] = field(default=None, repr=False)
    # Left foot orientation: shape [n_steps, 3] (roll, pitch, yaw)
    left_foot_rpy: Optional[np.ndarray] = field(default=None, repr=False)
    # Right foot pose: shape [n_steps, 3] position
    right_foot_pos: Optional[np.ndarray] = field(default=None, repr=False)
    # Right foot orientation: shape [n_steps, 3] (roll, pitch, yaw)
    right_foot_rpy: Optional[np.ndarray] = field(default=None, repr=False)

    # -- Per-timestep annotations ---------------------------------------------
    # Stance foot ID per step: shape [n_steps] (0=left, 1=right)
    stance_foot_id: Optional[np.ndarray] = field(default=None, repr=False)
    # Contact mask per step: shape [n_steps, 2] (left, right)
    contact_mask: Optional[np.ndarray] = field(default=None, repr=False)
    # Phase within the gait cycle: shape [n_steps] in [0, 1)
    phase: Optional[np.ndarray] = field(default=None, repr=False)

    # -- Metadata -------------------------------------------------------------
    generator_version: str = "unset"
    is_valid: bool = True
    validation_notes: str = ""

    @property
    def n_steps(self) -> int:
        """Number of timesteps in this trajectory."""
        if self.q_ref is not None:
            return self.q_ref.shape[0]
        if self.phase is not None:
            return self.phase.shape[0]
        return max(1, int(round(self.cycle_time / self.dt)))

    @property
    def n_joints(self) -> int:
        """Number of joints in q_ref."""
        if self.q_ref is not None:
            if self.q_ref.ndim < 2:
                return 0
            return self.q_ref.shape[1]
        return 0

    @property
    def command_key(self) -> Tuple[float, float, float]:
        return (self.command_vx, self.command_vy, self.command_yaw_rate)

    def validate(self) -> List[str]:
        """Return a list of validation issues (empty = pass)."""
        issues: List[str] = []
        n = self.n_steps
        if n < 1:
            issues.append("n_steps < 1")
            return issues

        def _check(name: str, arr: Optional[np.ndarray],
                   expected_shape: Tuple[int, ...]) -> None:
            if arr is None:
                return
            if not isinstance(arr, np.ndarray):
                issues.append(f"{name}: expected ndarray, got {type(arr).__name__}")
                return
            if arr.shape != expected_shape:
                issues.append(
                    f"{name}: expected shape {expected_shape}, got {arr.shape}"
                )

        nj = self.n_joints
        if self.q_ref is not None and self.q_ref.ndim != 2:
            issues.append(
                f"q_ref: expected 2-D array [n_steps, n_joints], "
                f"got {self.q_ref.ndim}-D with shape {self.q_ref.shape}"
            )
        elif nj > 0:
            _check("q_ref", self.q_ref, (n, nj))
            _check("dq_ref", self.dq_ref, (n, nj))

        _check("pelvis_pos", self.pelvis_pos, (n, 3))
        _check("pelvis_rpy", self.pelvis_rpy, (n, 3))
        _check("com_pos", self.com_pos, (n, 3))
        _check("left_foot_pos", self.left_foot_pos, (n, 3))
        _check("left_foot_rpy", self.left_foot_rpy, (n, 3))
        _check("right_foot_pos", self.right_foot_pos, (n, 3))
        _check("right_foot_rpy", self.right_foot_rpy, (n, 3))
        _check("stance_foot_id", self.stance_foot_id, (n,))
        _check("contact_mask", self.contact_mask, (n, 2))
        _check("phase", self.phase, (n,))

        if self.q_ref is None:
            issues.append("q_ref is required")
        if self.phase is None:
            issues.append("phase is required")
        if self.contact_mask is None:
            issues.append("contact_mask is required")
        if self.stance_foot_id is None:
            issues.append("stance_foot_id is required")

        return issues


# ---------------------------------------------------------------------------
# Preview window: what the policy sees at each timestep
# ---------------------------------------------------------------------------

@dataclass
class ReferencePreviewWindow:
    """Per-step reference preview exposed to the policy observation.

    This is the runtime interface between the reference library and the
    training env / policy.  It replaces the old ``LocomotionReferenceState``
    for the v0.20 path.
    """

    # -- Current nominal targets ----------------------------------------------
    q_ref: np.ndarray              # [n_joints]
    phase_sin: float
    phase_cos: float
    stance_foot_id: int            # 0=left, 1=right

    # -- Current task-space targets -------------------------------------------
    pelvis_height: float
    pelvis_roll: float
    pelvis_pitch: float
    swing_foot_pos: np.ndarray     # [3] in stance frame
    swing_foot_vel: np.ndarray     # [3] in stance frame
    next_foothold: np.ndarray      # [2] (x, y) in stance frame

    # -- Contact annotation ---------------------------------------------------
    contact_mask: np.ndarray       # [2] (left, right)

    # -- Future preview (for reference-conditioned policy) --------------------
    future_q_ref: np.ndarray       # [n_preview, n_joints]
    future_phase_sin: np.ndarray   # [n_preview]
    future_phase_cos: np.ndarray   # [n_preview]
    future_contact_mask: np.ndarray  # [n_preview, 2]

    def to_obs_array(self) -> np.ndarray:
        """Flatten to a 1-D observation vector for policy consumption.

        Layout:
          q_ref                          (n_joints)
          phase_sin, phase_cos           (2)
          stance_foot_id                 (1)
          next_foothold                  (2)
          swing_foot_pos                 (3)
          swing_foot_vel                 (3)
          pelvis_height, roll, pitch     (3)
          contact_mask                   (2)
          --- total base: n_joints + 16 ---
          future_q_ref                   (n_preview * n_joints)
          future_phase_sin/cos           (2 * n_preview)
          future_contact_mask            (2 * n_preview)
        """
        base = np.concatenate([
            self.q_ref.ravel().astype(np.float32),
            np.array([
                self.phase_sin, self.phase_cos,
                float(self.stance_foot_id),
                *self.next_foothold,
                *self.swing_foot_pos,
                *self.swing_foot_vel,
                self.pelvis_height, self.pelvis_roll, self.pelvis_pitch,
                *self.contact_mask,
            ], dtype=np.float32),
        ])
        future = np.concatenate([
            self.future_q_ref.ravel().astype(np.float32),
            self.future_phase_sin.astype(np.float32),
            self.future_phase_cos.astype(np.float32),
            self.future_contact_mask.ravel().astype(np.float32),
        ])
        return np.concatenate([base, future])


# ---------------------------------------------------------------------------
# Reference library: command-indexed collection of trajectories
# ---------------------------------------------------------------------------

@dataclass
class ReferenceLibraryMeta:
    """Library-level metadata."""
    generator: str = "unset"
    generator_version: str = "0.0.0"
    robot: str = "wildrobot_v2"
    dt: float = 0.02
    cycle_time: float = 0.50
    n_joints: int = 0
    command_range_vx: Tuple[float, float] = (-0.05, 0.25)
    command_range_vy: Tuple[float, float] = (0.0, 0.0)
    command_range_yaw: Tuple[float, float] = (0.0, 0.0)
    command_interval: float = 0.05
    created: str = ""


class ReferenceLibrary:
    """Command-indexed collection of reference trajectories.

    The library is the single source of locomotion semantics for v0.20.x.
    It stores pre-computed trajectories indexed by command key and supports
    nearest-neighbor lookup with optional interpolation.
    """

    def __init__(
        self,
        trajectories: Optional[List[ReferenceTrajectory]] = None,
        meta: Optional[ReferenceLibraryMeta] = None,
    ) -> None:
        self.meta = meta or ReferenceLibraryMeta()
        self._entries: Dict[Tuple[float, float, float], ReferenceTrajectory] = {}
        if trajectories:
            for traj in trajectories:
                self.add(traj)

    def add(self, traj: ReferenceTrajectory) -> None:
        """Add a trajectory to the library."""
        self._entries[traj.command_key] = traj

    @property
    def command_keys(self) -> List[Tuple[float, float, float]]:
        """All command keys in the library, sorted."""
        return sorted(self._entries.keys())

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: Tuple[float, float, float]) -> bool:
        return key in self._entries

    def __getitem__(self, key: Tuple[float, float, float]) -> ReferenceTrajectory:
        return self._entries[key]

    def lookup(
        self,
        vx: float,
        vy: float = 0.0,
        yaw_rate: float = 0.0,
    ) -> ReferenceTrajectory:
        """Nearest-neighbor command lookup."""
        if not self._entries:
            raise ValueError("Library is empty")
        query = np.array([vx, vy, yaw_rate])
        best_key = min(
            self._entries.keys(),
            key=lambda k: float(np.linalg.norm(np.array(k) - query)),
        )
        return self._entries[best_key]

    def get_preview(
        self,
        vx: float,
        step_index: int,
        n_preview: int = 5,
        vy: float = 0.0,
        yaw_rate: float = 0.0,
    ) -> ReferencePreviewWindow:
        """Extract a preview window at the given phase step.

        Parameters
        ----------
        vx : forward speed command
        step_index : integer index into the gait-cycle trajectory
        n_preview : number of future steps to include in the preview
        """
        traj = self.lookup(vx, vy, yaw_rate)
        n = traj.n_steps
        idx = step_index % n

        phase_val = traj.phase[idx] if traj.phase is not None else 0.0
        phase_sin = float(np.sin(2.0 * np.pi * phase_val))
        phase_cos = float(np.cos(2.0 * np.pi * phase_val))

        stance_id = int(traj.stance_foot_id[idx]) if traj.stance_foot_id is not None else 0
        contact = traj.contact_mask[idx] if traj.contact_mask is not None else np.ones(2)

        q = traj.q_ref[idx] if traj.q_ref is not None else np.zeros(1)

        # Task-space targets: derive from trajectory arrays
        pelvis_h = float(traj.pelvis_pos[idx, 2]) if traj.pelvis_pos is not None else 0.0
        pelvis_roll = float(traj.pelvis_rpy[idx, 0]) if traj.pelvis_rpy is not None else 0.0
        pelvis_pitch = float(traj.pelvis_rpy[idx, 1]) if traj.pelvis_rpy is not None else 0.0

        # Swing foot = the foot that is NOT the stance foot
        swing_id = 1 - stance_id
        if swing_id == 0:
            swing_pos = traj.left_foot_pos[idx] if traj.left_foot_pos is not None else np.zeros(3)
        else:
            swing_pos = traj.right_foot_pos[idx] if traj.right_foot_pos is not None else np.zeros(3)

        # Swing velocity from finite difference (or zero)
        next_idx = (idx + 1) % n
        if swing_id == 0 and traj.left_foot_pos is not None:
            swing_vel = (traj.left_foot_pos[next_idx] - traj.left_foot_pos[idx]) / traj.dt
        elif swing_id == 1 and traj.right_foot_pos is not None:
            swing_vel = (traj.right_foot_pos[next_idx] - traj.right_foot_pos[idx]) / traj.dt
        else:
            swing_vel = np.zeros(3)

        # Next foothold: stance-frame x,y of the next swing touchdown
        # For now, extract from the foot position at the next stance switch
        next_foothold = self._find_next_foothold(traj, idx, swing_id)

        # Future preview arrays
        future_indices = [(idx + 1 + k) % n for k in range(n_preview)]
        future_q = traj.q_ref[future_indices] if traj.q_ref is not None else np.zeros((n_preview, 1))
        future_phases = traj.phase[future_indices] if traj.phase is not None else np.zeros(n_preview)
        future_contacts = traj.contact_mask[future_indices] if traj.contact_mask is not None else np.ones((n_preview, 2))

        return ReferencePreviewWindow(
            q_ref=q,
            phase_sin=phase_sin,
            phase_cos=phase_cos,
            stance_foot_id=stance_id,
            pelvis_height=pelvis_h,
            pelvis_roll=pelvis_roll,
            pelvis_pitch=pelvis_pitch,
            swing_foot_pos=np.asarray(swing_pos, dtype=np.float32),
            swing_foot_vel=np.asarray(swing_vel, dtype=np.float32),
            next_foothold=np.asarray(next_foothold, dtype=np.float32),
            contact_mask=np.asarray(contact, dtype=np.float32),
            future_q_ref=np.asarray(future_q, dtype=np.float32),
            future_phase_sin=np.sin(2.0 * np.pi * future_phases).astype(np.float32),
            future_phase_cos=np.cos(2.0 * np.pi * future_phases).astype(np.float32),
            future_contact_mask=np.asarray(future_contacts, dtype=np.float32),
        )

    @staticmethod
    def _find_next_foothold(
        traj: ReferenceTrajectory,
        current_idx: int,
        swing_id: int,
    ) -> np.ndarray:
        """Find the touchdown position of the current swing foot.

        Touchdown is detected as the 0→1 transition in the contact mask
        for the swing foot, not just the first frame where contact is 1.
        This avoids returning the wrong foothold during double-support
        phases where the swing foot is already in contact.
        """
        n = traj.n_steps
        if traj.contact_mask is None:
            return np.zeros(2, dtype=np.float32)

        # Walk forward to find the 0→1 contact transition
        prev_contact = traj.contact_mask[current_idx, swing_id]
        for offset in range(1, n):
            future_idx = (current_idx + offset) % n
            curr_contact = traj.contact_mask[future_idx, swing_id]
            if prev_contact < 0.5 and curr_contact > 0.5:
                # Found touchdown transition — return foot position
                if swing_id == 0 and traj.left_foot_pos is not None:
                    return traj.left_foot_pos[future_idx, :2]
                elif swing_id == 1 and traj.right_foot_pos is not None:
                    return traj.right_foot_pos[future_idx, :2]
                break
            prev_contact = curr_contact

        return np.zeros(2, dtype=np.float32)

    # -- Validation -----------------------------------------------------------

    def validate(self) -> Dict[Tuple[float, float, float], List[str]]:
        """Validate all entries. Returns {command_key: [issues]}."""
        results = {}
        for key, traj in self._entries.items():
            issues = traj.validate()
            if issues:
                results[key] = issues
        return results

    # -- Persistence ----------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save library to a directory (metadata.json + per-entry .npz)."""
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)

        meta_dict = asdict(self.meta)
        # Convert tuples to lists for JSON serialization
        for k, v in meta_dict.items():
            if isinstance(v, tuple):
                meta_dict[k] = list(v)
        with open(out / "metadata.json", "w") as f:
            json.dump(meta_dict, f, indent=2)

        for i, (key, traj) in enumerate(sorted(self._entries.items())):
            arrays = {}
            scalars = {
                "command_vx": traj.command_vx,
                "command_vy": traj.command_vy,
                "command_yaw_rate": traj.command_yaw_rate,
                "dt": traj.dt,
                "cycle_time": traj.cycle_time,
                "generator_version": traj.generator_version,
                "is_valid": traj.is_valid,
                "validation_notes": traj.validation_notes,
            }
            for arr_name in (
                "q_ref", "dq_ref", "pelvis_pos", "pelvis_rpy", "com_pos",
                "left_foot_pos", "left_foot_rpy", "right_foot_pos",
                "right_foot_rpy", "stance_foot_id", "contact_mask", "phase",
            ):
                arr = getattr(traj, arr_name)
                if arr is not None:
                    arrays[arr_name] = arr

            np.savez(
                out / f"entry_{i:04d}.npz",
                **arrays,
                _scalars_json=json.dumps(scalars),
            )

    @classmethod
    def load(cls, path: str | Path) -> "ReferenceLibrary":
        """Load library from a directory."""
        src = Path(path)
        with open(src / "metadata.json") as f:
            meta_dict = json.load(f)
        # Convert lists back to tuples for tuple fields
        for k in ("command_range_vx", "command_range_vy", "command_range_yaw"):
            if k in meta_dict and isinstance(meta_dict[k], list):
                meta_dict[k] = tuple(meta_dict[k])
        meta = ReferenceLibraryMeta(**meta_dict)

        lib = cls(meta=meta)
        for npz_path in sorted(src.glob("entry_*.npz")):
            data = np.load(npz_path, allow_pickle=True)
            scalars = json.loads(str(data["_scalars_json"]))
            kwargs = {
                "command_vx": scalars["command_vx"],
                "command_vy": scalars["command_vy"],
                "command_yaw_rate": scalars["command_yaw_rate"],
                "dt": scalars["dt"],
                "cycle_time": scalars["cycle_time"],
                "generator_version": scalars.get("generator_version", "unset"),
                "is_valid": scalars.get("is_valid", True),
                "validation_notes": scalars.get("validation_notes", ""),
            }
            for arr_name in (
                "q_ref", "dq_ref", "pelvis_pos", "pelvis_rpy", "com_pos",
                "left_foot_pos", "left_foot_rpy", "right_foot_pos",
                "right_foot_rpy", "stance_foot_id", "contact_mask", "phase",
            ):
                if arr_name in data:
                    kwargs[arr_name] = data[arr_name]

            lib.add(ReferenceTrajectory(**kwargs))

        return lib

    # -- Summary --------------------------------------------------------------

    def summary(self) -> str:
        """Human-readable library summary."""
        lines = [
            f"ReferenceLibrary: {len(self)} entries",
            f"  generator: {self.meta.generator} v{self.meta.generator_version}",
            f"  robot: {self.meta.robot}",
            f"  dt={self.meta.dt}s  cycle_time={self.meta.cycle_time}s",
            f"  command_range_vx: {self.meta.command_range_vx}",
            f"  command_range_vy: {self.meta.command_range_vy}",
            f"  command_range_yaw: {self.meta.command_range_yaw}",
            f"  interval: {self.meta.command_interval}",
        ]
        for key in self.command_keys:
            traj = self._entries[key]
            tag = "OK" if traj.is_valid else "INVALID"
            lines.append(
                f"  [{tag}] vx={key[0]:+.3f} vy={key[1]:+.3f} "
                f"yaw={key[2]:+.3f}  steps={traj.n_steps}"
            )
        return "\n".join(lines)
