"""Runtime reference service (Layer 2) — v0.20.1 smoke.

Thin, JAX-jittable lookup over a single offline ``ReferenceTrajectory``.
Pre-stacks the trajectory's per-step arrays at construction time so the
per-step lookup inside an env's JIT'd step function is just integer
indexing + clamp.

Scope and design notes
----------------------

- **Single command per service instance.**  The v0.20.1 smoke only
  trains at ``vx=0.15``, so we deliberately skip multi-command
  interpolation here.  Use one ``RuntimeReferenceService`` per
  command bin and select externally if/when multi-command training
  resumes (deferred to a later milestone).

- **Window contents** (per
  ``training/docs/walking_training.md`` v0.20.1 Smoke Contract):
  - current reference slice: ``q_ref``, ``pelvis_pos``,
    ``left_foot_pos``, ``right_foot_pos``, ``contact_mask``, ``phase``
    encoded as ``(sin, cos)``, ``stance_foot_id``
  - optional very short future preview: 1-2 future anchor frames of
    the same fields (default: 2 anchors)
  - **no large dense preview window** — explicit per the doc
    (``walking_training.md`` v0.20.1 Observation contract)

- **Velocity reference signals** are computed per ``G2`` decision in
  the doc: finite differences of the position fields, computed once
  at service construction and stored alongside the position arrays.
  Angular velocity defaults to zero (the prior is yaw-stationary).

- **End-of-trajectory clamp** (per the library's multi-cycle
  contract): queries past ``n_steps - 1`` return the terminal frame.
  No wrap to cycle 0.  The future-preview slice also clamps —
  callers responsible for sizing ``episode_horizon <= n_steps``.

- **JAX vs NumPy parity** is preserved by ``lookup_np`` and
  ``lookup_jax`` reading from the same backing arrays.  The unit
  test ``tests/test_runtime_reference_service.py`` asserts
  byte-identical outputs across both paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from control.references.reference_library import ReferenceTrajectory


# ---------------------------------------------------------------------------
# Pre-stacked arrays + window dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RuntimeReferenceWindow:
    """Per-step reference window returned by the runtime service.

    Layout matches the v0.20.1 smoke observation contract; numeric
    types are float32 to keep the env obs path single-precision.
    """

    # -- Current frame --------------------------------------------------------
    q_ref: np.ndarray              # [n_joints]
    phase_sin: float
    phase_cos: float
    stance_foot_id: int            # 0=left, 1=right
    contact_mask: np.ndarray       # [2] (left, right) in {0, 1}

    # -- Current task-space targets (world frame; env converts as needed) ----
    pelvis_pos: np.ndarray         # [3]
    pelvis_vel: np.ndarray         # [3] finite-diff
    left_foot_pos: np.ndarray      # [3]
    right_foot_pos: np.ndarray     # [3]
    left_foot_vel: np.ndarray      # [3] finite-diff
    right_foot_vel: np.ndarray     # [3] finite-diff

    # -- Short future preview (n_anchor frames) ------------------------------
    future_q_ref: np.ndarray              # [n_anchor, n_joints]
    future_phase_sin: np.ndarray          # [n_anchor]
    future_phase_cos: np.ndarray          # [n_anchor]
    future_contact_mask: np.ndarray       # [n_anchor, 2]


@dataclass
class _StackedTrajectory:
    """All per-step arrays pre-stacked at construction.

    Kept as plain NumPy; the JAX lookup converts to ``jnp`` once at
    the call site (not at construction) so this stays usable from
    pure-NumPy code paths (tests, eval, viewer)."""

    q_ref: np.ndarray              # [n_steps, n_joints]
    phase_sin: np.ndarray          # [n_steps]
    phase_cos: np.ndarray          # [n_steps]
    stance_foot_id: np.ndarray     # [n_steps] (int32)
    contact_mask: np.ndarray       # [n_steps, 2]
    pelvis_pos: np.ndarray         # [n_steps, 3]
    pelvis_vel: np.ndarray         # [n_steps, 3]
    left_foot_pos: np.ndarray      # [n_steps, 3]
    right_foot_pos: np.ndarray     # [n_steps, 3]
    left_foot_vel: np.ndarray      # [n_steps, 3]
    right_foot_vel: np.ndarray     # [n_steps, 3]
    n_steps: int
    n_joints: int


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class RuntimeReferenceService:
    """JAX-jittable per-step lookup over a single reference trajectory.

    Construction is heavy (pre-stack + finite-diff); per-step lookup
    is integer indexing only.  Designed for single-command env use
    (one service per command bin; the smoke only needs vx=0.15)."""

    def __init__(
        self,
        trajectory: ReferenceTrajectory,
        n_anchor: int = 2,
    ) -> None:
        if trajectory.q_ref is None:
            raise ValueError(
                "RuntimeReferenceService requires q_ref on the trajectory."
            )
        if n_anchor < 0:
            raise ValueError(f"n_anchor must be >= 0, got {n_anchor}")

        self.command_key = trajectory.command_key
        self.dt = float(trajectory.dt)
        self.cycle_time = float(trajectory.cycle_time)
        self.n_anchor = int(n_anchor)
        self._stacked = self._build_stack(trajectory)
        # One-shot terminal-clamp warning: caller responsibility is
        # to size ``episode_horizon <= n_steps``; if a query lands
        # past the end, surface it once per service instance so a
        # misconfigured horizon shows up in logs instead of being
        # masked by the silent clamp.
        self._warned_overrun: bool = False

    # -- construction ---------------------------------------------------------

    @staticmethod
    def _build_stack(traj: ReferenceTrajectory) -> _StackedTrajectory:
        n = int(traj.q_ref.shape[0])
        n_joints = int(traj.q_ref.shape[1])

        q_ref = traj.q_ref.astype(np.float32)

        phase = (traj.phase if traj.phase is not None
                 else np.zeros(n, dtype=np.float32))
        phase_sin = np.sin(2.0 * np.pi * phase).astype(np.float32)
        phase_cos = np.cos(2.0 * np.pi * phase).astype(np.float32)

        stance = (traj.stance_foot_id.astype(np.int32) if traj.stance_foot_id is not None
                  else np.zeros(n, dtype=np.int32))

        contact = (traj.contact_mask.astype(np.float32) if traj.contact_mask is not None
                   else np.ones((n, 2), dtype=np.float32))

        # Position arrays: keep what the library provides, zero-fill otherwise.
        def _take_pos(arr: Optional[np.ndarray], shape3: tuple) -> np.ndarray:
            if arr is None:
                return np.zeros(shape3, dtype=np.float32)
            return arr.astype(np.float32)

        pelvis_pos = _take_pos(traj.pelvis_pos, (n, 3))
        left_foot_pos = _take_pos(traj.left_foot_pos, (n, 3))
        right_foot_pos = _take_pos(traj.right_foot_pos, (n, 3))

        # Velocity = finite diff (G2 decision).  Last frame copies the
        # previous step's velocity to avoid a discontinuous zero at the
        # terminal index (matches the library's clamp-at-end contract).
        def _fdiff(p: np.ndarray, dt: float) -> np.ndarray:
            v = np.zeros_like(p)
            if p.shape[0] >= 2:
                v[:-1] = (p[1:] - p[:-1]) / dt
                v[-1] = v[-2]
            return v

        pelvis_vel = _fdiff(pelvis_pos, traj.dt)
        left_foot_vel = _fdiff(left_foot_pos, traj.dt)
        right_foot_vel = _fdiff(right_foot_pos, traj.dt)

        return _StackedTrajectory(
            q_ref=q_ref,
            phase_sin=phase_sin,
            phase_cos=phase_cos,
            stance_foot_id=stance,
            contact_mask=contact,
            pelvis_pos=pelvis_pos,
            pelvis_vel=pelvis_vel,
            left_foot_pos=left_foot_pos,
            right_foot_pos=right_foot_pos,
            left_foot_vel=left_foot_vel,
            right_foot_vel=right_foot_vel,
            n_steps=n,
            n_joints=n_joints,
        )

    # -- properties -----------------------------------------------------------

    @property
    def n_steps(self) -> int:
        return self._stacked.n_steps

    @property
    def n_joints(self) -> int:
        return self._stacked.n_joints

    @property
    def stacked(self) -> _StackedTrajectory:
        """Underlying NumPy arrays.  Exposed for the JAX path to convert
        once at env-init time, not on every step."""
        return self._stacked

    # -- NumPy lookup (tests, eval, viewer) ----------------------------------

    def lookup_np(self, step_idx: int) -> RuntimeReferenceWindow:
        """NumPy lookup at ``step_idx`` (clamped to ``[0, n_steps-1]``).

        Future-preview indices are ``step_idx+1, step_idx+2, ...``,
        each clamped to ``n_steps-1``.  Matches the multi-cycle library
        contract (no wrap to cycle 0).

        Emits a one-shot ``RuntimeWarning`` per service instance if
        ``step_idx >= n_steps``; the JAX path can't issue Python
        warnings inside JIT, so this is the only place the overrun
        surfaces.  Callers running JAX-only loops should size their
        episode horizon to ``service.n_steps`` and check ``service``
        out-of-band if they want overrun detection."""
        s = self._stacked
        if step_idx >= s.n_steps and not self._warned_overrun:
            import warnings
            warnings.warn(
                f"RuntimeReferenceService.lookup_np: step_idx={step_idx} "
                f">= n_steps={s.n_steps} for command_key={self.command_key}. "
                f"Returning the terminal frame (no wrap to cycle 0).  "
                f"Ensure episode_horizon <= service.n_steps.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._warned_overrun = True
        idx = int(np.clip(step_idx, 0, s.n_steps - 1))
        future_idx = np.clip(
            np.arange(idx + 1, idx + 1 + self.n_anchor, dtype=np.int32),
            0, s.n_steps - 1,
        )

        return RuntimeReferenceWindow(
            q_ref=s.q_ref[idx].copy(),
            phase_sin=float(s.phase_sin[idx]),
            phase_cos=float(s.phase_cos[idx]),
            stance_foot_id=int(s.stance_foot_id[idx]),
            contact_mask=s.contact_mask[idx].copy(),
            pelvis_pos=s.pelvis_pos[idx].copy(),
            pelvis_vel=s.pelvis_vel[idx].copy(),
            left_foot_pos=s.left_foot_pos[idx].copy(),
            right_foot_pos=s.right_foot_pos[idx].copy(),
            left_foot_vel=s.left_foot_vel[idx].copy(),
            right_foot_vel=s.right_foot_vel[idx].copy(),
            future_q_ref=s.q_ref[future_idx].copy(),
            future_phase_sin=s.phase_sin[future_idx].copy(),
            future_phase_cos=s.phase_cos[future_idx].copy(),
            future_contact_mask=s.contact_mask[future_idx].copy(),
        )

    # -- JAX lookup (env step) -----------------------------------------------

    def lookup_jax(self, step_idx, jax_arrays):
        """JAX-pure lookup.

        ``jax_arrays`` is the dict returned by :meth:`to_jax_arrays`,
        constructed once at env init.  ``step_idx`` is a scalar JAX
        int.  Returns a ``RuntimeReferenceWindow``-shaped dict of JAX
        arrays (not the dataclass — JAX pytree handling is left to
        the env, which can wrap as it likes).

        Caller responsibility:
          - convert backing arrays once via :meth:`to_jax_arrays`
          - thread the resulting dict through their JIT'd step
          - episode horizon must satisfy ``horizon <= n_steps``"""
        import jax.numpy as jnp

        n_steps = jax_arrays["n_steps"]
        n_anchor = self.n_anchor

        idx = jnp.clip(step_idx, 0, n_steps - 1)
        future_offsets = jnp.arange(1, n_anchor + 1, dtype=jnp.int32)
        future_idx = jnp.clip(idx + future_offsets, 0, n_steps - 1)

        return {
            "q_ref":            jax_arrays["q_ref"][idx],
            "phase_sin":        jax_arrays["phase_sin"][idx],
            "phase_cos":        jax_arrays["phase_cos"][idx],
            "stance_foot_id":   jax_arrays["stance_foot_id"][idx],
            "contact_mask":     jax_arrays["contact_mask"][idx],
            "pelvis_pos":       jax_arrays["pelvis_pos"][idx],
            "pelvis_vel":       jax_arrays["pelvis_vel"][idx],
            "left_foot_pos":    jax_arrays["left_foot_pos"][idx],
            "right_foot_pos":   jax_arrays["right_foot_pos"][idx],
            "left_foot_vel":    jax_arrays["left_foot_vel"][idx],
            "right_foot_vel":   jax_arrays["right_foot_vel"][idx],
            "future_q_ref":            jax_arrays["q_ref"][future_idx],
            "future_phase_sin":        jax_arrays["phase_sin"][future_idx],
            "future_phase_cos":        jax_arrays["phase_cos"][future_idx],
            "future_contact_mask":     jax_arrays["contact_mask"][future_idx],
        }

    def to_jax_arrays(self) -> dict:
        """One-shot conversion of the backing stack into JAX device
        arrays.  Call once at env init; thread the result through the
        JIT'd step.  ``n_steps`` is included as a Python int constant
        so the JIT can fold it."""
        import jax.numpy as jnp
        s = self._stacked
        return {
            "q_ref":          jnp.asarray(s.q_ref),
            "phase_sin":      jnp.asarray(s.phase_sin),
            "phase_cos":      jnp.asarray(s.phase_cos),
            "stance_foot_id": jnp.asarray(s.stance_foot_id),
            "contact_mask":   jnp.asarray(s.contact_mask),
            "pelvis_pos":     jnp.asarray(s.pelvis_pos),
            "pelvis_vel":     jnp.asarray(s.pelvis_vel),
            "left_foot_pos":  jnp.asarray(s.left_foot_pos),
            "right_foot_pos": jnp.asarray(s.right_foot_pos),
            "left_foot_vel":  jnp.asarray(s.left_foot_vel),
            "right_foot_vel": jnp.asarray(s.right_foot_vel),
            "n_steps":        s.n_steps,
        }
