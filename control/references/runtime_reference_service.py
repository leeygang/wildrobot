"""Runtime reference service (Layer 2) — v0.20.1 smoke.

Thin, JAX-jittable lookup over a single offline ``ReferenceTrajectory``.
Pre-stacks the trajectory's per-step arrays at construction time so the
per-step lookup inside an env's JIT'd step function is just integer
indexing + clamp.

Scope and design notes
----------------------

- **Single command per service instance.**  The v0.20.1 smoke pins one
  q_ref operating point, so we deliberately skip multi-command
  interpolation here.  Use one ``RuntimeReferenceService`` per
  command bin and select externally when true command-conditioned
  reference lookup lands.

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
# JAX quaternion helpers (NEW-3 / C11)
# ---------------------------------------------------------------------------


def _yaw_from_quat_wxyz(quat_wxyz):
    """Extract yaw (rad) from a wxyz quaternion.

    Pure-JAX implementation: no ``scipy.spatial.transform`` (the v0.21.0
    C11 invariant — heading-frame correction in the env must stay
    JIT-traceable).  Mirrors
    ``policy_contract.jax.frames._yaw_from_quat_xyzw`` with the index
    layout shifted for the wxyz convention used by the offline
    ``RuntimeReferenceService`` / MuJoCo root quat.

    Returns a scalar ``jnp.float32`` (or whatever dtype the input
    promotes to).  The standard ZYX-Euler yaw formula::

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw       = atan2(siny_cosp, cosy_cosp)
    """
    import jax.numpy as jnp

    w = quat_wxyz[0]
    x = quat_wxyz[1]
    y = quat_wxyz[2]
    z = quat_wxyz[3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return jnp.arctan2(siny_cosp, cosy_cosp)


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

    # -- Realized FK reference (Phase 3 / TB-style mujoco_replay) -----------
    # Sourced from ``ReferenceTrajectory.body_pos / .body_quat / .site_pos``
    # populated by ``ZMPWalkGenerator._fixed_base_fk_replay``.  Shape is
    # ``[n_bodies, 3]`` etc per-frame; consumers index by body / site
    # name via the parent service's ``body_names`` / ``site_names``
    # tuples (stable across the trajectory).  Empty arrays when the
    # source trajectory predates Phase 3.
    body_pos: np.ndarray              # [n_bodies, 3]
    body_quat: np.ndarray             # [n_bodies, 4] wxyz
    body_lin_vel: np.ndarray          # [n_bodies, 3]
    body_ang_vel: np.ndarray          # [n_bodies, 3]
    site_pos: np.ndarray              # [n_sites, 3]

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
    # Realized FK arrays (Phase 3); shape ``[n_steps, n_bodies, 3]``
    # etc.  When the source trajectory lacks them (legacy assets), the
    # service synthesises empty per-step slices so the lookup return
    # shape stays stable at ``[0, 3]`` rather than raising.
    body_pos: np.ndarray           # [n_steps, n_bodies, 3]
    body_quat: np.ndarray          # [n_steps, n_bodies, 4]
    body_lin_vel: np.ndarray       # [n_steps, n_bodies, 3]
    body_ang_vel: np.ndarray       # [n_steps, n_bodies, 3]
    site_pos: np.ndarray           # [n_steps, n_sites, 3]
    n_steps: int
    n_joints: int
    n_bodies: int
    n_sites: int


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class RuntimeReferenceService:
    """JAX-jittable per-step lookup over a single reference trajectory.

    Construction is heavy (pre-stack + finite-diff); per-step lookup
    is integer indexing only.  Designed for fixed-q_ref env use
    (one service per command bin; the smoke pins one operating point)."""

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
        # Phase 3: stable name tuples for body / site indexing.  Empty
        # tuple when the source trajectory predates Phase 3 and didn't
        # supply names; downstream consumers can detect "no realized
        # FK data" by checking ``len(service.body_names) == 0``.
        self.body_names: tuple = (
            tuple(trajectory.body_names) if trajectory.body_names is not None else ()
        )
        self.site_names: tuple = (
            tuple(trajectory.site_names) if trajectory.site_names is not None else ()
        )
        self._stacked = self._build_stack(trajectory)
        # One-shot terminal-clamp warning: caller responsibility is
        # to size ``episode_horizon <= n_steps``; if a query lands
        # past the end, surface it once per service instance so a
        # misconfigured horizon shows up in logs instead of being
        # masked by the silent clamp.
        self._warned_overrun: bool = False

    # -- construction ---------------------------------------------------------

    @staticmethod
    def compute_command_integrated_path_state(
        *,
        t_since_reset_s,
        velocity_cmd_mps,
        velocity_cmd_y_mps=0.0,
        yaw_rate_cmd_rps=0.0,
        dt_s=0.02,
        default_root_pos_xyz=None,
    ) -> dict:
        """TB-style runtime path integration for constant command inputs.

        Mirrors ``toddlerbot.reference.motion_ref.MotionReference``
        ``integrate_path_state`` semantics for the v0.21 (vx, vy, wz)
        command space: local linear velocity ``[vx, vy, 0]`` and yaw
        rate ``[0, 0, wz]``.

        The helper is stateless and JAX-friendly: it reconstructs the
        path state from elapsed time plus command, so env code can
        consume a command-integrated torso/path target without mutating
        service state.  If ``default_root_pos_xyz`` is provided, the
        returned ``torso_pos`` matches TB's absolute-world semantics:
        ``path_rot.apply(default_root_pos_xyz) + path_pos``.

        v0.21.0 / NEW-5 extension:

          - ``velocity_cmd_y_mps`` (default ``0.0``) adds the local
            lateral component.  Closed-form 2D path sum:

                p_n = dt * Σ_{k=1..n} Rz(k * θ) · [vx, vy]

            with ``θ = wz * dt``.  Expanding gives

                C(n) = Σ_{k=1..n} cos(k * θ)
                S(n) = Σ_{k=1..n} sin(k * θ)
                p_n.x = dt * (vx * C(n) - vy * S(n))
                p_n.y = dt * (vx * S(n) + vy * C(n))

            and the half-angle closed form

                ratio = sin(n * θ / 2) / sin(θ / 2)
                C(n)  = ratio * cos((n + 1) * θ / 2)
                S(n)  = ratio * sin((n + 1) * θ / 2)

            with a ``θ -> 0`` fallback ``C(n) -> n``, ``S(n) -> 0``.
          - The closed form preserves backward-compat for the
            ``velocity_cmd_y_mps == 0`` (vx-only) case: ``S(n)`` term
            drops out and ``p_n.x = vx * dt * C(n)``,
            ``p_n.y = vx * dt * S(n)`` — identical to the pre-P3 form.

        ``lin_vel`` reports the instantaneous *body-frame* lin vel
        (matches TB ``integrate_path_state`` where ``lin_vel`` is
        reported pre-rotation), and ``ang_vel = [0, 0, wz]``.
        """
        import jax.numpy as jnp

        t = jnp.asarray(t_since_reset_s, dtype=jnp.float32)
        vx = jnp.asarray(velocity_cmd_mps, dtype=jnp.float32)
        vy = jnp.asarray(velocity_cmd_y_mps, dtype=jnp.float32)
        yaw_rate = jnp.asarray(yaw_rate_cmd_rps, dtype=jnp.float32)
        dt = jnp.maximum(jnp.asarray(dt_s, dtype=jnp.float32), jnp.float32(1e-8))

        # Step-count view of elapsed time keeps parity with the env's
        # discrete ctrl updates (t_since_reset = step_idx * dt).
        n_steps = jnp.maximum(jnp.rint(t / dt).astype(jnp.int32), 0)
        n = n_steps.astype(jnp.float32)
        theta = yaw_rate * dt

        half_theta = 0.5 * theta
        sin_half = jnp.sin(half_theta)
        sin_half_safe = jnp.where(
            jnp.abs(sin_half) > jnp.float32(1e-6),
            sin_half,
            jnp.float32(1.0),
        )
        ratio_closed = jnp.sin(0.5 * n * theta) / sin_half_safe
        ratio = jnp.where(
            jnp.abs(sin_half) > jnp.float32(1e-6), ratio_closed, n
        )
        C = ratio * jnp.cos(0.5 * (n + 1.0) * theta)
        S_ = ratio * jnp.sin(0.5 * (n + 1.0) * theta)
        # theta -> 0 fallback for S explicitly:
        #   cos((n+1)*0/2)=1, sin(...)=0; ratio collapses to n.
        S_ = jnp.where(
            jnp.abs(sin_half) > jnp.float32(1e-6),
            S_,
            jnp.float32(0.0),
        )

        px = dt * (vx * C - vy * S_)
        py = dt * (vx * S_ + vy * C)
        path_pos = jnp.asarray([px, py, jnp.float32(0.0)], dtype=jnp.float32)

        yaw = n * theta
        half_yaw = 0.5 * yaw
        yaw_cos = jnp.cos(yaw)
        yaw_sin = jnp.sin(yaw)
        path_rot = jnp.asarray(
            [jnp.cos(half_yaw), 0.0, 0.0, jnp.sin(half_yaw)], dtype=jnp.float32
        )  # wxyz
        default_root = (
            jnp.asarray(default_root_pos_xyz, dtype=jnp.float32)
            if default_root_pos_xyz is not None
            else jnp.zeros((3,), dtype=jnp.float32)
        )
        default_root_rotated = jnp.asarray(
            [
                yaw_cos * default_root[0] - yaw_sin * default_root[1],
                yaw_sin * default_root[0] + yaw_cos * default_root[1],
                default_root[2],
            ],
            dtype=jnp.float32,
        )
        torso_pos = (default_root_rotated + path_pos).astype(jnp.float32)
        lin_vel = jnp.asarray([vx, vy, 0.0], dtype=jnp.float32)
        ang_vel = jnp.asarray([0.0, 0.0, yaw_rate], dtype=jnp.float32)
        return {
            "path_pos": path_pos,
            "path_rot": path_rot,
            "torso_pos": torso_pos,
            "lin_vel": lin_vel,
            "ang_vel": ang_vel,
        }

    @staticmethod
    def incremental_path_state_step(
        *,
        prev_torso_pos,
        prev_path_rot_wxyz,
        velocity_cmd,
        dt_s,
    ):
        """One-step incremental path-state update (H7).

        Mirrors TB ``toddlerbot/reference/motion_ref.py::integrate_path_state``::

            delta_rot      = axis_angle_quat([0, 0, 1], wz * dt)  # wxyz
            path_rot       = quat_mul(prev_path_rot, delta_rot)   # wxyz
            lin_vel_world  = rotate_vec_by_quat([vx, vy, 0], path_rot)
            torso_pos      = prev_torso_pos + lin_vel_world * dt

        Used at runtime under ``cmd_resample_steps > 0`` where the
        closed-form integrator drifts because the cmd changes
        mid-episode (the closed form integrates the *new* cmd back
        from t=0).  The closed-form helper stays for unit tests
        (it's the analytic ground truth for constant cmd) and for
        library-build code paths.
        """
        import jax.numpy as jnp

        cmd = jnp.asarray(velocity_cmd, dtype=jnp.float32)
        dt = jnp.asarray(dt_s, dtype=jnp.float32)
        prev_pos = jnp.asarray(prev_torso_pos, dtype=jnp.float32)
        prev_rot = jnp.asarray(prev_path_rot_wxyz, dtype=jnp.float32)
        vx, vy, wz = cmd[0], cmd[1], cmd[2]
        half = 0.5 * wz * dt
        delta_rot = jnp.asarray(
            [jnp.cos(half), 0.0, 0.0, jnp.sin(half)], dtype=jnp.float32,
        )  # wxyz
        # quat_mul (wxyz):  (w1, v1) * (w2, v2)
        #   w = w1*w2 - v1 . v2
        #   v = w1*v2 + w2*v1 + v1 x v2
        w1, x1, y1, z1 = prev_rot[0], prev_rot[1], prev_rot[2], prev_rot[3]
        w2, x2, y2, z2 = (
            delta_rot[0], delta_rot[1], delta_rot[2], delta_rot[3]
        )
        new_w = w1 * w2 - (x1 * x2 + y1 * y2 + z1 * z2)
        new_x = w1 * x2 + w2 * x1 + (y1 * z2 - z1 * y2)
        new_y = w1 * y2 + w2 * y1 + (z1 * x2 - x1 * z2)
        new_z = w1 * z2 + w2 * z1 + (x1 * y2 - y1 * x2)
        new_rot = jnp.stack([new_w, new_x, new_y, new_z]).astype(jnp.float32)
        # rotate body-frame [vx, vy, 0] by new_rot (wxyz).  The
        # yaw-only (z-axis) restriction collapses the q * v * q^{-1}
        # expansion to a 2D rotation in xy:
        #   yaw = 2 * atan2(new_z, new_w)   (axis = +z)
        yaw = 2.0 * jnp.arctan2(new_z, new_w)
        c, s = jnp.cos(yaw), jnp.sin(yaw)
        vw_x = c * vx - s * vy
        vw_y = s * vx + c * vy
        lin_vel_world = jnp.stack([vw_x, vw_y, jnp.float32(0.0)])
        new_pos = (prev_pos + lin_vel_world * dt).astype(jnp.float32)
        return new_pos, new_rot

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

        # Phase 3 realized FK arrays (mujoco_replay output).  When the
        # source trajectory predates Phase 3, fall back to ``[n, 0, 3]``
        # / ``[n, 0, 4]`` shaped empties so the lookup return shape
        # stays well-defined and the service does not silently crash.
        def _take_3d(
            arr: Optional[np.ndarray], second_dim: int, last_dim: int = 3
        ) -> np.ndarray:
            if arr is None:
                return np.zeros((n, second_dim, last_dim), dtype=np.float32)
            return arr.astype(np.float32)

        n_bodies = (
            arr.shape[1] if (arr := traj.body_pos) is not None and arr.ndim == 3 else 0
        )
        n_sites = (
            arr.shape[1] if (arr := traj.site_pos) is not None and arr.ndim == 3 else 0
        )
        body_pos = _take_3d(traj.body_pos, n_bodies, 3)
        body_quat = _take_3d(traj.body_quat, n_bodies, 4)
        body_lin_vel = _take_3d(traj.body_lin_vel, n_bodies, 3)
        body_ang_vel = _take_3d(traj.body_ang_vel, n_bodies, 3)
        site_pos = _take_3d(traj.site_pos, n_sites, 3)

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
            body_pos=body_pos,
            body_quat=body_quat,
            body_lin_vel=body_lin_vel,
            body_ang_vel=body_ang_vel,
            site_pos=site_pos,
            n_steps=n,
            n_joints=n_joints,
            n_bodies=n_bodies,
            n_sites=n_sites,
        )

    # -- properties -----------------------------------------------------------

    @property
    def n_steps(self) -> int:
        return self._stacked.n_steps

    @property
    def n_joints(self) -> int:
        return self._stacked.n_joints

    @property
    def n_bodies(self) -> int:
        return self._stacked.n_bodies

    @property
    def n_sites(self) -> int:
        return self._stacked.n_sites

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
            body_pos=s.body_pos[idx].copy(),
            body_quat=s.body_quat[idx].copy(),
            body_lin_vel=s.body_lin_vel[idx].copy(),
            body_ang_vel=s.body_ang_vel[idx].copy(),
            site_pos=s.site_pos[idx].copy(),
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
            "body_pos":         jax_arrays["body_pos"][idx],
            "body_quat":        jax_arrays["body_quat"][idx],
            "body_lin_vel":     jax_arrays["body_lin_vel"][idx],
            "body_ang_vel":     jax_arrays["body_ang_vel"][idx],
            "site_pos":         jax_arrays["site_pos"][idx],
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
            "body_pos":       jnp.asarray(s.body_pos),
            "body_quat":      jnp.asarray(s.body_quat),
            "body_lin_vel":   jnp.asarray(s.body_lin_vel),
            "body_ang_vel":   jnp.asarray(s.body_ang_vel),
            "site_pos":       jnp.asarray(s.site_pos),
            "n_steps":        s.n_steps,
        }
