"""Runtime gait phase clock service.

Serves ``loc_ref_phase_sin_cos`` for the ``wr_obs_v8_cmd3d`` observation from
the bundled (bin-independent) phase table.  Mirrors the env / V6EvalAdapter
contract:

  - nearest-bin selection by L2 distance over the (vx, vy, wz) command keys
    (``WildRobotEnv._lookup_offline_window`` 3D path / ``V6EvalAdapter._select_bin_idx``)
  - per-bin absorbing-boundary clamp: ``step_idx`` is clamped to that bin's
    ``n_steps - 1`` before reading the phase
    (``RuntimeReferenceService.lookup_np``'s internal ``np.clip``)

Under the home-base residual contract the ONLY reference-derived obs channel is
``loc_ref_phase_sin_cos`` (q_ref / foot / pelvis channels are not in v8), so a
phase clock is the entire reference service the runtime needs.
"""

from __future__ import annotations

import numpy as np

from .runtime_policy_config import ReferencePhaseTable


class ReferencePhaseService:
    def __init__(self, table: ReferencePhaseTable) -> None:
        self._table = table

    @property
    def n_steps(self) -> int:
        """Global step-index advance bound (max across bins)."""
        return int(self._table.n_steps)

    def select_bin(self, velocity_cmd: np.ndarray) -> int:
        """Nearest-bin index by L2 distance over (vx, vy, wz) cmd keys."""
        cmd = _as_three_vec(velocity_cmd)
        diffs = self._table.cmd_keys - cmd[np.newaxis, :]
        return int(np.argmin(np.linalg.norm(diffs, axis=-1)))

    def phase_sin_cos(self, *, bin_idx: int, step_idx: int) -> np.ndarray:
        """Return ``[phase_sin, phase_cos]`` at the clamped step index.

        Clamps to the selected bin's ``n_steps - 1`` (absorbing boundary), then
        to the shared phase-array bound for safety.
        """
        bin_n = int(self._table.per_bin_n_steps[bin_idx])
        idx = min(max(int(step_idx), 0), max(bin_n - 1, 0))
        idx = min(idx, self._table.phase_sin.shape[0] - 1)
        return np.array(
            [float(self._table.phase_sin[idx]), float(self._table.phase_cos[idx])],
            dtype=np.float32,
        )


def _as_three_vec(velocity_cmd: np.ndarray) -> np.ndarray:
    cmd = np.asarray(velocity_cmd, dtype=np.float32).reshape(-1)
    if cmd.size == 1:
        return np.array([float(cmd[0]), 0.0, 0.0], dtype=np.float32)
    if cmd.size == 3:
        return cmd.astype(np.float32)
    raise ValueError(
        f"velocity_cmd must be scalar or length-3 (vx, vy, wz); got size {cmd.size}"
    )
