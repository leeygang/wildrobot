"""ZMP Preview Control planner (Kajita et al.).

Computes optimal center-of-mass trajectories from desired ZMP reference
points using an LQR formulation over the Linear Inverted Pendulum Model.

Reference
---------
- Kajita et al., "Biped Walking Pattern Generation by using Preview Control
  of ZMP", ICRA 2003.
- Tedrake et al., "A Closed-Form Solution for Real-Time ZMP Gait Generation
  and Feedback Stabilization", Humanoids 2015.
- ToddlerBot ``toddlerbot.algorithms.zmp_planner`` (github.com/hshi74/toddlerbot)

This implementation uses ``scipy.linalg.solve_continuous_are`` for the LQR
solution to avoid an external ``python-control`` dependency.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.linalg import expm, solve_continuous_are

GRAVITY = 9.81


# ---------------------------------------------------------------------------
# Piecewise polynomial helpers
# ---------------------------------------------------------------------------

class PPoly:
    """Piecewise polynomial evaluated at scalar time values."""

    def __init__(self, c: np.ndarray, x: np.ndarray) -> None:
        """
        Parameters
        ----------
        c : array, shape [order, n_segments, n_dims]
            Polynomial coefficients, highest power first.
        x : array, shape [n_segments + 1]
            Breakpoints.
        """
        self.c = c
        self.x = x

    def __call__(self, t: float) -> np.ndarray:
        n_seg = len(self.x) - 1
        idx = int(np.clip(np.searchsorted(self.x, t, side="right") - 1,
                          0, max(0, n_seg - 1)))
        dt = t - self.x[idx]
        c = self.c
        if c.shape[0] == 0:
            c = np.zeros((1,) + c.shape[1:], dtype=c.dtype)
        result = c[0, idx, :]
        for i in range(1, c.shape[0]):
            result = result * dt + c[i, idx, :]
        return result

    def derivative(self, order: int = 1) -> "PPoly":
        if order == 0:
            return self
        new_c = self.c[:-1] * np.arange(self.c.shape[0] - 1, 0, -1)[:, None, None]
        return PPoly(new_c, self.x).derivative(order - 1)


class ExpPlusPPoly:
    """Exponential-plus-polynomial trajectory representation."""

    def __init__(self, K: np.ndarray, A: np.ndarray,
                 alpha: np.ndarray, ppoly: PPoly) -> None:
        self.K = K
        self.A = A
        self.alpha = alpha
        self.ppoly = ppoly

    def value(self, t: float) -> np.ndarray:
        result = self.ppoly(t)
        n_seg = self.alpha.shape[1]
        seg = int(np.clip(np.searchsorted(self.ppoly.x, t, side="right") - 1,
                          0, max(0, n_seg - 1)))
        tj = self.ppoly.x[min(seg, len(self.ppoly.x) - 1)]
        exp_mat = expm(self.A * (t - tj))
        result += (self.K @ exp_mat @ self.alpha[:, seg:seg + 1]).flatten()
        return result

    def derivative(self, order: int) -> "ExpPlusPPoly":
        K_new = self.K.copy()
        for _ in range(order):
            K_new = K_new @ self.A
        return ExpPlusPPoly(K_new, self.A, self.alpha,
                            self.ppoly.derivative(order))


# ---------------------------------------------------------------------------
# LQR solver (replaces python-control dependency)
# ---------------------------------------------------------------------------

def _lqr(A: np.ndarray, B: np.ndarray,
          Q: np.ndarray, R: np.ndarray,
          N: np.ndarray | None = None
          ) -> Tuple[np.ndarray, np.ndarray]:
    """Solve the continuous-time LQR problem.

    Returns (K, S) where K is the gain matrix and S is the solution
    to the algebraic Riccati equation.
    """
    if N is not None:
        # Transform to standard form: Q' = Q - N R^{-1} N^T, A' = A - B R^{-1} N^T
        R_inv = np.linalg.inv(R)
        A_eff = A - B @ R_inv @ N.T
        Q_eff = Q - N @ R_inv @ N.T
    else:
        A_eff = A
        Q_eff = Q
        R_inv = np.linalg.inv(R)

    S = solve_continuous_are(A_eff, B, Q_eff, R)

    if N is not None:
        K = R_inv @ (B.T @ S + N.T)
    else:
        K = R_inv @ B.T @ S

    return K, S


# ---------------------------------------------------------------------------
# ZMP Planner
# ---------------------------------------------------------------------------

class ZMPPlanner:
    """ZMP Preview Control planner.

    Plans optimal COM trajectories by solving an LQR problem over the
    LIPM dynamics, given a sequence of desired ZMP reference points
    (typically footstep midpoints).
    """

    def __init__(self) -> None:
        self.planned = False

    def plan(
        self,
        time_steps: np.ndarray,
        zmp_d: List[np.ndarray],
        x0: np.ndarray,
        com_z: float,
        Qy: np.ndarray | None = None,
        R: np.ndarray | None = None,
    ) -> None:
        """Plan the COM trajectory.

        Parameters
        ----------
        time_steps : array, shape [n_transitions + 1]
            Time breakpoints for ZMP transitions (alternating
            double/single support phase boundaries).
        zmp_d : list of arrays, each shape [2]
            Desired ZMP x,y positions at each transition.
        x0 : array, shape [4]
            Initial state [com_x, com_y, com_vx, com_vy].
        com_z : float
            COM height (constant for LIPM).
        Qy, R : arrays, shape [2, 2]
            LQR cost weights for ZMP tracking and control effort.
        """
        if Qy is None:
            Qy = np.eye(2, dtype=np.float64)
        if R is None:
            R = 1e-1 * np.eye(2, dtype=np.float64)

        self.time_steps = time_steps.astype(np.float64)
        self.zmp_d = [z.astype(np.float64) for z in zmp_d]

        # State-space: x = [com_x, com_y, com_vx, com_vy], u = [com_ax, com_ay]
        A = np.zeros((4, 4), dtype=np.float64)
        A[:2, 2:] = np.eye(2)
        B = np.zeros((4, 2), dtype=np.float64)
        B[2:, :] = np.eye(2)
        self.C = np.zeros((2, 4), dtype=np.float64)
        self.C[:, :2] = np.eye(2)
        self.D = -com_z / GRAVITY * np.eye(2, dtype=np.float64)

        # LQR
        Q1 = self.C.T @ Qy @ self.C
        R1 = R + self.D.T @ Qy @ self.D
        N = self.C.T @ Qy @ self.D
        R1_inv = np.linalg.inv(R1)

        K_gain, S = _lqr(A, B, Q1, R1, N)
        self.K = -K_gain

        # Preview feedforward computation
        NB = N.T + B.T @ S
        A2 = NB.T @ R1_inv @ B.T - A.T
        B2 = 2.0 * (self.C.T - NB.T @ R1_inv @ self.D) @ Qy
        A2_inv = np.linalg.inv(A2)

        zmp_ref_last = self.zmp_d[-1]
        n_seg = len(self.zmp_d) - 1

        alpha = np.zeros((4, n_seg), dtype=np.float64)
        beta = [np.zeros((4, 1), dtype=np.float64) for _ in range(n_seg)]
        gamma = [np.zeros((2, 1), dtype=np.float64) for _ in range(n_seg)]
        c = [np.zeros((2, 1), dtype=np.float64) for _ in range(n_seg)]

        for t in range(n_seg - 1, -1, -1):
            c[t][:, 0] = self.zmp_d[t] - zmp_ref_last
            beta[t][:, 0] = -A2_inv @ B2 @ c[t][:, 0]
            gamma[t][:, 0] = (R1_inv @ self.D @ Qy @ c[t][:, 0]
                              - 0.5 * R1_inv @ B.T @ beta[t][:, 0])

            dt = self.time_steps[t + 1] - self.time_steps[t]
            A2exp = expm(A2 * dt)

            if t == n_seg - 1:
                vec4 = -beta[t]
            else:
                vec4 = alpha[:, t + 1:t + 2] + beta[t + 1] - beta[t]

            alpha[:, t] = (np.linalg.inv(A2exp) @ vec4).squeeze()

        # Build trajectory representations
        all_beta = np.transpose(np.stack(beta, axis=1), (2, 1, 0))
        all_gamma = np.transpose(np.stack(gamma, axis=1), (2, 1, 0))

        beta_traj = PPoly(all_beta, self.time_steps)
        self.s2 = ExpPlusPPoly(np.eye(4, dtype=np.float64), A2, alpha, beta_traj)

        gamma_traj = PPoly(all_gamma, self.time_steps)
        self.k2 = ExpPlusPPoly(-0.5 * R1_inv @ B.T, A2, alpha, gamma_traj)

        # Combined closed-loop + feedforward system
        Az = np.zeros((8, 8), dtype=np.float64)
        Az[:4, :4] = A + B @ self.K
        Az[:4, 4:] = -0.5 * B @ R1_inv @ B.T
        Az[4:, 4:] = A2
        Azi = np.linalg.inv(Az)

        Bz = np.zeros((8, 2), dtype=np.float64)
        Bz[:4, :] = B @ R1_inv @ self.D @ Qy
        Bz[4:, :] = B2

        a = np.zeros((8, n_seg), dtype=np.float64)
        a[4:, :] = alpha

        b = [np.zeros((4, 1), dtype=np.float64) for _ in range(n_seg)]
        i48 = np.zeros((4, 8), dtype=np.float64)
        i48[:, :4] = np.eye(4)

        x = x0.astype(np.float64).copy()
        x[:2] -= zmp_ref_last

        for t in range(n_seg):
            dt = self.time_steps[t + 1] - self.time_steps[t]
            b[t][:, 0] = -Azi[:4, :] @ Bz @ c[t][:, 0]
            a[:4, t] = x - b[t][:, 0]
            Az_exp = expm(Az * dt)
            x = i48 @ Az_exp @ a[:, t] + b[t].squeeze()
            b[t][:2, 0] += zmp_ref_last

        mat28 = np.zeros((2, 8), dtype=np.float64)
        mat28[:, :2] = np.eye(2)
        all_b = np.transpose(np.stack(b, axis=1), (2, 1, 0))
        b_traj = PPoly(all_b[..., :2], self.time_steps)

        self.com_pos = ExpPlusPPoly(mat28, Az, a, b_traj)
        self.com_vel = self.com_pos.derivative(1)
        self.com_acc = self.com_vel.derivative(1)
        self.planned = True

    def get_optim_com_acc(self, time: float, x: np.ndarray) -> np.ndarray:
        """Return optimal COM acceleration at the given time."""
        if not self.planned:
            raise ValueError("Must call plan() first.")
        yf = self.zmp_d[-1]
        x_bar = x.astype(np.float64).copy()
        x_bar[:2] -= yf
        return self.K @ x_bar + self.k2.value(time)

    def get_nominal_com(self, time: float) -> np.ndarray:
        """Return nominal COM position [x, y] at the given time."""
        if not self.planned:
            raise ValueError("Must call plan() first.")
        return self.com_pos.value(time)

    def get_nominal_com_vel(self, time: float) -> np.ndarray:
        """Return nominal COM velocity at the given time."""
        if not self.planned:
            raise ValueError("Must call plan() first.")
        return self.com_vel.value(time)
