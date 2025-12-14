"""Quaternion helpers used by the WildRobot env.

Provides a small, well-documented normalization + conversion utility so
tests can validate expected ordering (w,x,y,z) vs alternative (x,y,z,w).
"""
from __future__ import annotations

import math
import numpy as np


def normalize_quat_wxyz(q: np.ndarray) -> np.ndarray:
    """Return a (w,x,y,z) quaternion normalized for downstream use.

    Accepts arrays in either (w,x,y,z) or (x,y,z,w) ordering and returns a 1-D
    array in (w,x,y,z) ordering. Inputs may be 1-D length-4 or a 2-D array
    (N,4) in which case the first row is used.
    The heuristic used is conservative: if the last component is substantially
    larger than the first, treat the input as (x,y,z,w) and rotate it.
    """
    a = np.asarray(q, dtype=np.float32)
    if a.ndim == 2:
        a = a[0, :4]
    if a.size < 4:
        raise ValueError("Quaternion must have 4 components")
    # copy and cast
    qv = a[:4].astype(np.float32)
    # heuristic: if last element looks like scalar part, rotate xyzw->wxyz
    if abs(qv[3]) > (abs(qv[0]) + 0.3):
        qv = np.array([qv[3], qv[0], qv[1], qv[2]], dtype=np.float32)
    return qv


def quat_to_euler_wxyz(q: np.ndarray) -> tuple[float, float]:
    """Convert (w,x,y,z) quaternion to (roll, pitch) in radians.

    Only roll and pitch are returned because those are used for tilt-based
    termination checks in the environment.
    """
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)
    return roll, pitch
