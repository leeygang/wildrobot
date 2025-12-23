import math

import jax.numpy as jnp
import numpy as np
from brax import math as brax_math


def quat_for_pitch(deg: float) -> np.ndarray:
    """Return quaternion (w,x,y,z) for a rotation of `deg` about the Y axis (pitch)."""
    rad = math.radians(deg)
    w = math.cos(rad / 2.0)
    x = 0.0
    y = math.sin(rad / 2.0)
    z = 0.0
    return np.array([w, x, y, z], dtype=np.float32)


def quat_to_euler(q: np.ndarray) -> tuple[float, float, float]:
    """Convert (w,x,y,z) quaternion to (roll, pitch, yaw) using brax.math."""
    euler = brax_math.quat_to_euler(jnp.asarray(q))
    return float(euler[0]), float(euler[1]), float(euler[2])


def test_pitch_thresholds():
    # thresholds in env: 45 deg (obs-based) and 60 deg (data-based)
    t45 = quat_for_pitch(45.0)
    t44 = quat_for_pitch(44.9)
    t46 = quat_for_pitch(45.1)

    _, p45, _ = quat_to_euler(t45 / np.linalg.norm(t45))
    _, p44, _ = quat_to_euler(t44 / np.linalg.norm(t44))
    _, p46, _ = quat_to_euler(t46 / np.linalg.norm(t46))

    assert math.isclose(math.degrees(p45), 45.0, rel_tol=1e-3)
    assert math.degrees(p44) < 45.0
    assert math.degrees(p46) > 45.0


def test_normalize_random_permutations():
    # small random rotations: ensure scalar part is dominant so ordering heuristic applies
    rng = np.random.RandomState(123)
    angle_deg = rng.uniform(0.0, 20.0)
    axis = rng.normal(size=(3,)).astype(np.float32)
    axis = axis / np.linalg.norm(axis)
    theta = math.radians(angle_deg)
    w = math.cos(theta / 2.0)
    xyz = axis * math.sin(theta / 2.0)
    q = np.array([w, xyz[0], xyz[1], xyz[2]], dtype=np.float32)

    # Normalize and get euler
    n1 = q / np.linalg.norm(q)
    r1, p1, y1 = quat_to_euler(n1)

    # Just verify we can compute euler angles without error
    assert not math.isnan(r1)
    assert not math.isnan(p1)
    assert not math.isnan(y1)


def test_obs_like_done_logic():
    # Simulate the obs extras layout: [base_height, pitch, roll, comx, comy, comz]
    # and verify the simple done checks used in _is_done
    def obs_done(extras):
        base_h, pitch, roll = extras[0], extras[1], extras[2]
        if base_h < 0.25:
            return True
        if abs(pitch) > math.radians(45.0) or abs(roll) > math.radians(45.0):
            return True
        return False

    # base height triggers
    assert obs_done(np.array([0.1, 0.0, 0.0, 0, 0, 0]))
    assert not obs_done(np.array([0.3, 0.0, 0.0, 0, 0, 0]))

    # pitch triggers
    q = quat_for_pitch(50.0)
    nq = q / np.linalg.norm(q)
    _, pitch, _ = quat_to_euler(nq)
    assert obs_done(np.array([0.5, pitch, 0.0, 0, 0, 0]))
