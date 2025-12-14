import numpy as np
import math
from playground_amp.utils.quaternion import normalize_quat_wxyz, quat_to_euler_wxyz


def quat_for_pitch(deg: float) -> np.ndarray:
    """Return quaternion (w,x,y,z) for a rotation of `deg` about the Y axis (pitch)."""
    rad = math.radians(deg)
    w = math.cos(rad / 2.0)
    x = 0.0
    y = math.sin(rad / 2.0)
    z = 0.0
    return np.array([w, x, y, z], dtype=np.float32)


def test_pitch_thresholds():
    # thresholds in env: 45 deg (obs-based) and 60 deg (data-based)
    t45 = quat_for_pitch(45.0)
    t44 = quat_for_pitch(44.9)
    t46 = quat_for_pitch(45.1)

    _, p45 = quat_to_euler_wxyz(normalize_quat_wxyz(t45))
    _, p44 = quat_to_euler_wxyz(normalize_quat_wxyz(t44))
    _, p46 = quat_to_euler_wxyz(normalize_quat_wxyz(t46))

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
    q_xyzw = np.array([q[1], q[2], q[3], q[0]], dtype=np.float32)

    n1 = normalize_quat_wxyz(q)
    n2 = normalize_quat_wxyz(q_xyzw)
    assert np.allclose(n1, n2, atol=1e-6)


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
    _, pitch = quat_to_euler_wxyz(normalize_quat_wxyz(q))
    assert obs_done(np.array([0.5, pitch, 0.0, 0, 0, 0]))
