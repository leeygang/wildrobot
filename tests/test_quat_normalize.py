import numpy as np
from playground_amp.utils.quaternion import normalize_quat_wxyz, quat_to_euler_wxyz


def test_normalize_equivalence_wxyz_xyzw():
    # Example quaternion representing large pitch (w,x,y,z)
    q_wxyz = np.array([0.7071, 0.0, 0.7071, 0.0], dtype=np.float32)
    # Same rotation expressed as (x,y,z,w)
    q_xyzw = np.array([0.0, 0.7071, 0.0, 0.7071], dtype=np.float32)

    nq1 = normalize_quat_wxyz(q_wxyz)
    nq2 = normalize_quat_wxyz(q_xyzw)

    # normalized results should be approximately equal
    assert np.allclose(nq1, nq2, atol=1e-4)

    # converted Euler angles should indicate a large pitch ( > 60 deg )
    _, pitch = quat_to_euler_wxyz(nq1)
    assert abs(pitch) > np.deg2rad(60.0)
