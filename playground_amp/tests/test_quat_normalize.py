import math

import jax.numpy as jnp
import numpy as np
from brax import math as brax_math


def test_normalize_equivalence_wxyz_xyzw():
    # Example quaternion representing large pitch (w,x,y,z)
    q_wxyz = np.array([0.7071, 0.0, 0.7071, 0.0], dtype=np.float32)

    # Normalize using simple numpy
    nq = q_wxyz / np.linalg.norm(q_wxyz)

    # converted Euler angles should indicate a large pitch ( > 60 deg )
    euler = brax_math.quat_to_euler(jnp.asarray(nq))
    pitch = float(euler[1])
    assert abs(pitch) > np.deg2rad(60.0)
