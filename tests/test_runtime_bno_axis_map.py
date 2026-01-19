import numpy as np
import pytest


def test_axis_map_to_matrix_identity() -> None:
    from runtime.wr_runtime.hardware.bno085 import _axis_map_to_r_bs

    r = _axis_map_to_r_bs(["+X", "+Y", "+Z"])
    np.testing.assert_allclose(r, np.eye(3, dtype=np.float32))


def test_axis_map_to_matrix_rejects_duplicates() -> None:
    from runtime.wr_runtime.hardware.bno085 import _axis_map_to_r_bs

    with pytest.raises(ValueError):
        _axis_map_to_r_bs(["+X", "-X", "+Z"])


def test_axis_map_to_matrix_rejects_bad_format() -> None:
    from runtime.wr_runtime.hardware.bno085 import _axis_map_to_r_bs

    with pytest.raises(ValueError):
        _axis_map_to_r_bs(["X", "Y", "Z"])

