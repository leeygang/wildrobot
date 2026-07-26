from __future__ import annotations

import mujoco
import numpy as np
import pytest
from mujoco import mjx

from training.sim_adapter.foot_switches import (
    contact_forces_from_mujoco,
    contact_forces_from_mjx,
    resolve_foot_geom_ids,
)


def test_mjx_pyramidal_normal_forces_match_mujoco_and_robot_weight(
    mj_model,
    robot_config,
) -> None:
    assert int(mj_model.opt.cone) == int(mujoco.mjtCone.mjCONE_PYRAMIDAL)

    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
    for _ in range(1000):
        mujoco.mj_step(mj_model, mj_data)

    geom_ids = resolve_foot_geom_ids(
        mj_model,
        robot_config.get_foot_geom_names(),
    )
    native_forces = contact_forces_from_mujoco(
        mj_model,
        mj_data,
        geom_ids,
        np.zeros(6, dtype=np.float64),
    )
    mjx_forces = np.asarray(
        contact_forces_from_mjx(
            mjx.put_data(mj_model, mj_data),
            geom_ids,
            int(mj_model.opt.cone),
        )
    )

    np.testing.assert_allclose(mjx_forces, native_forces, rtol=1e-5, atol=1e-5)
    robot_weight_n = float(np.sum(mj_model.body_mass) * abs(mj_model.opt.gravity[2]))
    assert float(np.sum(mjx_forces)) == pytest.approx(robot_weight_n, rel=0.02)
