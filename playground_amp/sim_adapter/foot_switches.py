from __future__ import annotations

from typing import Iterable

import mujoco
import numpy as np


def resolve_foot_geom_ids(
    mj_model,
    foot_geom_names: Iterable[str],
) -> tuple[int, int, int, int]:
    geom_ids = []
    for name in foot_geom_names:
        geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id < 0:
            raise ValueError(f"Geom '{name}' not found in MJCF")
        geom_ids.append(int(geom_id))
    return tuple(geom_ids)  # type: ignore[return-value]


def switches_from_forces_np(forces: np.ndarray, threshold: float) -> np.ndarray:
    return (forces > threshold).astype(np.float32)


def switches_from_forces_jax(forces, threshold: float):
    import jax.numpy as jnp

    return jnp.where(forces > threshold, 1.0, 0.0).astype(jnp.float32)


def contact_forces_from_mujoco(
    mj_model,
    mj_data,
    geom_ids: tuple[int, int, int, int],
    contact_force: np.ndarray,
) -> np.ndarray:
    forces = np.zeros((4,), dtype=np.float32)
    for i in range(int(mj_data.ncon)):
        con = mj_data.contact[i]
        g1 = int(con.geom1)
        g2 = int(con.geom2)
        idx = -1
        if g1 in geom_ids:
            idx = geom_ids.index(g1)
        elif g2 in geom_ids:
            idx = geom_ids.index(g2)
        if idx < 0:
            continue
        mujoco.mj_contactForce(mj_model, mj_data, i, contact_force)
        forces[idx] += float(abs(contact_force[0]))
    return forces


def contact_forces_from_mjx(data, geom_ids: tuple[int, int, int, int]):
    import jax.numpy as jnp

    if hasattr(data, "_impl"):
        contact = data._impl.contact
        efc_force = data._impl.efc_force
    else:
        contact = data.contact
        efc_force = data.efc_force

    forces = []
    for geom_id in geom_ids:
        geom1_match = contact.geom1 == geom_id
        geom2_match = contact.geom2 == geom_id
        is_our_contact = geom1_match | geom2_match
        normal_forces = efc_force[contact.efc_address]
        our_forces = jnp.where(is_our_contact, jnp.abs(normal_forces), 0.0)
        forces.append(jnp.sum(our_forces))

    return jnp.stack(forces)
