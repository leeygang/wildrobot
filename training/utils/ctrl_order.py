"""Unified ctrl-ordering bridge between PolicySpec and MuJoCo model.

PolicySpec (from mujoco_robot_config.json) and MuJoCo (from XML) may list
actuators in different orders.  This module provides a single API for
converting ctrl arrays between the two orderings.

Usage:
    mapper = CtrlOrderMapper(mj_model, policy_spec)

    # JAX path (env):  PolicySpec → MuJoCo
    mj_ctrl = mapper.to_mj_jax(ctrl_policy_order)

    # NumPy path (viewer):  PolicySpec → MuJoCo
    mj_ctrl = mapper.to_mj_np(ctrl_policy_order)

    # Set individual actuator by name (viewer convenience)
    mapper.set_ctrl_by_name(mj_data, "left_knee_pitch", 0.99)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import numpy as np

if TYPE_CHECKING:
    import jax


class CtrlOrderMapper:
    """Maps ctrl arrays between PolicySpec order and MuJoCo model order.

    All internal training arrays (nominal_q_ref, _default_joint_qpos,
    action_to_ctrl output) use PolicySpec order.  MuJoCo's ``data.ctrl``
    uses model order (the order actuators appear in the XML).  This class
    bridges the two.
    """

    def __init__(self, mj_model: mujoco.MjModel, actuator_names: list[str]) -> None:
        """Build the permutation mapping.

        Args:
            mj_model: MuJoCo model.
            actuator_names: Actuator names in PolicySpec order
                (e.g. ``policy_spec.robot.actuator_names``).

        Raises:
            ValueError: If any name is not found in the MuJoCo model.
        """
        self._nu = mj_model.nu
        self._mj_model = mj_model
        self._actuator_names = list(actuator_names)

        # Build permutation: policy_idx → mj_ctrl_idx
        perm = []
        for name in actuator_names:
            mj_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if mj_id < 0:
                raise ValueError(
                    f"Actuator '{name}' from PolicySpec not found in MuJoCo model. "
                    f"Check that mujoco_robot_config.json actuated_joints match the MJCF."
                )
            perm.append(mj_id)
        self._perm_np = np.asarray(perm, dtype=np.intp)

        # Lazy JAX array — created on first use to avoid importing jax at
        # module level for pure-MuJoCo callers (viewer).
        self._perm_jax = None

    @property
    def actuator_names(self) -> list[str]:
        """Actuator names in PolicySpec order (read-only view)."""
        return list(self._actuator_names)

    @property
    def policy_to_mj_order(self) -> np.ndarray:
        """Permutation array (numpy): policy_idx → mj_ctrl_idx."""
        return self._perm_np

    @property
    def policy_to_mj_order_jax(self) -> "jax.Array":
        """Permutation array (JAX): policy_idx → mj_ctrl_idx."""
        if self._perm_jax is None:
            import jax.numpy as jp
            self._perm_jax = jp.asarray(self._perm_np, dtype=jp.int32)
        return self._perm_jax

    # ── NumPy path (viewer, eval scripts) ──────────────────────────────

    def to_mj_np(self, ctrl_policy_order: np.ndarray) -> np.ndarray:
        """Permute ctrl from PolicySpec order to MuJoCo model order (numpy)."""
        out = np.zeros(self._nu, dtype=ctrl_policy_order.dtype)
        out[self._perm_np] = ctrl_policy_order
        return out

    def set_ctrl_by_name(
        self, mj_data: mujoco.MjData, name: str, value: float
    ) -> None:
        """Set a single actuator's ctrl by name."""
        mj_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if mj_id < 0:
            raise ValueError(f"Actuator '{name}' not found in MuJoCo model")
        mj_data.ctrl[mj_id] = value

    def set_all_ctrl(
        self, mj_data: mujoco.MjData, ctrl_policy_order: np.ndarray
    ) -> None:
        """Set all actuator ctrl values from a PolicySpec-ordered array."""
        mj_data.ctrl[:] = self.to_mj_np(ctrl_policy_order)

    def set_ctrl_from_qpos(self, mj_data: mujoco.MjData) -> None:
        """Set each actuator's ctrl to match its current joint qpos."""
        for i in range(self._mj_model.nu):
            name = self._mj_model.actuator(i).name
            jid = mujoco.mj_name2id(
                self._mj_model, mujoco.mjtObj.mjOBJ_JOINT, name
            )
            if jid >= 0:
                mj_data.ctrl[i] = mj_data.qpos[int(self._mj_model.jnt_qposadr[jid])]

    # ── JAX path (env) ─────────────────────────────────────────────────

    def to_mj_jax(self, ctrl_policy_order: "jax.Array") -> "jax.Array":
        """Permute ctrl from PolicySpec order to MuJoCo model order (JAX)."""
        import jax.numpy as jp
        return jp.zeros(self._nu, dtype=ctrl_policy_order.dtype).at[
            self.policy_to_mj_order_jax
        ].set(ctrl_policy_order)
