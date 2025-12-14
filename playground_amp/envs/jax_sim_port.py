"""Prototype pure-JAX simulator port helpers.

This module provides a tiny, self-contained prototype demonstrating how to
represent simulator `Data` as a JAX pytree and implement a simple jitted
`step_fn` + `scan` pattern. It's a focused starting point for the long-term
pure-JAX migration described in the project plan.

Notes:
- This is a prototype and intentionally small. It implements a synthetic
  integrator for joint positions/velocities to stand in for the full MuJoCo
  dynamics while exercising the data-pytree + jax.jit/vmap workflow.
- Replace the synthetic dynamics with actual MJX-compatible math when ready.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

try:
    import jax
    import jax.numpy as jnp
except Exception:
    jax = None
    jnp = None


@dataclass
class JaxData:
    qpos: Any
    qvel: Any
    ctrl: Any


def tree_flatten(data: JaxData):
    children = (data.qpos, data.qvel, data.ctrl)
    aux = None
    return children, aux


def tree_unflatten(aux, children):
    qpos, qvel, ctrl = children
    return JaxData(qpos=qpos, qvel=qvel, ctrl=ctrl)


if jax is not None:
    jax.tree_util.register_pytree_node(JaxData, tree_flatten, tree_unflatten)


def make_data(nq: int, nv: int, batch: int = 1):
    qpos = jnp.zeros((batch, nq), dtype=jnp.float32)
    qvel = jnp.zeros((batch, nv), dtype=jnp.float32)
    ctrl = jnp.zeros((batch, nv), dtype=jnp.float32)
    return JaxData(qpos=qpos, qvel=qvel, ctrl=ctrl)


def simple_dynamics_step(data: JaxData, dt: float = 0.02):
    """A tiny integrator: qvel += ctrl*dt; qpos += qvel*dt.

    Args:
        data: JaxData (batch, nq)/(batch, nv)...
    Returns:
        new_data: JaxData updated after one substep.
    """
    qvel = data.qvel + data.ctrl * dt
    qpos = data.qpos + qvel * dt
    return JaxData(qpos=qpos, qvel=qvel, ctrl=data.ctrl)


def step_fn(data: JaxData, action: jnp.ndarray, dt: float = 0.02):
    """Apply control `action` (assumed to match nv) and step once.

    This function is JAX-friendly: pure, relies only on JAX arrays, and can
    be jitted and vmapped across batch dimension.
    """
    # broadcast action into ctrl shape if needed
    ctrl = action
    if ctrl.ndim == 1:
        # single env
        ctrl = ctrl[None, :]
    newd = JaxData(qpos=data.qpos, qvel=data.qvel, ctrl=ctrl)
    return simple_dynamics_step(newd, dt=dt)


def run_steps(data: JaxData, actions: jnp.ndarray, dt: float = 0.02):
    """Run a sequence of actions using lax.scan and return final data.

    actions shape: (T, nv) or (T, batch, nv)
    """
    if jax is None:
        raise RuntimeError("JAX not available")

    def _step(d, a):
        nd = step_fn(d, a, dt=dt)
        return nd, nd

    final, seq = jax.lax.scan(_step, data, actions)
    return final, seq


if __name__ == "__main__":
    # quick smoke test
    if jax is None:
        print("JAX not available — skipping jax_sim_port smoke test")
    else:
        nq = 6
        nv = 6
        data = make_data(nq=nq, nv=nv, batch=1)
        actions = jnp.ones((5, nv), dtype=jnp.float32) * 0.1
        final, seq = run_steps(data, actions, dt=0.02)
        print("final.qpos shape:", final.qpos.shape)
        print("final.qpos:", final.qpos)

