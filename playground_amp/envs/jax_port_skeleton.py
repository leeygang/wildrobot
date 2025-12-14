"""Small skeleton demonstrating a JAX-style step_fn using `data.replace`.

This file is a non-invasive starting point for porting `WildRobotEnv` to a
pure-JAX functional API. It is intentionally lightweight and safe to import.
"""
from __future__ import annotations

try:
    import jax
    import jax.numpy as jnp
except Exception:
    jax = None
    jnp = None

try:
    import mujoco.mjx as mjx
except Exception:
    mjx = None


def step_fn_example(mjx_model, data, action, n_substeps: int = 1):
    """A minimal pure function that would step the simulator `n_substeps`.

    Returns the resulting `data`. In real port, this will be jitted and used
    inside `jax.lax.scan` or `jax.jit`.
    """
    if jax is None or mjx is None:
        raise RuntimeError("JAX or mjx not available for step_fn_example")

    def _step(d, _):
        # In a true port, use d.replace(...) and mjx.step returning a new data
        nd = mjx.step(mjx_model, d)
        return nd, None

    data, _ = jax.lax.scan(_step, data, None, length=n_substeps)
    return data
