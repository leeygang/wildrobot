"""Small JAX-native training stub that uses optax and demonstrates param pytrees.

This is a minimal starter to begin converting the training pipeline to optax/Flax.
"""
import os
try:
    import jax
    import jax.numpy as jnp
    import optax
except Exception:
    jax = None
    jnp = None
    optax = None

def simple_update_step(params, grads, lr=1e-3):
    if optax is None:
        raise RuntimeError("optax not installed")
    tx = optax.sgd(lr)
    opt_state = tx.init(params)
    updates, new_state = tx.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_state

def make_dummy_params():
    # simple pytree dict of arrays
    return {"w": jnp.zeros((10, 10)), "b": jnp.zeros((10,))} if jnp is not None else None

def run_smoke():
    if jax is None:
        print("JAX not available; skipping JAX training stub.")
        return
    params = make_dummy_params()
    grads = {"w": jnp.ones((10, 10)) * 0.01, "b": jnp.ones((10,)) * 0.01}
    new_params, _ = simple_update_step(params, grads)
    print("Updated param w mean:", float(jnp.mean(new_params["w"])))

if __name__ == '__main__':
    run_smoke()
