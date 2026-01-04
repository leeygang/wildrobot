from __future__ import annotations

import jax.numpy as jnp

from policy_contract.jax.state import PolicyState
from policy_contract.spec import PolicySpec


def postprocess_action(
    *,
    spec: PolicySpec,
    state: PolicyState,
    action_raw: jnp.ndarray,
) -> tuple[jnp.ndarray, PolicyState]:
    if spec.action.postprocess_id == "none":
        action = action_raw.astype(jnp.float32)
        return action, state.replace(prev_action=action)

    if spec.action.postprocess_id == "lowpass_v1":
        alpha = spec.action.postprocess_params.get("alpha")
        if alpha is None:
            raise ValueError("postprocess_params.alpha is required for lowpass_v1")
        alpha = float(alpha)
        action = alpha * state.prev_action + (1.0 - alpha) * action_raw
        action = action.astype(jnp.float32)
        return action, state.replace(prev_action=action)

    raise ValueError(f"Unsupported postprocess_id: {spec.action.postprocess_id}")
