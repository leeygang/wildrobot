from __future__ import annotations

from flax import struct
import jax.numpy as jnp

from policy_contract.spec import PolicySpec


@struct.dataclass
class PolicyState:
    prev_action: jnp.ndarray

    @classmethod
    def init(cls, spec: PolicySpec, *, mode: str = "zeros") -> "PolicyState":
        if mode == "zeros":
            return cls(prev_action=jnp.zeros((spec.model.action_dim,), dtype=jnp.float32))
        if mode == "from_pose":
            raise ValueError("PolicyState.init(mode='from_pose') is not implemented yet")
        raise ValueError(f"Unknown PolicyState.init mode: {mode}")
