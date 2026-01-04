from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from policy_contract.spec import PolicySpec


@dataclass(frozen=True)
class PolicyState:
    prev_action: np.ndarray

    @classmethod
    def init(cls, spec: PolicySpec, *, mode: str = "zeros") -> "PolicyState":
        if mode == "zeros":
            return cls(prev_action=np.zeros((spec.model.action_dim,), dtype=np.float32))
        if mode == "from_pose":
            raise ValueError("PolicyState.init(mode='from_pose') is not implemented yet")
        raise ValueError(f"Unknown PolicyState.init mode: {mode}")
