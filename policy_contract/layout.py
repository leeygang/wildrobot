from __future__ import annotations

from typing import Dict

from policy_contract.spec import ObservationSpec


def get_slices(observation: ObservationSpec) -> Dict[str, slice]:
    """Compute name->slice mapping for an observation layout."""
    idx = 0
    out: Dict[str, slice] = {}
    for field in observation.layout:
        size = int(field.size)
        out[field.name] = slice(idx, idx + size)
        idx += size
    return out
