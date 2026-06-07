"""Runtime validation helpers."""

from __future__ import annotations

from typing import Any

__all__ = [
    "load_runtime_realism_profile",
]


def __getattr__(name: str) -> Any:
    if name == "load_runtime_realism_profile":
        from .realism_profile import load_runtime_realism_profile

        return load_runtime_realism_profile
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
