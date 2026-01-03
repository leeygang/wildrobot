"""Compatibility shim.

The runtime package was renamed from `wildrobot_runtime` to `wr_runtime`.
Prefer importing `wr_runtime` directly.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "Package 'wildrobot_runtime' is deprecated; use 'wr_runtime' instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export top-level symbols if needed.
from wr_runtime.config import RuntimeConfig, load_config  # noqa: F401
