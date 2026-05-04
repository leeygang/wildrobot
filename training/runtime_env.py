"""Process-start runtime environment guards for training entrypoints.

These settings must be applied before importing JAX-heavy modules.
"""

from __future__ import annotations

import os
import platform


def configure_training_runtime_env() -> None:
    """Set conservative defaults for local MJX training.

    Rationale:
    - ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` avoids JAX grabbing almost all
      GPU memory at process start, which can surface as opaque BLAS/cuBLAS
      initialization failures when another process is already using the GPU.
    - ``MUJOCO_GL=egl`` matches the rest of the repo's Linux headless scripts.

    Existing user overrides are respected.
    """

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    if platform.system() == "Linux":
        os.environ.setdefault("MUJOCO_GL", "egl")

