"""WildRobot training environment — v0.20.1 placeholder.

The previous 6533-line ``WildRobotEnv`` was deleted at v0.20.1
(commits ``0ffc473`` removed v1/v2 reference modules; this file
replaces the env that consumed them).  The v3-only rewrite is the
next-session task.  See:

  - ``training/docs/v0201_env_wiring.md`` §10 — 8-step rewrite plan
  - ``training/docs/walking_training.md`` v0.20.1 section — smoke spec
  - tasks #50 (rewrite) and #51 (cleanup) in the project task list

Why this placeholder exists:

  External review (round-3) of commit ``0ffc473`` flagged that the
  branch was not safe for shared use because ``import
  training.envs.wildrobot_env`` failed with
  ``ModuleNotFoundError`` on the deleted ``locomotion_contract``.
  The placeholder makes the import graph fully clean: the module
  imports without error, the public symbols (``WildRobotEnv``,
  ``WildRobotEnvState``) are present, and any attempt to construct
  the env raises ``NotImplementedError`` with a clear pointer to
  the rewrite plan.  This prevents opaque failures and signals
  intent to anyone touching the branch before the rewrite lands.

After the rewrite this file is replaced with the real v3-only env
(~1500-2000 lines per the design note).  Symbols expected in that
rewrite, listed here so consumers know what to expect:

  - ``WildRobotEnv``: ``mjx_env.MjxEnv`` subclass; constructed with
    a ``TrainingConfig`` whose ``env.loc_ref_version`` is
    ``"v3_offline_library"``
  - ``WildRobotEnvState``: ``mjx_env.State`` subclass adding
    ``pipeline_state`` (Brax wrapper compat) and ``rng``
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jp
from mujoco_playground._src import mjx_env


__all__ = [
    "WildRobotEnv",
    "WildRobotEnvState",
    "_PLACEHOLDER_REASON",
]


_PLACEHOLDER_REASON = (
    "WildRobotEnv is a placeholder pending the v0.20.1 v3-only rewrite. "
    "See training/docs/v0201_env_wiring.md §10 for the rewrite plan, and "
    "tasks #50 (rewrite) / #51 (cleanup) in the project task list.  The "
    "v1/v2 reference stack and the v0.19.5c-era env that consumed it were "
    "deleted at v0.20.1 (commit 0ffc473 onward); the v3 rewrite restores "
    "this module."
)


class WildRobotEnvState(mjx_env.State):
    """Shape-preserving placeholder for the post-rewrite state class.

    The rewrite will keep this exact class shape (``pipeline_state``
    + ``rng`` extending ``mjx_env.State``) so consumers that
    pattern-match on it don't need to change."""

    pipeline_state: Any = None
    rng: jp.ndarray = None


class WildRobotEnv:
    """Placeholder class.  Raises ``NotImplementedError`` on
    construction with a pointer to the rewrite plan."""

    def __init__(self, *args, **kwargs) -> None:  # pragma: no cover
        raise NotImplementedError(_PLACEHOLDER_REASON)
