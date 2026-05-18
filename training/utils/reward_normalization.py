"""Normalize WR and TB reward-term metrics to a single canonical unit.

Canonical unit
--------------
    per-step weighted post-dt reward contribution = weight × raw × dt

This is the quantity each environment step adds to the PPO reward scalar
``reward = sum(weight × raw) × dt``.  Comparing reward decompositions in
this unit lets us put WildRobot and ToddlerBot numbers on the same axis.

Source-unit semantics (confirmed in code, 2026-05-17)
-----------------------------------------------------
WR ``reward/<term>`` (logged in W&B and replayed in offline-run analysis):
  - ``training/envs/wildrobot_env.py:1693-1751`` (``_aggregate_reward``)
      pre_dt[k]  = weight × raw_term
      contrib[k] = pre_dt[k] × dt
      terminal_metrics_dict["reward/<k>"] = contrib[k]  # set every step
  - ``training/core/metrics_registry.py``: each ``reward/<k>`` is
    ``Reducer.MEAN`` over the (T, N) rollout.
  -> Already in canonical units; no conversion needed.

TB ``state.metrics[<term>]`` and printed ``Mean episode <term>``:
  - ``toddlerbot/locomotion/mjx_env.py:1700-1768``
      reward_dict[k] = weight × raw_term            # pre-dt
      reward = sum(reward_dict.values()) * self.dt  # × dt applied here
      state.metrics.update(reward_dict)             # pre-dt values land in metrics
  - ``brax/envs/wrappers/training.py:117-124`` (``EpisodeWrapper.step``)
      episode_metrics[k] += state.metrics[k]        # per-episode sum, pre-dt
  - ``toddlerbot/locomotion/train_mjx.py:417`` prints
      ``Mean episode <term> = mean(episode_metrics[<term>])``
  -> A per-episode sum of pre-dt weighted values; to canonical:
       canonical = (mean_episode_value / mean_episode_length) * dt
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


CANONICAL_UNIT = "per-step weighted post-dt contribution (weight × raw × dt)"
WR_SOURCE_UNIT = (
    "rollout-mean per-step weighted post-dt contribution "
    "(MEAN over (T,N) of weight × raw × dt)"
)
TB_SOURCE_UNIT = (
    "rollout-mean per-episode sum of weighted pre-dt contributions "
    "(mean over episodes of Σ_step weight × raw)"
)


def wr_canonical_per_step(reward_metric_value: float) -> float:
    """Return the WR ``reward/<term>`` value unchanged.

    Provided for symmetry with :func:`tb_canonical_per_step` so callers
    can treat both repos through a uniform interface.  WR's emission path
    already produces the canonical unit (see module docstring).
    """
    return float(reward_metric_value)


def tb_canonical_per_step(
    mean_episode_value: float,
    mean_episode_length: float,
    dt: float,
) -> float:
    """Convert a TB ``Mean episode <term>`` value to canonical per-step.

    Args:
        mean_episode_value: TB's printed/logged ``Mean episode <term>``,
            i.e. the rollout mean of ``Σ_step (weight × raw)``.
        mean_episode_length: TB's ``Mean episode length`` (steps).
        dt: TB's control timestep (``self.dt``).

    Returns:
        Per-step weighted post-dt contribution
        (= ``mean_episode_value / mean_episode_length * dt``).
    """
    if mean_episode_length <= 0:
        raise ValueError(
            f"mean_episode_length must be > 0, got {mean_episode_length!r}"
        )
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt!r}")
    return float(mean_episode_value) / float(mean_episode_length) * float(dt)


@dataclass(frozen=True)
class RewardRow:
    """One reward term, with both its source-unit and canonical values."""

    term: str
    source_value: float
    canonical_per_step: float


def format_reward_comparison(
    rows: List[RewardRow],
    *,
    source_unit: str,
    canonical_unit: str = CANONICAL_UNIT,
) -> str:
    """Render reward rows as a fixed-width table with explicit unit headers.

    Both unit strings are written into the header so a reader can never
    accidentally compare a TB ``Mean episode`` value to a WR ``reward/``
    value without seeing what conversion (if any) was applied.
    """
    header = (
        f"# source unit:    {source_unit}\n"
        f"# canonical unit: {canonical_unit}\n"
        f"{'term':<28s} {'source_value':>14s} {'canonical_per_step':>22s}\n"
    )
    body = "\n".join(
        f"{row.term:<28s} {row.source_value:>14.4f} {row.canonical_per_step:>22.6f}"
        for row in rows
    )
    return header + body + ("\n" if rows else "")
