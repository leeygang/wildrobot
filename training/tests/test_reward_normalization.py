"""Pin the semantics of canonical reward-term normalization.

These tests pin the conversion math + report labelling used to put
WildRobot ``reward/<term>`` values and ToddlerBot ``Mean episode <term>``
values onto the same per-step weighted-post-dt axis.

The expected values come straight from the audit summary in
``training/utils/reward_normalization.py`` (module docstring).  Updating
the conversion math without updating those references should fail one or
more tests here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure project root is importable when running these tests from anywhere.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from training.utils.reward_normalization import (  # noqa: E402
    CANONICAL_UNIT,
    RewardRow,
    TB_SOURCE_UNIT,
    WR_SOURCE_UNIT,
    format_reward_comparison,
    tb_canonical_per_step,
    wr_canonical_per_step,
)


# Prompt fixture (analysis-prompt example, dt=0.02 / mean ep len = 50.70):
#   alive                 source=  50.70  -> canonical=0.0200
#   feet_phase            source= 266.99  -> canonical=0.1053
#   penalty_action_rate   source=-667.15  -> canonical=-0.2632
_TB_DT = 0.02
_TB_MEAN_EP_LEN = 50.70


# -----------------------------------------------------------------------------
# WR side
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "term,value",
    [
        ("alive", 0.02),
        ("feet_phase", 0.1053),
        ("action_rate", -0.2632),
    ],
)
def test_wr_canonical_is_identity(term: str, value: float) -> None:
    """WR ``reward/<term>`` is already canonical (weight × raw × dt mean).

    Pinned by env code: ``_aggregate_reward`` applies × dt to each
    weighted term, and the metrics registry MEAN-reduces over (T, N).
    """
    assert wr_canonical_per_step(value) == pytest.approx(value)
    assert wr_canonical_per_step(value) == pytest.approx(value, abs=0.0)


def test_wr_canonical_matches_synthetic_env_emission() -> None:
    """Mock the env path: take a rollout of per-step (weight × raw × dt),
    apply the registry's MEAN reducer, confirm the result is what the
    canonical helper returns.
    """
    weight = 2.0
    raw_per_step = [0.5, 0.6, 0.4, 0.5]  # raw_term values over a tiny rollout
    dt = 0.02
    per_step_weighted_post_dt = [weight * r * dt for r in raw_per_step]
    rollout_mean = sum(per_step_weighted_post_dt) / len(per_step_weighted_post_dt)
    assert wr_canonical_per_step(rollout_mean) == pytest.approx(rollout_mean)


# -----------------------------------------------------------------------------
# TB side
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "term,mean_episode_value,expected_canonical",
    [
        ("alive", 50.70, 0.0200),
        ("feet_phase", 266.99, 0.1053),
        ("penalty_action_rate", -667.15, -0.2632),
    ],
)
def test_tb_canonical_conversion(
    term: str, mean_episode_value: float, expected_canonical: float
) -> None:
    """TB ``Mean episode <term>`` -> canonical per-step weighted post-dt.

    Pinned formula (module docstring): canonical = value / len * dt.
    """
    got = tb_canonical_per_step(
        mean_episode_value, _TB_MEAN_EP_LEN, _TB_DT
    )
    assert got == pytest.approx(expected_canonical, abs=5e-4)


def test_tb_canonical_matches_synthetic_brax_episode_sum() -> None:
    """Mimic the TB+brax path on a tiny synthetic rollout, verify
    normalization recovers the per-step weighted-post-dt mean.

    The synthetic rollout simulates ToddlerBot's
    ``state.metrics[k] = weight × raw`` (pre-dt), brax's
    ``episode_metrics[k] += state.metrics[k]`` (per-episode sum), and the
    final ``Mean episode k`` (mean across episodes).
    """
    weight = 5.0
    dt = 0.02
    per_step_raw = [0.8, 0.9, 0.7, 0.8, 0.9]  # 5-step episode
    per_step_weighted = [weight * r for r in per_step_raw]
    episode_sum = sum(per_step_weighted)        # ≈ episode_metrics[k]
    episode_length = len(per_step_raw)          # ≈ episode_metrics['length']
    expected_per_step_canonical = (
        sum(weight * r * dt for r in per_step_raw) / episode_length
    )
    assert tb_canonical_per_step(episode_sum, episode_length, dt) == pytest.approx(
        expected_per_step_canonical
    )


@pytest.mark.parametrize("bad_len", [0, -1, -10.0])
def test_tb_canonical_rejects_nonpositive_length(bad_len: float) -> None:
    with pytest.raises(ValueError, match="mean_episode_length"):
        tb_canonical_per_step(100.0, bad_len, _TB_DT)


@pytest.mark.parametrize("bad_dt", [0, -0.02])
def test_tb_canonical_rejects_nonpositive_dt(bad_dt: float) -> None:
    with pytest.raises(ValueError, match="dt"):
        tb_canonical_per_step(100.0, _TB_MEAN_EP_LEN, bad_dt)


# -----------------------------------------------------------------------------
# Report formatting
# -----------------------------------------------------------------------------


def test_format_reward_comparison_includes_unit_labels() -> None:
    """The rendered table must surface both source and canonical units
    so a reader cannot accidentally compare incomparable numbers."""
    rows = [
        RewardRow(
            term="alive",
            source_value=50.70,
            canonical_per_step=tb_canonical_per_step(50.70, _TB_MEAN_EP_LEN, _TB_DT),
        ),
        RewardRow(
            term="feet_phase",
            source_value=266.99,
            canonical_per_step=tb_canonical_per_step(266.99, _TB_MEAN_EP_LEN, _TB_DT),
        ),
        RewardRow(
            term="penalty_action_rate",
            source_value=-667.15,
            canonical_per_step=tb_canonical_per_step(-667.15, _TB_MEAN_EP_LEN, _TB_DT),
        ),
    ]
    report = format_reward_comparison(rows, source_unit=TB_SOURCE_UNIT)

    assert TB_SOURCE_UNIT in report, (
        f"source unit string missing from report header:\n{report}"
    )
    assert CANONICAL_UNIT in report, (
        f"canonical unit string missing from report header:\n{report}"
    )
    # Every term appears with both its source value and the canonical
    # value formatted to a recognisable precision.
    for row in rows:
        assert row.term in report
        assert f"{row.source_value:.4f}" in report
        assert f"{row.canonical_per_step:.6f}" in report


def test_format_reward_comparison_wr_unit_string_renders() -> None:
    """A WR-side report should label the source as 'already canonical'
    so it's obvious no conversion was needed."""
    rows = [
        RewardRow(
            term="alive",
            source_value=0.02,
            canonical_per_step=wr_canonical_per_step(0.02),
        ),
    ]
    report = format_reward_comparison(rows, source_unit=WR_SOURCE_UNIT)
    assert WR_SOURCE_UNIT in report
    assert CANONICAL_UNIT in report


def test_format_reward_comparison_handles_empty_rows() -> None:
    report = format_reward_comparison([], source_unit=TB_SOURCE_UNIT)
    assert TB_SOURCE_UNIT in report
    assert CANONICAL_UNIT in report
