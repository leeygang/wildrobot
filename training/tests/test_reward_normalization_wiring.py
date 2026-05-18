"""Pin that the reward-normalization utility is actually wired into the
training W&B emitter and the offline-run analyzer.

These complement ``test_reward_normalization.py`` (which pins the math
primitives) by verifying:

  - ``training/core/experiment_tracking.build_wandb_metrics`` routes
    ``reward/<term>`` through ``wr_canonical_per_step`` so the canonical
    contract is part of the code path, not just docs.
  - The offline-run analyzer's CHANGELOG block surfaces the canonical
    unit label on the WR reward table and renders a WR-vs-TB comparison
    section that converts TB ``Mean episode <term>`` values via
    ``tb_canonical_per_step``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# -----------------------------------------------------------------------------
# Training-side wiring (build_wandb_metrics)
# -----------------------------------------------------------------------------


@pytest.fixture()
def fake_iteration_metrics():
    """Build a minimal duck-typed IterationMetrics-like object sufficient
    for ``build_wandb_metrics``.  Field set matches what
    ``create_training_metrics_from_iteration`` reads.
    """
    obj = types.SimpleNamespace(
        episode_reward=0.91,
        total_loss=0.10,
        policy_loss=0.05,
        value_loss=0.05,
        entropy_loss=0.001,
        clip_fraction=0.10,
        approx_kl=0.01,
        success_rate=0.5,
        episode_length=200.0,
        task_reward_mean=0.91,
        env_metrics={
            # Required by create_training_metrics_from_iteration
            "forward_velocity": 0.20,
            "tracking/avg_torque": 1.5,
            "velocity_command": 0.20,
            "tracking/vel_error": 0.05,
            "tracking/max_torque": 1.5,
            "height": 0.46,
            # Reward-term canonical values (per-step weighted post-dt mean)
            "reward/action_rate": -0.2632,
            "reward/alive": 0.02,
            "reward/feet_phase": 0.1053,
        },
    )
    return obj


def test_build_wandb_metrics_emits_reward_terms_in_canonical_unit(
    fake_iteration_metrics,
) -> None:
    """The W&B emitter must preserve canonical values bit-for-bit (the
    routing through ``wr_canonical_per_step`` is identity for WR)."""
    from training.core.experiment_tracking import build_wandb_metrics

    m, _missing = build_wandb_metrics(
        iteration=1,
        metrics=fake_iteration_metrics,
        steps_per_sec=829.0,
        reward_terms=["reward/action_rate", "reward/alive", "reward/feet_phase"],
    )
    assert m["reward/action_rate"] == pytest.approx(-0.2632)
    assert m["reward/alive"] == pytest.approx(0.02)
    assert m["reward/feet_phase"] == pytest.approx(0.1053)


def test_build_wandb_metrics_calls_canonical_helper(
    monkeypatch, fake_iteration_metrics
) -> None:
    """Regression pin: every ``reward/<term>`` read passes through
    ``_wr_canonical_per_step`` so the contract is testable, not just
    documented.  We swap in a tracking wrapper and assert it observes
    one call per emitted term.
    """
    import training.core.experiment_tracking as et

    calls = []
    original = et._wr_canonical_per_step

    def tracking(value):
        calls.append(value)
        return original(value)

    monkeypatch.setattr(et, "_wr_canonical_per_step", tracking)

    terms = ["reward/action_rate", "reward/alive", "reward/feet_phase"]
    et.build_wandb_metrics(
        iteration=1,
        metrics=fake_iteration_metrics,
        steps_per_sec=829.0,
        reward_terms=terms,
    )
    assert len(calls) == len(terms), (
        f"expected exactly {len(terms)} calls to wr_canonical_per_step, "
        f"got {len(calls)}: {calls}"
    )


# -----------------------------------------------------------------------------
# Analyzer-side wiring (analyze_offline_run._build_tb_comparison_block,
# _changelog_block reward-term header)
# -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def analyzer_module():
    """Load the analyzer script as an importable module.  The script
    lives outside the importable package tree (under skills/) so we
    locate it via importlib."""
    script_path = (
        _PROJECT_ROOT
        / "skills"
        / "wildrobot-training-analyze"
        / "scripts"
        / "analyze_offline_run.py"
    )
    assert script_path.exists(), script_path
    spec = importlib.util.spec_from_file_location(
        "analyze_offline_run_under_test", str(script_path)
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # required so @dataclass works
    spec.loader.exec_module(module)
    return module


def test_analyzer_tb_comparison_block_uses_canonical_conversion(
    analyzer_module,
) -> None:
    """TB ``Mean episode <X>`` values must be converted to canonical
    per-step weighted post-dt via ``tb_canonical_per_step``; the rendered
    table must surface both source and canonical columns plus the unit
    label so a reader cannot silently compare incomparable numbers."""
    block_lines = analyzer_module._build_tb_comparison_block(
        tb_comparison={
            "dt": 0.02,
            "mean_episode_length": 50.70,
            "terms": {
                "alive": 50.70,
                "feet_phase": 266.99,
                "action_rate": -667.15,
            },
        },
        wr_canonical_by_key={
            "reward/alive": 0.0200,
            "reward/feet_phase": 0.1053,
            "reward/action_rate": -0.2632,
        },
        reward_rows=[
            ("reward/alive (w=10, -done)", "reward/alive"),
            ("reward/feet_phase (w=1.0)", "reward/feet_phase"),
            ("reward/action_rate (w=-1.0)", "reward/action_rate"),
        ],
    )
    rendered = "\n".join(block_lines)

    # Header must declare canonical and both source units.
    assert "Canonical unit:" in rendered
    assert "WR source unit:" in rendered
    assert "TB source unit:" in rendered

    # Each term row must carry the WR canonical + TB source + TB canonical.
    for label_substr, wr_canonical, tb_source, tb_canonical in [
        ("alive", "0.0200", "50.7000", "0.0200"),
        ("feet_phase", "0.1053", "266.9900", "0.1053"),
        ("action_rate", "-0.2632", "-667.1500", "-0.2632"),
    ]:
        row = [
            line
            for line in block_lines
            if line.startswith("|") and label_substr in line
        ]
        assert row, f"missing comparison row for {label_substr!r}"
        cells = row[0]
        assert wr_canonical in cells, (label_substr, wr_canonical, cells)
        assert tb_source in cells, (label_substr, tb_source, cells)
        assert tb_canonical in cells, (label_substr, tb_canonical, cells)


def test_analyzer_tb_comparison_block_handles_tb_only_terms(
    analyzer_module,
) -> None:
    """TB terms with no WR counterpart should still render (n/a on the
    WR side)."""
    block_lines = analyzer_module._build_tb_comparison_block(
        tb_comparison={
            "dt": 0.02,
            "mean_episode_length": 50.70,
            "terms": {
                "tb_only_signal": 12.34,
            },
        },
        wr_canonical_by_key={},
        reward_rows=[],
    )
    rendered = "\n".join(block_lines)
    assert "(TB-only) tb_only_signal" in rendered
    assert "12.3400" in rendered
    assert "n/a" in rendered  # WR cell


def test_analyzer_changelog_block_renders_canonical_unit_header(
    analyzer_module,
) -> None:
    """When the analyzer renders the v0201 reward-term table, it should
    print the canonical-unit label so readers cannot mis-interpret the
    numbers as TB-style episode totals."""
    keys = analyzer_module.EvalKeySet(
        prefix="Evaluate/",
        success="Evaluate/mean_reward",
        ep_len="Evaluate/mean_episode_length",
    )
    walking = analyzer_module.WalkingSummary(
        enabled=True,
        forward_vel_mean=0.20,
        velocity_cmd_mean=0.20,
        velocity_error_mean=0.05,
        verdict="moving",
        tracking_status="tracking",
        stability="stable",
    )
    best_row = {
        "Evaluate/mean_reward": 10.0,
        "Evaluate/mean_episode_length": 500.0,
        # One reward/<term> so the table prints at least one canonical row.
        "reward/alive": 0.02,
    }
    out = analyzer_module._changelog_block(
        version="0.20.1-smokeX",
        run_dir=Path("/tmp/fake-run"),
        ckpt_dir=None,
        best_ckpt=None,
        keys=keys,
        best_it=100,
        best_row=best_row,
        walking=walking,
        regression=None,
    )
    assert "Reward terms" in out
    assert "weight × raw × dt" in out, out
    # The lone canonical reward row should appear.
    assert "0.0200" in out


def test_analyzer_changelog_block_emits_comparison_when_tb_supplied(
    tmp_path, analyzer_module,
) -> None:
    keys = analyzer_module.EvalKeySet(
        prefix="Evaluate/",
        success="Evaluate/mean_reward",
        ep_len="Evaluate/mean_episode_length",
    )
    walking = analyzer_module.WalkingSummary(
        enabled=True,
        forward_vel_mean=0.20,
        velocity_cmd_mean=0.20,
        velocity_error_mean=0.05,
        verdict="moving",
        tracking_status="tracking",
        stability="stable",
    )
    best_row = {
        "Evaluate/mean_reward": 10.0,
        "Evaluate/mean_episode_length": 500.0,
        "reward/alive": 0.02,
        "reward/feet_phase": 0.1053,
        "reward/action_rate": -0.2632,
    }
    tb_comparison = {
        "dt": 0.02,
        "mean_episode_length": 50.70,
        "terms": {
            "alive": 50.70,
            "feet_phase": 266.99,
            "action_rate": -667.15,
        },
    }
    out = analyzer_module._changelog_block(
        version="0.20.1-smokeX",
        run_dir=Path("/tmp/fake-run"),
        ckpt_dir=None,
        best_ckpt=None,
        keys=keys,
        best_it=100,
        best_row=best_row,
        walking=walking,
        regression=None,
        tb_comparison=tb_comparison,
    )
    assert "WR vs TB reward-term comparison" in out
    assert "Canonical unit:" in out
    assert "TB source unit:" in out
    # Spot-check the converted TB values appear (per prompt fixture).
    for canonical in ("0.0200", "0.1053", "-0.2632"):
        assert canonical in out, (canonical, out)
    for source in ("50.7000", "266.9900", "-667.1500"):
        assert source in out, (source, out)


def test_analyzer_tb_comparison_rejects_malformed_terms(analyzer_module) -> None:
    with pytest.raises(ValueError, match="terms"):
        analyzer_module._build_tb_comparison_block(
            tb_comparison={"dt": 0.02, "mean_episode_length": 50.7, "terms": "not-a-dict"},
            wr_canonical_by_key={},
            reward_rows=[],
        )


def test_analyzer_tb_comparison_json_fixture_roundtrips(
    tmp_path, analyzer_module
) -> None:
    """A JSON file matching the CLI contract should parse and render."""
    fixture = {
        "dt": 0.02,
        "mean_episode_length": 50.70,
        "terms": {"alive": 50.70, "feet_phase": 266.99},
    }
    path = tmp_path / "tb_comparison.json"
    path.write_text(json.dumps(fixture))
    parsed = json.loads(path.read_text())
    block = analyzer_module._build_tb_comparison_block(
        tb_comparison=parsed,
        wr_canonical_by_key={"reward/alive": 0.02, "reward/feet_phase": 0.1053},
        reward_rows=[
            ("reward/alive", "reward/alive"),
            ("reward/feet_phase", "reward/feet_phase"),
        ],
    )
    rendered = "\n".join(block)
    assert "0.0200" in rendered and "0.1053" in rendered
