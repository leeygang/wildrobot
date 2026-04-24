"""YAML round-trip assertion for the v0.20.1 smoke config.

Catches the class of bug found in smoke6: a YAML key under
``reward_weights:`` is added but the ``_parse_reward_weights_config``
loader is not updated to read it, so the value silently falls through
to the dataclass default (typically 0.0).  The training run completes,
the reward term contributes 0 every step, and the run looks like a
"dead term" failure when really the term was never enabled.

Smoke6 specifically: YAML set ``lin_vel_z: 1.0`` and ``ang_vel_xy: 2.0``
but the loader returned 0.0 for both.  Per-step reward contribution was
0 across all 16 eval iters from iter 0 onward.  Cost: one full smoke
run (~20M env steps) of compute on a misdiagnosed "TB phase signals
dead at training time" hypothesis.

This test fails loud at config-load time on any silent drop.

Spec: training/docs/walking_training.md v0.20.1-smoke6-prep3 §.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

_SMOKE_CFG = Path("training/configs/ppo_walking_v0201_smoke.yaml")
if not _SMOKE_CFG.exists():
    pytest.skip(
        f"{_SMOKE_CFG.name} not found",
        allow_module_level=True,
    )

from training.configs.training_config import load_training_config


def _flatten_yaml_section(d: dict, prefix: str = "") -> dict:
    """Return ``{key_path: value}`` for every leaf scalar / list in ``d``.

    Skips nested dicts (those are separate config sections with their own
    parsers) but follows them to find leaf scalars at the same nesting
    depth.  For the v0.20.1 smoke config, ``reward_weights:`` and ``env:``
    are flat — every value under them is a scalar or list.
    """
    out = {}
    for key, value in d.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.update(_flatten_yaml_section(value, path))
        else:
            out[path] = value
    return out


@pytest.fixture(scope="module")
def smoke_yaml_raw() -> dict:
    with open(_SMOKE_CFG) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def smoke_cfg():
    return load_training_config(str(_SMOKE_CFG))


def test_reward_weights_round_trip(smoke_yaml_raw, smoke_cfg) -> None:
    """Every key under ``reward_weights:`` in YAML round-trips to the
    dataclass field at the same value.

    Catches: YAML adds a new reward-weight key, dataclass adds the
    matching field with a default, but ``_parse_reward_weights_config``
    forgets to thread the YAML value through.  Result: silent drop to
    default.
    """
    yaml_rewards = smoke_yaml_raw.get("reward_weights", {})
    if not yaml_rewards:
        pytest.skip("smoke YAML has no reward_weights: section")

    rw = smoke_cfg.reward_weights
    failures: list[str] = []
    for key, yaml_value in yaml_rewards.items():
        if not hasattr(rw, key):
            failures.append(
                f"YAML key reward_weights.{key} = {yaml_value!r} but "
                f"RewardWeightsConfig has no field named {key!r}"
            )
            continue
        loaded_value = getattr(rw, key)
        # Allow numeric type drift (float vs int) but require value equality.
        if loaded_value != yaml_value:
            failures.append(
                f"YAML key reward_weights.{key} = {yaml_value!r} but "
                f"loaded config has {key} = {loaded_value!r} "
                f"(silent drop or override?)"
            )

    if failures:
        msg = (
            "reward_weights round-trip failed for "
            f"{len(failures)} key(s).  Most likely cause: "
            "_parse_reward_weights_config in training/configs/training_config.py "
            "is missing a `rewards.get(...)` line for the new key.\n  "
            + "\n  ".join(failures)
        )
        pytest.fail(msg)


def test_env_round_trip(smoke_yaml_raw, smoke_cfg) -> None:
    """Every scalar / list key under ``env:`` in YAML round-trips to the
    EnvConfig dataclass.

    Same bug class as reward_weights — when a smoke promotes a new env
    knob (e.g. ``cmd_resample_steps`` for smoke7's multi-cmd curriculum),
    we want the loader to thread it through, not silently drop to default.
    """
    yaml_env = smoke_yaml_raw.get("env", {})
    if not yaml_env:
        pytest.skip("smoke YAML has no env: section")

    env_cfg = smoke_cfg.env
    failures: list[str] = []
    for key, yaml_value in yaml_env.items():
        # Some env keys are nested (e.g. push schedules); skip dicts here
        # — they go through their own parsers.
        if isinstance(yaml_value, dict):
            continue
        if not hasattr(env_cfg, key):
            # env: is a permissive section historically (legacy keys).
            # Don't fail on unknown keys here; the reward-weights test is
            # the load-bearing one for the smoke6 bug class.  Just record.
            continue
        loaded_value = getattr(env_cfg, key)
        if loaded_value != yaml_value:
            failures.append(
                f"YAML key env.{key} = {yaml_value!r} but "
                f"loaded config has {key} = {loaded_value!r}"
            )

    if failures:
        msg = (
            "env round-trip failed for "
            f"{len(failures)} key(s).\n  "
            + "\n  ".join(failures)
        )
        pytest.fail(msg)


def test_smoke6_critical_weights_nonzero(smoke_cfg) -> None:
    """Pin the specific weights smoke6 needed.

    Hard-coded so the test is robust against future YAML refactors that
    might rename keys: smoke6+ requires these reward weights to be
    nonzero in the smoke config.  If a future smoke deliberately zeros
    one, update this test alongside the YAML change.
    """
    rw = smoke_cfg.reward_weights
    expected = {
        "ref_q_track": 5.0,
        "ref_body_quat_track": 5.0,
        "torso_pos_xy": 2.0,
        "ref_contact_match": 1.0,
        "lin_vel_z": 1.0,
        "ang_vel_xy": 2.0,
        "cmd_forward_velocity_track": 5.0,
        "feet_air_time": 500.0,
        "feet_clearance": 1.0,
        "feet_distance": 1.0,
        "alive": 10.0,
    }
    failures: list[str] = []
    for key, expected_value in expected.items():
        actual = getattr(rw, key)
        if actual != expected_value:
            failures.append(f"{key}: expected {expected_value}, got {actual}")
    if failures:
        pytest.fail(
            "smoke6 critical reward weights mismatch — either the YAML or "
            "the loader has drifted from the documented smoke6 contract:\n  "
            + "\n  ".join(failures)
        )
