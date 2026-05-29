"""Tests for v0.21.0 smoke2 vx-range: env.cmd_sampler_walk_vx_positive_only.

Background — smoke2 post-mortem (lblub15y, 2026-05-24):
TB's `_sample_walk_command` ellipse legitimately emits negative vx via the
``sin_theta < 0`` branch (wildrobot_env.py:3055: ``x_max = -x_lo`` when
``sin_theta < 0``), and it can shrink |vx| near zero via |sin_theta| even
when min_velocity is positive.  WR smoke2's basin-break contract is
forward-only at [min_velocity, max_velocity], so this flag makes the
walk-branch vx range literal without disabling the rest of the 3D sampler.

Tests:
  * Field has correct default (False) for back-compat.
  * When True: sampled walk-branch vx stays in [min_velocity, max_velocity].
  * When False (default): sampled walk-branch vx spans both signs
    (preserves TB symmetric-ellipse behavior).
  * vy is NOT clamped under either setting (library has symmetric +/- vy
    bins, so negative vy_cmd has a real reference).
  * Smoke2 YAML opts in (config-load test).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jp
import pytest

from training.configs.training_config import load_training_config
from training.configs.training_runtime_config import EnvConfig
from training.envs.wildrobot_env import WildRobotEnv


_SMOKE1 = Path("training/configs/ppo_walking_v0210_smoke1_lateral_yaw.yaml")
_SMOKE2 = Path("training/configs/ppo_walking_v0210_smoke2.yaml")


# ---------------------------------------------------------------------------
# Dataclass default + back-compat
# ---------------------------------------------------------------------------
def test_cmd_sampler_walk_vx_positive_only_default_false() -> None:
    """Default must be False so legacy YAMLs (smoke1, smoke14, ...) are
    byte-for-byte identical."""
    env_cfg = EnvConfig()
    assert env_cfg.cmd_sampler_walk_vx_positive_only is False


# ---------------------------------------------------------------------------
# Smoke2 YAML opts in; smoke1 does not
# ---------------------------------------------------------------------------
def test_smoke2_yaml_opts_in() -> None:
    if not _SMOKE2.exists():
        pytest.skip(f"{_SMOKE2.name} not found")
    cfg = load_training_config(str(_SMOKE2))
    assert cfg.env.cmd_sampler_walk_vx_positive_only is True


def test_smoke1_yaml_does_not_set_flag() -> None:
    """smoke1 must remain byte-equivalent (default False) so smoke1
    reproducibility from disk is preserved."""
    if not _SMOKE1.exists():
        pytest.skip(f"{_SMOKE1.name} not found")
    cfg = load_training_config(str(_SMOKE1))
    assert cfg.env.cmd_sampler_walk_vx_positive_only is False


# ---------------------------------------------------------------------------
# Sampler behaviour: clamp on, clamp off
# ---------------------------------------------------------------------------
def _build_env_with_walk_vx_clamp(*, clamp: bool) -> WildRobotEnv:
    """Load smoke1 as a base (3D sampler is already enabled there) and
    override the new flag via dataclass replacement."""
    if not _SMOKE1.exists():
        pytest.skip(f"{_SMOKE1.name} not found")
    cfg = load_training_config(str(_SMOKE1))
    new_env = dataclasses.replace(cfg.env, cmd_sampler_walk_vx_positive_only=clamp)
    cfg = dataclasses.replace(cfg, env=new_env)
    return WildRobotEnv(config=cfg)


def _sample_n_walk_commands(env: WildRobotEnv, n: int) -> list[tuple[float, float, float]]:
    """Call ``_sample_walk_command`` directly N times with distinct
    PRNG keys.  Returns the (vx, vy, wz) triples as Python floats."""
    out = []
    for i in range(n):
        rng_a, rng_b = jax.random.split(jax.random.PRNGKey(i))
        sampled = env._sample_walk_command(rng_a, rng_b)
        out.append((float(sampled[0]), float(sampled[1]), float(sampled[2])))
    return out


def test_walk_branch_vx_uses_config_range_when_clamp_on() -> None:
    """With clamp=True, every walk-branch vx sample must stay inside
    the configured forward range.  This catches the smoke2 bug where
    vx was positive but still shrunk toward zero by abs(sin(theta))."""
    env = _build_env_with_walk_vx_clamp(clamp=True)
    samples = _sample_n_walk_commands(env, 200)
    vxs = [vx for vx, _, _ in samples]
    n_neg = sum(1 for v in vxs if v < 0)
    assert n_neg == 0, (
        f"Expected zero negative vx samples under clamp=True; got {n_neg}/{len(vxs)}.  "
        f"Sample of negatives: {[v for v in vxs if v < 0][:5]}"
    )
    min_vx = float(env._config.env.min_velocity)
    max_vx = float(env._config.env.max_velocity)
    out_of_range = [v for v in vxs if v < min_vx or v > max_vx]
    assert not out_of_range, (
        f"Expected vx samples in [{min_vx}, {max_vx}] under clamp=True; "
        f"got out-of-range sample(s): {out_of_range[:5]}"
    )
    expected_mean = 0.5 * (min_vx + max_vx)
    actual_mean = sum(vxs) / len(vxs)
    assert abs(actual_mean - expected_mean) < 0.02, (
        f"Expected roughly uniform vx samples under clamp=True; "
        f"mean={actual_mean:.4f}, expected midpoint={expected_mean:.4f}"
    )
    # wz is always pinned to 0 in the walk branch.
    wzs = [wz for _, _, wz in samples]
    assert all(abs(wz) < 1e-6 for wz in wzs)


def test_walk_branch_vx_spans_both_signs_when_clamp_off() -> None:
    """With clamp=False (default / TB-symmetric), walk-branch vx must
    span both signs in expectation.  Asserts both >0 and <0 samples
    appear over many trials — this is the smoke1 baseline behavior."""
    env = _build_env_with_walk_vx_clamp(clamp=False)
    samples = _sample_n_walk_commands(env, 200)
    vxs = [vx for vx, _, _ in samples]
    n_pos = sum(1 for v in vxs if v > 0.01)
    n_neg = sum(1 for v in vxs if v < -0.01)
    # TB ellipse is symmetric around 0 → ~50% each side; allow generous
    # slack for the 200-sample variance.
    assert n_pos > 50, f"expected >50 positive vx samples; got {n_pos}/200"
    assert n_neg > 50, f"expected >50 negative vx samples; got {n_neg}/200"


def test_walk_branch_vy_not_clamped_by_vx_flag() -> None:
    """vy must still span both signs even when vx clamp is on — the
    flag is vx-specific (WR library has symmetric +/- vy bins so
    negative vy_cmd has a real reference)."""
    env = _build_env_with_walk_vx_clamp(clamp=True)
    samples = _sample_n_walk_commands(env, 200)
    vys = [vy for _, vy, _ in samples]
    n_pos = sum(1 for v in vys if v > 0.01)
    n_neg = sum(1 for v in vys if v < -0.01)
    assert n_pos > 50, f"expected >50 positive vy samples under vx-clamp; got {n_pos}/200"
    assert n_neg > 50, f"expected >50 negative vy samples under vx-clamp; got {n_neg}/200"
