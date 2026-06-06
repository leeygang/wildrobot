"""Tests for v0.21.0 smoke9 lateral curriculum:
``env.cmd_sampler_walk_vy_exact_signed_bins``.

Background — smoke7/8 both learned a ONE-SIDED lateral mode (pass +vy
probes, still move +vy under -vy).  smoke8 widened ``min/max_velocity_y``
to +-0.065 and raised ``alpha_y`` 30->120, but kept TB's ellipse sampler
``vy = uniform(deadzone_y, |bound|) * cos(theta)``.  Measured train-side
``tracking/velocity_cmd_vy_abs`` was only ~0.026 (run gbis9wc4) — STILL
below the inherited ~0.05 +y bias — so the signed lateral teaching signal
never overcame the bias and smoke8's hypothesis was never actually tested.

smoke9's only lever de-projects vy into the balanced exact signed bins
``{min_velocity_y, 0, max_velocity_y}`` (each 1/3) so the train-side
lateral command matches the +-0.065 probes and over-samples both signs
equally.

Tests:
  * Field default is False (smoke7/8 unaffected, byte-for-byte).
  * When True: walk-branch vy is EXACTLY in {y_lo, 0, y_hi}, balanced
    across the two nonzero signs, and NEVER ellipse-projected (the test
    fails if any vy magnitude collapses to an intermediate cos(theta)
    value).
  * When False: vy is ellipse-projected (continuous magnitudes, both
    signs) — the smoke7/8 baseline.
  * The smoke9 YAML opts in; smoke8 does not.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import jax
import pytest

from training.configs.training_config import load_training_config
from training.configs.training_runtime_config import EnvConfig
from training.envs.wildrobot_env import WildRobotEnv


_SMOKE8 = Path("training/configs/ppo_walking_v0210_smoke8_2d_signed_vy.yaml")
_SMOKE9 = Path("training/configs/ppo_walking_v0210_smoke9_exact_signed_vy.yaml")


# ---------------------------------------------------------------------------
# Dataclass default + back-compat
# ---------------------------------------------------------------------------
def test_cmd_sampler_walk_vy_exact_signed_bins_default_false() -> None:
    """Default must be False so smoke7/8 (ellipse vy) are byte-identical."""
    env_cfg = EnvConfig()
    assert env_cfg.cmd_sampler_walk_vy_exact_signed_bins is False


# ---------------------------------------------------------------------------
# YAML opt-in: smoke9 yes, smoke8 no
# ---------------------------------------------------------------------------
def test_smoke9_yaml_opts_in() -> None:
    if not _SMOKE9.exists():
        pytest.skip(f"{_SMOKE9.name} not found")
    cfg = load_training_config(str(_SMOKE9))
    assert cfg.env.cmd_sampler_walk_vy_exact_signed_bins is True


def test_smoke8_yaml_does_not_set_flag() -> None:
    """smoke8 must remain byte-equivalent (default False) — its lateral
    command stays cos(theta)-projected."""
    if not _SMOKE8.exists():
        pytest.skip(f"{_SMOKE8.name} not found")
    cfg = load_training_config(str(_SMOKE8))
    assert cfg.env.cmd_sampler_walk_vy_exact_signed_bins is False


# ---------------------------------------------------------------------------
# Sampler behaviour
# ---------------------------------------------------------------------------
def _build_env_with_vy_exact_bins(*, exact: bool) -> WildRobotEnv:
    """Load smoke8 as a base (3D branched sampler + vy range +-0.065 +
    vx-positive-only are all already set) and override only the new flag."""
    if not _SMOKE8.exists():
        pytest.skip(f"{_SMOKE8.name} not found")
    cfg = load_training_config(str(_SMOKE8))
    new_env = dataclasses.replace(
        cfg.env, cmd_sampler_walk_vy_exact_signed_bins=exact
    )
    cfg = dataclasses.replace(cfg, env=new_env)
    return WildRobotEnv(config=cfg)


def _sample_n_walk_commands(
    env: WildRobotEnv, n: int
) -> list[tuple[float, float, float]]:
    out = []
    for i in range(n):
        rng_a, rng_b = jax.random.split(jax.random.PRNGKey(i))
        sampled = env._sample_walk_command(rng_a, rng_b)
        out.append((float(sampled[0]), float(sampled[1]), float(sampled[2])))
    return out


def test_vy_is_exact_signed_bins_when_on() -> None:
    """With the flag on, every walk-branch vy must be EXACTLY one of
    {y_lo, 0, y_hi}.  Fails if any value is ellipse-projected (an
    intermediate magnitude) — that is the smoke8 bug this stage fixes."""
    env = _build_env_with_vy_exact_bins(exact=True)
    y_lo = float(env._config.env.min_velocity_y)
    y_hi = float(env._config.env.max_velocity_y)
    allowed = {round(y_lo, 6), 0.0, round(y_hi, 6)}

    samples = _sample_n_walk_commands(env, 300)
    vys = [vy for _, vy, _ in samples]

    off_grid = [v for v in vys if round(v, 6) not in allowed]
    assert not off_grid, (
        f"Expected vy strictly in {sorted(allowed)} under exact-bins=True; "
        f"got {len(off_grid)} ellipse-projected (off-grid) value(s), e.g. "
        f"{off_grid[:5]}.  vy must NOT collapse through cos(theta)."
    )

    # Both nonzero signs must appear and be roughly balanced (1/3 each).
    n_neg = sum(1 for v in vys if v < -1e-6)
    n_pos = sum(1 for v in vys if v > 1e-6)
    n_zero = sum(1 for v in vys if abs(v) <= 1e-6)
    assert n_neg > 50, f"expected ~1/3 negative vy bins; got {n_neg}/300"
    assert n_pos > 50, f"expected ~1/3 positive vy bins; got {n_pos}/300"
    assert n_zero > 50, f"expected ~1/3 zero vy bins; got {n_zero}/300"
    # Balanced signs: the |+| and |-| counts should be within a generous
    # band of each other (this is the property smoke7/8 lacked).
    assert abs(n_neg - n_pos) < 50, (
        f"expected balanced signed vy bins; got -{n_neg} vs +{n_pos}"
    )


def test_vy_is_ellipse_projected_when_off() -> None:
    """With the flag off (default), vy uses the TB cos(theta) ellipse:
    continuous magnitudes spanning both signs, almost never landing
    exactly on the +-0.065 bound.  This is the smoke7/8 baseline."""
    env = _build_env_with_vy_exact_bins(exact=False)
    y_hi = float(env._config.env.max_velocity_y)

    samples = _sample_n_walk_commands(env, 300)
    vys = [vy for _, vy, _ in samples]

    n_neg = sum(1 for v in vys if v < -1e-6)
    n_pos = sum(1 for v in vys if v > 1e-6)
    assert n_neg > 50, f"expected >50 negative ellipse vy; got {n_neg}/300"
    assert n_pos > 50, f"expected >50 positive ellipse vy; got {n_pos}/300"

    # The ellipse projection means |vy| is continuous and the realized
    # mean magnitude is well below the bound (the smoke8 collapse).  Very
    # few samples should sit exactly on +-y_hi.
    on_bound = sum(1 for v in vys if abs(abs(v) - y_hi) < 1e-4)
    assert on_bound < 15, (
        f"ellipse vy should rarely hit the bound exactly; got {on_bound}/300 "
        f"at |vy|={y_hi} (looks de-projected, not ellipse)."
    )
    mean_abs = sum(abs(v) for v in vys) / len(vys)
    assert mean_abs < y_hi, (
        f"ellipse mean |vy| ({mean_abs:.4f}) should be below the bound "
        f"({y_hi}); the cos(theta) projection is the smoke8 collapse."
    )


def test_vx_unaffected_by_vy_exact_bins() -> None:
    """The vy flag must not touch vx — smoke8 keeps
    cmd_sampler_walk_vx_positive_only=True, so vx stays in
    [min_velocity, max_velocity] under either vy setting."""
    env = _build_env_with_vy_exact_bins(exact=True)
    min_vx = float(env._config.env.min_velocity)
    max_vx = float(env._config.env.max_velocity)
    samples = _sample_n_walk_commands(env, 200)
    vxs = [vx for vx, _, _ in samples]
    out_of_range = [v for v in vxs if v < min_vx - 1e-6 or v > max_vx + 1e-6]
    assert not out_of_range, (
        f"vx must stay in [{min_vx}, {max_vx}] regardless of vy bins; "
        f"got out-of-range {out_of_range[:5]}"
    )
