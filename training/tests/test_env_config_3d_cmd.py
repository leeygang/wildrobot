"""v0.21.0 P4.1 — EnvConfig accepts vy / wz command ranges.

Divergence from the spec body in
``training/docs/.tmp_p4_original_spec.md`` (P4.1):

    The spec imports ``LocomotionEnvConfig`` from
    ``training.configs.training_runtime_config``.  The actual class in
    this repo is named ``EnvConfig`` (P3 divergence #1).  The tests use
    ``EnvConfig`` accordingly.
"""

from __future__ import annotations

import pytest

from training.configs.training_runtime_config import EnvConfig


def test_env_config_accepts_vy_and_yaw_rate_ranges() -> None:
    cfg = EnvConfig(
        min_velocity=0.18,
        max_velocity=0.26,
        min_velocity_y=-0.13,
        max_velocity_y=0.13,
        max_yaw_rate=0.25,
    )
    assert cfg.min_velocity_y == pytest.approx(-0.13)
    assert cfg.max_velocity_y == pytest.approx(0.13)
    assert cfg.max_yaw_rate == pytest.approx(0.25)


def test_env_config_defaults_vy_wz_to_zero() -> None:
    cfg = EnvConfig(min_velocity=0.18, max_velocity=0.26)
    assert cfg.min_velocity_y == 0.0
    assert cfg.max_velocity_y == 0.0
    assert cfg.max_yaw_rate == 0.0


def test_env_config_defaults_cmd_sampler_3d_branched_false() -> None:
    """P4-fix: the TB-branched 3D sampler is opt-in.  Legacy YAML and
    fresh ``EnvConfig`` instances must default to ``False`` so smoke14 /
    smoke12b / smoke7 use the v0.20.x scalar-vx sampler (forward-only)."""
    cfg = EnvConfig(min_velocity=0.18, max_velocity=0.26)
    assert cfg.cmd_sampler_3d_branched is False
