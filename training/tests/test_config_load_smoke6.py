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
from control.zmp.zmp_walk import ZMPWalkGenerator


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


def test_smoke7_critical_reward_weights(smoke_cfg) -> None:
    """Pin the smoke7 / TB-active reward weights.

    Hard-coded so the test is robust against future YAML refactors that
    might rename keys.  The expected values reflect the v0.20.1
    TB-active alignment sweep (walking_training.md Appendix B):
      - Phase 1 lowered alive 10→1, cmd_forward 5→2, ref_body_quat 5→2.5
        (those were aligned to the commented-out walk.gin ZMP variant;
        TB-active values are 1.0 / 2.0 / 2.5 in walk.gin:111-127).
      - Phase 2 added ref_feet_z_track (+5.0, dense per-step foot-z),
        penalty_pose (-0.5, per-joint anchor with WR negative-yaml
        convention), penalty_feet_ori (-5.0, anti-tippy-toe).
    If a future smoke deliberately changes one, update this test
    alongside the YAML change.
    """
    rw = smoke_cfg.reward_weights
    expected = {
        # Imitation block (TB-active aligned at the magnitudes here).
        "ref_q_track": 5.0,
        "ref_body_quat_track": 2.5,        # Phase 1: 5.0 → 2.5 (walk.gin:117).
        "torso_pos_xy": 2.0,
        "ref_contact_match": 1.0,
        "lin_vel_z": 1.0,
        "ang_vel_xy": 2.0,
        # Phase 2 — dense foot-z tracker (TB feet_phase substitute).
        "ref_feet_z_track": 5.0,
        # Task block.
        "cmd_forward_velocity_track": 2.0, # Phase 1: 5.0 → 2.0 (walk.gin:111).
        # TB shaping family.
        "feet_air_time": 500.0,
        "feet_clearance": 1.0,
        "feet_distance": 1.0,
        # Phase 2 — anti-exploit penalties (negative yaml weight × positive
        # raw squared-error, WR convention; magnitudes match TB walk.gin
        # 0.5 / 5.0 since TB uses +scale × negative function value).
        "penalty_pose": -0.5,
        "penalty_feet_ori": -5.0,
        # Survival — Phase 1c switched to dense per-step at TB-active
        # weight 1.0 (was 10.0 with -done semantics).
        "alive": 1.0,
    }
    failures: list[str] = []
    for key, expected_value in expected.items():
        actual = getattr(rw, key)
        if actual != expected_value:
            failures.append(f"{key}: expected {expected_value}, got {actual}")
    if failures:
        pytest.fail(
            "smoke7 critical reward weights mismatch — either the YAML or "
            "the loader has drifted from the documented smoke7 / TB-active "
            "contract:\n  "
            + "\n  ".join(failures)
        )


def test_smoke7_critical_env_settings(smoke_cfg) -> None:
    """Pin the smoke7 multi-cmd + DR contract.

    smoke7 enables WR's already-wired multi-cmd + DR plumbing to test
    whether the TB curriculum (not the reward family) is what breaks
    the shuffle exploit.  Same protective pattern as the smoke6 weights
    test: catches silent env-loader drift that would otherwise turn
    smoke7 into another null run like smoke6 was.
    """
    e = smoke_cfg.env
    expected = {
        # Multi-cmd curriculum (TB anti-shuffle lever #1).  Phase 9D
        # (2026-05-09): cycle_time scaled 0.72 → 0.96 s; operating point
        # shifted vx 0.265 → 0.20 to preserve step/leg ≈ TB's 0.256
        # under the longer cycle.  max_velocity stays at 0.30 (now 1.5×
        # operating point — generous robustness pressure).
        "min_velocity": 0.0,
        "max_velocity": 0.30,
        "cmd_resample_steps": 150,
        "cmd_zero_chance": 0.2,
        "cmd_deadzone": 0.05,
        # Eval-cmd override pins eval rollouts at the Phase 9D
        # operating point so G4 stays comparable across smokes.
        "eval_velocity_cmd": 0.20,
        # The fixed offline q_ref trajectory must match the Phase 9D eval
        # operating point.  Otherwise PPO is asked to track an off-operating
        # prior while being evaluated against a different command.
        "loc_ref_offline_command_vx": 0.20,
        # DR (TB anti-shuffle lever #2)
        "domain_randomization_enabled": True,
        "domain_rand_friction_range": [0.4, 1.0],
        "domain_rand_mass_scale_range": [0.9, 1.1],
        "domain_rand_kp_scale_range": [0.9, 1.1],
        "domain_rand_frictionloss_scale_range": [0.8, 1.2],
        "domain_rand_joint_offset_rad": 0.03,
        # IMU noise (sim2real hardening)
        "imu_gyro_noise_std": 0.05,
        "imu_quat_noise_deg": 2.0,
        # Push events: NOT enabled (TB walk default add_push: False)
        "push_enabled": False,
        # action_delay_steps unchanged from smoke6 (TB n_steps_delay=1)
        "action_delay_steps": 1,
        # Phase 3 (walking_training.md Appendix B.2): TB-form smooth
        # backlash DR (mjx_config.py:209-210, mjx_env.py:2073-2080).
        "domain_rand_backlash_range": [0.02, 0.10],
        "domain_rand_backlash_activation": 0.1,
        # Phase 2: per-joint penalty_pose vector default (scalar fallback
        # when a joint isn't listed in penalty_pose_weights_per_joint).
        "penalty_pose_weight_default": 0.0,
    }
    failures: list[str] = []
    for key, expected_value in expected.items():
        actual = getattr(e, key)
        if actual != expected_value:
            failures.append(f"{key}: expected {expected_value!r}, got {actual!r}")
    if failures:
        pytest.fail(
            "smoke7 critical env settings mismatch — either the YAML or "
            "the loader has drifted from the documented smoke7 contract:\n  "
            + "\n  ".join(failures)
        )


def test_smoke7_offline_reference_vx_does_not_snap_to_default_bin(smoke_cfg) -> None:
    """The fixed q_ref operating point must be generated exactly.

    The default ZMP library grid has 0.05 m/s bins, so a plain default
    library lookup would snap any non-grid value (e.g. Phase 9A's 0.265
    or Phase 9D's 0.20 if the grid drifts) to its nearest grid bin.
    The env builds an explicit one-bin library for the configured
    operating point; keep that contract covered at the
    generator/library boundary.
    """
    offline_vx = smoke_cfg.env.loc_ref_offline_command_vx
    lib = ZMPWalkGenerator().build_library_for_vx_values([offline_vx])
    traj = lib.lookup(offline_vx)

    assert traj.command_vx == pytest.approx(offline_vx)
