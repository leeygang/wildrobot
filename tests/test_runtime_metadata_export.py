"""Bundle export metadata: runtime_policy_config carries what the runtime needs.

Validates ``build_runtime_metadata`` against the actual smoke9
``training_config.yaml`` env block (without building the heavy ZMP phase table —
that is injected as a small synthetic reference).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from conftest import make_v8_spec, make_reference_dict
from training.exports.runtime_metadata import build_runtime_metadata

_SMOKE9_CKPT = Path(
    "training/checkpoints/"
    "ppo_walking_v0210_smoke9_exact_signed_vy_v00210_20260606_091415-j6kaw1ij/"
    "training_config.yaml"
)


def _smoke9_env() -> dict:
    if not _SMOKE9_CKPT.exists():
        pytest.skip(f"{_SMOKE9_CKPT} not found")
    data = yaml.safe_load(_SMOKE9_CKPT.read_text())
    return data["env"]


def test_metadata_matches_smoke9_training_config() -> None:
    env = _smoke9_env()
    spec = make_v8_spec()
    meta = build_runtime_metadata(
        env=env, spec=spec, reference=make_reference_dict()
    )

    # Action contract — the load-bearing fields the runtime needs and that
    # policy_spec.json does NOT carry.
    assert meta["loc_ref_residual_base"] == "home"
    assert meta["loc_ref_residual_mode"] == "absolute"
    assert meta["action_delay_steps"] == 1
    assert meta["action_filter_alpha"] == 0.0
    assert meta["loc_ref_residual_scale"] == pytest.approx(0.2)
    assert meta["ctrl_dt"] == pytest.approx(0.02)
    assert meta["control_hz"] == pytest.approx(50.0)
    assert meta["actor_obs_layout_id"] == "wr_obs_v8_cmd3d"
    assert meta["action_mapping_id"] == "pos_target_rad_v1"
    assert meta["loc_ref_command_conditioned"] is True
    assert meta["loc_ref_command_axes_3d"] is True
    assert meta["default_velocity_cmd"] == pytest.approx([0.13, 0.0, 0.0])
    assert meta["loc_ref_walking_joint_offsets_rad"] == {}
    assert meta["residual_base_offset_per_actuator"] == pytest.approx(
        [0.0] * len(spec.robot.actuator_names)
    )

    # Per-joint residual scales preserved + resolved into actuator order.
    pj = meta["loc_ref_residual_scale_per_joint"]
    assert pj["left_knee_pitch"] == pytest.approx(0.978)
    assert pj["left_hip_pitch"] == pytest.approx(0.64)
    assert pj["left_ankle_pitch"] == pytest.approx(0.33)

    per_act = meta["residual_scale_per_actuator"]
    names = list(spec.robot.actuator_names)
    assert len(per_act) == len(names)
    # Listed leg joint -> its per-joint scale; unlisted joint -> scalar fallback.
    assert per_act[names.index("left_knee_pitch")] == pytest.approx(0.978)
    assert per_act[names.index("waist_yaw")] == pytest.approx(0.2)
    assert per_act[names.index("left_shoulder_pitch")] == pytest.approx(0.2)


def test_metadata_resolves_walking_offsets_in_actuator_order(tmp_path) -> None:
    from runtime.wr_runtime.control.runtime_policy_config import RuntimePolicyConfig

    env = _smoke9_env()
    env["loc_ref_walking_joint_offsets_rad"] = {
        "left_hip_roll": 0.03,
        "right_hip_roll": -0.03,
        "left_ankle_roll": -0.03,
        "right_ankle_roll": 0.03,
    }
    spec = make_v8_spec()
    meta = build_runtime_metadata(
        env=env, spec=spec, reference=make_reference_dict()
    )

    names = list(spec.robot.actuator_names)
    resolved = meta["residual_base_offset_per_actuator"]
    assert resolved[names.index("left_hip_roll")] == pytest.approx(0.03)
    assert resolved[names.index("right_hip_roll")] == pytest.approx(-0.03)
    assert resolved[names.index("left_ankle_roll")] == pytest.approx(-0.03)
    assert resolved[names.index("right_ankle_roll")] == pytest.approx(0.03)

    path = tmp_path / "runtime_policy_config.json"
    path.write_text(json.dumps(meta), encoding="utf-8")
    runtime_config = RuntimePolicyConfig.from_json(path)
    assert runtime_config.loc_ref_walking_joint_offsets_rad == env[
        "loc_ref_walking_joint_offsets_rad"
    ]
    assert runtime_config.residual_base_offset_per_actuator == pytest.approx(resolved)

def test_metadata_reference_block_roundtrips_into_runtime_config(tmp_path) -> None:
    """The metadata 'reference' block must load into the runtime loader."""
    from runtime.wr_runtime.control.runtime_policy_config import (
        ReferencePhaseTable,
        RuntimePolicyConfig,
    )

    env = _smoke9_env()
    spec = make_v8_spec()
    ref = make_reference_dict(n_steps=96, n_cycle=48)
    meta = build_runtime_metadata(env=env, spec=spec, reference=ref)
    table = ReferencePhaseTable.from_dict(meta["reference"])
    assert table.n_steps == 96
    assert table.cmd_keys.shape == (4, 3)
    assert table.phase_sin.shape == (96,)

    path = tmp_path / "runtime_policy_config.json"
    path.write_text(json.dumps(meta), encoding="utf-8")
    runtime_config = RuntimePolicyConfig.from_json(path)
    assert runtime_config.loc_ref_walking_joint_offsets_rad == {}
    assert runtime_config.residual_base_offset_per_actuator == pytest.approx(
        [0.0] * spec.model.action_dim
    )
