#!/usr/bin/env python3
"""Phase 5 lifecycle/reproducibility checks for WR reference assets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from control.references.reference_library import (
    ReferenceLibrary,
    ReferenceLibraryMeta,
    ReferenceTrajectory,
)
from control.zmp.zmp_walk import ZMPWalkGenerator
from tools import reference_geometry_parity


def _dummy_library(vx_values: list[float]) -> ReferenceLibrary:
    n, n_joints = 12, 8
    phase = np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float32)
    trajectories: list[ReferenceTrajectory] = []
    for vx in sorted({round(float(v), 4) for v in vx_values}):
        trajectories.append(
            ReferenceTrajectory(
                command_vx=vx,
                dt=0.02,
                cycle_time=0.24,
                q_ref=np.zeros((n, n_joints), dtype=np.float32),
                phase=phase,
                stance_foot_id=np.zeros(n, dtype=np.float32),
                contact_mask=np.ones((n, 2), dtype=np.float32),
            )
        )
    meta = ReferenceLibraryMeta(
        generator="test_phase5",
        generator_version="1.0",
        dt=0.02,
        cycle_time=0.24,
        n_joints=n_joints,
        command_range_vx=(trajectories[0].command_vx, trajectories[-1].command_vx),
        command_interval=0.05,
    )
    return ReferenceLibrary(trajectories, meta)


def test_build_library_for_vx_values_exact_bins():
    gen = ZMPWalkGenerator()
    bins = [0.10, 0.15, 0.20]
    lib = gen.build_library_for_vx_values(bins)
    actual_bins = [key[0] for key in lib.command_keys]
    assert actual_bins == bins


def test_phase5_wr_cache_reuses_saved_library(tmp_path, monkeypatch):
    def _fake_build(self, vx_values, interval=None):  # noqa: ANN001
        return _dummy_library(list(vx_values))

    monkeypatch.setattr(
        reference_geometry_parity.ZMPWalkGenerator,
        "build_library_for_vx_values",
        _fake_build,
    )
    monkeypatch.setattr(
        reference_geometry_parity,
        "_wr_generator_fingerprint",
        lambda: "generator_sig_v1",
    )

    args = argparse.Namespace(
        vx=0.15,
        nominal_only=False,
        wr_library_path=None,
        wr_library_cache_dir=str(tmp_path / "wr_cache"),
        wr_library_no_cache=False,
        wr_library_rebuild=False,
    )

    _lib1, lifecycle1 = reference_geometry_parity._resolve_wr_reference_library(args)
    _lib2, lifecycle2 = reference_geometry_parity._resolve_wr_reference_library(args)

    assert lifecycle1.source == "cache-build"
    assert lifecycle2.source == "cache-hit"
    assert lifecycle1.path == lifecycle2.path
    assert (Path(lifecycle1.path) / "metadata.json").exists()


def test_phase5_wr_cache_invalidates_on_generator_change(tmp_path, monkeypatch):
    build_calls = {"count": 0}

    def _fake_build(self, vx_values, interval=None):  # noqa: ANN001
        build_calls["count"] += 1
        return _dummy_library(list(vx_values))

    monkeypatch.setattr(
        reference_geometry_parity.ZMPWalkGenerator,
        "build_library_for_vx_values",
        _fake_build,
    )

    args = argparse.Namespace(
        vx=0.15,
        nominal_only=False,
        wr_library_path=None,
        wr_library_cache_dir=str(tmp_path / "wr_cache"),
        wr_library_no_cache=False,
        wr_library_rebuild=False,
    )

    monkeypatch.setattr(
        reference_geometry_parity,
        "_wr_generator_fingerprint",
        lambda: "generator_sig_v1",
    )
    _lib1, lifecycle1 = reference_geometry_parity._resolve_wr_reference_library(args)
    _lib2, lifecycle2 = reference_geometry_parity._resolve_wr_reference_library(args)
    assert lifecycle1.source == "cache-build"
    assert lifecycle2.source == "cache-hit"
    assert build_calls["count"] == 1

    monkeypatch.setattr(
        reference_geometry_parity,
        "_wr_generator_fingerprint",
        lambda: "generator_sig_v2",
    )
    _lib3, lifecycle3 = reference_geometry_parity._resolve_wr_reference_library(args)
    assert lifecycle3.source == "cache-build"
    assert lifecycle3.path != lifecycle1.path
    assert lifecycle3.generator_fingerprint == "generator_sig_v2"
    assert build_calls["count"] == 2
    assert (Path(lifecycle3.path) / "build_spec.json").exists()


def test_wr_generator_fingerprint_changes_on_asset_change(tmp_path, monkeypatch):
    wildrobot_root = tmp_path / "wildrobot"
    (wildrobot_root / "control" / "zmp").mkdir(parents=True)
    (wildrobot_root / "control" / "references").mkdir(parents=True)
    (wildrobot_root / "assets" / "v2").mkdir(parents=True)

    (wildrobot_root / "control" / "zmp" / "zmp_walk.py").write_text(
        "def a():\n    return 1\n", encoding="utf-8"
    )
    (wildrobot_root / "control" / "zmp" / "zmp_planner.py").write_text(
        "def b():\n    return 2\n", encoding="utf-8"
    )
    (wildrobot_root / "control" / "references" / "reference_library.py").write_text(
        "def c():\n    return 3\n", encoding="utf-8"
    )
    scene_path = wildrobot_root / "assets" / "v2" / "scene_flat_terrain.xml"
    scene_path.write_text("<mujoco><worldbody/></mujoco>\n", encoding="utf-8")
    (wildrobot_root / "assets" / "v2" / "mujoco_robot_config.json").write_text(
        '{"actuated_joint_specs":[]}\n', encoding="utf-8"
    )

    monkeypatch.setattr(
        reference_geometry_parity,
        "_repo_roots",
        lambda: (wildrobot_root, wildrobot_root.parent / "toddlerbot"),
    )
    fp_before = reference_geometry_parity._wr_generator_fingerprint()
    scene_path.write_text("<mujoco><worldbody><body/></worldbody></mujoco>\n", encoding="utf-8")
    fp_after = reference_geometry_parity._wr_generator_fingerprint()
    assert fp_before != fp_after
