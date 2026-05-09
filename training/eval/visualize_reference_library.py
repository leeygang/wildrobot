#!/usr/bin/env python3
"""Reference library viewer and trace logger for v0.20.0.

Consumes the offline ``ReferenceLibrary`` schema and provides:
  - visual playback of individual command-bin trajectories
  - command-sweep trace plots
  - validation summary

Usage
-----
  # View a single command bin
  uv run python training/eval/visualize_reference_library.py \
      --library-path /tmp/ref_library --vx 0.15

  # Command sweep summary
  uv run python training/eval/visualize_reference_library.py \
      --library-path /tmp/ref_library --sweep

  # Validate all entries
  uv run python training/eval/visualize_reference_library.py \
      --library-path /tmp/ref_library --validate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _load_library(path: str):
    """Load library, importing lazily to keep the script self-contained."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from control.references.reference_library import ReferenceLibrary
    return ReferenceLibrary.load(path)


def _print_summary(lib) -> None:
    print(lib.summary())
    print()


def _validate(lib) -> bool:
    issues = lib.validate()
    if not issues:
        print("All entries pass validation.")
        return True
    for key, entry_issues in issues.items():
        print(f"  [{key}]:")
        for issue in entry_issues:
            print(f"    - {issue}")
    return False


def _view_entry(lib, vx: float, vy: float = 0.0, yaw: float = 0.0) -> None:
    traj = lib.lookup(vx, vy, yaw)
    n = traj.n_steps
    matched = traj.command_key

    print(f"Trajectory for command vx={vx:.3f} (matched key={matched})")
    print(f"  n_steps={n}  dt={traj.dt}  cycle_time={traj.cycle_time}")
    print(f"  n_joints={traj.n_joints}")
    print(f"  generator={traj.generator_version}")
    print(f"  valid={traj.is_valid}")
    print()

    # Per-step summary
    print(f"{'step':>5} {'phase':>6} {'stance':>6} {'contact':>10} "
          f"{'pelvis_h':>9} {'pelvis_p':>9}")
    print("-" * 55)
    for i in range(n):
        phase = traj.phase[i] if traj.phase is not None else 0.0
        stance = int(traj.stance_foot_id[i]) if traj.stance_foot_id is not None else -1
        contact_str = ""
        if traj.contact_mask is not None:
            cl, cr = traj.contact_mask[i]
            contact_str = f"L={cl:.0f} R={cr:.0f}"
        pelvis_h = traj.pelvis_pos[i, 2] if traj.pelvis_pos is not None else 0.0
        pelvis_p = traj.pelvis_rpy[i, 1] if traj.pelvis_rpy is not None else 0.0
        print(f"{i:5d} {phase:6.3f} {'L' if stance == 0 else 'R':>6} "
              f"{contact_str:>10} {pelvis_h:9.4f} {pelvis_p:9.4f}")

    # Preview window test
    from control.references.reference_library import ReferencePreviewWindow
    preview = lib.get_preview(vx, step_index=0, n_preview=5)
    obs = preview.to_obs_array()
    print(f"\nPreview window obs vector: shape={obs.shape}, "
          f"range=[{obs.min():.3f}, {obs.max():.3f}]")


def _sweep_summary(lib) -> None:
    print("Command sweep summary:")
    print(f"{'vx':>7} {'vy':>7} {'yaw':>7} {'steps':>6} {'valid':>6} "
          f"{'pelvis_h_mean':>14} {'phase_range':>12}")
    print("-" * 70)
    for key in lib.command_keys:
        traj = lib[key]
        pelvis_h = np.mean(traj.pelvis_pos[:, 2]) if traj.pelvis_pos is not None else 0.0
        phase_range = ""
        if traj.phase is not None:
            phase_range = f"[{traj.phase.min():.2f},{traj.phase.max():.2f}]"
        valid = "OK" if traj.is_valid else "FAIL"
        print(f"{key[0]:7.3f} {key[1]:7.3f} {key[2]:7.3f} "
              f"{traj.n_steps:6d} {valid:>6} {pelvis_h:14.4f} {phase_range:>12}")


def _create_dummy_library(out_path: str) -> None:
    """Create a minimal dummy library for schema testing."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from control.references.reference_library import (
        ReferenceLibrary,
        ReferenceLibraryMeta,
        ReferenceTrajectory,
    )

    n_joints = 8  # 4 per leg for WildRobot v2
    dt = 0.02
    cycle_time = 0.50
    n_steps = int(round(cycle_time / dt))

    meta = ReferenceLibraryMeta(
        generator="dummy_test",
        generator_version="0.0.1",
        robot="wildrobot_v2",
        dt=dt,
        cycle_time=cycle_time,
        n_joints=n_joints,
        command_range_vx=(0.0, 0.20),
        command_interval=0.05,
        created="2026-04-17",
    )

    trajectories = []
    for vx in [0.0, 0.05, 0.10, 0.15, 0.20]:
        phase = np.linspace(0, 1, n_steps, endpoint=False).astype(np.float32)

        # Stance switches at mid-cycle
        stance = np.zeros(n_steps, dtype=np.float32)
        stance[n_steps // 2:] = 1.0

        # Contact mask: swing foot lifts during single support
        contact = np.ones((n_steps, 2), dtype=np.float32)
        swing_start = n_steps // 8
        swing_end = n_steps // 2 - n_steps // 8
        contact[swing_start:swing_end, 1] = 0.0  # right foot swings first half
        contact[n_steps // 2 + swing_start:n_steps // 2 + swing_end, 0] = 0.0

        # Placeholder joint targets (zeros + small oscillation)
        q = np.zeros((n_steps, n_joints), dtype=np.float32)
        q[:, 0] = 0.1 * np.sin(2 * np.pi * phase)  # hip pitch oscillation

        # Pelvis: constant height, small pitch
        pelvis_pos = np.zeros((n_steps, 3), dtype=np.float32)
        pelvis_pos[:, 2] = 0.42
        pelvis_rpy = np.zeros((n_steps, 3), dtype=np.float32)
        pelvis_rpy[:, 1] = 0.01 * np.sin(2 * np.pi * phase)

        # COM tracks pelvis
        com_pos = pelvis_pos.copy()

        # Foot positions: symmetric about origin
        left_foot = np.zeros((n_steps, 3), dtype=np.float32)
        left_foot[:, 1] = 0.08  # lateral offset
        right_foot = np.zeros((n_steps, 3), dtype=np.float32)
        right_foot[:, 1] = -0.08

        # Add forward stepping motion proportional to vx
        if vx > 0:
            step_len = vx * cycle_time / 2.0
            # Right foot swings forward in first half
            for i in range(swing_start, swing_end):
                frac = (i - swing_start) / max(1, swing_end - swing_start - 1)
                right_foot[i, 0] = step_len * frac
                right_foot[i, 2] = 0.02 * np.sin(np.pi * frac)
            right_foot[swing_end:n_steps // 2, 0] = step_len
            # Left foot swings forward in second half
            ls = n_steps // 2 + swing_start
            le = n_steps // 2 + swing_end
            for i in range(ls, le):
                frac = (i - ls) / max(1, le - ls - 1)
                left_foot[i, 0] = step_len * frac
                left_foot[i, 2] = 0.02 * np.sin(np.pi * frac)
            left_foot[le:, 0] = step_len

        foot_rpy = np.zeros((n_steps, 3), dtype=np.float32)

        traj = ReferenceTrajectory(
            command_vx=vx,
            dt=dt,
            cycle_time=cycle_time,
            q_ref=q,
            pelvis_pos=pelvis_pos,
            pelvis_rpy=pelvis_rpy,
            com_pos=com_pos,
            left_foot_pos=left_foot,
            left_foot_rpy=foot_rpy.copy(),
            right_foot_pos=right_foot,
            right_foot_rpy=foot_rpy.copy(),
            stance_foot_id=stance,
            contact_mask=contact,
            phase=phase,
            generator_version="dummy_0.0.1",
        )
        trajectories.append(traj)

    lib = ReferenceLibrary(trajectories, meta)
    lib.save(out_path)
    print(f"Saved dummy library to {out_path}")
    _print_summary(lib)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reference library viewer and trace logger (v0.20.0-A)")
    parser.add_argument("--library-path", type=str,
                        help="Path to reference library directory")
    parser.add_argument("--vx", type=float, default=0.20,
                        help="Forward speed command for single-entry view (default 0.20 = Phase 9D operating point, TB-step/leg-matched at cycle_time=0.96 s)")
    parser.add_argument("--sweep", action="store_true",
                        help="Print command sweep summary")
    parser.add_argument("--validate", action="store_true",
                        help="Validate all library entries")
    parser.add_argument("--create-dummy", type=str, default=None,
                        help="Create a dummy library at the given path for testing")
    args = parser.parse_args()

    if args.create_dummy:
        _create_dummy_library(args.create_dummy)
        return

    if not args.library_path:
        parser.error("--library-path is required (or use --create-dummy)")

    lib = _load_library(args.library_path)
    _print_summary(lib)

    if args.validate:
        ok = _validate(lib)
        if not ok:
            sys.exit(1)
    elif args.sweep:
        _sweep_summary(lib)
    else:
        _view_entry(lib, args.vx)


if __name__ == "__main__":
    main()
