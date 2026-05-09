#!/usr/bin/env python3
"""v0.20.0-C closeout sweep — produces the CHANGELOG artifact.

Runs all 32 measurement cases defined in
``training/docs/reference_design.md`` v0.20.0-C "Closeout matrix":

  - 4 kinematic baseline rows  (1 per vx, deterministic)
  - 4 fixed-base PD baselines  (1 per vx, deterministic)
  - 12 free-floating C1 rows   (4 vx x 3 seeds, no stabilizer)
  - 12 free-floating C2 rows   (4 vx x 3 seeds, --c2-stabilizer)

For each row, parses survival, touchdown counts and stride per
side, overall + worst-joint RMSE, any-joint saturation %, and (C2
only) harness clip-saturation per channel.  Scores against the
C1 / C2 metric gates and emits a single Markdown table plus a
pass/fail summary suitable for pasting into ``training/CHANGELOG.md``.

Usage::

    uv run python tools/v0200c_closeout.py --output /tmp/v0200c_artifact.md
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


# Closeout sweep parameters (frozen by reference_design.md v0.20.0-C).
# Phase 9D-revisited (2026-05-09): cycle_time scaled to WR's leg
# pendulum (0.72 → 0.96 s); operating point shifted vx 0.265 → 0.20
# to preserve step_length_per_leg ≈ TB's 0.256 under the longer cycle.
# Closeout matrix brackets the new operating point with vx=0.15
# (below) and vx=0.25 (above) for regression / robustness checks.
_VX_BINS = (0.15, 0.20, 0.25)
_SEEDS = (0, 1, 2)
_HORIZON = 200
_PRINT_EVERY = 200    # only print start + end inside each replay
_VIEWER = "training/eval/view_zmp_in_mujoco.py"

# Viewer runner: mjpython on macOS for the GUI loop, but headless
# runs only need a normal Python.  Override via V0200C_RUNNER env
# var (e.g. ``V0200C_RUNNER="uv run python"`` if mjpython is not
# installed locally).  Default tries mjpython first and falls back
# to plain ``uv run python`` when it is missing on PATH.
def _detect_runner() -> List[str]:
    explicit = os.environ.get("V0200C_RUNNER")
    if explicit:
        return explicit.split()
    if shutil.which("mjpython") is not None:
        return ["uv", "run", "mjpython"]
    return ["uv", "run", "python"]


_RUNNER: Optional[List[str]] = None  # set in main()
_LIBRARY_PATH: Optional[Path] = None  # set in main() after _build_pinned_library

# Gate thresholds (frozen by v0.20.0-C external review).
_RMSE_PER_JOINT_BUDGET = 0.40
_RMSE_OVERALL_BUDGET = 0.25
_FIXED_BASE_SAT_BUDGET = 0.05         # 5 %, per C1 servo-trackability
_C1_HORIZON_FACTOR = 2.0              # 2 * cycle_time / ctrl_dt
_CYCLE_TIME = 0.64
_CTRL_DT = 0.02
_C1_SURVIVAL_BUDGET = math.ceil(_C1_HORIZON_FACTOR * _CYCLE_TIME / _CTRL_DT)
_C2_SAT_TARGET = 0.20
_C2_SAT_HARDFAIL = 0.25
_C2_HARNESS_CLIP_TARGET = 0.10
_C2_HARNESS_CLIP_HARDFAIL = 0.25
_C2_STEP_LEN_BUDGET = 0.03
_C2_STEP_RATIO_BUDGET = 0.5

_LEG_JOINTS = (
    "L_hip_pitch", "R_hip_pitch", "L_hip_roll", "R_hip_roll",
    "L_knee_pitch", "R_knee_pitch", "L_ankle_pitch", "R_ankle_pitch",
)


@dataclass
class RunResult:
    mode: str
    vx: float
    seed: Optional[int]
    cmd_line: str
    log_path: Optional[str] = None
    survival_steps: Optional[int] = None
    overall_rmse: Optional[float] = None
    worst_joint: Optional[str] = None
    worst_joint_rmse: Optional[float] = None
    any_sat_pct: Optional[float] = None
    L_touchdowns: Optional[int] = None
    R_touchdowns: Optional[int] = None
    L_stride_m: Optional[float] = None
    R_stride_m: Optional[float] = None
    L_step_m: Optional[float] = None
    R_step_m: Optional[float] = None
    L_ratio: Optional[float] = None
    R_ratio: Optional[float] = None
    pitch_clip_pct: Optional[float] = None
    roll_clip_pct: Optional[float] = None
    cp_clip_pct: Optional[float] = None
    any_clip_pct: Optional[float] = None     # union, contract metric
    fail_reasons: List[str] = field(default_factory=list)


class SubprocessFailure(RuntimeError):
    """Raised when a closeout subprocess returns non-zero."""


def _shell(cmd: List[str], *, check: bool = True) -> str:
    """Run a command, return combined stdout+stderr.

    If ``check`` is True (default), raises SubprocessFailure on
    non-zero exit so the closeout never silently scores a broken
    subprocess as a row.  Pass ``check=False`` for git introspection
    where empty output is acceptable.
    """
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    out = (proc.stdout or "") + (proc.stderr or "")
    if check and proc.returncode != 0:
        raise SubprocessFailure(
            f"Command {' '.join(cmd)!r} failed with rc={proc.returncode}.\n"
            f"--- combined output ---\n{out[-2000:]}"
        )
    return out


def _build_cmd(mode: str, vx: float, seed: Optional[int]) -> List[str]:
    assert _RUNNER is not None, "_RUNNER must be set before _build_cmd"
    cmd = [
        *_RUNNER, _VIEWER,
        "--vx", f"{vx:.3f}", "--headless",
        "--horizon", str(_HORIZON), "--print-every", str(_PRINT_EVERY),
    ]
    if _LIBRARY_PATH is not None:
        # Phase 9D pinned library — use exact closeout bins so non-grid
        # values cannot snap to a default 0.05-step bin via nearest-
        # neighbour lookup.  See `_build_pinned_library` below.
        cmd.extend(["--library-path", str(_LIBRARY_PATH)])
    if mode == "kinematic":
        cmd.append("--kinematic")
    elif mode == "fixed-base":
        cmd.append("--fixed-base")
    elif mode == "C1":
        if seed is not None:
            cmd.extend(["--seed", str(seed)])
    elif mode == "C2":
        cmd.append("--c2-stabilizer")
        if seed is not None:
            cmd.extend(["--seed", str(seed)])
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return cmd


def _parse(out: str, mode: str, vx: float, seed: Optional[int],
           cmd: List[str]) -> RunResult:
    r = RunResult(mode=mode, vx=vx, seed=seed,
                  cmd_line=" ".join(cmd))

    m = re.search(r"Log saved to: (\S+)", out)
    if m:
        r.log_path = m.group(1)

    # Survival: TERMINATED at step N → N+1 ctrl steps; otherwise full horizon.
    m = re.search(r"TERMINATED at step (\d+)", out)
    if m:
        r.survival_steps = int(m.group(1)) + 1
    else:
        m = re.search(r"Summary \((\d+)/\d+ steps\)", out)
        if m:
            r.survival_steps = int(m.group(1))

    m = re.search(r"overall rmse\s*:\s*([0-9.]+) rad", out)
    if m:
        r.overall_rmse = float(m.group(1))

    m = re.search(r"worst joint\s*:\s*(\S+)\s+rmse=([0-9.]+) rad", out)
    if m:
        r.worst_joint = m.group(1)
        r.worst_joint_rmse = float(m.group(2))

    m = re.search(r"any-joint saturation:\s*([0-9.]+)% of steps", out)
    if m:
        r.any_sat_pct = float(m.group(1))

    # C2-only harness clip-sat lines.
    m = re.search(r"pitch_PD\s*:\s*([0-9.]+)%", out)
    if m:
        r.pitch_clip_pct = float(m.group(1))
    m = re.search(r"roll_PD\s*:\s*([0-9.]+)%", out)
    if m:
        r.roll_clip_pct = float(m.group(1))
    m = re.search(r"CP_nudge\s*:\s*([0-9.]+)%", out)
    if m:
        r.cp_clip_pct = float(m.group(1))
    # Aggregate "any" — the contract metric, union of channel clips.
    m = re.search(r"any\s*\(gate\)\s*:\s*([0-9.]+)%", out)
    if m:
        r.any_clip_pct = float(m.group(1))

    # Touchdown lines: "left  touchdowns= 6  stride=+0.0...m ... step=+0.0...m  realized/cmd=+1.04"
    for side, attr_t, attr_str, attr_step, attr_ratio in (
        ("left", "L_touchdowns", "L_stride_m", "L_step_m", "L_ratio"),
        ("right", "R_touchdowns", "R_stride_m", "R_step_m", "R_ratio"),
    ):
        line = re.search(
            rf"{side}\s+touchdowns=\s*(\d+)\s*"
            r"(?:stride=([+-]?[0-9.]+)m\s+\(min=[+-]?[0-9.]+\)\s+"
            r"step=([+-]?[0-9.]+)m\s+realized/cmd=([+-]?[0-9.]+))?",
            out,
        )
        if line:
            setattr(r, attr_t, int(line.group(1)))
            if line.group(2) is not None:
                setattr(r, attr_str, float(line.group(2)))
                setattr(r, attr_step, float(line.group(3)))
                setattr(r, attr_ratio, float(line.group(4)))

    # Fixed-base "count only" alternate format.
    if mode == "fixed-base":
        line = re.search(r"left\s+count=\s*(\d+)\s+right count=\s*(\d+)", out)
        if line:
            r.L_touchdowns = int(line.group(1))
            r.R_touchdowns = int(line.group(2))

    return r


def _score(r: RunResult) -> RunResult:
    """Annotate fail_reasons in place against C1 / C2 gates."""
    if r.mode == "kinematic":
        if r.L_step_m is None or r.L_step_m < _C2_STEP_LEN_BUDGET:
            r.fail_reasons.append(
                f"kinematic L step {r.L_step_m} < {_C2_STEP_LEN_BUDGET}"
            )
        if r.L_ratio is None or r.L_ratio < 0.95:
            r.fail_reasons.append(
                f"kinematic L ratio {r.L_ratio} < 0.95"
            )
    elif r.mode == "fixed-base":
        if r.overall_rmse is not None and r.overall_rmse > _RMSE_OVERALL_BUDGET:
            r.fail_reasons.append(
                f"overall RMSE {r.overall_rmse:.3f} > {_RMSE_OVERALL_BUDGET}"
            )
        if (r.worst_joint_rmse is not None
                and r.worst_joint_rmse > _RMSE_PER_JOINT_BUDGET):
            r.fail_reasons.append(
                f"worst-joint RMSE {r.worst_joint_rmse:.3f} > "
                f"{_RMSE_PER_JOINT_BUDGET} ({r.worst_joint})"
            )
        if r.any_sat_pct is not None and r.any_sat_pct > _FIXED_BASE_SAT_BUDGET * 100:
            r.fail_reasons.append(
                f"any-joint sat {r.any_sat_pct:.1f}% > "
                f"{_FIXED_BASE_SAT_BUDGET*100:.0f}%"
            )
    elif r.mode == "C1":
        if r.survival_steps is None or r.survival_steps < _C1_SURVIVAL_BUDGET:
            r.fail_reasons.append(
                f"survival {r.survival_steps} < {_C1_SURVIVAL_BUDGET} ctrl steps"
            )
        if (r.L_touchdowns or 0) < 1 or (r.R_touchdowns or 0) < 1:
            r.fail_reasons.append(
                f"touchdowns L={r.L_touchdowns} R={r.R_touchdowns}; need ≥ 1+1"
            )
    elif r.mode == "C2":
        if r.L_step_m is None or r.R_step_m is None:
            r.fail_reasons.append("not enough touchdowns for stride")
        else:
            if r.L_step_m < _C2_STEP_LEN_BUDGET or r.R_step_m < _C2_STEP_LEN_BUDGET:
                r.fail_reasons.append(
                    f"step length L={r.L_step_m:.4f} R={r.R_step_m:.4f} "
                    f"min < {_C2_STEP_LEN_BUDGET}"
                )
            if (r.L_ratio is None or r.R_ratio is None
                    or r.L_ratio < _C2_STEP_RATIO_BUDGET
                    or r.R_ratio < _C2_STEP_RATIO_BUDGET):
                r.fail_reasons.append(
                    f"realized/cmd ratio L={r.L_ratio} R={r.R_ratio} "
                    f"min < {_C2_STEP_RATIO_BUDGET}"
                )
        if r.any_sat_pct is not None and r.any_sat_pct >= _C2_SAT_HARDFAIL * 100:
            r.fail_reasons.append(
                f"any-joint sat {r.any_sat_pct:.1f}% >= "
                f"{_C2_SAT_HARDFAIL*100:.0f}% hard fail"
            )
        # Contract metric: aggregate "any harness output pinned" %.
        # Per-channel %s are reported but not gated separately
        # (see reference_design.md v0.20.0-C C2 metric gate, last
        # bullet under "Metric gate (C2 ...)").
        if (r.any_clip_pct is not None
                and r.any_clip_pct >= _C2_HARNESS_CLIP_HARDFAIL * 100):
            r.fail_reasons.append(
                f"harness any-channel clip {r.any_clip_pct:.1f}% >= "
                f"{_C2_HARNESS_CLIP_HARDFAIL*100:.0f}% hard fail"
            )
        if (r.worst_joint_rmse is not None
                and r.worst_joint_rmse > _RMSE_PER_JOINT_BUDGET):
            r.fail_reasons.append(
                f"worst-joint RMSE {r.worst_joint_rmse:.3f} > "
                f"{_RMSE_PER_JOINT_BUDGET} ({r.worst_joint})"
            )
    return r


def _emit_markdown(results: List[RunResult], harness_sha: str,
                   library_hash: str) -> str:
    lines: List[str] = []
    lines.append("## v0.20.0-C closeout artifact")
    lines.append("")
    lines.append(f"- harness commit sha: `{harness_sha}`")
    lines.append(f"- reference library hash: `{library_hash}`")
    lines.append(f"- horizon per row: {_HORIZON} ctrl steps "
                 f"({_HORIZON * _CTRL_DT:.1f} s)")
    lines.append(f"- C1 survival budget: "
                 f"`ceil({_C1_HORIZON_FACTOR} * {_CYCLE_TIME} / {_CTRL_DT}) "
                 f"= {_C1_SURVIVAL_BUDGET}` ctrl steps")
    lines.append(f"- seed family: numpy default_rng (seeds {list(_SEEDS)})")
    lines.append(f"- IC perturbation (Q2): pelvis x/y +/- 0.02 m, "
                 f"yaw +/- 5 deg, vx +/- 0.10 m/s")
    lines.append("")
    lines.append("### Raw matrix (32 measurement cases)")
    lines.append("")
    lines.append(
        "| mode | vx | seed | survived | L_td | R_td | L_step (m) | R_step (m) "
        "| L/cmd | R/cmd | overall RMSE | worst joint (RMSE) | any sat % "
        "| pitch_clip % | roll_clip % | CP_clip % | any_clip % | pass |"
    )
    lines.append(
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
    )
    for r in results:
        seed_str = "n/a" if r.seed is None else str(r.seed)
        worst = (f"{r.worst_joint}={r.worst_joint_rmse:.3f}"
                 if r.worst_joint else "—")
        passed = "✓" if not r.fail_reasons else "✗"
        lines.append(
            f"| {r.mode} | {r.vx:.3f} | {seed_str} "
            f"| {_fmt(r.survival_steps)} | {_fmt(r.L_touchdowns)} "
            f"| {_fmt(r.R_touchdowns)} | {_fmt(r.L_step_m, '.4f')} "
            f"| {_fmt(r.R_step_m, '.4f')} | {_fmt(r.L_ratio, '.2f')} "
            f"| {_fmt(r.R_ratio, '.2f')} | {_fmt(r.overall_rmse, '.4f')} "
            f"| {worst} | {_fmt(r.any_sat_pct, '.1f')} "
            f"| {_fmt(r.pitch_clip_pct, '.1f')} "
            f"| {_fmt(r.roll_clip_pct, '.1f')} "
            f"| {_fmt(r.cp_clip_pct, '.1f')} "
            f"| {_fmt(r.any_clip_pct, '.1f')} | {passed} |"
        )

    # Per-mode aggregate verdict.
    lines.append("")
    lines.append("### Aggregate verdict by mode")
    lines.append("")
    lines.append("| mode | rows | passed | failed | gate verdict |")
    lines.append("|---|---|---|---|---|")
    for mode in ("kinematic", "fixed-base", "C1", "C2"):
        rs = [r for r in results if r.mode == mode]
        passed = [r for r in rs if not r.fail_reasons]
        verdict = "PASS" if len(passed) == len(rs) else "FAIL"
        lines.append(f"| {mode} | {len(rs)} | {len(passed)} | "
                     f"{len(rs) - len(passed)} | **{verdict}** |")

    # Failure detail.
    failed_rows = [r for r in results if r.fail_reasons]
    if failed_rows:
        lines.append("")
        lines.append("### Failure detail")
        lines.append("")
        for r in failed_rows:
            seed_str = "n/a" if r.seed is None else str(r.seed)
            lines.append(f"- **{r.mode}** vx={r.vx:.3f} seed={seed_str}: "
                         + "; ".join(r.fail_reasons))

    # Command-line transcript.
    lines.append("")
    lines.append("### Command lines (per row)")
    lines.append("")
    lines.append("```")
    for r in results:
        seed_str = "n/a" if r.seed is None else str(r.seed)
        lines.append(f"# {r.mode} vx={r.vx:.3f} seed={seed_str}")
        lines.append(r.cmd_line)
    lines.append("```")
    return "\n".join(lines)


def _fmt(v, spec: str = "") -> str:
    if v is None:
        return "—"
    if spec:
        return f"{v:{spec}}"
    return str(v)


def _build_pinned_library(repo_root: Path) -> Path:
    """Pre-generate a ZMP library covering exactly _VX_BINS and save it.

    The viewer's default ``gen.build_library()`` uses ``interval=0.05``
    which produces bins {0.0, 0.05, 0.10, 0.15, 0.20, 0.25}.  Building
    a pinned library with exactly ``_VX_BINS`` and passing
    ``--library-path`` to each viewer invocation guarantees the
    closeout actually tests the bins it claims to test, even if a
    future operating-point shift introduces non-grid values (the
    historical Phase 9A bins 0.265 and 0.30 were the original failure
    case this guard was added for).
    """
    sys.path.insert(0, str(repo_root))
    from control.zmp.zmp_walk import ZMPWalkGenerator  # noqa: WPS433

    out_dir = Path("/tmp/v0200c_closeout_lib")
    print(
        f"Pre-generating pinned ZMP library for closeout bins "
        f"{list(_VX_BINS)} -> {out_dir} ...",
        file=sys.stderr,
    )
    gen = ZMPWalkGenerator()
    lib = gen.build_library_for_vx_values(list(_VX_BINS))
    lib.save(str(out_dir))
    print(f"  saved library at {out_dir}", file=sys.stderr)
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=str, default="/tmp/v0200c_artifact.md")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the matrix without running anything")
    args = ap.parse_args()

    global _RUNNER, _LIBRARY_PATH
    _RUNNER = _detect_runner()
    print(f"Viewer runner: {' '.join(_RUNNER)}", file=sys.stderr)

    repo_root = Path(__file__).resolve().parents[1]
    harness_sha = _shell(
        ["git", "rev-parse", "--short=10", "HEAD"], check=False
    ).strip()
    # Library hash: hash of the generator config + library entries.
    # ZMP libraries are regenerated on every run; instead emit the hash
    # of the generator module file as the deterministic source-of-truth.
    zmp_walk_path = repo_root / "control" / "zmp" / "zmp_walk.py"
    library_hash = _shell(
        ["git", "hash-object", str(zmp_walk_path)], check=False
    ).strip()[:10]

    plan: List = []
    for vx in _VX_BINS:
        plan.append(("kinematic", vx, None))
    for vx in _VX_BINS:
        plan.append(("fixed-base", vx, None))
    for vx in _VX_BINS:
        for s in _SEEDS:
            plan.append(("C1", vx, s))
    for vx in _VX_BINS:
        for s in _SEEDS:
            plan.append(("C2", vx, s))

    if args.dry_run:
        for mode, vx, seed in plan:
            print(f"{mode:11s} vx={vx:.3f} seed={seed}")
        print(f"total: {len(plan)} rows")
        return

    # Build the pinned library once before launching subprocesses so all
    # 32 rows use the same on-disk asset (and so the asset is generated
    # against the current git checkout, not whatever happens to be in
    # the viewer's default cache).
    _LIBRARY_PATH = _build_pinned_library(repo_root)

    results: List[RunResult] = []
    print(f"Running {len(plan)} rows ...", file=sys.stderr)
    for i, (mode, vx, seed) in enumerate(plan, 1):
        cmd = _build_cmd(mode, vx, seed)
        print(f"  [{i:2d}/{len(plan)}] {mode:11s} vx={vx:.3f} "
              f"seed={seed} ...", file=sys.stderr, end=" ", flush=True)
        try:
            out = _shell(cmd, check=True)
        except SubprocessFailure as exc:
            # Surface the failure immediately — silent rows produced
            # the round-1 false-pass count when mjpython was missing
            # locally.  Re-raise so the artifact never claims a row
            # that didn't actually run.
            print("SUBPROCESS FAILED", file=sys.stderr)
            raise SystemExit(
                f"\nv0.20.0-C closeout aborted at row {i}/{len(plan)} "
                f"({mode} vx={vx:.3f} seed={seed}).\n{exc}"
            )
        r = _score(_parse(out, mode, vx, seed, cmd))
        results.append(r)
        verdict = "✓" if not r.fail_reasons else "✗"
        survived = r.survival_steps if r.survival_steps is not None else "—"
        print(f"survived={survived} {verdict}", file=sys.stderr)

    out_md = _emit_markdown(results, harness_sha, library_hash)
    Path(args.output).write_text(out_md)
    print(f"\nArtifact written to {args.output}", file=sys.stderr)
    print(f"Total: {sum(1 for r in results if not r.fail_reasons)}/"
          f"{len(results)} rows pass.", file=sys.stderr)


if __name__ == "__main__":
    main()
