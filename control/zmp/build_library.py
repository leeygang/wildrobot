#!/usr/bin/env python3
"""Build a ZMP reference library for WildRobot v2.

Generates a command-indexed offline reference library using ZMP preview
control (Kajita et al.) and saves it to disk.

Usage:
  # 1D (vx-only) library — v0.20.x default
  uv run python control/zmp/build_library.py --output /tmp/zmp_ref_library
  uv run python control/zmp/build_library.py --output /tmp/zmp_ref_library --vx-max 0.20

  # 3D (vx + vy + yaw-rate) library — v0.21.0 lateral/yaw prior
  # NOTE: bin lists that contain negative values MUST use the
  #   --flag=value (equals-sign) form. With a plain space, argparse mis-
  #   parses the leading '-' as the next flag and errors out.
  #
  # Canonical grid (4 vx x 5 vy x 5 wz -> 24 bins after TB-split dedup):
  uv run python control/zmp/build_library.py --output /tmp/v0210_lib --mode 3d \\
      --vx-bins=0.0,0.18,0.20,0.22,0.26 \\
      --vy-bins=-0.10,-0.05,0.0,0.05,0.10 \\
      --yaw-rate-bins=-0.20,-0.10,0.0,0.10,0.20

  # Same grid built from symmetric ranges instead of explicit bins
  # (no negative literals -> bare flags are fine):
  uv run python control/zmp/build_library.py --output /tmp/v0210_lib --mode 3d \\
      --vx-bins=0.0,0.18,0.20,0.22,0.26 \\
      --vy-max 0.10 --vy-interval 0.05 \\
      --yaw-rate-max 0.20 --yaw-rate-interval 0.10

  # Minimal smoke (1 vx, 2 vy, 1 wz=0 -> a handful of bins):
  uv run python control/zmp/build_library.py --output /tmp/v0210_lib_smoke --mode 3d \\
      --vx-bins=0.0,0.20 --vy-bins=0.0,0.10 --yaw-rate-bins=0.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from control.zmp.zmp_walk import ZMPWalkConfig, ZMPWalkGenerator


def _parse_bins(spec: str) -> list[float]:
    return [float(v.strip()) for v in spec.split(",") if v.strip()]


def _linear_range(vmin: float, vmax: float, interval: float) -> list[float]:
    """Inclusive range [vmin, vmax] stepped by interval. Used for vx."""
    if interval <= 0:
        raise ValueError(f"interval must be > 0, got {interval}")
    if vmax < vmin:
        raise ValueError(f"vmax ({vmax}) < vmin ({vmin})")
    n = int(round((vmax - vmin) / interval))
    return [round(vmin + i * interval, 6) for i in range(n + 1)]


def _symmetric_range(maxv: float, interval: float) -> list[float]:
    """Symmetric range [-maxv, ..., 0, ..., +maxv]. maxv=0 -> [0.0]."""
    if maxv < 0:
        raise ValueError(f"max must be >= 0, got {maxv}")
    if maxv == 0:
        return [0.0]
    if interval <= 0:
        raise ValueError(f"interval must be > 0, got {interval}")
    n = int(round(maxv / interval))
    return [round(i * interval, 6) for i in range(-n, n + 1)]


def main() -> None:
    # Canonical defaults live in ZMPWalkConfig (the planner). build_library.py
    # is a thin CLI wrapper and MUST NOT override them silently; previous
    # cycle_time=0.50 / step_height=0.025 defaults drifted from the
    # 0.96 / 0.05 planner defaults and produced libraries whose lateral
    # references had insufficient hip-roll authority (~4.6 deg vs the
    # ~6.7 deg geometric lean threshold), causing visible backward drift
    # in --vy playback. See CHANGELOG v0.21.0 lateral-prior diagnostic.
    _DEFAULT_CFG = ZMPWalkConfig()

    parser = argparse.ArgumentParser(
        description="Build ZMP reference library for WildRobot v2")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for the library")
    parser.add_argument(
        "--mode",
        choices=["1d", "3d"],
        default="1d",
        help=("'1d' (default): vx-only library. '3d': TB-split "
              "(vx, vy, 0) linear + (0, 0, wz) pure-yaw bins, dedup at zero."),
    )
    # vx axis (used in both modes).
    parser.add_argument("--vx-min", type=float, default=0.0)
    parser.add_argument("--vx-max", type=float, default=0.25)
    parser.add_argument("--vx-interval", type=float, default=0.05)
    parser.add_argument(
        "--vx-bins",
        type=str,
        default="",
        help=(
            "Optional comma-separated explicit vx bins (e.g. "
            "'0.10,0.15,0.20,0.25'). Overrides --vx-min/--vx-max/--vx-interval."
        ),
    )
    # vy axis (3d mode only). Symmetric around 0; vy-max=0 collapses to [0.0].
    parser.add_argument("--vy-max", type=float, default=0.0,
                        help="3d mode: symmetric vy half-range (m/s). 0 -> [0.0].")
    parser.add_argument("--vy-interval", type=float, default=0.05)
    parser.add_argument(
        "--vy-bins",
        type=str,
        default="",
        help=("3d mode: optional comma-separated explicit vy bins (e.g. "
              "'-0.10,-0.05,0.0,0.05,0.10'). Overrides --vy-max/--vy-interval."),
    )
    # yaw-rate axis (3d mode only). Symmetric around 0; max=0 -> [0.0].
    parser.add_argument("--yaw-rate-max", type=float, default=0.0,
                        dest="yaw_rate_max",
                        help="3d mode: symmetric yaw-rate half-range (rad/s).")
    parser.add_argument("--yaw-rate-interval", type=float, default=0.10,
                        dest="yaw_rate_interval")
    parser.add_argument(
        "--yaw-rate-bins",
        type=str,
        default="",
        dest="yaw_rate_bins",
        help=("3d mode: optional comma-separated explicit yaw-rate bins (e.g. "
              "'-0.20,-0.10,0.0,0.10,0.20'). Overrides --yaw-rate-max/-interval."),
    )
    # ZMPWalkConfig knobs (apply to both modes). Defaults pulled from
    # ZMPWalkConfig() so build_library cannot drift from the planner.
    parser.add_argument("--cycle-time", type=float,
                        default=_DEFAULT_CFG.cycle_time_s,
                        help=f"Walking cycle time (default "
                             f"{_DEFAULT_CFG.cycle_time_s}s, Froude-scaled "
                             f"to WR CoM height)")
    parser.add_argument("--dt", type=float, default=_DEFAULT_CFG.dt_s)
    parser.add_argument("--step-height", type=float,
                        default=_DEFAULT_CFG.foot_step_height_m,
                        help=f"Swing-foot peak height (default "
                             f"{_DEFAULT_CFG.foot_step_height_m}m)")
    args = parser.parse_args()

    cfg = ZMPWalkConfig(
        cycle_time_s=args.cycle_time,
        dt_s=args.dt,
        foot_step_height_m=args.step_height,
    )

    gen = ZMPWalkGenerator(cfg)

    print(f"Building ZMP reference library (mode={args.mode}):")
    print(f"  cycle_time={cfg.cycle_time_s}s  dt={cfg.dt_s}s  "
          f"com_height={cfg.com_height_m:.4f}m (derived)")

    if args.mode == "1d":
        if args.vx_bins.strip():
            vx_bins = _parse_bins(args.vx_bins)
            print(f"  explicit vx bins: {vx_bins}")
            lib = gen.build_library_for_vx_values(vx_bins, interval=args.vx_interval)
        else:
            print(f"  vx range: [{args.vx_min}, {args.vx_max}] at "
                  f"{args.vx_interval} m/s")
            lib = gen.build_library(
                command_range_vx=(args.vx_min, args.vx_max),
                interval=args.vx_interval,
            )
    else:  # 3d
        if args.vx_bins.strip():
            vx_bins = _parse_bins(args.vx_bins)
        else:
            vx_bins = _linear_range(args.vx_min, args.vx_max, args.vx_interval)

        if args.vy_bins.strip():
            vy_bins = _parse_bins(args.vy_bins)
        else:
            vy_bins = _symmetric_range(args.vy_max, args.vy_interval)

        if args.yaw_rate_bins.strip():
            wz_bins = _parse_bins(args.yaw_rate_bins)
        else:
            wz_bins = _symmetric_range(args.yaw_rate_max, args.yaw_rate_interval)

        print(f"  vx bins ({len(vx_bins)}): {vx_bins}")
        print(f"  vy bins ({len(vy_bins)}): {vy_bins}")
        print(f"  yaw-rate bins ({len(wz_bins)}): {wz_bins}")
        n_linear = len(vx_bins) * len(vy_bins)
        n_yaw = len(wz_bins)
        n_dedup = 1 if (0.0 in vx_bins and 0.0 in vy_bins and 0.0 in wz_bins) else 0
        print(f"  expected entries: {n_linear} linear + {n_yaw} pure-yaw "
              f"- {n_dedup} dedup = {n_linear + n_yaw - n_dedup}")
        print()
        lib = gen.build_library_for_3d_values(
            vx_values=vx_bins,
            vy_values=vy_bins,
            yaw_rate_values=wz_bins,
        )

    # Validate
    issues = lib.validate()
    if issues:
        print(f"\nValidation issues found:")
        for key, entry_issues in issues.items():
            print(f"  {key}: {entry_issues}")
        print("\nLibrary NOT saved due to validation failures.")
        sys.exit(1)
    else:
        print(f"\nAll {len(lib)} entries pass validation.")

    # Save
    lib.save(args.output)
    print(f"\nSaved to {args.output}")
    print(lib.summary())


if __name__ == "__main__":
    main()
