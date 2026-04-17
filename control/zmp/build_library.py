#!/usr/bin/env python3
"""Build a ZMP reference library for WildRobot v2.

Generates a command-indexed offline reference library using ZMP preview
control (Kajita et al.) and saves it to disk.

Usage:
  uv run python control/zmp/build_library.py --output /tmp/zmp_ref_library
  uv run python control/zmp/build_library.py --output /tmp/zmp_ref_library --vx-max 0.20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from control.zmp.zmp_walk import ZMPWalkConfig, ZMPWalkGenerator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build ZMP reference library for WildRobot v2")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for the library")
    parser.add_argument("--vx-min", type=float, default=0.0)
    parser.add_argument("--vx-max", type=float, default=0.25)
    parser.add_argument("--vx-interval", type=float, default=0.05)
    parser.add_argument("--cycle-time", type=float, default=0.50)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--com-height", type=float, default=0.42)
    parser.add_argument("--step-height", type=float, default=0.025)
    args = parser.parse_args()

    cfg = ZMPWalkConfig(
        cycle_time_s=args.cycle_time,
        dt_s=args.dt,
        com_height_m=args.com_height,
        foot_step_height_m=args.step_height,
    )

    gen = ZMPWalkGenerator(cfg)

    print(f"Building ZMP reference library:")
    print(f"  vx range: [{args.vx_min}, {args.vx_max}] at {args.vx_interval} m/s")
    print(f"  cycle_time={cfg.cycle_time_s}s  dt={cfg.dt_s}s  com_height={cfg.com_height_m}m")
    print()

    lib = gen.build_library(
        command_range_vx=(args.vx_min, args.vx_max),
        interval=args.vx_interval,
    )

    # Validate
    issues = lib.validate()
    if issues:
        print(f"\nValidation issues found:")
        for key, entry_issues in issues.items():
            print(f"  {key}: {entry_issues}")
    else:
        print(f"\nAll {len(lib)} entries pass validation.")

    # Save
    lib.save(args.output)
    print(f"\nSaved to {args.output}")
    print(lib.summary())


if __name__ == "__main__":
    main()
