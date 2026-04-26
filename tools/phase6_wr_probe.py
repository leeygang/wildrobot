#!/usr/bin/env python3
"""Phase 6 WR-only probe: run the parity tool's WR-side helpers without
needing the TB venv.

Reports the load-bearing FK / smoothness / geometry numbers for WR at
``vx=0.15`` so a tuning change can be compared before/after.  The TB
side is read from the cached ``tools/parity_report.json`` (TB code is
untouched between Phase 1 and Phase 6, so cached TB numbers stay valid).
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from control.zmp.zmp_walk import ZMPWalkConfig
from tools.reference_geometry_parity import (  # noqa: E402
    _GRAVITY_MPS2,
    _summarize_wildrobot,
    _wr_dims,
    _wr_fk_and_smoothness,
)


def main() -> int:
    cfg = ZMPWalkConfig()
    print(f"WR cycle_time_s        = {cfg.cycle_time_s:.4f}")
    print(f"WR foot_step_height_m  = {cfg.foot_step_height_m:.4f}")
    print(f"WR com_height_m        = {cfg.com_height_m:.4f}")
    print(f"WR leg_length_m        = {cfg.upper_leg_m + cfg.lower_leg_m:.4f}")

    geom = _summarize_wildrobot()
    print()
    print("=== WR P0 geometry summary ===")
    print(f"  total failures      = {geom.total_failures}")
    print(f"  in-scope failures   = {geom.in_scope_failures}")
    print(f"  worst stance z (m)  = {geom.worst_stance_z:+.4f}")
    print(f"  worst swing z (m)   = {geom.worst_swing_z:+.4f}")

    gait, smooth = _wr_fk_and_smoothness(0.15)
    print()
    print("=== WR P1A FK gait at vx=0.15 ===")
    print(f"  step_length_mean_m       = {gait.step_length_mean_m:.4f}")
    print(f"  swing_clearance_mean_m   = {gait.swing_clearance_mean_m:.4f}")
    print(f"  swing_clearance_min_m    = {gait.swing_clearance_min_m:.4f}")
    print(f"  touchdown_rate_hz        = {gait.touchdown_rate_hz:.4f}")
    print(f"  touchdown_speed_proxy_mps= {gait.touchdown_speed_proxy_mps:.4f}")
    print(f"  double_support_frac      = {gait.double_support_frac:.4f}")
    print(f"  foot_z_step_max_m        = {gait.foot_z_step_max_m:.4f}")

    print()
    print("=== WR P2 smoothness at vx=0.15 ===")
    print(f"  swing_foot_z_step_max_m  = {smooth.swing_foot_z_step_max_m:.4f}")
    print(f"  shared_leg_q_step_max_rad= {smooth.shared_leg_q_step_max_rad:.4f}")
    print(f"  pelvis_z_step_max_m      = {smooth.pelvis_z_step_max_m:.4f}")
    print(f"  contact_flips_per_cycle  = {smooth.contact_flips_per_cycle:.4f}")

    dims = _wr_dims()
    leg = dims["leg_length_m"]
    com_h = dims["com_height_m"]
    print()
    print("=== WR size-normalised view at vx=0.15 ===")
    print(f"  step_length_per_leg               = {gait.step_length_mean_m / leg:.4f}")
    print(f"  swing_clearance_per_com_height    = {gait.swing_clearance_mean_m / com_h:.4f}")
    print(f"  cadence_froude_norm               = {gait.touchdown_rate_hz * (com_h / _GRAVITY_MPS2) ** 0.5:.4f}")
    if gait.swing_clearance_mean_m > 0:
        print(f"  swing_foot_z_step_per_clearance   = {smooth.swing_foot_z_step_max_m / gait.swing_clearance_mean_m:.4f}")

    out = {
        "cfg": {
            "cycle_time_s": cfg.cycle_time_s,
            "foot_step_height_m": cfg.foot_step_height_m,
            "com_height_m": cfg.com_height_m,
            "leg_length_m": cfg.upper_leg_m + cfg.lower_leg_m,
        },
        "geometry": asdict(geom),
        "fk_gait": asdict(gait),
        "smoothness": asdict(smooth),
        "dims": dims,
    }
    out_path = Path(__file__).resolve().parent / "phase6_wr_probe.json"
    out_path.write_text(json.dumps(out, indent=2))
    print()
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
