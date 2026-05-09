#!/usr/bin/env python3
"""Phase 6 WR-only probe.

Scope: this probe only measures WR-side P0 geometry, P1A FK gait, and P2
smoothness at a single nominal ``vx`` (default 0.15) by re-using the parity
tool's WR-side helpers.  It is a tuning-slice before/after tool, not a
full-parity replacement.

Explicit non-goals:
- Does NOT run TB itself (TB code is untouched, but TB still needs its own
  venv with ``joblib`` / ``lz4`` to recompute its baseline).
- Does NOT measure P1 closed-loop trackability under the current WR config.
  P1 needs the full ``tools/reference_geometry_parity.py`` run.
- Does NOT re-judge the doc's "on par or better" decision rule
  (``training/docs/reference_architecture_comparison.md`` § "What 'On Par
  or Better' Means").  That rule requires P0 + normalised P1A + P2 + P1 +
  G4/G7 all to pass; this probe covers only the first three layers.

For convenience the probe loads the **cached** TB numbers from
``tools/parity_report.json`` (commit ``c1b815d`` baseline, TB-side
unchanged between Phase 1 and Phase 6) and prints them next to WR so
before/after tuning runs can read the absolute and per-clearance ratios
directly.  Treat the TB numbers as cached, not freshly measured.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from control.zmp.zmp_walk import ZMPWalkConfig, ZMPWalkGenerator
from tools.reference_geometry_parity import (  # noqa: E402
    _GRAVITY_MPS2,
    _P1_VX_BINS,
    _VX_BINS,
    _summarize_wildrobot,
    _wr_dims,
    _wr_fk_and_smoothness,
)


def _load_cached_tb(parity_path: Path, baseline: str, vx: float) -> dict:
    """Return cached TB FK + smoothness + normalised metrics at ``vx``.

    Returns an empty dict if the cache file is missing so the probe still
    runs WR-only on a fresh checkout.  Looks up the named TB baseline
    (``toddlerbot_2xc`` or ``toddlerbot_2xm``) at the requested ``vx``.
    """
    if not parity_path.exists():
        return {}
    cached = json.loads(parity_path.read_text())
    out: dict = {"source": str(parity_path)}
    for key in ("fk_gait_metrics", "smoothness_metrics", "normalized_fk_metrics"):
        for entry in cached.get(key, []):
            if entry.get("name") == baseline and abs(float(entry.get("vx", -1)) - vx) < 1e-6:
                out[key] = entry
                break
    return out


def _fmt_ratio(wr_val: float, tb_val: float, gate_dir: str, gate_factor: float) -> str:
    """Return ``"WR/TB=R.RRRx (gate ≤/≥ Rx → PASS/FAIL)"`` for printing."""
    if tb_val == 0 or wr_val != wr_val or tb_val != tb_val:  # NaN-safe
        return "n/a (TB=0 or NaN)"
    ratio = wr_val / tb_val
    if gate_dir == "le":
        verdict = "PASS" if ratio <= gate_factor else "FAIL"
        gate = f"≤ {gate_factor}×"
    else:
        verdict = "PASS" if ratio >= gate_factor else "FAIL"
        gate = f"≥ {gate_factor}×"
    return f"{ratio:.3f}× TB ({gate} → {verdict})"


def main() -> int:
    cfg = ZMPWalkConfig()
    print(f"WR cycle_time_s        = {cfg.cycle_time_s:.4f}")
    print(f"WR foot_step_height_m  = {cfg.foot_step_height_m:.4f}")
    print(f"WR com_height_m        = {cfg.com_height_m:.4f}")
    print(f"WR leg_length_m        = {cfg.upper_leg_m + cfg.lower_leg_m:.4f}")

    # Phase 5 requires an explicit pre-built library across the bins
    # used by both the geometry summary (_VX_BINS) and the FK/smoothness
    # probe at the nominal vx.
    required_bins = sorted({round(float(v), 4) for v in (*_VX_BINS, *_P1_VX_BINS, 0.265)})
    lib = ZMPWalkGenerator().build_library_for_vx_values(required_bins)

    geom = _summarize_wildrobot(lib)
    print()
    print("=== WR P0 geometry summary ===")
    print(f"  total failures      = {geom.total_failures}")
    print(f"  in-scope failures   = {geom.in_scope_failures}")
    print(f"  worst stance z (m)  = {geom.worst_stance_z:+.4f}")
    print(f"  worst swing z (m)   = {geom.worst_swing_z:+.4f}")

    gait, smooth = _wr_fk_and_smoothness(lib, 0.265)
    print()
    print("=== WR P1A FK gait at vx=0.265 ===")
    print(f"  step_length_mean_m       = {gait.step_length_mean_m:.4f}")
    print(f"  swing_clearance_mean_m   = {gait.swing_clearance_mean_m:.4f}")
    print(f"  swing_clearance_min_m    = {gait.swing_clearance_min_m:.4f}")
    print(f"  touchdown_rate_hz        = {gait.touchdown_rate_hz:.4f}")
    print(f"  touchdown_speed_proxy_mps= {gait.touchdown_speed_proxy_mps:.4f}")
    print(f"  double_support_frac      = {gait.double_support_frac:.4f}")
    print(f"  foot_z_step_max_m        = {gait.foot_z_step_max_m:.4f}")

    print()
    print("=== WR P2 smoothness at vx=0.265 ===")
    print(f"  swing_foot_z_step_max_m  = {smooth.swing_foot_z_step_max_m:.4f}")
    print(f"  shared_leg_q_step_max_rad= {smooth.shared_leg_q_step_max_rad:.4f}")
    print(f"  pelvis_z_step_max_m      = {smooth.pelvis_z_step_max_m:.4f}")
    print(f"  contact_flips_per_cycle  = {smooth.contact_flips_per_cycle:.4f}")

    dims = _wr_dims()
    leg = dims["leg_length_m"]
    com_h = dims["com_height_m"]
    wr_step_per_leg = gait.step_length_mean_m / leg
    wr_swing_per_h = gait.swing_clearance_mean_m / com_h
    wr_cadence_norm = gait.touchdown_rate_hz * (com_h / _GRAVITY_MPS2) ** 0.5
    wr_step_per_clr = (
        smooth.swing_foot_z_step_max_m / gait.swing_clearance_mean_m
        if gait.swing_clearance_mean_m > 0
        else float("nan")
    )
    print()
    print("=== WR size-normalised view at vx=0.265 ===")
    print(f"  step_length_per_leg               = {wr_step_per_leg:.4f}")
    print(f"  swing_clearance_per_com_height    = {wr_swing_per_h:.4f}")
    print(f"  cadence_froude_norm               = {wr_cadence_norm:.4f}")
    print(f"  swing_foot_z_step_per_clearance   = {wr_step_per_clr:.4f}")

    # Cached TB-2xc comparison (TB code untouched between Phase 1 and
    # Phase 6, so cached numbers are still the right baseline).  This
    # comparison covers P1A (FK gait) and P2 (smoothness) only — P1
    # closed-loop trackability is OUT OF SCOPE for this probe.
    parity_path = Path(__file__).resolve().parent / "parity_report.json"
    tb = _load_cached_tb(parity_path, "toddlerbot_2xc", 0.15)  # TB still operates at vx=0.15; this is the baseline we match against via step/leg scaling
    print()
    if tb:
        fk_tb = tb.get("fk_gait_metrics", {})
        sm_tb = tb.get("smoothness_metrics", {})
        nf_tb = tb.get("normalized_fk_metrics", {})
        print(f"=== TB-2xc cached @ vx=0.265 (source: {tb['source']}) ===")
        print(f"  step_length_mean_m       = {fk_tb.get('step_length_mean_m', float('nan')):.4f}")
        print(f"  swing_clearance_mean_m   = {fk_tb.get('swing_clearance_mean_m', float('nan')):.4f}")
        print(f"  touchdown_rate_hz        = {fk_tb.get('touchdown_rate_hz', float('nan')):.4f}")
        print(f"  swing_foot_z_step_max_m  = {sm_tb.get('swing_foot_z_step_max_m', float('nan')):.4f}")
        print(f"  shared_leg_q_step_max_rad= {sm_tb.get('shared_leg_q_step_max_rad', float('nan')):.4f}")
        print(f"  step_length_per_leg      = {nf_tb.get('step_length_per_leg', float('nan')):.4f}")
        print(f"  swing_clearance_per_com_height  = {nf_tb.get('swing_clearance_per_com_height', float('nan')):.4f}")
        print(f"  cadence_froude_norm      = {nf_tb.get('cadence_froude_norm', float('nan')):.4f}")
        print(f"  swing_foot_z_step_per_clearance = {nf_tb.get('swing_foot_z_step_per_clearance', float('nan')):.4f}")
        print()
        print("=== WR vs TB-2xc gate verdicts (P1A + P2) ===")
        print(f"  P1A step_length_mean_m (≥0.90× TB): {_fmt_ratio(gait.step_length_mean_m, fk_tb.get('step_length_mean_m', 0.0), 'ge', 0.90)}")
        print(f"  P1A swing_clearance_mean_m (≥0.85× TB): {_fmt_ratio(gait.swing_clearance_mean_m, fk_tb.get('swing_clearance_mean_m', 0.0), 'ge', 0.85)}")
        print(f"  P1A touchdown_rate_hz (≤1.20× TB):  {_fmt_ratio(gait.touchdown_rate_hz, fk_tb.get('touchdown_rate_hz', 0.0), 'le', 1.20)}")
        print(f"  P2  swing_foot_z_step_max_m (≤1.10× TB): {_fmt_ratio(smooth.swing_foot_z_step_max_m, sm_tb.get('swing_foot_z_step_max_m', 0.0), 'le', 1.10)}")
        print(f"  P2  shared_leg_q_step_max (≤1.10× TB):   {_fmt_ratio(smooth.shared_leg_q_step_max_rad, sm_tb.get('shared_leg_q_step_max_rad', 0.0), 'le', 1.10)}")
        print(f"  norm step_length_per_leg (≥0.85× TB):    {_fmt_ratio(wr_step_per_leg, nf_tb.get('step_length_per_leg', 0.0), 'ge', 0.85)}")
        print(f"  norm swing_clearance_per_h (≥0.85× TB):  {_fmt_ratio(wr_swing_per_h, nf_tb.get('swing_clearance_per_com_height', 0.0), 'ge', 0.85)}")
        print(f"  norm cadence_froude_norm (≤1.20× TB):    {_fmt_ratio(wr_cadence_norm, nf_tb.get('cadence_froude_norm', 0.0), 'le', 1.20)}")
        print(f"  norm swing_step_per_clearance (≤1.10× TB): {_fmt_ratio(wr_step_per_clr, nf_tb.get('swing_foot_z_step_per_clearance', 0.0), 'le', 1.10)}")
        print()
        print("NOTE: P1 closed-loop trackability is NOT measured by this probe.")
        print("      Run `uv run ./tools/reference_geometry_parity.py` for P1.")
    else:
        print(f"(no cached TB baseline at {parity_path}; skipping WR-vs-TB table)")

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
        "normalized": {
            "step_length_per_leg": wr_step_per_leg,
            "swing_clearance_per_com_height": wr_swing_per_h,
            "cadence_froude_norm": wr_cadence_norm,
            "swing_foot_z_step_per_clearance": wr_step_per_clr,
        },
        "dims": dims,
        "tb_cached_2xc": tb,
        "scope": {
            "measured": ["P0", "P1A", "P2", "size-normalised P1A/P2"],
            "not_measured": ["P1 closed-loop trackability"],
        },
    }
    out_path = Path(__file__).resolve().parent / "phase6_wr_probe.json"
    out_path.write_text(json.dumps(out, indent=2))
    print()
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
