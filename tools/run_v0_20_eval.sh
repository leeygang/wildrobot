#!/usr/bin/env bash
# Run the full v0.20 re-evaluation suite end-to-end.
#
# Captures the three load-bearing diagnostics that populate the
# `v0.20.1-reference-quality-snapshot` table in
# `training/CHANGELOG.md`:
#
#   1. Phase 10 closed-loop diagnostic (WR-only) at vx ∈ {0.10, 0.15, 0.20}
#      — closed-loop survival, contact-match, hip_roll saturation
#   2. v0.20.0-C per-frame deterministic probe at vx=0.15
#      — G7 baseline (lever-flip, pelvis-x lag, free-float survival)
#   3. Full WR vs TB reference geometry parity report
#      — every absolute + normalised P0 / P1A / P2 / P1 closed-loop gate
#
# All three artifacts are written to /tmp/v0_20_eval_<timestamp>/ for
# archiving alongside the canonical `tools/parity_report.json` (which
# the parity tool overwrites on every successful run).
#
# Usage (run from the wildrobot repo root):
#   bash tools/run_v0_20_eval.sh
#
# Requirements:
#   - `uv` available on PATH (project standard)
#   - For step 3 only: ToddlerBot venv reachable to
#     `tools/reference_geometry_parity.py::_pick_python_with_modules`.
#     If unreachable, step 3 will exit non-zero with a clear error
#     message; steps 1 + 2 still complete and their artifacts remain
#     in the output directory.
#
# After running, paste the contents of:
#   - ${OUT_DIR}/phase10.log          (Phase 10 summary tables)
#   - ${OUT_DIR}/per_frame_probe.log  (G7 baseline)
#   - ${OUT_DIR}/parity_report.log    (WR vs TB gate verdicts)
# back into the working session to refresh the snapshot table in
# `training/CHANGELOG.md`.

set -euo pipefail

TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="/tmp/v0_20_eval_${TS}"
mkdir -p "${OUT_DIR}"

echo "================================================================"
echo "WildRobot v0.20 re-evaluation suite"
echo "Started:          $(date)"
echo "Output directory: ${OUT_DIR}"
echo "================================================================"

run_step () {
    local label="$1"
    local logfile="$2"
    shift 2
    echo
    echo "=== ${label} ==="
    echo "Logging to: ${logfile}"
    echo "Command:    $*"
    echo
    # `set +e` so a failed step doesn't abort the whole script — the
    # parity step in particular may fail when the TB venv is
    # unreachable, but the phase10 + per-frame artifacts should still
    # be saved.
    set +e
    "$@" 2>&1 | tee "${logfile}"
    local rc="${PIPESTATUS[0]}"
    set -e
    if [[ "${rc}" -ne 0 ]]; then
        echo "  [warn] step exited with rc=${rc}; continuing"
    fi
}

# ----------------------------------------------------------------------
# 1. Phase 10 closed-loop diagnostic (WR-only)
# ----------------------------------------------------------------------
# vx bins are inherited from `phase10_diagnostic.py`'s default
# (post Phase 9A: [0.25, 0.265, 0.30] — bracket around the new
# operating point).  Override with V0_20_PHASE10_VX="..." if you
# need a different bracket (e.g. for cross-vx regression checks).
run_step \
    "[1/3] Phase 10 closed-loop diagnostic (vx defaults from phase10_diagnostic.py)" \
    "${OUT_DIR}/phase10.log" \
    uv run python tools/phase10_diagnostic.py \
        ${V0_20_PHASE10_VX:+--vx ${V0_20_PHASE10_VX}} \
        --horizon 200 \
        --wr-library-rebuild \
        --json-out "${OUT_DIR}/phase10.json"

# ----------------------------------------------------------------------
# 2. Per-frame G7 deterministic probe
# ----------------------------------------------------------------------
# vx is inherited from `v0200c_per_frame_probe.py`'s default (post
# Phase 9A: 0.265, the operating point).  Override with
# V0_20_PROBE_VX=0.15 if you need a different bin.
run_step \
    "[2/3] Per-frame G7 deterministic probe (vx default from v0200c_per_frame_probe.py)" \
    "${OUT_DIR}/per_frame_probe.log" \
    uv run python tools/v0200c_per_frame_probe.py \
        ${V0_20_PROBE_VX:+--vx ${V0_20_PROBE_VX}} --horizon 200

# ----------------------------------------------------------------------
# 3. Full WR vs TB parity report (needs TB venv)
# ----------------------------------------------------------------------
run_step \
    "[3/3] WR vs TB reference geometry parity report" \
    "${OUT_DIR}/parity_report.log" \
    uv run python tools/reference_geometry_parity.py

# Copy the canonical parity_report.json artifact (overwritten by the
# parity tool on every successful run) into the timestamped output
# dir so this run's exact verdicts stay archived.
if [[ -f tools/parity_report.json ]]; then
    cp tools/parity_report.json "${OUT_DIR}/parity_report.json"
fi

echo
echo "================================================================"
echo "Done.  Artifacts saved in:"
ls -la "${OUT_DIR}"
echo
echo "Finished: $(date)"
echo "================================================================"
