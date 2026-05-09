#!/usr/bin/env bash
# Visualize the WR operating-point bracket in the MuJoCo viewer.
#
# Generates a single ZMP reference library covering the post-Phase-9A
# bracket {0.20, 0.265, 0.30} once, then opens the MuJoCo viewer at
# each one in turn so you can compare gait shape side-by-side.
#
#   - 0.20  — lower edge (slow-walk regression check)
#   - 0.265 — Phase 9A operating point (TB step/leg-matched)
#   - 0.30  — upper robustness bracket
#
# History: this helper originally swept the pre-Phase-9A candidates
# {0.15, 0.19, 0.21, 0.25} when picking the operating point.  After
# the Phase 9A decision (vx=0.265, TB step/leg-matched), it now
# brackets the chosen point.  Override with VX_OPTIONS="0.15,0.19,..."
# to re-run the historical sweep.
#
# Usage (run from the wildrobot repo root):
#   bash tools/visualize_vx_options.sh
#   VX_OPTIONS="0.15,0.19,0.21,0.25" bash tools/visualize_vx_options.sh
#   VIZ_MODE=fixed_base bash tools/visualize_vx_options.sh
#
# What you'll see in each viewer (default kinematic mode):
#   - kinematic replay of the prior at that vx (no falling — gait shape only)
# Other modes (VIZ_MODE) drop into PD / closed-loop replays; see below.
#
# Close the viewer window (or press Escape) to advance to the next vx.
# After all bins, re-run individual ones with:
#   uv run mjpython training/eval/view_zmp_in_mujoco.py \
#       --library-path /tmp/wr_vx_options_lib --vx <bin>
#
# macOS: the viewer requires `mjpython` (not `python`).  Linux: either
# works.  This script picks the right one automatically.

set -euo pipefail

# Default: post-Phase-9A bracket around the chosen operating point (0.265).
# Override via env var, e.g. VX_OPTIONS="0.15,0.19,0.21,0.25".
VX_OPTIONS_CSV="${VX_OPTIONS:-0.20,0.265,0.30}"
IFS=',' read -r -a VX_OPTIONS <<< "${VX_OPTIONS_CSV}"

# Encode the requested bracket into the default cache directory.  Library
# lookup is nearest-neighbor (see control/references/reference_library.py
# `ReferenceLibrary.lookup`), so a cache built for one bracket would silently
# snap requests for a different bracket onto the nearest stale bin.  Different
# brackets get different cache dirs.  An explicit LIB_DIR is left untouched —
# the caller owns cache management in that case.
VX_OPTIONS_TAG="$(printf '%s\n' "${VX_OPTIONS[@]}" | sort -n | paste -sd '_' -)"
LIB_DIR="${LIB_DIR:-/tmp/wr_vx_options_lib_${VX_OPTIONS_TAG}}"

# Pick mjpython on macOS, plain python elsewhere.
if [[ "$(uname -s)" == "Darwin" ]]; then
    PY_LAUNCHER="mjpython"
else
    PY_LAUNCHER="python"
fi

echo "================================================================"
echo "WR vx-bracket visual evaluation"
echo "Library directory: ${LIB_DIR}"
echo "Candidate vx bins: ${VX_OPTIONS[*]}"
echo "Viewer launcher: uv run ${PY_LAUNCHER}"
echo "================================================================"

# 1. Pre-generate the library (one-time, ~5-10 s per bin).
if [[ ! -d "${LIB_DIR}" ]]; then
    echo
    echo "[1/2] Generating reference library at ${LIB_DIR} ..."
    # Bash arrays expand space-separated; join with commas for Python list literal.
    VX_CSV="$(IFS=,; echo "${VX_OPTIONS[*]}")"
    uv run python -c "
from control.zmp.zmp_walk import ZMPWalkGenerator
vx_options = [${VX_CSV}]
gen = ZMPWalkGenerator()
lib = gen.build_library_for_vx_values(vx_options)
lib.save('${LIB_DIR}')
print(f'Library saved to ${LIB_DIR} with bins: {vx_options}')
"
else
    echo "Library already exists at ${LIB_DIR}; reusing."
    echo "(Delete ${LIB_DIR} to force regeneration.)"
fi

# 2. Sequentially open the viewer at each vx.
#
# Mode selection (env var VIZ_MODE):
#   kinematic   — pelvis teleported along planned trajectory, gravity-free
#                 (best for comparing gait shape; doesn't fall)
#   fixed_base  — pelvis pinned at standing pose, legs swing via PD
#                 (best for checking actuator tracking)
#   free_float  — free-floating PD ctrl (will fall in ~1 s on WR;
#                 useful for spotting catastrophic divergence only)
#   c2_stab     — free-floating + C2 validation stabilizer
#                 (closed-loop with bounded balance harness)
VIZ_MODE="${VIZ_MODE:-kinematic}"
case "${VIZ_MODE}" in
    kinematic)   MODE_FLAG=(--kinematic) ;;
    fixed_base)  MODE_FLAG=(--fixed-base) ;;
    free_float)  MODE_FLAG=() ;;
    c2_stab)     MODE_FLAG=(--c2-stabilizer) ;;
    *)
        echo "Unknown VIZ_MODE='${VIZ_MODE}'.  Use one of: kinematic, fixed_base, free_float, c2_stab"
        exit 1
        ;;
esac

echo
echo "[2/2] Opening viewer at each vx in turn (mode=${VIZ_MODE})."
echo "    Close the viewer window or press Escape to advance."
echo "    Override mode: VIZ_MODE=fixed_base bash tools/visualize_vx_options.sh"
echo
for vx in "${VX_OPTIONS[@]}"; do
    case "${vx}" in
        0.15) regime="pre-Phase-9A op pt — known shuffle (step/leg=0.144)" ;;
        0.19) regime="Froude-matched — mild shuffle (step/leg=0.183)" ;;
        0.20) regime="lower bracket — slow-walk regression check (step/leg≈0.190)" ;;
        0.21) regime="threshold — just out of shuffle (step/leg=0.200)" ;;
        0.25) regime="healthy walk (step/leg=0.241)" ;;
        0.265) regime="Phase 9A operating point — TB step/leg-matched (step/leg=0.252)" ;;
        0.30) regime="upper robustness bracket — fast walk (step/leg≈0.286)" ;;
        *)    regime="custom" ;;
    esac
    echo
    echo "----------------------------------------------------------------"
    echo "vx = ${vx} m/s  —  ${regime}"
    echo "Mode: ${VIZ_MODE}"
    echo "Close the viewer to advance ..."
    echo "----------------------------------------------------------------"
    uv run "${PY_LAUNCHER}" training/eval/view_zmp_in_mujoco.py \
        --library-path "${LIB_DIR}" \
        --vx "${vx}" \
        "${MODE_FLAG[@]}"
done

echo
echo "================================================================"
echo "Done.  Library at ${LIB_DIR} retained for re-runs."
echo "================================================================"
