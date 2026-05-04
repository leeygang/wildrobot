#!/usr/bin/env bash
# Refresh assets/v2/wildrobot.xml from a pre-staged Onshape export.
#
# Usage:
#   ./assets/update_xml.sh
#
# Prerequisites:
#   The raw Onshape export must already exist at:
#     assets/v2/onshape_export/wildrobot.xml
#     assets/v2/onshape_export/assets/   (mesh files: .stl, .part)
#   Stage it manually (e.g. via Onshape's onshape-to-robot CLI invoked
#   outside this pipeline, then commit to git).
#
# Pipeline:
#   1. Sync onshape_export/wildrobot.xml -> v2/wildrobot.xml.
#   2. Mirror onshape_export/assets/    -> v2/assets/   (rsync --delete).
#   3. Reorder actuators to canonical order (assets/reorder_actuators.py).
#   4. Run post_process.py on v2/wildrobot.xml. The first post_process step
#      (inject_additional_xml) splices sensors.xml + joints_properties.xml
#      into the synced MJCF; remaining steps normalize bodies/geoms/defaults
#      and emit mujoco_robot_config.json.
#   5. Print the actuated-joint summary.
#
# This script no longer invokes onshape-to-robot directly. The
# `--export-onshape` flag and the v1 variant have been removed.

set -euo pipefail

echo "=============================="
echo "WildRobot MJCF update pipeline"
echo "Platform: $(uname -s)"
echo "=============================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VARIANT="v2"
VARIANT_DIR="${SCRIPT_DIR}/${VARIANT}"
EXPORT_DIR="${VARIANT_DIR}/onshape_export"

# Detect Python command
if command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_CMD="python"
else
    echo "Error: Python not found"
    exit 1
fi

echo "Using Python: $PYTHON_CMD"

if [[ $# -gt 0 ]]; then
    case "$1" in
        -h|--help)
            cat <<EOF
Usage: $0 [--help]

Refreshes assets/${VARIANT}/wildrobot.xml from the pre-staged Onshape export
at assets/${VARIANT}/onshape_export/, then runs reorder_actuators followed
by post_process.

This script does not invoke onshape-to-robot. Stage the export manually at
assets/${VARIANT}/onshape_export/ before running.
EOF
            exit 0
            ;;
        *)
            echo "Error: unknown option '$1'"
            echo "Usage: $0 [--help]"
            exit 1
            ;;
    esac
fi

# --- Preflight ---------------------------------------------------------------

if [[ ! -d "$VARIANT_DIR" ]]; then
    echo "Error: variant directory not found: $VARIANT_DIR"
    exit 1
fi
if [[ ! -d "$EXPORT_DIR" ]]; then
    echo "Error: onshape export directory not found: $EXPORT_DIR"
    echo "       Stage the raw Onshape export at this path first."
    exit 1
fi
if [[ ! -f "$EXPORT_DIR/wildrobot.xml" ]]; then
    echo "Error: missing $EXPORT_DIR/wildrobot.xml"
    exit 1
fi
if [[ ! -d "$EXPORT_DIR/assets" ]]; then
    echo "Error: missing $EXPORT_DIR/assets/ (mesh directory)"
    exit 1
fi
command -v rsync >/dev/null 2>&1 || {
    echo "Error: rsync not found (required for mesh sync with --delete)"
    exit 1
}

echo ""
echo "=============================="
echo "Updating assets/${VARIANT}"
echo "=============================="

# --- 1. Sync wildrobot.xml ---------------------------------------------------

echo ""
echo "Syncing wildrobot.xml from onshape_export/..."
cp -f "$EXPORT_DIR/wildrobot.xml" "$VARIANT_DIR/wildrobot.xml"

# --- 2. Mirror meshes (with delete) ------------------------------------------

echo "Mirroring meshes onshape_export/assets/ -> ${VARIANT}/assets/ (with delete)..."
mkdir -p "$VARIANT_DIR/assets"
rsync -a --delete "$EXPORT_DIR/assets/" "$VARIANT_DIR/assets/"

# --- 3. Reorder actuators ----------------------------------------------------

echo ""
echo "Reordering actuators to canonical order..."
$PYTHON_CMD "$SCRIPT_DIR/reorder_actuators.py" \
    --xml "$VARIANT_DIR/wildrobot.xml" \
    --order "$VARIANT_DIR/actuator_order.txt"

# --- 4. Post-process ---------------------------------------------------------

echo ""
echo "Running post_process.py..."
$PYTHON_CMD "$SCRIPT_DIR/post_process.py" "$VARIANT_DIR/wildrobot.xml"

if [[ ! -f "$VARIANT_DIR/mujoco_robot_config.json" ]]; then
    echo "Error: post_process did not produce mujoco_robot_config.json in ${VARIANT_DIR}"
    exit 1
fi

# --- 5. Joint summary --------------------------------------------------------

print_joint_summary() {
    "$PYTHON_CMD" - "$VARIANT" "$VARIANT_DIR/mujoco_robot_config.json" <<'PY'
import json
import sys
from pathlib import Path

variant = sys.argv[1]
cfg_path = Path(sys.argv[2])
data = json.loads(cfg_path.read_text())
specs = data.get("actuated_joint_specs", [])

def sort_key(entry):
    name = str(entry.get("name", ""))
    if name.startswith("left_"):
        base = name[len("left_"):]
        side_order = 0
    elif name.startswith("right_"):
        base = name[len("right_"):]
        side_order = 1
    else:
        base = name
        side_order = 2
    return (base, side_order, name)

ordered = sorted(specs, key=sort_key)

print(f"\nJoint summary (assets/{variant}):")
for joint in ordered:
    name = str(joint.get("name", "unknown"))
    rng = joint.get("range", ["?", "?"])
    sign = float(joint.get("policy_action_sign", 1.0))
    print(f"  {name}: range[{rng[0]}, {rng[1]}], policy_action_sign: {sign:+.1f}")
PY
}

print_joint_summary

echo ""
echo "✓ Updated assets/${VARIANT}"
echo ""
echo "Render command (copy/paste):"
echo "  ${PYTHON_CMD} assets/render_models.py --scene-file assets/${VARIANT}/scene_flat_terrain.xml"
