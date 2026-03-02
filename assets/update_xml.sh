#!/usr/bin/env bash
# Update WildRobot MJCF from Onshape for a specific asset variant.
#
# Usage:
#   ./assets/update_xml.sh --version v1
#   ./assets/update_xml.sh --version v2
#   ./assets/update_xml.sh --version all
#   ./assets/update_xml.sh --version v2 --export-onshape
#
# Notes:
# - By default skips Onshape export and reuses existing files in `assets/<version>/`.
# - Runs `assets/post_process.py` to regenerate `mujoco_robot_config.json` next to the MJCF.
# - With `--export-onshape`, runs onshape export and refreshes generated assets first.

set -euo pipefail

echo "=============================="
echo "onshape-to-robot Pipeline"
echo "Platform: $(uname -s)"
echo "=============================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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

VERSION=""
EXPORT_ONSHAPE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --version)
            VERSION="${2:-}"
            shift 2
            ;;
        --export-onshape)
            EXPORT_ONSHAPE=1
            shift
            ;;
        -h|--help)
            echo "Usage: $0 --version {v1|v2|all} [--export-onshape]"
            echo ""
            echo "Options:"
            echo "  --version      Asset variant to process (v1, v2, all)"
            echo "  --export-onshape  Run onshape-to-robot and refresh assets before reorder/post_process"
            exit 0
            ;;
        *)
            echo "Error: unknown option '$1'"
            echo "Usage: $0 --version {v1|v2|all} [--export-onshape]"
            exit 1
            ;;
    esac
done

if [[ -z "$VERSION" ]]; then
    echo "Error: missing --version {v1|v2|all}"
    exit 1
fi

if [[ "$EXPORT_ONSHAPE" -eq 1 ]]; then
    command -v onshape-to-robot >/dev/null 2>&1 || {
        echo "Error: onshape-to-robot not found. Install with: pip install onshape-to-robot"
        exit 1
    }
fi

UPDATED_VARIANTS=()

print_joint_summary() {
    local variant="$1"
    local robot_cfg_path="$2"

    if [[ ! -f "$robot_cfg_path" ]]; then
        echo "[joint-summary] Missing robot config: $robot_cfg_path"
        return
    fi

    "$PYTHON_CMD" - "$variant" "$robot_cfg_path" <<'PY'
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

update_variant() {
    local variant="$1"
    local variant_dir="${SCRIPT_DIR}/${variant}"

    if [[ ! -d "$variant_dir" ]]; then
        echo "Error: variant directory not found: $variant_dir"
        exit 1
    fi
    if [[ ! -f "$variant_dir/config.json" ]]; then
        echo "Error: missing config.json for variant '${variant}': $variant_dir/config.json"
        exit 1
    fi

    echo ""
    echo "=============================="
    echo "Updating assets/${variant}"
    echo "=============================="

    pushd "$variant_dir" >/dev/null

    if [[ "$EXPORT_ONSHAPE" -eq 1 ]]; then
        # Clean meshes folder to avoid stale assets after export.
        # onshape-to-robot will re-create/populate this folder as needed.
        rm -rf assets
        mkdir -p assets

        # Ensure generated model outputs are refreshed (avoid silently reusing stale files).
        # Keep existing mujoco_robot_config.json until post_process succeeds to avoid losing
        # the last valid config if generation fails midway.
        rm -f wildrobot.xml wildrobot.urdf

        # Step 1: Run onshape-to-robot (reads ./config.json)
        echo ""
        echo "Running onshape-to-robot in ${variant_dir}..."
        onshape-to-robot .

        if [[ ! -f wildrobot.xml ]]; then
            echo "Error: onshape-to-robot did not produce wildrobot.xml in ${variant_dir}"
            exit 1
        fi
    else
        echo ""
        echo "Skipping onshape export for ${variant}; reusing existing files in ${variant_dir}."
        if [[ ! -f wildrobot.xml ]]; then
            echo "Error: default no-export mode requires existing ${variant_dir}/wildrobot.xml"
            echo "       Use --export-onshape to regenerate from Onshape."
            exit 1
        fi
    fi

    # Step 1.5: Enforce canonical actuator order (v2 only)
    if [[ "$variant" != "v1" ]]; then
        echo ""
        echo "Reordering actuators to canonical order (v2)..."
        $PYTHON_CMD "${SCRIPT_DIR}/reorder_actuators.py" \
            --xml "${variant_dir}/wildrobot.xml" \
            --order "${variant_dir}/actuator_order.txt"
    fi

    # Step 2: Run post-process to regenerate mujoco_robot_config.json next to the MJCF
    echo ""
    echo "Running post_process.py..."
    $PYTHON_CMD "${SCRIPT_DIR}/post_process.py" "wildrobot.xml"
    if [[ ! -f mujoco_robot_config.json ]]; then
        echo "Error: post_process did not produce mujoco_robot_config.json in ${variant_dir}"
        exit 1
    fi

    print_joint_summary "$variant" "mujoco_robot_config.json"

    echo ""
    echo "✓ Updated assets/${variant}"
    UPDATED_VARIANTS+=("$variant")

    popd >/dev/null
}

case "$VERSION" in
    v1|v2)
        update_variant "$VERSION"
        ;;
    all)
        update_variant "v1"
        update_variant "v2"
        ;;
    *)
        echo "Error: invalid --version '$VERSION' (expected v1, v2, or all)"
        exit 1
        ;;
esac

echo ""
echo "✓ Pipeline completed successfully!"

if [[ ${#UPDATED_VARIANTS[@]} -gt 0 ]]; then
    echo ""
    echo "Render commands (copy/paste):"
    for variant in "${UPDATED_VARIANTS[@]}"; do
        echo "  ${PYTHON_CMD} assets/render_models.py --scene-file assets/${variant}/scene_flat_terrain.xml"
    done
fi
