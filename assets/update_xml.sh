#!/usr/bin/env bash
# Update WildRobot MJCF from Onshape for a specific asset variant.
#
# Usage:
#   ./assets/update_xml.sh --version v1
#   ./assets/update_xml.sh --version v2
#   ./assets/update_xml.sh --version all
#
# Notes:
# - Runs `onshape-to-robot` inside `assets/<version>/` using that folder's config.json.
# - Runs `assets/post_process.py` to regenerate `robot_config.yaml` next to the MJCF.

set -euo pipefail

echo "=============================="
echo "onshape-to-robot Pipeline"
echo "Platform: $(uname -s)"
echo "=============================="

# Check if commands exist
command -v onshape-to-robot >/dev/null 2>&1 || { 
    echo "Error: onshape-to-robot not found. Install with: pip install onshape-to-robot"
    exit 1
}

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
if [[ "${1:-}" == "--version" ]]; then
    VERSION="${2:-}"
fi
if [[ -z "$VERSION" ]]; then
    echo "Error: missing --version {v1|v2|all}"
    exit 1
fi

update_variant() {
    local variant="$1"
    local variant_dir="${SCRIPT_DIR}/${variant}"
    local ts
    ts="$(date +%Y%m%d_%H%M%S)"

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

    # Backup existing generated outputs (keep hand-authored XML like keyframes/sensors).
    local backup_dir=".backup_${ts}"
    mkdir -p "$backup_dir"
    if [[ -f wildrobot.xml ]]; then mv wildrobot.xml "$backup_dir/" ; fi
    if [[ -f wildrobot.urdf ]]; then mv wildrobot.urdf "$backup_dir/" ; fi
    if [[ -d assets ]]; then mv assets "$backup_dir/assets" ; fi

    # Step 1: Run onshape-to-robot (reads ./config.json)
    echo ""
    echo "Running onshape-to-robot in ${variant_dir}..."
    onshape-to-robot .

    # Step 2: Run post-process to regenerate robot_config.yaml next to the MJCF
    echo ""
    echo "Running post_process.py..."
    $PYTHON_CMD "${SCRIPT_DIR}/post_process.py" "wildrobot.xml"

    echo ""
    echo "✓ Updated assets/${variant}"

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
