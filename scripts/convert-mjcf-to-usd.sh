#!/bin/bash

PROJECT_ROOT="$HOME/projects/wildrobot"
# Set the IsaacLab path - adjust this to match your actual setup
ISAACLAB_PATH="$HOME/projects/IsaacLab"

# Check if IsaacLab exists
if [ ! -d "$ISAACLAB_PATH" ]; then
    echo "Error: IsaacLab not found at $ISAACLAB_PATH"
    echo "Please update ISAACLAB_PATH in this script to point to your IsaacLab installation"
    exit 1
fi

# Check if isaaclab.sh exists
if [ ! -f "$ISAACLAB_PATH/isaaclab.sh" ]; then
    echo "Error: isaaclab.sh not found at $ISAACLAB_PATH/isaaclab.sh"
    exit 1
fi

echo "Converting MJCF to USD..."
echo "Project root: $PROJECT_ROOT"
echo "IsaacLab path: $ISAACLAB_PATH"

cd "$ISAACLAB_PATH"

VERBOSE=""
if echo "$*" | grep -q "\-\-verbose"; then
    echo "Verbose mode enabled..."
    $VERBOSE="--verbose"
fi


./isaaclab.sh -p scripts/tools/convert_mjcf.py \
    "$PROJECT_ROOT/assets/mjcf/wildrobot.xml" \
    "$PROJECT_ROOT/assets/usd/wildrobot.usd" \
    $VERBOSE \
    --import-sites \
    --make-instanceable \
    --fix-base

# Verify USD file was created successfully
if [ -f "$PROJECT_ROOT/assets/usd/wildrobot.usd" ]; then
    USD_SIZE=$(stat -f%z "$PROJECT_ROOT/assets/usd/wildrobot.usd" 2>/dev/null || stat -c%s "$PROJECT_ROOT/assets/usd/wildrobot.usd" 2>/dev/null)
    if [ "$USD_SIZE" -gt 1000 ]; then
        echo "✓ Conversion complete! USD saved to $PROJECT_ROOT/assets/usd/wildrobot.usd"
        echo "  File size: $(numfmt --to=iec-i --suffix=B $USD_SIZE 2>/dev/null || echo "$USD_SIZE bytes")"
    else
        echo "✗ Error: USD file created but appears to be empty or corrupted (size: $USD_SIZE bytes)"
        exit 1
    fi
else
    echo "✗ Error: USD file was not created at $PROJECT_ROOT/assets/usd/wildrobot.usd"
    exit 1
fi
