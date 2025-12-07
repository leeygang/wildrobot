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

# Check if --nogui flag is passed
if [[ "$*" == *"--nogui"* ]]; then
    echo "Running in headless mode (no GUI)..."
    ./isaaclab.sh -p scripts/tools/convert_mjcf.py \
      "$PROJECT_ROOT/assets/mjcf/wildrobot.xml" \
      "$PROJECT_ROOT/assets/usd/wildrobot.usd" \
      --import-sites \
      --make-instanceable \
      --headless
else
    ./isaaclab.sh -p scripts/tools/convert_mjcf.py \
      "$PROJECT_ROOT/assets/mjcf/wildrobot.xml" \
      "$PROJECT_ROOT/assets/usd/wildrobot.usd" \
      --import-sites \
      --make-instanceable
fi

echo "Conversion complete! USD saved to $PROJECT_ROOT/assets/usd/wildrobot.usd"
