#!/bin/bash

# Sync wildrobot files from Mac to Ubuntu GPU machine
#
# Usage:
#   ./scp_to_remote.sh [--public] <filename>    # Sync single file/directory
#   ./scp_to_remote.sh [--public] --all         # Sync all essential training files
#   ./scp_to_remote.sh [--public] --data        # Sync only data files (motions, AMP)
#   ./scp_to_remote.sh [--public] --code        # Sync only code files
#
# Options:
#   --public    Use $LINUX_PUBLIC_IP instead of linux-pc.local
#
# Examples:
#   ./scp_to_remote.sh --public --code
#   ./scp_to_remote.sh --public training/train.py
#   ./scp_to_remote.sh --all                    # Uses linux-pc.local

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Remote destination
REMOTE_USER="leeygang"
REMOTE_HOST="linux-pc.local"  # Default
REMOTE_BASE="~/projects/wildrobot"

# Parse --public option
if [[ "$1" == "--public" ]]; then
    if [ -z "$LINUX_PUBLIC_IP" ]; then
        echo -e "${RED}Error: LINUX_PUBLIC_IP environment variable is not set${NC}"
        echo "Set it with: export LINUX_PUBLIC_IP=<your-ip>"
        exit 1
    fi
    REMOTE_HOST="$LINUX_PUBLIC_IP"
    shift
fi

echo -e "${YELLOW}Remote host:${NC} $REMOTE_HOST"

# Get script directory and move to wildrobot root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

sync_file() {
    local FILE="$1"
    local REMOTE_PATH="$REMOTE_BASE/$FILE"

    if [ ! -e "$FILE" ]; then
        echo -e "${RED}Error: '$FILE' does not exist${NC}"
        return 1
    fi

    # Create remote directory if needed
    local REMOTE_DIR=$(dirname "$REMOTE_PATH")
    ssh "$REMOTE_USER@$REMOTE_HOST" "mkdir -p $REMOTE_DIR" 2>/dev/null || true

    if [ -d "$FILE" ]; then
        echo -e "${YELLOW}Syncing directory:${NC} $FILE"
        rsync -avz --progress "$FILE/" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/"
    else
        echo -e "${YELLOW}Syncing file:${NC} $FILE"
        scp "$FILE" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"
    fi

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Synced:${NC} $FILE"
    else
        echo -e "${RED}✗ Failed:${NC} $FILE"
        return 1
    fi
}

sync_dir_no_cache() {
    # Sync directory excluding __pycache__, .pyc, .pyo, .egg-info, etc.
    local DIR="$1"
    local REMOTE_PATH="$REMOTE_BASE/$DIR"

    if [ ! -d "$DIR" ]; then
        echo -e "${RED}Error: Directory '$DIR' does not exist${NC}"
        return 1
    fi

    # Create remote directory if needed
    ssh "$REMOTE_USER@$REMOTE_HOST" "mkdir -p $REMOTE_PATH" 2>/dev/null || true

    echo -e "${YELLOW}Syncing directory (excluding cache):${NC} $DIR"
    rsync -avz --progress \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='*.pyo' \
        --exclude='*.egg-info' \
        --exclude='.pytest_cache' \
        --exclude='.mypy_cache' \
        --exclude='.ruff_cache' \
        "$DIR/" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/"

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Synced:${NC} $DIR"
    else
        echo -e "${RED}✗ Failed:${NC} $DIR"
        return 1
    fi
}

sync_data() {
    echo -e "\n${YELLOW}=== Syncing Data Files ===${NC}\n"

    # Motion data (retargeted) - in assets/motions
    if [ -d "assets/motions" ]; then
        sync_file "assets/motions"
    fi

    # AMP formatted data - in training/data
    if [ -d "training/data" ]; then
        sync_file "training/data"
    fi

    echo -e "\n${GREEN}✓ Data sync complete${NC}"
}

sync_code() {
    echo -e "\n${YELLOW}=== Syncing Code Files ===${NC}\n"

    # Robot config (critical - code depends on this)
    sync_file "assets/robot_config.yaml"

    # Scene/model files
    sync_file "assets/scene_flat_terrain.xml"

    # Training scripts
    sync_file "training/train.py"

    # Training modules (exclude __pycache__)
    if [ -d "training/training" ]; then
        sync_dir_no_cache "training/training"
    fi

    # AMP modules (exclude __pycache__)
    if [ -d "training/amp" ]; then
        sync_dir_no_cache "training/amp"
    fi

    # Environment (exclude __pycache__)
    if [ -d "training/envs" ]; then
        sync_dir_no_cache "training/envs"
    fi

    # Configs (exclude __pycache__)
    if [ -d "training/configs" ]; then
        sync_dir_no_cache "training/configs"
    fi

    # Utils (exclude __pycache__)
    if [ -d "training/utils" ]; then
        sync_dir_no_cache "training/utils"
    fi

    # Documentation
    if [ -f "training/phase3_rl_training_plan.md" ]; then
        sync_file "training/phase3_rl_training_plan.md"
    fi
    if [ -f "training/reference_data_generation.md" ]; then
        sync_file "training/reference_data_generation.md"
    fi

    echo -e "\n${GREEN}✓ Code sync complete${NC}"
}

sync_all() {
    echo -e "\n${YELLOW}======================================${NC}"
    echo -e "${YELLOW}  Syncing All Training Files${NC}"
    echo -e "${YELLOW}======================================${NC}"

    sync_code
    sync_data

    echo -e "\n${GREEN}======================================${NC}"
    echo -e "${GREEN}  All files synced to $REMOTE_HOST${NC}"
    echo -e "${GREEN}======================================${NC}"

    echo -e "\n${YELLOW}To start training on Ubuntu:${NC}"
    echo "  ssh $REMOTE_USER@$REMOTE_HOST"
    echo "  cd ~/projects/wildrobot"
    echo "  uv run python training/train.py"
}

# Main
if [ $# -eq 0 ]; then
    echo "Usage: $0 <filename|--all|--data|--code>"
    echo ""
    echo "Options:"
    echo "  <filename>   Sync specific file or directory"
    echo "  --all        Sync all essential training files"
    echo "  --data       Sync only data files (motions, AMP)"
    echo "  --code       Sync only code files"
    echo ""
    echo "Examples:"
    echo "  $0 training/train.py"
    echo "  $0 data/amp/walking_motions_merged.pkl"
    echo "  $0 --all"
    exit 1
fi

case "$1" in
    --all)
        sync_all
        ;;
    --data)
        sync_data
        ;;
    --code)
        sync_code
        ;;
    *)
        sync_file "$1"
        ;;
esac
