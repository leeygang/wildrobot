#!/bin/bash

# Sync wildrobot files from Mac to Ubuntu GPU machine (linux-pc.local)
#
# Usage:
#   ./scp_to_remote.sh <filename>           # Sync single file/directory
#   ./scp_to_remote.sh --all                # Sync all essential training files
#   ./scp_to_remote.sh --data               # Sync only data files (motions, AMP)
#   ./scp_to_remote.sh --code               # Sync only code files
#
# Examples:
#   ./scp_to_remote.sh playground_amp/train_amp.py
#   ./scp_to_remote.sh data/amp/walking_motions_merged.pkl
#   ./scp_to_remote.sh --all

set -e

# Remote destination
REMOTE_USER="leeygang"
REMOTE_HOST="linux-pc.local"
REMOTE_BASE="~/projects/wildrobot"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory (wildrobot root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

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

sync_data() {
    echo -e "\n${YELLOW}=== Syncing Data Files ===${NC}\n"

    # Motion data (retargeted)
    if [ -d "assets/motions" ]; then
        sync_file "assets/motions"
    fi

    # AMP formatted data
    if [ -d "data/amp" ]; then
        sync_file "data/amp"
    fi

    echo -e "\n${GREEN}✓ Data sync complete${NC}"
}

sync_code() {
    echo -e "\n${YELLOW}=== Syncing Code Files ===${NC}\n"

    # Training scripts
    sync_file "playground_amp/train_amp.py"
    sync_file "playground_amp/train.py"

    # Training modules
    if [ -d "playground_amp/training" ]; then
        sync_file "playground_amp/training"
    fi

    # AMP modules
    if [ -d "playground_amp/amp" ]; then
        sync_file "playground_amp/amp"
    fi

    # Environment
    if [ -d "playground_amp/envs" ]; then
        sync_file "playground_amp/envs"
    fi

    # Configs
    if [ -d "playground_amp/configs" ]; then
        sync_file "playground_amp/configs"
    fi

    # Phase 3 training plan
    sync_file "playground_amp/phase3_rl_training_plan.md"

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
    echo "  uv run python playground_amp/train_amp.py \\"
    echo "      --use-custom-loop \\"
    echo "      --iterations 3000 \\"
    echo "      --num-envs 2048 \\"
    echo "      --amp-weight 1.0 \\"
    echo "      --amp-data data/amp/walking_motions_merged.pkl"
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
    echo "  $0 playground_amp/train_amp.py"
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
