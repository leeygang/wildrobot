#!/bin/bash

# Sync wildrobot files from Ubuntu GPU machine (linux-pc.local) to Mac
#
# Usage:
#   ./scp_from_remote.sh <filename>           # Copy single file/directory
#   ./scp_from_remote.sh --checkpoints        # List available checkpoints
#   ./scp_from_remote.sh --latest             # Copy latest checkpoint
#   ./scp_from_remote.sh <checkpoint_name>    # Copy specific checkpoint folder
#   ./scp_from_remote.sh --wandb-runs         # List available W&B runs
#   ./scp_from_remote.sh --wandb <run_name>   # Copy specific W&B run folder
#
# Examples:
#   ./scp_from_remote.sh checkpoints/wildrobot/final_amp_policy.pkl
#   ./scp_from_remote.sh videos/policy.mp4
#   ./scp_from_remote.sh --latest
#   ./scp_from_remote.sh wildrobot_amp_20251220-180000
#   ./scp_from_remote.sh --wandb-runs
#   ./scp_from_remote.sh --wandb run-20251220_183447-gef6ixl2

set -e

# Remote destination
REMOTE_USER="leeygang"
REMOTE_HOST="linux-pc.local"
REMOTE_BASE="/home/leeygang/projects/wildrobot"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Get script directory and move to wildrobot root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

list_checkpoints() {
    echo -e "\n${YELLOW}=== Available Checkpoints on Remote ===${NC}\n"

    ssh "$REMOTE_USER@$REMOTE_HOST" "ls -lht $REMOTE_BASE/checkpoints/ 2>/dev/null || echo 'No checkpoints directory found'"

    echo -e "\n${CYAN}To copy a checkpoint:${NC}"
    echo "  ./scp_from_remote.sh <checkpoint_folder_name>"
    echo "  ./scp_from_remote.sh --latest"
}

list_wandb_runs() {
    echo -e "\n${YELLOW}=== Available W&B Runs on Remote ===${NC}\n"

    ssh "$REMOTE_USER@$REMOTE_HOST" "ls -lht $REMOTE_BASE/wandb/ 2>/dev/null | grep '^d' | head -20 || echo 'No wandb directory found'"

    echo -e "\n${CYAN}To copy a W&B run:${NC}"
    echo "  ./scp_from_remote.sh --wandb <run_name>"
    echo ""
    echo -e "${CYAN}Example:${NC}"
    echo "  ./scp_from_remote.sh --wandb run-20251220_183447-gef6ixl2"
}

get_latest_checkpoint() {
    # Get the most recent checkpoint directory
    ssh "$REMOTE_USER@$REMOTE_HOST" "ls -t $REMOTE_BASE/checkpoints/ 2>/dev/null | head -1"
}

copy_file() {
    local FILE="$1"
    local REMOTE_PATH="$REMOTE_BASE/$FILE"
    local LOCAL_PATH="$FILE"

    # Create local directory if needed
    local LOCAL_DIR=$(dirname "$LOCAL_PATH")
    mkdir -p "$LOCAL_DIR" 2>/dev/null || true

    echo -e "${YELLOW}Copying from remote:${NC} $FILE"
    echo -e "  Remote: $REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"
    echo -e "  Local:  $LOCAL_PATH"
    echo ""

    # Check if it's a directory on remote
    if ssh "$REMOTE_USER@$REMOTE_HOST" "[ -d '$REMOTE_PATH' ]" 2>/dev/null; then
        echo -e "${CYAN}Detected directory, using rsync...${NC}"
        rsync -avz --progress "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/" "$LOCAL_PATH/"
    else
        scp "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH" "$LOCAL_PATH"
    fi

    if [ $? -eq 0 ]; then
        echo -e "\n${GREEN}✓ Transfer completed successfully!${NC}"
    else
        echo -e "\n${RED}✗ Transfer failed!${NC}"
        return 1
    fi
}

copy_checkpoint() {
    local CHECKPOINT_NAME="$1"
    local REMOTE_PATH="$REMOTE_BASE/checkpoints/$CHECKPOINT_NAME"
    local LOCAL_PATH="checkpoints/$CHECKPOINT_NAME"

    # Create local checkpoints directory
    mkdir -p checkpoints

    echo -e "${YELLOW}Copying checkpoint folder:${NC} $CHECKPOINT_NAME"
    echo -e "  Remote: $REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"
    echo -e "  Local:  $LOCAL_PATH"
    echo ""

    # Use rsync for efficient checkpoint copying
    if command -v rsync &> /dev/null; then
        echo -e "${CYAN}Using rsync for efficient transfer...${NC}"
        rsync -avz --progress "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/" "$LOCAL_PATH/"
        RESULT=$?
    else
        echo -e "${CYAN}Using scp (rsync not found)...${NC}"
        scp -r "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH" checkpoints/
        RESULT=$?
    fi

    if [ $RESULT -eq 0 ]; then
        echo -e "\n${GREEN}✓ Checkpoint transfer completed!${NC}"
        echo ""
        echo -e "${YELLOW}Checkpoint contents:${NC}"
        ls -lh "$LOCAL_PATH" 2>/dev/null | head -10
        echo ""
        echo -e "${CYAN}To visualize the policy:${NC}"
        echo "  python playground_amp/visualize_policy.py --checkpoint $LOCAL_PATH/final_amp_policy.pkl"
        echo ""
        echo -e "${CYAN}To test the policy:${NC}"
        echo "  python playground_amp/test_policy.py --checkpoint $LOCAL_PATH/final_amp_policy.pkl"
    else
        echo -e "\n${RED}✗ Transfer failed!${NC}"
        return 1
    fi
}

copy_wandb_run() {
    local RUN_NAME="$1"
    local REMOTE_PATH="$REMOTE_BASE/wandb/$RUN_NAME"
    local LOCAL_PATH="wandb/$RUN_NAME"

    # Create local wandb directory
    mkdir -p wandb

    echo -e "${YELLOW}Copying W&B run folder:${NC} $RUN_NAME"
    echo -e "  Remote: $REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"
    echo -e "  Local:  $LOCAL_PATH"
    echo ""

    # Check if the run exists on remote
    if ! ssh "$REMOTE_USER@$REMOTE_HOST" "[ -d '$REMOTE_PATH' ]" 2>/dev/null; then
        echo -e "${RED}✗ W&B run not found on remote: $RUN_NAME${NC}"
        echo ""
        echo -e "${CYAN}Available runs:${NC}"
        ssh "$REMOTE_USER@$REMOTE_HOST" "ls -1 $REMOTE_BASE/wandb/ 2>/dev/null | grep '^run-' | head -10"
        return 1
    fi

    # Use rsync for efficient copying
    if command -v rsync &> /dev/null; then
        echo -e "${CYAN}Using rsync for efficient transfer...${NC}"
        rsync -avz --progress "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/" "$LOCAL_PATH/"
        RESULT=$?
    else
        echo -e "${CYAN}Using scp (rsync not found)...${NC}"
        scp -r "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH" wandb/
        RESULT=$?
    fi

    if [ $RESULT -eq 0 ]; then
        echo -e "\n${GREEN}✓ W&B run transfer completed!${NC}"
        echo ""
        echo -e "${YELLOW}Run contents:${NC}"
        ls -lh "$LOCAL_PATH" 2>/dev/null | head -15
        echo ""

        # Show run info if available
        if [ -f "$LOCAL_PATH/files/config.yaml" ]; then
            echo -e "${YELLOW}Run config:${NC}"
            head -30 "$LOCAL_PATH/files/config.yaml" 2>/dev/null
            echo ""
        fi

        # Show summary if available
        if [ -f "$LOCAL_PATH/files/wandb-summary.json" ]; then
            echo -e "${YELLOW}Run summary:${NC}"
            cat "$LOCAL_PATH/files/wandb-summary.json" 2>/dev/null | head -5
            echo ""
        fi

        echo -e "${CYAN}To view in W&B:${NC}"
        echo "  Open the run URL from the logs, or"
        echo "  wandb sync $LOCAL_PATH"
    else
        echo -e "\n${RED}✗ Transfer failed!${NC}"
        return 1
    fi
}

# Main
if [ $# -eq 0 ]; then
    echo "Usage: $0 <filename|--checkpoints|--latest|--wandb-runs|--wandb <run>|checkpoint_name>"
    echo ""
    echo "Options:"
    echo "  <filename>       Copy specific file or directory from remote"
    echo "  --checkpoints    List available checkpoints on remote"
    echo "  --latest         Copy the most recent checkpoint"
    echo "  --wandb-runs     List available W&B runs on remote"
    echo "  --wandb <run>    Copy specific W&B run folder by name"
    echo "  <checkpoint>     Copy specific checkpoint folder by name"
    echo ""
    echo "Examples:"
    echo "  $0 checkpoints/wildrobot/final_amp_policy.pkl"
    echo "  $0 --checkpoints"
    echo "  $0 --latest"
    echo "  $0 wildrobot_amp_20251220-180000"
    echo "  $0 --wandb-runs"
    echo "  $0 --wandb run-20251220_183447-gef6ixl2"
    exit 1
fi

case "$1" in
    --checkpoints)
        list_checkpoints
        ;;
    --wandb-runs)
        list_wandb_runs
        ;;
    --wandb)
        if [ -z "$2" ]; then
            echo -e "${RED}Error: --wandb requires a run name${NC}"
            echo ""
            echo "Usage: $0 --wandb <run_name>"
            echo "Example: $0 --wandb run-20251220_183447-gef6ixl2"
            echo ""
            echo "To list available runs:"
            echo "  $0 --wandb-runs"
            exit 1
        fi
        copy_wandb_run "$2"
        ;;
    --latest)
        LATEST=$(get_latest_checkpoint)
        if [ -z "$LATEST" ]; then
            echo -e "${RED}No checkpoints found on remote${NC}"
            exit 1
        fi
        echo -e "${GREEN}Latest checkpoint: $LATEST${NC}"
        copy_checkpoint "$LATEST"
        ;;
    *)
        # Check if it looks like a checkpoint folder name (contains date pattern)
        if [[ "$1" =~ ^[a-zA-Z_]+[0-9]{8}-[0-9]{6}$ ]] || [[ "$1" =~ ^wildrobot ]]; then
            # Check if it exists in checkpoints on remote
            if ssh "$REMOTE_USER@$REMOTE_HOST" "[ -d '$REMOTE_BASE/checkpoints/$1' ]" 2>/dev/null; then
                copy_checkpoint "$1"
            else
                # Try as regular file path
                copy_file "$1"
            fi
        else
            copy_file "$1"
        fi
        ;;
esac
