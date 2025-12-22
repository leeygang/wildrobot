#!/bin/bash

# Sync wildrobot files from Ubuntu GPU machine to Mac
#
# Usage:
#   ./scp_from_remote.sh [--public] <filename>           # Copy single file/directory
#   ./scp_from_remote.sh [--public] --checkpoints        # List available checkpoints
#   ./scp_from_remote.sh [--public] --latest             # Copy latest checkpoint
#   ./scp_from_remote.sh [--public] <checkpoint_name>    # Copy specific checkpoint folder
#   ./scp_from_remote.sh [--public] --logs               # List available training logs/runs
#   ./scp_from_remote.sh [--public] --log <run_name>     # Copy specific training log/run folder
#
# Options:
#   --public    Use $LINUX_PUBLIC_IP instead of linux-pc.local
#
# Examples:
#   ./scp_from_remote.sh --public --checkpoints
#   ./scp_from_remote.sh --public --latest
#   ./scp_from_remote.sh --log run_20251220_183447

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Remote destination
REMOTE_USER="leeygang"
REMOTE_HOST="linux-pc.local"  # Default
REMOTE_BASE="/home/leeygang/projects/wildrobot"

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

list_checkpoints() {
    echo -e "\n${YELLOW}=== Available Checkpoints on Remote ===${NC}\n"

    ssh "$REMOTE_USER@$REMOTE_HOST" "ls -lht $REMOTE_BASE/playground_amp/checkpoints/ 2>/dev/null || echo 'No checkpoints directory found'"

    echo -e "\n${CYAN}To copy a checkpoint:${NC}"
    echo "  ./scp_from_remote.sh <checkpoint_folder_name>"
    echo "  ./scp_from_remote.sh --latest"
}

list_logs() {
    echo -e "\n${YELLOW}=== Available Training Logs on Remote ===${NC}\n"

    ssh "$REMOTE_USER@$REMOTE_HOST" "ls -lht $REMOTE_BASE/playground_amp/logs/ 2>/dev/null | grep '^d' | head -20 || echo 'No logs directory found'"

    echo -e "\n${CYAN}To copy a training log:${NC}"
    echo "  ./scp_from_remote.sh --log <run_name>"
    echo ""
    echo -e "${CYAN}Example:${NC}"
    echo "  ./scp_from_remote.sh --log run_20251220_183447"
}

get_latest_checkpoint() {
    # Get the most recent checkpoint directory
    ssh "$REMOTE_USER@$REMOTE_HOST" "ls -t $REMOTE_BASE/playground_amp/checkpoints/ 2>/dev/null | head -1"
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
    local REMOTE_PATH="$REMOTE_BASE/playground_amp/checkpoints/$CHECKPOINT_NAME"
    local LOCAL_PATH="playground_amp/checkpoints/$CHECKPOINT_NAME"

    # Create local checkpoints directory
    mkdir -p playground_amp/checkpoints

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

copy_log() {
    local RUN_NAME="$1"
    local REMOTE_PATH="$REMOTE_BASE/playground_amp/logs/$RUN_NAME"
    local LOCAL_PATH="playground_amp/logs/$RUN_NAME"

    # Create local logs directory
    mkdir -p playground_amp/logs

    echo -e "${YELLOW}Copying training log folder:${NC} $RUN_NAME"
    echo -e "  Remote: $REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"
    echo -e "  Local:  $LOCAL_PATH"
    echo ""

    # Check if the run exists on remote
    if ! ssh "$REMOTE_USER@$REMOTE_HOST" "[ -d '$REMOTE_PATH' ]" 2>/dev/null; then
        echo -e "${RED}✗ Training log not found on remote: $RUN_NAME${NC}"
        echo ""
        echo -e "${CYAN}Available logs:${NC}"
        ssh "$REMOTE_USER@$REMOTE_HOST" "ls -1 $REMOTE_BASE/playground_amp/logs/ 2>/dev/null | grep '^run_' | head -10"
        return 1
    fi

    # Use rsync for efficient copying
    if command -v rsync &> /dev/null; then
        echo -e "${CYAN}Using rsync for efficient transfer...${NC}"
        rsync -avz --progress "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/" "$LOCAL_PATH/"
        RESULT=$?
    else
        echo -e "${CYAN}Using scp (rsync not found)...${NC}"
        scp -r "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH" playground_amp/logs/
        RESULT=$?
    fi

    if [ $RESULT -eq 0 ]; then
        echo -e "\n${GREEN}✓ Training log transfer completed!${NC}"
        echo ""
        echo -e "${YELLOW}Log contents:${NC}"
        ls -lh "$LOCAL_PATH" 2>/dev/null | head -15
        echo ""

        # Show run info if available (W&B stores files in wandb subfolder)
        if [ -d "$LOCAL_PATH/wandb" ]; then
            echo -e "${YELLOW}W&B files:${NC}"
            ls -lh "$LOCAL_PATH/wandb/" 2>/dev/null | head -10
            echo ""
        fi

        echo -e "${CYAN}To sync with W&B:${NC}"
        echo "  wandb sync $LOCAL_PATH/wandb/"
    else
        echo -e "\n${RED}✗ Transfer failed!${NC}"
        return 1
    fi
}

# Main
if [ $# -eq 0 ]; then
    echo "Usage: $0 <filename|--checkpoints|--latest|--logs|--log <run>|checkpoint_name>"
    echo ""
    echo "Options:"
    echo "  <filename>       Copy specific file or directory from remote"
    echo "  --checkpoints    List available checkpoints on remote"
    echo "  --latest         Copy the most recent checkpoint"
    echo "  --logs           List available training logs on remote"
    echo "  --log <run>      Copy specific training log folder by name"
    echo "  <checkpoint>     Copy specific checkpoint folder by name"
    echo ""
    echo "Examples:"
    echo "  $0 playground_amp/checkpoints/wildrobot_amp_20251220_180000/final_amp_policy.pkl"
    echo "  $0 --checkpoints"
    echo "  $0 --latest"
    echo "  $0 wildrobot_amp_20251220_180000"
    echo "  $0 --logs"
    echo "  $0 --log run_20251220_183447"
    exit 1
fi

case "$1" in
    --checkpoints)
        list_checkpoints
        ;;
    --logs)
        list_logs
        ;;
    --log)
        if [ -z "$2" ]; then
            echo -e "${RED}Error: --log requires a run name${NC}"
            echo ""
            echo "Usage: $0 --log <run_name>"
            echo "Example: $0 --log run_20251220_183447"
            echo ""
            echo "To list available logs:"
            echo "  $0 --logs"
            exit 1
        fi
        copy_log "$2"
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
        if [[ "$1" =~ ^[a-zA-Z_]+[0-9]{8}_[0-9]{6}$ ]] || [[ "$1" =~ ^wildrobot ]]; then
            # Check if it exists in checkpoints on remote
            if ssh "$REMOTE_USER@$REMOTE_HOST" "[ -d '$REMOTE_BASE/playground_amp/checkpoints/$1' ]" 2>/dev/null; then
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
