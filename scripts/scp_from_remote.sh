#!/bin/bash

# Sync wildrobot files from Ubuntu GPU machine to Mac
#
# Usage:
#   ./scp_from_remote.sh [--public] <filename>           # Copy single file/directory
#   ./scp_from_remote.sh [--public] --checkpoints        # List available checkpoints
#   ./scp_from_remote.sh [--public] --latest             # Copy latest checkpoint
#   ./scp_from_remote.sh [--public] <checkpoint_name>    # Copy specific checkpoint folder
#   ./scp_from_remote.sh [--public] --logs               # List available wandb runs
#   ./scp_from_remote.sh [--public] --run <run_name>     # Copy both checkpoint and wandb log for a run
#
# Options:
#   --public    Use $LINUX_PUBLIC_IP instead of linux-pc.local
#
# Examples:
#   ./scp_from_remote.sh --public --checkpoints
#   ./scp_from_remote.sh --public --latest
#   ./scp_from_remote.sh --run run-20251228_011308-xw1fu3n6

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
    echo -e "\n${YELLOW}=== Available W&B Runs on Remote ===${NC}\n"

    ssh "$REMOTE_USER@$REMOTE_HOST" "ls -lht $REMOTE_BASE/playground_amp/wandb/ 2>/dev/null | grep '^d' | grep 'run-' | head -20 || echo 'No wandb runs found'"

    echo -e "\n${CYAN}To copy a run (checkpoint + wandb log):${NC}"
    echo "  ./scp_from_remote.sh --run <run_name>"
    echo ""
    echo -e "${CYAN}Example:${NC}"
    echo "  ./scp_from_remote.sh --run run-20251228_011308-xw1fu3n6"
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

copy_wandb_log() {
    local RUN_NAME="$1"
    local REMOTE_PATH="$REMOTE_BASE/playground_amp/wandb/$RUN_NAME"
    local LOCAL_PATH="playground_amp/wandb/$RUN_NAME"

    # Create local wandb directory
    mkdir -p playground_amp/wandb

    echo -e "${YELLOW}Copying W&B run folder:${NC} $RUN_NAME"
    echo -e "  Remote: $REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"
    echo -e "  Local:  $LOCAL_PATH"
    echo ""

    # Check if the run exists on remote
    if ! ssh "$REMOTE_USER@$REMOTE_HOST" "[ -d '$REMOTE_PATH' ]" 2>/dev/null; then
        echo -e "${RED}✗ W&B run not found on remote: $RUN_NAME${NC}"
        return 1
    fi

    # Use rsync for efficient copying
    if command -v rsync &> /dev/null; then
        echo -e "${CYAN}Using rsync for efficient transfer...${NC}"
        rsync -avz --progress "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/" "$LOCAL_PATH/"
        RESULT=$?
    else
        echo -e "${CYAN}Using scp (rsync not found)...${NC}"
        scp -r "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH" playground_amp/wandb/
        RESULT=$?
    fi

    if [ $RESULT -eq 0 ]; then
        echo -e "\n${GREEN}✓ W&B run transfer completed!${NC}"
        echo ""
        echo -e "${YELLOW}Run contents:${NC}"
        ls -lh "$LOCAL_PATH" 2>/dev/null | head -15
        echo ""
        echo -e "${CYAN}To sync with W&B:${NC}"
        echo "  wandb sync $LOCAL_PATH"
    else
        echo -e "\n${RED}✗ Transfer failed!${NC}"
        return 1
    fi
}

copy_run() {
    local RUN_NAME="$1"

    echo -e "\n${YELLOW}=== Copying Run: $RUN_NAME ===${NC}\n"

    # Extract run_id from wandb run name
    # Format: run-YYYYMMDD_HHMMSS-xxxxxxxx -> YYYYMMDD_HHMMSS-xxxxxxxx
    # Example: run-20251228_205534-uf665cr6 -> 20251228_205534-uf665cr6
    local RUN_ID=$(echo "$RUN_NAME" | sed 's/^run-//')

    if [ -z "$RUN_ID" ] || [ "$RUN_ID" = "$RUN_NAME" ]; then
        echo -e "${RED}✗ Could not extract run_id from run name: $RUN_NAME${NC}"
        echo "Expected format: run-YYYYMMDD_HHMMSS-xxxxxxxx"
        return 1
    fi

    # Check if wandb run exists on remote
    local WANDB_REMOTE="$REMOTE_BASE/playground_amp/wandb/$RUN_NAME"
    if ! ssh "$REMOTE_USER@$REMOTE_HOST" "[ -d '$WANDB_REMOTE' ]" 2>/dev/null; then
        echo -e "${RED}✗ W&B run not found on remote: $RUN_NAME${NC}"
        echo ""
        echo -e "${CYAN}Available W&B runs:${NC}"
        ssh "$REMOTE_USER@$REMOTE_HOST" "ls -1 $REMOTE_BASE/playground_amp/wandb/ 2>/dev/null | grep '^run-' | head -10"
        return 1
    fi

    # Find matching checkpoint folder based on the run_id suffix
    # New checkpoint format: {config_name}_v{version}_{timestamp}-{wandb_run_id}
    # Example: ppo_walking_v01005_20251228_205534-uf665cr6
    # We match the suffix: 20251228_205534-uf665cr6
    local CHECKPOINT_NAME=$(ssh "$REMOTE_USER@$REMOTE_HOST" "ls -1 $REMOTE_BASE/playground_amp/checkpoints/ 2>/dev/null | grep '$RUN_ID$' | head -1")

    # Step 1: Copy wandb log
    echo -e "${CYAN}Step 1/2: Copying W&B log...${NC}"
    echo ""
    copy_wandb_log "$RUN_NAME"
    WANDB_RESULT=$?

    # Step 2: Copy checkpoint if found
    echo ""
    echo -e "${CYAN}Step 2/2: Copying checkpoint...${NC}"
    echo ""

    if [ -n "$CHECKPOINT_NAME" ]; then
        copy_checkpoint "$CHECKPOINT_NAME"
        CHECKPOINT_RESULT=$?
    else
        echo -e "${YELLOW}⚠ No matching checkpoint found for run_id: $RUN_ID${NC}"
        echo ""
        echo -e "${CYAN}This could mean:${NC}"
        echo "  1. Training hasn't saved a checkpoint yet (not reached checkpoint interval)"
        echo "  2. Training failed before saving a checkpoint"
        echo "  3. Checkpoint was saved with a different naming scheme"
        echo ""
        echo -e "${CYAN}Available checkpoints (most recent first):${NC}"
        ssh "$REMOTE_USER@$REMOTE_HOST" "ls -1t $REMOTE_BASE/playground_amp/checkpoints/ 2>/dev/null | head -10"
        echo ""
        echo -e "${CYAN}To copy a specific checkpoint manually:${NC}"
        echo "  ./scp_from_remote.sh <checkpoint_folder_name>"
        CHECKPOINT_RESULT=1
    fi

    # Summary
    echo ""
    echo -e "${YELLOW}=== Transfer Summary ===${NC}"
    if [ $WANDB_RESULT -eq 0 ]; then
        echo -e "${GREEN}✓ W&B Log: $RUN_NAME${NC}"
    else
        echo -e "${RED}✗ W&B Log: $RUN_NAME (failed)${NC}"
    fi

    if [ $CHECKPOINT_RESULT -eq 0 ]; then
        echo -e "${GREEN}✓ Checkpoint: $CHECKPOINT_NAME${NC}"
    else
        echo -e "${YELLOW}⚠ Checkpoint: not found or failed${NC}"
    fi
}

# Main
if [ $# -eq 0 ]; then
    echo "Usage: $0 <filename|--checkpoints|--latest|--logs|--run <name>|checkpoint_name>"
    echo ""
    echo "Options:"
    echo "  <filename>       Copy specific file or directory from remote"
    echo "  --checkpoints    List available checkpoints on remote"
    echo "  --latest         Copy the most recent checkpoint"
    echo "  --logs           List available W&B runs on remote"
    echo "  --run <name>     Copy both checkpoint and W&B log for a run"
    echo "  <checkpoint>     Copy specific checkpoint folder by name"
    echo ""
    echo "Examples:"
    echo "  $0 playground_amp/checkpoints/wildrobot_amp_20251220_180000/final_amp_policy.pkl"
    echo "  $0 --checkpoints"
    echo "  $0 --latest"
    echo "  $0 wildrobot_amp_20251220_180000"
    echo "  $0 --logs"
    echo "  $0 --run run-20251228_011308-xw1fu3n6"
    exit 1
fi

case "$1" in
    --checkpoints)
        list_checkpoints
        ;;
    --logs)
        list_logs
        ;;
    --run)
        if [ -z "$2" ]; then
            echo -e "${RED}Error: --run requires a run name${NC}"
            echo ""
            echo "Usage: $0 --run <run_name>"
            echo "Example: $0 --run run-20251228_011308-xw1fu3n6"
            echo ""
            echo "To list available runs:"
            echo "  $0 --logs"
            exit 1
        fi
        copy_run "$2"
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
