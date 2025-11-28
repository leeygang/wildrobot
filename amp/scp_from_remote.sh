#!/bin/bash

# SCP from Remote - AMP Version
# Copy files from remote amp/ directory to local amp/ directory

# Parse flags
USE_PUBLIC=false
FILENAME=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --public)
            USE_PUBLIC=true
            shift
            ;;
        *)
            FILENAME="$1"
            shift
            ;;
    esac
done

# Check if filename argument is provided
if [ -z "$FILENAME" ]; then
    echo "Usage: $0 [--public] <filename|checkpoint_folder_name>"
    echo ""
    echo "Options:"
    echo "  --public    Use public IP from \$LINUX_PUBLIC_IP and \$LINUX_PUBLIC_PORT"
    echo ""
    echo "Examples:"
    echo "  $0 walk.py                                       # Copy from local network"
    echo "  $0 --public walk.py                              # Copy from public IP"
    echo "  $0 logs/experiment.log                           # Copy from subdirectory"
    echo "  $0 phase1_contact_flat_20251127-123456           # Copy checkpoint folder"
    echo "  $0 --public phase1_contact_flat_20251127-123456  # Copy checkpoint via public IP"
    echo "  $0 baseline_flat_20251127-123456                 # Copy checkpoint folder"
    echo ""
    echo "For checkpoint folders:"
    echo "  - Automatically detects if it's a checkpoint folder name"
    echo "  - Copies from checkpoints/ directory on remote"
    echo "  - Saves to local checkpoints/ directory"
    exit 1
fi

# Remote configuration
REMOTE_USER="leeygang"
REMOTE_BASE_PATH="~/projects/wildrobot/amp"

# Configure remote host based on --public flag
if [ "$USE_PUBLIC" = true ]; then
    if [ -z "$LINUX_PUBLIC_IP" ]; then
        echo "Error: LINUX_PUBLIC_IP environment variable not set"
        echo "Set it with: export LINUX_PUBLIC_IP=your.public.ip"
        exit 1
    fi

    REMOTE_HOST="$LINUX_PUBLIC_IP"

    # Optional: Use custom SSH port if set
    if [ -n "$LINUX_PUBLIC_PORT" ]; then
        SCP_PORT_FLAG="-P $LINUX_PUBLIC_PORT"
        RSYNC_PORT_FLAG="-e ssh -p $LINUX_PUBLIC_PORT"
        echo "Using public IP: $REMOTE_HOST (port $LINUX_PUBLIC_PORT)"
    else
        SCP_PORT_FLAG=""
        RSYNC_PORT_FLAG=""
        echo "Using public IP: $REMOTE_HOST"
    fi
else
    REMOTE_HOST="linux-pc.local"
    SCP_PORT_FLAG=""
    RSYNC_PORT_FLAG=""
    echo "Using local network: $REMOTE_HOST"
fi

# Check if this looks like a checkpoint folder name
# Patterns:
#   - phase1_contact_flat_YYYYMMDD-HHMMSS
#   - phase2_imitation_flat_YYYYMMDD-HHMMSS
#   - phase3_hybrid_flat_YYYYMMDD-HHMMSS
#   - baseline_flat_YYYYMMDD-HHMMSS
#   - quickverify_*_YYYYMMDD-HHMMSS
if [[ "$FILENAME" =~ ^(phase[123]_[a-z]+|baseline|quickverify_[a-z_]+)_(flat|rough)_[0-9]{8}-[0-9]{6}$ ]]; then
    echo "Detected checkpoint folder: $FILENAME"

    # Set paths for checkpoint folder
    REMOTE_PATH="$REMOTE_BASE_PATH/checkpoints/$FILENAME"
    LOCAL_PATH="checkpoints/$FILENAME"

    # Create local checkpoints directory if it doesn't exist
    mkdir -p checkpoints

    echo "Copying checkpoint folder from remote..."
    echo "Remote: $REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"
    echo "Local:  $LOCAL_PATH"
    echo ""

    # Use rsync for better checkpoint copying (preserves structure, shows progress)
    # Falls back to scp if rsync is not available
    if command -v rsync &> /dev/null; then
        echo "Using rsync for efficient transfer..."
        if [ -n "$RSYNC_PORT_FLAG" ]; then
            rsync -avz --progress $RSYNC_PORT_FLAG "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/" "$LOCAL_PATH/"
        else
            rsync -avz --progress "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/" "$LOCAL_PATH/"
        fi
        RESULT=$?
    else
        echo "Using scp (rsync not found)..."
        scp -r $SCP_PORT_FLAG "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH" checkpoints/
        RESULT=$?
    fi

else
    # Regular file/directory copy
    REMOTE_PATH="$REMOTE_BASE_PATH/$FILENAME"
    LOCAL_PATH="$FILENAME"

    echo "Copying '$FILENAME' from remote..."
    echo "Remote: $REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"
    echo "Local:  $LOCAL_PATH"
    echo ""

    scp -r $SCP_PORT_FLAG "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH" "$LOCAL_PATH"
    RESULT=$?
fi

# Check if transfer was successful
if [ $RESULT -eq 0 ]; then
    echo ""
    echo "✅ Transfer completed successfully!"

    # Show checkpoint info if it was a checkpoint folder
    if [[ "$FILENAME" =~ ^(phase[123]_[a-z]+|baseline|quickverify_[a-z_]+)_(flat|rough)_[0-9]{8}-[0-9]{6}$ ]]; then
        echo ""
        echo "Checkpoint folder contents:"
        ls -lh "$LOCAL_PATH" | head -10
        echo ""
        echo "📊 Next steps:"
        echo ""

        # Check if final_policy.pkl exists
        if [ -f "$LOCAL_PATH/final_policy.pkl" ]; then
            echo "Final policy found! To visualize:"
            echo "  python visualize_policy.py --checkpoint $LOCAL_PATH/final_policy.pkl --output videos/${FILENAME}.mp4"
        fi

        # Check for config
        if [ -f "$LOCAL_PATH/config.json" ]; then
            echo ""
            echo "Config saved at: $LOCAL_PATH/config.json"
        fi

        echo ""
        echo "To analyze training logs:"
        echo "  cat $LOCAL_PATH/config.json | jq ."
    fi
else
    echo ""
    echo "❌ Transfer failed!"
    exit 1
fi
