#!/bin/bash

# Check if filename argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <filename|checkpoint_folder_name>"
    echo ""
    echo "Examples:"
    echo "  $0 myfile.txt                                    # Copy a file"
    echo "  $0 videos/policy.mp4                             # Copy from subdirectory"
    echo "  $0 wildrobot_locomotion_flat_20251126-170623     # Copy checkpoint folder"
    echo ""
    echo "For checkpoint folders:"
    echo "  - Automatically detects if it's a checkpoint folder name"
    echo "  - Copies from checkpoints/ directory on remote"
    echo "  - Saves to local checkpoints/ directory"
    exit 1
fi

# Get the filename from the first argument
FILENAME="$1"

# Remote destination
REMOTE_USER="leeygang"
REMOTE_HOST="linux-pc.local"
REMOTE_BASE_PATH="~/projects/wildrobot/playground"

# Check if this looks like a checkpoint folder name
# Pattern: wildrobot_locomotion_flat_YYYYMMDD-HHMMSS
if [[ "$FILENAME" =~ ^wildrobot_locomotion_flat_[0-9]{8}-[0-9]{6}$ ]]; then
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
        rsync -avz --progress "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/" "$LOCAL_PATH/"
        RESULT=$?
    else
        echo "Using scp (rsync not found)..."
        scp -r "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH" checkpoints/
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

    scp -r "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH" "$LOCAL_PATH"
    RESULT=$?
fi

# Check if transfer was successful
if [ $RESULT -eq 0 ]; then
    echo ""
    echo "✅ Transfer completed successfully!"

    # Show checkpoint info if it was a checkpoint folder
    if [[ "$FILENAME" =~ ^wildrobot_locomotion_flat_[0-9]{8}-[0-9]{6}$ ]]; then
        echo ""
        echo "Checkpoint folder contents:"
        ls -lh "$LOCAL_PATH" | head -10
        echo ""
        echo "To visualize the final policy:"
        echo "  python visualize_policy.py --checkpoint $LOCAL_PATH/final_policy.pkl --output videos/policy.mp4"
        echo ""
        echo "To convert intermediate checkpoint (e.g., 5M steps):"
        echo "  python convert_orbax_to_pkl.py --checkpoint $LOCAL_PATH/000005000000 --config default.yaml --output 5m_policy.pkl"
    fi
else
    echo ""
    echo "❌ Transfer failed!"
    exit 1
fi
