#!/bin/bash

# SCP to Remote - AMP Version
# Copy files from local amp/ directory to remote amp/ directory

# Parse flags
USE_PUBLIC=false
FILES=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --public)
            USE_PUBLIC=true
            shift
            ;;
        *)
            FILES+=("$1")
            shift
            ;;
    esac
done

# Check if at least one file is provided
if [ ${#FILES[@]} -eq 0 ]; then
    echo "Usage: $0 [--public] <file1> [file2] [file3] ..."
    echo ""
    echo "Options:"
    echo "  --public    Use public IP from \$LINUX_PUBLIC_IP and \$LINUX_PUBLIC_PORT"
    echo ""
    echo "Examples:"
    echo "  $0 walk.py                              # Copy single file (local network)"
    echo "  $0 walk.py train.py                     # Copy multiple files"
    echo "  $0 --public walk.py train.py            # Copy multiple files via public IP"
    echo "  $0 phase1_contact.yaml baseline.yaml   # Copy config files"
    echo "  $0 rewards/                             # Copy entire directory"
    exit 1
fi

# Remote destination
REMOTE_USER="leeygang"
REMOTE_PATH="~/projects/wildrobot/amp/"

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
        echo "Using public IP: $REMOTE_HOST (port $LINUX_PUBLIC_PORT)"
    else
        SCP_PORT_FLAG=""
        echo "Using public IP: $REMOTE_HOST"
    fi
else
    REMOTE_HOST="linux-pc.local"
    SCP_PORT_FLAG=""
    echo "Using local network: $REMOTE_HOST"
fi

# Track files and check existence
TOTAL_FILES=${#FILES[@]}
VALID_FILES=()
INVALID_FILES=()

echo ""
echo "Checking $TOTAL_FILES file(s)..."
echo ""

# Check all files exist first
for FILENAME in "${FILES[@]}"; do
    if [ ! -e "$FILENAME" ]; then
        echo "❌ Error: File '$FILENAME' does not exist - skipping"
        INVALID_FILES+=("$FILENAME")
    else
        VALID_FILES+=("$FILENAME")
        if [ -d "$FILENAME" ]; then
            echo "✓ Directory: $FILENAME"
        else
            echo "✓ File: $FILENAME"
        fi
    fi
done

# Exit if no valid files
if [ ${#VALID_FILES[@]} -eq 0 ]; then
    echo ""
    echo "❌ No valid files to transfer!"
    exit 1
fi

# Transfer all valid files in a single scp command (ONE password prompt!)
echo ""
echo "Transferring ${#VALID_FILES[@]} file(s) to $REMOTE_HOST..."
echo "(You will be prompted for password once)"
echo ""

# Build scp command with all files
# Use -r flag to handle both files and directories
scp -r $SCP_PORT_FLAG "${VALID_FILES[@]}" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"

# Check if scp was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "="*60
    echo "✅ All transfers completed successfully!"
    echo ""
    echo "Transferred files:"
    for file in "${VALID_FILES[@]}"; do
        echo "  ✓ $file"
    done

    if [ ${#INVALID_FILES[@]} -gt 0 ]; then
        echo ""
        echo "Skipped files (not found):"
        for file in "${INVALID_FILES[@]}"; do
            echo "  ✗ $file"
        done
    fi
else
    echo ""
    echo "❌ Transfer failed!"
    exit 1
fi
