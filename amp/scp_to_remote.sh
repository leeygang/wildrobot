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

# Track success/failure
TOTAL_FILES=${#FILES[@]}
SUCCESS_COUNT=0
FAILED_FILES=()

echo ""
echo "Transferring $TOTAL_FILES file(s)..."
echo ""

# Loop through all files
for FILENAME in "${FILES[@]}"; do
    # Check if file exists
    if [ ! -e "$FILENAME" ]; then
        echo "❌ Error: File '$FILENAME' does not exist - skipping"
        FAILED_FILES+=("$FILENAME")
        continue
    fi

    # Determine if it's a directory or file
    if [ -d "$FILENAME" ]; then
        echo "📁 Copying directory '$FILENAME'..."
        scp -r $SCP_PORT_FLAG "$FILENAME" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH$FILENAME"
    else
        echo "📄 Copying file '$FILENAME'..."
        scp $SCP_PORT_FLAG "$FILENAME" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH$FILENAME"
    fi

    # Check if scp was successful
    if [ $? -eq 0 ]; then
        echo "   ✅ Success"
        ((SUCCESS_COUNT++))
    else
        echo "   ❌ Failed"
        FAILED_FILES+=("$FILENAME")
    fi
    echo ""
done

# Summary
echo "="*60
echo "Transfer Summary:"
echo "  Total: $TOTAL_FILES"
echo "  Success: $SUCCESS_COUNT"
echo "  Failed: ${#FAILED_FILES[@]}"

if [ ${#FAILED_FILES[@]} -gt 0 ]; then
    echo ""
    echo "Failed files:"
    for file in "${FAILED_FILES[@]}"; do
        echo "  - $file"
    done
    exit 1
else
    echo ""
    echo "✅ All transfers completed successfully!"
fi
