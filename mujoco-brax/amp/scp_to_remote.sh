#!/bin/bash

# SCP to Remote - AMP Version
# Copy files from local amp/ directory to remote amp/ directory

# Parse flags
USE_PUBLIC=false
USE_AMP=false
FILES=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --public)
            USE_PUBLIC=true
            shift
            ;;
        --amp)
            USE_AMP=true
            shift
            ;;
        *)
            FILES+=("$1")
            shift
            ;;
    esac
done

# If --amp flag is set, collect all .py and .yaml files (including subdirectories)
if [ "$USE_AMP" = true ]; then
    echo "🔍 Collecting all .py and .yaml files from amp/ and common/ (including subdirectories)..."
    echo ""

    # Directories to exclude from search (as regex pattern for grep)
    EXCLUDE_PATTERN="\.venv|__pycache__|\.git|training_logs|checkpoints|wandb|\.pytest_cache|models"

    # Collect files from amp/ directory (including subdirectories like tests/)
    AMP_FILES=()
    if [ -d "." ]; then
        while IFS= read -r file; do
            AMP_FILES+=("$file")
        done < <(find . -type f \( -name "*.py" -o -name "*.yaml" \) | grep -vE "$EXCLUDE_PATTERN" | sort)
    fi

    # Collect files from common/ directory (including subdirectories)
    COMMON_FILES=()
    if [ -d "../common" ]; then
        while IFS= read -r file; do
            COMMON_FILES+=("$file")
        done < <(find ../common -type f \( -name "*.py" -o -name "*.yaml" \) 2>/dev/null | grep -vE "$EXCLUDE_PATTERN" | sort)
    fi

    # Add collected files to FILES array
    FILES=("${AMP_FILES[@]}" "${COMMON_FILES[@]}")

    echo "Found ${#AMP_FILES[@]} files in amp/ (including subdirectories)"
    echo "Found ${#COMMON_FILES[@]} files in common/ (including subdirectories)"
    echo ""
    echo "Excluded: .venv, __pycache__, .git, training_logs, checkpoints, wandb, .pytest_cache, models"
    echo ""
fi

# Check if at least one file is provided
if [ ${#FILES[@]} -eq 0 ]; then
    echo "Usage: $0 [--public] [--amp] [file1] [file2] [file3] ..."
    echo ""
    echo "Options:"
    echo "  --public    Use public IP from \$LINUX_PUBLIC_IP and \$LINUX_PUBLIC_PORT"
    echo "  --amp       Copy all .py and .yaml files from amp/ and common/ directories"
    echo ""
    echo "Examples:"
    echo "  $0 walk.py                              # Copy single file (local network)"
    echo "  $0 walk.py train.py                     # Copy multiple files"
    echo "  $0 --public walk.py train.py            # Copy multiple files via public IP"
    echo "  $0 phase1_contact.yaml baseline.yaml   # Copy config files"
    echo "  $0 rewards/                             # Copy entire directory"
    echo "  $0 --amp                                # Copy all .py and .yaml from amp/ and common/"
    echo "  $0 --amp --public                       # Copy all .py and .yaml via public IP"
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

# Transfer all valid files preserving directory structure
echo ""
echo "Transferring ${#VALID_FILES[@]} file(s) to $REMOTE_HOST..."
echo "(You will be prompted for password once)"
echo ""


# Separate amp/ files and common/ files
AMP_ONLY_FILES=()
COMMON_ONLY_FILES=()

for file in "${VALID_FILES[@]}"; do
    if [[ "$file" == ../common/* ]]; then
        # Strip ../common/ prefix for remote path
        COMMON_ONLY_FILES+=("$file")
    else
        # amp/ files (including subdirectories like tests/)
        AMP_ONLY_FILES+=("$file")
    fi
done

# Transfer amp/ files with --relative (preserves tests/ structure)
if [ ${#AMP_ONLY_FILES[@]} -gt 0 ]; then
    echo "Transferring ${#AMP_ONLY_FILES[@]} amp/ files (preserving structure)..."
    if [ "$USE_PUBLIC" = true ] && [ -n "$LINUX_PUBLIC_PORT" ]; then
        rsync -avz --relative -e "ssh -p $LINUX_PUBLIC_PORT" "${AMP_ONLY_FILES[@]}" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"
    else
        rsync -avz --relative "${AMP_ONLY_FILES[@]}" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"
    fi

    if [ $? -ne 0 ]; then
        echo "❌ amp/ transfer failed!"
        exit 1
    fi
fi

# Transfer common/ files to remote common/ (without --relative)
if [ ${#COMMON_ONLY_FILES[@]} -gt 0 ]; then
    echo "Transferring ${#COMMON_ONLY_FILES[@]} common/ files..."

    # Create remote common/ directory first
    if [ "$USE_PUBLIC" = true ] && [ -n "$LINUX_PUBLIC_PORT" ]; then
        ssh -p "$LINUX_PUBLIC_PORT" "$REMOTE_USER@$REMOTE_HOST" "mkdir -p ~/projects/wildrobot/common"
        rsync -avz -e "ssh -p $LINUX_PUBLIC_PORT" "${COMMON_ONLY_FILES[@]}" "$REMOTE_USER@$REMOTE_HOST:~/projects/wildrobot/common/"
    else
        ssh "$REMOTE_USER@$REMOTE_HOST" "mkdir -p ~/projects/wildrobot/common"
        rsync -avz "${COMMON_ONLY_FILES[@]}" "$REMOTE_USER@$REMOTE_HOST:~/projects/wildrobot/common/"
    fi

    if [ $? -ne 0 ]; then
        echo "❌ common/ transfer failed!"
        exit 1
    fi
fi

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
