#!/bin/bash

# SCP to Remote - AMP Version
# Copy files from local amp/ directory to remote amp/ directory

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
    echo "Usage: $0 [--public] <filename>"
    echo ""
    echo "Options:"
    echo "  --public    Use public IP from \$LINUX_PUBLIC_IP and \$LINUX_PUBLIC_PORT"
    echo ""
    echo "Examples:"
    echo "  $0 walk.py                    # Copy to local network (linux-pc.local)"
    echo "  $0 --public walk.py           # Copy to public IP"
    echo "  $0 phase1_contact.yaml"
    exit 1
fi

# Check if file exists
if [ ! -e "$FILENAME" ]; then
    echo "Error: File '$FILENAME' does not exist"
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

# Determine if it's a directory or file
if [ -d "$FILENAME" ]; then
    echo "Copying directory '$FILENAME' to $REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH$FILENAME"
    scp -r $SCP_PORT_FLAG "$FILENAME" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH$FILENAME"
else
    echo "Copying file '$FILENAME' to $REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH$FILENAME"
    scp $SCP_PORT_FLAG "$FILENAME" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH$FILENAME"
fi

# Check if scp was successful
if [ $? -eq 0 ]; then
    echo "✅ Transfer completed successfully!"
else
    echo "❌ Transfer failed!"
    exit 1
fi
