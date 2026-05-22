#!/bin/bash
# Blue-Green Deployment Script for VLA Policy

set -e

MODEL_DIR="models/vla"
ACTIVE_LINK="$MODEL_DIR/active"
BACKUP_LINK="$MODEL_DIR/backup"

usage() {
    echo "Usage: $0 [--model <path>] [--policy-id <id>] [--rollback]"
    exit 1
}

ROLLBACK=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL_PATH="$2"; shift 2 ;;
        --policy-id) POLICY_ID="$2"; shift 2 ;;
        --rollback) ROLLBACK=true; shift ;;
        *) usage ;;
    esac
done

if [ "$ROLLBACK" = true ]; then
    echo "Rolling back to previous policy..."
    if [ -L "$BACKUP_LINK" ]; then
        CURRENT=$(readlink -f "$ACTIVE_LINK")
        mv "$ACTIVE_LINK" "$ACTIVE_LINK.old"
        cp -r "$BACKUP_LINK" "$ACTIVE_LINK"
        rm -rf "$ACTIVE_LINK.old"
        echo "Rollback complete"
    else
        echo "No backup found for rollback"
        exit 1
    fi
    exit 0
fi

# Validate
if [ -z "$MODEL_PATH" ] || [ -z "$POLICY_ID" ]; then
    usage
fi

# Deploy
echo "Deploying policy: $POLICY_ID"
echo "Model path: $MODEL_PATH"

# Backup current active
if [ -L "$ACTIVE_LINK" ]; then
    rm -rf "$BACKUP_LINK"
    cp -r "$(readlink -f $ACTIVE_LINK)" "$BACKUP_LINK"
    echo "Backed up current policy"
fi

# Atomic symlink swap
ln -sfn "$MODEL_PATH" "$ACTIVE_LINK.tmp"
mv "$ACTIVE_LINK.tmp" "$ACTIVE_LINK"

echo "Policy deployed: $POLICY_ID"
echo "Active: $(readlink -f $ACTIVE_LINK)"
