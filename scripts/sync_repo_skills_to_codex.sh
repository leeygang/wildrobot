#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="${ROOT_DIR}/skills"
CODEX_HOME_DIR="${CODEX_HOME:-${HOME}/.codex}"
DEST_DIR="${CODEX_HOME_DIR}/skills"

DRY_RUN=0
DELETE_EXTRA=0

usage() {
  cat <<'EOF'
Sync repo-local skills into the active Codex skills directory.

Usage:
  ./scripts/sync_repo_skills_to_codex.sh [--dry-run] [--delete]

Options:
  --dry-run   Show what would be copied without changing files.
  --delete    Delete skills in destination that do not exist in repo.
  -h, --help  Show this help.

Environment:
  CODEX_HOME  Optional override for Codex home. Defaults to ~/.codex
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --delete)
      DELETE_EXTRA=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ ! -d "$SRC_DIR" ]]; then
  echo "Source skills directory not found: $SRC_DIR" >&2
  exit 1
fi

mkdir -p "$DEST_DIR"

echo "Syncing skills"
echo "  source:      $SRC_DIR"
echo "  destination: $DEST_DIR"
echo "  dry-run:     $DRY_RUN"
echo "  delete:      $DELETE_EXTRA"

RSYNC_ARGS=(-a)
if [[ "$DRY_RUN" -eq 1 ]]; then
  RSYNC_ARGS+=(--dry-run)
fi
if [[ "$DELETE_EXTRA" -eq 1 ]]; then
  RSYNC_ARGS+=(--delete)
fi

if command -v rsync >/dev/null 2>&1; then
  rsync "${RSYNC_ARGS[@]}" "${SRC_DIR}/" "${DEST_DIR}/"
else
  if [[ "$DELETE_EXTRA" -eq 1 ]]; then
    echo "--delete requires rsync, but rsync is not installed." >&2
    exit 1
  fi
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "rsync not found; dry-run cannot be simulated safely." >&2
    exit 1
  fi
  cp -R "${SRC_DIR}/." "${DEST_DIR}/"
fi

echo "Skill sync complete."
