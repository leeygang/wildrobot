#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG_PATH="${CONFIG_PATH:-training/configs/ppo_standing_push.yaml}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-training/checkpoints/staged_v0149}"
INITIAL_RESUME="${INITIAL_RESUME:-training/checkpoints/ppo_standing_push_v00148_20260310_203611-l1g2jb6r/checkpoint_200_26214400.pkl}"
NUM_ENVS="${NUM_ENVS:-}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-}"
SEED="${SEED:-}"

# Three-step unattended plan for v0.14.9:
#   stage 1: 80 iterations  -> first signal
#   stage 2: 40 iterations  -> extend to 120 total for go/no-go
#   stage 3: 80 iterations  -> extend to 200 total if still improving
STAGE_ITERS=(
  "${STAGE1_ITERS:-80}"
  "${STAGE2_ITERS:-40}"
  "${STAGE3_ITERS:-80}"
)

timestamp="$(date +%Y%m%d_%H%M%S)"
run_root="${CHECKPOINT_ROOT}/run_${timestamp}"
mkdir -p "$run_root"

latest_job_dir() {
  find "$run_root" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' \
    | sort -n \
    | tail -n1 \
    | cut -d' ' -f2-
}

latest_checkpoint_in_dir() {
  local dir="$1"
  find "$dir" -maxdepth 1 -type f -name 'checkpoint_*.pkl' -printf '%f\n' \
    | sort -V \
    | tail -n1 \
    | sed "s|^|$dir/|"
}

best_checkpoint_in_dir() {
  local dir="$1"
  find "$dir" -maxdepth 1 -type f -name 'best_checkpoint_*.pkl' -printf '%f\n' \
    | sort -V \
    | tail -n1 \
    | sed "s|^|$dir/|"
}

print_stage_banner() {
  local stage_idx="$1"
  local iters="$2"
  local resume_from="$3"
  echo
  echo "============================================================"
  echo "Stage ${stage_idx}: run ${iters} iterations"
  echo "Checkpoint root: ${run_root}"
  echo "Config: ${CONFIG_PATH}"
  if [[ -n "$resume_from" ]]; then
    echo "Resume from: ${resume_from}"
  fi
  echo "============================================================"
}

resume_ckpt="$INITIAL_RESUME"

if [[ -n "$resume_ckpt" && ! -f "$resume_ckpt" ]]; then
  echo "ERROR: initial resume checkpoint not found: ${resume_ckpt}" >&2
  exit 1
fi

for idx in "${!STAGE_ITERS[@]}"; do
  stage_num="$((idx + 1))"
  stage_iters="${STAGE_ITERS[$idx]}"

  print_stage_banner "$stage_num" "$stage_iters" "$resume_ckpt"

  cmd=(
    uv run python training/train.py
    --config "$CONFIG_PATH"
    --iterations "$stage_iters"
    --checkpoint-dir "$run_root"
  )

  if [[ -n "$resume_ckpt" ]]; then
    cmd+=(--resume "$resume_ckpt")
  fi
  if [[ -n "$NUM_ENVS" ]]; then
    cmd+=(--num-envs "$NUM_ENVS")
  fi
  if [[ -n "$CHECKPOINT_INTERVAL" ]]; then
    cmd+=(--checkpoint-interval "$CHECKPOINT_INTERVAL")
  fi
  if [[ -n "$SEED" ]]; then
    cmd+=(--seed "$SEED")
  fi

  printf 'Command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'

  "${cmd[@]}"

  job_dir="$(latest_job_dir)"
  if [[ -z "$job_dir" || ! -d "$job_dir" ]]; then
    echo "ERROR: could not find checkpoint job directory under ${run_root}" >&2
    exit 1
  fi

  latest_ckpt="$(latest_checkpoint_in_dir "$job_dir")"
  best_ckpt="$(best_checkpoint_in_dir "$job_dir" || true)"

  if [[ -z "$latest_ckpt" || ! -f "$latest_ckpt" ]]; then
    echo "ERROR: could not find latest checkpoint in ${job_dir}" >&2
    exit 1
  fi

  resume_ckpt="$latest_ckpt"

  echo
  echo "Stage ${stage_num} complete"
  echo "  job dir:     ${job_dir}"
  echo "  latest ckpt: ${latest_ckpt}"
  if [[ -n "$best_ckpt" && -f "$best_ckpt" ]]; then
    echo "  best ckpt:   ${best_ckpt}"
  fi
done

echo
echo "============================================================"
echo "Staged training finished"
echo "Run root: ${run_root}"
echo "Final resume checkpoint: ${resume_ckpt}"
echo "============================================================"
