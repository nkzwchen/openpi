#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export WANDB_MODE="${WANDB_MODE:-offline}"

DATASET_REPO_ID="${DATASET_REPO_ID:-nkzwchen/dataset_combined_pos_orange_01}"
DATASET_ROOT="${DATASET_ROOT:-/root/rivermind-data/train_dataset/lerobot/nkzwchen/dataset_combined_pos_orange_01}"
JOB_NAME="${JOB_NAME:-so101_act}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/${JOB_NAME}_$(date +%Y%m%d_%H%M%S)}"

BATCH_SIZE="${BATCH_SIZE:-8}"
STEPS="${STEPS:-100000}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SAVE_FREQ="${SAVE_FREQ:-20000}"
LOG_FREQ="${LOG_FREQ:-200}"
DEVICE="${DEVICE:-cuda}"

uv run lerobot-train \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --dataset.root="${DATASET_ROOT}" \
  --policy.type=act \
  --policy.device="${DEVICE}" \
  --policy.push_to_hub=false \
  --job_name="${JOB_NAME}" \
  --output_dir="${OUTPUT_DIR}" \
  --batch_size="${BATCH_SIZE}" \
  --steps="${STEPS}" \
  --num_workers="${NUM_WORKERS}" \
  --save_freq="${SAVE_FREQ}" \
  --log_freq="${LOG_FREQ}" \
  --eval_freq=0 \
  --wandb.enable=false \
  "$@"
