#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG_NAME="${CONFIG_NAME:-pi05_so101}"
EXP_NAME="${EXP_NAME:-pi05_so101_pos_orange}"
DATASET_DIR="${DATASET_DIR:-/root/rivermind-data/train_dataset/lerobot/nkzwchen/dataset_combined_pos_orange_01}"
WEIGHTS_DIR="${WEIGHTS_DIR:-/root/rivermind-data/openpi/checkpoints/pi05_base}"
ASSET_ID="${ASSET_ID:-dataset_combined_pos_orange_01}"

BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-30000}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
KEEP_PERIOD="${KEEP_PERIOD:-10000}"
FSDP_DEVICES="${FSDP_DEVICES:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
WANDB_ENABLED="${WANDB_ENABLED:-false}"
OVERWRITE="${OVERWRITE:-false}"
RESUME="${RESUME:-false}"

export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"

export PYTHONPATH="${PWD}/src:${PWD}/packages/openpi-client/src:${PYTHONPATH:-}"

run_python() {
  .venv/bin/python "$@"
}

flag() {
  local name="$1"
  local value="$2"
  if [[ "$value" == "true" ]]; then
    printf -- "--%s" "$name"
  else
    printf -- "--no-%s" "$name"
  fi
}

if [[ ! -d "$DATASET_DIR" ]]; then
  echo "Dataset directory not found: $DATASET_DIR" >&2
  exit 1
fi

if [[ ! -d "$WEIGHTS_DIR" ]]; then
  echo "Weights directory not found: $WEIGHTS_DIR" >&2
  exit 1
fi

run_python scripts/compute_norm_stats.py \
  --config-name "$CONFIG_NAME" \
  --repo-id "$DATASET_DIR" \
  --asset-id "$ASSET_ID"

run_python scripts/train.py "$CONFIG_NAME" \
  --exp-name "$EXP_NAME" \
  --data.repo-id "$DATASET_DIR" \
  --data.assets.asset-id "$ASSET_ID" \
  --weight-loader.params-path "$WEIGHTS_DIR" \
  --batch-size "$BATCH_SIZE" \
  --num-train-steps "$NUM_TRAIN_STEPS" \
  --log-interval "$LOG_INTERVAL" \
  --save-interval "$SAVE_INTERVAL" \
  --keep-period "$KEEP_PERIOD" \
  --fsdp-devices "$FSDP_DEVICES" \
  --num-workers "$NUM_WORKERS" \
  --overwrite \
  "$(flag wandb-enabled "$WANDB_ENABLED")" \
  "$(flag overwrite "$OVERWRITE")" \
  "$(flag resume "$RESUME")"
