#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM="false"

if [ -f scripts/train_local.sh ]; then
  source scripts/train_local.sh
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_PATH="${CONFIG_PATH:-configs/train/train_wan22_ti2v_5b_action_adaln.yaml}"
DATASET_NUM_WORKERS="${DATASET_NUM_WORKERS:-8}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-10000}"
SAVE_STEPS="${SAVE_STEPS:-5}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-}"
USE_GRADIENT_CHECKPOINTING="${USE_GRADIENT_CHECKPOINTING:-1}"
DETERMINISTIC="${DETERMINISTIC:-0}"
CKPT_PATH="${CKPT_PATH:-}"
RESUME_FROM="${RESUME_FROM:-}"
OUTPUT_PATH="${OUTPUT_PATH:-}"

CMD=(
  "${PYTHON_BIN}" scripts/train.py
  --config "${CONFIG_PATH}"
  --model_paths "${MODEL_DIR}"
  --dataset_base_path "${DATASET_DIR}"
  --dataset_metadata_path "${DATASET_METADATA_PATH}"
  --action_stat_path "${ACTION_STAT_PATH}"
  --dataset_num_workers "${DATASET_NUM_WORKERS}"
  --max_train_steps "${MAX_TRAIN_STEPS}"
  --save_steps "${SAVE_STEPS}"
)

if [ -n "${GRADIENT_ACCUMULATION_STEPS}" ]; then
  CMD+=(--gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}")
fi

if [ "${USE_GRADIENT_CHECKPOINTING}" = "1" ]; then
  CMD+=(--use_gradient_checkpointing)
fi

if [ "${DETERMINISTIC}" = "1" ]; then
  CMD+=(--deterministic)
fi

if [ -n "${OUTPUT_PATH}" ]; then
  CMD+=(--output_path "${OUTPUT_PATH}")
fi

if [ -n "${CKPT_PATH}" ]; then
  CMD+=(--ckpt_path "${CKPT_PATH}")
fi

if [ -n "${RESUME_FROM}" ]; then
  CMD+=(--resume_from "${RESUME_FROM}")
fi

"${CMD[@]}"
