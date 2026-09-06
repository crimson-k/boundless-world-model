#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM="false"

if [ -f scripts/train_local.sh ]; then
  source scripts/train_local.sh
fi

PYTHON_BIN="${PYTHON_BIN:-/home/fangxuebin/.conda/envs/BWM/bin/python}"
MODEL_DIR="${MODEL_DIR:-/data1/common_model/modelscope/Wan-AI/Wan2.2-TI2V-5B/}"
DATASET_DIR="${DATASET_DIR:-/data1/fangxuebin/boundless-world-model/converted_dataset_task1}"
TRAIN_METADATA_PATH="${TRAIN_METADATA_PATH:-${DATASET_DIR}/metadata_train.jsonl}"
VAL_METADATA_PATH="${VAL_METADATA_PATH:-${DATASET_DIR}/metadata_test.jsonl}"
ACTION_STAT_PATH="${ACTION_STAT_PATH:-/data1/fangxuebin/boundless-world-model/converted_dataset_task1/stat.json}"
CONFIG_PATH="${CONFIG_PATH:-configs/train/train_wan22_ti2v_5b_action_adaln.yaml}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-configs/train/accelerate_fsdp_multi_gpu.yaml}"
MACHINE_RANK="${MACHINE_RANK:-}"
DATASET_NUM_WORKERS="${DATASET_NUM_WORKERS:-1}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-1000}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
DETERMINISTIC="${DETERMINISTIC:-0}"
USE_WANDB="${USE_WANDB:-0}"
USE_SWANLAB="${USE_SWANLAB:-0}"
RUN_NAME="${RUN_NAME:-train_wan22_ti2v_5b_action_adaln_pefm_soft_w1}"
RESUME_FROM="${RESUME_FROM:-}"
OUTPUT_PATH="${OUTPUT_PATH:-outputs/training/train_wan22_ti2v_5b_action_adaln_pefm_soft_w1}"
EVALUATOR_PATH="${EVALUATOR_PATH:-}"
EVALUATOR_LOSS_WEIGHT="${EVALUATOR_LOSS_WEIGHT:-0.5}"
LOG_PATH="${LOG_PATH:-outputs/training/train_wan22_ti2v_5b_action_adaln_pefm_soft_w1/train.log}"

LAUNCH_CMD=(
  "${PYTHON_BIN}" -m accelerate.commands.launch
  --config_file "${ACCELERATE_CONFIG}"
)

if [ -n "${MACHINE_RANK}" ]; then
  LAUNCH_CMD+=(--machine_rank "${MACHINE_RANK}")
fi

TRAIN_CMD=(
  scripts/train.py
  --config "${CONFIG_PATH}"
  --model_paths "${MODEL_DIR}"
  --dataset_base_path "${DATASET_DIR}"
  --train_dataset_metadata_path "${TRAIN_METADATA_PATH}"
  --val_dataset_metadata_path "${VAL_METADATA_PATH}"
  --action_stat_path "${ACTION_STAT_PATH}"
  --dataset_num_workers "${DATASET_NUM_WORKERS}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --max_train_steps "${MAX_TRAIN_STEPS}"
  --save_steps "${SAVE_STEPS}"
  --evaluator_loss_weight "${EVALUATOR_LOSS_WEIGHT}"
  --use_gradient_checkpointing
)

if [ -n "${EVALUATOR_PATH}" ]; then
  TRAIN_CMD+=(--evaluator_path "${EVALUATOR_PATH}")
fi

if [ "${DETERMINISTIC}" = "1" ]; then
  TRAIN_CMD+=(--deterministic)
fi

if [ "${USE_WANDB}" = "1" ]; then
  TRAIN_CMD+=(--use_wandb)
fi

if [ "${USE_SWANLAB}" = "1" ]; then
  TRAIN_CMD+=(--use_swanlab)
fi

if [ -n "${RUN_NAME}" ]; then
  TRAIN_CMD+=(--run_name "${RUN_NAME}")
fi

if [ -n "${OUTPUT_PATH}" ]; then
  TRAIN_CMD+=(--output_path "${OUTPUT_PATH}")
fi

if [ -n "${RESUME_FROM}" ]; then
  TRAIN_CMD+=(--resume_from "${RESUME_FROM}")
fi

"${LAUNCH_CMD[@]}" "${TRAIN_CMD[@]}" 2>&1 | tee -a "${LOG_PATH}"
