#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOCAL_CONFIG="${LOCAL_CONFIG:-${SCRIPT_DIR}/local.sh}"

if [[ -f "${LOCAL_CONFIG}" ]]; then
  # shellcheck source=/dev/null
  source "${LOCAL_CONFIG}"
fi

cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM="false"

CONFIG_PATH="${CONFIG_PATH:-configs/infer/infer.yaml}"
MODEL_PATHS="${MODEL_PATHS:-/data1/common_model/modelscope/Wan-AI/Wan2.2-TI2V-5B}"
CKPT_PATH="${CKPT_PATH:-outputs/training/train_wan22_ti2v_5b_action_adaln_pefm_w1/step-1000.safetensors}"
DATASET_BASE_PATH="${DATASET_BASE_PATH:-converted_dataset_task1}"
DATASET_METADATA_PATH="${DATASET_METADATA_PATH:-converted_dataset_task1/metadata.jsonl}"
ACTION_STAT_PATH="${ACTION_STAT_PATH:-converted_dataset_task1/stat.json}"
OUTPUT_PATH="${OUTPUT_PATH:-outputs/infer/sft_iter1000_pefm_task1}"
MAX_SAMPLES="${MAX_SAMPLES:-50}"
PYTHON_BIN="${PYTHON_BIN:-/home/fangxuebin/.conda/envs/BWM/bin/python}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-configs/train/accelerate_multi_gpu.yaml}"

"${PYTHON_BIN}" -m accelerate.commands.launch \
  --config_file "${ACCELERATE_CONFIG}" \
  scripts/infer_distributed.py \
  --config "${CONFIG_PATH}" \
  --model_paths "${MODEL_PATHS}" \
  --ckpt_path "${CKPT_PATH}" \
  --dataset_base_path "${DATASET_BASE_PATH}" \
  --dataset_metadata_path "${DATASET_METADATA_PATH}" \
  --action_stat_path "${ACTION_STAT_PATH}" \
  --output_path "${OUTPUT_PATH}" \
  --max_samples "${MAX_SAMPLES}"
  2>&1 | tee -a "${OUTPUT_PATH}/infer-2.log"
