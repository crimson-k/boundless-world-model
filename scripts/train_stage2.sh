#!/bin/bash
set -euo pipefail

# 每个阶段在独立 tmux session 中启动，先检查 A 的验证指标，再开始 B/C。
BWM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PEFM_ROOT="${PEFM_ROOT:-${BWM_ROOT}/../pefm}"
PYTHON_BIN="${PYTHON_BIN:-/home/fangxuebin/.conda/envs/BWM/bin/python}"
STAGE="${1:?usage: train_stage2.sh capture|cache|predictor|no-actions|adapter|baseline|align|predict}"
shift
export PYTHONPATH="${BWM_ROOT}:${PEFM_ROOT}:${PYTHONPATH:-}"
MODEL_DIR="${MODEL_DIR:-/data1/common_model/modelscope/Wan-AI/Wan2.2-TI2V-5B}"
: "${DATASET_DIR:?set DATASET_DIR to the existing dataset directory}"
: "${TEACHER_BUNDLE:?set TEACHER_BUNDLE to the existing frozen_evaluator.pt}"
TRAIN_METADATA_PATH="${TRAIN_METADATA_PATH:-${DATASET_DIR}/metadata_train.jsonl}"
VALID_METADATA_PATH="${VALID_METADATA_PATH:-${DATASET_DIR}/metadata_test.jsonl}"
ACTION_STAT_PATH="${ACTION_STAT_PATH:-${DATASET_DIR}/stat.json}"
BWM_CONFIG="${BWM_ROOT}/configs/train/train_wan22_ti2v_5b_action_adaln.yaml"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${BWM_ROOT}/configs/train/accelerate_fsdp_multi_gpu.yaml}"
SELECTED_BLOCK="${SELECTED_BLOCK:-15}"
DIT_DIM="${DIT_DIM:-3072}"
SEED="${SEED:-0}"
HEIGHT="${HEIGHT:-192}"
WIDTH="${WIDTH:-256}"
SIGMA_MAX="${SIGMA_MAX:-1.0}"
# FM 始终保留完整噪声采样范围，辅助损失的区间由 Adapter checkpoint 固定。
NOISE_SAMPLING='{"min_timestep_boundary":0.0,"max_timestep_boundary":1.0}'
RUN_ROOT="${RUN_ROOT:-${BWM_ROOT}/outputs/stage2_B}"
CAPTURE_ROOT="${RUN_ROOT}/hidden/block_${SELECTED_BLOCK}"
CACHE_ROOT="${RUN_ROOT}/embeddings"
PREDICTOR_BUNDLE="${PREDICTOR_BUNDLE:-${RUN_ROOT}/predictor/predictor_best.pt}"
ADAPTER_OUTPUT="${RUN_ROOT}/adapter/block_${SELECTED_BLOCK}"
ADAPTER_BUNDLE="${ADAPTER_BUNDLE:-${ADAPTER_OUTPUT}/adapter_best.pt}"
cd "${BWM_ROOT}"

case "${STAGE}" in
  capture)
    for split in train valid; do
      metadata="${TRAIN_METADATA_PATH}"
      if [ "$split" = valid ]; then metadata="${VALID_METADATA_PATH}"; fi
      "${PYTHON_BIN}" scripts/capture_stage2_hidden.py --config "${BWM_CONFIG}" \
        --model_paths "${MODEL_DIR}" --dataset_base_path "${DATASET_DIR}" \
        --dataset_metadata_path "${metadata}" --action_stat_path "${ACTION_STAT_PATH}" \
        --capture_output "${CAPTURE_ROOT}/${split}" --capture_selected_block "${SELECTED_BLOCK}" \
        --height "${HEIGHT}" --width "${WIDTH}" \
        --min_timestep_boundary 0 --max_timestep_boundary 1 --seed "${SEED}" "$@"
    done ;;
  cache)
    for split in train valid; do
      "${PYTHON_BIN}" -m pefm.training.train_predictor cache --teacher-bundle "${TEACHER_BUNDLE}" \
        --train-manifest "${CAPTURE_ROOT}/${split}/manifest.jsonl" \
        --output "${CACHE_ROOT}/${split}" "$@"
    done ;;
  predictor|no-actions)
    extra=()
    if [ "${STAGE}" = no-actions ]; then extra+=(--no-actions); fi
    "${PYTHON_BIN}" -m pefm.training.train_predictor train \
      --train-manifest "${CACHE_ROOT}/train/manifest.jsonl" \
      --valid-manifest "${CACHE_ROOT}/valid/manifest.jsonl" \
      --output "${RUN_ROOT}/${STAGE}" --seed "${SEED}" "${extra[@]}" "$@" ;;
  adapter)
    "${PYTHON_BIN}" -m pefm.training.train_adapter --teacher-bundle "${TEACHER_BUNDLE}" \
      --predictor-bundle "${PREDICTOR_BUNDLE}" --train-manifest "${CAPTURE_ROOT}/train/manifest.jsonl" \
      --valid-manifest "${CAPTURE_ROOT}/valid/manifest.jsonl" --output "${ADAPTER_OUTPUT}" \
      --dit-dim "${DIT_DIM}" --selected-block "${SELECTED_BLOCK}" \
      --noise-sampling "${NOISE_SAMPLING}" --sigma-max "${SIGMA_MAX}" --seed "${SEED}" "$@" ;;
  baseline|align|predict)
    weight="${STAGE2_LOSS_WEIGHT:-0.1}"
    pred_scale=1
    if [ "${STAGE}" = baseline ]; then weight=0; fi
    if [ "${STAGE}" = align ]; then pred_scale=0; fi
    export NCCL_CUMEM_HOST_ENABLE="${NCCL_CUMEM_HOST_ENABLE:-0}"
    "${PYTHON_BIN}" -m accelerate.commands.launch --config_file "${ACCELERATE_CONFIG}" scripts/train.py \
      --config "${BWM_CONFIG}" --model_paths "${MODEL_DIR}" --dataset_base_path "${DATASET_DIR}" \
      --dataset_metadata_path "${TRAIN_METADATA_PATH}" --val_dataset_metadata_path "${VALID_METADATA_PATH}" \
      --action_stat_path "${ACTION_STAT_PATH}" --trainable dit --evaluator_loss_weight 0 \
      --stage2_teacher_path "${TEACHER_BUNDLE}" --stage2_adapter_path "${ADAPTER_BUNDLE}" \
      --stage2_loss_weight "${weight}" --stage2_align_scale 1 --stage2_pred_scale "${pred_scale}" \
      --stage2_selected_block "${SELECTED_BLOCK}" --output_path "${RUN_ROOT}/bwm_${STAGE}" \
      --height "${HEIGHT}" --width "${WIDTH}" \
        --min_timestep_boundary 0 --max_timestep_boundary 1 --seed "${SEED}" "$@" ;;
  *) echo "Unknown phase: ${STAGE}" >&2; exit 2 ;;
esac
