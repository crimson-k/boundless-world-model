#!/bin/bash
set -euo pipefail

# Run one phase or the complete sequence:
#   MODEL_DIR=... DATASET_DIR=... TRAIN_METADATA_PATH=... \
#   VALID_METADATA_PATH=... ACTION_STAT_PATH=... bash scripts/train_stage2.sh all
# Individual phases: teacher, capture, adapter, bwm.

BWM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PEFM_ROOT="${PEFM_ROOT:-/data1/fangxuebin/pefm}"
PYTHON_BIN="${PYTHON_BIN:-/home/fangxuebin/.conda/envs/BWM/bin/python}"
STAGE="${1:-all}"

export PYTHONPATH="${BWM_ROOT}:${PEFM_ROOT}:${PYTHONPATH:-}"

PEFM_TEACHER_CONFIG="${PEFM_TEACHER_CONFIG:-${BWM_ROOT}/configs/stage2/pefm_teacher.yaml}"
TEACHER_OUTPUT="${TEACHER_OUTPUT:-${BWM_ROOT}/outputs/stage2/teacher}"
TEACHER_BUNDLE="${TEACHER_BUNDLE:-${TEACHER_OUTPUT}/frozen_evaluator.pt}"
VJEPA_CHECKPOINT="${VJEPA_CHECKPOINT:-/data1/fangxuebin/models/vjepa2/vjepa2-ac-vitg.pt}"
BWM_CONFIG="${BWM_CONFIG:-${BWM_ROOT}/configs/train/train_wan22_ti2v_5b_action_adaln.yaml}"
SELECTED_BLOCK="${SELECTED_BLOCK:-20}"
DIT_DIM="${DIT_DIM:-3072}"
MIN_TIMESTEP_BOUNDARY="${MIN_TIMESTEP_BOUNDARY:-0.0}"
MAX_TIMESTEP_BOUNDARY="${MAX_TIMESTEP_BOUNDARY:-1.0}"
NOISE_SAMPLING="{\"min_timestep_boundary\":${MIN_TIMESTEP_BOUNDARY},\"max_timestep_boundary\":${MAX_TIMESTEP_BOUNDARY}}"

CAPTURE_ROOT="${CAPTURE_ROOT:-${BWM_ROOT}/outputs/stage2_hidden/block_${SELECTED_BLOCK}}"
ADAPTER_OUTPUT="${ADAPTER_OUTPUT:-${BWM_ROOT}/outputs/stage2/adapter/block_${SELECTED_BLOCK}}"
ADAPTER_BUNDLE="${ADAPTER_BUNDLE:-${ADAPTER_OUTPUT}/best_adapter.pt}"
STAGE2_LOSS_WEIGHT="${STAGE2_LOSS_WEIGHT:-0.1}"
STAGE2_WEIGHT_SCHEDULE="${STAGE2_WEIGHT_SCHEDULE:-constant}"
BWM_OUTPUT="${BWM_OUTPUT:-${BWM_ROOT}/outputs/training/stage2_block_${SELECTED_BLOCK}}"

export DATASET_DIR TEACHER_OUTPUT VJEPA_CHECKPOINT

run_teacher() {
  echo "[Stage A] train and export the frozen V-JEPA--RSSM Teacher"
  cd "${PEFM_ROOT}"
  "${PYTHON_BIN}" -m pefm.training.train --config "${PEFM_TEACHER_CONFIG}"
}

capture_split() {
  local split="$1"
  local metadata="$2"
  local output="${CAPTURE_ROOT}/${split}"
  cd "${BWM_ROOT}"
  "${PYTHON_BIN}" scripts/capture_stage2_hidden.py \
    --config "${BWM_CONFIG}" \
    --model_paths "${MODEL_DIR:?set MODEL_DIR}" \
    --dataset_base_path "${DATASET_DIR:?set DATASET_DIR}" \
    --dataset_metadata_path "${metadata}" \
    --action_stat_path "${ACTION_STAT_PATH:?set ACTION_STAT_PATH}" \
    --capture_output "${output}" \
    --capture_selected_block "${SELECTED_BLOCK}" \
    --min_timestep_boundary "${MIN_TIMESTEP_BOUNDARY}" \
    --max_timestep_boundary "${MAX_TIMESTEP_BOUNDARY}"
}

run_capture() {
  echo "[Phase B input] capture train/validation DiT hidden states"
  capture_split train "${TRAIN_METADATA_PATH:?set TRAIN_METADATA_PATH}"
  capture_split valid "${VALID_METADATA_PATH:?set VALID_METADATA_PATH}"
}

run_adapter() {
  echo "[Phase B] train and export the spatially aligned Adapter"
  test -f "${TEACHER_BUNDLE}"
  cd "${PEFM_ROOT}"
  "${PYTHON_BIN}" -m pefm.training.train_adapter \
    --config pefm/training/config/adapter.yaml \
    --teacher-bundle "${TEACHER_BUNDLE}" \
    --train-manifest "${CAPTURE_ROOT}/train/manifest.jsonl" \
    --valid-manifest "${CAPTURE_ROOT}/valid/manifest.jsonl" \
    --output "${ADAPTER_OUTPUT}" \
    --dit-dim "${DIT_DIM}" \
    --selected-block "${SELECTED_BLOCK}" \
    --noise-sampling "${NOISE_SAMPLING}"
}

run_bwm() {
  echo "[Phase C] train Wan DiT with flow matching and 17-step prior KL"
  test -f "${TEACHER_BUNDLE}"
  test -f "${ADAPTER_BUNDLE}"
  cd "${BWM_ROOT}"
  "${PYTHON_BIN}" -m accelerate.commands.launch scripts/train.py \
    --config "${BWM_CONFIG}" \
    --model_paths "${MODEL_DIR:?set MODEL_DIR}" \
    --dataset_base_path "${DATASET_DIR:?set DATASET_DIR}" \
    --dataset_metadata_path "${TRAIN_METADATA_PATH:?set TRAIN_METADATA_PATH}" \
    --val_dataset_metadata_path "${VALID_METADATA_PATH:?set VALID_METADATA_PATH}" \
    --action_stat_path "${ACTION_STAT_PATH:?set ACTION_STAT_PATH}" \
    --trainable dit \
    --evaluator_loss_weight 0 \
    --stage2_teacher_path "${TEACHER_BUNDLE}" \
    --stage2_adapter_path "${ADAPTER_BUNDLE}" \
    --stage2_loss_weight "${STAGE2_LOSS_WEIGHT}" \
    --stage2_selected_block "${SELECTED_BLOCK}" \
    --stage2_weight_schedule "${STAGE2_WEIGHT_SCHEDULE}" \
    --output_path "${BWM_OUTPUT}" \
    --min_timestep_boundary "${MIN_TIMESTEP_BOUNDARY}" \
    --max_timestep_boundary "${MAX_TIMESTEP_BOUNDARY}"
}

case "${STAGE}" in
  teacher) run_teacher ;;
  capture) run_capture ;;
  adapter) run_adapter ;;
  bwm) run_bwm ;;
  all)
    run_teacher
    run_capture
    run_adapter
    run_bwm
    ;;
  *)
    echo "usage: $0 {teacher|capture|adapter|bwm|all}" >&2
    exit 2
    ;;
esac
