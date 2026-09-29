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

DATASET_DIR="${DATASET_DIR:-/data1/fangxuebin/boundless-world-model/converted_dataset_bwm_6tasks_256x192}"
PEFM_TEACHER_CONFIG="${PEFM_TEACHER_CONFIG:-${BWM_ROOT}/configs/stage2/pefm_teacher.yaml}"
TEACHER_OUTPUT="${TEACHER_OUTPUT:-${BWM_ROOT}/outputs/stage2_256x192/teacher}"
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
ADAPTER_BUNDLE="${ADAPTER_BUNDLE:-}"
ADAPTER_SELECTION="${ADAPTER_SELECTION:-${ADAPTER_OUTPUT}/adapter_selection.json}"
ADAPTER_NUM_PROCESSES="${ADAPTER_NUM_PROCESSES:-4}"
ADAPTER_MIXED_PRECISION="${ADAPTER_MIXED_PRECISION:-bf16}"
ADAPTER_MAIN_PROCESS_PORT="${ADAPTER_MAIN_PROCESS_PORT:-29511}"
STAGE2_LOSS_WEIGHT="${STAGE2_LOSS_WEIGHT:-0.1}"
STAGE2_WEIGHT_SCHEDULE="${STAGE2_WEIGHT_SCHEDULE:-constant}"
BWM_OUTPUT="${BWM_OUTPUT:-${BWM_ROOT}/outputs/training/stage2_block_${SELECTED_BLOCK}}"
BWM_MAX_TRAIN_STEPS="${BWM_MAX_TRAIN_STEPS:-8000}"
BWM_SAVE_STEPS="${BWM_SAVE_STEPS:-2000}"

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
    --model_paths "${MODEL_DIR:-/data1/common_model/modelscope/Wan-AI/Wan2.2-TI2V-5B/}" \
    --dataset_base_path "${DATASET_DIR}" \
    --dataset_metadata_path "${metadata}" \
    --action_stat_path "${ACTION_STAT_PATH:-${DATASET_DIR}/stat.json}" \
    --capture_output "${output}" \
    --capture_selected_block "${SELECTED_BLOCK}" \
    --stage2_teacher_path "${TEACHER_BUNDLE}" \
    --min_timestep_boundary "${MIN_TIMESTEP_BOUNDARY}" \
    --max_timestep_boundary "${MAX_TIMESTEP_BOUNDARY}"
}

run_capture() {
  echo "[Phase B input] capture train/validation DiT hidden states"
  capture_split train "${TRAIN_METADATA_PATH:-${DATASET_DIR}/metadata_train.jsonl}"
  capture_split valid "${VALID_METADATA_PATH:-${DATASET_DIR}/metadata_test.jsonl}"
}

run_adapter() {
  echo "[Phase B] train and export the spatially aligned Adapter on ${ADAPTER_NUM_PROCESSES} GPUs"
  test -f "${TEACHER_BUNDLE}"
  cd "${PEFM_ROOT}"
  "${PYTHON_BIN}" -m accelerate.commands.launch \
    --multi_gpu \
    --num_processes "${ADAPTER_NUM_PROCESSES}" \
    --mixed_precision "${ADAPTER_MIXED_PRECISION}" \
    --main_process_port "${ADAPTER_MAIN_PROCESS_PORT}" \
    -m pefm.training.train_adapter \
    --config pefm/training/config/adapter.yaml \
    --teacher-bundle "${TEACHER_BUNDLE}" \
    --train-manifest "${CAPTURE_ROOT}/train/manifest.jsonl" \
    --valid-manifest "${CAPTURE_ROOT}/valid/manifest.jsonl" \
    --output "${ADAPTER_OUTPUT}" \
    --dit-dim "${DIT_DIM}" \
    --selected-block "${SELECTED_BLOCK}" \
    --mixed-precision "${ADAPTER_MIXED_PRECISION}" \
    --noise-sampling "${NOISE_SAMPLING}"
}

run_bwm() {
  echo "[Phase C] train Wan DiT with flow matching and 17-step prior KL"
  test -f "${TEACHER_BUNDLE}"
  if [ -z "${ADAPTER_BUNDLE}" ]; then
    test -f "${ADAPTER_SELECTION}"
    ADAPTER_BUNDLE="$(${PYTHON_BIN} -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["best_path"])' "${ADAPTER_SELECTION}")"
    case "${ADAPTER_BUNDLE}" in
      /*) ;;
      *) ADAPTER_BUNDLE="${ADAPTER_OUTPUT}/${ADAPTER_BUNDLE}" ;;
    esac
  fi
  test -f "${ADAPTER_BUNDLE}"
  echo "[Phase C] using Adapter bundle ${ADAPTER_BUNDLE}"
  cd "${BWM_ROOT}"
  "${PYTHON_BIN}" -m accelerate.commands.launch \
    --config_file "${BWM_ROOT}/configs/train/accelerate_fsdp_multi_gpu.yaml" \
    scripts/train.py \
    --config "${BWM_CONFIG}" \
    --model_paths "${MODEL_DIR:-/data1/common_model/modelscope/Wan-AI/Wan2.2-TI2V-5B/}" \
    --dataset_base_path "${DATASET_DIR}" \
    --dataset_metadata_path "${TRAIN_METADATA_PATH:-${DATASET_DIR}/metadata_train.jsonl}" \
    --val_dataset_metadata_path "${VALID_METADATA_PATH:-${DATASET_DIR}/metadata_test.jsonl}" \
    --action_stat_path "${ACTION_STAT_PATH:-${DATASET_DIR}/stat.json}" \
    --trainable dit \
    --stage2_teacher_path "${TEACHER_BUNDLE}" \
    --stage2_adapter_path "${ADAPTER_BUNDLE}" \
    --stage2_loss_weight "${STAGE2_LOSS_WEIGHT}" \
    --stage2_selected_block "${SELECTED_BLOCK}" \
    --stage2_weight_schedule "${STAGE2_WEIGHT_SCHEDULE}" \
    --max_train_steps "${BWM_MAX_TRAIN_STEPS}" \
    --save_steps "${BWM_SAVE_STEPS}" \
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
