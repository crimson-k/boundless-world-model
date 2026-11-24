#!/bin/bash
set -euo pipefail

# Run one phase or the complete sequence:
#   MODEL_DIR=... DATASET_DIR=... TRAIN_METADATA_PATH=... \
#   VALID_METADATA_PATH=... ACTION_STAT_PATH=... bash scripts/train_stage2.sh all
# Two training stages: evaluator and bwm (teacher is an alias for evaluator).

BWM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PEFM_ROOT="${PEFM_ROOT:-${BWM_ROOT}/../pefm}"
PYTHON_BIN="${PYTHON_BIN:-/home/fangxuebin/.conda/envs/BWM/bin/python}"
STAGE="${1:-all}"

export PYTHONPATH="${BWM_ROOT}:${PEFM_ROOT}:${PYTHONPATH:-}"

DATASET_DIR="${DATASET_DIR:-${BWM_ROOT}/converted_dataset_bwm_6tasks_256x192}"
PEFM_TEACHER_CONFIG="${PEFM_TEACHER_CONFIG:-${BWM_ROOT}/configs/stage2/pefm_teacher.yaml}"
TEACHER_OUTPUT="${TEACHER_OUTPUT:-${BWM_ROOT}/outputs/stage2_256x192/teacher}"
TEACHER_NUM_PROCESSES="${TEACHER_NUM_PROCESSES:-8}"
TEACHER_MAIN_PROCESS_PORT="${TEACHER_MAIN_PROCESS_PORT:-29521}"
REWARD_LABELS="${REWARD_LABELS:-${PEFM_ROOT}/outputs/reward_labels/v23_episodes_20261025}"
TEACHER_BUNDLE="${TEACHER_BUNDLE:-${TEACHER_OUTPUT}/frozen_evaluator.pt}"
VJEPA_CHECKPOINT="${VJEPA_CHECKPOINT:-/data1/fangxuebin/models/vjepa2/vjepa2-ac-vitg.pt}"
BWM_CONFIG="${BWM_CONFIG:-${BWM_ROOT}/configs/train/train_wan22_ti2v_5b_action_adaln.yaml}"
MIN_TIMESTEP_BOUNDARY="${MIN_TIMESTEP_BOUNDARY:-0.0}"
MAX_TIMESTEP_BOUNDARY="${MAX_TIMESTEP_BOUNDARY:-1.0}"

STAGE2_LOSS_WEIGHT="${STAGE2_LOSS_WEIGHT:-0.1}"
STAGE2_WEIGHT_SCHEDULE="${STAGE2_WEIGHT_SCHEDULE:-constant}"
BWM_OUTPUT="${BWM_OUTPUT:-${BWM_ROOT}/outputs/training/stage2_latent_evaluator}"
BWM_MAX_TRAIN_STEPS="${BWM_MAX_TRAIN_STEPS:-8000}"
BWM_LEARNING_RATE="${BWM_LEARNING_RATE:-1e-5}"
BWM_SAVE_STEPS="${BWM_SAVE_STEPS:-2000}"
BWM_GRADIENT_ACCUMULATION_STEPS="${BWM_GRADIENT_ACCUMULATION_STEPS:-8}"
BWM_RESUME_FROM="${BWM_RESUME_FROM:-}"
BWM_ACCELERATE_CONFIG="${BWM_ACCELERATE_CONFIG:-${BWM_ROOT}/configs/stage2/accelerate_fsdp_fast_multi_gpu.yaml}"
BWM_FSDP_SYNC_EACH_BATCH="${BWM_FSDP_SYNC_EACH_BATCH:-0}"
BWM_FUSED_ADAMW="${BWM_FUSED_ADAMW:-1}"
BWM_VAE_LATENT_CACHE_DIR="${BWM_VAE_LATENT_CACHE_DIR-auto}"
BWM_MAIN_PROCESS_PORT="${BWM_MAIN_PROCESS_PORT:-29500}"

export DATASET_DIR TEACHER_OUTPUT VJEPA_CHECKPOINT REWARD_LABELS

run_evaluator() {
  echo "[Stage 1] train the VAE-latent evaluator with semantic and reward supervision"
  cd "${PEFM_ROOT}"
  local launch_args=()
  if [ "${TEACHER_NUM_PROCESSES}" -gt 1 ]; then
    launch_args+=(--multi_gpu)
  fi
  "${PYTHON_BIN}" -m accelerate.commands.launch \
    "${launch_args[@]}" \
    --num_processes "${TEACHER_NUM_PROCESSES}" \
    --num_cpu_threads_per_process 2 \
    --mixed_precision no \
    --main_process_port "${TEACHER_MAIN_PROCESS_PORT}" \
    -m pefm.training.train --config "${PEFM_TEACHER_CONFIG}"
}

run_bwm() {
  echo "[Stage 2] train Wan DiT; recovered clean latents enter the frozen evaluator"
  test -f "${TEACHER_BUNDLE}"
  local resume_args=()
  if [ -n "${BWM_RESUME_FROM}" ]; then
    test -d "${BWM_RESUME_FROM}"
    resume_args=(--resume_from "${BWM_RESUME_FROM}")
  fi
  local performance_args=(--fsdp_sync_each_batch "${BWM_FSDP_SYNC_EACH_BATCH}")
  if [ "${BWM_FUSED_ADAMW}" = "1" ]; then
    performance_args+=(--fused_adamw)
  fi
  if [ "${BWM_VAE_LATENT_CACHE_DIR}" = "auto" ]; then
    local cached_default="${BWM_ROOT}/outputs/cache/stage2_vae_6tasks_81f_256x192"
    BWM_VAE_LATENT_CACHE_DIR=""
    if [ -f "${cached_default}/metadata_train/manifest.json" ] && [ -f "${cached_default}/metadata_test/manifest.json" ]; then
      BWM_VAE_LATENT_CACHE_DIR="${cached_default}"
    fi
  fi
  if [ -n "${BWM_VAE_LATENT_CACHE_DIR}" ]; then
    performance_args+=(--vae_latent_cache_dir "${BWM_VAE_LATENT_CACHE_DIR}")
  fi
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  cd "${BWM_ROOT}"
  "${PYTHON_BIN}" -m accelerate.commands.launch \
    --config_file "${BWM_ACCELERATE_CONFIG}" \
    --main_process_port "${BWM_MAIN_PROCESS_PORT}" \
    scripts/train.py \
    --config "${BWM_CONFIG}" \
    --model_paths "${MODEL_DIR:-/data1/common_model/modelscope/Wan-AI/Wan2.2-TI2V-5B/}" \
    --dataset_base_path "${DATASET_DIR}" \
    --dataset_metadata_path "${TRAIN_METADATA_PATH:-${DATASET_DIR}/metadata_train.jsonl}" \
    --val_dataset_metadata_path "${VALID_METADATA_PATH:-${DATASET_DIR}/metadata_test.jsonl}" \
    --action_stat_path "${ACTION_STAT_PATH:-${DATASET_DIR}/stat.json}" \
    --trainable dit \
    --stage2_teacher_path "${TEACHER_BUNDLE}" \
    --stage2_loss_weight "${STAGE2_LOSS_WEIGHT}" \
    --stage2_weight_schedule "${STAGE2_WEIGHT_SCHEDULE}" \
    --learning_rate "${BWM_LEARNING_RATE}" \
    --max_train_steps "${BWM_MAX_TRAIN_STEPS}" \
    --save_steps "${BWM_SAVE_STEPS}" \
    --batch_size 1 \
    --gradient_accumulation_steps "${BWM_GRADIENT_ACCUMULATION_STEPS}" \
    --output_path "${BWM_OUTPUT}" \
    --min_timestep_boundary "${MIN_TIMESTEP_BOUNDARY}" \
    --max_timestep_boundary "${MAX_TIMESTEP_BOUNDARY}" \
    "${performance_args[@]}" \
    "${resume_args[@]}"
}

case "${STAGE}" in
  evaluator|teacher) run_evaluator ;;
  bwm) run_bwm ;;
  all)
    run_evaluator
    run_bwm
    ;;
  *)
    echo "usage: $0 {evaluator|bwm|all}" >&2
    exit 2
    ;;
esac
