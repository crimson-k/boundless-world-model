#!/bin/bash

# Copy this file to scripts/train_local.sh and fill in machine-local paths.
# scripts/train_local.sh is ignored by git.

# Leave CUDA_VISIBLE_DEVICES unset to expose all GPUs on this node.
# export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

PYTHON_BIN="/path/to/python"
CONFIG_PATH="configs/train/train_wan22_ti2v_5b_action_adaln.yaml"
ACCELERATE_CONFIG="configs/train/accelerate_multi_gpu.yaml"
MODEL_DIR="/path/to/Wan2.2-TI2V-5B"
DATASET_DIR="/path/to/RoboTwin2.0_lerobot"
DATASET_METADATA_PATH="${DATASET_DIR}/episodes_train_cam_high_720p_len57_adjust_bottle.jsonl"
ACTION_STAT_PATH="${DATASET_DIR}/stat.json"
OUTPUT_PATH="/path/to/output_dir"
