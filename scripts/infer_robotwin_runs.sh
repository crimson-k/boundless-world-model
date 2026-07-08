#!/bin/bash
set -euo pipefail

# Usage:
#   PREP_ONLY=1 EPISODES=0 scripts/infer_robotwin_runs.sh
#   scripts/infer_robotwin_runs.sh
#   scripts/infer_robotwin_runs.sh /data/fangxuebin/RoboTwin/data/dummy_task/multiple_interventions
#   ROBOTWIN_RUN_ROOTS="/data/fangxuebin/RoboTwin/data/dummy_task/multiple_interventions" scripts/infer_robotwin_runs.sh
#   ROBOTWIN_MAX_SAMPLES=1 scripts/infer_robotwin_runs.sh
#   GPU_IDS=0,1,2 ROBOTWIN_CHUNK_SIZE=13 scripts/infer_robotwin_runs.sh
#
# Override local paths:
#   MODEL_PATHS=/path/to/Wan2.2-TI2V-5B \
#   CKPT_PATH=/path/to/checkpoint.safetensors \
#   ACTION_STAT_PATH=/path/to/stat.json \
#   scripts/infer_robotwin_runs.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOCAL_CONFIG="${LOCAL_CONFIG:-${SCRIPT_DIR}/local.sh}"

if [[ -f "${LOCAL_CONFIG}" ]]; then
  # shellcheck source=/dev/null
  source "${LOCAL_CONFIG}"
fi

cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CONFIG_PATH="${CONFIG_PATH:-configs/infer/infer.yaml}"
MODEL_PATHS="${MODEL_PATHS:-models/Wan2.2-TI2V-5B}"
CKPT_PATH="${CKPT_PATH:-ckpt/BLM/step-12000.safetensors}"
ACTION_STAT_PATH="${ACTION_STAT_PATH:-demo/stat.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/inference_robotwin_runs}"
WORK_ROOT="${WORK_ROOT:-outputs/inference_robotwin_runs_metadata}"
ROBOTWIN_MAX_SAMPLES="${ROBOTWIN_MAX_SAMPLES:-0}"
ROBOTWIN_CHUNK_SIZE="${ROBOTWIN_CHUNK_SIZE:-13}"
GPU_IDS="${GPU_IDS:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
ACTION_TYPE="${ACTION_TYPE:-eef_abs}"
ROTATION_QUAT_ORDER="${ROTATION_QUAT_ORDER:-xyzw}"
EPISODES="${EPISODES:-}"
DRY_RUN="${DRY_RUN:-0}"
PREP_ONLY="${PREP_ONLY:-0}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/logs}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
INFER_ORIGINAL="${INFER_ORIGINAL:-1}"
INFER_CROPS="${INFER_CROPS:-1}"
CROP_ROOT_NAME="${CROP_ROOT_NAME:-crop_intervention}"
ROBOTWIN_RUN_ROOTS="${ROBOTWIN_RUN_ROOTS:-}"

IFS=', ' read -r -a GPU_ID_LIST <<< "${GPU_IDS}"
if (( ${#GPU_ID_LIST[@]} == 0 )); then
  echo "GPU_IDS must contain at least one GPU id." >&2
  exit 1
fi
if (( ROBOTWIN_CHUNK_SIZE <= 0 )); then
  echo "ROBOTWIN_CHUNK_SIZE must be positive." >&2
  exit 1
fi

RUN_DIRS=(
  "/data/fangxuebin/RoboTwin/data/adjust_bottle/multiple_interventions/run_0001"
  "/data/fangxuebin/RoboTwin/data/beat_block_hammer/multiple_interventions/run_0001"
  "/data/fangxuebin/RoboTwin/data/click_alarmclock/multiple_interventions/run_0001"
  "/data/fangxuebin/RoboTwin/data/place_a2b_left/multiple_interventions/run_0001"
  "/data/fangxuebin/RoboTwin/data/stamp_seal/multiple_interventions/run_0001"
)

TASK_NAMES=(
  "adjust_bottle"
  "beat_block_hammer"
  "click_alarmclock"
  "place_a2b_left"
  "stamp_seal"
)

RUN_NAMES=(
  "run_0001"
  "run_0001"
  "run_0001"
  "run_0001"
  "run_0001" 
)

add_discovered_run_dir() {
  local run_dir="$1"
  local run_name
  local task_config_dir
  local task_dir
  local task_name

  run_dir="$(cd "${run_dir}" && pwd)"
  run_name="$(basename "${run_dir}")"
  task_config_dir="$(dirname "${run_dir}")"
  task_dir="$(dirname "${task_config_dir}")"
  task_name="$(basename "${task_dir}")"

  RUN_DIRS+=("${run_dir}")
  TASK_NAMES+=("${task_name}")
  RUN_NAMES+=("${run_name}")
}

discover_run_root() {
  local root="$1"
  local run_dir

  if [[ ! -d "${root}" ]]; then
    echo "[discover skip] Missing path: ${root}" >&2
    return
  fi

  root="$(cd "${root}" && pwd)"
  if [[ "$(basename "${root}")" == run_* ]]; then
    add_discovered_run_dir "${root}"
    return
  fi

  while IFS= read -r run_dir; do
    add_discovered_run_dir "${run_dir}"
  done < <(find "${root}" -mindepth 1 -maxdepth 1 -type d -name 'run_*' | sort)
}

DISCOVERY_ROOTS=()
if (( $# > 0 )); then
  DISCOVERY_ROOTS+=("$@")
fi
if [[ -n "${ROBOTWIN_RUN_ROOTS}" ]]; then
  # Whitespace-separated list of parent dirs or run dirs.
  # Use command-line args if paths ever contain spaces.
  read -r -a env_roots <<< "${ROBOTWIN_RUN_ROOTS}"
  DISCOVERY_ROOTS+=("${env_roots[@]}")
fi

if (( ${#DISCOVERY_ROOTS[@]} > 0 )); then
  RUN_DIRS=()
  TASK_NAMES=()
  RUN_NAMES=()
  for root in "${DISCOVERY_ROOTS[@]}"; do
    discover_run_root "${root}"
  done
fi

if (( ${#RUN_DIRS[@]} == 0 )); then
  echo "No RUN_DIRS configured or discovered." >&2
  exit 1
fi
if (( ${#RUN_DIRS[@]} != ${#TASK_NAMES[@]} || ${#RUN_DIRS[@]} != ${#RUN_NAMES[@]} )); then
  echo "RUN_DIRS, TASK_NAMES, and RUN_NAMES must have the same length." >&2
  exit 1
fi

prepare_run_metadata() {
  local run_dir="$1"
  local task_name="$2"
  local metadata_branch="$3"
  local prepared_dir="${WORK_ROOT}/${task_name}/${metadata_branch}"

  mkdir -p "${prepared_dir}"

  ROTATION_QUAT_ORDER="${ROTATION_QUAT_ORDER}" EPISODES="${EPISODES}" "${PYTHON_BIN}" - "${run_dir}" "${task_name}" "${prepared_dir}" <<'PY'
import json
import os
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

run_dir = Path(sys.argv[1])
task_name = sys.argv[2]
prepared_dir = Path(sys.argv[3])
action_dir = prepared_dir / "actions"
metadata_path = prepared_dir / "metadata.jsonl"
action_dir.mkdir(parents=True, exist_ok=True)

quat_order = os.environ.get("ROTATION_QUAT_ORDER", "xyzw").lower()
if quat_order not in {"xyzw", "wxyz"}:
    raise ValueError(f"ROTATION_QUAT_ORDER must be xyzw or wxyz, got {quat_order!r}")

episodes_env = os.environ.get("EPISODES", "").strip()
episode_filter = None
if episodes_env:
    episode_filter = {
        int(value)
        for value in re.split(r"[\s,]+", episodes_env)
        if value.strip()
    }


def episode_index(path: Path) -> int:
    match = re.fullmatch(r"episode(\d+)", path.stem)
    if match is None:
        raise ValueError(f"Unexpected episode filename: {path.name}")
    return int(match.group(1))


def quat_to_euler(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    if quat_order == "wxyz":
        quat = quat[:, [1, 2, 3, 0]]
    return Rotation.from_quat(quat).as_euler("xyz", degrees=False).astype(np.float32)


def eef_pose_14(h5: h5py.File) -> np.ndarray:
    left_pose = np.asarray(h5["endpose/left_endpose"], dtype=np.float32)
    right_pose = np.asarray(h5["endpose/right_endpose"], dtype=np.float32)
    left_gripper = np.asarray(h5["endpose/left_gripper"], dtype=np.float32).reshape(-1, 1)
    right_gripper = np.asarray(h5["endpose/right_gripper"], dtype=np.float32).reshape(-1, 1)
    left = np.concatenate(
        [left_pose[:, :3], quat_to_euler(left_pose[:, 3:7]), left_gripper],
        axis=1,
    )
    right = np.concatenate(
        [right_pose[:, :3], quat_to_euler(right_pose[:, 3:7]), right_gripper],
        axis=1,
    )
    return np.concatenate([left, right], axis=1).astype(np.float32)


def joint_action_14(h5: h5py.File, fallback_length: int) -> np.ndarray:
    if "joint_action/vector" in h5:
        return np.asarray(h5["joint_action/vector"], dtype=np.float32)
    return np.zeros((fallback_length, 14), dtype=np.float32)


video_paths = {episode_index(path): path for path in (run_dir / "video").glob("episode*.mp4")}
hdf5_paths = {episode_index(path): path for path in (run_dir / "data").glob("episode*.hdf5")}
episode_ids = sorted(set(video_paths) & set(hdf5_paths))
if episode_filter is not None:
    episode_ids = [episode_id for episode_id in episode_ids if episode_id in episode_filter]

if not episode_ids:
    raise RuntimeError(f"No matching video/episode*.mp4 and data/episode*.hdf5 files found in {run_dir}")

rows = []
skipped_episodes = []
for episode_id in episode_ids:
    hdf5_path = hdf5_paths[episode_id]
    try:
        with h5py.File(hdf5_path, "r") as h5:
            state_pose = eef_pose_14(h5)
            action = joint_action_14(h5, state_pose.shape[0])
            length = int(min(state_pose.shape[0], action.shape[0]))
            state_pose = state_pose[:length]
            action = action[:length]
    except Exception as exc:
        skipped_episodes.append((episode_id, hdf5_path, repr(exc)))
        print(
            f"[metadata warning] skip unreadable episode{episode_id}: {hdf5_path} ({exc!r})",
            file=sys.stderr,
        )
        continue

    parquet_path = action_dir / f"episode{episode_id}.parquet"
    table = pa.Table.from_pydict(
        {
            "observation.state": [row.tolist() for row in state_pose],
            "action": [row.tolist() for row in action],
        }
    )
    pq.write_table(table, parquet_path)

    rows.append(
        {
            "episode_index": episode_id,
            "source_episode_index": episode_id,
            "length": length,
            "start_frame": 0,
            "end_frame": length - 1,
            "video": str(video_paths[episode_id]),
            "action": str(parquet_path.resolve()),
            "task": task_name,
            "prompt": "",
        }
    )

if not rows:
    print(
        f"[metadata warning] no readable episodes found in {run_dir}; "
        f"skipped {len(skipped_episodes)} episode(s).",
        file=sys.stderr,
    )

with metadata_path.open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(metadata_path)
PY
}

prepare_todo_metadata() {
  local metadata_path="$1"
  local output_path="$2"
  local task_name="$3"
  local metadata_branch="$4"
  local todo_path="${WORK_ROOT}/${task_name}/${metadata_branch}/metadata_todo.jsonl"

  SKIP_EXISTING="${SKIP_EXISTING}" "${PYTHON_BIN}" - "${metadata_path}" "${output_path}" "${todo_path}" <<'PY'
import json
import os
import sys
from pathlib import Path

metadata_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
todo_path = Path(sys.argv[3])
skip_existing = os.environ.get("SKIP_EXISTING", "1") != "0"

todo_rows = []
skipped = 0
with metadata_path.open("r", encoding="utf-8") as f:
    for line in f:
        text = line.strip()
        if not text:
            continue
        row = json.loads(text)
        episode_output = output_path / f"episode{int(row['episode_index'])}.mp4"
        if skip_existing and episode_output.exists() and episode_output.stat().st_size > 0:
            skipped += 1
            continue
        todo_rows.append(row)

todo_path.parent.mkdir(parents=True, exist_ok=True)
with todo_path.open("w", encoding="utf-8") as f:
    for row in todo_rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"{todo_path}\t{len(todo_rows)}\t{skipped}")
PY
}

prepare_chunk_metadata() {
  local metadata_path="$1"
  local task_name="$2"
  local metadata_branch="$3"
  local chunk_start="$4"
  local chunk_count="$5"
  local chunk_path="${WORK_ROOT}/${task_name}/${metadata_branch}/chunks/todo_start${chunk_start}_count${chunk_count}.jsonl"

  "${PYTHON_BIN}" - "${metadata_path}" "${chunk_path}" "${chunk_start}" "${chunk_count}" <<'PY'
import json
import sys
from pathlib import Path

metadata_path = Path(sys.argv[1])
chunk_path = Path(sys.argv[2])
chunk_start = int(sys.argv[3])
chunk_count = int(sys.argv[4])

rows = []
with metadata_path.open("r", encoding="utf-8") as f:
    for line in f:
        text = line.strip()
        if text:
            rows.append(json.loads(text))

chunk_rows = rows[chunk_start : chunk_start + chunk_count]
chunk_path.parent.mkdir(parents=True, exist_ok=True)
with chunk_path.open("w", encoding="utf-8") as f:
    for row in chunk_rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

episode_ids = [str(int(row["episode_index"])) for row in chunk_rows]
if episode_ids:
    episode_label = f"{episode_ids[0]}-{episode_ids[-1]}" if len(episode_ids) > 1 else episode_ids[0]
else:
    episode_label = "none"

print(f"{chunk_path}\t{len(chunk_rows)}\t{episode_label}")
PY
}

echo "RoboTwin inference batch"
echo "  Config: ${CONFIG_PATH}"
echo "  Model paths: ${MODEL_PATHS}"
echo "  Checkpoint: ${CKPT_PATH}"
echo "  Action stats: ${ACTION_STAT_PATH}"
echo "  Output root: ${OUTPUT_ROOT}"
echo "  Work root: ${WORK_ROOT}"
echo "  Action type: ${ACTION_TYPE}"
echo "  Max samples per run: ${ROBOTWIN_MAX_SAMPLES} (0 means all)"
echo "  Chunk size: ${ROBOTWIN_CHUNK_SIZE}"
echo "  GPU ids: ${GPU_ID_LIST[*]}"
echo "  Skip existing outputs: ${SKIP_EXISTING}"
echo "  Require CUDA: ${REQUIRE_CUDA}"
echo "  Infer original runs: ${INFER_ORIGINAL}"
echo "  Infer cropped ranges: ${INFER_CROPS}"
echo "  Crop root name: ${CROP_ROOT_NAME}"
echo "  Runs configured: ${#RUN_DIRS[@]}"
for index in "${!RUN_DIRS[@]}"; do
  echo "    - ${TASK_NAMES[$index]}/${RUN_NAMES[$index]}: ${RUN_DIRS[$index]}"
done
echo ""

mkdir -p "${LOG_ROOT}"

active_jobs=0
next_gpu_index=0
failure_count=0

wait_for_one_job() {
  local status=0
  set +e
  wait -n
  status=$?
  set -e
  active_jobs=$((active_jobs - 1))
  if (( status != 0 )); then
    failure_count=$((failure_count + 1))
    echo "[error] One inference job failed with status ${status}." >&2
  fi
}

launch_job() {
  local task_name="$1"
  local run_label="$2"
  local run_dir="$3"
  local metadata_path="$4"
  local output_path="$5"
  local max_samples="$6"
  local episode_label="$7"
  local gpu_id="${GPU_ID_LIST[$next_gpu_index]}"
  local safe_label="${run_label//\//_}"
  local log_path="${LOG_ROOT}/${task_name}_${safe_label}_episodes${episode_label}_count${max_samples}_gpu${gpu_id}.log"

  next_gpu_index=$(((next_gpu_index + 1) % ${#GPU_ID_LIST[@]}))

  cmd=(
    "${PYTHON_BIN}" scripts/infer.py
    --config "${CONFIG_PATH}"
    --model_paths "${MODEL_PATHS}"
    --ckpt_path "${CKPT_PATH}"
    --dataset_base_path "${run_dir}"
    --dataset_metadata_path "${metadata_path}"
    --action_stat_path "${ACTION_STAT_PATH}"
    --action_type "${ACTION_TYPE}"
    --output_path "${output_path}"
    --start_index 0
    --max_samples "${max_samples}"
  )

  echo "[job] ${task_name}/${run_label} episodes=${episode_label} count=${max_samples} gpu=${gpu_id}"
  echo "      log=${log_path}"
  printf '      command: CUDA_VISIBLE_DEVICES=%q' "${gpu_id}"
  printf ' %q' "${cmd[@]}"
  printf '\n'

  if [[ "${DRY_RUN}" == "1" ]]; then
    return
  fi

  (
    export CUDA_VISIBLE_DEVICES="${gpu_id}"
    export PYTHONUNBUFFERED=1
    if [[ "${REQUIRE_CUDA}" == "1" ]]; then
      "${PYTHON_BIN}" - <<'PY'
import sys
import torch

if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    sys.exit("CUDA preflight failed: torch cannot see a CUDA device in this process.")

print(f"CUDA preflight ok: {torch.cuda.get_device_name(0)}")
PY
    fi
    "${cmd[@]}"
  ) >"${log_path}" 2>&1 &

  active_jobs=$((active_jobs + 1))
  if (( active_jobs >= ${#GPU_ID_LIST[@]} )); then
    wait_for_one_job
  fi
}

schedule_dataset() {
  local dataset_dir="$1"
  local task_name="$2"
  local run_label="$3"
  local metadata_branch="$4"
  local output_path="$5"

  if [[ ! -d "${dataset_dir}" ]]; then
    echo "[skip] Missing dataset dir: ${dataset_dir}" >&2
    return
  fi

  metadata_path="$(prepare_run_metadata "${dataset_dir}" "${task_name}" "${metadata_branch}")"
  todo_info="$(prepare_todo_metadata "${metadata_path}" "${output_path}" "${task_name}" "${metadata_branch}")"
  IFS=$'\t' read -r todo_metadata_path todo_rows skipped_rows <<< "${todo_info}"
  total_rows="$(wc -l < "${metadata_path}")"

  if (( total_rows == 0 )); then
    echo "[skip] ${task_name}/${run_label}: no readable source episodes"
    return
  fi

  if (( todo_rows == 0 )); then
    echo "[skip] ${task_name}/${run_label}: all ${total_rows} episode output(s) already exist"
    return
  fi

  remaining_rows="${todo_rows}"
  if (( ROBOTWIN_MAX_SAMPLES > 0 && ROBOTWIN_MAX_SAMPLES < remaining_rows )); then
    remaining_rows="${ROBOTWIN_MAX_SAMPLES}"
  fi

  echo "[run] ${task_name}/${run_label}"
  echo "  Source: ${dataset_dir}"
  echo "  Metadata: ${metadata_path}"
  echo "  Todo metadata: ${todo_metadata_path}"
  echo "  Output: ${output_path}"
  echo "  Existing outputs skipped: ${skipped_rows}/${total_rows}"
  echo "  Samples scheduled: ${remaining_rows}/${todo_rows} todo rows"

  if [[ "${PREP_ONLY}" == "1" ]]; then
    return
  fi

  chunk_start=0
  scheduled=0
  while (( scheduled < remaining_rows )); do
    chunk_count="${ROBOTWIN_CHUNK_SIZE}"
    if (( scheduled + chunk_count > remaining_rows )); then
      chunk_count=$((remaining_rows - scheduled))
    fi

    chunk_info="$(prepare_chunk_metadata "${todo_metadata_path}" "${task_name}" "${metadata_branch}" "${chunk_start}" "${chunk_count}")"
    IFS=$'\t' read -r chunk_metadata_path actual_chunk_count episode_label <<< "${chunk_info}"
    if (( actual_chunk_count == 0 )); then
      break
    fi

    launch_job \
      "${task_name}" \
      "${run_label}" \
      "${dataset_dir}" \
      "${chunk_metadata_path}" \
      "${output_path}" \
      "${actual_chunk_count}" \
      "${episode_label}"

    chunk_start=$((chunk_start + chunk_count))
    scheduled=$((scheduled + actual_chunk_count))
  done
}

for index in "${!RUN_DIRS[@]}"; do
  run_dir="${RUN_DIRS[$index]}"
  task_name="${TASK_NAMES[$index]}"
  run_name="${RUN_NAMES[$index]}"

  if [[ ! -d "${run_dir}" ]]; then
    echo "[skip] Missing run dir: ${run_dir}" >&2
    continue
  fi

  if [[ "${INFER_ORIGINAL}" == "1" ]]; then
    schedule_dataset \
      "${run_dir}" \
      "${task_name}" \
      "${run_name}" \
      "${run_name}" \
      "${OUTPUT_ROOT}/${task_name}/${run_name}"
  fi

  if [[ "${INFER_CROPS}" == "1" ]]; then
    crop_root="${run_dir}/${CROP_ROOT_NAME}"
    if [[ ! -d "${crop_root}" ]]; then
      echo "[skip] ${task_name}/${run_name}: no ${CROP_ROOT_NAME} directory"
      continue
    fi

    while IFS= read -r crop_dir; do
      range_name="$(basename "${crop_dir}")"
      if [[ ! -d "${crop_dir}/data" || ! -d "${crop_dir}/video" ]]; then
        echo "[skip] ${task_name}/${run_name}/${CROP_ROOT_NAME}/${range_name}: missing data/ or video/"
        continue
      fi

      schedule_dataset \
        "${crop_dir}" \
        "${task_name}" \
        "${run_name}/${CROP_ROOT_NAME}/${range_name}" \
        "${run_name}/${CROP_ROOT_NAME}/${range_name}" \
        "${OUTPUT_ROOT}/${task_name}/${run_name}/${CROP_ROOT_NAME}/${range_name}"
    done < <(find "${crop_root}" -mindepth 1 -maxdepth 1 -type d | sort)
  fi
done

while (( active_jobs > 0 )); do
  wait_for_one_job
done

if (( failure_count > 0 )); then
  echo "[failed] ${failure_count} inference job(s) failed. Check logs under ${LOG_ROOT}." >&2
  exit 1
fi

echo "[done] All scheduled RoboTwin inference jobs completed."
