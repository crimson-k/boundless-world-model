#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-600}"
INFER_SCRIPT="${INFER_SCRIPT:-${REPO_ROOT}/scripts/infer_robotwin_runs.sh}"

WATCH_DIRS=(
  "${REPO_ROOT}/models"
  "${REPO_ROOT}/ckpt/BLM/.cache/huggingface/download"
)

cd "${REPO_ROOT}"

if [[ ! -x "${INFER_SCRIPT}" ]]; then
  echo "Inference script is not executable or missing: ${INFER_SCRIPT}" >&2
  exit 1
fi

echo "Waiting for downloads to complete."
echo "  Check interval: ${CHECK_INTERVAL_SECONDS}s"
echo "  Inference script: ${INFER_SCRIPT}"
printf '  Watch dirs:\n'
printf '    %s\n' "${WATCH_DIRS[@]}"
echo ""

while true; do
  mapfile -t incomplete_files < <(
    find "${WATCH_DIRS[@]}" -type f -name '*.incomplete' -print 2>/dev/null | sort
  )

  if (( ${#incomplete_files[@]} == 0 )); then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] No .incomplete files remain. Starting inference."
    break
  fi

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Still downloading: ${#incomplete_files[@]} .incomplete file(s)."
  for path in "${incomplete_files[@]}"; do
    size="$(du -h "${path}" 2>/dev/null | awk '{print $1}')"
    printf '  %8s  %s\n' "${size:-?}" "${path}"
  done
  echo "Next check in ${CHECK_INTERVAL_SECONDS}s."
  sleep "${CHECK_INTERVAL_SECONDS}"
done

exec "${INFER_SCRIPT}"
