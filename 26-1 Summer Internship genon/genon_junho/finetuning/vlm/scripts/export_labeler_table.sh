#!/usr/bin/env bash
# JSONL(표) → labeler_ready_table export (b200)
#
#   cd .../train/vlm && bash scripts/export_labeler_table.sh
#   MAX_SAMPLES=100 bash scripts/export_labeler_table.sh   # pilot

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TRAIN_DATA_ROOT="${TRAIN_DATA_ROOT:-${VLM_ROOT}/_train_data}"
JSONL_DIR="${JSONL_DIR:-${TRAIN_DATA_ROOT}/table_src_6902/data}"
IMAGE_ROOT="${IMAGE_ROOT:-${TRAIN_DATA_ROOT}/table_src_6902}"
OUT_DIR="${OUT_DIR:-${TRAIN_DATA_ROOT}/labeler_ready_table}"
MAX_SAMPLES="${MAX_SAMPLES-}"

PYTHON="${PYTHON:-python3}"
if [[ -x "${VLM_ROOT}/.venv/bin/python" ]]; then
  PYTHON="${VLM_ROOT}/.venv/bin/python"
elif [[ -x "/NHNHOME/WORKSPACE/0426030039_A/shkim/tsr_test/train/vlm/.venv/bin/python" ]]; then
  PYTHON="/NHNHOME/WORKSPACE/0426030039_A/shkim/tsr_test/train/vlm/.venv/bin/python"
fi

ARGS=(
  -m data.export_jsonl_to_labeler_tsr
  --train-data-root "${TRAIN_DATA_ROOT}"
  --jsonl-dir "${JSONL_DIR}"
  --image-root "${IMAGE_ROOT}"
  --image-root "${TRAIN_DATA_ROOT}/chandra_table_layout_divhtml_16886"
  --out-dir "${OUT_DIR}"
)

if [[ -n "${MAX_SAMPLES}" ]]; then
  ARGS+=(--max-samples "${MAX_SAMPLES}")
fi

if [[ "${OVERWRITE:-0}" == "1" ]]; then
  ARGS+=(--overwrite)
fi

echo "TRAIN_DATA_ROOT=${TRAIN_DATA_ROOT}"
echo "JSONL_DIR=${JSONL_DIR}"
echo "OUT_DIR=${OUT_DIR}"
echo "MAX_SAMPLES=${MAX_SAMPLES:-ALL}"

cd "${VLM_ROOT}"
"${PYTHON}" "${ARGS[@]}"
