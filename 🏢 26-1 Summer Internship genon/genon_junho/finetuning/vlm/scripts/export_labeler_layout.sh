#!/usr/bin/env bash
# JSONL → labeler_ready_layout export (b200 / jhshin _train_data)
#
# 사용:
#   cd /NHNHOME/WORKSPACE/0426030039_A/jhshin/tsr_test/train/vlm
#   bash scripts/export_labeler_layout.sh
#
# pilot 100장:
#   MAX_SAMPLES=100 bash scripts/export_labeler_layout.sh
#
# 전량 export:
#   MAX_SAMPLES= bash scripts/export_labeler_layout.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TRAIN_DATA_ROOT="${TRAIN_DATA_ROOT:-${VLM_ROOT}/_train_data}"
JSONL="${JSONL:-${TRAIN_DATA_ROOT}/layout_src_9984/labeler_converter_layout_source_9984.jsonl}"
IMAGE_ROOT="${IMAGE_ROOT:-${TRAIN_DATA_ROOT}/chandra_table_layout_divhtml_16886}"
OUT_DIR="${OUT_DIR:-${TRAIN_DATA_ROOT}/labeler_ready_layout}"
# MAX_SAMPLES 미설정 시 전량 export. pilot: MAX_SAMPLES=100
MAX_SAMPLES="${MAX_SAMPLES-}"

PYTHON="${PYTHON:-python3}"
if [[ -x "${VLM_ROOT}/.venv/bin/python" ]]; then
  PYTHON="${VLM_ROOT}/.venv/bin/python"
elif [[ -x "/NHNHOME/WORKSPACE/0426030039_A/shkim/tsr_test/train/vlm/.venv/bin/python" ]]; then
  PYTHON="/NHNHOME/WORKSPACE/0426030039_A/shkim/tsr_test/train/vlm/.venv/bin/python"
fi

ARGS=(
  -m data.export_jsonl_to_labeler_layout
  --train-data-root "${TRAIN_DATA_ROOT}"
  --jsonl "${JSONL}"
  --image-root "${IMAGE_ROOT}"
  --image-root "${TRAIN_DATA_ROOT}/layout_src_9984"
  --out-dir "${OUT_DIR}"
)

if [[ -n "${MAX_SAMPLES}" ]]; then
  ARGS+=(--max-samples "${MAX_SAMPLES}")
fi

if [[ "${OVERWRITE:-0}" == "1" ]]; then
  ARGS+=(--overwrite)
fi

echo "TRAIN_DATA_ROOT=${TRAIN_DATA_ROOT}"
echo "JSONL=${JSONL}"
echo "OUT_DIR=${OUT_DIR}"
echo "MAX_SAMPLES=${MAX_SAMPLES:-ALL}"

cd "${VLM_ROOT}"
"${PYTHON}" "${ARGS[@]}"
