#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Filter predictions rows where TEDS-S is not 1.0.

Usage:
  bash scripts/filter_teds_s_not_one.sh <run_dir> [output_jsonl] [report_html] [chunk_size]

Examples:
  bash scripts/filter_teds_s_not_one.sh \
    eval_results/20260321/e15_without_ocr/e15final_noreason_nothink_b16_api_on_e15test200_retry5_b16_m10000_20260321_105947

  bash scripts/filter_teds_s_not_one.sh \
    eval_results/my_run \
    eval_results/my_run/predictions_teds_s_ne_1.jsonl \
    eval_results/my_run/report_teds_s_ne_1.html \
    100
USAGE
}

if [[ $# -lt 1 || $# -gt 4 ]]; then
  usage
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

RUN_DIR="$1"
INPUT_JSONL="${RUN_DIR}/predictions.jsonl"
OUTPUT_JSONL="${2:-${RUN_DIR}/predictions_teds_s_ne_1.jsonl}"
REPORT_HTML="${3:-${RUN_DIR}/report_teds_s_ne_1.html}"
CHUNK_SIZE="${4:-100}"

if [[ ! -f "${INPUT_JSONL}" ]]; then
  echo "[ERROR] Input not found: ${INPUT_JSONL}" >&2
  exit 1
fi
if ! [[ "${CHUNK_SIZE}" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] chunk_size must be a non-negative integer: ${CHUNK_SIZE}" >&2
  exit 1
fi

python3 -m data.filter_predictions_by_teds_s \
  --input "${INPUT_JSONL}" \
  --output "${OUTPUT_JSONL}" \
  --report_html "${REPORT_HTML}" \
  --metric_field teds_structure \
  --op ne \
  --target 1.0 \
  --atol 1e-12 \
  --chunk_size "${CHUNK_SIZE}"
