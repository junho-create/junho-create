#!/usr/bin/env bash
set -euo pipefail

# dots.ocr(dots-mocr) on 6000_test held-out 200 samples
# Requires network access to dots endpoint (default: 192.168.75.174:26001)

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"

TEST_DATA="${TEST_DATA:-/NHNHOME/WORKSPACE/0426030039_A/shkim/tsr_test/train/vlm/_train_data/20260317_4_6000/data_split/test.jsonl}"
OUT_DIR="${OUT_DIR:-eval_results/dots_ocr_6000test/dots_ocr_noreason_nothink_b8_6000test200}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
API_URL="${API_URL:-http://192.168.75.174:26001/v1/chat/completions}"
API_MODEL="${API_MODEL:-dots-mocr}"
CONCURRENCY="${CONCURRENCY:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16000}"
TIMEOUT="${TIMEOUT:-360}"

"${PYTHON_BIN}" -m eval.evaluate_dots_ocr \
  --test_data "${TEST_DATA}" \
  --output_dir "${OUT_DIR}" \
  --max_samples 200 \
  --api_url "${API_URL}" \
  --api_model "${API_MODEL}" \
  --concurrency "${CONCURRENCY}" \
  --timeout "${TIMEOUT}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --nested_teds_mode split_mean \
  --normalize_empty_cells
