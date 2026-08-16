#!/usr/bin/env bash
# 174 서버 labeler_feat 루트에서 실행
# input/pdfs_batch 에 ~1만 페이지 분량 PDF만 두고 convert_only → JSONL export
set -euo pipefail

LABELER_ROOT="${LABELER_ROOT:-/home/vlm_train/labeler_feat}"
CONFIG="${CONFIG:-config/layout_collect.yaml}"
TARGET="${TARGET:-10000}"
EXPORT_DIR="${EXPORT_DIR:-${LABELER_ROOT}/output/layout_export}"
OUTPUT_DIR="${OUTPUT_DIR:-${LABELER_ROOT}/output/layout_batch}"

cd "${LABELER_ROOT}"
source .venv/bin/activate

mkdir -p input/pdfs_batch "${EXPORT_DIR}"

echo "[1/3] labeler convert_only (input=input/pdfs_batch, config=${CONFIG})"
labeler --config "${CONFIG}"

echo "[2/3] export JSONL (target=${TARGET}, ERROR 제외)"
python3 "${LABELER_ROOT}/scripts/export_layout_dataset.py" \
  --output-dir "${OUTPUT_DIR}" \
  --out-jsonl "${EXPORT_DIR}/layout_train.jsonl" \
  --copy-images-dir "${EXPORT_DIR}/images" \
  --target "${TARGET}"

echo "[3/3] done → ${EXPORT_DIR}/layout_train.jsonl"
cat "${EXPORT_DIR}/layout_train.summary.json"
