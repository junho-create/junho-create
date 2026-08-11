#!/bin/bash
# =============================================================================
# unified_base(JSON) 모델을 동일 held-out test set 으로
# "OCR 넣을 때 vs 안 넣을 때" 두 번 평가해 비교한다(동일 GT → 직접 비교 가능).
#
# - WITH OCR : _train_data/chandra_table_layout_divhtml_16886/data/test.jsonl (ocr_info 보유)
# - NO   OCR : 위에서 ocr_info 제거 + prompt_style=chandra_no_ocr 로 생성한 짝
#
# 출력: 각 모드별 metrics_unified.json / predictions_unified.jsonl
#       (predictions 에는 bbox/category/text 가 pred_elements 로 보존된다)
#
# 사용(로컬 vLLM, LoRA 어댑터):
#   BACKEND=vllm \
#   MODEL=output/unified_base_ocr_20260616/student_sft/final \
#   BASE_MODEL=/NHNHOME/WORKSPACE/0426030039_A/shkim/models/Qwen3.5-9B \
#   bash scripts/eval_unified_ocr_compare.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# 비교 manifest 원본(통합 JSON test split, ocr_info 보유)
SOURCE_DATA="${SOURCE_DATA:-_train_data/chandra_table_layout_divhtml_16886/data/test.jsonl}"
COMPARE_DIR="${COMPARE_DIR:-eval_data/ocr_compare}"
WITH_OCR_DATA="${WITH_OCR_DATA:-${COMPARE_DIR}/test_with_ocr.jsonl}"
NO_OCR_DATA="${NO_OCR_DATA:-${COMPARE_DIR}/test_no_ocr.jsonl}"
OUT_ROOT="${OUT_ROOT:-eval_results/ocr_compare_$(date +%Y%m%d_%H%M%S)}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

# 비교 manifest 짝이 없으면 자동 생성(prompt_style 을 OCR 유무에 맞게 정규화).
if [[ ! -f "${WITH_OCR_DATA}" || ! -f "${NO_OCR_DATA}" ]]; then
  echo "[prep] building OCR/NO-OCR manifests under ${COMPARE_DIR}"
  python data/make_ocr_compare_manifests.py --in "${SOURCE_DATA}" --out-dir "${COMPARE_DIR}"
fi

echo "=============================================="
echo "  OCR vs NO-OCR comparison (same GT)"
echo "  with_ocr: ${WITH_OCR_DATA}"
echo "  no_ocr:   ${NO_OCR_DATA}"
echo "  out_root: ${OUT_ROOT}"
echo "=============================================="

echo "[1/2] WITH OCR ..."
TEST_DATA="${WITH_OCR_DATA}" OUTPUT_DIR="${OUT_ROOT}/with_ocr" MAX_SAMPLES="${MAX_SAMPLES}" \
  bash scripts/eval_unified.sh

echo "[2/2] NO OCR ..."
TEST_DATA="${NO_OCR_DATA}" OUTPUT_DIR="${OUT_ROOT}/no_ocr" MAX_SAMPLES="${MAX_SAMPLES}" \
  bash scripts/eval_unified.sh

echo "=============================================="
echo "  DONE. compare:"
echo "   with_ocr: ${OUT_ROOT}/with_ocr/metrics_unified.json"
echo "   no_ocr:   ${OUT_ROOT}/no_ocr/metrics_unified.json"
echo "=============================================="
