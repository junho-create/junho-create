#!/bin/bash
# =============================================================================
# unified_base(JSON 레이아웃 디텍션) 학습 종료를 기다린 뒤 자동 추론한다.
#
# 추론 1) genos500 (NO OCR 전용 — OCR 소스 없음):
#    - 입력: 이미지만. manifest = eval_data/500_genos_val_set/inference_manifest_noocr.jsonl
#    - 출력: [{bbox,category,text}] JSON (라벨러 UI 확인용, bbox 포함)
# 추론 2) held-out test set OCR vs NO-OCR 비교(동일 GT):
#    - scripts/eval_unified_ocr_compare.sh 호출
#
# 사용:
#   tmux new -s tsr_eval
#   source <venv>/bin/activate
#   bash scripts/wait_train_and_infer_unified_base.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

CONFIG_MARKER="${CONFIG_MARKER:-exp_unified_base_smoke.yaml}"
TRAIN_OUTPUT="${TRAIN_OUTPUT:-output/unified_base_ocr_20260616/student_sft}"
BASE_MODEL="${BASE_MODEL:-/NHNHOME/WORKSPACE/0426030039_A/shkim/models/Qwen3.5-9B}"

GENOS_ROOT="${GENOS_ROOT:-eval_data/500_genos_val_set}"
GENOS_NOOCR_MANIFEST="${GENOS_NOOCR_MANIFEST:-${GENOS_ROOT}/inference_manifest_noocr.jsonl}"
GENOS_OUT="${GENOS_OUT:-eval_results/genos500_unified_base_$(date +%Y%m%d_%H%M%S)}"

# OCR/NO-OCR 비교 원본(통합 JSON test split). 비교 manifest 한 쌍은 compare 스크립트가 자동 생성.
COMPARE_SOURCE_DATA="${COMPARE_SOURCE_DATA:-_train_data/chandra_table_layout_divhtml_16886/data/test.jsonl}"
COMPARE_OUT_ROOT="${COMPARE_OUT_ROOT:-eval_results/ocr_compare_unified_base_$(date +%Y%m%d_%H%M%S)}"

POLL_SEC="${POLL_SEC:-60}"
NO_THINKING="${NO_THINKING:-true}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
# 학습 종료 후 GPU 전부 유휴. TP=1 이므로 1장이면 충분.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

LOG="logs/wait_train_and_infer_unified_base_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs
exec > >(tee -a "${LOG}") 2>&1

echo "=============================================="
echo "  Wait unified_base train -> auto infer"
echo "  config_marker:${CONFIG_MARKER}"
echo "  train_output: ${TRAIN_OUTPUT}"
echo "  base_model:   ${BASE_MODEL}"
echo "  genos noocr:  ${GENOS_NOOCR_MANIFEST} -> ${GENOS_OUT}"
echo "  compare src:  ${COMPARE_SOURCE_DATA} -> ${COMPARE_OUT_ROOT}"
echo "  CUDA:         ${CUDA_VISIBLE_DEVICES}"
echo "  log:          ${LOG}"
echo "=============================================="

_is_training_running() {
  pgrep -f "distill\\.student_sft.*${CONFIG_MARKER}" >/dev/null 2>&1 \
    || pgrep -f "train\\.train_qlora.*${CONFIG_MARKER}" >/dev/null 2>&1 \
    || pgrep -f "torchrun.*${CONFIG_MARKER}" >/dev/null 2>&1
}

_resolve_model_path() {
  local explicit="${MODEL:-}"
  if [[ -n "${explicit}" && -f "${explicit}/adapter_config.json" ]]; then
    echo "${explicit}"; return 0
  fi
  if [[ -f "${TRAIN_OUTPUT}/final/adapter_config.json" ]]; then
    echo "${TRAIN_OUTPUT}/final"; return 0
  fi
  local latest
  latest="$(ls -d "${TRAIN_OUTPUT}"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)"
  if [[ -n "${latest}" && -f "${latest}/adapter_config.json" ]]; then
    echo "${latest}"; return 0
  fi
  return 1
}

# --- 1) 학습 종료 대기 -------------------------------------------------------
echo "[wait] training process monitoring started..."
while _is_training_running; do
  echo "[wait] $(date '+%F %T') training still running..."
  sleep "${POLL_SEC}"
done

echo "[wait] training process not found. waiting for saved adapter..."
while ! _resolve_model_path >/dev/null 2>&1; do
  if _is_training_running; then
    echo "[wait] training restarted; continue waiting..."
    sleep "${POLL_SEC}"; continue
  fi
  echo "[wait] $(date '+%F %T') adapter not ready yet..."
  sleep "${POLL_SEC}"
done

MODEL_PATH="$(_resolve_model_path)"
echo "[ready] using MODEL=${MODEL_PATH}"

# --- 2) genos500 NO-OCR 추론 -------------------------------------------------
echo "[infer] genos500 NO-OCR ..."
BACKEND=vllm \
MODEL="${MODEL_PATH}" \
BASE_MODEL="${BASE_MODEL}" \
TEST_DATA="${GENOS_NOOCR_MANIFEST}" \
OUTPUT_DIR="${GENOS_OUT}" \
NO_THINKING="${NO_THINKING}" \
MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
bash scripts/eval_unified.sh

# --- 3) (옵션) test set OCR vs NO-OCR 비교 ----------------------------------
# 기본은 genos500 추론만 수행한다. 비교까지 자동으로 돌리려면 RUN_COMPARE=true.
if [[ "${RUN_COMPARE:-false}" == "true" ]]; then
  echo "[infer] held-out test OCR vs NO-OCR comparison ..."
  BACKEND=vllm \
  MODEL="${MODEL_PATH}" \
  BASE_MODEL="${BASE_MODEL}" \
  NO_THINKING="${NO_THINKING}" \
  MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
  SOURCE_DATA="${COMPARE_SOURCE_DATA}" \
  OUT_ROOT="${COMPARE_OUT_ROOT}" \
  bash scripts/eval_unified_ocr_compare.sh
  echo "[done] ocr compare:     ${COMPARE_OUT_ROOT}"
else
  echo "[skip] OCR/NO-OCR 비교는 RUN_COMPARE=true 일 때만 실행(기본 genos500 추론만)."
fi

echo "[done] all inference finished at $(date '+%F %T')"
echo "[done] genos500 no-ocr: ${GENOS_OUT}"
