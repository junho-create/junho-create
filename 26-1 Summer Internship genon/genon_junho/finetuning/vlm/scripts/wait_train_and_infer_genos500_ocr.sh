#!/bin/bash
# =============================================================================
# 지정한 학습(CONFIG_MARKER) 종료를 기다린 뒤, genos 500장에 대해
# OCR 사용(with_ocr) 매니페스트로 통합 추론을 자동 실행한다.
#
# - 입력: eval_data/500_genos_val_set/inference_manifest_with_ocr.jsonl
#   (이미 chandra_with_ocr / bbox_scale=1000 / output_format=html / ocr 0-1000 정렬됨)
#   → evaluate_unified.py 가 prompt_style=chandra_with_ocr 로 OCR 프롬프트를 사용.
# - 백엔드: vllm (학습으로 나온 LoRA 어댑터를 로컬에서 직접 로딩).
#
# 필수 env:
#   CONFIG_MARKER  : 학습 프로세스 감지용 config 파일명 (예: exp_chandra_layout.yaml)
#   TRAIN_OUTPUT   : 학습 출력 디렉터리 (예: output/chandra_layout_20260622/student_sft)
#   OUTPUT_DIR     : 추론 결과 저장 디렉터리
#   CUDA_VISIBLE_DEVICES : 추론에 쓸 GPU (학습 종료 후 유휴 GPU 1장)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

CONFIG_MARKER="${CONFIG_MARKER:?CONFIG_MARKER required}"
TRAIN_OUTPUT="${TRAIN_OUTPUT:?TRAIN_OUTPUT required}"
BASE_MODEL="${BASE_MODEL:-/NHNHOME/WORKSPACE/0426030039_A/shkim/models/Qwen3.5-9B}"
EVAL_ROOT="${EVAL_ROOT:-eval_data/500_genos_val_set}"
MANIFEST="${MANIFEST:-${EVAL_ROOT}/inference_manifest_with_ocr.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-eval_results/genos500_ocr_$(date +%Y%m%d_%H%M%S)}"
POLL_SEC="${POLL_SEC:-60}"
NO_THINKING="${NO_THINKING:-true}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-24576}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"

LOG="logs/wait_infer_ocr_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs
exec > >(tee -a "${LOG}") 2>&1

echo "=============================================="
echo "  Wait train -> auto infer genos500 (WITH OCR)"
echo "  config_marker:${CONFIG_MARKER}"
echo "  train_output: ${TRAIN_OUTPUT}"
echo "  base_model:   ${BASE_MODEL}"
echo "  manifest:     ${MANIFEST}"
echo "  output_dir:   ${OUTPUT_DIR}"
echo "  CUDA:         ${CUDA_VISIBLE_DEVICES}"
echo "  max_model_len:${MAX_MODEL_LEN}"
echo "  poll:         ${POLL_SEC}s"
echo "  log:          ${LOG}"
echo "=============================================="

_is_training_running() {
  pgrep -f "distill\\.student_sft.*${CONFIG_MARKER}" >/dev/null 2>&1 \
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
  # WAIT_FOR_FINAL_ONLY=1 이면 학습 종료로 생성되는 final 만 사용한다.
  # (중간 checkpoint 로 fallback 하지 않으므로, 학습이 GPU 를 점유 중인 상태에서
  #  ckpt 로 추론을 시작했다가 GPU OOM 으로 죽는 사고를 방지한다.)
  if [[ "${WAIT_FOR_FINAL_ONLY:-0}" == "1" ]]; then
    return 1
  fi
  local latest
  latest="$(ls -d "${TRAIN_OUTPUT}"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)"
  if [[ -n "${latest}" && -f "${latest}/adapter_config.json" ]]; then
    echo "${latest}"; return 0
  fi
  return 1
}

# --- 1) 학습 종료 대기 -------------------------------------------------------
echo "[wait] training(${CONFIG_MARKER}) monitoring started..."
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

# --- 2) genos500 WITH-OCR 추론 ----------------------------------------------
echo "[infer] genos500 WITH-OCR ..."
BACKEND=vllm \
MODEL="${MODEL_PATH}" \
BASE_MODEL="${BASE_MODEL}" \
TEST_DATA="${MANIFEST}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
NO_THINKING="${NO_THINKING}" \
MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
bash scripts/eval_unified.sh

echo "[done] genos500 with-ocr inference finished at $(date '+%F %T')"
echo "[done] results: ${OUTPUT_DIR}"
