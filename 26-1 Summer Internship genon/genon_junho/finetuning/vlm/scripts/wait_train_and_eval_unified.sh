#!/bin/bash
# =============================================================================
# unified_smoke 학습 종료를 기다린 뒤 vLLM 통합 평가를 자동 실행한다.
#
# 사용:
#   tmux new -s tsr_eval
#   source .venv/bin/activate
#   bash scripts/wait_train_and_eval_unified.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

TRAIN_OUTPUT="${TRAIN_OUTPUT:-output/unified_smoke/student_sft}"
BASE_MODEL="${BASE_MODEL:-/NHNHOME/WORKSPACE/0426030039_A/shkim/models/Qwen3.5-9B}"
TEST_DATA="${TEST_DATA:-_train_data/unified_smoke/data/test.jsonl}"
POLL_SEC="${POLL_SEC:-60}"
CONFIG_MARKER="${CONFIG_MARKER:-exp_unified_smoke.yaml}"

LOG="logs/wait_train_and_eval_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

exec > >(tee -a "${LOG}") 2>&1

echo "=============================================="
echo "  Wait train → auto eval (unified)"
echo "  project: ${PROJECT_ROOT}"
echo "  train_output: ${TRAIN_OUTPUT}"
echo "  poll: ${POLL_SEC}s"
echo "  log: ${LOG}"
echo "=============================================="

_is_training_running() {
  pgrep -f "distill\\.student_sft.*${CONFIG_MARKER}" >/dev/null 2>&1 \
    || pgrep -f "torchrun.*distill\\.student_sft" >/dev/null 2>&1
}

_resolve_model_path() {
  local explicit="${MODEL:-}"
  if [[ -n "${explicit}" && -f "${explicit}/adapter_config.json" ]]; then
    echo "${explicit}"
    return 0
  fi

  if [[ -f "${TRAIN_OUTPUT}/final/adapter_config.json" ]]; then
    echo "${TRAIN_OUTPUT}/final"
    return 0
  fi

  local latest
  latest="$(ls -d "${TRAIN_OUTPUT}"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)"
  if [[ -n "${latest}" && -f "${latest}/adapter_config.json" ]]; then
    echo "${latest}"
    return 0
  fi

  return 1
}

echo "[wait] training process monitoring started..."
while _is_training_running; do
  echo "[wait] $(date '+%F %T') training still running..."
  sleep "${POLL_SEC}"
done

echo "[wait] training process not found. waiting for saved adapter..."
while ! _resolve_model_path >/dev/null 2>&1; do
  if _is_training_running; then
    echo "[wait] training restarted; continue waiting..."
    continue
  fi
  echo "[wait] $(date '+%F %T') adapter not ready yet..."
  sleep "${POLL_SEC}"
done

MODEL_PATH="$(_resolve_model_path)"
echo "[ready] using MODEL=${MODEL_PATH}"

BACKEND=vllm \
MODEL="${MODEL_PATH}" \
BASE_MODEL="${BASE_MODEL}" \
TEST_DATA="${TEST_DATA}" \
bash scripts/eval_unified.sh

echo "[done] auto eval finished at $(date '+%F %T')"
