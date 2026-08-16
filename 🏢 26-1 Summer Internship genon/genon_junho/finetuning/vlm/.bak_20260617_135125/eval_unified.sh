#!/bin/bash
# =============================================================================
# Unified (Table + Layout) 모델 평가 실행 스크립트
# - eval/evaluate_unified.py 를 호출한다.
# - 기본 backend=api (서버 vLLM 서빙: http://192.168.75.173:8010).
# - 학습으로 나온 LoRA/머지 모델을 로컬에서 직접 평가하려면 BACKEND=vllm 사용.
#
# 사용 예:
#   # (A) 서버 API 평가 (서빙 중인 통합 모델)
#   API_URL=http://192.168.75.173:8010/v1/chat/completions \
#   API_MODEL=<served-model-name> \
#   TEST_DATA=_train_data/unified_smoke/data/test.jsonl \
#   bash scripts/eval_unified.sh
#
#   # (B) 로컬 vLLM 평가 (LoRA 어댑터 경로)
#   BACKEND=vllm \
#   MODEL=output/unified_smoke/student_sft/checkpoint-300 \
#   BASE_MODEL=/home/vlm_train/models/Qwen3.5-9B \
#   TEST_DATA=_train_data/unified_smoke/data/test.jsonl \
#   bash scripts/eval_unified.sh
#
# 출력 형식(파이프라인)은 TEST_DATA 의 prompt_style 로 자동 감지된다:
#   - JSON 셋 (_train_data/unified_smoke/...)      → table=TEDS, layout=IoU F1
#   - HTML 셋 (_train_data/unified_html_smoke/...)  → table=TEDS, layout=HTML-tree TEDS
#   예) TEST_DATA=_train_data/unified_html_smoke/data/test.jsonl bash scripts/eval_unified.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

BACKEND=${BACKEND:-api}
TEST_DATA=${TEST_DATA:-_train_data/unified_smoke/data/test.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-eval_results/unified_$(date +%Y%m%d_%H%M%S)}
IOU_THRESHOLD=${IOU_THRESHOLD:-0.5}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-10000}
MAX_SAMPLES=${MAX_SAMPLES:-}
# 학습이 JSON-only(thinking 미포함)이므로 평가도 thinking 을 끈다(기본 true).
# thinking 을 다시 켜려면 NO_THINKING=false 로 실행.
NO_THINKING=${NO_THINKING:-true}
# 긴 표/페이지에서 입력(이미지+OCR)+출력이 한도를 넘어 표가 잘리거나 빈 응답이 나오는 것을
# 줄이기 위해 추론 컨텍스트 한도를 상향한다(vllm 백엔드). api 백엔드는 서빙 측 --max-model-len 으로 맞춘다.
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}

ARGS=(
  --backend "${BACKEND}"
  --test_data "${TEST_DATA}"
  --output_dir "${OUTPUT_DIR}"
  --iou_threshold "${IOU_THRESHOLD}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
)

if [[ "${NO_THINKING}" == "true" ]]; then
  ARGS+=(--no-thinking)
fi

if [[ -n "${MAX_SAMPLES}" ]]; then
  ARGS+=(--max_samples "${MAX_SAMPLES}")
fi

if [[ "${BACKEND}" == "api" ]]; then
  ARGS+=(--api_url "${API_URL:-${TABLE_VLM_URL:-}}")
  ARGS+=(--api_model "${API_MODEL:-${TABLE_VLM_MODEL:-}}")
else
  ARGS+=(--model "${MODEL:?MODEL is required for backend=${BACKEND}}")
  [[ -n "${BASE_MODEL:-}" ]] && ARGS+=(--base_model "${BASE_MODEL}")
  ARGS+=(--batch_size "${BATCH_SIZE:-8}")
  ARGS+=(--max_model_len "${MAX_MODEL_LEN}")
fi

mkdir -p logs
python -m eval.evaluate_unified "${ARGS[@]}" \
  2>&1 | tee "logs/eval_unified_$(date +%Y%m%d_%H%M%S).log"
