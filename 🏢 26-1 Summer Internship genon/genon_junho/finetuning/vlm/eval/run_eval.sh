#!/bin/bash
# ==============================================================================
# 평가 + HTML 리포트 통합 실행 스크립트
#
# 저장 정책:
#   root dir: <yyyymmdd>_<n>
#   run dir:  <root>/<yyyymmdd>_<n>_<model_short>
#   images:   <root>/images
#
# 사용법:
#   bash eval/run_eval.sh --model <checkpoint_path> --test_data <eval.jsonl>
#   bash eval/run_eval.sh --backend api --model mymodel --api_url http://... --test_data eval.jsonl
#   bash eval/run_eval.sh --output_dir eval_results
# ==============================================================================

set -euo pipefail

EVAL_ARGS=()
OUTPUT_BASE="eval_results"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output_dir)
            OUTPUT_BASE="$2"
            shift 2
            ;;
        *)
            EVAL_ARGS+=("$1")
            shift
            ;;
    esac
done

get_arg_value() {
    local key="$1"
    local next_is_value="0"
    local arg
    for arg in "${EVAL_ARGS[@]}"; do
        if [[ "$next_is_value" == "1" ]]; then
            printf '%s' "$arg"
            return 0
        fi
        if [[ "$arg" == "$key" ]]; then
            next_is_value="1"
        fi
    done
    return 1
}

sanitize_slug() {
    local raw="$1"
    local slug
    slug="$(printf '%s' "$raw" | tr '[:upper:]' '[:lower:]' | sed -E 's#^.+/##; s/[^a-z0-9]+/_/g; s/^_+//; s/_+$//; s/_+/_/g')"
    if [[ -z "$slug" ]]; then
        slug="model"
    fi
    printf '%s' "$slug"
}

DATE_PREFIX="$(date +%Y%m%d)"
N=1
while [[ -d "${OUTPUT_BASE}/${DATE_PREFIX}_${N}" ]]; do
    N=$((N + 1))
done

ROOT_ID="${DATE_PREFIX}_${N}"
ROOT_DIR="${OUTPUT_BASE}/${ROOT_ID}"

MODEL_REF="$(get_arg_value --api_model || true)"
if [[ -z "${MODEL_REF}" ]]; then
    MODEL_REF="$(get_arg_value --model || true)"
fi
if [[ -z "${MODEL_REF}" ]]; then
    MODEL_REF="model"
fi
MODEL_SHORT="$(sanitize_slug "${MODEL_REF}")"

RUN_DIR="${ROOT_DIR}/${ROOT_ID}_${MODEL_SHORT}"
IMAGES_DIR="${ROOT_DIR}/images"
mkdir -p "${RUN_DIR}" "${IMAGES_DIR}"

EVAL_ARGS+=(--output_dir "${RUN_DIR}")

echo "============================================================"
echo "  Evaluation Pipeline"
echo "============================================================"
echo "  Output root: ${ROOT_DIR}"
echo "  Run dir:     ${RUN_DIR}"
echo "  Image dir:   ${IMAGES_DIR}"

echo ""
echo "[Step 1/2] Running inference and computing metrics..."
python -m eval.evaluate "${EVAL_ARGS[@]}"

echo ""
echo "[Step 2/2] Generating HTML reports..."
VIS_COMMON_ARGS=(
    --image_mode external
    --shared_image_dir "${IMAGES_DIR}"
)

python -m eval.visualize \
    --predictions "${RUN_DIR}/predictions.jsonl" \
    --metrics "${RUN_DIR}/metrics.json" \
    --output "${RUN_DIR}/report.html" \
    "${VIS_COMMON_ARGS[@]}"

for complexity in simple medium complex; do
    python -m eval.visualize \
        --predictions "${RUN_DIR}/predictions_${complexity}.jsonl" \
        --metrics "${RUN_DIR}/metrics_${complexity}.json" \
        --output "${RUN_DIR}/report_${complexity}.html" \
        "${VIS_COMMON_ARGS[@]}"
done

echo ""
echo "============================================================"
echo "  Done!"
echo "  Root dir:            ${ROOT_DIR}"
echo "  Run dir:             ${RUN_DIR}"
echo "  Images dir:          ${IMAGES_DIR}"
echo "  Predictions:         ${RUN_DIR}/predictions.jsonl"
echo "  Metrics:             ${RUN_DIR}/metrics.json"
echo "  Full report:         ${RUN_DIR}/report.html"
echo "  Simple report:       ${RUN_DIR}/report_simple.html"
echo "  Medium report:       ${RUN_DIR}/report_medium.html"
echo "  Complex report:      ${RUN_DIR}/report_complex.html"
echo "============================================================"
