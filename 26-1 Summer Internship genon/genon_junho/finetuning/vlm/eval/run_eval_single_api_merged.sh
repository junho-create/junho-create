#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  # single GPU
  bash eval/run_eval_single_api_merged.sh \
    --base_model_path /home/vlm_train/models/Qwen3.5-9B \
    --adapter_path /home/vlm_train/qwen3_vl_tsr/output/e15_qwen35_9b_6000/student_sft/checkpoint-400 \
    --model_alias Qwen/Qwen3.5-9B-e15-ckpt400-merged \
    --tag e15ckpt400_noreason_nothink_b16 \
    --test_data /home/vlm_train/qwen3_vl_tsr/_train_data/20260317_4_6000/data_split/test.jsonl \
    --gpu_ids 0 \
    --port 8030

  # multi GPU (tp auto=number of gpu ids)
  bash eval/run_eval_single_api_merged.sh \
    --base_model_path /home/vlm_train/models/Qwen3.5-9B \
    --adapter_path /home/vlm_train/qwen3_vl_tsr/output/e15_qwen35_9b_6000/student_sft/checkpoint-700 \
    --model_alias Qwen/Qwen3.5-9B-e15-ckpt700-merged \
    --tag e15ckpt700_noreason_nothink_b16 \
    --test_data /home/vlm_train/qwen3_vl_tsr/_train_data/20260317_4_6000/data_split/test.jsonl \
    --gpu_ids 0,1,2,3 \
    --port 8030

Notes:
- vLLM serve command intentionally does NOT include --reasoning-parser qwen3.
- Defaults are aligned to prior e14 API eval runs:
  batch_size=16, max_new_tokens=10000, retries=5, no-thinking.
- OCR usage control:
  --use_ocr true|false
  (prompt_style 미지정 시 true=with_ocr, false=without_ocr)
- Report image rendering control:
  --report_image_mode embed|external|none
  --report_shared_image_dir <dir>  (external 모드에서 비어있으면 ${OUT_DIR}/images 사용)
  --report_image_root <dir>        (image_path 상대경로 해석 보정)
USAGE
}

BASE_MODEL_PATH="/home/vlm_train/models/Qwen3.5-9B"
ADAPTER_PATH=""
MERGED_PATH=""
MODEL_ALIAS=""
TAG=""
TEST_DATA=""
TEST_TAG="test"
GPU_IDS="0"
PORT="8030"
MAX_SAMPLES="200"
BATCH_SIZE="16"
MAX_NEW_TOKENS="10000"
TEMPERATURE="0.0"
RETRIES="5"
EMPTY_RETRY_BACKOFF="0.5"
API_TIMEOUT="180"
PROMPT_STYLE=""
PROMPT_STYLE_EXPLICIT="0"
USE_OCR="true"
GT_SOURCE="dataset"
NORMALIZE_EMPTY_CELLS="true"
EMPTY_CELL_TOKEN="__EMPTY__"
NESTED_TEDS_MODE="split_mean"
OUT_ROOT="eval_results"
LOG_DIR="logs"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
MAX_MODEL_LEN="262144"
MAX_NUM_SEQS="32"
TENSOR_PARALLEL_SIZE="auto"
FORCE_MERGE="0"
SKIP_REPORT="0"
PYTHON_BIN="${PYTHON_BIN:-python}"
REPORT_IMAGE_MODE="embed"
REPORT_SHARED_IMAGE_DIR=""
REPORT_IMAGE_ROOT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --base_model_path)
            BASE_MODEL_PATH="$2"
            shift 2
            ;;
        --adapter_path)
            ADAPTER_PATH="$2"
            shift 2
            ;;
        --merged_path)
            MERGED_PATH="$2"
            shift 2
            ;;
        --model_alias)
            MODEL_ALIAS="$2"
            shift 2
            ;;
        --tag)
            TAG="$2"
            shift 2
            ;;
        --test_data)
            TEST_DATA="$2"
            shift 2
            ;;
        --test_tag)
            TEST_TAG="$2"
            shift 2
            ;;
        --gpu_ids)
            GPU_IDS="$2"
            shift 2
            ;;
        --gpu_id)
            GPU_IDS="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --max_samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --max_new_tokens)
            MAX_NEW_TOKENS="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --retries)
            RETRIES="$2"
            shift 2
            ;;
        --empty_retry_backoff)
            EMPTY_RETRY_BACKOFF="$2"
            shift 2
            ;;
        --api_timeout)
            API_TIMEOUT="$2"
            shift 2
            ;;
        --prompt_style)
            PROMPT_STYLE="$2"
            PROMPT_STYLE_EXPLICIT="1"
            shift 2
            ;;
        --use_ocr)
            USE_OCR="$2"
            shift 2
            ;;
        --gt_source)
            GT_SOURCE="$2"
            shift 2
            ;;
        --normalize_empty_cells)
            NORMALIZE_EMPTY_CELLS="$2"
            shift 2
            ;;
        --empty_cell_token)
            EMPTY_CELL_TOKEN="$2"
            shift 2
            ;;
        --nested_teds_mode)
            NESTED_TEDS_MODE="$2"
            shift 2
            ;;
        --out_root)
            OUT_ROOT="$2"
            shift 2
            ;;
        --log_dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --run_stamp)
            RUN_STAMP="$2"
            shift 2
            ;;
        --max_model_len)
            MAX_MODEL_LEN="$2"
            shift 2
            ;;
        --max_num_seqs)
            MAX_NUM_SEQS="$2"
            shift 2
            ;;
        --tensor_parallel_size)
            TENSOR_PARALLEL_SIZE="$2"
            shift 2
            ;;
        --force_merge)
            FORCE_MERGE="1"
            shift
            ;;
        --skip_report)
            SKIP_REPORT="1"
            shift
            ;;
        --python_bin)
            PYTHON_BIN="$2"
            shift 2
            ;;
        --report_image_mode)
            REPORT_IMAGE_MODE="$2"
            shift 2
            ;;
        --report_shared_image_dir)
            REPORT_SHARED_IMAGE_DIR="$2"
            shift 2
            ;;
        --report_image_root)
            REPORT_IMAGE_ROOT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ -z "${ADAPTER_PATH}" || -z "${MODEL_ALIAS}" || -z "${TAG}" || -z "${TEST_DATA}" ]]; then
    echo "[ERROR] Missing required args: --adapter_path, --model_alias, --tag, --test_data" >&2
    usage
    exit 1
fi

if [[ ! -d "${ADAPTER_PATH}" ]]; then
    echo "[ERROR] Adapter path not found: ${ADAPTER_PATH}" >&2
    exit 1
fi
if [[ ! -f "${TEST_DATA}" ]]; then
    echo "[ERROR] Test data not found: ${TEST_DATA}" >&2
    exit 1
fi

case "${USE_OCR,,}" in
    true|1|yes|y)
        USE_OCR="true"
        ;;
    false|0|no|n)
        USE_OCR="false"
        ;;
    *)
        echo "[ERROR] --use_ocr must be true/false: ${USE_OCR}" >&2
        exit 1
        ;;
esac

REPORT_IMAGE_MODE="${REPORT_IMAGE_MODE,,}"
if [[ "${REPORT_IMAGE_MODE}" != "embed" && "${REPORT_IMAGE_MODE}" != "external" && "${REPORT_IMAGE_MODE}" != "none" ]]; then
    echo "[ERROR] --report_image_mode must be embed|external|none: ${REPORT_IMAGE_MODE}" >&2
    exit 1
fi

if [[ "${PROMPT_STYLE_EXPLICIT}" != "1" ]]; then
    if [[ "${USE_OCR}" == "true" ]]; then
        PROMPT_STYLE="chandra_table_with_ocr"
    else
        PROMPT_STYLE="chandra_table_without_ocr"
    fi
fi

GPU_IDS="${GPU_IDS//./,}"
GPU_IDS="${GPU_IDS// /}"
if [[ ! "${GPU_IDS}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "[ERROR] Invalid --gpu_ids: ${GPU_IDS} (expected 0 or 0,1,2)" >&2
    exit 1
fi

IFS=',' read -r -a GPU_ARR <<< "${GPU_IDS}"
GPU_COUNT="${#GPU_ARR[@]}"
MERGE_GPU_ID="${GPU_ARR[0]}"

if [[ "${TENSOR_PARALLEL_SIZE}" == "auto" || -z "${TENSOR_PARALLEL_SIZE}" ]]; then
    TP_SIZE="${GPU_COUNT}"
else
    TP_SIZE="${TENSOR_PARALLEL_SIZE}"
fi

if ! [[ "${TP_SIZE}" =~ ^[0-9]+$ ]] || [[ "${TP_SIZE}" -lt 1 ]]; then
    echo "[ERROR] Invalid tensor_parallel_size: ${TP_SIZE}" >&2
    exit 1
fi

if [[ "${TP_SIZE}" -gt "${GPU_COUNT}" ]]; then
    echo "[ERROR] tensor_parallel_size=${TP_SIZE} > visible gpus in --gpu_ids (${GPU_COUNT})" >&2
    exit 1
fi

if [[ -z "${MERGED_PATH}" ]]; then
    base_dir="$(dirname "${BASE_MODEL_PATH}")"
    base_name="$(basename "${BASE_MODEL_PATH}")"
    MERGED_PATH="${base_dir}/${base_name}-${TAG}-merged-full"
fi

mkdir -p "${OUT_ROOT}" "${LOG_DIR}"

GPU_LABEL="${GPU_IDS//,/}"
LOG_VLLM="${LOG_DIR}/vllm_qwen35_${TAG}_gpus${GPU_LABEL}_tmux_${RUN_STAMP}.log"
LOG_MERGE="${LOG_DIR}/merge_qwen35_${TAG}_full_${RUN_STAMP}.log"
LOG_EVAL="${LOG_DIR}/eval_${TAG}_on_${TEST_TAG}_retry${RETRIES}_b${BATCH_SIZE}_${RUN_STAMP}.log"
OUT_DIR="${OUT_ROOT}/${TAG}_api_on_${TEST_TAG}_retry${RETRIES}_b${BATCH_SIZE}_m${MAX_NEW_TOKENS}_${RUN_STAMP}"

exec > >(tee -a "${LOG_EVAL}") 2>&1

echo "[INFO] TAG=${TAG}"
echo "[INFO] GPU_IDS=${GPU_IDS}"
echo "[INFO] TP_SIZE=${TP_SIZE}"
echo "[INFO] PORT=${PORT}"
echo "[INFO] BASE_MODEL_PATH=${BASE_MODEL_PATH}"
echo "[INFO] ADAPTER_PATH=${ADAPTER_PATH}"
echo "[INFO] MERGED_PATH=${MERGED_PATH}"
echo "[INFO] MODEL_ALIAS=${MODEL_ALIAS}"
echo "[INFO] TEST_DATA=${TEST_DATA}"
echo "[INFO] MAX_SAMPLES=${MAX_SAMPLES}"
echo "[INFO] RETRIES=${RETRIES}"
echo "[INFO] BATCH_SIZE=${BATCH_SIZE}"
echo "[INFO] USE_OCR=${USE_OCR}"
echo "[INFO] PROMPT_STYLE=${PROMPT_STYLE}"
echo "[INFO] NESTED_TEDS_MODE=${NESTED_TEDS_MODE}"
echo "[INFO] OUT_DIR=${OUT_DIR}"
echo "[INFO] REPORT_IMAGE_MODE=${REPORT_IMAGE_MODE}"
echo "[INFO] REPORT_SHARED_IMAGE_DIR=${REPORT_SHARED_IMAGE_DIR:-<auto>}"
echo "[INFO] REPORT_IMAGE_ROOT=${REPORT_IMAGE_ROOT:-<auto>}"
echo "[INFO] SPEED_OPTS=parallel_tmux + merged_full + batch_size${BATCH_SIZE} + no_thinking + max_num_seqs${MAX_NUM_SEQS} + tp${TP_SIZE} + ocr_${USE_OCR}"

# vLLM internals use torch.distributed even for single-GPU workers.
# When multiple eval workers start concurrently on one host, ensure each
# worker has a unique rendezvous port to avoid EADDRINUSE collisions.
DIST_PORT="${VLLM_DIST_PORT:-$((PORT + 10000))}"
DP_DIST_PORT="${VLLM_DP_MASTER_PORT_OVERRIDE:-$((DIST_PORT + 100))}"
if ! [[ "${DIST_PORT}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] Invalid VLLM_DIST_PORT: ${DIST_PORT}" >&2
    exit 1
fi
if ! [[ "${DP_DIST_PORT}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] Invalid VLLM_DP_MASTER_PORT_OVERRIDE: ${DP_DIST_PORT}" >&2
    exit 1
fi
echo "[INFO] VLLM_DIST_PORT=${DIST_PORT}"
echo "[INFO] VLLM_DP_MASTER_PORT=${DP_DIST_PORT}"

API_TRUNCATE_PROMPT_TOKENS=0
if [[ "${MAX_MODEL_LEN}" =~ ^[0-9]+$ ]] && [[ "${MAX_NEW_TOKENS}" =~ ^[0-9]+$ ]]; then
    if (( MAX_MODEL_LEN > MAX_NEW_TOKENS )); then
        API_TRUNCATE_PROMPT_TOKENS=$((MAX_MODEL_LEN - MAX_NEW_TOKENS))
    fi
fi
if (( API_TRUNCATE_PROMPT_TOKENS > 0 )); then
    echo "[INFO] API_TRUNCATE_PROMPT_TOKENS=${API_TRUNCATE_PROMPT_TOKENS}"
else
    echo "[INFO] API_TRUNCATE_PROMPT_TOKENS=disabled"
fi

SERVER_PID=""
cleanup() {
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
        kill "${SERVER_PID}" >/dev/null 2>&1 || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

if [[ "${FORCE_MERGE}" == "1" || ! -f "${MERGED_PATH}/config.json" ]]; then
    echo "[INFO] merge start"
    CUDA_VISIBLE_DEVICES="${MERGE_GPU_ID}" "${PYTHON_BIN}" - "${BASE_MODEL_PATH}" "${ADAPTER_PATH}" "${MERGED_PATH}" >"${LOG_MERGE}" 2>&1 <<'PY'
import os
import sys
import torch
from peft import PeftModel
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

base_model_path = sys.argv[1]
lora_adapter_path = sys.argv[2]
output_path = sys.argv[3]

cfg = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
model_type = getattr(cfg, "model_type", "unknown")
print(f"Detected model_type: {model_type}")
if model_type not in {"qwen2_vl", "qwen2_5_vl"}:
    print(f"Warning: Unknown model_type '{model_type}', falling back to AutoModelForImageTextToText")

model = AutoModelForImageTextToText.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)
model = PeftModel.from_pretrained(model, lora_adapter_path)
model = model.merge_and_unload()

os.makedirs(output_path, exist_ok=True)
model.save_pretrained(output_path)
processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
processor.save_pretrained(output_path)
print("DONE")
PY
    echo "[INFO] merge done"
else
    echo "[INFO] merge skipped (reuse existing merged model)"
fi

export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export VLLM_PLUGINS=""
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK="True"

VLLM_BIN="$(dirname "${PYTHON_BIN}")/vllm"
if [[ ! -x "${VLLM_BIN}" ]]; then
    VLLM_BIN="$(command -v vllm || true)"
fi
if [[ -z "${VLLM_BIN}" ]]; then
    echo "[ERROR] vLLM executable not found. Checked: $(dirname "${PYTHON_BIN}")/vllm and PATH(vllm)." >&2
    exit 1
fi

CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
MASTER_ADDR="127.0.0.1" \
MASTER_PORT="${DIST_PORT}" \
VLLM_PORT="${DIST_PORT}" \
VLLM_DP_MASTER_PORT="${DP_DIST_PORT}" \
"${VLLM_BIN}" serve "${MERGED_PATH}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --served-model-name "${MODEL_ALIAS}" \
    --master-port "${DIST_PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --skip-mm-profiling \
    --enforce-eager \
    >"${LOG_VLLM}" 2>&1 &
SERVER_PID="$!"

READY="0"
for _ in $(seq 1 180); do
    if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
        READY="1"
        break
    fi
    if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
        echo "[ERROR] vLLM process exited before ready. Check: ${LOG_VLLM}" >&2
        tail -n 80 "${LOG_VLLM}" || true
        exit 1
    fi
    sleep 2
done

if [[ "${READY}" != "1" ]]; then
    echo "[ERROR] vLLM readiness timeout on port ${PORT}. Check: ${LOG_VLLM}" >&2
    tail -n 80 "${LOG_VLLM}" || true
    exit 1
fi

echo "[INFO] vLLM ready on ${PORT}"

EVAL_CMD=(
    "${PYTHON_BIN}" -m eval.evaluate
    --model "${MODEL_ALIAS}"
    --backend api
    --api_url "http://127.0.0.1:${PORT}/v1/chat/completions"
    --api_model "${MODEL_ALIAS}"
    --test_data "${TEST_DATA}"
    --output_dir "${OUT_DIR}"
    --batch_size "${BATCH_SIZE}"
    --max_new_tokens "${MAX_NEW_TOKENS}"
    --temperature "${TEMPERATURE}"
    --api_timeout "${API_TIMEOUT}"
    --api_empty_response_retries "${RETRIES}"
    --api_empty_retry_backoff "${EMPTY_RETRY_BACKOFF}"
    --api_empty_retry_disable_thinking
    --api_force_chat_template_no_thinking
    --prompt_style "${PROMPT_STYLE}"
    --gt_source "${GT_SOURCE}"
    --nested_teds_mode "${NESTED_TEDS_MODE}"
    --no-thinking
)
if [[ "${NORMALIZE_EMPTY_CELLS}" == "true" ]]; then
    EVAL_CMD+=(--normalize_empty_cells --empty_cell_token "${EMPTY_CELL_TOKEN}")
fi
if [[ "${MAX_SAMPLES}" != "" && "${MAX_SAMPLES}" != "0" ]]; then
    EVAL_CMD+=(--max_samples "${MAX_SAMPLES}")
fi
if (( API_TRUNCATE_PROMPT_TOKENS > 0 )); then
    EVAL_CMD+=(--api_truncate_prompt_tokens "${API_TRUNCATE_PROMPT_TOKENS}")
fi

"${EVAL_CMD[@]}"

if [[ "${SKIP_REPORT}" != "1" ]]; then
    EFFECTIVE_SHARED_IMAGE_DIR="${REPORT_SHARED_IMAGE_DIR}"
    if [[ "${REPORT_IMAGE_MODE}" == "external" && -z "${EFFECTIVE_SHARED_IMAGE_DIR}" ]]; then
        EFFECTIVE_SHARED_IMAGE_DIR="${OUT_DIR}/images"
    fi

    REPORT_BASE_CMD=(
        "${PYTHON_BIN}" -m eval.visualize
        --image_mode "${REPORT_IMAGE_MODE}"
    )
    if [[ -n "${EFFECTIVE_SHARED_IMAGE_DIR}" ]]; then
        REPORT_BASE_CMD+=(--shared_image_dir "${EFFECTIVE_SHARED_IMAGE_DIR}")
    fi
    if [[ -n "${REPORT_IMAGE_ROOT}" ]]; then
        REPORT_BASE_CMD+=(--image_root "${REPORT_IMAGE_ROOT}")
    fi

    "${REPORT_BASE_CMD[@]}" \
        --predictions "${OUT_DIR}/predictions.jsonl" \
        --metrics "${OUT_DIR}/metrics.json" \
        --output "${OUT_DIR}/report.html"

    for complexity in simple medium complex; do
        pred_path="${OUT_DIR}/predictions_${complexity}.jsonl"
        metric_path="${OUT_DIR}/metrics_${complexity}.json"
        if [[ -f "${pred_path}" && -f "${metric_path}" ]]; then
            "${REPORT_BASE_CMD[@]}" \
                --predictions "${pred_path}" \
                --metrics "${metric_path}" \
                --output "${OUT_DIR}/report_${complexity}.html"
        fi
    done
fi

echo "DONE TAG=${TAG} OUT_DIR=${OUT_DIR} LOG_EVAL=${LOG_EVAL} LOG_VLLM=${LOG_VLLM} LOG_MERGE=${LOG_MERGE}"
