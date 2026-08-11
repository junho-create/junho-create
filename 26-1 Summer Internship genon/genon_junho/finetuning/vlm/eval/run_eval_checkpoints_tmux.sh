#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  # run inside train/vlm on training server
  bash eval/run_eval_checkpoints_tmux.sh \
    --exp_name e15 \
    --adapter_root /home/vlm_train/qwen3_vl_tsr/output/e15_qwen35_9b_6000/student_sft \
    --test_data /home/vlm_train/qwen3_vl_tsr/_train_data/20260317_4_6000/data_split/test.jsonl \
    --model_gpu_map "400:0;500:1;600:2;final:3"

Dynamic mapping examples:
  --model_gpu_map "400:0;500:1;600:2;700:3"
  --model_gpu_map "700:0,1,2,3"
  --model_gpu_map "200:0,1;400:2.3"   # '.' is treated as ','
  --model_gpu_map "100:0;200:0;300:0"

Notes:
- tmux session per model mapping entry is created.
- launch mode:
  - parallel (default): 각 매핑을 즉시 병렬 실행
  - sequential: 중복 GPU만 락 기반 순차 실행(비중복 GPU는 병렬 실행)
- OCR usage control:
  --use_ocr true|false
  (prompt_style 미지정 시 true=with_ocr, false=without_ocr)
- Report image rendering control:
  --report_image_mode embed|external|none
  --report_shared_image_dir <dir>
  --report_image_root <dir>
- vLLM serve does NOT use --reasoning-parser qwen3.
- Defaults match prior e14 eval profile:
  batch_size=16, max_new_tokens=10000, retries=5,
  --no-thinking, --api_force_chat_template_no_thinking.
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

EXP_NAME="e15"
BASE_MODEL_PATH="/home/vlm_train/models/Qwen3.5-9B"
ADAPTER_ROOT="/home/vlm_train/qwen3_vl_tsr/output/e15_qwen35_9b_6000/student_sft"
TEST_DATA="/home/vlm_train/qwen3_vl_tsr/_train_data/20260317_4_6000/data_split/test.jsonl"
TEST_TAG=""
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
START_PORT="8030"
MAX_MODEL_LEN="262144"
MAX_NUM_SEQS="32"
TENSOR_PARALLEL_SIZE="auto"
FORCE_MERGE="0"
SKIP_REPORT="0"
PYTHON_BIN="${PYTHON_BIN:-python}"
LAUNCH_MODE="parallel"
REPORT_IMAGE_MODE="embed"
REPORT_SHARED_IMAGE_DIR=""
REPORT_IMAGE_ROOT=""

# Default requested mapping
MODEL_GPU_MAP="400:0;500:1;600:2;final:3"

# Backward compatibility (single-gpu style)
CHECKPOINTS_CSV=""
GPUS_CSV=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --project_root)
            PROJECT_ROOT="$2"
            shift 2
            ;;
        --exp_name)
            EXP_NAME="$2"
            shift 2
            ;;
        --base_model_path)
            BASE_MODEL_PATH="$2"
            shift 2
            ;;
        --adapter_root)
            ADAPTER_ROOT="$2"
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
        --start_port)
            START_PORT="$2"
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
        --model_gpu_map)
            MODEL_GPU_MAP="$2"
            shift 2
            ;;
        --checkpoints)
            CHECKPOINTS_CSV="$2"
            shift 2
            ;;
        --gpus)
            GPUS_CSV="$2"
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
        --launch_mode)
            LAUNCH_MODE="$2"
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

if ! command -v tmux >/dev/null 2>&1; then
    echo "[ERROR] tmux not found. Install tmux first." >&2
    exit 1
fi
if [[ "${LAUNCH_MODE}" == "sequential" ]] && ! command -v flock >/dev/null 2>&1; then
    echo "[ERROR] flock not found. Install util-linux first for launch_mode=sequential." >&2
    exit 1
fi

if [[ ! -d "${PROJECT_ROOT}" ]]; then
    echo "[ERROR] project_root not found: ${PROJECT_ROOT}" >&2
    exit 1
fi
if [[ ! -d "${ADAPTER_ROOT}" ]]; then
    echo "[ERROR] adapter_root not found: ${ADAPTER_ROOT}" >&2
    exit 1
fi
if [[ ! -f "${TEST_DATA}" ]]; then
    echo "[ERROR] test_data not found: ${TEST_DATA}" >&2
    exit 1
fi
if [[ "${LAUNCH_MODE}" != "parallel" && "${LAUNCH_MODE}" != "sequential" ]]; then
    echo "[ERROR] launch_mode must be 'parallel' or 'sequential': ${LAUNCH_MODE}" >&2
    exit 1
fi
REPORT_IMAGE_MODE="${REPORT_IMAGE_MODE,,}"
if [[ "${REPORT_IMAGE_MODE}" != "embed" && "${REPORT_IMAGE_MODE}" != "external" && "${REPORT_IMAGE_MODE}" != "none" ]]; then
    echo "[ERROR] --report_image_mode must be embed|external|none: ${REPORT_IMAGE_MODE}" >&2
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
if [[ "${PROMPT_STYLE_EXPLICIT}" != "1" ]]; then
    if [[ "${USE_OCR}" == "true" ]]; then
        PROMPT_STYLE="chandra_table_with_ocr"
    else
        PROMPT_STYLE="chandra_table_without_ocr"
    fi
fi

declare -a CHECKPOINTS=()
declare -a GPU_GROUPS=()

# Backward compatibility parser: --checkpoints 400,500 --gpus 0,1
if [[ -n "${CHECKPOINTS_CSV}" || -n "${GPUS_CSV}" ]]; then
    if [[ -z "${CHECKPOINTS_CSV}" || -z "${GPUS_CSV}" ]]; then
        echo "[ERROR] --checkpoints and --gpus must be provided together" >&2
        exit 1
    fi
    IFS=',' read -r -a ckpts_old <<< "${CHECKPOINTS_CSV}"
    IFS=',' read -r -a gpus_old <<< "${GPUS_CSV}"
    if [[ "${#ckpts_old[@]}" -ne "${#gpus_old[@]}" ]]; then
        echo "[ERROR] checkpoints count != gpus count" >&2
        exit 1
    fi
    for i in "${!ckpts_old[@]}"; do
        ckpt_raw="${ckpts_old[$i]}"
        if [[ "${ckpt_raw}" == "final" ]]; then
            ckpt_norm="final"
        elif [[ "${ckpt_raw}" =~ ^(ckpt|checkpoint-)?([0-9]+)$ ]]; then
            ckpt_norm="${BASH_REMATCH[2]}"
        else
            echo "[ERROR] invalid checkpoint id in --checkpoints: ${ckpt_raw}" >&2
            exit 1
        fi
        CHECKPOINTS+=("${ckpt_norm}")
        GPU_GROUPS+=("${gpus_old[$i]}")
    done
else
    map_compact="${MODEL_GPU_MAP// /}"
    if [[ -z "${map_compact}" ]]; then
        echo "[ERROR] model_gpu_map is empty" >&2
        exit 1
    fi

    IFS=';' read -r -a pairs <<< "${map_compact}"
    for pair in "${pairs[@]}"; do
        [[ -z "${pair}" ]] && continue
        if [[ "${pair}" != *:* ]]; then
            echo "[ERROR] invalid mapping entry (missing ':'): ${pair}" >&2
            exit 1
        fi

        ckpt="${pair%%:*}"
        gpu_spec="${pair#*:}"

        if [[ -z "${ckpt}" || -z "${gpu_spec}" ]]; then
            echo "[ERROR] invalid mapping entry (empty ckpt/gpu): ${pair}" >&2
            exit 1
        fi

        if [[ "${ckpt}" == "final" ]]; then
            ckpt_norm="final"
        elif [[ "${ckpt}" =~ ^(ckpt|checkpoint-)?([0-9]+)$ ]]; then
            ckpt_norm="${BASH_REMATCH[2]}"
        else
            echo "[ERROR] checkpoint must be integer/final (or ckpt prefix): ${ckpt}" >&2
            exit 1
        fi

        gpu_spec="${gpu_spec//./,}"
        gpu_spec="${gpu_spec// /}"

        if [[ ! "${gpu_spec}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
            echo "[ERROR] invalid gpu list in '${pair}' (use 0 or 0,1,2): ${gpu_spec}" >&2
            exit 1
        fi

        CHECKPOINTS+=("${ckpt_norm}")
        GPU_GROUPS+=("${gpu_spec}")
    done
fi

if [[ "${#CHECKPOINTS[@]}" -eq 0 ]]; then
    echo "[ERROR] No valid checkpoint mapping entries found" >&2
    exit 1
fi

if [[ -z "${TEST_TAG}" ]]; then
    if [[ -n "${MAX_SAMPLES}" && "${MAX_SAMPLES}" != "0" ]]; then
        TEST_TAG="${EXP_NAME}test${MAX_SAMPLES}"
    else
        TEST_TAG="${EXP_NAME}test"
    fi
fi

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "${PROJECT_ROOT}/${LOG_DIR}" "${PROJECT_ROOT}/${OUT_ROOT}"

WORKER_SCRIPT="${SCRIPT_DIR}/run_eval_single_api_merged.sh"
if [[ ! -f "${WORKER_SCRIPT}" ]]; then
    echo "[ERROR] worker script missing: ${WORKER_SCRIPT}" >&2
    exit 1
fi

echo "[INFO] project_root=${PROJECT_ROOT}"
echo "[INFO] exp_name=${EXP_NAME}"
echo "[INFO] adapter_root=${ADAPTER_ROOT}"
echo "[INFO] test_data=${TEST_DATA}"
echo "[INFO] model_gpu_map=${MODEL_GPU_MAP}"
echo "[INFO] use_ocr=${USE_OCR}"
echo "[INFO] prompt_style=${PROMPT_STYLE}"
echo "[INFO] launch_mode=${LAUNCH_MODE}"
echo "[INFO] report_image_mode=${REPORT_IMAGE_MODE}"
echo "[INFO] report_shared_image_dir=${REPORT_SHARED_IMAGE_DIR:-<auto>}"
echo "[INFO] report_image_root=${REPORT_IMAGE_ROOT:-<auto>}"
echo "[INFO] run_stamp=${RUN_STAMP}"

# duplicate session name guard
if declare -A _tmp_assoc 2>/dev/null; then
    declare -A SESSION_NAME_COUNT=()
    HAS_ASSOC=1
else
    HAS_ASSOC=0
fi

declare -a SEQ_LAUNCH_LINES=()
declare -a SEQ_META_LINES=()

for i in "${!CHECKPOINTS[@]}"; do
    ckpt="${CHECKPOINTS[$i]}"
    gpu_ids="${GPU_GROUPS[$i]}"
    port="$((START_PORT + i))"

    if [[ "${ckpt}" == "final" ]]; then
        suffix="final"
        adapter_path="${ADAPTER_ROOT}/final"
    else
        suffix="ckpt${ckpt}"
        adapter_path="${ADAPTER_ROOT}/checkpoint-${ckpt}"
    fi

    if [[ ! -d "${adapter_path}" ]]; then
        echo "[ERROR] adapter path not found: ${adapter_path}" >&2
        exit 1
    fi

    tag="${EXP_NAME}${suffix}_noreason_nothink_b${BATCH_SIZE}"
    model_alias="Qwen/Qwen3.5-9B-${EXP_NAME}-${suffix}-merged"
    merged_path="$(dirname "${BASE_MODEL_PATH}")/$(basename "${BASE_MODEL_PATH}")-${EXP_NAME}-${suffix}-merged-full"

    session_base="eval_${tag}_${TEST_TAG}"
    if [[ "${HAS_ASSOC}" == "1" ]]; then
        seen="${SESSION_NAME_COUNT[${session_base}]:-0}"
        if [[ "${seen}" -eq 0 ]]; then
            session_name="${session_base}"
        else
            session_name="${session_base}_${seen}"
        fi
        SESSION_NAME_COUNT["${session_base}"]=$((seen + 1))
    else
        session_name="${session_base}_${i}"
    fi

    cmd=(
        bash "${WORKER_SCRIPT}"
        --base_model_path "${BASE_MODEL_PATH}"
        --adapter_path "${adapter_path}"
        --merged_path "${merged_path}"
        --model_alias "${model_alias}"
        --tag "${tag}"
        --test_data "${TEST_DATA}"
        --test_tag "${TEST_TAG}"
        --gpu_ids "${gpu_ids}"
        --port "${port}"
        --max_samples "${MAX_SAMPLES}"
        --batch_size "${BATCH_SIZE}"
        --max_new_tokens "${MAX_NEW_TOKENS}"
        --temperature "${TEMPERATURE}"
        --retries "${RETRIES}"
        --empty_retry_backoff "${EMPTY_RETRY_BACKOFF}"
        --api_timeout "${API_TIMEOUT}"
        --use_ocr "${USE_OCR}"
        --prompt_style "${PROMPT_STYLE}"
        --gt_source "${GT_SOURCE}"
        --normalize_empty_cells "${NORMALIZE_EMPTY_CELLS}"
        --empty_cell_token "${EMPTY_CELL_TOKEN}"
        --nested_teds_mode "${NESTED_TEDS_MODE}"
        --out_root "${OUT_ROOT}"
        --log_dir "${LOG_DIR}"
        --run_stamp "${RUN_STAMP}"
        --max_model_len "${MAX_MODEL_LEN}"
        --max_num_seqs "${MAX_NUM_SEQS}"
        --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}"
        --python_bin "${PYTHON_BIN}"
        --report_image_mode "${REPORT_IMAGE_MODE}"
        --report_shared_image_dir "${REPORT_SHARED_IMAGE_DIR}"
        --report_image_root "${REPORT_IMAGE_ROOT}"
    )
    if [[ "${FORCE_MERGE}" == "1" ]]; then
        cmd+=(--force_merge)
    fi
    if [[ "${SKIP_REPORT}" == "1" ]]; then
        cmd+=(--skip_report)
    fi

    cmd_str=""
    for token in "${cmd[@]}"; do
        cmd_str+="$(printf '%q' "${token}") "
    done
    cmd_str="${cmd_str% }"

    if [[ "${LAUNCH_MODE}" == "parallel" ]]; then
        if tmux has-session -t "${session_name}" 2>/dev/null; then
            echo "[WARN] tmux session already exists, replacing: ${session_name}"
            tmux kill-session -t "${session_name}"
        fi

        tmux new-session -d -s "${session_name}" "cd $(printf '%q' "${PROJECT_ROOT}") && ${cmd_str}"

        echo "[STARTED] session=${session_name} gpus=${gpu_ids} port=${port} ckpt=${ckpt}"
        echo "          adapter=${adapter_path}"
        echo "          merged=${merged_path}"
        echo "          alias=${model_alias}"
    else
        seq_desc="ckpt=${ckpt} gpus=${gpu_ids} port=${port}"
        printf -v launch_line 'launch_with_locks %q %q %q &' "${seq_desc}" "${gpu_ids}" "${cmd_str}"
        SEQ_LAUNCH_LINES+=("${launch_line}")
        SEQ_META_LINES+=("${seq_desc}")
    fi
done

echo ""
if [[ "${LAUNCH_MODE}" == "parallel" ]]; then
    echo "[INFO] tmux sessions started."
    echo "[INFO] list: tmux ls | grep '^eval_${EXP_NAME}'"
    echo "[INFO] attach: tmux attach -t <session_name>"
    echo "[INFO] logs: ${PROJECT_ROOT}/${LOG_DIR}"
else
    seq_session="eval_${EXP_NAME}_${TEST_TAG}_sequential"
    if tmux has-session -t "${seq_session}" 2>/dev/null; then
        echo "[WARN] tmux session already exists, replacing: ${seq_session}"
        tmux kill-session -t "${seq_session}"
    fi

    seq_runner_rel="${LOG_DIR}/run_${EXP_NAME}_${TEST_TAG}_${RUN_STAMP}_sequential.sh"
    seq_runner_abs="${PROJECT_ROOT}/${seq_runner_rel}"
    {
        echo '#!/usr/bin/env bash'
        echo 'set -euo pipefail'
        echo "LOCK_DIR=$(printf '%q' "${PROJECT_ROOT}/${LOG_DIR}/gpu_locks_${EXP_NAME}_${TEST_TAG}_${RUN_STAMP}")"
        cat <<'EOF'
mkdir -p "${LOCK_DIR}"
if ! command -v flock >/dev/null 2>&1; then
    echo "[ERROR] flock not found. Install util-linux first." >&2
    exit 1
fi

launch_with_locks() {
    local desc="$1"
    local gpu_spec="$2"
    local cmd_str="$3"

    local normalized="${gpu_spec//./,}"
    normalized="${normalized// /}"
    if [[ ! "${normalized}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
        echo "[ERROR] invalid gpu spec in sequential runner: ${gpu_spec}" >&2
        return 2
    fi

    local sorted_gpus
    sorted_gpus="$(tr ',' '\n' <<< "${normalized}" | sed '/^$/d' | sort -n -u)"

    local -a lock_fds=()
    local gpu
    while IFS= read -r gpu; do
        [[ -z "${gpu}" ]] && continue
        local lock_file="${LOCK_DIR}/gpu${gpu}.lock"
        local fd
        eval "exec {fd}>\"${lock_file}\""
        flock "${fd}"
        lock_fds+=("${fd}")
    done <<< "${sorted_gpus}"

    echo "[START] ${desc}"
    set +e
    bash -lc "${cmd_str}"
    local rc=$?
    set -e

    local i
    for ((i=${#lock_fds[@]}-1; i>=0; i--)); do
        eval "exec ${lock_fds[$i]}>&-"
    done

    if [[ "${rc}" -eq 0 ]]; then
        echo "[DONE] ${desc}"
    else
        echo "[FAIL] ${desc} rc=${rc}" >&2
    fi
    return "${rc}"
}

declare -a JOB_PIDS=()
EOF
        for line in "${SEQ_LAUNCH_LINES[@]}"; do
            echo "${line}"
            echo 'JOB_PIDS+=("$!")'
        done
        cat <<'EOF'

overall_status=0
for pid in "${JOB_PIDS[@]}"; do
    if ! wait "${pid}"; then
        overall_status=1
    fi
done
exit "${overall_status}"
EOF
    } > "${seq_runner_abs}"
    chmod +x "${seq_runner_abs}"

    tmux new-session -d -s "${seq_session}" "cd $(printf '%q' "${PROJECT_ROOT}") && bash $(printf '%q' "${seq_runner_rel}")"

    echo "[STARTED] session=${seq_session} mode=sequential(entries with duplicate GPUs are serialized)"
    for m in "${SEQ_META_LINES[@]}"; do
        echo "          ${m}"
    done
    echo "[INFO] attach: tmux attach -t ${seq_session}"
    echo "[INFO] script: ${seq_runner_abs}"
    echo "[INFO] logs: ${PROJECT_ROOT}/${LOG_DIR}"
fi
