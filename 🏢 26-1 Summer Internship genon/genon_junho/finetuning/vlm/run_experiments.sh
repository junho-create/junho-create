#!/bin/bash
# =============================================================================
# TSR 실험 마스터 스크립트
# 3가지 학습 방법 × 2가지 OCR 설정 = 6가지 실험
#
# Usage:
#   PHASE=all bash run_experiments.sh        # 전체 실행
#   PHASE=prepare bash run_experiments.sh    # Step 0: AIHub 압축 해제
#   PHASE=shared_data bash run_experiments.sh # Step 1: 공유 데이터 생성
#   PHASE=e1 bash run_experiments.sh         # Step 2: E1 (SFT + OCR on)
#   PHASE=e2 bash run_experiments.sh         # Step 3: E2 (SFT + OCR off)
#   PHASE=e3 bash run_experiments.sh         # Step 4: E3 (Response Distill + OCR on)
#   PHASE=e4 bash run_experiments.sh         # Step 5: E4 (Response Distill + OCR off)
#   PHASE=e5 bash run_experiments.sh         # Step 6: E5 (Logit KD + OCR on)
#   PHASE=e6 bash run_experiments.sh         # Step 7: E6 (Logit KD + OCR off)
#   PHASE=baseline bash run_experiments.sh   # Step 8: 베이스라인 평가
#   PHASE=aggregate bash run_experiments.sh  # Step 9: 결과 집계
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PHASE=${PHASE:-"all"}
NUM_GPUS=${NUM_GPUS:-4}
MASTER_PORT=${MASTER_PORT:-29510}
SEED=${SEED:-42}
SAMPLE_COUNT=${SAMPLE_COUNT:-3000}
SAMPLE_SELECTION_MODE=${SAMPLE_SELECTION_MODE:-"full_analysis"}
SAMPLE_MIN_COMPLEX_RATIO=${SAMPLE_MIN_COMPLEX_RATIO:-0.30}
SAMPLE_HARD_FIRST=${SAMPLE_HARD_FIRST:-"true"}

# AIHub 경로
AIHUB_TRAIN_DIR=${AIHUB_TRAIN_DIR:-"../aihub_table_data/Training"}
AIHUB_EXTRACT_DIR=${AIHUB_EXTRACT_DIR:-"./data/extracted/aihub"}
EXTRACT_MAX_ZIPS=${EXTRACT_MAX_ZIPS:-}

export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-"false"}
export WANDB_MODE=${WANDB_MODE:-"online"}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"expandable_segments:True"}

# vLLM spawn 강제
if [ -z "${VLLM_WORKER_MULTIPROC_METHOD:-}" ]; then
    export VLLM_WORKER_MULTIPROC_METHOD="spawn"
fi

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && [ "${NUM_GPUS}" -gt 0 ]; then
    CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((NUM_GPUS - 1)))"
    export CUDA_VISIBLE_DEVICES
fi

# Config 경로
CFG_E1="config/exp_sft_ocr_on.yaml"
CFG_E2="config/exp_sft_ocr_off.yaml"
CFG_E3="config/exp_resp_ocr_on.yaml"
CFG_E4="config/exp_resp_ocr_off.yaml"
CFG_E5="config/exp_logit_ocr_on.yaml"
CFG_E6="config/exp_logit_ocr_off.yaml"

# 데이터 경로
SHARED_DIR="./data/experiments/shared"
SHARED_SAMPLED="${SHARED_DIR}/sampled_${SAMPLE_COUNT}.jsonl"
SHARED_TRAIN="${SHARED_DIR}/train.jsonl"
SHARED_EVAL="${SHARED_DIR}/eval.jsonl"
SHARED_TRAIN_OCR_OFF="${SHARED_DIR}/train_ocr_off.jsonl"
SHARED_EVAL_OCR_OFF="${SHARED_DIR}/eval_ocr_off.jsonl"
EVAL_OCR_ON="${SHARED_DIR}/eval.jsonl"
EVAL_OCR_OFF="${SHARED_DIR}/eval_ocr_off.jsonl"

# 기존 생성 데이터 재사용 경로
EXISTING_SAMPLED_JSONL=${EXISTING_SAMPLED_JSONL:-"./data/processed/aihub_training_sampled_ocr_${SAMPLE_COUNT}.jsonl"}
EXISTING_TRAIN_JSONL=${EXISTING_TRAIN_JSONL:-"./data/processed/train.jsonl"}
EXISTING_EVAL_JSONL=${EXISTING_EVAL_JSONL:-"./data/processed/eval.jsonl"}
LEGACY_E3_DIR=${LEGACY_E3_DIR:-"./data/distill/modes/output_lora_3000"}
LEGACY_E4_DIR=${LEGACY_E4_DIR:-"./data/distill/modes/output_lora_3000_ocr_off"}

# vLLM 평가 기본값
EVAL_BACKEND=${EVAL_BACKEND:-"vllm"}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-16}
EVAL_MAX_NEW_TOKENS=${EVAL_MAX_NEW_TOKENS:-1024}
EVAL_TEMPERATURE=${EVAL_TEMPERATURE:-0.0}
EVAL_GPU_MEMORY_UTILIZATION=${EVAL_GPU_MEMORY_UTILIZATION:-0.9}
EVAL_MAX_MODEL_LEN=${EVAL_MAX_MODEL_LEN:-8192}
EVAL_MAX_PIXELS=${EVAL_MAX_PIXELS:-1048576}
EVAL_MIN_PIXELS=${EVAL_MIN_PIXELS:-262144}
EVAL_TP_SIZE=${EVAL_TP_SIZE:-"${NUM_GPUS}"}

log_step() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  $1"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

line_count() {
    local path="$1"
    if [ ! -f "${path}" ]; then
        echo 0
        return 0
    fi
    wc -l < "${path}" | tr -d ' '
}

file_has_lines() {
    local path="$1"
    local min_lines="${2:-1}"
    [ -f "${path}" ] || return 1
    local lines
    lines="$(line_count "${path}")"
    [ "${lines:-0}" -ge "${min_lines}" ]
}

copy_if_missing() {
    local src="$1"
    local dst="$2"
    local tag="${3:-data}"
    if [ -f "${src}" ] && [ ! -f "${dst}" ]; then
        mkdir -p "$(dirname "${dst}")"
        cp -v "${src}" "${dst}"
        echo "  [reuse:${tag}] ${src} -> ${dst}"
    fi
}

bootstrap_experiment_data() {
    local exp_dir="$1"
    shift
    mkdir -p "${exp_dir}"
    for src_dir in "$@"; do
        [ -d "${src_dir}" ] || continue
        for f in synthetic_raw.jsonl synthetic_filtered.jsonl train.jsonl eval.jsonl; do
            copy_if_missing "${src_dir}/${f}" "${exp_dir}/${f}" "legacy"
        done
    done
}

run_eval_vllm() {
    local model="$1"
    local test_data="$2"
    local output_dir="$3"
    local base_model="${4:-}"

    EVAL_CMD=(
        bash eval/run_eval.sh
        --backend "${EVAL_BACKEND}"
        --model "${model}"
        --test_data "${test_data}"
        --output_dir "${output_dir}"
        --batch_size "${EVAL_BATCH_SIZE}"
        --max_new_tokens "${EVAL_MAX_NEW_TOKENS}"
        --temperature "${EVAL_TEMPERATURE}"
        --gpu_memory_utilization "${EVAL_GPU_MEMORY_UTILIZATION}"
        --max_model_len "${EVAL_MAX_MODEL_LEN}"
        --max_pixels "${EVAL_MAX_PIXELS}"
        --min_pixels "${EVAL_MIN_PIXELS}"
        --tp_size "${EVAL_TP_SIZE}"
    )
    if [ -n "${base_model}" ]; then
        EVAL_CMD+=(--base_model "${base_model}")
    fi

    "${EVAL_CMD[@]}"
}

run_logit_with_retry() {
    local cfg="$1"
    local label="$2"

    echo "  [${label}] Logit distillation (attempt 1: default)..."
    if PHASE=logit CONFIG="${cfg}" NUM_GPUS="${NUM_GPUS}" \
        bash distill/run_distill.sh; then
        return 0
    fi

    echo "  [${label}] Warning: logit distillation failed on default settings."
    echo "  [${label}] Retrying with VLLM_DISABLE_CUSTOM_ALL_REDUCE=1..."
    if VLLM_DISABLE_CUSTOM_ALL_REDUCE=1 PHASE=logit CONFIG="${cfg}" NUM_GPUS="${NUM_GPUS}" \
        bash distill/run_distill.sh; then
        return 0
    fi

    echo "  [${label}] ERROR: logit distillation failed even after retry."
    return 1
}

# =============================================================================
# Step 0: AIHub 압축 해제
# =============================================================================
step_prepare() {
    log_step "Step 0: AIHub 압축 해제"

    PREP_ARGS=(
        --input "${AIHUB_TRAIN_DIR}"
        --output_dir "${AIHUB_EXTRACT_DIR}"
    )
    if [ -n "${EXTRACT_MAX_ZIPS:-}" ]; then
        PREP_ARGS+=(--max_zips "${EXTRACT_MAX_ZIPS}")
    fi
    python -m data.extract_aihub "${PREP_ARGS[@]}"
    echo "  ✓ AIHub extraction complete"
}

# =============================================================================
# Step 1: 공유 데이터 생성
# =============================================================================
step_shared_data() {
    log_step "Step 1: 공유 데이터 생성 (${SAMPLE_COUNT}개, seed=${SEED})"

    mkdir -p "${SHARED_DIR}"

    # 1a. OCR 포함 샘플링 (기존 3000 샘플 재사용 우선)
    if [ ! -f "${SHARED_SAMPLED}" ]; then
        if file_has_lines "${EXISTING_SAMPLED_JSONL}" "${SAMPLE_COUNT}"; then
            copy_if_missing "${EXISTING_SAMPLED_JSONL}" "${SHARED_SAMPLED}" "sampled"
        fi
    fi
    if [ ! -f "${SHARED_SAMPLED}" ]; then
        echo "  [1a] Sampling ${SAMPLE_COUNT} records with OCR..."
        SAMPLE_HARD_FIRST_FLAG=()
        if [ "${SAMPLE_HARD_FIRST}" = "true" ]; then
            SAMPLE_HARD_FIRST_FLAG+=(--hard_first)
        else
            SAMPLE_HARD_FIRST_FLAG+=(--no-hard_first)
        fi
        python -m data.sample_aihub_with_ocr \
            --input_dir "${AIHUB_EXTRACT_DIR}/Training" \
            --output "${SHARED_SAMPLED}" \
            --sample_count "${SAMPLE_COUNT}" \
            --min_span_ratio 0.70 \
            --min_complex_ratio "${SAMPLE_MIN_COMPLEX_RATIO}" \
            --selection_mode "${SAMPLE_SELECTION_MODE}" \
            "${SAMPLE_HARD_FIRST_FLAG[@]}" \
            --prompt_style chandra_table_with_ocr \
            --bbox_scale 1024 \
            --seed "${SEED}"
    else
        echo "  [1a] Skipped: ${SHARED_SAMPLED} already exists"
    fi

    # 1b. Train/Eval 분리 (기존 split 재사용 우선)
    if [ ! -f "${SHARED_TRAIN}" ] || [ ! -f "${SHARED_EVAL}" ]; then
        copy_if_missing "${EXISTING_TRAIN_JSONL}" "${SHARED_TRAIN}" "train"
        copy_if_missing "${EXISTING_EVAL_JSONL}" "${SHARED_EVAL}" "eval"
    fi
    if [ ! -f "${SHARED_TRAIN}" ] || [ ! -f "${SHARED_EVAL}" ]; then
        echo "  [1b] Splitting train/eval (eval_ratio=0.1)..."
        python -m data.split_dataset \
            --input "${SHARED_SAMPLED}" \
            --output_dir "${SHARED_DIR}/" \
            --eval_ratio 0.1 \
            --seed "${SEED}"
    else
        echo "  [1b] Skipped: ${SHARED_TRAIN}, ${SHARED_EVAL} already exist"
    fi

    # 1c. OCR-off 버전 생성
    if [ ! -f "${SHARED_TRAIN_OCR_OFF}" ]; then
        echo "  [1c] Converting train to OCR-off..."
        python -m data.convert_prompt_style \
            --input "${SHARED_TRAIN}" \
            --output "${SHARED_TRAIN_OCR_OFF}" \
            --prompt_style chandra_table_without_ocr
    else
        echo "  [1c] Skipped: ${SHARED_TRAIN_OCR_OFF} already exists"
    fi

    if [ ! -f "${SHARED_EVAL_OCR_OFF}" ]; then
        echo "  [1d] Converting eval to OCR-off..."
        python -m data.convert_prompt_style \
            --input "${SHARED_EVAL}" \
            --output "${SHARED_EVAL_OCR_OFF}" \
            --prompt_style chandra_table_without_ocr
    else
        echo "  [1d] Skipped: ${SHARED_EVAL_OCR_OFF} already exists"
    fi

    echo "  ✓ Shared data ready"
}

# =============================================================================
# E1: SFT + OCR on (Teacher 불필요)
# =============================================================================
step_e1() {
    log_step "Step 2: E1 - SFT + OCR on"
    if [ -d "./output/e1_sft_ocr_on/student_sft/final" ]; then
        echo "  Skipped: ./output/e1_sft_ocr_on/student_sft/final already exists"
        return 0
    fi
    PHASE=sft CONFIG="${CFG_E1}" NUM_GPUS="${NUM_GPUS}" MASTER_PORT="${MASTER_PORT}" \
        bash distill/run_distill.sh
    echo "  ✓ E1 SFT complete"
}

# =============================================================================
# E2: SFT + OCR off (Teacher 불필요)
# =============================================================================
step_e2() {
    log_step "Step 3: E2 - SFT + OCR off"
    if [ -d "./output/e2_sft_ocr_off/student_sft/final" ]; then
        echo "  Skipped: ./output/e2_sft_ocr_off/student_sft/final already exists"
        return 0
    fi
    PHASE=sft CONFIG="${CFG_E2}" NUM_GPUS="${NUM_GPUS}" MASTER_PORT="${MASTER_PORT}" \
        bash distill/run_distill.sh
    echo "  ✓ E2 SFT complete"
}

# =============================================================================
# E3: Response Distill + OCR on (generate → filter → sft)
# =============================================================================
step_e3() {
    log_step "Step 4: E3 - Response Distill + OCR on"
    local e3_data="./data/experiments/e3_resp_ocr_on"
    local e3_sft_final="./output/e3_resp_ocr_on/student_sft/final"

    bootstrap_experiment_data "${e3_data}" "${LEGACY_E3_DIR}" "./data/distill/modes/output_lora_3000"

    if file_has_lines "${e3_data}/synthetic_raw.jsonl" "${SAMPLE_COUNT}"; then
        echo "  [4a] Skipped: synthetic_raw.jsonl already has >= ${SAMPLE_COUNT} samples"
    else
        echo "  [4a] Teacher generate..."
        PHASE=generate CONFIG="${CFG_E3}" NUM_GPUS="${NUM_GPUS}" \
            bash distill/run_distill.sh
    fi

    if file_has_lines "${e3_data}/synthetic_filtered.jsonl" 1 && \
       file_has_lines "${e3_data}/train.jsonl" 1 && \
       file_has_lines "${e3_data}/eval.jsonl" 1; then
        echo "  [4b] Skipped: filtered/train/eval already exist"
    else
        echo "  [4b] Quality filter..."
        PHASE=filter CONFIG="${CFG_E3}" \
            bash distill/run_distill.sh
    fi

    if [ -d "${e3_sft_final}" ]; then
        echo "  [4c] Skipped: ${e3_sft_final} already exists"
    else
        echo "  [4c] Student SFT..."
        PHASE=sft CONFIG="${CFG_E3}" NUM_GPUS="${NUM_GPUS}" MASTER_PORT="${MASTER_PORT}" \
            bash distill/run_distill.sh
    fi

    echo "  ✓ E3 Response Distill complete"
}

# =============================================================================
# E4: Response Distill + OCR off (generate → filter → sft)
# =============================================================================
step_e4() {
    log_step "Step 5: E4 - Response Distill + OCR off"
    local e4_data="./data/experiments/e4_resp_ocr_off"
    local e4_sft_final="./output/e4_resp_ocr_off/student_sft/final"

    bootstrap_experiment_data "${e4_data}" "${LEGACY_E4_DIR}" "./data/distill/modes/output_lora_3000_ocr_off"

    if file_has_lines "${e4_data}/synthetic_raw.jsonl" "${SAMPLE_COUNT}"; then
        echo "  [5a] Skipped: synthetic_raw.jsonl already has >= ${SAMPLE_COUNT} samples"
    else
        echo "  [5a] Teacher generate..."
        PHASE=generate CONFIG="${CFG_E4}" NUM_GPUS="${NUM_GPUS}" \
            bash distill/run_distill.sh
    fi

    if file_has_lines "${e4_data}/synthetic_filtered.jsonl" 1 && \
       file_has_lines "${e4_data}/train.jsonl" 1 && \
       file_has_lines "${e4_data}/eval.jsonl" 1; then
        echo "  [5b] Skipped: filtered/train/eval already exist"
    else
        echo "  [5b] Quality filter..."
        PHASE=filter CONFIG="${CFG_E4}" \
            bash distill/run_distill.sh
    fi

    if [ -d "${e4_sft_final}" ]; then
        echo "  [5c] Skipped: ${e4_sft_final} already exists"
    else
        echo "  [5c] Student SFT..."
        PHASE=sft CONFIG="${CFG_E4}" NUM_GPUS="${NUM_GPUS}" MASTER_PORT="${MASTER_PORT}" \
            bash distill/run_distill.sh
    fi

    echo "  ✓ E4 Response Distill complete"
}

# =============================================================================
# E5: Logit KD + OCR on (E3 데이터/SFT 복사 → logit distill)
# =============================================================================
step_e5() {
    log_step "Step 6: E5 - Logit KD + OCR on (based on E3)"

    E3_DATA="./data/experiments/e3_resp_ocr_on"
    E5_DATA="./data/experiments/e5_logit_ocr_on"
    E3_SFT="./output/e3_resp_ocr_on/student_sft"
    E5_SFT="./output/e5_logit_ocr_on/student_sft"
    E5_LOGIT_FINAL="./output/e5_logit_ocr_on/student_logit_distill/final"

    if [ -d "${E5_LOGIT_FINAL}" ]; then
        echo "  Skipped: ${E5_LOGIT_FINAL} already exists"
        return 0
    fi

    # E3의 데이터와 SFT 체크포인트를 E5로 복사
    echo "  [6a] Copying E3 data and SFT checkpoint to E5..."
    mkdir -p "${E5_DATA}"
    for f in train.jsonl eval.jsonl synthetic_filtered.jsonl; do
        if [ -f "${E3_DATA}/${f}" ] && [ ! -f "${E5_DATA}/${f}" ]; then
            cp -v "${E3_DATA}/${f}" "${E5_DATA}/${f}"
        fi
    done

    mkdir -p "${E5_SFT}"
    if [ ! -d "${E5_SFT}/final" ] && [ -d "${E3_SFT}/final" ]; then
        cp -rv "${E3_SFT}/final" "${E5_SFT}/final"
    elif [ ! -d "${E5_SFT}/final" ]; then
        echo "  ERROR: E3 SFT checkpoint not found at ${E3_SFT}/final"
        echo "  Run E3 first: PHASE=e3 bash run_experiments.sh"
        exit 1
    else
        echo "  [6a] Reusing existing E5 student_sft/final"
    fi
    for f in train.jsonl eval.jsonl synthetic_filtered.jsonl; do
        if [ ! -f "${E5_DATA}/${f}" ]; then
            echo "  ERROR: missing ${E5_DATA}/${f} (run E3 first)"
            exit 1
        fi
    done

    # Logit distillation (save_logits + distill)
    run_logit_with_retry "${CFG_E5}" "6b"

    echo "  ✓ E5 Logit KD complete"
}

# =============================================================================
# E6: Logit KD + OCR off (E4 데이터/SFT 복사 → logit distill)
# =============================================================================
step_e6() {
    log_step "Step 7: E6 - Logit KD + OCR off (based on E4)"

    E4_DATA="./data/experiments/e4_resp_ocr_off"
    E6_DATA="./data/experiments/e6_logit_ocr_off"
    E4_SFT="./output/e4_resp_ocr_off/student_sft"
    E6_SFT="./output/e6_logit_ocr_off/student_sft"
    E6_LOGIT_FINAL="./output/e6_logit_ocr_off/student_logit_distill/final"

    if [ -d "${E6_LOGIT_FINAL}" ]; then
        echo "  Skipped: ${E6_LOGIT_FINAL} already exists"
        return 0
    fi

    # E4의 데이터와 SFT 체크포인트를 E6으로 복사
    echo "  [7a] Copying E4 data and SFT checkpoint to E6..."
    mkdir -p "${E6_DATA}"
    for f in train.jsonl eval.jsonl synthetic_filtered.jsonl; do
        if [ -f "${E4_DATA}/${f}" ] && [ ! -f "${E6_DATA}/${f}" ]; then
            cp -v "${E4_DATA}/${f}" "${E6_DATA}/${f}"
        fi
    done

    mkdir -p "${E6_SFT}"
    if [ ! -d "${E6_SFT}/final" ] && [ -d "${E4_SFT}/final" ]; then
        cp -rv "${E4_SFT}/final" "${E6_SFT}/final"
    elif [ ! -d "${E6_SFT}/final" ]; then
        echo "  ERROR: E4 SFT checkpoint not found at ${E4_SFT}/final"
        echo "  Run E4 first: PHASE=e4 bash run_experiments.sh"
        exit 1
    else
        echo "  [7a] Reusing existing E6 student_sft/final"
    fi
    for f in train.jsonl eval.jsonl synthetic_filtered.jsonl; do
        if [ ! -f "${E6_DATA}/${f}" ]; then
            echo "  ERROR: missing ${E6_DATA}/${f} (run E4 first)"
            exit 1
        fi
    done

    # Logit distillation (save_logits + distill)
    run_logit_with_retry "${CFG_E6}" "7b"

    echo "  ✓ E6 Logit KD complete"
}

# =============================================================================
# 베이스라인 평가
# =============================================================================
step_baseline() {
    log_step "Step 8: 베이스라인 평가"

    # B2: Student 원본 (OCR on)
    echo "  [8a] B2: Student zero-shot + OCR on..."
    run_eval_vllm \
        "Qwen/Qwen3-VL-8B-Instruct" \
        "${EVAL_OCR_ON}" \
        "eval_results/baseline_student_ocr_on/"

    # B2: Student 원본 (OCR off)
    echo "  [8b] B2: Student zero-shot + OCR off..."
    run_eval_vllm \
        "Qwen/Qwen3-VL-8B-Instruct" \
        "${EVAL_OCR_OFF}" \
        "eval_results/baseline_student_ocr_off/"

    # B1: Teacher (OCR on) - vLLM 필요
    echo "  [8c] B1: Teacher zero-shot + OCR on..."
    run_eval_vllm \
        "Qwen/Qwen3-VL-235B-A22B-Instruct-FP8" \
        "${EVAL_OCR_ON}" \
        "eval_results/baseline_teacher_ocr_on/"

    echo "  ✓ Baseline evaluation complete"
}

# =============================================================================
# 개별 실험 평가 헬퍼
# =============================================================================
eval_experiment() {
    local exp_id="$1"
    local config="$2"
    local eval_data="$3"
    local model_dir="$4"

    log_step "Evaluating: ${exp_id}"

    # logit distill 결과가 있으면 우선 사용
    local logit_final="${model_dir%/student_sft}/../student_logit_distill/final"
    local ckpt=""
    if [ -d "${logit_final}" ]; then
        ckpt="${logit_final}"
    elif [ -d "${model_dir}/final" ]; then
        ckpt="${model_dir}/final"
    else
        ckpt="$(ls -td "${model_dir}"/checkpoint-* 2>/dev/null | head -1 || true)"
    fi

    if [ -z "${ckpt}" ]; then
        echo "  WARNING: No checkpoint found for ${exp_id}, skipping"
        return 0
    fi

    echo "  Model: ${ckpt}"
    run_eval_vllm \
        "${ckpt}" \
        "${eval_data}" \
        "eval_results/${exp_id}/"
}

# =============================================================================
# 전체 실험 평가
# =============================================================================
step_eval_all() {
    log_step "Evaluating all experiments"

    eval_experiment "e1_sft_ocr_on" "${CFG_E1}" "${EVAL_OCR_ON}" \
        "./output/e1_sft_ocr_on/student_sft"
    eval_experiment "e2_sft_ocr_off" "${CFG_E2}" "${EVAL_OCR_OFF}" \
        "./output/e2_sft_ocr_off/student_sft"
    eval_experiment "e3_resp_ocr_on" "${CFG_E3}" "${EVAL_OCR_ON}" \
        "./output/e3_resp_ocr_on/student_sft"
    eval_experiment "e4_resp_ocr_off" "${CFG_E4}" "${EVAL_OCR_OFF}" \
        "./output/e4_resp_ocr_off/student_sft"
    eval_experiment "e5_logit_ocr_on" "${CFG_E5}" "${EVAL_OCR_ON}" \
        "./output/e5_logit_ocr_on/student_sft"
    eval_experiment "e6_logit_ocr_off" "${CFG_E6}" "${EVAL_OCR_OFF}" \
        "./output/e6_logit_ocr_off/student_sft"

    echo "  ✓ All experiments evaluated"
}

# =============================================================================
# 결과 집계
# =============================================================================
step_aggregate() {
    log_step "Step 9: 결과 집계"
    python -m eval.aggregate_results \
        --results_dir eval_results/ \
        --output eval_results/comparison_table.json
    echo "  ✓ Aggregation complete"
}

# =============================================================================
# 메인 디스패치
# =============================================================================

echo "=============================================="
echo "  TSR Experiment Pipeline"
echo "=============================================="
echo "  Phase:   ${PHASE}"
echo "  GPUs:    ${NUM_GPUS}"
echo "  Seed:    ${SEED}"
echo "  Samples: ${SAMPLE_COUNT}"
echo "=============================================="

case "${PHASE}" in
    all)
        step_prepare
        step_shared_data
        step_e1
        step_e2
        step_e3
        step_e4
        step_e5
        step_e6
        step_baseline
        step_eval_all
        step_aggregate
        ;;
    prepare)
        step_prepare
        ;;
    shared_data)
        step_shared_data
        ;;
    e1)
        step_e1
        ;;
    e2)
        step_e2
        ;;
    e3)
        step_e3
        ;;
    e4)
        step_e4
        ;;
    e5)
        step_e5
        ;;
    e6)
        step_e6
        ;;
    baseline)
        step_baseline
        ;;
    eval_all)
        step_eval_all
        ;;
    aggregate)
        step_aggregate
        ;;
    *)
        echo "Unsupported PHASE: ${PHASE}"
        echo ""
        echo "Supported phases:"
        echo "  all         - 전체 파이프라인 실행"
        echo "  prepare     - Step 0: AIHub 압축 해제"
        echo "  shared_data - Step 1: 공유 데이터 생성"
        echo "  e1          - Step 2: E1 (SFT + OCR on)"
        echo "  e2          - Step 3: E2 (SFT + OCR off)"
        echo "  e3          - Step 4: E3 (Response Distill + OCR on)"
        echo "  e4          - Step 5: E4 (Response Distill + OCR off)"
        echo "  e5          - Step 6: E5 (Logit KD + OCR on, requires E3)"
        echo "  e6          - Step 7: E6 (Logit KD + OCR off, requires E4)"
        echo "  baseline    - Step 8: 베이스라인 평가 (B1, B2)"
        echo "  eval_all    - 전체 실험 평가 (E1~E6)"
        echo "  aggregate   - Step 9: 결과 집계 및 비교 표"
        exit 1
        ;;
esac

echo ""
echo "=============================================="
echo "  Phase '${PHASE}' Complete!"
echo "=============================================="
