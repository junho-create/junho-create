#!/bin/bash
# =============================================================================
# Knowledge Distillation Pipeline
# Teacher (Large) -> Student (Small) 전체 파이프라인
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

CONFIG=${CONFIG:-"config/distill_config.yaml"}
PHASE=${PHASE:-"all"}        # all / prepare / generate / filter / sft / logit / feature / eval
NUM_GPUS=${NUM_GPUS:-4}
MASTER_PORT=${MASTER_PORT:-29510}

# prepare phase options (shared extraction path)
AIHUB_TRAIN_DIR=${AIHUB_TRAIN_DIR:-"../aihub_table_data/Training"}
AIHUB_EXTRACT_DIR=${AIHUB_EXTRACT_DIR:-"./data/extracted/aihub"}
EXTRACT_MAX_ZIPS=${EXTRACT_MAX_ZIPS:-}
MAX_IMAGES=${MAX_IMAGES:-}
MAX_SAMPLES=${MAX_SAMPLES:-}

export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-"false"}
export WANDB_MODE=${WANDB_MODE:-"online"}
export WANDB_PROJECT=${WANDB_PROJECT:-"tsr_vlm_train"}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"expandable_segments:True"}

# Torch + CUDA13 wheel 조합에서 NVRTC builtins를 찾지 못하는 경우를 방지한다.
configure_cuda_wheel_libpath() {
    local python_bin
    local lib_paths

    python_bin="${PYTHON_BIN:-python}"
    if ! command -v "${python_bin}" >/dev/null 2>&1; then
        return 0
    fi

    lib_paths="$("${python_bin}" - <<'PY'
import os
import site

candidates = []
for base in site.getsitepackages():
    for rel in (
        "nvidia/cu13/lib",
        "nvidia/cuda_nvrtc/lib",
        "nvidia/cuda_runtime/lib",
        "nvidia/cublas/lib",
        "nvidia/cudnn/lib",
    ):
        path = os.path.join(base, rel)
        if os.path.isdir(path):
            candidates.append(path)

print(":".join(candidates))
PY
)"

    if [ -n "${lib_paths}" ]; then
        export LD_LIBRARY_PATH="${lib_paths}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
        echo "  LD_LIBRARY_PATH(cuda wheels): ${lib_paths}"
    fi
}

configure_cuda_wheel_libpath

# vLLM + CUDA worker에서는 fork 방식이 실패할 수 있으므로 spawn을 기본/강제한다.
if [ "${VLLM_WORKER_MULTIPROC_METHOD:-}" = "fork" ]; then
    echo "  Warning: VLLM_WORKER_MULTIPROC_METHOD=fork is incompatible with CUDA workers. Forcing spawn."
    export VLLM_WORKER_MULTIPROC_METHOD="spawn"
elif [ -z "${VLLM_WORKER_MULTIPROC_METHOD:-}" ]; then
    export VLLM_WORKER_MULTIPROC_METHOD="spawn"
fi

# vLLM/torch subprocess 환경에서 GPU 가시성을 명확히 하기 위해
# CUDA_VISIBLE_DEVICES가 비어 있으면 NUM_GPUS 기준으로 기본 설정한다.
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && [ "${NUM_GPUS}" -gt 0 ]; then
    CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((NUM_GPUS - 1)))"
    export CUDA_VISIBLE_DEVICES
fi

config_bool() {
    local config_path="$1"
    local key_path="$2"
    python - "$config_path" "$key_path" <<'PY'
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1], "r"))
path = sys.argv[2].split(".")
cur = cfg
for key in path:
    if not isinstance(cur, dict) or key not in cur:
        print("false")
        raise SystemExit(0)
    cur = cur[key]
print("true" if bool(cur) else "false")
PY
}

config_value() {
    local config_path="$1"
    local key_path="$2"
    local fallback="$3"
    python - "$config_path" "$key_path" "$fallback" <<'PY'
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1], "r"))
path = sys.argv[2].split(".")
fallback = sys.argv[3]
cur = cfg
for key in path:
    if not isinstance(cur, dict) or key not in cur:
        print(fallback)
        raise SystemExit(0)
    cur = cur[key]
print(cur if cur is not None else fallback)
PY
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

run_student_sft() {
    SFT_ARGS=(--config "${CONFIG}")
    if [ -n "${MAX_SAMPLES}" ]; then
        SFT_ARGS+=(--max_train_samples "${MAX_SAMPLES}")
    fi

    if [ "${NUM_GPUS}" -gt 1 ]; then
        torchrun --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT}" \
            -m distill.student_sft \
            "${SFT_ARGS[@]}"
    else
        python -m distill.student_sft "${SFT_ARGS[@]}"
    fi
}

LOGIT_ENABLED="$(config_bool "${CONFIG}" "logit_distillation.enabled")"
FEATURE_ENABLED="$(config_bool "${CONFIG}" "feature_distillation.enabled")"
EVAL_DATA="$(config_value "${CONFIG}" "student_sft.eval_file" "data/processed/eval.jsonl")"
EVAL_BACKEND=${EVAL_BACKEND:-"vllm"}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-16}
EVAL_MAX_NEW_TOKENS=${EVAL_MAX_NEW_TOKENS:-1024}
EVAL_TEMPERATURE=${EVAL_TEMPERATURE:-0.0}
EVAL_TP_SIZE=${EVAL_TP_SIZE:-"${NUM_GPUS}"}
EVAL_GPU_MEMORY_UTILIZATION=${EVAL_GPU_MEMORY_UTILIZATION:-0.9}
EVAL_MAX_MODEL_LEN=${EVAL_MAX_MODEL_LEN:-8192}
EVAL_MAX_PIXELS=${EVAL_MAX_PIXELS:-1048576}
EVAL_MIN_PIXELS=${EVAL_MIN_PIXELS:-262144}

add_phase() {
    PLAN_KEYS+=("$1")
    PLAN_LABELS+=("$2")
}

run_phase() {
    local phase_label="$1"
    local phase_func="$2"

    CURRENT_PHASE=$((CURRENT_PHASE + 1))
    local percent=$((CURRENT_PHASE * 100 / TOTAL_PHASES))
    local started_at
    started_at="$(date +%s)"

    echo ""
    echo "━━━ [${CURRENT_PHASE}/${TOTAL_PHASES}] (${percent}%) ${phase_label} ━━━"
    "${phase_func}"

    local ended_at
    ended_at="$(date +%s)"
    local elapsed=$((ended_at - started_at))
    echo "  ✓ ${phase_label} complete (${elapsed}s)"
}

phase_prepare() {
    PREP_ARGS=(
        --input "${AIHUB_TRAIN_DIR}"
        --output_dir "${AIHUB_EXTRACT_DIR}"
    )
    if [ -n "${EXTRACT_MAX_ZIPS}" ]; then
        PREP_ARGS+=(--max_zips "${EXTRACT_MAX_ZIPS}")
    fi
    python -m data.extract_aihub "${PREP_ARGS[@]}"
}

phase_generate() {
    local gen_output
    local gen_target
    gen_output="$(config_value "${CONFIG}" "synthetic_generation.output_path" "")"
    gen_target="$(config_value "${CONFIG}" "synthetic_generation.target_samples" "0")"
    if [[ "${gen_target}" =~ ^[0-9]+$ ]] && [ "${gen_target}" -gt 0 ] && \
       file_has_lines "${gen_output}" "${gen_target}"; then
        echo "  Skipped generate: ${gen_output} already has >= ${gen_target} samples"
        return 0
    fi

    GENERATE_ARGS=(--config "${CONFIG}")
    if [ -n "${MAX_IMAGES}" ]; then
        GENERATE_ARGS+=(--max_images "${MAX_IMAGES}")
    fi
    python -m distill.teacher_generate "${GENERATE_ARGS[@]}"
}

phase_filter() {
    local filter_output
    local filter_dir
    filter_output="$(config_value "${CONFIG}" "quality_filter.output_path" "")"
    filter_dir="$(dirname "${filter_output:-.}")"
    if file_has_lines "${filter_output}" 1 && \
       file_has_lines "${filter_dir}/train.jsonl" 1 && \
       file_has_lines "${filter_dir}/eval.jsonl" 1; then
        echo "  Skipped filter: ${filter_output}, train.jsonl, eval.jsonl already exist"
        return 0
    fi

    python -m distill.quality_filter --config "${CONFIG}"
}

phase_sft() {
    local sft_output_dir
    sft_output_dir="$(config_value "${CONFIG}" "student_sft.output_dir" "")"
    if [ -n "${sft_output_dir}" ] && [ -d "${sft_output_dir}/final" ]; then
        echo "  Skipped SFT: ${sft_output_dir}/final already exists"
        return 0
    fi

    run_student_sft
}

phase_logit() {
    local teacher_logits_dir
    local logit_output_dir
    teacher_logits_dir="$(config_value "${CONFIG}" "logit_distillation.teacher_logits_dir" "")"
    logit_output_dir="$(config_value "${CONFIG}" "logit_distillation.output_dir" "")"

    if [ -n "${logit_output_dir}" ] && [ -d "${logit_output_dir}/final" ]; then
        echo "  Skipped logit distillation: ${logit_output_dir}/final already exists"
        return 0
    fi

    if [ -n "${teacher_logits_dir}" ] && [ -f "${teacher_logits_dir}/metadata.json" ] && \
       ls "${teacher_logits_dir}"/logits_*.npz >/dev/null 2>&1; then
        echo "  [4a] Skipped: precomputed teacher logits found at ${teacher_logits_dir}"
    else
        echo "  [4a] Pre-computing teacher logits..."
        LOGIT_ARGS=(--config "${CONFIG}")
        if [ -n "${MAX_SAMPLES}" ]; then
            LOGIT_ARGS+=(--max_samples "${MAX_SAMPLES}")
        fi
        python -m distill.save_teacher_logits "${LOGIT_ARGS[@]}"
    fi

    if [ "${NUM_GPUS}" -gt 1 ]; then
        echo "  [4b] Training with logit distillation (DDP: ${NUM_GPUS} GPUs)..."
        torchrun --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT}" \
            -m distill.logit_distill_trainer \
            --config "${CONFIG}" \
            --mode offline
    else
        echo "  [4b] Training with logit distillation (single-process)..."
        python -m distill.logit_distill_trainer \
            --config "${CONFIG}" \
            --mode offline
    fi
}

phase_feature() {
    if [ "${FEATURE_ENABLED}" != "true" ]; then
        echo "Feature distillation is disabled in config (feature_distillation.enabled=false)."
        exit 1
    fi
    echo "  WARNING: Teacher + Student are loaded together; single-process only."
    python -m distill.feature_distill_trainer --config "${CONFIG}"
}

phase_eval() {
    STUDENT_CKPT="$( (ls -td output/student_*/checkpoint-* output/student_*/final 2>/dev/null || true) | head -1 )"

    if [ -n "${STUDENT_CKPT}" ]; then
        echo "  Evaluating: ${STUDENT_CKPT}"
        echo "  Eval backend: ${EVAL_BACKEND}"

        EVAL_ARGS=(
            --model "${STUDENT_CKPT}"
            --test_data "${EVAL_DATA}"
            --output_dir "eval_results/student/"
            --backend "${EVAL_BACKEND}"
            --max_new_tokens "${EVAL_MAX_NEW_TOKENS}"
            --temperature "${EVAL_TEMPERATURE}"
            --max_pixels "${EVAL_MAX_PIXELS}"
            --min_pixels "${EVAL_MIN_PIXELS}"
        )

        if [ "${EVAL_BACKEND}" = "vllm" ]; then
            EVAL_ARGS+=(
                --batch_size "${EVAL_BATCH_SIZE}"
                --tensor_parallel_size "${EVAL_TP_SIZE}"
                --gpu_memory_utilization "${EVAL_GPU_MEMORY_UTILIZATION}"
                --max_model_len "${EVAL_MAX_MODEL_LEN}"
            )
        fi

        python -m eval.evaluate "${EVAL_ARGS[@]}"

        python -m eval.visualize \
            --predictions eval_results/student/predictions.jsonl \
            --output eval_results/student/comparison.html
    else
        echo "  No student checkpoint found."
    fi
}

PLAN_KEYS=()
PLAN_LABELS=()

case "${PHASE}" in
    all)
        add_phase "prepare" "Phase 0: AIHub unified extraction"
        add_phase "generate" "Phase 1: Teacher Synthetic Data Generation"
        add_phase "filter" "Phase 2: Quality Filtering & Balancing"
        add_phase "sft" "Phase 3: Student SFT (Output Distillation)"
        if [ "${LOGIT_ENABLED}" = "true" ]; then
            add_phase "logit" "Phase 4: Logit Distillation"
        fi
        add_phase "eval" "Final Evaluation"
        ;;
    prepare)
        add_phase "prepare" "Phase 0: AIHub unified extraction"
        ;;
    generate)
        add_phase "generate" "Phase 1: Teacher Synthetic Data Generation"
        ;;
    filter)
        add_phase "filter" "Phase 2: Quality Filtering & Balancing"
        ;;
    sft)
        add_phase "sft" "Phase 3: Student SFT (Output Distillation)"
        ;;
    logit)
        add_phase "logit" "Phase 4: Logit Distillation"
        ;;
    feature)
        add_phase "feature" "Phase 5: Feature Distillation (Experimental)"
        ;;
    eval)
        add_phase "eval" "Final Evaluation"
        ;;
    *)
        echo "Unsupported PHASE: ${PHASE}"
        echo "Supported: all / prepare / generate / filter / sft / logit / feature / eval"
        exit 1
        ;;
esac

TOTAL_PHASES=${#PLAN_KEYS[@]}
CURRENT_PHASE=0

echo "=============================================="
echo "  Knowledge Distillation Pipeline"
echo "=============================================="
echo "  Project: ${PROJECT_ROOT}"
echo "  Config:  ${CONFIG}"
echo "  Phase:   ${PHASE}"
echo "  GPUs:    ${NUM_GPUS}"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<not-set>}"
echo "  VLLM_WORKER_MULTIPROC_METHOD: ${VLLM_WORKER_MULTIPROC_METHOD:-<not-set>}"
echo "  Steps:   ${TOTAL_PHASES}"
echo "=============================================="

if [ "${PHASE}" = "all" ] && [ "${LOGIT_ENABLED}" != "true" ]; then
    echo "  Note: Phase 4(Logit Distillation) is disabled in config."
fi

for i in "${!PLAN_KEYS[@]}"; do
    phase_key="${PLAN_KEYS[$i]}"
    phase_label="${PLAN_LABELS[$i]}"

    case "${phase_key}" in
        prepare) run_phase "${phase_label}" phase_prepare ;;
        generate) run_phase "${phase_label}" phase_generate ;;
        filter) run_phase "${phase_label}" phase_filter ;;
        sft) run_phase "${phase_label}" phase_sft ;;
        logit) run_phase "${phase_label}" phase_logit ;;
        feature) run_phase "${phase_label}" phase_feature ;;
        eval) run_phase "${phase_label}" phase_eval ;;
        *)
            echo "Internal error: unknown planned phase ${phase_key}"
            exit 1
            ;;
    esac
done

echo ""
echo "=============================================="
echo "  Distillation Pipeline Complete!"
echo "=============================================="
