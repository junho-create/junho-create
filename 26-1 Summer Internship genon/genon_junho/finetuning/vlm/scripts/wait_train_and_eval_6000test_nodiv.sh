#!/bin/bash
# =============================================================================
# e18_nodiv 학습 종료를 기다린 뒤, 6000_test(200장)에 대해
# "저장된 모든 체크포인트(100스텝 간격) + final" 을 빠짐없이 TEDS 평가한다.
#
# - e18align 용(wait_train_and_eval_6000test.sh)을 복제하되, 체크포인트 선택을
#   sparse(200/400/.../1200)에서 -> "모든 체크포인트" 로 변경.
# - 평가 하네스: eval/run_eval_checkpoints_tmux.sh (e18 6000_test 평가와 동일 파이프라인)
# - GPU: 학습 종료 후 유휴해진 0~3 사용, sequential(GPU 락) 실행.
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

CONFIG_MARKER="${CONFIG_MARKER:-exp_e18_nodiv_20260624.yaml}"
ADAPTER_ROOT="${ADAPTER_ROOT:-${PROJECT_ROOT}/output/e18_nodiv_20260624/student_sft}"
BASE_MODEL="${BASE_MODEL:-/NHNHOME/WORKSPACE/0426030039_A/shkim/models/Qwen3.5-9B}"
TEST_DATA="${TEST_DATA:-/NHNHOME/WORKSPACE/0426030039_A/shkim/tsr_test/train/vlm/_train_data/20260317_4_6000/data_split/test.jsonl}"
PYTHON_BIN="${PYTHON_BIN:-/NHNHOME/WORKSPACE/0426030039_A/shkim/tsr_test/train/vlm/.venv/bin/python}"
GPUS="${GPUS:-0,1,2,3}"
EXP_NAME="${EXP_NAME:-e18_nodiv}"
OUT_ROOT="${OUT_ROOT:-eval_results/e18_nodiv_6000test}"
POLL_SEC="${POLL_SEC:-60}"

LOG="logs/wait_eval_6000test_e18nodiv_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs
exec > >(tee -a "${LOG}") 2>&1

echo "=============================================="
echo "  Wait train -> auto eval 6000_test(200)  [e18_nodiv / ALL checkpoints]"
echo "  config_marker:${CONFIG_MARKER}"
echo "  adapter_root: ${ADAPTER_ROOT}"
echo "  test_data:    ${TEST_DATA}"
echo "  gpus:         ${GPUS}"
echo "  out_root:     ${OUT_ROOT}"
echo "  log:          ${LOG}"
echo "=============================================="

_is_training_running() {
  pgrep -f "distill\\.student_sft.*${CONFIG_MARKER}" >/dev/null 2>&1 \
    || pgrep -f "torchrun.*${CONFIG_MARKER}" >/dev/null 2>&1
}

echo "[wait] training(${CONFIG_MARKER}) monitoring started..."
while _is_training_running; do
  echo "[wait] $(date '+%F %T') training still running..."
  sleep "${POLL_SEC}"
done

echo "[wait] training process not found. waiting for final adapter..."
while [[ ! -f "${ADAPTER_ROOT}/final/adapter_config.json" ]]; do
  if _is_training_running; then sleep "${POLL_SEC}"; continue; fi
  echo "[wait] $(date '+%F %T') final adapter not ready yet..."
  sleep "${POLL_SEC}"
done
echo "[wait] final adapter detected. flush 대기 60s..."
sleep 60

# 체크포인트 -> GPU 매핑: 저장된 "모든" 체크포인트(100스텝 간격) + final, 라운드로빈 GPU.
MAP="$(${PYTHON_BIN} - "${ADAPTER_ROOT}" "${GPUS}" <<'PY'
import os, sys, glob, re
root, gpus = sys.argv[1], [g for g in sys.argv[2].split(",") if g != ""]
steps = sorted(int(re.search(r"checkpoint-(\d+)", p).group(1))
               for p in glob.glob(os.path.join(root, "checkpoint-*")))
entries = [str(s) for s in steps] + ["final"]
print(";".join(f"{e}:{gpus[i % len(gpus)]}" for i, e in enumerate(entries)))
PY
)"
echo "[eval] model_gpu_map=${MAP}"

bash eval/run_eval_checkpoints_tmux.sh \
  --exp_name "${EXP_NAME}" \
  --base_model_path "${BASE_MODEL}" \
  --adapter_root "${ADAPTER_ROOT}" \
  --test_data "${TEST_DATA}" \
  --max_samples 200 \
  --batch_size 48 \
  --max_new_tokens 16000 \
  --api_timeout 360 \
  --max_num_seqs 48 \
  --retries 5 \
  --launch_mode sequential \
  --out_root "${OUT_ROOT}" \
  --use_ocr true \
  --prompt_style chandra_with_ocr \
  --nested_teds_mode split_mean \
  --report_image_mode external \
  --report_shared_image_dir "" \
  --python_bin "${PYTHON_BIN}" \
  --model_gpu_map "${MAP}"

echo "[done] 6000_test eval(all checkpoints) launched -> ${OUT_ROOT}"
