#!/bin/bash
# tablelayout: chandra_all_e18align_20260623의 모든 checkpoint -> 6000_test(200) TEDS 평가 (GPU 4-7)
set -uo pipefail
source /NHNHOME/WORKSPACE/0426030039_A/shkim/tsr_test/train/vlm/.venv/bin/activate
cd /NHNHOME/WORKSPACE/0426030039_A/jhshin/tsr_test/train/vlm
unset TMUX

BASE_MODEL=/NHNHOME/WORKSPACE/0426030039_A/shkim/models/Qwen3.5-9B
TEST_DATA=/NHNHOME/WORKSPACE/0426030039_A/shkim/tsr_test/train/vlm/_train_data/20260317_4_6000/data_split/test.jsonl
PYTHON_BIN=/NHNHOME/WORKSPACE/0426030039_A/shkim/tsr_test/train/vlm/.venv/bin/python
GPUS=4,5,6,7
POLL_SEC=60
LOG="logs/eval_tablelayout_ckpts_6000test_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs
exec > >(tee -a "${LOG}") 2>&1

_build_map() {
  local adapter_root="$1"
  local gpus_csv="$2"
  "${PYTHON_BIN}" - "${adapter_root}" "${gpus_csv}" <<'PY'
import os, sys, glob, re
root, gpus = sys.argv[1], [g for g in sys.argv[2].split(",") if g.strip()]
steps = sorted(
    int(re.search(r"checkpoint-(\d+)", p).group(1))
    for p in glob.glob(os.path.join(root, "checkpoint-*"))
)
if not steps:
    raise SystemExit(f"[ERROR] no checkpoints under {root}")
entries = [str(s) for s in steps]
print(";".join(f"{e}:{gpus[i % len(gpus)]}" for i, e in enumerate(entries)))
PY
}

_count_ckpt_metrics() {
  local out_root="$1"
  local ckpt_prefix="$2"
  find "${out_root}" -maxdepth 2 -name "metrics.json" 2>/dev/null \
    | grep -E "/${ckpt_prefix}ckpt[0-9]+" | wc -l
}

_wait_ckpt_metrics() {
  local label="$1"
  local out_root="$2"
  local ckpt_prefix="$3"
  local expected="$4"
  echo "[wait:${label}] expecting ${expected} checkpoint metrics under ${out_root}"
  while true; do
    local done
    done="$(_count_ckpt_metrics "${out_root}" "${ckpt_prefix}")"
    echo "[wait:${label}] $(date '+%F %T') metrics=${done}/${expected}"
    if [[ "${done}" -ge "${expected}" ]]; then
      echo "[wait:${label}] all checkpoint metrics ready."
      return 0
    fi
    sleep "${POLL_SEC}"
  done
}

_run_eval() {
  local exp_name="$1"
  local adapter_root="$2"
  local out_root="$3"
  local start_port="$4"
  local launch_mode="$5"
  local gpus_csv="$6"

  local map
  map="$(_build_map "${adapter_root}" "${gpus_csv}")"
  local ckpt_count
  ckpt_count="$(echo "${map}" | tr ';' '\n' | wc -l)"
  echo "=============================================="
  echo "[${exp_name}] adapter_root=${adapter_root}"
  echo "[${exp_name}] out_root=${out_root}"
  echo "[${exp_name}] start_port=${start_port} launch_mode=${launch_mode}"
  echo "[${exp_name}] checkpoint count=${ckpt_count}"
  echo "[${exp_name}] model_gpu_map=${map}"
  echo "=============================================="

  bash eval/run_eval_checkpoints_tmux.sh \
    --exp_name "${exp_name}" \
    --base_model_path "${BASE_MODEL}" \
    --adapter_root "${adapter_root}" \
    --test_data "${TEST_DATA}" \
    --max_samples 200 \
    --batch_size 48 \
    --max_new_tokens 16000 \
    --api_timeout 360 \
    --max_num_seqs 48 \
    --retries 5 \
    --launch_mode "${launch_mode}" \
    --out_root "${out_root}" \
    --use_ocr true \
    --prompt_style chandra_with_ocr \
    --nested_teds_mode split_mean \
    --report_image_mode external \
    --report_shared_image_dir "" \
    --python_bin "${PYTHON_BIN}" \
    --start_port "${start_port}" \
    --model_gpu_map "${map}"
}

echo "[start] tablelayout checkpoint eval batch at $(date '+%F %T')"
echo "[log] ${LOG}"

# e18 parity table+layout: checkpoint 전체 — GPU 4-7 sequential
V2_OUT="eval_results/tablelayout_6000test"
V2_ROOT="/NHNHOME/WORKSPACE/0426030039_A/jhshin/tsr_test/train/vlm/output/chandra_all_e18align_20260623/student_sft"
V2_COUNT="$("${PYTHON_BIN}" - "${V2_ROOT}" <<'PY'
import os, sys, glob
print(len(glob.glob(os.path.join(sys.argv[1], "checkpoint-*"))))
PY
)"
_run_eval "tablelayout" "${V2_ROOT}" "${V2_OUT}" 8200 sequential "${GPUS}"

echo "[started] tablelayout ${V2_COUNT} checkpoints (sequential tmux, GPU 4-7)"
echo "[info] completion: tmux attach -t eval_tablelayout_tablelayouttest200_sequential"
echo "[ALL LAUNCHED] log=${LOG}"
