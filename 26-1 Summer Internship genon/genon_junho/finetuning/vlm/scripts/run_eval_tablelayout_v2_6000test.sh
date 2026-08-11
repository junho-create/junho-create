#!/bin/bash
# tablelayout (chandra_all_e18align_20260623, table+layout e18 parity) -> 6000_test(200) TEDS 평가. GPU 5, port 8040(충돌 회피).
set -uo pipefail
source /NHNHOME/WORKSPACE/0426030039_A/shkim/tsr_test/train/vlm/.venv/bin/activate
cd /NHNHOME/WORKSPACE/0426030039_A/jhshin/tsr_test/train/vlm
unset TMUX

BASE_MODEL=/NHNHOME/WORKSPACE/0426030039_A/shkim/models/Qwen3.5-9B
TEST_DATA=/NHNHOME/WORKSPACE/0426030039_A/shkim/tsr_test/train/vlm/_train_data/20260317_4_6000/data_split/test.jsonl
PYTHON_BIN=/NHNHOME/WORKSPACE/0426030039_A/shkim/tsr_test/train/vlm/.venv/bin/python

bash eval/run_eval_checkpoints_tmux.sh \
  --exp_name "tablelayout" \
  --base_model_path "${BASE_MODEL}" \
  --adapter_root "/NHNHOME/WORKSPACE/0426030039_A/jhshin/tsr_test/train/vlm/output/chandra_all_e18align_20260623/student_sft" \
  --test_data "${TEST_DATA}" \
  --max_samples 200 \
  --batch_size 48 \
  --max_new_tokens 16000 \
  --api_timeout 360 \
  --max_num_seqs 48 \
  --retries 5 \
  --launch_mode sequential \
  --out_root "eval_results/tablelayout_6000test" \
  --use_ocr true \
  --prompt_style chandra_with_ocr \
  --nested_teds_mode split_mean \
  --report_image_mode external \
  --report_shared_image_dir "" \
  --python_bin "${PYTHON_BIN}" \
  --start_port 8040 \
  --model_gpu_map "final:5"
echo "[eval_tablelayout] FINISHED exit=$? at $(date '+%F %T')"
