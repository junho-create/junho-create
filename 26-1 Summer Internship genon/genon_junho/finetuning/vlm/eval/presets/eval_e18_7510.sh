#!/usr/bin/env bash
set -euo pipefail

# Run this script on training server:
#   ssh -p 2222 root@192.168.75.174
#   cd /home/vlm_train/qwen3_vl_tsr
#   bash eval/presets/eval_e15.sh

MODEL_GPU_MAP="400:0;500:1;600:2;700:3;800:4;900:5;final:6"
LAUNCH_MODE="sequential"
USE_OCR="true"
NESTED_TEDS_MODE="split_mean"
BATCH_SIZE="48"
API_TIMEOUT="360"
MAX_NUM_SEQS="48"
MAX_NEW_TOKENS="16000"
OUT_ROOT="eval_results/$(date +%Y%m%d)/e18_with_ocr_7510"
REPORT_IMAGE_MODE="external"
REPORT_SHARED_IMAGE_DIR=""
# Example alternatives:
# MODEL_GPU_MAP="400:0;500:1;600:2;700:3"
# MODEL_GPU_MAP="700:0,1,2,3"
# MODEL_GPU_MAP="200:0,1;400:2.3"
# MODEL_GPU_MAP="100:0;200:0;300:0"
# LAUNCH_MODE="parallel"
# USE_OCR="false"
# OUT_ROOT="eval_results/custom_e18_with_ocr"
# REPORT_IMAGE_MODE="embed"
# REPORT_SHARED_IMAGE_DIR="eval_results/shared_images/e18_with_ocr"

bash eval/run_eval_checkpoints_tmux.sh \
  --exp_name e18_7510 \
  --base_model_path /NHNHOME/WORKSPACE/0426030039_A/shkim/models/Qwen3.5-9B \
  --adapter_root /NHNHOME/WORKSPACE/0426030039_A/shkim/tsr_test/train/vlm/output/e18_qwen35_9b_7510 \
  --test_data /NHNHOME/WORKSPACE/0426030039_A/shkim/tsr_test/train/vlm/_train_data/20260317_4_6000/data_split/test.jsonl \
  --max_samples 200 \
  --batch_size "${BATCH_SIZE}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --api_timeout "${API_TIMEOUT}" \
  --max_num_seqs "${MAX_NUM_SEQS}" \
  --retries 5 \
  --launch_mode "${LAUNCH_MODE}" \
  --out_root "${OUT_ROOT}" \
  --use_ocr "${USE_OCR}" \
  --nested_teds_mode "${NESTED_TEDS_MODE}" \
  --report_image_mode "${REPORT_IMAGE_MODE}" \
  --report_shared_image_dir "${REPORT_SHARED_IMAGE_DIR}" \
  --python_bin /NHNHOME/WORKSPACE/0426030039_A/shkim/tsr_test/train/vlm/.venv/bin/python \
  --model_gpu_map "${MODEL_GPU_MAP}"
