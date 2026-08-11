#!/usr/bin/env bash
set -euo pipefail

# Run this script on training server:
#   ssh -p 2222 root@192.168.75.174
#   cd /home/vlm_train/qwen3_vl_tsr
#   bash eval/presets/eval_e15.sh

MODEL_GPU_MAP="final:0,1,2,3"
LAUNCH_MODE="sequential"
USE_OCR="true"
NESTED_TEDS_MODE="split_mean"
BATCH_SIZE="24"
API_TIMEOUT="360"
MAX_NUM_SEQS="24"
MAX_NEW_TOKENS="16000"
OUT_ROOT="eval_results/$(date +%Y%m%d)/e17_with_ocr_nested_final"
REPORT_IMAGE_MODE="external"
REPORT_SHARED_IMAGE_DIR=""
# Example alternatives:
# MODEL_GPU_MAP="400:0;500:1;600:2;700:3"
# MODEL_GPU_MAP="700:0,1,2,3"
# MODEL_GPU_MAP="200:0,1;400:2.3"
# MODEL_GPU_MAP="100:0;200:0;300:0"
# LAUNCH_MODE="parallel"
# USE_OCR="false"
# OUT_ROOT="eval_results/custom_e15_with_ocr"
# REPORT_IMAGE_MODE="embed"
# REPORT_SHARED_IMAGE_DIR="eval_results/shared_images/e15_with_ocr"

bash eval/run_eval_checkpoints_tmux.sh \
  --exp_name e17 \
  --adapter_root /home/vlm_train/qwen3_vl_tsr/output/e17_qwen35_9b_6910/student_sft \
  --test_data /home/vlm_train/qwen3_vl_tsr/_train_data/20260406_nested_synthetic_200_2/data/test.jsonl \
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
  --model_gpu_map "${MODEL_GPU_MAP}"
