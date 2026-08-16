#!/usr/bin/env bash
set -euo pipefail

# Run this script on training server:
#   ssh -p 2222 root@192.168.75.174
#   cd /home/workspace/tsr_test/genos/doc_parser/train/vlm
#   bash eval/presets/eval_20260409_vsgrpo_v6.sh

MODEL_GPU_MAP="300:0;800:1;1200:2"
LAUNCH_MODE="sequential"
USE_OCR="true"
NESTED_TEDS_MODE="split_mean"
BATCH_SIZE="16"
API_TIMEOUT="180"
MAX_NUM_SEQS="32"
MAX_NEW_TOKENS="10000"
OUT_ROOT="eval_results/$(date +%Y%m%d)/vsgrpo_v6_with_ocr"
REPORT_IMAGE_MODE="external"
REPORT_SHARED_IMAGE_DIR=""
# Example alternatives:
# MODEL_GPU_MAP="300:0;800:1;1200:2"
# MODEL_GPU_MAP="1200:0,1,2,3"
# MODEL_GPU_MAP="300:0,1;800:2,3"
# MODEL_GPU_MAP="300:0;800:0;1200:0"
# LAUNCH_MODE="parallel"
# USE_OCR="false"
# OUT_ROOT="eval_results/custom_vsgrpo_v6_with_ocr"
# REPORT_IMAGE_MODE="embed"
# REPORT_SHARED_IMAGE_DIR="eval_results/shared_images/vsgrpo_v6_with_ocr"

bash eval/run_eval_checkpoints_tmux.sh \
  --exp_name vsgrpo_v6 \
  --adapter_root /home/workspace/tsr_test/outputs/vsgrpo_v6 \
  --test_data /home/vlm_train/qwen3_vl_tsr/_train_data/20260317_4_6000/data_split/test.jsonl \
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
