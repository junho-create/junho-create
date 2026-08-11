#!/usr/bin/env bash
set -euo pipefail

# Run this script on training server:
#   ssh -p 2222 root@192.168.75.174
#   cd /home/vlm_train/qwen3_vl_tsr
#   bash eval/presets/eval_e15.sh

MODEL_GPU_MAP="200:0;300:1;400:2;500:3;600:0;700:1;final:2"
LAUNCH_MODE="sequential"
USE_OCR="true"
OUT_ROOT="eval_results/$(date +%Y%m%d)/e16_with_ocr"
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
  --exp_name e16 \
  --adapter_root /home/vlm_train/qwen3_vl_tsr/output/e16_qwen35_9b_6300/student_sft \
  --test_data /home/vlm_train/qwen3_vl_tsr/_train_data/20260317_4_6000/data_split/test.jsonl \
  --max_samples 200 \
  --batch_size 16 \
  --max_new_tokens 10000 \
  --retries 5 \
  --launch_mode "${LAUNCH_MODE}" \
  --out_root "${OUT_ROOT}" \
  --use_ocr "${USE_OCR}" \
  --report_image_mode "${REPORT_IMAGE_MODE}" \
  --report_shared_image_dir "${REPORT_SHARED_IMAGE_DIR}" \
  --model_gpu_map "${MODEL_GPU_MAP}"
