#!/bin/bash
# e18_nodiv 학습 런처 (shkim venv 활성화 후 jhshin 워크스페이스에서 SFT 실행)
set -uo pipefail
source /NHNHOME/WORKSPACE/0426030039_A/shkim/tsr_test/train/vlm/.venv/bin/activate
cd /NHNHOME/WORKSPACE/0426030039_A/jhshin/tsr_test/train/vlm
mkdir -p logs
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NUM_GPUS=4 \
MASTER_PORT=29661 \
WANDB_MODE=online \
WANDB_PROJECT=tsr_vlm_train \
CONFIG=config/exp_e18_nodiv_20260624.yaml \
PHASE=sft \
bash distill/run_distill.sh 2>&1 | tee "logs/train_e18nodiv_$(date +%Y%m%d_%H%M%S).log"
echo "[train_e18nodiv] FINISHED exit=$? at $(date '+%F %T')"
