#!/bin/bash
# =============================================================================
# Unified (Table + Layout) Student SFT 실행 스크립트
# - scripts/train.sh 와 동일하게 distill/run_distill.sh (PHASE=sft) 를 호출한다.
# - 사전에 data/build_unified_dataset.py 로 통합 데이터셋을 만들어 두어야 한다.
#
# 출력 형식(파이프라인) 전환 — CONFIG 환경변수로 선택:
#   # (1) JSON 파이프라인 (기본)
#   bash scripts/train_unified.sh
#   # (2) HTML 파이프라인
#   CONFIG=config/exp_unified_html_smoke.yaml bash scripts/train_unified.sh
#
# 데이터 빌드(형식별):
#   python -m data.build_unified_dataset ... --output-format json   # JSON 셋
#   python -m data.build_unified_dataset ... --output-format html \
#       --out-dir ./_train_data/unified_html_smoke                   # HTML 셋
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

mkdir -p logs

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3} \
NUM_GPUS=${NUM_GPUS:-4} \
MASTER_PORT=${MASTER_PORT:-29661} \
WANDB_MODE=${WANDB_MODE:-online} \
WANDB_PROJECT=${WANDB_PROJECT:-tsr_vlm_train} \
CONFIG=${CONFIG:-config/exp_unified_smoke.yaml} \
PHASE=sft \
bash distill/run_distill.sh 2>&1 | tee "logs/train_unified_$(date +%Y%m%d_%H%M%S).log"
