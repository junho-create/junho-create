#!/bin/bash
# =============================================================================
# Qwen3-VL TSR Fine-tuning 실행 스크립트
# =============================================================================

set -e

# --- Configuration ---
CONFIG=${CONFIG:-"config/training_config.yaml"}
LORA_CONFIG=${LORA_CONFIG:-"config/lora_config.yaml"}
NUM_GPUS=${NUM_GPUS:-4}
PHASE=${PHASE:-"both"}        # phase1 / phase2 / both
MERGE=${MERGE:-"false"}
SKIP_EVAL=${SKIP_EVAL:-"false"}
MASTER_PORT=${MASTER_PORT:-29500}

# --- Environment ---
export WANDB_PROJECT="qwen3_vl_tsr"
export TOKENIZERS_PARALLELISM="false"
export NCCL_P2P_DISABLE="1"   # 필요 시

# Wandb 비활성화 (디버그 시)
# export WANDB_MODE="disabled"

echo "=============================================="
echo "  Qwen3-VL TSR Fine-tuning (QLoRA + DDP)"
echo "=============================================="
echo "  GPUs:       $NUM_GPUS"
echo "  Phase:      $PHASE"
echo "  Config:     $CONFIG"
echo "  LoRA:       $LORA_CONFIG"
echo "=============================================="

# --- Pre-checks ---
echo "[1/4] Validating dataset..."
python -m data.validate_dataset --input data/processed/train.jsonl --no-check-images 2>&1 | tail -5

# --- Training ---
echo ""
echo "[2/4] Starting training (phase=$PHASE)..."

MERGE_FLAG=""
if [ "$MERGE" = "true" ]; then
    MERGE_FLAG="--merge"
fi

if [ "$NUM_GPUS" -gt 1 ]; then
    # Multi-GPU: torchrun + DDP (QLoRA와 호환)
    torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
        train/train_qlora.py \
        --config $CONFIG \
        --lora_config $LORA_CONFIG \
        --phase $PHASE \
        $MERGE_FLAG
else
    # Single GPU
    python train/train_qlora.py \
        --config $CONFIG \
        --lora_config $LORA_CONFIG \
        --phase $PHASE \
        $MERGE_FLAG
fi

# --- Post-training Evaluation ---
if [ "$SKIP_EVAL" = "true" ]; then
    echo ""
    echo "[3/4] Skipping evaluation (SKIP_EVAL=true)"
else
    echo ""
    echo "[3/4] Running evaluation..."
    CHECKPOINT=$(ls -td output/*/final 2>/dev/null | head -1)
    if [ -n "$CHECKPOINT" ]; then
        python -m eval.evaluate \
            --model $CHECKPOINT \
            --test_data data/processed/eval.jsonl \
            --output_dir eval_results/
        echo ""
        echo "[4/4] Generating visualization..."
        python -m eval.visualize \
            --predictions eval_results/predictions.jsonl \
            --output eval_results/comparison.html
    else
        echo "  No checkpoint found, skipping evaluation."
    fi
fi

echo ""
echo "=============================================="
echo "  Training complete!"
echo "=============================================="
