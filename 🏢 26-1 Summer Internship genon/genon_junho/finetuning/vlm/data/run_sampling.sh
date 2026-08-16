#!/bin/bash
set -euo pipefail

# === AIHub 샘플링 파이프라인 ===
# 사용법: bash data/run_sampling.sh [all|train|validation|test]

# === 설정 ===
INPUT_DIR="./data/extracted/aihub/Training"
INDEX_FILE="./data/index/training_index.jsonl"
OUTPUT_DIR="./data/experiments/e7"
SEED=42

RATIO_COMPLEX=0.30
RATIO_MEDIUM=0.40
RATIO_SIMPLE=0.30

TRAIN_COUNT=10000
VAL_COUNT=1000
TEST_COUNT=500

PROMPT_STYLE="chandra_table_with_ocr"
BBOX_SCALE=1024

# 실행 모드: all | train | validation | test
MODE="${1:-all}"

echo "=== AIHub 샘플링 파이프라인 ==="
echo "  모드: $MODE"
echo "  출력: $OUTPUT_DIR"

# === Step 1: 데이터 분석 (공통, skip 가능) ===
echo ""
echo "[Step 1] 데이터 분석 + 인덱싱"
python -m data.analyze_aihub \
  --input_dir "$INPUT_DIR" --output "$INDEX_FILE" --skip

mkdir -p "$OUTPUT_DIR"

case "$MODE" in
  all)
    # test → validation → train 순서 (exclude 체인으로 중복 방지)
    echo ""
    echo "[Step 2] test 샘플링 ($TEST_COUNT건)"
    python -m data.sample_aihub \
      --index "$INDEX_FILE" --output "$OUTPUT_DIR/test_raw.jsonl" \
      --count $TEST_COUNT \
      --ratio_complex $RATIO_COMPLEX --ratio_medium $RATIO_MEDIUM --ratio_simple $RATIO_SIMPLE \
      --seed $((SEED + 2))

    echo ""
    echo "[Step 3] validation 샘플링 ($VAL_COUNT건)"
    python -m data.sample_aihub \
      --index "$INDEX_FILE" --output "$OUTPUT_DIR/validation_raw.jsonl" \
      --count $VAL_COUNT \
      --ratio_complex $RATIO_COMPLEX --ratio_medium $RATIO_MEDIUM --ratio_simple $RATIO_SIMPLE \
      --seed $((SEED + 1)) \
      --exclude "$OUTPUT_DIR/test_raw.jsonl"

    echo ""
    echo "[Step 4] train 샘플링 ($TRAIN_COUNT건)"
    python -m data.sample_aihub \
      --index "$INDEX_FILE" --output "$OUTPUT_DIR/train_raw.jsonl" \
      --count $TRAIN_COUNT \
      --ratio_complex $RATIO_COMPLEX --ratio_medium $RATIO_MEDIUM --ratio_simple $RATIO_SIMPLE \
      --seed $SEED \
      --exclude "$OUTPUT_DIR/test_raw.jsonl" "$OUTPUT_DIR/validation_raw.jsonl"

    # 모든 split에 OCR 추가
    for SPLIT in train validation test; do
      echo ""
      echo "[Step 5] ${SPLIT} OCR 추가"
      python -m data.add_ocr \
        --input "$OUTPUT_DIR/${SPLIT}_raw.jsonl" \
        --output "$OUTPUT_DIR/${SPLIT}.jsonl" \
        --prompt_style "$PROMPT_STYLE" --bbox_scale $BBOX_SCALE
    done
    ;;

  train)
    echo ""
    echo "[Step 2] train 샘플링 ($TRAIN_COUNT건)"
    python -m data.sample_aihub \
      --index "$INDEX_FILE" --output "$OUTPUT_DIR/train_raw.jsonl" \
      --count $TRAIN_COUNT \
      --ratio_complex $RATIO_COMPLEX --ratio_medium $RATIO_MEDIUM --ratio_simple $RATIO_SIMPLE \
      --seed $SEED

    echo ""
    echo "[Step 3] train OCR 추가"
    python -m data.add_ocr \
      --input "$OUTPUT_DIR/train_raw.jsonl" --output "$OUTPUT_DIR/train.jsonl" \
      --prompt_style "$PROMPT_STYLE" --bbox_scale $BBOX_SCALE
    ;;

  validation)
    echo ""
    echo "[Step 2] validation 샘플링 ($VAL_COUNT건)"
    python -m data.sample_aihub \
      --index "$INDEX_FILE" --output "$OUTPUT_DIR/validation_raw.jsonl" \
      --count $VAL_COUNT \
      --ratio_complex $RATIO_COMPLEX --ratio_medium $RATIO_MEDIUM --ratio_simple $RATIO_SIMPLE \
      --seed $((SEED + 1))

    echo ""
    echo "[Step 3] validation OCR 추가"
    python -m data.add_ocr \
      --input "$OUTPUT_DIR/validation_raw.jsonl" --output "$OUTPUT_DIR/validation.jsonl" \
      --prompt_style "$PROMPT_STYLE" --bbox_scale $BBOX_SCALE
    ;;

  test)
    echo ""
    echo "[Step 2] test 샘플링 ($TEST_COUNT건)"
    python -m data.sample_aihub \
      --index "$INDEX_FILE" --output "$OUTPUT_DIR/test_raw.jsonl" \
      --count $TEST_COUNT \
      --ratio_complex $RATIO_COMPLEX --ratio_medium $RATIO_MEDIUM --ratio_simple $RATIO_SIMPLE \
      --seed $((SEED + 2))

    echo ""
    echo "[Step 3] test OCR 추가"
    python -m data.add_ocr \
      --input "$OUTPUT_DIR/test_raw.jsonl" --output "$OUTPUT_DIR/test.jsonl" \
      --prompt_style "$PROMPT_STYLE" --bbox_scale $BBOX_SCALE
    ;;

  *)
    echo "사용법: bash data/run_sampling.sh [all|train|validation|test]"
    exit 1
    ;;
esac

echo ""
echo "=== 완료 ==="
