# ==============================================================================
# 기존 파이프라인 (레이블링 도구 데이터)
# ==============================================================================

# 1. 데이터 변환
python -m data.convert_to_chat --input raw_labels.jsonl --image_dir images/ --output data/processed/dataset.jsonl
python -m data.convert_to_chat --input ../../tsr_lable_tool/dataset_test/images/dataset_train.jsonl --image_dir ../../tsr_lable_tool/dataset_test/images/train --output data/processed/dataset.jsonl

# 2. 검증
python -m data.validate_dataset --input data/processed/dataset.jsonl

# 3. 분리
python -m data.split_dataset --input data/processed/dataset.jsonl --output_dir data/processed/

# 4. 학습
bash train/run_train.sh

# 5. 평가
python -m eval.evaluate --model output/.../final --test_data data/processed/eval.jsonl


# ==============================================================================
# AIHub 데이터 파이프라인
# ==============================================================================

# === 공통 압축 해제 (train/distill 공용) ===
python -m data.extract_aihub \
    --input ../aihub_table_data/Training \
    --output_dir ./data/extracted/aihub \
    --max_zips 1

# === AIHub 전체 변환 (Training, ~40만장) ===
# python -m data.convert_aihub \
#     --input ./data/extracted/aihub/Training \
#     --output data/processed/dataset.jsonl

# === AIHub 테스트 실행 (50건, 최소 스텝으로 파이프라인 검증) ===

# 1. 데이터 변환 (50건만)
python -m data.convert_aihub \
    --input ./data/extracted/aihub/Training \
    --output data/processed/dataset.jsonl \
    --max-samples 50

# 2. 검증
python -m data.validate_dataset --input data/processed/dataset.jsonl --no-check-images

# 3. train/eval 분리
python -m data.split_dataset --input data/processed/dataset.jsonl --output_dir data/processed/

# 4. 테스트 학습 (phase1만, wandb off)
CONFIG=config/training_config_test.yaml PHASE=phase1 NUM_GPUS=4 bash train/run_train.sh

# 5. 평가 + HTML 리포트 생성
bash eval/run_eval.sh \
    --model output/qwen3_vl_tsr_qlora_test/checkpoint-10 \
    --test_data data/processed/eval.jsonl \
    --output_dir eval_results
