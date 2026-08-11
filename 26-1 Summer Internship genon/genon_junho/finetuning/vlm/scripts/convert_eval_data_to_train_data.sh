# cd /Users/shkim/_shkim/01.source/tsr_labs/train/vlm

python -m data.merge_jsonl --input_dirs _train_data/20260325_10 --output _train_data/20260325_10/merged.jsonl

python3 -m data.convert_predictions_to_train_ocr \
  --input _train_data/20260325_10/merged.jsonl \
  --output _train_data/20260325_10/train_ocr_filtered.jsonl \
  --ocr_lookup_jsonl _train_data/20260325_10/data_ocr/train_ocr.jsonl \
  --drop_empty_html

# python3 -m data.convert_predictions_to_train_ocr \
#   --input eval_results/20260321/e15_without_ocr/e15final_noreason_nothink_b16_api_on_e15test200_retry5_b16_m10000_20260321_105947/predictions.jsonl \
#   --output _train_data/20260320_5_3000/data_ocr/train_ocr_from_e15final_with_lookup.jsonl \
#   --ocr_lookup_jsonl _train_data/20260320_5_3000/data_ocr/train_ocr.jsonl \
#   --drop_empty_html
