cd /home/vlm_train/qwen3_vl_tsr

python3 -m data.add_ocr \
  --input _train_data/20260320_5_3000/train_raw.jsonl \
  --output _train_data/20260320_5_3000/data_ocr/train_ocr.jsonl \
  --prompt_style chandra_table_with_ocr \
  --bbox_scale 1024 \
  --ocr_lang korean
#   --image_root _train_data/20260320_5_3000 \
