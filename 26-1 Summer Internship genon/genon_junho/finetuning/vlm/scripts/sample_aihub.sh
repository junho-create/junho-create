cd /home/vlm_train/qwen3_vl_tsr

# Parameter notes:
# --index: analyzed AI-Hub index JSONL input
# --output: sampled train_raw JSONL output path
# --count: number of samples to draw
# --use_complex_detail: split complex into nested/col/row/mix buckets
# --ratio_complex_nested|col|row|mix: detailed complex sampling ratios
# --ratio_medium|simple: medium/simple sampling ratios
# --no-hard_first: randomize pool order instead of hard-example-first
# --max_per_signature: max samples per structure_signature to control duplicates
# --seed: deterministic sampling seed
# --exclude: JSONL files to exclude by image filename
# --no_thinking: disable thinking field generation
# tee .../sample_aihub.log: save stdout/stderr run log
python3 -m data.sample_aihub \
  --index data/index/training_index_v2.jsonl \
  --output data/processed/sample/20260320_154758_sample_aihub_3000_excl_3sets_filename/train_raw.jsonl \
  --count 3000 \
  --use_complex_detail \
  --ratio_complex_nested 0.0 \
  --ratio_complex_col 0.3 \
  --ratio_complex_row 0.2 \
  --ratio_complex_mix 0.3 \
  --ratio_medium 0.1 \
  --ratio_simple 0.1 \
  --no-hard_first \
  --max_per_signature 4 \
  --seed 20260320 \
  --exclude \
    _train_data/20260225_1_3000/data_split/train.jsonl \
    _train_data/20260225_1_3000/data_split/valid.jsonl \
    _train_data/20260225_1_3000/data_split/test.jsonl \
    _train_data/20260311_2_3000/data_split/train.jsonl \
    _train_data/20260311_2_3000/data_split/valid.jsonl \
    _train_data/20260311_2_3000/data_split/test.jsonl \
    _train_data/20260316_3_3000/train_raw.jsonl \
  --no_thinking |& tee data/processed/sample/20260320_154758_sample_aihub_3000_excl_3sets_filename/sample_aihub.log

# python3 -m data.sample_aihub \
#   --index data/index/training_index_v2.jsonl \
#   --count 3000 \
#   --use_complex_detail \
#   --ratio_complex_col 0.3 \
#   --ratio_complex_row 0.2 \
#   --ratio_complex_mix 0.3 \
#   --ratio_medium 0.1 \
#   --ratio_simple 0.1 \
#   --no-hard_first \
#   --max_per_signature 4 \
#   --seed 20260320 \
#   --exclude <3-set union>
