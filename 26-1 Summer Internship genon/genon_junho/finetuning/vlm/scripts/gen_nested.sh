# 20260406
python -m data.generate_nested_tables \
--index_path data/index/training_index.jsonl \
--output_dir _train_data/20260406_nested_synthetic_200_2 \
--count 200 \
--n_child_per_parent 1 \
--gt_quality_filter \
--seed 42 \
--exclude_jsonl _train_data/20260401_nested_synthetic_100_1/nested_synthetic.jsonl \
--dedup_mode html_or_source_pair


# 20260414: add another 200 samples, but exclude both previous sets to avoid duplicates
python -m data.generate_nested_tables \
--index_path data/index/training_index.jsonl \
--output_dir _train_data/20260414_nested_synthetic_200_3 \
--count 200 \
--n_child_per_parent 1 \
--gt_quality_filter \
--seed 42 \
--exclude_jsonl _train_data/20260401_nested_synthetic_100_1/nested_synthetic.jsonl \
--exclude_jsonl _train_data/20260406_nested_synthetic_200_2/nested_synthetic.jsonl \
--dedup_mode html_or_source_pair