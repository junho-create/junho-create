# python test_table.py --input_dir "samples_png" --output_dir "output" --debug --postprocess none
TARGET=${TARGET:-self_serving}
python test_table.py \
  --input_dir "samples_png" \
  --output_dir "output" \
  --postprocess none \
  --target "${TARGET}"
