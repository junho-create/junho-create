cd /home/vlm_train/qwen3_vl_tsr

BASE_MODEL_PATH="/home/vlm_train/models/Qwen3.5-9B"
LORA_ADAPTER_PATH="/home/vlm_train/qwen3_vl_tsr/output/e15_qwen35_9b_6000/student_sft/final"
MERGED_PATH="/home/vlm_train/models/Qwen3.5-9B-e15-final-merged-full"

CUDA_VISIBLE_DEVICES=0 python3 - "$BASE_MODEL_PATH" "$LORA_ADAPTER_PATH" "$MERGED_PATH" <<'PY'
import os, sys, torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

base_model_path, lora_adapter_path, output_path = sys.argv[1], sys.argv[2], sys.argv[3]

model = AutoModelForImageTextToText.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)
model = PeftModel.from_pretrained(model, lora_adapter_path)
model = model.merge_and_unload()

os.makedirs(output_path, exist_ok=True)
model.save_pretrained(output_path)
AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True).save_pretrained(output_path)
print("DONE:", output_path)
PY
