import time
import subprocess
import os
import glob
import json

print("Waiting for cmcv pipelines to finish...")
while True:
    try:
        # Check if cmcv run is still active
        output = subprocess.check_output(["pgrep", "-f", "ocr_filter.cli cmcv run"])
        if not output.strip():
            break
    except subprocess.CalledProcessError:
        # pgrep returns 1 when no processes are matched
        break
    time.sleep(60)

print("CMCV pipelines finished. Merging results...")
shard_files = glob.glob("/NHNHOME/WORKSPACE/0426030039_A/jhyeo/ocr_filter_result/_accel/cmcv_results_s*.jsonl")
main_file = "/NHNHOME/WORKSPACE/0426030039_A/jhyeo/ocr_filter_result/cmcv_results.jsonl"

existing_ids = set()
if os.path.exists(main_file):
    with open(main_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    d = json.loads(line)
                    existing_ids.add(d["id"])
                except:
                    pass

with open(main_file, "a", encoding="utf-8") as out_f:
    for shard in shard_files:
        with open(shard, "r", encoding="utf-8") as in_f:
            for line in in_f:
                if line.strip():
                    try:
                        d = json.loads(line)
                        if d["id"] not in existing_ids:
                            out_f.write(line)
                            existing_ids.add(d["id"])
                    except:
                        pass

print("Killing existing model servers (vLLM and Paddle) to free all 8 GPUs...")
subprocess.call(["pkill", "-f", "vllm.entrypoints"])
subprocess.call(["pkill", "-f", "PaddleOCRVL"])
subprocess.call(["pkill", "-f", "chandra_all"])
time.sleep(10)

print("Starting Judge model (Qwen-397B) on all 8 GPUs...")
os.chdir("/NHNHOME/WORKSPACE/0426030039_A/jhyeo/ocr_file_filter")
# Launch the serve script in the background
subprocess.Popen(["./venv/bin/python3", "-m", "ocr_filter.cli", "models", "serve", "--manifest", "configs/models.yaml", "--only", "judge"])

print("Waiting for Judge model to be ready on port 8004...")
while True:
    try:
        output = subprocess.check_output(["curl", "-s", "http://localhost:8004/v1/models"])
        if b"Qwen" in output or b"data" in output:
            break
    except:
        pass
    time.sleep(10)

print("Judge model is up! Starting hardcase pipeline (V2)...")
subprocess.check_call(["./venv/bin/python3", "-m", "ocr_filter.cli", "hardcase", "run", "--config", "configs/new_docs.yaml"])

print("Generating final complete HTML gallery...")
os.chdir("/NHNHOME/WORKSPACE/0426030039_A/jhyeo/ocr_filter_result")
subprocess.check_call([
    "../ocr_file_filter/venv/bin/python3", "-m", "ocr_filter.cli", "report", "gallery",
    "--config", "../ocr_file_filter/configs/new_docs.yaml",
    "--cmcv-results", "cmcv_results.jsonl",
    "--out", "gallery_final_morning.html",
    "--per-tier", "50"
])

print("All nightly jobs completed successfully!")
