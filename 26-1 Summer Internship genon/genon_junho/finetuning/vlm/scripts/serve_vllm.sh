export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_PLUGINS=""
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

CUDA_VISIBLE_DEVICES=0,1,2,3 \
vllm serve /home/vlm_train/models/Qwen3.5-9B-e14-final-merged-full \
  --host 0.0.0.0 \
  --port 8030 \
  --served-model-name Qwen/Qwen3.5-9B-e14-final-merged \
  --max-model-len 262144 \
  --max-num-seqs 32 \
  --tensor-parallel-size 4


# 평가서버
/workspace/tsr_eval/venvs/tsr_eval/bin/python -m vllm.entrypoints.openai.api_server \
  --model /workspace/tsr_eval/models/Qwen3.5-9B-e16-final-merged-full \
  --host 0.0.0.0 \
  --port 8012 \
  --tokenizer /workspace/tsr_eval/models/Qwen3.5-9B-e16-final-merged-full \
  --served-model-name e16final \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 1 \
  --skip-mm-profiling \
  --enforce-eager

CUDA_VISIBLE_DEVICES=1 /workspace/tsr_eval/venvs/tsr_eval/bin/python -m vllm.entrypoints.openai.api_server \
 /workspace/tsr_eval/models/dots.mocr \
 --tensor-parallel-size 1 \
 --gpu-memory-utilization 0.9 \
 --chat-template-content-format string \
 --served-model-name model \
 --trust-remote-code
