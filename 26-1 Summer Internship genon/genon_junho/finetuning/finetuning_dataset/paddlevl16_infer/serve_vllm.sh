#!/usr/bin/env bash
# PaddleOCR-VL-1.6 을 vLLM OpenAI 서버로 띄운다.
#
# paddlex 의 genai vllm 플러그인은 vllm==0.10.2 를 요구해서 이 환경(0.17.0)에서 못 쓴다.
# 대신 vLLM 이 PaddleOCRVLForConditionalGeneration 을 자체 지원하므로 직접 serve 한다.
# 기동 로그에 나오는 "Failed to load plugin register_paddlex_genai_models" 는 무시해도 된다.
#
# --api-server-count 4 는 성능이 아니라 안정성 때문에 필요하다. API 서버가 하나면 HF fast
# tokenizer 를 여러 요청이 동시에 잡으면서 400 "Already borrowed" 가 청크마다 터지고,
# 클라이언트가 페이지 단위 직렬 재시도로 떨어져 처리량이 1/4 로 주저앉는다.
set -euo pipefail

GPU="${1:-0}"
PORT="${2:-8118}"
MODEL_DIR="${MODEL_DIR:-/root/.paddlex/official_models/PaddleOCR-VL-1.6}"
LOG_DIR="$(dirname "$0")/logs"
mkdir -p "$LOG_DIR"

CUDA_VISIBLE_DEVICES="$GPU" nohup python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" \
  --served-model-name PaddleOCR-VL-1.6-0.9B \
  --host 127.0.0.1 --port "$PORT" \
  --gpu-memory-utilization 0.85 \
  --max-model-len 16384 \
  --limit-mm-per-prompt '{"image":1}' \
  --api-server-count 4 \
  --trust-remote-code \
  > "$LOG_DIR/vllm_server_gpu${GPU}_${PORT}.log" 2>&1 &

echo "vLLM 기동 중 (gpu=$GPU port=$PORT pid=$!). 준비되면 http://127.0.0.1:$PORT/v1/models 가 200 을 준다."
