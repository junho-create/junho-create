#!/usr/bin/env bash
# dots.ocr(rednote-hilab/dots.ocr)를 vLLM OpenAI 서버로 띄운다.
#
# paddle 쪽(../paddlevl16_infer/serve_vllm.sh)과 달리 --api-server-count 를 안 준다.
# 거기서 필요했던 건 paddlex 클라이언트가 한 페이지를 블록 수십 개로 쪼개 쏘는 통에
# tokenizer 경합이 났기 때문인데, dots 는 페이지당 요청 1개라 그 문제가 없다.
#
# 모델은 dots.mocr 가 아니라 dots.ocr 이어야 한다. 이름이 비슷하고 아키텍처도 똑같이
# DotsOCRForCausalLM 이라 헷갈리는데, mocr 은 마크다운 전용이라 레이아웃 프롬프트를
# 줘도 bbox 없이 본문 마크다운만 뱉는다(temperature 를 바꿔도 마찬가지). 표별 crop 을
# 만들려면 bbox 가 필요하므로 레이아웃 모델인 dots.ocr 을 쓴다.
set -euo pipefail

GPU="${1:-2}"
PORT="${2:-8119}"
MODEL="${MODEL:-rednote-hilab/dots.ocr}"
LOG_DIR="$(dirname "$0")/logs"
mkdir -p "$LOG_DIR"

CUDA_VISIBLE_DEVICES="$GPU" nohup python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name dots.ocr \
  --host 127.0.0.1 --port "$PORT" \
  --gpu-memory-utilization 0.85 \
  --max-model-len 25600 \
  --limit-mm-per-prompt '{"image":1}' \
  --trust-remote-code \
  > "$LOG_DIR/vllm_dots_gpu${GPU}_${PORT}.log" 2>&1 &

echo "dots.ocr 기동 중 (gpu=$GPU port=$PORT pid=$!)."
