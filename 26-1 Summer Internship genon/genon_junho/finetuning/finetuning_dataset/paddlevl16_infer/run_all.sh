#!/usr/bin/env bash
# 샤드 여러 개로 전체 5,030 페이지를 추론한다. serve_vllm.sh 가 먼저 떠 있어야 한다.
#
# 워커 하나는 CPU 단일 스레드(crop·인코딩)라 vLLM 을 25% 밖에 못 채운다. 샤드를 늘려
# 서버를 포화시키는 게 목적이라, NSHARD 는 GPU 가 아니라 서버 처리량을 보고 정한다.
set -euo pipefail

cd "$(dirname "$0")"
NSHARD="${NSHARD:-6}"
LAYOUT_GPU="${LAYOUT_GPU:-3}"   # 레이아웃 검출용 (vLLM 이 쓰는 GPU 와 달라야 함)
PORT="${PORT:-8118}"

curl -s -m 5 "http://127.0.0.1:$PORT/v1/models" > /dev/null || {
  echo "vLLM 서버($PORT)가 안 떠 있다. ./serve_vllm.sh 0 $PORT 먼저 실행할 것." >&2
  exit 1
}

mkdir -p logs
for i in $(seq 0 $((NSHARD - 1))); do
  CUDA_VISIBLE_DEVICES="$LAYOUT_GPU" nohup python3 -u run_infer.py \
    --shard "$i/$NSHARD" --chunk 32 --concurrency 64 \
    --server-url "http://127.0.0.1:$PORT/v1" \
    > "logs/infer_shard$i.log" 2>&1 &
  echo "shard $i/$NSHARD 시작 (pid $!)"
done

echo "진행: tail -f logs/infer_shard0.log  |  끝나면: python3 merge_shards.py"
