#!/bin/bash
# e19 학습 감시 스크립트
# - 지금 tmux(jhyeo)에서 도는 학습이 (SIGTERM 등으로) 죽으면,
#   최신 checkpoint를 찾아 resume_from_adapter로 이어붙여 자동 재시작한다.
# - 학습이 살아있는 동안에는 30초마다 확인만 하고 아무것도 안 한다(중복 실행 방지).
# - 정상 완료(exit 0)로 끝나면 감시도 종료한다.
set -uo pipefail

VLM=/home/jhyeo/finetuning/vlm
cd "$VLM"

RUN_ROOT="output/e19_qwen35_9b_37818"
BASE_CONFIG="config/e19_qwen35_9b_37818.yaml"
GEN_DIR="config/_watchdog"
LOG_DIR="_run_logs"
mkdir -p "$GEN_DIR" "$LOG_DIR"
WLOG="$LOG_DIR/e19_watchdog.log"

GRACE_ITERS=6   # 10초 x 6 = 60초 동안 계속 없으면 진짜 죽은 것으로 판단

log() { echo "[$(date '+%F %T')] $*" >> "$WLOG"; }
alive() { pgrep -f "distill.student_sft --config" >/dev/null; }

log "watchdog 시작 (감시 대상: distill.student_sft)"

while true; do
  if alive; then
    sleep 30
    continue
  fi

  # 프로세스가 안 보임 -> grace period 동안 재확인 (일시적 공백 오탐 방지)
  log "학습 프로세스 미검출. ${GRACE_ITERS}0초간 재확인..."
  reappeared=0
  for _ in $(seq 1 $GRACE_ITERS); do
    sleep 10
    if alive; then reappeared=1; break; fi
  done
  if [ "$reappeared" -eq 1 ]; then
    log "프로세스 재등장. 계속 감시."
    continue
  fi

  # 진짜 죽음 -> 최신 checkpoint로 재시작
  latest=$(find "${RUN_ROOT}"* -maxdepth 2 -type d -name "checkpoint-*" \
             -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
  ts=$(date +%Y%m%d_%H%M%S)
  gen="$GEN_DIR/resume_${ts}.yaml"
  cp "$BASE_CONFIG" "$gen"

  python3 - "$gen" "$latest" "$ts" <<'PY'
import sys
path, ckpt, ts = sys.argv[1:4]
t = open(path).read()
t = t.replace('output_dir: "./output/e19_qwen35_9b_37818/student_sft"',
              f'output_dir: "./output/e19_qwen35_9b_37818_resume_{ts}/student_sft"')
t = t.replace('run_name: "e19_qwen35_9b_37818"',
              f'run_name: "e19_qwen35_9b_37818_resume_{ts}"')
if ckpt:
    t = t.replace('student_sft:',
                  f'student_sft:\n  resume_from_adapter: "{ckpt}"', 1)
open(path, 'w').write(t)
PY

  log "재시작. resume_from_adapter=${latest:-<none, fresh>} config=$gen"

  # 혹시 남은 좀비 프로세스가 포트를 쥐고 있으면 정리 (student_sft 없을 때만 도달)
  pkill -f "torchrun --nproc_per_node=3 --master_port=29650" 2>/dev/null || true
  sleep 3

  source .venv/bin/activate
  CUDA_VISIBLE_DEVICES=0,2,3 \
  NUM_GPUS=3 \
  MASTER_PORT=29650 \
  WANDB_MODE=online \
  WANDB_PROJECT="tsr_vlm_train" \
  WANDB_ENTITY="aisearch260330-genon" \
  CONFIG="$gen" \
  PHASE=sft \
  bash distill/run_distill.sh >> "$LOG_DIR/e19_watchdog_run_${ts}.log" 2>&1
  rc=$?

  log "재시작 학습 종료 (exit=$rc)"
  if [ "$rc" -eq 0 ]; then
    log "정상 완료로 판단. 감시 종료."
    break
  fi
  sleep 20
done
