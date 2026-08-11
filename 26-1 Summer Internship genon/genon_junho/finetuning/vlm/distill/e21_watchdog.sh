#!/bin/bash
# e21 학습 감시 스크립트 (e19_watchdog.sh 기반, 경로만 e21로 교체)
# - 학습이 (SIGTERM 등으로) 죽으면 최신 checkpoint를 찾아 resume_from_adapter로
#   이어붙여 자동 재시작한다.
# - 살아있는 동안에는 30초마다 확인만. 정상 완료(exit 0)면 감시도 종료.
#
# 실행(메인 학습 launch 후 별도 tmux 창에서):
#   cd /home/jhyeo/finetuning/vlm && source .venv/bin/activate
#   bash distill/e21_watchdog.sh
set -uo pipefail

VLM=/home/jhyeo/finetuning/vlm
cd "$VLM"

RUN_ROOT="output/e21_qwen35_9b_base_grid_e20data"
BASE_CONFIG="config/e21_qwen35_9b_grid_e20data.yaml"
GEN_DIR="config/_watchdog"
LOG_DIR="_run_logs"
mkdir -p "$GEN_DIR" "$LOG_DIR"
WLOG="$LOG_DIR/e21_watchdog.log"

GRACE_ITERS=6

log() { echo "[$(date '+%F %T')] $*" >> "$WLOG"; }
alive() { pgrep -f "distill.student_sft --config" >/dev/null; }

log "watchdog 시작 (감시 대상: distill.student_sft)"

while true; do
  if alive; then
    sleep 30
    continue
  fi

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

  latest=$(find "${RUN_ROOT}"* -maxdepth 2 -type d -name "checkpoint-*" \
             -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
  ts=$(date +%Y%m%d_%H%M%S)
  gen="$GEN_DIR/resume_${ts}.yaml"
  cp "$BASE_CONFIG" "$gen"

  python3 - "$gen" "$latest" "$ts" <<'PY'
import sys
path, ckpt, ts = sys.argv[1:4]
t = open(path).read()
t = t.replace('output_dir: "./output/e21_qwen35_9b_base_grid_e20data/student_sft"',
              f'output_dir: "./output/e21_qwen35_9b_base_grid_e20data_resume_{ts}/student_sft"')
t = t.replace('run_name: "e21_qwen35_9b_base_grid_e20data"',
              f'run_name: "e21_qwen35_9b_base_grid_e20data_resume_{ts}"')
if ckpt:
    t = t.replace('student_sft:',
                  f'student_sft:\n  resume_from_adapter: "{ckpt}"', 1)
open(path, 'w').write(t)
PY

  log "재시작. resume_from_adapter=${latest:-<none, fresh>} config=$gen"

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
  bash distill/run_distill.sh >> "$LOG_DIR/e21_watchdog_run_${ts}.log" 2>&1
  rc=$?

  log "재시작 학습 종료 (exit=$rc)"
  if [ "$rc" -eq 0 ]; then
    log "정상 완료로 판단. 감시 종료."
    break
  fi
  sleep 20
done
