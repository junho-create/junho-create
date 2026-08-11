#!/bin/bash
# e20 학습 run 을 주기적으로 WandB 클라우드에 sync 한다.
# (학습 프로세스의 실시간 업로더가 죽어 로컬에만 쌓이는 상황 대응)
# - 매 INTERVAL 초마다 wandb sync 시도. 라이브 파일이라 tail 에서 EOF 에러가 나도
#   그 직전까지는 업로드되므로, 반복하면 web UI 가 점진적으로 따라온다.
# - 학습 프로세스가 사라지면 마지막으로 한 번 더 sync(그땐 파일이 완결돼 전량 업로드)하고 종료.
#
# 사용:
#   cd /home/jhyeo/finetuning/vlm && source .venv/bin/activate
#   bash wandb_sync_loop.sh                       # tmux 안에서 포그라운드
#   nohup bash wandb_sync_loop.sh >/dev/null 2>&1 &   # 백그라운드
set -uo pipefail
cd /home/jhyeo/finetuning/vlm

INTERVAL="${INTERVAL:-300}"   # 초 (기본 5분)
RUN_DIR="${RUN_DIR:-}"

# RUN_DIR 미지정이면 가장 최근 run 디렉토리 자동 선택
if [ -z "$RUN_DIR" ]; then
  RUN_DIR=$(ls -dt wandb/run-*/ 2>/dev/null | head -1)
fi
RUN_DIR="${RUN_DIR%/}"

LOG="_run_logs/wandb_sync_loop.log"
mkdir -p _run_logs
echo "[$(date '+%F %T')] sync loop 시작. RUN_DIR=$RUN_DIR interval=${INTERVAL}s" >> "$LOG"

# --no-skip-synced: 이미 synced 로 표시된 run 도 다시 처리해 새 데이터를 이어 올림
# --no-skip-online: online run 도 건너뛰지 않음
SYNC_FLAGS="--no-skip-synced --no-skip-online"

while true; do
  wandb sync $SYNC_FLAGS "$RUN_DIR" >> "$LOG" 2>&1
  last=$(grep -oE "uploading history steps [0-9]+-[0-9]+" "$LOG" | tail -1)
  echo "[$(date '+%F %T')] synced (${last:-no-new}; tail EOF 경고는 정상)" >> "$LOG"

  # 학습 프로세스가 끝났으면 마지막 sync 후 종료
  if ! pgrep -f "distill.student_sft --config" >/dev/null; then
    sleep 10
    wandb sync $SYNC_FLAGS "$RUN_DIR" >> "$LOG" 2>&1
    echo "[$(date '+%F %T')] 학습 종료 감지 → 최종 sync 완료. 루프 종료." >> "$LOG"
    break
  fi
  sleep "$INTERVAL"
done
