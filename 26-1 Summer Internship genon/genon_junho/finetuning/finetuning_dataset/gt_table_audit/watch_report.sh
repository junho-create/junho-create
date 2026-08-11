#!/bin/bash
# 감사가 도는 동안 리포트를 주기적으로 다시 만든다.
# 브라우저에서 results/review.html 를 열어두고 새로고침하면 최신 결과가 보인다.
#
# 사용: ./watch_report.sh [주기초=300]
cd "$(dirname "$0")"
INTERVAL="${1:-300}"
while true; do
  python3 report.py --max 600 >results/report.log 2>&1
  DONE=$(wc -l <results/audit.jsonl)
  echo "$(date +%H:%M:%S)  $DONE 건 처리  → results/review.html 갱신"
  # 감사 프로세스가 끝났으면 마지막으로 한 번 더 만들고 종료
  if ! pgrep -f "run_audit.py" >/dev/null; then
    python3 report.py --max 600 >results/report.log 2>&1
    echo "$(date +%H:%M:%S)  감사 종료 — 최종 리포트 생성 완료 ($DONE 건)"
    exit 0
  fi
  sleep "$INTERVAL"
done
