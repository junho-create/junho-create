#!/bin/bash
# e18_nodiv 대기->평가 런처. 학습 종료까지 폴링 후 모든 체크포인트 6000_test 평가.
set -uo pipefail
source /NHNHOME/WORKSPACE/0426030039_A/shkim/tsr_test/train/vlm/.venv/bin/activate
cd /NHNHOME/WORKSPACE/0426030039_A/jhshin/tsr_test/train/vlm
# 평가 하네스가 자체적으로 detached tmux 세션을 생성하므로 부모 TMUX 컨텍스트 해제.
unset TMUX
bash scripts/wait_train_and_eval_6000test_nodiv.sh
echo "[eval_e18nodiv] WRAPPER FINISHED at $(date '+%F %T')"
