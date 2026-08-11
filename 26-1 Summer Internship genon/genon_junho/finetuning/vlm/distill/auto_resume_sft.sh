#!/bin/bash
# 학습 프로세스가 (원인 불명의 SIGTERM 등으로) 죽으면 최신 체크포인트를 찾아
# resume_from_adapter로 이어붙인 파생 config를 만들어 자동으로 재시작한다.
set -uo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BASE_CONFIG="config/exp_chandra_combined_37818.yaml"
RUN_ROOT="output/chandra_combined_37818_20260720"
GEN_CONFIG_DIR="config/_autoresume"
LOG_DIR="_run_logs"
mkdir -p "$GEN_CONFIG_DIR" "$LOG_DIR"

export CUDA_VISIBLE_DEVICES=0,2,3
export NUM_GPUS=3
export MASTER_PORT=29650
export WANDB_MODE=online
export WANDB_PROJECT="${WANDB_PROJECT:-genon tuning}"
export WANDB_ENTITY="${WANDB_ENTITY:-aisearch260330-genon}"
if [ -f /home/jhyeo/.env ]; then
  set -a
  # shellcheck disable=SC1091
  source /home/jhyeo/.env
  set +a
fi
export PHASE=sft

attempt=0
while true; do
  attempt=$((attempt + 1))

  latest_ckpt=$(find "${RUN_ROOT}"* -maxdepth 2 -type d -name "checkpoint-*" -printf '%T@ %p\n' 2>/dev/null \
    | sort -n | tail -1 | cut -d' ' -f2-)

  gen_config="${GEN_CONFIG_DIR}/attempt_${attempt}.yaml"
  cp "$BASE_CONFIG" "$gen_config"

  run_suffix="resume${attempt}"
  new_output_dir="./${RUN_ROOT}_${run_suffix}"
  new_run_name="chandra_combined_37818_20260720_${run_suffix}"

  python3 - "$gen_config" "$latest_ckpt" "$new_output_dir" "$new_run_name" <<'PY'
import sys, re
path, ckpt, out_dir, run_name = sys.argv[1:5]
with open(path) as f:
    text = f.read()
text = text.replace(
    'output_dir: "./output/chandra_combined_37818_20260720/student_sft"',
    f'output_dir: "{out_dir}/student_sft"'
)
text = text.replace(
    'run_name: "chandra_combined_37818_20260720"',
    f'run_name: "{run_name}"'
)
if ckpt:
    text = text.replace(
        'student_sft:',
        f'student_sft:\n  resume_from_adapter: "{ckpt}"',
        1,
    )
with open(path, "w") as f:
    f.write(text)
print(f"[auto_resume] attempt={sys.argv[0] if False else ''}", file=sys.stderr)
PY

  echo "[auto_resume] attempt ${attempt}: resume_from_adapter=${latest_ckpt:-<none, fresh start>} output_dir=${new_output_dir}"

  CONFIG="$gen_config" bash distill/run_distill.sh \
    > "${LOG_DIR}/full_run_autoresume_attempt${attempt}_$(date +%Y%m%d_%H%M%S).log" 2>&1
  exit_code=$?

  echo "[auto_resume] attempt ${attempt} exited with code ${exit_code}"

  if [ "$exit_code" -eq 0 ]; then
    echo "[auto_resume] training finished normally (exit 0). stopping supervisor."
    break
  fi

  if [ "$attempt" -ge 50 ]; then
    echo "[auto_resume] too many attempts (${attempt}), giving up."
    break
  fi

  sleep 20
done
