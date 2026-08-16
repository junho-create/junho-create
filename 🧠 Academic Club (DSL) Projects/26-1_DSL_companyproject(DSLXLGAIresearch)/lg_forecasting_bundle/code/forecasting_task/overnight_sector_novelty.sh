#!/bin/bash
# Waits for the already-running 3-ticker sector-novelty smoke test to finish,
# then (if it succeeded) launches the full 100-ticker comparison against the
# current best checkpoint (dlinear_final_compliant, hit_rate=0.5900):
#   does adding sector-level novelty consensus to the news_weight_mult signal
#   help, hurt, or do nothing? (macro was ruled out earlier -- identical text
#   across all 100 tickers on a given day makes it useless as a per-sample
#   weighting signal; sector is shared only within ~9-10 tickers per cluster,
#   so it's worth testing even though a weaker signal than target/related.)
set -uo pipefail
cd /root/LG-AI
LOG=logs/overnight_sector_novelty.log
exec > "$LOG" 2>&1
echo "[$(date)] overnight_sector_novelty.sh starting (PID $$)"

SMOKE_LOG=/tmp/claude-0/-root/fd6c5516-108f-4373-8e86-0a8292ac75ba/scratchpad/smoke_sector_novelty.log
SMOKE_DIR=/tmp/claude-0/-root/fd6c5516-108f-4373-8e86-0a8292ac75ba/scratchpad/smoke_sector_novelty

echo "[$(date)] Waiting for smoke test to finish (poll every 30s, up to 90min)..."
for i in $(seq 1 180); do
    if [ -f "${SMOKE_DIR}/results.txt" ]; then
        echo "[$(date)] Smoke test succeeded:"
        cat "${SMOKE_DIR}/results.txt"
        break
    fi
    if ! pgrep -f "smoke_sector_novelty" > /dev/null; then
        echo "[$(date)] Smoke test process is gone but no results.txt -- it crashed or was OOM-killed. Aborting."
        tail -30 "$SMOKE_LOG"
        exit 1
    fi
    sleep 30
done
if [ ! -f "${SMOKE_DIR}/results.txt" ]; then
    echo "[$(date)] Smoke test still hasn't produced results.txt after 90min -- aborting."
    exit 1
fi

USER_JOB_PATTERN="run_cartography|run_kaggle_weighted|generate_submission|newText2Signal|run_maxDA|verify_gpu"
yield_to_user_jobs() {
    echo "[$(date)] Waiting for user/infra jobs (${USER_JOB_PATTERN}) to be gone for 3 consecutive checks (60s apart)..."
    local clear_count=0
    while [ "$clear_count" -lt 3 ]; do
        if pgrep -f "$USER_JOB_PATTERN" > /dev/null; then
            echo "[$(date)] a job is still running -- waiting."
            clear_count=0
        else
            clear_count=$((clear_count + 1))
        fi
        sleep 60
    done
    echo "[$(date)] confirmed gone -- proceeding."
}
wait_for_memory() {
    while true; do
        local usage=$(cat /sys/fs/cgroup/memory/memory.usage_in_bytes 2>/dev/null || echo 0)
        local limit=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || echo 1)
        local pct=$(( usage * 100 / limit ))
        if [ "$pct" -ge 70 ] || pgrep -f "$USER_JOB_PATTERN" > /dev/null; then
            echo "[$(date)] cgroup memory at ${pct}% or a job is active -- waiting before launching."
            sleep 60
        else
            break
        fi
    done
}

yield_to_user_jobs
wait_for_memory

echo "[$(date)] === Full 100-ticker run: target+related+sector novelty ==="
python3 -u -m forecasting_task.run_DoubleAdapt --backbone dlinear --lr 0.002 --reg 1.0 \
    --use_target_novelty --use_related_novelty --use_sector_novelty --news_weight_mult 10 --freeze_online 1 \
    --logdir logs/dlinear_final_plus_sector_novelty
echo "[$(date)] result:"
cat logs/dlinear_final_plus_sector_novelty/results.txt 2>/dev/null || echo "(failed)"

{
    echo "=== sector novelty experiment summary ($(date)) ==="
    echo "baseline (target+related novelty only, no sector): $(cat logs/dlinear_final_compliant/results.txt 2>/dev/null | tr '\n' ' ')"
    echo "with sector novelty added: $(cat logs/dlinear_final_plus_sector_novelty/results.txt 2>/dev/null | tr '\n' ' ')"
} > logs/overnight_sector_novelty_summary.txt
cat logs/overnight_sector_novelty_summary.txt

echo "[$(date)] === OVERNIGHT SECTOR NOVELTY EXPERIMENT COMPLETE ==="
