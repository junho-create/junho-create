#!/bin/bash
# Overnight pipeline v3: push the two winning knobs (news_weight_mult, reg) further,
# since both independently beat the prior best (0.5557) and haven't been combined yet.
#   mult=5,reg=0.5 -> 0.5722 (best so far)
#   mult=0(off),reg=1.0 -> 0.5617
# Grid: mult in {5,7,10} x reg in {0.5,1.0}, skipping the one combo already done.
set -uo pipefail
cd /root/LG-AI
LOG=logs/overnight_pipeline3.log
exec > "$LOG" 2>&1
echo "[$(date)] overnight_pipeline3.sh starting (PID $$)"

USER_JOB_PATTERN="run_cartography|run_kaggle_weighted|generate_submission|newText2Signal|run_maxDA"
yield_to_user_jobs() {
    echo "[$(date)] Waiting for user jobs (${USER_JOB_PATTERN}) to be gone for 3 consecutive checks (60s apart)..."
    local clear_count=0
    while [ "$clear_count" -lt 3 ]; do
        if pgrep -f "$USER_JOB_PATTERN" > /dev/null; then
            echo "[$(date)] a user job is still running -- waiting."
            clear_count=0
        else
            clear_count=$((clear_count + 1))
        fi
        sleep 60
    done
    echo "[$(date)] user jobs confirmed gone -- proceeding."
}
wait_for_memory() {
    while true; do
        local usage=$(cat /sys/fs/cgroup/memory/memory.usage_in_bytes 2>/dev/null || echo 0)
        local limit=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || echo 1)
        local pct=$(( usage * 100 / limit ))
        if [ "$pct" -ge 70 ] || pgrep -f "$USER_JOB_PATTERN" > /dev/null; then
            echo "[$(date)] cgroup memory at ${pct}% or a user job is active -- waiting before launching next run."
            sleep 60
        else
            break
        fi
    done
}

yield_to_user_jobs

declare -A COMBOS
COMBOS[mult5_reg10]="--news_weight_mult 5 --reg 1.0"
COMBOS[mult7_reg05]="--news_weight_mult 7 --reg 0.5"
COMBOS[mult7_reg10]="--news_weight_mult 7 --reg 1.0"
COMBOS[mult10_reg05]="--news_weight_mult 10 --reg 0.5"
COMBOS[mult10_reg10]="--news_weight_mult 10 --reg 1.0"

for tag in mult5_reg10 mult7_reg05 mult7_reg10 mult10_reg05 mult10_reg10; do
    if [ -f "logs/dlinear_${tag}/results.txt" ]; then
        echo "[$(date)] --- ${tag} already done -- skipping ---"
        continue
    fi
    yield_to_user_jobs
    wait_for_memory
    echo "[$(date)] --- ${tag} (${COMBOS[$tag]}) ---"
    python3 -u -m forecasting_task.run_DoubleAdapt --backbone dlinear --lr 0.002 \
        --use_target_novelty --use_related_novelty ${COMBOS[$tag]} \
        --logdir "logs/dlinear_${tag}"
    echo "[$(date)] ${tag} result:"
    cat "logs/dlinear_${tag}/results.txt" 2>/dev/null || echo "(failed)"
done

{
    echo "=== overnight_pipeline3 summary ($(date)) ==="
    echo "reference points:"
    echo "  baseline (no news weight, reg=0.5): 0.5557"
    echo "  mult=5, reg=0.5 (prior run):        0.5722"
    echo "  mult=0, reg=1.0 (prior run):        0.5617"
    echo ""
    for tag in mult5_reg10 mult7_reg05 mult7_reg10 mult10_reg05 mult10_reg10; do
        echo "$tag: $(cat logs/dlinear_${tag}/results.txt 2>/dev/null | tr '\n' ' ')"
    done
} > logs/overnight_pipeline3_summary.txt
cat logs/overnight_pipeline3_summary.txt

echo "[$(date)] === OVERNIGHT PIPELINE 3 COMPLETE ==="
