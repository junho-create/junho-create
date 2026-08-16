#!/bin/bash
# Follow-up pipeline (run 2 for the night):
#   A. Yield to the user's own run_cartography job (heavy, ~99% CPU, memory near cgroup limit)
#      so the two heavy jobs never compete for the ~128.8GB cgroup memory limit at once.
#   B. Small hyperparameter sweep on the best backbone found so far (DLinear + DoubleAdapt,
#      Step1 price + text-PCA features only, no derived macro/sector/novelty flags).
#   C. Re-run Step5 attribution using the DLinear checkpoint instead of the earlier GRU
#      checkpoint, since the GRU checkpoint's attribution signal looked miscalibrated
#      (permuting every group, including price, improved hit rate).
set -uo pipefail
cd /root/LG-AI
LOG=logs/followup_pipeline.log
exec > "$LOG" 2>&1
echo "[$(date)] followup_pipeline.sh starting (PID $$)"

BASE_FLAGS=""  # same curated set as ensemble_* runs: no macro/sector/target/related derived flags

# --- Yield to any of the user's own heavy jobs, by name pattern ---
# Covers everything we've seen the user launch directly in this session so far.
USER_JOB_PATTERN="run_cartography|run_kaggle_weighted|generate_submission|newText2Signal"
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

yield_to_user_jobs

# Extra safety: also wait if cgroup memory is already high, OR a user job reappears
# (e.g. its own panel/embedding build spikes memory a few seconds after this check passes).
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

# --- Stage B: DLinear hyperparameter sweep ---
echo "[$(date)] === Stage B: DLinear hyperparameter sweep ==="
declare -A EXTRA
EXTRA[lr_low]="--lr 0.0005"
EXTRA[lr_high]="--lr 0.002"
EXTRA[seqlen_short]="--seq_len 40"
EXTRA[seqlen_long]="--seq_len 90"
EXTRA[reg_low]="--reg 0.1"
EXTRA[reg_high]="--reg 1.0"

for tag in lr_low lr_high seqlen_short seqlen_long reg_low reg_high; do
    if [ -f "logs/dlinear_hp_${tag}/results.txt" ]; then
        echo "[$(date)] --- dlinear_hp_${tag} already done -- skipping ---"
        continue
    fi
    yield_to_user_jobs
    wait_for_memory
    echo "[$(date)] --- dlinear_hp_${tag} (${EXTRA[$tag]}) ---"
    python3 -u -m forecasting_task.run_DoubleAdapt $BASE_FLAGS --backbone dlinear ${EXTRA[$tag]} \
        --logdir "logs/dlinear_hp_${tag}"
    echo "[$(date)] dlinear_hp_${tag} result:"
    cat "logs/dlinear_hp_${tag}/results.txt" 2>/dev/null || echo "(failed)"
done

{
    echo "=== DLinear hyperparameter sweep (baseline: lr=0.001 seq_len=60 reg=0.5 -> hit_rate=0.5495) ==="
    for tag in lr_low lr_high seqlen_short seqlen_long reg_low reg_high; do
        echo "$tag (${EXTRA[$tag]}): $(cat logs/dlinear_hp_${tag}/results.txt 2>/dev/null | tr '\n' ' ')"
    done
} > logs/dlinear_hp_summary.txt
cat logs/dlinear_hp_summary.txt

# --- Stage C: attribution using the DLinear checkpoint ---
wait_for_memory
echo "[$(date)] === Stage C: attribution on DLinear checkpoint ==="
python3 -u -m forecasting_task.attribution --logdir logs/ensemble_dlinear
cat logs/ensemble_dlinear/attribution.csv 2>/dev/null || echo "(attribution failed)"

echo "[$(date)] === FOLLOWUP PIPELINE COMPLETE ==="
