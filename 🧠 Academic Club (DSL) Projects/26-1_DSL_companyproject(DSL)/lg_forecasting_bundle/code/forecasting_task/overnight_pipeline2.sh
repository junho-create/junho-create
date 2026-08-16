#!/bin/bash
# Overnight pipeline v2: news-weighted loss experiment (newly implemented) +
# remaining easy items from PROGRESS.md's "다음 방향" list.
#   A. Smoke test (3 tickers) the new --news_weight_mult sample-weighting mechanism
#      before trusting it on a full run.
#   B. news_weight_mult sweep on DLinear+lr=0.002 (current best config).
#   C. lr=0.002 + reg=1.0 combo (not yet tried).
#   D. Alternate primary_emb_model sweep (bert/gemini/qwen) on DLinear+lr=0.002.
set -uo pipefail
cd /root/LG-AI
LOG=logs/overnight_pipeline2.log
exec > "$LOG" 2>&1
echo "[$(date)] overnight_pipeline2.sh starting (PID $$)"

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
wait_for_memory

# --- Stage A: smoke test the new news_weight_mult mechanism (3 tickers, small/fast) ---
echo "[$(date)] === Stage A: smoke test news_weight_mult ==="
python3 -u -m forecasting_task.run_DoubleAdapt --tickers AAPL,MSFT,NVDA --backbone dlinear \
    --use_target_novelty --use_related_novelty --news_weight_mult 3.0 \
    --logdir logs/smoke_news_weight
if [ ! -f logs/smoke_news_weight/results.txt ]; then
    echo "[$(date)] FATAL: smoke test for news_weight_mult failed -- aborting rest of pipeline. Check log above."
    exit 1
fi
echo "[$(date)] smoke test passed:"
cat logs/smoke_news_weight/results.txt

# --- Stage B: news_weight_mult sweep on best config (DLinear, lr=0.002) ---
echo "[$(date)] === Stage B: news_weight_mult sweep ==="
for MULT in 2 3 5; do
    if [ -f "logs/news_weight_mult${MULT}/results.txt" ]; then
        echo "[$(date)] --- news_weight_mult=${MULT} already done -- skipping ---"
        continue
    fi
    yield_to_user_jobs
    wait_for_memory
    echo "[$(date)] --- news_weight_mult=${MULT} ---"
    python3 -u -m forecasting_task.run_DoubleAdapt --backbone dlinear --lr 0.002 \
        --use_target_novelty --use_related_novelty --news_weight_mult "$MULT" \
        --logdir "logs/news_weight_mult${MULT}"
    echo "[$(date)] news_weight_mult=${MULT} result:"
    cat "logs/news_weight_mult${MULT}/results.txt" 2>/dev/null || echo "(failed)"
done

# --- Stage C: lr=0.002 + reg=1.0 combo (untried point from the HP sweep) ---
if [ ! -f "logs/dlinear_lr002_reg10/results.txt" ]; then
    yield_to_user_jobs
    wait_for_memory
    echo "[$(date)] === Stage C: lr=0.002 + reg=1.0 combo ==="
    python3 -u -m forecasting_task.run_DoubleAdapt --backbone dlinear --lr 0.002 --reg 1.0 \
        --logdir logs/dlinear_lr002_reg10
    cat logs/dlinear_lr002_reg10/results.txt 2>/dev/null || echo "(failed)"
fi

# --- Stage D: alternate primary embedding model sweep ---
echo "[$(date)] === Stage D: alternate primary_emb_model sweep ==="
for EMB in bert gemini qwen; do
    if [ -f "logs/dlinear_emb_${EMB}/results.txt" ]; then
        echo "[$(date)] --- emb=${EMB} already done -- skipping ---"
        continue
    fi
    yield_to_user_jobs
    wait_for_memory
    echo "[$(date)] --- primary_emb_model=${EMB} ---"
    python3 -u -m forecasting_task.run_DoubleAdapt --backbone dlinear --lr 0.002 \
        --primary_emb_model "${EMB}" \
        --logdir "logs/dlinear_emb_${EMB}"
    echo "[$(date)] emb=${EMB} result:"
    cat "logs/dlinear_emb_${EMB}/results.txt" 2>/dev/null || echo "(failed)"
done

# --- Final summary ---
{
    echo "=== overnight_pipeline2 summary ($(date)) ==="
    echo "baseline (DLinear lr=0.002, no news weight): 0.5557 (from logs/dlinear_hp_lr_high, prior run)"
    echo ""
    echo "-- news_weight_mult sweep --"
    for MULT in 2 3 5; do
        echo "mult=${MULT}: $(cat logs/news_weight_mult${MULT}/results.txt 2>/dev/null | tr '\n' ' ')"
    done
    echo ""
    echo "-- lr=0.002 + reg=1.0 --"
    echo "$(cat logs/dlinear_lr002_reg10/results.txt 2>/dev/null | tr '\n' ' ')"
    echo ""
    echo "-- alternate embedding models (lr=0.002) --"
    for EMB in bert gemini qwen; do
        echo "emb=${EMB}: $(cat logs/dlinear_emb_${EMB}/results.txt 2>/dev/null | tr '\n' ' ')"
    done
} > logs/overnight_pipeline2_summary.txt
cat logs/overnight_pipeline2_summary.txt

echo "[$(date)] === OVERNIGHT PIPELINE 2 COMPLETE ==="
