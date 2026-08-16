#!/bin/bash
# Autonomous experiment driver: runs a queue of run_DoubleAdapt.py jobs, 2 at a time
# (one per GPU), and writes a running summary to /root/LG-AI/logs/_queue_summary.txt.
set -u
cd /root/LG-AI
SUMMARY=/root/LG-AI/logs/_queue_summary.txt
echo "=== Experiment queue started $(date) ===" > "$SUMMARY"

BASE_ARGS="--backbone dlinear --lr 0.002 --reg 1.0 --use_target_novelty --use_related_novelty --news_weight_mult 10 --freeze_online 1 --seed 7"

run_job() {
    # run_job <gpu> <logdir> <extra args...>
    local gpu=$1; local logdir=$2; shift 2
    CUDA_VISIBLE_DEVICES=$gpu python3 -u -m forecasting_task.run_DoubleAdapt $BASE_ARGS --logdir "$logdir" "$@" \
        > "/tmp/queue_$(basename $logdir).log" 2>&1
}

report() {
    local logdir=$1
    if [ -f "$logdir/results.txt" ]; then
        local hr=$(grep -oP '(?<=weighted_hit_rate=)[0-9.]+' "$logdir/results.txt" | head -1)
        echo "$(date '+%H:%M:%S') $logdir -> weighted_hit_rate=$hr" | tee -a "$SUMMARY"
    else
        echo "$(date '+%H:%M:%S') $logdir -> FAILED (no results.txt, check /tmp/queue_$(basename $logdir).log)" | tee -a "$SUMMARY"
    fi
}

echo "--- Round 1: 6-month-valid retrain (GPU0) + consensus=5/6 on baseline split (GPU1) ---" | tee -a "$SUMMARY"
run_job 0 logs/dlinear_valid6mo \
    --train_start 2019-01-01 --train_end 2022-06-30 --valid_start 2022-07-01 --valid_end 2022-12-31 \
    --test_start 2023-01-01 --test_end 2023-12-01 &
PID0=$!
run_job 1 logs/dlinear_consensus_5of6 --novelty_consensus_frac 0.8333333333333334 &
PID1=$!
wait $PID0 $PID1
report logs/dlinear_valid6mo
report logs/dlinear_consensus_5of6

echo "--- Round 2: consensus=3/6 (GPU0) + consensus=2/6 (GPU1) on baseline split ---" | tee -a "$SUMMARY"
run_job 0 logs/dlinear_consensus_3of6 --novelty_consensus_frac 0.5 &
PID0=$!
run_job 1 logs/dlinear_consensus_2of6 --novelty_consensus_frac 0.3333333333333333 &
PID1=$!
wait $PID0 $PID1
report logs/dlinear_consensus_3of6
report logs/dlinear_consensus_2of6

# Decide whether the 6-month-valid split beat the 0.5900 baseline.
BASELINE=0.5900
VALID6MO_HR=$(grep -oP '(?<=weighted_hit_rate=)[0-9.]+' logs/dlinear_valid6mo/results.txt 2>/dev/null | head -1)
echo "--- Checking whether valid6mo ($VALID6MO_HR) beat baseline ($BASELINE) ---" | tee -a "$SUMMARY"

BEAT=$(python3 -c "print(1 if float('$VALID6MO_HR' or 0) > $BASELINE else 0)" 2>/dev/null || echo 0)

if [ "$BEAT" = "1" ]; then
    echo "valid6mo split WON ($VALID6MO_HR > $BASELINE) -- re-running consensus sweep on THIS split" | tee -a "$SUMMARY"
    NEWSPLIT="--train_start 2019-01-01 --train_end 2022-06-30 --valid_start 2022-07-01 --valid_end 2022-12-31 --test_start 2023-01-01 --test_end 2023-12-01"

    echo "--- Round 3: consensus=5/6 + 3/6 on WINNING split ---" | tee -a "$SUMMARY"
    run_job 0 logs/dlinear_v6mo_consensus_5of6 $NEWSPLIT --novelty_consensus_frac 0.8333333333333334 &
    PID0=$!
    run_job 1 logs/dlinear_v6mo_consensus_3of6 $NEWSPLIT --novelty_consensus_frac 0.5 &
    PID1=$!
    wait $PID0 $PID1
    report logs/dlinear_v6mo_consensus_5of6
    report logs/dlinear_v6mo_consensus_3of6

    echo "--- Round 4: consensus=2/6 on WINNING split ---" | tee -a "$SUMMARY"
    run_job 0 logs/dlinear_v6mo_consensus_2of6 $NEWSPLIT --novelty_consensus_frac 0.3333333333333333 &
    PID0=$!
    wait $PID0
    report logs/dlinear_v6mo_consensus_2of6
else
    echo "valid6mo split did NOT beat baseline ($VALID6MO_HR vs $BASELINE) -- consensus sweep on baseline split (already done above) is final" | tee -a "$SUMMARY"
fi

echo "=== Experiment queue finished $(date) ===" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"
echo "=== FINAL SUMMARY (all runs, sorted by hit rate) ===" | tee -a "$SUMMARY"
for d in logs/dlinear_final_compliant logs/dlinear_valid6mo logs/dlinear_consensus_5of6 logs/dlinear_consensus_3of6 logs/dlinear_consensus_2of6 logs/dlinear_v6mo_consensus_5of6 logs/dlinear_v6mo_consensus_3of6 logs/dlinear_v6mo_consensus_2of6; do
    if [ -f "$d/results.txt" ]; then
        hr=$(grep -oP '(?<=weighted_hit_rate=)[0-9.]+' "$d/results.txt" | head -1)
        echo "$hr $d"
    fi
done | sort -rn | tee -a "$SUMMARY"
