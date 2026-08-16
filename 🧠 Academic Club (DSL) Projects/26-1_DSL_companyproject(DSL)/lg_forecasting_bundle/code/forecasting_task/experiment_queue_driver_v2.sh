#!/bin/bash
# Sequential (one-at-a-time) driver -- the container's memory cgroup limit (~120GiB) can't
# survive two parallel panel builds (each peaks ~75-90GiB), so unlike v1 this never overlaps jobs.
set -u
cd /root/LG-AI
SUMMARY=/root/LG-AI/logs/_queue_summary.txt

BASE_ARGS="--backbone dlinear --lr 0.002 --reg 1.0 --use_target_novelty --use_related_novelty --news_weight_mult 10 --freeze_online 1 --seed 7"

run_job() {
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

echo "--- Sequential re-run: consensus=5/6 (rerun, was OOM-killed) ---" | tee -a "$SUMMARY"
run_job 0 logs/dlinear_consensus_5of6 --novelty_consensus_frac 0.8333333333333334
report logs/dlinear_consensus_5of6

echo "--- Sequential re-run: consensus=2/6 (rerun, was killed to free memory) ---" | tee -a "$SUMMARY"
run_job 0 logs/dlinear_consensus_2of6 --novelty_consensus_frac 0.3333333333333333
report logs/dlinear_consensus_2of6

echo "=== Sequential queue finished $(date) ===" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"
echo "=== FINAL SUMMARY (all runs, sorted by hit rate) ===" | tee -a "$SUMMARY"
for d in logs/dlinear_final_compliant logs/dlinear_valid6mo logs/dlinear_consensus_5of6 logs/dlinear_consensus_3of6 logs/dlinear_consensus_2of6; do
    if [ -f "$d/results.txt" ]; then
        hr=$(grep -oP '(?<=weighted_hit_rate=)[0-9.]+' "$d/results.txt" | head -1)
        echo "$hr $d"
    fi
done | sort -rn | tee -a "$SUMMARY"
