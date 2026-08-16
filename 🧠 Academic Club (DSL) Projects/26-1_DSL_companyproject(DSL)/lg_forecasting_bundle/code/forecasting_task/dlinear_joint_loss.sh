#!/bin/bash
# Follow-up: DLinear was today's best backbone (0.5495 hit rate); joint MSE+direction-BCE
# loss gave a monotonic hit-rate improvement on GRU. Test whether combining the two
# (best backbone + the loss that helped) stacks further.
set -uo pipefail
cd /root/LG-AI
LOG=logs/dlinear_joint_loss.log
exec > "$LOG" 2>&1
echo "[$(date)] dlinear_joint_loss.sh starting (PID $$)"

for AW in 0.1 0.3; do
    echo "[$(date)] --- backbone=dlinear aux_weight=$AW ---"
    python3 -u -m forecasting_task.run_DoubleAdapt --backbone dlinear --aux_weight "$AW" \
        --logdir "logs/dlinear_joint_aw${AW}"
    echo "[$(date)] aux_weight=$AW result:"
    cat "logs/dlinear_joint_aw${AW}/results.txt" 2>/dev/null || echo "(failed)"
done

{
    echo ""
    echo "=== DLinear + joint MSE+direction-BCE loss ==="
    echo "baseline (aux_weight=0, from ensemble_dlinear): $(cat logs/ensemble_dlinear/results.txt 2>/dev/null | tr '\n' ' ')"
    for AW in 0.1 0.3; do
        echo "aux_weight=$AW: $(cat logs/dlinear_joint_aw${AW}/results.txt 2>/dev/null | tr '\n' ' ')"
    done
} >> logs/overnight_summary.txt

echo "[$(date)] === DLINEAR JOINT LOSS SWEEP COMPLETE ==="
