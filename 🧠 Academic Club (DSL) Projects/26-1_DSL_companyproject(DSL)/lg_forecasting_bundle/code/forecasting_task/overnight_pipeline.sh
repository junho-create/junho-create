#!/bin/bash
# Fully unattended overnight pipeline:
#   0. Yield to the user's run_large_model.sh (run_newText2Signal) if/when it starts,
#      so the two heavy jobs never compete for the ~128GB cgroup memory limit at once.
#   1. Step5 attribution on the existing full_all_steps_gru checkpoint.
#   2. Decide which Step2-4 feature groups to keep (drop any whose permutation
#      *didn't hurt* hit rate -- i.e. real signal wasn't actually helping).
#   3. Train all 5 backbones on that curated feature set (panel is cached after the
#      first backbone, so backbones 2-5 skip the ~40min rebuild).
#   4. Ensemble the 5 backbones' predictions (avg-price and majority-sign-vote) and
#      report hit rate.
set -uo pipefail
cd /root/LG-AI
LOG=logs/overnight_pipeline.log
exec > "$LOG" 2>&1
echo "[$(date)] overnight_pipeline.sh starting (PID $$)"

# --- Stage 0: yield to the user's own newText2Signal job if it shows up ---
# (matches both run_newText2Signal and run_maxDAnewText2Signal)
if [ -n "${SKIP_STAGE0:-}" ]; then
    echo "[$(date)] SKIP_STAGE0 set -- skipping grace-period wait."
elif pgrep -f "newText2Signal" > /dev/null; then
    echo "[$(date)] newText2Signal detected -- waiting for it to finish before starting."
    while pgrep -f "newText2Signal" > /dev/null; do
        sleep 60
    done
    echo "[$(date)] newText2Signal finished."
else
    echo "[$(date)] Waiting up to 30min for newText2Signal to appear (grace period)..."
    for i in $(seq 1 30); do
        if pgrep -f "newText2Signal" > /dev/null; then
            echo "[$(date)] newText2Signal detected -- waiting for it to finish before starting."
            while pgrep -f "newText2Signal" > /dev/null; do
                sleep 60
            done
            echo "[$(date)] newText2Signal finished."
            break
        fi
        sleep 60
    done
fi
echo "[$(date)] Proceeding with our own pipeline."

# --- Stage 1: attribution ---
ATTR_CSV=logs/full_all_steps_gru/attribution.csv
if [ ! -f "$ATTR_CSV" ]; then
    echo "[$(date)] === Stage 1: attribution ==="
    python3 -u -m forecasting_task.attribution --logdir logs/full_all_steps_gru
fi
if [ ! -f "$ATTR_CSV" ]; then
    echo "[$(date)] ERROR: attribution.csv still missing after running attribution -- aborting."
    exit 1
fi
cat "$ATTR_CSV"

# --- Stage 2: decide feature flags from attribution deltas ---
echo "[$(date)] === Stage 2: deciding feature flags ==="
FLAGS=$(python3 - <<'PYEOF'
import pandas as pd
df = pd.read_csv("logs/full_all_steps_gru/attribution.csv")
df = df[df["group"] != "baseline"]
flag_map = {"macro": "--use_macro", "sector": "--use_sector",
            "target": "--use_target_novelty", "related": "--use_related_novelty"}
keep = [flag_map[row["group"]] for _, row in df.iterrows()
        if row["group"] in flag_map and row["delta"] < 0]
print(" ".join(keep))
PYEOF
)
echo "Selected flags: $FLAGS"
echo "$FLAGS" > logs/overnight_selected_flags.txt

# --- Stage 3: train all 5 backbones on the curated feature set ---
echo "[$(date)] === Stage 3: training backbones ==="
for BB in gru patchtst dlinear timesnet itransformer; do
    echo "[$(date)] --- backbone=$BB ---"
    python3 -u -m forecasting_task.run_DoubleAdapt $FLAGS --backbone "$BB" --logdir "logs/ensemble_$BB"
    echo "[$(date)] $BB result:"
    cat "logs/ensemble_$BB/results.txt" 2>/dev/null || echo "(no results.txt -- this backbone failed)"
done

# --- Stage 4: ensemble ---
echo "[$(date)] === Stage 4: ensembling ==="
python3 - <<'PYEOF'
import numpy as np
import pandas as pd

backbones = ["gru", "patchtst", "dlinear", "timesnet", "itransformer"]
dfs = {}
for bb in backbones:
    try:
        d = pd.read_csv(f"logs/ensemble_{bb}/predictions.csv", parse_dates=["date"]).set_index(["date", "ticker"])
        dfs[bb] = d
        print(f"loaded {bb}: {d.shape}")
    except Exception as e:
        print(f"skip {bb}: {e}")

if not dfs:
    print("No backbone predictions available -- nothing to ensemble.")
else:
    common_idx = None
    for d in dfs.values():
        common_idx = d.index if common_idx is None else common_idx.intersection(d.index)

    any_df = next(iter(dfs.values()))
    anchor = any_df["anchor_close"].reindex(common_idx)
    true_close = any_df["true_close"].reindex(common_idx)

    avg_pred = sum(d["pred_close"].reindex(common_idx) for d in dfs.values()) / len(dfs)
    signs = pd.DataFrame({bb: np.sign(d["pred_close"].reindex(common_idx) - anchor) for bb, d in dfs.items()})
    majority_sign = np.sign(signs.sum(axis=1))

    valid = true_close.notna()
    true_sign = np.sign(true_close[valid] - anchor[valid])
    avg_hit = (np.sign(avg_pred[valid] - anchor[valid]) == true_sign).mean()
    maj_hit = (majority_sign[valid] == true_sign).mean()

    with open("logs/overnight_summary.txt", "w") as f:
        f.write(f"Feature flags used: {open('logs/overnight_selected_flags.txt').read().strip()}\n")
        f.write(f"Backbones ensembled ({len(dfs)}): {list(dfs.keys())}\n")
        f.write(f"Average-price ensemble hit rate: {avg_hit:.4f}\n")
        f.write(f"Majority-vote-of-sign hit rate: {maj_hit:.4f}\n")
        f.write("\nPer-backbone individual results:\n")
        for bb in backbones:
            try:
                with open(f"logs/ensemble_{bb}/results.txt") as rf:
                    f.write(f"  {bb}: {rf.read().strip()}\n")
            except FileNotFoundError:
                f.write(f"  {bb}: FAILED\n")
    print(open("logs/overnight_summary.txt").read())
PYEOF

# --- Stage 5: joint MSE+direction-BCE loss experiment (GRU, curated features) ---
# Tests whether a directional gradient signal on every training batch (JointMSEDirLoss)
# improves test hit rate over plain MSE -- unlike the earlier hit_rate-based checkpoint
# SELECTION experiment (reverted: too noisy on a small validation set), this supplies
# the directional signal densely during training itself.
echo "[$(date)] === Stage 5: joint MSE+direction-BCE loss sweep (GRU) ==="
for AW in 0.1 0.3; do
    echo "[$(date)] --- aux_weight=$AW ---"
    python3 -u -m forecasting_task.run_DoubleAdapt $FLAGS --backbone gru --aux_weight "$AW" \
        --logdir "logs/joint_loss_aw${AW}"
    echo "[$(date)] aux_weight=$AW result:"
    cat "logs/joint_loss_aw${AW}/results.txt" 2>/dev/null || echo "(failed)"
done

{
    echo ""
    echo "=== Stage 5: joint MSE+direction-BCE loss (GRU, same curated features) ==="
    echo "baseline (aux_weight=0, from Stage 3): $(cat logs/ensemble_gru/results.txt 2>/dev/null | tr '\n' ' ')"
    for AW in 0.1 0.3; do
        echo "aux_weight=$AW: $(cat logs/joint_loss_aw${AW}/results.txt 2>/dev/null | tr '\n' ' ')"
    done
} >> logs/overnight_summary.txt

echo "[$(date)] === PIPELINE COMPLETE ==="
