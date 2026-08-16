#!/bin/bash
# Tests the new supervised text-direction signal (ported from V7Model's
# TextClassifier, forecasting_task/preprocessing/text_signal.py) on top of the
# Step1+textPCA feature set, on both of today's two most relevant backbones:
# DLinear (today's best plain result, 0.5495) and GRU (today's joint-loss
# testbed, 0.5265 plain / 0.5336 with aux_weight=0.3).
set -uo pipefail
cd /root/LG-AI
LOG=logs/text_signal_experiment.log
exec > "$LOG" 2>&1
echo "[$(date)] text_signal_experiment.sh starting (PID $$)"

for BB in dlinear gru; do
    echo "[$(date)] --- backbone=$BB + use_text_signal ---"
    python3 -u -m forecasting_task.run_DoubleAdapt --backbone "$BB" --use_text_signal \
        --logdir "logs/${BB}_textsignal"
    echo "[$(date)] $BB+textsignal result:"
    cat "logs/${BB}_textsignal/results.txt" 2>/dev/null || echo "(failed)"
done

echo "[$(date)] === attribution on dlinear+textsignal checkpoint ==="
python3 -u -m forecasting_task.attribution --logdir logs/dlinear_textsignal
cat logs/dlinear_textsignal/attribution.csv 2>/dev/null

{
    echo ""
    echo "=== Text-direction signal (V7-ported, supervised pretrained classifier) ==="
    echo "baseline dlinear (no text_signal, from ensemble_dlinear): $(cat logs/ensemble_dlinear/results.txt 2>/dev/null | tr '\n' ' ')"
    echo "dlinear + text_signal: $(cat logs/dlinear_textsignal/results.txt 2>/dev/null | tr '\n' ' ')"
    echo "baseline gru (no text_signal, from ensemble_gru): $(cat logs/ensemble_gru/results.txt 2>/dev/null | tr '\n' ' ')"
    echo "gru + text_signal: $(cat logs/gru_textsignal/results.txt 2>/dev/null | tr '\n' ' ')"
} >> logs/overnight_summary.txt

echo "[$(date)] === TEXT SIGNAL EXPERIMENT COMPLETE ==="
