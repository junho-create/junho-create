#!/bin/bash

LOG_FILE="/root/run_maxDA.log"

echo "Checking for other claude processes..." > $LOG_FILE

while pgrep -f "claude" | grep -v $$ > /dev/null; do
    echo "Other process is still running. Waiting 3600 seconds..." >> $LOG_FILE
    sleep 3600
done

echo "All other processes finished. Starting Max DA model training..." >> $LOG_FILE

export CUDA_VISIBLE_DEVICES=0

# Run the training script with larger architecture sizes AND new DA Loss
nohup python3 -u -m forecasting_task.run_maxDAnewText2Signal \
    --data_path /root/LG-AI/data/test.csv \
    --emb_dir /root/LG-AI/data \
    --primary_emb gemini_textemb.parquet \
    --n_clusters 11 \
    --num_epoch 30 \
    --text_epochs 15 \
    --text_hidden 512 \
    --text_layers 4 \
    --text_heads 8 \
    --d_model 256 \
    --n_heads 8 \
    --e_layers 4 \
    --d_ff 512 \
    >> $LOG_FILE 2>&1 &

echo "Max DA Training process launched with PID $!" >> $LOG_FILE
