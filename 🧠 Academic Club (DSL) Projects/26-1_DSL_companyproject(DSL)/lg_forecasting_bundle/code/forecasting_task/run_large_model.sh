#!/bin/bash

LOG_FILE="/root/run_test_large.log"

echo "Waiting for other claude/python processes to finish..." > $LOG_FILE

# Loop until 'claude' process is no longer running (excluding our own grep)
while pgrep -f "claude" | grep -v $$ > /dev/null; do
    echo "Other process is still running. Waiting 60 seconds..." >> $LOG_FILE
    sleep 60
done

echo "All other processes finished. Starting large model training..." >> $LOG_FILE

export CUDA_VISIBLE_DEVICES=0

# Run the training script with larger architecture sizes
nohup python3 -u -m forecasting_task.run_newText2Signal \
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

echo "Training process launched with PID $!" >> $LOG_FILE
