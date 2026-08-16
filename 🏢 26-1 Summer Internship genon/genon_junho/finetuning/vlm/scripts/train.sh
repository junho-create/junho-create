
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NUM_GPUS=4 \
MASTER_PORT=29651 \
WANDB_MODE=online \
WANDB_PROJECT=tsr_vlm_train \
CONFIG=config/exp_e17_6910.yaml \
PHASE=sft \
bash distill/run_distill.sh 2>&1 | tee logs/train_e17_$(date +%Y%m%d_%H%M%S).log
