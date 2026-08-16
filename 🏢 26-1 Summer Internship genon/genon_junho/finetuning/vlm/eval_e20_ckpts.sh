#!/bin/bash
# e20 체크포인트 6개를 combined_e20/test.jsonl(픽셀 좌표)로 평가한다.
# - finetuning/vlm 의 patched evaluate_unified 사용(픽셀 프롬프트 bbox_space/W/H 반영).
# - transformers 백엔드(이 venv엔 vllm 없음). GPU 0/2/3 에 3개씩 2웨이브 병렬.
# - 산출: eval_results/e20_ckpt_eval/<ckpt>/metrics_unified.json (layout IoU/F1, TEDS 등)
#
# 사용(tmux 안에서):
#   cd /home/jhyeo/finetuning/vlm && source .venv/bin/activate
#   N=60 bash eval_e20_ckpts.sh
set -uo pipefail
cd /home/jhyeo/finetuning/vlm
source .venv/bin/activate 2>/dev/null || true

BASE=/home/jhyeo/ocr_file_filter/_models/Qwen3.5-9B
TEST=../finetuning_dataset/combined_e20/test.jsonl
CKROOT=output/e20_qwen35_9b_pixel_bboxw/student_sft
OUTROOT=eval_results/e20_ckpt_eval
N="${N:-60}"                 # 평가 이미지 수 (장당 ~230s 이므로 조절)
MAXNEW="${MAXNEW:-4096}"
mkdir -p "$OUTROOT"

# 평가할 체크포인트 (초반 3 + eval최적 3)
CKPTS=(200 500 1000 3700 3800 4000)
# GPU 배치: 0,2,3 순환
GPUS=(0 2 3)

run_one() {  # $1=ckpt  $2=gpu
  local ck=$1 gpu=$2
  local out="$OUTROOT/checkpoint-$ck"
  echo "[$(date '+%T')] start ckpt-$ck on GPU$gpu -> $out"
  CUDA_VISIBLE_DEVICES=$gpu python -m eval.evaluate_unified \
    --test_data "$TEST" \
    --backend transformers \
    --model "$CKROOT/checkpoint-$ck" \
    --base_model "$BASE" \
    --max_samples "$N" \
    --max_new_tokens "$MAXNEW" \
    --max_pixels 262144 --min_pixels 65536 \
    --iou_threshold 0.5 \
    --no-thinking \
    --output_dir "$out" \
    > "$OUTROOT/log_ckpt-$ck.log" 2>&1
  echo "[$(date '+%T')] done ckpt-$ck (GPU$gpu)"
}

# 3개씩 웨이브로 실행(각 웨이브 병렬, 웨이브 종료 대기)
i=0
for ck in "${CKPTS[@]}"; do
  gpu=${GPUS[$((i % 3))]}
  run_one "$ck" "$gpu" &
  i=$((i+1))
  if [ $((i % 3)) -eq 0 ]; then
    wait   # 3개 끝날 때까지 대기 후 다음 웨이브
  fi
done
wait

echo ""
echo "===== 요약 (layout IoU / F1 / TEDS) ====="
for ck in "${CKPTS[@]}"; do
  m="$OUTROOT/checkpoint-$ck/metrics_unified.json"
  if [ -f "$m" ]; then
    python - "$ck" "$m" <<'PY'
import json,sys
ck,m=sys.argv[1],sys.argv[2]
d=json.load(open(m))
lay=d.get("layout",{}); tab=d.get("table",{})
iou=lay.get("avg_layout_mean_iou"); f1=lay.get("avg_layout_f1")
cat=lay.get("avg_layout_category_accuracy"); teds=tab.get("avg_teds")
fmt=lambda x: f"{x:.4f}" if isinstance(x,(int,float)) else str(x)
print(f"ckpt-{ck}: layout_IoU={fmt(iou)}  layout_F1={fmt(f1)}  cat_acc={fmt(cat)}  TEDS={fmt(teds)}")
PY
  else
    echo "ckpt-$ck: (metrics 없음 - 로그 확인 $OUTROOT/log_ckpt-$ck.log)"
  fi
done
