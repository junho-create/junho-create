#!/usr/bin/env bash
# GitLab(task/272-teds-bbox-matching) run_eval_genos500.sh 를 로컬 저장소에 맞춘 버전.
# dp-bench(NID/TEDS) 평가를 직접 돌려 <OUT_DIR>/eval_result_gitlab.txt 를 생성한다.
# 사용: 아래 ▼설정▼ 만 바꾸고  ->  bash scripts/run_eval_genos500.sh
#
# 평가 코드는 GitLab 그대로 가져온 upstage-dp-bench/{evaluate.py, src/table_evaluation.py} 사용.

set -euo pipefail
cd "$(dirname "$0")/.."   # genos500_dpbench 루트로 이동

# ============================ ▼ 여기만 보고 바꾸면 됨 ▼ ============================
# 평가할 모델 폴더명 (dp_out/eval_excluded_8docs_v2/<MODEL> 안의 pred_dp_bench_full.json 사용)
MODEL="${MODEL:-unified_base_ocr_20260617}"   # dots_ocr | unified_base_20260616 | unified_base_ocr_20260617

PYTHON="${PYTHON:-python3}"
REF="${REF:-genos-500-set/reference_dp_bench.json}"          # 정답(GT) dp-bench json
IMAGES_DIR="${IMAGES_DIR:-genos-500-set/qwen_infer_images}"  # GT=image 정규화용. qwen_infer_images=모델 실제 추론 이미지(DCTN, 권장). page_images 도 가능

# --- 테이블(TEDS) 매칭 방식 (이슈 #272) ---
TABLE_MATCH_MODE="${TABLE_MATCH_MODE:-bbox}"                 # index(기존) | bbox(테이블별 IoU 매칭)
# 미검출 GT 처리 방식: 공백으로 여러 개 지정하면 한 번에 모두 계산해 결과에 함께 기록.
#   zero=미검출 0점(옵션 b) | skip=미검출 제외(옵션 a)
TABLE_BBOX_UNMATCHED_GTS="${TABLE_BBOX_UNMATCHED_GTS:-zero skip}"
TABLE_BBOX_IOU_THR="${TABLE_BBOX_IOU_THR:-0.5}"             # 매칭 인정 최소 IoU

# --- 좌표계 정규화 (bbox 모드에서만 사용) ---
#   dots_ocr            : GT=image / PRED=fraction_bl
#   unified_base_*      : GT=image / PRED=grid (PRED_GRID=1024)
GT_COORD_SPACE="${GT_COORD_SPACE:-image}"
PRED_COORD_SPACE="${PRED_COORD_SPACE:-grid}"                # dots_ocr 이면 fraction_bl 로 바꿀 것
# 현 chandra HTML 출력은 0~1000 정규화 그리드. (레거시 unified=1024 모델 평가 시 PRED_GRID=1024 로 오버라이드)
PRED_GRID="${PRED_GRID:-1000}"                             # chandra(0~1000 그리드 출력)

# --- NID 표 텍스트 주입 (레이아웃 평가) ---
# true(기본): 표/pred 정렬이 어긋나면 표 HTML 을 텍스트로 변환해 NID 비교에 포함(레거시 호환).
# false: 표 텍스트를 NID 에서 완전히 제외(표는 오직 TEDS 로만 평가). chandra 평가는 false 사용.
NID_CONVERT_TABLE_TEXT="${NID_CONVERT_TABLE_TEXT:-true}"
# ============================ ▲ 설정 끝 ▲ ============================

# 출력 루트(기본: 레거시 excluded 폴더). 500개 전체 평가 등 새 폴더로 보내려면 OUT_ROOT 오버라이드.
OUT_ROOT="${OUT_ROOT:-dp_out/eval_excluded_8docs_v2}"
OUT_DIR="$OUT_ROOT/$MODEL"
PRED="$OUT_DIR/pred_dp_bench_full.json"
RESULT="$OUT_DIR/eval_result_gitlab.txt"

[ -f "$PRED" ] || { echo "pred 없음: $PRED"; exit 1; }
cd upstage-dp-bench   # evaluate.py 는 'from src...' import 때문에 이 폴더에서 실행해야 함
REF="../$REF"; PRED="../$PRED"; IMAGES_DIR="../$IMAGES_DIR"

n_steps=$(( 1 + $(echo $TABLE_BBOX_UNMATCHED_GTS | wc -w) ))
echo "[1/$n_steps] layout 평가 (NID, convert_table_text=$NID_CONVERT_TABLE_TEXT)"
NID_FLAGS=""
if [ "$NID_CONVERT_TABLE_TEXT" = "true" ]; then
  NID_FLAGS="--convert_table_to_text_for_index --convert_table_to_text_on_mismatch"
fi
NID=$("$PYTHON" evaluate.py \
  --ref_path "$REF" --pred_path "$PRED" --mode layout \
  $NID_FLAGS | grep -i "NID Score")

# 미검출 처리 방식별로 TEDS 를 각각 계산해 모은다.
TEDS_BLOCKS=""
step=2
for UG in $TABLE_BBOX_UNMATCHED_GTS; do
  echo "[$step/$n_steps] table 평가 (TEDS, $TABLE_MATCH_MODE, unmatched_gt=$UG)"
  OUT=$("$PYTHON" evaluate.py \
    --ref_path "$REF" --pred_path "$PRED" --mode table \
    --table_match_mode "$TABLE_MATCH_MODE" \
    --table_bbox_unmatched_gt "$UG" \
    --table_bbox_iou_thr "$TABLE_BBOX_IOU_THR" \
    --gt_coord_space "$GT_COORD_SPACE" \
    --pred_coord_space "$PRED_COORD_SPACE" \
    --pred_grid "$PRED_GRID" \
    --images_dir "$IMAGES_DIR" \
    | grep -iE "bbox-match|TEDS Score|TEDS-S Score|Warning")
  label="(b) zero=미검출 0점"; [ "$UG" = "skip" ] && label="(a) skip=미검출 제외"
  TEDS_BLOCKS="${TEDS_BLOCKS}--- unmatched_gt=${UG} ${label} ---
${OUT}

"
  step=$((step+1))
done

{
  echo "model : $MODEL"
  echo "pred  : $PRED"
  echo "ref   : $REF"
  echo "table_match_mode : $TABLE_MATCH_MODE (iou_thr=$TABLE_BBOX_IOU_THR)"
  echo "coord : gt=$GT_COORD_SPACE pred=$PRED_COORD_SPACE pred_grid=$PRED_GRID"
  echo "nid_convert_table_text : $NID_CONVERT_TABLE_TEXT"
  echo "images_dir : $IMAGES_DIR"
  echo "=============================="
  echo "$NID"
  echo "------------------------------"
  printf '%s' "$TEDS_BLOCKS"
} | tee "../$RESULT"

echo ""
echo "완료 -> $OUT_DIR/eval_result_gitlab.txt"
