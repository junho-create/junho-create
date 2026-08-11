#!/usr/bin/env bash
# 이슈 doc_parser#318 Phase 3: 표 매칭 2방식(bbox IoU vs OmniDocBench 텍스트) 비교
# — b200에서 실제 3모델(dots_ocr, chandra_layout, chandra_tablelayout)로 실행.
# LLM 불필요. LLM 매칭(pdf-parse-bench)은 엔드포인트 확보 후 --ppb_dir 로 추가.
#
# 전제: Phase 1b 완료 상태 (hckim/tsr_test_issue318 + hckim/upstream_OmniDocBench,
#       result/ 에 pred_md_<tag>_quick_match_table_result.json 존재)
#
# 필요 env:
#   OMNIDOCBENCH_ROOT, GT_JSON, IMAGES_DIR (Phase 1b와 동일)
#   DOTS_PRED                  dots dp-bench pred JSON (fraction_bl 좌표)
#   CHANDRA_LAYOUT_JSONL / CHANDRA_TABLELAYOUT_JSONL  (grid 1000 좌표)
#   PY_EVAL                    dp-bench 평가용 python (rapidfuzz/apted/lxml/bs4/pillow)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
DPB="$HERE/../upstage-dp-bench"
PREP="$HERE/../scripts/prepare/convert_qwen_pred_to_dpbench.py"
OUT_DIR="${OUT_DIR:-$HERE/../table_match_compare_out}"
PY_EVAL="${PY_EVAL:?dp-bench 의존성 설치된 python 경로 필요}"
OMNIDOCBENCH_ROOT="${OMNIDOCBENCH_ROOT:?}"
GT_JSON="${GT_JSON:?}"
IMAGES_DIR="${IMAGES_DIR:?}"

mkdir -p "$OUT_DIR"

# chandra div-HTML -> dp-bench JSON 변환 (없을 때만)
convert_chandra() {  # $1=태그 $2=jsonl
  local out="$OUT_DIR/pred_dpbench_$1.json"
  if [ ! -f "$out" ]; then
    echo "== [$1] predictions_unified.jsonl -> dp-bench JSON"
    "$PY_EVAL" "$PREP" --input "$2" --output "$out"
  fi
  echo "$out"
}

run_one() {  # $1=태그 $2=pred_json $3=pred_coord_space $4=pred_grid
  local tag="$1" pred="$2" space="$3" grid="$4"
  local dump="$OUT_DIR/bbox_match_dump_$tag.json"
  echo "== [$tag] bbox 매칭 평가 (space=$space grid=$grid)"
  (cd "$DPB" && "$PY_EVAL" evaluate.py --mode table \
    --ref_path "$GT_JSON" --pred_path "$pred" \
    --table_match_mode bbox --table_bbox_unmatched_gt skip \
    --gt_coord_space image --pred_coord_space "$space" --pred_grid "$grid" \
    --images_dir "$IMAGES_DIR" \
    --table_match_dump "$dump") | grep -E "bbox-match|TEDS"

  local omni="$OMNIDOCBENCH_ROOT/result/pred_md_${tag}_quick_match_table_result.json"
  echo "== [$tag] bbox vs OmniDocBench 비교"
  "$PY_EVAL" "$HERE/compare_table_matching.py" \
    --bbox_dump "$dump" --omni_result "$omni" \
    --output "$OUT_DIR/disagreements_$tag.json"
}

if [ -n "${DOTS_PRED:-}" ]; then
  # 좌표공간 기본값=fraction_bl(dots.ocr v2 계열). 다른 산출물이면 env 로 교체.
  run_one dots_ocr "$DOTS_PRED" "${DOTS_SPACE:-fraction_bl}" "${DOTS_GRID:-1024}"
fi
if [ -n "${CHANDRA_LAYOUT_JSONL:-}" ]; then
  p=$(convert_chandra chandra_layout "$CHANDRA_LAYOUT_JSONL" | tail -1)
  run_one chandra_layout "$p" grid 1000
fi
if [ -n "${CHANDRA_TABLELAYOUT_JSONL:-}" ]; then
  p=$(convert_chandra chandra_tablelayout "$CHANDRA_TABLELAYOUT_JSONL" | tail -1)
  run_one chandra_tablelayout "$p" grid 1000
fi

echo "완료. 산출물: $OUT_DIR (bbox_match_dump_*.json, disagreements_*.json)"
