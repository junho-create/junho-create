#!/usr/bin/env bash
# 이슈 doc_parser#318: genos 500-set 3모델을 OmniDocBench(원본)로 평가하는 wrapper.
#
# 흐름: GT 변환(1회) -> 모델별 pred 변환 -> 모델별 config 생성 -> pdf_validation.py 실행.
# 결과는 $OMNIDOCBENCH_ROOT/result/ 에 "<pred폴더명>_quick_match_*" 로 남는다.
#
# 필요 env:
#   OMNIDOCBENCH_ROOT  upstream OmniDocBench 클론 경로(.venv 포함, CDM 제외 실행)
#   GT_JSON            reference_dp_bench.json 경로
#   IMAGES_DIR         500장 이미지 폴더
#   OUT_DIR            변환 산출물 폴더 (기본 ./omnidocbench_out)
#   DOTS_PRED          dots.ocr dp-bench pred JSON (없으면 스킵)
#   CHANDRA_LAYOUT_JSONL / CHANDRA_TABLELAYOUT_JSONL  predictions_unified.jsonl (없으면 스킵)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
OMNIDOCBENCH_ROOT="${OMNIDOCBENCH_ROOT:?upstream OmniDocBench 경로 필요}"
GT_JSON="${GT_JSON:?reference_dp_bench.json 경로 필요}"
IMAGES_DIR="${IMAGES_DIR:?이미지 폴더 필요}"
OUT_DIR="${OUT_DIR:-$HERE/../omnidocbench_out}"
PY="${PY:-$OMNIDOCBENCH_ROOT/.venv/bin/python}"

mkdir -p "$OUT_DIR"

# 1) GT 변환 (1회)
GT_OMNI="$OUT_DIR/gt_omnidocbench.json"
if [ ! -f "$GT_OMNI" ]; then
  "$PY" "$HERE/convert_gt_to_omnidocbench.py" \
    --gt "$GT_JSON" --images_dir "$IMAGES_DIR" --output "$GT_OMNI"
fi

# WITH_CDM=1 이면 display_formula에 CDM을 추가한다.
# 전제: pdflatex(+한글 CJK)·magick(또는 shim)·gs 가 PATH에 있어야 함.
# 상세 절차: CDM_SETUP_b200.md
CDM_METRIC_LINES=""
if [ "${WITH_CDM:-0}" = "1" ]; then
  CDM_METRIC_LINES="
      - CDM
      cdm_workers: ${CDM_WORKERS:-4}"
fi

run_one() {  # $1=태그 $2=포맷 $3=pred 경로
  local tag="$1" fmt="$2" pred="$3"
  local md_dir="$OUT_DIR/pred_md_$tag"
  echo "== [$tag] pred 변환 ($fmt)"
  "$PY" "$HERE/convert_pred_to_markdown.py" --format "$fmt" --input "$pred" --output_dir "$md_dir"

  local cfg="$OUT_DIR/config_$tag.yaml"
  cat > "$cfg" <<EOF
end2end_eval:
  metrics:
    text_block:
      metric:
      - Edit_dist
    display_formula:
      metric:
      - Edit_dist${CDM_METRIC_LINES}
    table:
      metric:
      - TEDS
      - Edit_dist
      teds_workers: ${TEDS_WORKERS:-8}
    reading_order:
      metric:
      - Edit_dist
  dataset:
    dataset_name: end2end_dataset
    ground_truth:
      data_path: $GT_OMNI
    prediction:
      data_path: $md_dir
    match_method: quick_match
    match_workers: ${MATCH_WORKERS:-8}
    quick_match_truncated_timeout_sec: 300
    match_timeout_sec: 420
    timeout_fallback_max_chunk_span: 10
    timeout_fallback_order_penalty: 0.10
EOF
  echo "== [$tag] OmniDocBench 평가 실행"
  (cd "$OMNIDOCBENCH_ROOT" && "$PY" pdf_validation.py --config "$cfg")
}

# 한 모델이 실패해도 나머지는 계속 실행하고, 마지막에 실패 목록을 보고한다.
FAILED=""
if [ -n "${DOTS_PRED:-}" ]; then
  run_one dots_ocr dpbench "$DOTS_PRED" || { echo "== [dots_ocr] 실패"; FAILED="$FAILED dots_ocr"; }
fi
if [ -n "${CHANDRA_LAYOUT_JSONL:-}" ]; then
  run_one chandra_layout chandra "$CHANDRA_LAYOUT_JSONL" || { echo "== [chandra_layout] 실패"; FAILED="$FAILED chandra_layout"; }
fi
if [ -n "${CHANDRA_TABLELAYOUT_JSONL:-}" ]; then
  run_one chandra_tablelayout chandra "$CHANDRA_TABLELAYOUT_JSONL" || { echo "== [chandra_tablelayout] 실패"; FAILED="$FAILED chandra_tablelayout"; }
fi
# 임의의 dp-bench 형식 pred 추가 평가: EXTRA_PREDS="태그1=/경로1.json 태그2=/경로2.json"
# (MinerU/PaddleOCR-VL 등 dp-bench JSON만 있으면 어떤 모델이든 같은 파이프라인으로 측정)
for item in ${EXTRA_PREDS:-}; do
  tag="${item%%=*}"; pred="${item#*=}"
  run_one "$tag" dpbench "$pred" || { echo "== [$tag] 실패"; FAILED="$FAILED $tag"; }
done

if [ -n "$FAILED" ]; then
  echo "실패 모델:$FAILED (로그 확인 필요)"
  exit 1
fi
echo "완료. 결과: $OMNIDOCBENCH_ROOT/result/pred_md_<tag>_quick_match_*"
