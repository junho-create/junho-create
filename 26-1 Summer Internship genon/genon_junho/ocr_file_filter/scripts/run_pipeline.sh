#!/usr/bin/env bash
# 신규 원본 이미지 배치 → 최종 GT jsonl 까지 전체 파이프라인 (batch5400 라이브 실행에서
# 검증됨, 2026-07-31~08-01). 자세한 설명/트러블슈팅은 같은 디렉터리의 PIPELINE.md 참고.
#
# 사용:
#   scripts/run_pipeline.sh <config.yaml> <원본이미지_디렉터리> [최대_uncovered_임계값]
#
# 예:
#   scripts/run_pipeline.sh configs/batch5400.yaml \
#       /home/jhyeo/ocr_filter_result/additional_dataset/5400dataextract 0.25
#
# 전제조건:
#   - <config.yaml> 은 configs/batch5400.yaml 을 복사해 paths.work_dir 만 바꾼 것
#   - GPU 3장(H100 80GB) 필요, 4번째 GPU는 이 컨테이너에서 다른 프로세스가 이미 점유
#     (nvidia-smi 에는 보이지만 우리가 못 내림 — configs/models.yaml external_b 주석 참고)
#   - configs/models.yaml 이 이미 이 머신 기준으로 맞춰져 있음(경로/포트/gpu 배정).
#     다른 머신이면 이 파일부터 다시 검증할 것.
set -euo pipefail

CONFIG="${1:?사용법: run_pipeline.sh <config.yaml> <input_dir> [max_uncovered_new]}"
INPUT_DIR="${2:?사용법: run_pipeline.sh <config.yaml> <input_dir> [max_uncovered_new]}"
MAX_UNCOVERED_NEW="${3:-0.25}"   # 0.05(기본값)는 구형 파이프라인 캘리브레이션 — 이 새
                                  # 파이프라인(dots.ocr 직접 채택 포함)엔 안 맞는다.
                                  # PIPELINE.md "임계값 재보정" 절 참고, 배치마다 재확인 권장.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"

WORK_DIR="$(python3 -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['paths']['work_dir'])" "$CONFIG")"
echo "== config=$CONFIG  input=$INPUT_DIR  work_dir=$WORK_DIR  max_uncovered_new=$MAX_UNCOVERED_NEW =="
mkdir -p "$WORK_DIR"

step() { echo; echo "########## $* ##########"; }

# ── 1) io: jpg 직접 인제스트 (렌더링 없음, PDF 아님) ─────────────────────────────
step "1/9 io build-images"
python3 -m ocr_filter.cli io build-images \
    --config "$CONFIG" --input-dir "$INPUT_DIR" --out "$WORK_DIR/unified.jsonl"

# ── 2) CMCV 3모델 기동 (target/dots.ocr/paddle, GPU 0/2/3) ─────────────────────
step "2/9 models serve (target external_a external_b)"
python3 -m ocr_filter.cli models serve --only target external_a external_b
python3 -m ocr_filter.cli models status || true

# ── 3) CMCV 실행 (resume-safe: 재실행하면 done id 건너뜀) ──────────────────────
step "3/9 cmcv run"
# CMCV 는 레코드당 target(GPU0) + dots(GPU2) 를 **순차로** 부른다. 두 서버가 서로 다른
# GPU 라 동시요청이 겹쳐도 경합이 없고 vLLM 이 알아서 배치한다 — 6단계 hardcase judge 의
# workers=4 제약(Chromium 인스턴스 경합)은 여기엔 해당되지 않는다.
# batch5400 은 4로 돌았지만 대형 스캔 이미지 코퍼스에선 레코드당 ~29초라 4로는 7.2 s/page
# 밖에 안 나온다(실측). 큰 배치에서는 CMCV_WORKERS 로 올릴 것.
python3 -m ocr_filter.cli cmcv run --config "$CONFIG" \
    --unified "$WORK_DIR/unified.jsonl" --out "$WORK_DIR/cmcv_results.jsonl" \
    --workers "${CMCV_WORKERS:-4}"

# ── 4) rescue: Hard 티어 N:M 그룹 매칭 (CPU 전용, GPU 서버 살아있어도 무관) ─────
step "4/9 rescue"
python3 -m ocr_filter.cli rescue run --config "$CONFIG" \
    --unified "$WORK_DIR/unified.jsonl" --cmcv-results "$WORK_DIR/cmcv_results.jsonl" \
    --out "$WORK_DIR/rescue_results.jsonl"

# ── 5) CMCV 3모델 내리고 judge(122B, GPU 0+2+3 전부) 기동 ──────────────────────
#     3-GPU 제약 때문에 동시 서빙 불가 — 반드시 순차.
step "5/9 models stop (cmcv) -> serve (judge)"
python3 -m ocr_filter.cli models stop --only target external_a external_b
sleep 5
python3 -m ocr_filter.cli models serve --only judge

# `models status` 는 헬스체크를 딱 한 번만 찍고 지나간다 — 122B(PP=3)는 로딩에
# 2~3분 걸리는데(실측 165초) 여기서 기다리지 않고 바로 6단계로 넘어가면 hardcase
# judge 가 서버가 아직 안 뜬 상태로 시작해서 전체 레코드가 ConnectionError 로
# 즉시 실패한다(resolved=False 100%) — 실제로 55600dataextract 5코퍼스 배치 중
# multimodal 코퍼스에서 이렇게 Hard 티어 3,192건이 통째로 날아간 뒤 발견했다.
# 준비될 때까지 폴링해서 막는다.
echo "judge(122B) 로딩 대기..."
JUDGE_READY=0
for i in $(seq 1 60); do
  if curl -s -m 5 http://127.0.0.1:8004/v1/models 2>/dev/null | grep -q "Qwen"; then
    echo "  준비됨 ($((i * 15))초)"
    JUDGE_READY=1
    break
  fi
  sleep 15
done
python3 -m ocr_filter.cli models status || true
if [ "$JUDGE_READY" -ne 1 ]; then
  echo "judge 서버가 15분 안에 준비되지 않았다 — _work/serve/judge.log 확인 필요" >&2
  exit 1
fi

# ── 6) hardcase judge: workers=4 고정 ───────────────────────────────────────────
#     workers=32 로 돌리면 render-then-verify(playwright chromium, 표 렌더용)가
#     동시 인스턴스 경합으로 대량 실패한다(2026-07-31 실측, 표 렌더 실패 급증 후
#     확인). 4 는 batch5400(3,483건) 전체를 에러 없이 완주 검증된 값 — 늘리고
#     싶으면 Chromium 인스턴스 수부터 별도로 검증할 것.
step "6/9 hardcase judge (workers=4, ETA 약 3~6시간/3000여건)"
python3 -m ocr_filter.cli hardcase judge --config "$CONFIG" --workers 4
python3 -m ocr_filter.cli models stop --only judge

# ── 7) 3단계 결과 머지 → final_dataset.jsonl ────────────────────────────────────
step "7/9 build_final_output.py"
python3 scripts/build_final_output.py \
    --cmcv-results "$WORK_DIR/cmcv_results.jsonl" \
    --rescue "$WORK_DIR/rescue_results.jsonl" \
    --hardcase-judge "$WORK_DIR/hardcase_judge.jsonl" \
    --out "$WORK_DIR/final_dataset.jsonl"

# ── 8) export gt-html → add_ocr(OCR 커버리지 부여, GPU 없으면 CPU) → filter_e21grid ──
step "8/9 export -> add_ocr -> filter_e21grid"
python3 -m ocr_filter.cli export gt-html \
    --final-dataset "$WORK_DIR/final_dataset.jsonl" --unified "$WORK_DIR/unified.jsonl" \
    --out "$WORK_DIR/gt_html.jsonl"

# add_ocr: GPU 가 CMCV/judge 로 이미 꽉 찼을 수 있어 기본 CPU. GPU 비었으면
# --ocr_device gpu:0 등으로 바꿔서 훨씬 빠르게 돌릴 수 있음 (PIPELINE.md 참고).
( cd /home/jhyeo/finetuning/vlm && python3 -m data.add_ocr \
    --input "$WORK_DIR/gt_html.jsonl" --output "$WORK_DIR/with_ocr.jsonl" \
    --ocr_device "${ADD_OCR_DEVICE:-cpu}" --ocr_lang korean --bbox_scale 1024 )

# filter_e21grid.py 는 디렉터리+split 파일명(train.jsonl 등) 구조를 요구 — 단일
# 파일이면 이렇게 감싸야 함.
mkdir -p "$WORK_DIR/_e21grid_in"
cp "$WORK_DIR/with_ocr.jsonl" "$WORK_DIR/_e21grid_in/train.jsonl"
python3 /home/jhyeo/finetuning/finetuning_dataset/filter_e21grid.py \
    --data_dir "$WORK_DIR/_e21grid_in" --out_dir "$WORK_DIR/_e21grid_out" \
    --splits train --max_uncovered_new "$MAX_UNCOVERED_NEW"
cp "$WORK_DIR/_e21grid_out/train.jsonl" "$WORK_DIR/final_gt.jsonl"

# ── 9) 표 태그를 paddle 로 교체 (combined_e24_table_refined 규칙 이식) ──────────
step "9/9 table paddle refine"
python3 scripts/apply_table_paddle_refine.py \
    --gt-html "$WORK_DIR/final_gt.jsonl" \
    --unified "$WORK_DIR/unified.jsonl" \
    --cmcv-results "$WORK_DIR/cmcv_results.jsonl" \
    --out "$WORK_DIR/final_gt_tablefix.jsonl"
mv "$WORK_DIR/final_gt.jsonl" "$WORK_DIR/final_gt_before_tablefix.jsonl"
mv "$WORK_DIR/final_gt_tablefix.jsonl" "$WORK_DIR/final_gt.jsonl"

echo
echo "== 완료: $WORK_DIR/final_gt.jsonl =="
wc -l "$WORK_DIR/final_gt.jsonl"
