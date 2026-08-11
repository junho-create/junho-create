#!/usr/bin/env bash
# 55600dataextract 의 5개 코퍼스를 **차례로** run_pipeline.sh 에 태워 GT 를 만든다.
# (ArxivFormula 는 별도 수식 전용 파이프라인으로 이미 처리했으므로 제외)
#
#   tmux new-session -d -s gt5 "bash scripts/run_5corpora.sh 2>&1 | tee _work/run_5corpora.log"
#
# 코퍼스당 9,100장 × 5 = 45,500장. batch5400(5,400장, cmcv~2h + judge~3~4h) 기준으로
# 환산하면 코퍼스당 9~11시간, 전체 50시간 안팎이다.
#
# run_pipeline.sh 는 set -euo pipefail 이라 한 코퍼스가 실패하면 거기서 죽는다. 여기서는
# 실패를 기록하고 **다음 코퍼스로 넘어간다** — 10시간짜리 하나가 엎어졌다고 나머지 40시간을
# 버릴 이유가 없다. 모든 단계가 resume-safe 라 나중에 그 코퍼스만 다시 돌리면 이어서 간다.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"

BASE="/home/jhyeo/ocr_filter_result/additional_dataset/55600dataextract"
LOGDIR="$REPO_ROOT/_work/corpora"
mkdir -p "$LOGDIR"

# 사용자가 지정한 순서 그대로. slug|디렉터리명
# multimodal 은 제외 — judge 헬스체크 누락 버그(2026-08-08 수정)로 한 번 망가졌다가
# hardcase judge 재처리 + build_final_output/export/calibrate/tablefix 를 수동으로 다시
# 돌려 이미 완료됐다(final_gt.jsonl 646건). run_pipeline.sh 는 이제 고쳐져 있으니 이
# 목록에서 빠진 건 "건너뛰는 게 아니라 이미 끝났다"는 뜻이다.
#
# 2026-08-10: 사용자 지시로 pubadmin 까지만 돌리고 중단. ocr023public/ocr025finlog/
# paper32 는 나중에 필요하면 이 배열에 다시 넣고 재실행할 것 — 전 단계가 resume-safe라
# unified/cmcv 캐시가 있으면 그대로 재사용된다.
CORPORA=(
  "pubadmin|공공행정문서 OCR"
)

# add_ocr 은 judge 를 내린 뒤(8단계)에 도는지라 GPU 가 비어 있다. CPU 로 돌리면
# 2,262장에 30분이라 9,100장 × 5 코퍼스면 그것만 10시간이 넘는다.
export ADD_OCR_DEVICE="${ADD_OCR_DEVICE:-gpu:0}"

# ── 처리량 튜닝 (전부 이 머신에서 실측하고 넣은 값) ──────────────────────────────
# paddle 사전 배치는 청크(200장)를 순차로 돌고 청크마다 모델을 새로 로드해서, 9,100장이면
# 그것만 3.8시간이다. 4병렬로 1.49시간(0.59 s/page). 클라이언트가 CPU 단일 스레드라
# 병렬화 이득이 그대로 나온다 — paddlevl16_infer/run_all.sh 가 샤드 6개를 쓰는 이유와 같다.
export OCR_FILTER_PADDLE_PARALLEL="${OCR_FILTER_PADDLE_PARALLEL:-4}"
# CMCV 레코드 루프는 workers=4 에서 7.2 s/page(레코드당 target+dots 약 29초). 두 모델이
# 다른 GPU 에 있어 동시요청 경합이 없다. 6단계 judge 의 workers=4 제약과는 무관.
export CMCV_WORKERS="${CMCV_WORKERS:-16}"

step() { echo; echo -e "\033[1;36m########## [$(date '+%m-%d %H:%M:%S')] $* ##########\033[0m"; }

gpu_cleanup() {
  # 우리 서버만 내린다. GPU1(77GB)/GPU3(2GB) 점유는 **다른 컨테이너 소유**라
  # ps 에도 안 잡히고 우리가 못 내린다(PIPELINE.md 함정 6). 그걸 죽이려 들면 안 된다.
  python3 -m ocr_filter.cli models stop --only target external_a external_b judge >/dev/null 2>&1 || true
  sleep 8
  # vLLM 이 죽어도 워커 서브프로세스가 GPU 를 계속 물고 있는 경우가 있다(함정 7).
  # 우리 프로세스 트리에 남은 것만 정리한다.
  if pgrep -f "VLLM::Worker_PP" >/dev/null 2>&1; then
    echo "  잔여 VLLM Worker 정리"
    pkill -KILL -f "VLLM::Worker_PP" || true
    sleep 5
  fi
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed 's/^/  GPU /'
}

declare -a RESULT
for entry in "${CORPORA[@]}"; do
  SLUG="${entry%%|*}"
  NAME="${entry#*|}"
  CONFIG="configs/batch_${SLUG}.yaml"
  INPUT="$BASE/$NAME"
  WORK_DIR="$(python3 -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['paths']['work_dir'])" "$CONFIG")"
  LOG="$LOGDIR/${SLUG}.log"

  step "코퍼스 $SLUG — $NAME"
  echo "  config=$CONFIG"
  echo "  input =$INPUT ($(find "$INPUT" -type f -iname '*.jpg' | wc -l)장)"
  echo "  work  =$WORK_DIR"
  echo "  log   =$LOG"

  gpu_cleanup

  if bash scripts/run_pipeline.sh "$CONFIG" "$INPUT" 0.25 >> "$LOG" 2>&1; then
    # ── uncovered 임계값 재보정 (PIPELINE.md 함정 4) ──
    # 0.25 는 batch5400 캘리브레이션이라 코퍼스가 바뀌면 통과율이 달라진다.
    # 필터/표교체는 모델 호출이 없어 몇 분이면 다시 돈다.
    step "$SLUG — uncovered 임계값 재보정"
    if python3 scripts/calibrate_uncovered.py --work-dir "$WORK_DIR" --apply >> "$LOG" 2>&1; then
      THR=$(grep -o 'CHOSEN_THRESHOLD=[0-9.]*' "$LOG" | tail -1 | cut -d= -f2)
      python3 scripts/apply_table_paddle_refine.py \
        --gt-html "$WORK_DIR/final_gt.jsonl" --unified "$WORK_DIR/unified.jsonl" \
        --cmcv-results "$WORK_DIR/cmcv_results.jsonl" \
        --out "$WORK_DIR/final_gt_tablefix.jsonl" >> "$LOG" 2>&1 \
        && mv "$WORK_DIR/final_gt_tablefix.jsonl" "$WORK_DIR/final_gt.jsonl"
      N=$(wc -l < "$WORK_DIR/final_gt.jsonl" 2>/dev/null || echo 0)
      RESULT+=("OK   $SLUG  final_gt=${N}행  thr=${THR:-?}")
    else
      N=$(wc -l < "$WORK_DIR/final_gt.jsonl" 2>/dev/null || echo 0)
      RESULT+=("WARN $SLUG  재보정 실패, 0.25 결과 유지  final_gt=${N}행")
    fi
  else
    RESULT+=("FAIL $SLUG  — $LOG 확인 (resume-safe: 같은 명령 재실행하면 이어서 감)")
  fi
  echo "  -> ${RESULT[-1]}"
done

gpu_cleanup
step "전체 완료"
printf '%s\n' "${RESULT[@]}"
