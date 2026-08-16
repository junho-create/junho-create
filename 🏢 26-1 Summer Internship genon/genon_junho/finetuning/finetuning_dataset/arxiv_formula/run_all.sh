#!/usr/bin/env bash
# ArxivFormula 수식 GT 전체 파이프라인. tmux 안에서 돌린다.
#
#   tmux new-session -d -s afpipe "bash run_all.sh 2>&1 | tee logs/pipeline.log"
#
# 전제: dots(8119)/paddle(8118) 추론이 이미 돌고 있거나 끝나 있을 것.
# 이 스크립트는 추론이 끝나기를 기다린 뒤 나머지를 쭉 진행한다.
set -uo pipefail

WD="$(cd "$(dirname "$0")" && pwd)"
OFR=/home/jhyeo/ocr_file_filter
cd "$WD"

step() { echo -e "\n\033[1;36m=== [$(date '+%H:%M:%S')] $* ===\033[0m"; }
die()  { echo -e "\033[1;31m[FATAL] $*\033[0m"; exit 1; }

# ── 1. 추론 완료 대기 ─────────────────────────────────────────────────────────
step "1/8 추론 완료 대기 (dots + paddle)"
while pgrep -f "run_infer.py .*arxiv_formula" > /dev/null; do
  D=$(cat dots_out/*.jsonl 2>/dev/null | wc -l)
  P=$(cat paddle_out/*.jsonl 2>/dev/null | wc -l)
  echo "  [$(date '+%H:%M:%S')] dots $D / 9100,  paddle $P / 9100"
  sleep 120
done
echo "  추론 프로세스 종료됨"

step "2/8 paddle 샤드 병합"
# 병합은 한 번만 필요하다. merge_shards.py 는 성공해도 샤드를 지우므로, 재실행 시에는
# 샤드가 없는 게 정상이다 — 그때 다시 부르면 "빠진 페이지" 로 오판해 실패한다.
if ls paddle_out/*.shard*.jsonl > /dev/null 2>&1; then
  python3 "$WD/../paddlevl16_infer/merge_shards.py" \
    --dir "$WD/paddle_out" --manifest "$WD/manifest.jsonl" || echo "  (병합 경고 — 아래 커버리지로 판정)"
else
  echo "  샤드 없음 — 이미 병합됨"
fi
# 판정은 merge 의 종료코드가 아니라 **실제 커버리지**로 한다.
python3 - <<'PY' || die "커버리지 부족"
import json, sys
from pathlib import Path
wd = Path("/home/jhyeo/finetuning/finetuning_dataset/arxiv_formula")
man = {json.loads(l)["key"] for l in (wd / "manifest.jsonl").open(encoding="utf-8") if l.strip()}
for name in ("dots_out", "paddle_out"):
    got = set()
    for sp in ("train", "valid", "test"):
        p = wd / name / f"{sp}.jsonl"
        if p.is_file():
            got |= {json.loads(l)["key"] for l in p.open(encoding="utf-8") if l.strip()}
    miss = man - got
    print(f"  {name}: {len(got)}/{len(man)} (누락 {len(miss)})")
    # 몇 장 빠지는 건 합의 대상이 줄 뿐이라 치명적이지 않다. 대량 누락만 막는다.
    if len(miss) > len(man) * 0.01:
        print(f"    누락 예시: {sorted(miss)[:5]}")
        sys.exit(1)
PY

# ── 3. 합의 + 크롭 ────────────────────────────────────────────────────────────
step "3/8 수식 합의(IoU>=0.5) + 크롭 생성"
python3 build_formula_consensus.py || die "consensus 실패"
[ -s formula_pairs.jsonl ] || die "합의 결과가 비었다"

# ── 4. 서버 교체 (3 GPU 에 dots/paddle 과 122B 를 동시에 못 올린다) ──────────────
step "4/8 dots/paddle 서버 내리고 122B judge 올리기"
pkill -f "vllm.entrypoints.openai.api_server.*8119" || true
pkill -f "vllm.entrypoints.openai.api_server.*8118" || true
tmux kill-session -t afdots 2>/dev/null || true
tmux kill-session -t afpaddle 2>/dev/null || true
sleep 20
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

( cd "$OFR" && python3 -m ocr_filter.cli models serve --only judge ) || die "judge serve 실패"
echo "  122B 로딩 대기..."
for i in $(seq 1 120); do
  curl -s -m 5 http://127.0.0.1:8004/v1/models | grep -q Qwen && { echo "  준비됨 ($((i*15))s)"; break; }
  sleep 15
done
curl -s -m 5 http://127.0.0.1:8004/v1/models | grep -q Qwen || die "122B 가 안 떴다"

# ── 5. 중재 (파일럿 -> 전량) ──────────────────────────────────────────────────
step "5/8 judge 중재 — 파일럿 300 페이지"
python3 arbitrate_formulas.py --limit 300 --workers 8 || die "중재(파일럿) 실패"

step "5/8 judge 중재 — 전량 (재개 모드)"
python3 arbitrate_formulas.py --workers 8 || die "중재(전량) 실패"

# ── 6~8 ──────────────────────────────────────────────────────────────────────
step "6/8 데이터셋 빌드 (crop + layout)"
python3 build_formula_datasets.py || die "데이터셋 빌드 실패"

step "7/8 검증"
python3 verify_datasets.py
VERIFY=$?

step "8/8 완료  (검증 종료코드=$VERIFY)"
echo "산출물:"
echo "  $WD/dataset_formula_crop/{train,valid,test}.jsonl"
echo "  $WD/dataset_formula_layout/{train,valid,test}.jsonl"
echo "  judge 가 직접 쓴 수식 검색: grep '\"formula_source\": \"judge_authored\"' dataset_formula_crop/*.jsonl"
exit $VERIFY
