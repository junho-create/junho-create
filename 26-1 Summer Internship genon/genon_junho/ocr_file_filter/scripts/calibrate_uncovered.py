#!/usr/bin/env python3
"""`filter_e21grid.py --max_uncovered_new` 임계값을 배치마다 재보정한다.

PIPELINE.md 함정 4: 기본값 0.05 는 **구형 파이프라인** 캘리브레이션이다. 이 파이프라인은
Medium 라벨로 dots.ocr 출력을 그대로 채택하는데 dots 는 범용 OCR 만큼 촘촘히 segment 하지
않아서, 멀쩡한 라벨조차 uncovered_ocr 중앙값이 0.15~0.18 로 나온다. 0.05 로 돌리면
batch5400 에서 79% 가 날아갔다. 코퍼스마다 문서 종류가 달라 분포도 달라지므로
**배치마다 다시 잡아야 한다.**

방법은 PIPELINE.md 가 권한 그대로 "목표 통과 건수에 맞춰 임계값을 역산"이다.
`--dry_run` 으로 후보 임계값을 훑어 통과율을 재고, 목표(기본 33%)에 가장 가까운 값을 고른다.

사용:
    python3 scripts/calibrate_uncovered.py --work-dir /home/.../batch_multimodal
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

FILTER = "/home/jhyeo/finetuning/finetuning_dataset/filter_e21grid.py"
CANDIDATES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
LINE_RE = re.compile(r"\[train\]\s+(\d+)\s*->\s*(\d+)")


def run_filter(in_dir: Path, out_dir: Path, thr: float, dry: bool) -> tuple[int, int]:
    cmd = [sys.executable, FILTER, "--data_dir", str(in_dir), "--out_dir", str(out_dir),
           "--splits", "train", "--max_uncovered_new", str(thr)]
    if dry:
        cmd.append("--dry_run")
    p = subprocess.run(cmd, capture_output=True, text=True)
    m = LINE_RE.search(p.stdout)
    if not m:
        print(p.stdout[-1500:], file=sys.stderr)
        print(p.stderr[-800:], file=sys.stderr)
        raise SystemExit(f"filter_e21grid 출력에서 통과 건수를 못 읽었다 (thr={thr})")
    return int(m.group(1)), int(m.group(2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--target-rate", type=float, default=0.33,
                    help="목표 통과율. PIPELINE.md 가 '전체의 30%대'를 실전값으로 든다")
    ap.add_argument("--apply", action="store_true", help="고른 임계값으로 실제 필터까지 수행")
    args = ap.parse_args()

    wd = Path(args.work_dir)
    in_dir, out_dir = wd / "_e21grid_in", wd / "_e21grid_out"
    src = wd / "with_ocr.jsonl"
    if not src.is_file():
        raise SystemExit(f"with_ocr.jsonl 이 없다: {src}")
    # 존재 여부가 아니라 **최신성**으로 판단한다. with_ocr.jsonl 이 새로 갱신됐는데
    # (예: hardcase judge 재처리 후 재조립) _e21grid_in/train.jsonl 이 예전 캐시 그대로면
    # 스윕 전체가 옛 데이터로 도는데도 조용히 통과한다 — 실제로 multimodal 코퍼스에서
    # judge 3,192건을 복구한 뒤 재조립했는데 이 캐시가 안 갱신돼 323건짜리 예전 결과로
    # 재보정이 돌 뻔했다. mtime 비교로 막는다.
    stale = (not (in_dir / "train.jsonl").is_file()
             or (in_dir / "train.jsonl").stat().st_mtime < src.stat().st_mtime)
    if stale:
        in_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, in_dir / "train.jsonl")

    print(f"== uncovered 임계값 스윕 ({wd.name}) ==")
    results = []
    for thr in CANDIDATES:
        total, kept = run_filter(in_dir, out_dir, thr, dry=True)
        rate = kept / max(total, 1)
        results.append((thr, total, kept, rate))
        print(f"  thr={thr:<5} {total} -> {kept}  ({rate * 100:.1f}%)")

    best = min(results, key=lambda r: abs(r[3] - args.target_rate))
    thr, total, kept, rate = best
    print(f"\n선택: max_uncovered_new={thr}  ({kept}/{total} = {rate * 100:.1f}%, "
          f"목표 {args.target_rate * 100:.0f}%)")

    if args.apply:
        run_filter(in_dir, out_dir, thr, dry=False)
        shutil.copy(out_dir / "train.jsonl", wd / "final_gt.jsonl")
        print(f"적용 완료 -> {wd / 'final_gt.jsonl'}")
    print(f"CHOSEN_THRESHOLD={thr}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
