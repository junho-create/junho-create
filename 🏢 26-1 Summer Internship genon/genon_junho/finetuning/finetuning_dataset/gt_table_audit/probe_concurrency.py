#!/usr/bin/env python3
"""776 서빙이 동시 요청을 몇 개까지 소화하는지 재본다.

같은 페이지 묶음을 동시성만 바꿔가며 돌려 호출당 지연과 처리량을 비교한다.
이미 렌더된 이미지를 재사용하므로 judge 호출 비용만 든다.

사용:
    PYTHONPATH=/home/jhyeo/ocr_file_filter/labeler python3 probe_concurrency.py
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import render as render_mod
from judge import JudgeError, extract_json, make_judge, validate_verdict
from run_audit import budget_for, build_prompt, get_renderer

BASE_DIR = Path(__file__).parent
WORK_DIR = BASE_DIR / "work" / "probe"


def one_call(row: dict, judge, timeout_note: list) -> dict:
    t0 = time.time()
    try:
        overlay = render_mod.make_overlay(row, WORK_DIR / "overlays" / f"{row['key']}.jpg")
        rendered = render_mod.render_tables(row, get_renderer(), WORK_DIR / "renders")
        raw = judge.call_with_images(
            system_prompt="",
            user_text=build_prompt(row["n_tables"]),
            images=[str(overlay), str(rendered)],
            max_tokens=budget_for(row["n_tables"], 16384),
        )
        validate_verdict(extract_json(raw), row["n_tables"])
        return {"key": row["key"], "ok": True, "sec": round(time.time() - t0, 1)}
    except JudgeError as e:
        return {"key": row["key"], "ok": False, "sec": round(time.time() - t0, 1),
                "err": str(e)[:60]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(BASE_DIR / "manifest.jsonl"))
    ap.add_argument("--levels", type=int, nargs="*", default=[1, 2, 4, 8])
    ap.add_argument("--per-level", type=int, default=4, help="동시성 수준마다 호출 수")
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.manifest, encoding="utf-8")]
    pool_rows = [r for r in rows if r["n_tables"] == 1 and not r["static_flags"]]

    judge = make_judge(timeout=args.timeout)
    cursor = 0
    print(f"{'동시성':>6} {'성공':>5} {'호출당 평균':>11} {'최대':>7} {'전체':>7} {'처리량':>10}")
    for level in args.levels:
        batch = pool_rows[cursor: cursor + args.per_level]
        cursor += args.per_level
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=level) as pool:
            res = list(pool.map(lambda r: one_call(r, judge, []), batch))
        wall = time.time() - t0
        ok = [r for r in res if r["ok"]]
        avg = sum(r["sec"] for r in ok) / len(ok) if ok else 0
        mx = max((r["sec"] for r in ok), default=0)
        print(f"{level:>6} {len(ok)}/{len(res):>3} {avg:>10.1f}s {mx:>6.1f}s "
              f"{wall:>6.1f}s {len(ok)/wall*60:>8.1f} 건/분")
        for r in res:
            if not r["ok"]:
                print(f"         실패 {r['key']} {r['sec']}s {r.get('err')}")

    u = judge.usage
    if u["calls"]:
        print(f"\n토큰: 호출 {u['calls']}회 / 입력 평균 {u['prompt_tokens'] // u['calls']} "
              f"/ 출력 평균 {u['completion_tokens'] // u['calls']} (최대 {u['max_completion']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
