#!/usr/bin/env python3
"""판별력 테스트 — 멀쩡한 GT 를 인위로 훼손했을 때 judge 가 잡아내는지 확인한다.

`ocr_filter/hardcase/prompts.py:68-77` 에 지표를 잘게 쪼갠 judge 가 판별력을 잃어
"박스를 20% 밀고 절반을 지운" 합성 결함을 5점/PASS 로 통과시킨 사례가 기록돼 있다.
같은 함정을 밟지 않으려고 전량 실행 전에 반드시 돌린다.

훼손 종류:
    drop_row    — 본문 행 1개 삭제
    swap_number — 셀 안의 숫자 자릿수 바꾸기
    break_span  — rowspan/colspan 값 변경 (없으면 헤더 셀 1개 삭제)

사용:
    PYTHONPATH=/home/jhyeo/ocr_file_filter/labeler python3 sanity_corrupt.py --n 10
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from bs4 import BeautifulSoup

import render as render_mod
from judge import JudgeError, extract_json, make_judge, validate_verdict
from run_audit import budget_for, build_prompt, close_renderer, get_renderer

BASE_DIR = Path(__file__).parent
WORK_DIR = BASE_DIR / "work" / "sanity"

_print_lock = threading.Lock()


def say(msg: str) -> None:
    """즉시 flush — 중간에 죽어도 어디까지 갔는지는 남게."""
    with _print_lock:
        print(msg, flush=True)


def drop_row(html: str) -> str | None:
    soup = BeautifulSoup(html, "html.parser")
    rows = soup.find_all("tr")
    if len(rows) < 4:
        return None
    rows[len(rows) // 2].decompose()
    return str(soup)


def swap_number(html: str) -> str | None:
    soup = BeautifulSoup(html, "html.parser")
    for cell in soup.find_all(["td", "th"]):
        text = cell.get_text(strip=True)
        m = re.fullmatch(r"[\d,\.]{3,}", text)
        if m and any(c.isdigit() for c in text):
            digits = [c for c in text if c.isdigit()]
            if len(set(digits)) < 2:
                continue
            # 첫 두 숫자를 맞바꿔 "그럴듯하지만 틀린" 값을 만든다
            i = text.index(digits[0])
            j = text.index(digits[1], i + 1)
            new = list(text)
            new[i], new[j] = new[j], new[i]
            cell.string = "".join(new)
            return str(soup)
    return None


def break_span(html: str) -> str | None:
    soup = BeautifulSoup(html, "html.parser")
    for cell in soup.find_all(["td", "th"]):
        for attr in ("rowspan", "colspan"):
            if cell.has_attr(attr):
                try:
                    v = int(cell[attr])
                except ValueError:
                    continue
                cell[attr] = str(max(1, v - 1)) if v > 2 else str(v + 1)
                return str(soup)
    cells = soup.find_all(["td", "th"])
    if len(cells) < 6:
        return None
    cells[1].decompose()
    return str(soup)


CORRUPTIONS = {"drop_row": drop_row, "swap_number": swap_number, "break_span": break_span}


def judge_row(row: dict, judge) -> dict:
    overlay = render_mod.make_overlay(row, WORK_DIR / "overlays" / f"{row['key']}.jpg")
    rendered = render_mod.render_tables(row, get_renderer(), WORK_DIR / "renders")
    raw = judge.call_with_images(
        system_prompt="",
        user_text=build_prompt(row["n_tables"]),
        images=[str(overlay), str(rendered)],
        max_tokens=budget_for(row["n_tables"], 8192),
    )
    return validate_verdict(extract_json(raw), row["n_tables"])


def run_one(r: dict, judge, flag_threshold: int) -> dict:
    """페이지 1장에 대해 원본 + 훼손 3종을 판정한다 (페이지 내부는 직렬)."""
    entry = {"key": r["key"], "image_path": r["image_path"]}
    try:
        base = judge_row(r, judge)
        entry["clean_min_score"] = base["overall"]["min_table_score"]
        entry["clean_summary"] = base["overall"]["summary"][:120]
    except (JudgeError, Exception) as e:
        entry["clean_error"] = f"{type(e).__name__}: {e}"
        say(f"{r['key']}: clean judge 실패 {e}")
        return entry

    entry["variants"] = {}
    for name, fn in CORRUPTIONS.items():
        bad_html = fn(r["tables"][0]["html"])
        if bad_html is None:
            entry["variants"][name] = {"skipped": "적용 불가"}
            continue
        bad_row = copy.deepcopy(r)
        bad_row["key"] = f"{r['key']}__{name}"
        bad_row["tables"][0]["html"] = bad_html
        try:
            v = judge_row(bad_row, judge)
        except Exception as e:
            entry["variants"][name] = {"error": f"{type(e).__name__}: {e}"}
            continue
        ms = v["overall"]["min_table_score"]
        entry["variants"][name] = {
            "min_score": ms,
            "caught": ms <= flag_threshold,
            "summary": v["tables"][0]["summary"][:120],
            "error_types": sorted({
                e for m in v["tables"][0]["metrics"].values() for e in m["error_types"]
            }),
        }

    line = f"{r['key']}  clean={entry['clean_min_score']}"
    for name, res in entry["variants"].items():
        mark = "O" if res.get("caught") else ("-" if "skipped" in res else "X")
        val = res.get("min_score", res.get("skipped", res.get("error")))
        line += f"  {name}={val}[{mark}]"
    say(line)
    return entry


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(BASE_DIR / "manifest.jsonl"))
    ap.add_argument("--n", type=int, default=10, help="테스트할 페이지 수")
    ap.add_argument("--seed", type=int, default=318)
    ap.add_argument("--flag-threshold", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4, help="페이지 단위 병렬")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--reasoning-effort", default="low",
                    help="low | medium | high | default | off")
    ap.add_argument("--out", default=str(BASE_DIR / "results" / "sanity.json"))
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.manifest, encoding="utf-8")]
    # 표 1개 + 정적 플래그 없음 + 행이 충분한 페이지만 (훼손이 의미 있으려면)
    cands = [
        r for r in rows
        if r["n_tables"] == 1 and not r["static_flags"]
        and r["tables"][0]["html"].count("<tr") >= 5
    ]
    random.Random(args.seed).shuffle(cands)
    picked = cands[: args.n]
    print(f"후보 {len(cands)}건 중 {len(picked)}건 선택\n")

    judge = make_judge(timeout=args.timeout, reasoning_effort=args.reasoning_effort)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = []

    def flush_report():
        """중간에 죽어도 여기까지의 결과는 남게 매번 덮어쓴다."""
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                            encoding="utf-8")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_one, r, judge, args.flag_threshold): r for r in picked}
        for fut in as_completed(futs):
            r = futs[fut]
            try:
                report.append(fut.result())
            except Exception as e:
                say(f"{r['key']}: 처리 실패 {type(e).__name__}: {e}")
                report.append({"key": r["key"], "clean_error": f"{type(e).__name__}: {e}"})
            flush_report()
        for _ in range(args.workers):
            pool.submit(close_renderer)

    # 집계
    total = caught = 0
    clean_pass = 0
    for e in report:
        if e.get("clean_min_score", 0) > args.flag_threshold:
            clean_pass += 1
        for res in (e.get("variants") or {}).values():
            if "min_score" not in res:
                continue
            total += 1
            caught += bool(res["caught"])
    n_clean = sum(1 for e in report if "clean_min_score" in e)
    print(f"\n원본 PASS(오탐 아님): {clean_pass}/{n_clean}")
    print(f"훼손 검출률: {caught}/{total}" + (f" ({caught/total*100:.0f}%)" if total else ""))
    u = judge.usage
    if u["calls"]:
        print(f"토큰: 호출 {u['calls']}회 / 입력 평균 {u['prompt_tokens'] // u['calls']} "
              f"/ 출력 평균 {u['completion_tokens'] // u['calls']} "
              f"(최대 {u['max_completion']}) / max_tokens 소진 {u['truncated']}회")
    print(f"→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
