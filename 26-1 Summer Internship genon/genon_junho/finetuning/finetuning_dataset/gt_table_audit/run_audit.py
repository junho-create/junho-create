#!/usr/bin/env python3
"""combined_e24_refined layout 페이지의 Table GT 를 LLM-judge 로 감사한다.

페이지 1장 = judge 호출 1회. 원본 페이지(표 영역에 #N 오버레이) 와 GT 표 HTML 렌더를
나란히 넣고, 표별로 6개 지표를 매기게 한다.

**재시도 없음** — 호출/파싱이 한 번에 안 되면 그 페이지는 JUDGE_ERROR 로 기록하고
바로 다음으로 넘어간다. JUDGE_ERROR 도 사람 검수 큐에 포함된다.

사용:
    export PYTHONPATH=/home/jhyeo/ocr_file_filter/labeler
    python3 run_audit.py --preflight
    python3 run_audit.py --limit 100 --workers 8
    python3 run_audit.py --workers 8                 # 전량 (audit.jsonl 이어서)
    python3 run_audit.py --retry-errors --workers 8  # JUDGE_ERROR 건만 재실행
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import render as render_mod
from judge import JudgeError, extract_json, make_judge, validate_verdict

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
WORK_DIR = BASE_DIR / "work"
AUDIT_PATH = RESULTS_DIR / "audit.jsonl"
PROMPT_PATH = BASE_DIR / "judge_prompt.md"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("audit")
logging.getLogger("httpx").setLevel(logging.WARNING)

_thread_local = threading.local()
_write_lock = threading.Lock()


def get_renderer():
    """worker thread 마다 Renderer 1개. labeler/core/pipeline.py:330 패턴."""
    r = getattr(_thread_local, "renderer", None)
    if r is None:
        r = render_mod.new_renderer()
        _thread_local.renderer = r
    return r


def close_renderer(_ignored=None):
    """생성한 스레드에서 teardown 되도록 worker 당 1회 submit 한다."""
    r = getattr(_thread_local, "renderer", None)
    if r is not None:
        try:
            r.stop()
        except Exception as e:
            logger.warning("renderer stop 실패: %s", e)
        _thread_local.renderer = None


def build_prompt(n_tables: int) -> str:
    return PROMPT_PATH.read_text(encoding="utf-8").replace("{{N_TABLES}}", str(n_tables))


def budget_for(n_tables: int, cap: int) -> int:
    """표 개수에 맞춰 max_tokens 를 잡는다.

    thinking 이 켜져 있어서(끄면 판별력이 무너진다 — judge.py 주석 참고) reasoning
    토큰이 이 예산을 함께 쓴다. 빠듯하게 잡으면 content 가 빈 채로 돌아온다.

    실측(표 1개, 이미지 2장): 출력 평균 5,228 / 최대 9,511 토큰. 7,500 을 줬더니
    26회 중 2회가 finish_reason=length 로 잘렸다. 표 1개에 12,000 을 준다.
    """
    return min(cap, 10000 + 2000 * n_tables)


def decide_status(row: dict, verdict: dict | None, flag_threshold: int) -> str:
    if row["static_flags"]:
        return "FLAG"
    if verdict is None:
        return "FLAG"
    if verdict["unlabeled_tables"]:
        return "FLAG"
    if verdict["overall"]["min_table_score"] <= flag_threshold:
        return "FLAG"
    return "PASS"


def process_page(row: dict, judge, flag_threshold: int, keep_raw: bool,
                 skip_static: bool = True, token_cap: int = 8192) -> dict:
    key = row["key"]
    out = {
        "key": key,
        "split": row["split"],
        "image_path": row["image_path"],
        "n_tables": row["n_tables"],
        "static_flags": row["static_flags"],
        "bboxes": [t["bbox"] for t in row["tables"]],
    }
    t0 = time.time()

    # 1) 렌더 (실패해도 사람이 봐야 하므로 상태만 남기고 계속 진행하지 않는다)
    try:
        overlay = render_mod.make_overlay(row, WORK_DIR / "overlays" / f"{key}.jpg")
        rendered = render_mod.render_tables(row, get_renderer(), WORK_DIR / "renders")
    except Exception as e:
        out.update(status="RENDER_ERROR", error=f"{type(e).__name__}: {e}",
                   elapsed=round(time.time() - t0, 1))
        return out

    out["overlay_path"] = str(overlay)
    out["render_path"] = str(rendered)

    # 2) 정적 플래그가 있으면 judge 결과와 무관하게 어차피 FLAG 다.
    #    호출을 아예 건너뛴다 — 판정이 달라지지 않는데 토큰만 쓴다.
    if skip_static and row["static_flags"]:
        out.update(status="FLAG", skipped_judge="static_flags",
                   elapsed=round(time.time() - t0, 1))
        return out

    # 3) judge — 호출 1회, 실패 시 즉시 JUDGE_ERROR
    try:
        raw = judge.call_with_images(
            system_prompt="",
            user_text=build_prompt(row["n_tables"]),
            images=[str(overlay), str(rendered)],
            max_tokens=budget_for(row["n_tables"], token_cap),
        )
        verdict = validate_verdict(extract_json(raw), row["n_tables"])
    except JudgeError as e:
        out.update(status="JUDGE_ERROR", error=str(e),
                   elapsed=round(time.time() - t0, 1))
        if keep_raw:
            raw_dir = WORK_DIR / "raw_errors"
            raw_dir.mkdir(parents=True, exist_ok=True)
            (raw_dir / f"{key}.txt").write_text(locals().get("raw", "") or "",
                                                encoding="utf-8")
        return out
    except Exception as e:
        out.update(status="JUDGE_ERROR", error=f"{type(e).__name__}: {e}",
                   elapsed=round(time.time() - t0, 1))
        return out

    out["judge"] = verdict
    out["min_score"] = verdict["overall"]["min_table_score"]
    out["status"] = decide_status(row, verdict, flag_threshold)
    out["elapsed"] = round(time.time() - t0, 1)
    return out


def load_done(path: Path) -> dict[str, str]:
    """이미 처리한 key → status. 중간에 끊겨도 이어서 돌리려고 쓴다."""
    done: dict[str, str] = {}
    if not path.exists():
        return done
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue  # 쓰다 만 마지막 줄
        done[r["key"]] = r.get("status", "")
    return done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(BASE_DIR / "manifest.jsonl"))
    ap.add_argument("--out", default=str(AUDIT_PATH))
    ap.add_argument("--workers", type=int, default=4,
                    help="776 서빙 처리량이 4에서 정점(1.7건/분). 8은 오히려 느리다")
    ap.add_argument("--limit", type=int, help="처리할 최대 페이지 수")
    ap.add_argument("--split", choices=("train", "valid", "test"))
    ap.add_argument("--keys", nargs="*", help="특정 key 만")
    ap.add_argument("--keys-file", help="key 목록 파일 (한 줄에 하나)")
    ap.add_argument("--sample", type=int, help="무작위 N건 (seed 고정)")
    ap.add_argument("--seed", type=int, default=318)
    ap.add_argument("--flag-threshold", type=int, default=4,
                    help="표의 어떤 지표든 이 값 이하면 FLAG (기본 4 = 완벽하지 않으면 FLAG)")
    ap.add_argument("--reasoning-effort", default="default",
                    help="thinking 길이. low 가 기본 — off 는 판별력이 무너지니 쓰지 말 것")
    ap.add_argument("--timeout", type=int, default=600,
                    help="이 시간을 넘기면 끊고 JUDGE_ERROR (사람 검수 큐로)")
    ap.add_argument("--max-tokens", type=int, default=24000,
                    help="max_tokens 상한. 실제로는 표 개수에 따라 더 작게 잡는다")
    ap.add_argument("--judge-static-flagged", action="store_true",
                    help="정적 플래그가 붙은 페이지도 judge 를 태운다 (기본은 건너뜀 — "
                         "어차피 FLAG 라 판정이 안 바뀌는데 토큰만 쓴다)")
    ap.add_argument("--retry-errors", action="store_true",
                    help="기존 결과의 JUDGE_ERROR/RENDER_ERROR 건만 다시 실행")
    ap.add_argument("--preflight", action="store_true",
                    help="이미지 2장 judge 호출 1건만 해보고 끝낸다")
    ap.add_argument("--keep-raw", action="store_true",
                    help="파싱 실패한 원문을 work/raw_errors/ 에 남긴다")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(args.manifest, encoding="utf-8")]

    if args.split:
        rows = [r for r in rows if r["split"] == args.split]
    if args.keys_file:
        want = {l.strip() for l in open(args.keys_file, encoding="utf-8") if l.strip()}
        rows = [r for r in rows if r["key"] in want]
    if args.keys:
        want = set(args.keys)
        rows = [r for r in rows if r["key"] in want]

    out_path = Path(args.out)
    done = load_done(out_path)

    if args.retry_errors:
        bad = {k for k, s in done.items() if s in ("JUDGE_ERROR", "RENDER_ERROR")}
        rows = [r for r in rows if r["key"] in bad]
        # 재실행분은 새 파일에 쓰고, 나중에 합칠 때 나중 줄이 이기게 한다.
        logger.info("재실행 대상 %d건", len(rows))
    else:
        rows = [r for r in rows if r["key"] not in done]

    if args.sample:
        random.Random(args.seed).shuffle(rows)
        rows = rows[: args.sample]
    if args.limit:
        rows = rows[: args.limit]

    if args.preflight:
        rows = rows[:1]
        if not rows:
            logger.error("preflight 할 페이지가 없다")
            return 1

    if not rows:
        logger.info("처리할 페이지 없음 (이미 %d건 완료)", len(done))
        return 0

    n_static = sum(1 for r in rows if r["static_flags"])
    n_calls = len(rows) - (0 if args.judge_static_flagged else n_static)
    logger.info("대상 %d 페이지 / 이미 완료 %d건 / judge 호출 예정 %d회 "
                "(정적 플래그 %d건 건너뜀) / workers=%d / flag_threshold=%d",
                len(rows), len(done), n_calls, len(rows) - n_calls,
                args.workers, args.flag_threshold)

    judge = make_judge(timeout=args.timeout, max_tokens=args.max_tokens,
                       reasoning_effort=args.reasoning_effort)
    counts: dict[str, int] = {}
    t_start = time.time()

    fout = out_path.open("a", encoding="utf-8")
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {
                pool.submit(process_page, r, judge, args.flag_threshold, args.keep_raw,
                            not args.judge_static_flagged, args.max_tokens): r
                for r in rows
            }
            for i, fut in enumerate(as_completed(futs), start=1):
                r = futs[fut]
                try:
                    res = fut.result()
                except Exception as e:  # process_page 밖에서 터진 경우
                    logger.exception("페이지 처리 실패: %s", r["key"])
                    res = {"key": r["key"], "split": r["split"], "status": "JUDGE_ERROR",
                           "error": f"{type(e).__name__}: {e}", "n_tables": r["n_tables"]}
                counts[res["status"]] = counts.get(res["status"], 0) + 1
                with _write_lock:
                    fout.write(json.dumps(res, ensure_ascii=False) + "\n")
                    fout.flush()
                if i % 20 == 0 or i == len(rows):
                    rate = i / max(1e-6, time.time() - t_start)
                    eta = (len(rows) - i) / rate / 60
                    logger.info("%d/%d  %s  %.2f p/s  ETA %.1f분",
                                i, len(rows), counts, rate, eta)
            # worker 마다 Renderer teardown (생성한 스레드에서 돌아야 한다)
            for _ in range(args.workers):
                pool.submit(close_renderer)
    finally:
        fout.close()

    logger.info("완료: %s  (%.1f분)", counts, (time.time() - t_start) / 60)
    u = judge.usage
    if u["calls"]:
        logger.info("토큰: 호출 %d회 / 입력 %d (평균 %d) / 출력 %d (평균 %d, 최대 %d) "
                    "/ max_tokens 소진 %d회",
                    u["calls"], u["prompt_tokens"], u["prompt_tokens"] // u["calls"],
                    u["completion_tokens"], u["completion_tokens"] // u["calls"],
                    u["max_completion"], u["truncated"])
        (RESULTS_DIR / "usage.json").write_text(
            json.dumps(u, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.preflight:
        last = json.loads(out_path.read_text(encoding="utf-8").strip().split("\n")[-1])
        print(json.dumps(last, ensure_ascii=False, indent=2)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
