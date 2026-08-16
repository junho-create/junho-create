#!/usr/bin/env python3
"""audit.jsonl 을 사람이 볼 수 있는 형태로 정리한다.

    flagged.jsonl  — FLAG / JUDGE_ERROR / RENDER_ERROR 만, 심각한 것부터
    summary.json   — split 별 집계, 지표 평균, error_type 빈도
    review.html    — 오버레이 이미지 + GT 렌더 이미지를 나란히 보여주는 검수 페이지

review.html 은 기본적으로 이미지를 **상대 경로로 참조**한다 (base64 로 박으면
5,030건에서 파일이 수 GB 가 된다). 다른 머신으로 옮겨 볼 거면 `--embed` 를 쓰되
`--max` 로 건수를 제한하라.

사용:
    python3 report.py
    python3 report.py --max 300 --status FLAG
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"

METRIC_KEYS = (
    "table_coverage",
    "basic_structure",
    "complex_structure",
    "cell_correspondence",
    "text_accuracy",
    "auxiliary_visual_fidelity",
)
METRIC_LABELS = {
    "table_coverage": "범위",
    "basic_structure": "기본구조",
    "complex_structure": "병합구조",
    "cell_correspondence": "셀대응",
    "text_accuracy": "텍스트",
    "auxiliary_visual_fidelity": "시각",
}
FLAGGED_STATUSES = ("FLAG", "JUDGE_ERROR", "RENDER_ERROR")
# 상태별 정렬 우선순위 — 점수가 낮은 것과 판정 실패를 위로 올린다.
STATUS_ORDER = {"RENDER_ERROR": 0, "JUDGE_ERROR": 1, "FLAG": 2, "PASS": 3}


def load_audit(path: Path) -> list[dict]:
    """같은 key 가 여러 번 나오면(재실행분) 나중 줄이 이긴다."""
    by_key: dict[str, dict] = {}
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        by_key[r["key"]] = r
    return list(by_key.values())


def sort_key(r: dict) -> tuple:
    return (STATUS_ORDER.get(r.get("status"), 9), r.get("min_score", 0), r["key"])


def build_summary(rows: list[dict]) -> dict:
    by_split: dict[str, Counter] = defaultdict(Counter)
    status_total = Counter()
    error_types = Counter()
    static_flags = Counter()
    judge_errors = Counter()
    metric_sum = Counter()
    metric_n = Counter()
    score_hist = Counter()
    n_tables_judged = 0

    for r in rows:
        st = r.get("status", "?")
        status_total[st] += 1
        by_split[r.get("split", "?")][st] += 1
        for f in r.get("static_flags") or []:
            static_flags[f.split("@")[0]] += 1
        if st == "JUDGE_ERROR":
            judge_errors[str(r.get("error", ""))[:40]] += 1
        v = r.get("judge")
        if not v:
            continue
        for t in v["tables"]:
            n_tables_judged += 1
            score_hist[t["min_score"]] += 1
            for k in METRIC_KEYS:
                m = t["metrics"][k]
                metric_sum[k] += m["score"]
                metric_n[k] += 1
                for e in m["error_types"]:
                    error_types[e] += 1

    return {
        "total_pages": len(rows),
        "status": dict(status_total),
        "by_split": {k: dict(v) for k, v in sorted(by_split.items())},
        "tables_judged": n_tables_judged,
        "metric_avg": {k: round(metric_sum[k] / metric_n[k], 3)
                       for k in METRIC_KEYS if metric_n[k]},
        "table_min_score_hist": dict(sorted(score_hist.items())),
        "error_types": dict(error_types.most_common()),
        "static_flags": dict(static_flags.most_common()),
        "judge_error_kinds": dict(judge_errors.most_common(10)),
    }


def _img_src(path: str | None, embed: bool, out_dir: Path) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.exists():
        return ""
    if not embed:
        # out_dir 밖(work/)에 있으므로 relative_to 로는 안 되고 os.path.relpath 가 필요하다.
        return html.escape(os.path.relpath(p, out_dir))
    mime = "image/jpeg" if p.suffix.lower() in (".jpg", ".jpeg") else "image/png"
    return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode("ascii")


def _table_block(t: dict) -> str:
    cells = []
    for k in METRIC_KEYS:
        m = t["metrics"][k]
        cls = "s5" if m["score"] == 5 else ("s4" if m["score"] == 4 else "sbad")
        title = html.escape(m["reason"] or "")
        cells.append(f'<span class="m {cls}" title="{title}">'
                     f'{METRIC_LABELS[k]} {m["score"]}</span>')
    reasons = [f"<li><b>{METRIC_LABELS[k]}</b>: {html.escape(t['metrics'][k]['reason'])}"
               f" <i>{'/'.join(t['metrics'][k]['error_types'])}</i></li>"
               for k in METRIC_KEYS if t["metrics"][k]["reason"]]
    reason_html = f"<ul class='reasons'>{''.join(reasons)}</ul>" if reasons else ""
    summ = html.escape(t.get("summary") or "")
    return (f'<div class="tbl"><div class="tblhead">#{t["index"]} '
            f'<small>min={t["min_score"]}</small></div>'
            f'<div class="metrics">{"".join(cells)}</div>'
            f'{f"<div class=summary>{summ}</div>" if summ else ""}{reason_html}</div>')


def build_review_html(rows: list[dict], embed: bool, out_dir: Path, title: str) -> str:
    parts = [f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<title>{html.escape(title)}</title><style>
 body{{font-family:system-ui,-apple-system,sans-serif;margin:0;background:#f6f7f9;color:#111}}
 header{{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;padding:10px 16px;z-index:5}}
 h1{{font-size:16px;margin:0}}
 .item{{background:#fff;margin:14px;border:1px solid #ddd;border-radius:6px;overflow:hidden}}
 .bar{{display:flex;gap:12px;align-items:center;padding:8px 12px;border-bottom:1px solid #eee;
      background:#fafafa;flex-wrap:wrap}}
 .key{{font-family:ui-monospace,monospace;font-weight:600}}
 .badge{{padding:2px 8px;border-radius:10px;font-size:12px;font-weight:700;color:#fff}}
 .FLAG{{background:#e67e22}} .JUDGE_ERROR{{background:#8e44ad}}
 .RENDER_ERROR{{background:#c0392b}} .PASS{{background:#27ae60}}
 .path{{color:#888;font-size:12px;word-break:break-all}}
 .cols{{display:grid;grid-template-columns:1fr 1fr;gap:10px;padding:10px}}
 .col h3{{font-size:13px;margin:0 0 6px;color:#555}}
 .col img{{max-width:100%;border:1px solid #ccc;background:#fff}}
 .verdict{{padding:0 12px 12px}}
 .tbl{{border-top:1px solid #eee;padding:8px 0}}
 .tblhead{{font-weight:700;color:#b00}}
 .metrics{{display:flex;gap:6px;flex-wrap:wrap;margin:6px 0}}
 .m{{font-size:12px;padding:2px 7px;border-radius:4px;border:1px solid #ccc}}
 .s5{{background:#eafaf1;border-color:#a9dfbf}} .s4{{background:#fef9e7;border-color:#f7dc6f}}
 .sbad{{background:#fdedec;border-color:#f5b7b1;font-weight:700}}
 .summary{{font-size:13px;margin:4px 0}}
 .reasons{{font-size:12px;color:#444;margin:4px 0 0 16px}}
 .err{{color:#c0392b;font-family:ui-monospace,monospace;font-size:13px;padding:10px 12px}}
 .flags{{font-size:12px;color:#c0392b}}
</style></head><body>
<header><h1>{html.escape(title)} — {len(rows)}건</h1></header>"""]

    for r in rows:
        st = r.get("status", "?")
        ov = _img_src(r.get("overlay_path"), embed, out_dir)
        rd = _img_src(r.get("render_path"), embed, out_dir)
        flags = ", ".join(r.get("static_flags") or [])
        parts.append(f'<div class="item"><div class="bar">'
                     f'<span class="badge {st}">{st}</span>'
                     f'<span class="key">{html.escape(r["key"])}</span>'
                     f'<span>표 {r.get("n_tables", "?")}개</span>'
                     + (f'<span>min={r["min_score"]}</span>' if "min_score" in r else "")
                     + (f'<span class="flags">정적: {html.escape(flags)}</span>' if flags else "")
                     + f'<span class="path">{html.escape(r.get("image_path", ""))}</span></div>')
        if ov or rd:
            ov_tag = f'<img src="{ov}" loading="lazy">' if ov else "—"
            rd_tag = f'<img src="{rd}" loading="lazy">' if rd else "—"
            parts.append('<div class="cols">'
                         f'<div class="col"><h3>원본 페이지 (표 영역 #N)</h3>{ov_tag}</div>'
                         f'<div class="col"><h3>GT HTML 렌더</h3>{rd_tag}</div></div>')
        if r.get("error"):
            parts.append(f'<div class="err">{html.escape(str(r["error"]))}</div>')
        v = r.get("judge")
        if v:
            body = "".join(_table_block(t) for t in v["tables"])
            if v["unlabeled_tables"]:
                body += ('<div class="tbl"><div class="tblhead">미라벨 표</div><ul class="reasons">'
                         + "".join(f"<li>{html.escape(str(u.get('where', '')))}</li>"
                                   for u in v["unlabeled_tables"]) + "</ul></div>")
            if v["overall"].get("summary"):
                body += f'<div class="summary"><b>총평</b> {html.escape(v["overall"]["summary"])}</div>'
            parts.append(f'<div class="verdict">{body}</div>')
        parts.append("</div>")

    parts.append("</body></html>")
    return "\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", default=str(RESULTS_DIR / "audit.jsonl"))
    ap.add_argument("--out-dir", default=str(RESULTS_DIR))
    ap.add_argument("--max", type=int, default=400, help="review.html 에 넣을 최대 건수")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--status", nargs="*", default=list(FLAGGED_STATUSES),
                    help="review.html 에 넣을 status (기본: 검수 대상 전부)")
    ap.add_argument("--embed", action="store_true",
                    help="이미지를 base64 로 HTML 에 박는다 (다른 머신에서 열 때)")
    ap.add_argument("--html-name", default="review.html")
    args = ap.parse_args()

    audit_path = Path(args.audit)
    if not audit_path.exists():
        print(f"{audit_path} 없음 — run_audit.py 를 먼저 돌려라")
        return 1

    rows = load_audit(audit_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = build_summary(rows)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    flagged = sorted((r for r in rows if r.get("status") in FLAGGED_STATUSES), key=sort_key)
    with (out_dir / "flagged.jsonl").open("w", encoding="utf-8") as f:
        for r in flagged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (out_dir / "flagged_keys.txt").open("w", encoding="utf-8") as f:
        for r in flagged:
            f.write(r["key"] + "\n")

    want = set(args.status)
    shown = sorted((r for r in rows if r.get("status") in want), key=sort_key)
    total_shown = len(shown)
    shown = shown[args.offset: args.offset + args.max]
    title = f"GT 표 검수 [{'/'.join(sorted(want))}] {args.offset + 1}–{args.offset + len(shown)} / {total_shown}"
    (out_dir / args.html_name).write_text(
        build_review_html(shown, args.embed, out_dir, title), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n검수 대상 {len(flagged)}/{len(rows)}건 → {out_dir / 'flagged.jsonl'}")
    print(f"review.html: {len(shown)}건 표시 (전체 {total_shown}건) → {out_dir / args.html_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
