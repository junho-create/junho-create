"""[6] hardcase judge 결과 리뷰용 HTML 갤러리 — 좌: 원본, 우: 라벨(bbox 오버레이), 상태(PASS/FAIL) 배지.

    python -m ocr_filter.hardcase.review_gallery \
        --judge-out ... --judge-out ... --out review.html [--limit N]

여러 --judge-out 을 줄 수 있다(레플리카별로 나눠 돌린 결과 파일들을 한 갤러리로 합쳐서 봄).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ocr_filter.report.render import draw_boxes, to_data_uri

_CARD_TEMPLATE = """
<article class="card {status_class}">
  <header class="card-head">
    <span class="chip {status_class}">{status_text}</span>
    <span class="cid">{id_short}</span>
    <span class="score">score <b>{score}</b></span>
    {revised_badge}
  </header>
  <div class="images">
    <figure><figcaption>원본</figcaption><img src="{orig_uri}" loading="lazy" alt="원본 이미지"></figure>
    <figure><figcaption>라벨 &middot; {n_elements}개 요소</figcaption><img src="{label_uri}" loading="lazy" alt="라벨 오버레이"></figure>
  </div>
  {issues_html}
</article>
"""

_PAGE_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8">
<title>hardcase judge 리뷰 — {n_total}건</title>
<style>
:root {{
  --bg: #12151a;
  --card: #1b1f26;
  --border: #272d36;
  --fg: #e6e9ed;
  --fg-muted: #8b93a1;
  --accent: #4fa3ff;
  --pass: #3ddc84;
  --pass-ink: #042415;
  --fail: #ff6b6b;
  --fail-ink: #300707;
  --revised: #f5a623;
  --revised-ink: #2e1c00;
  --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  --mono: ui-monospace, "SF Mono", "Cascadia Code", "Roboto Mono", Consolas, monospace;
}}
@media (prefers-color-scheme: light) {{
  :root {{
    --bg: #f5f6f8; --card: #ffffff; --border: #dde1e6; --fg: #1a1d23; --fg-muted: #5b6470;
    --accent: #2f6fe0; --pass: #1e9e5a; --pass-ink: #eafff3; --fail: #d64545; --fail-ink: #fff0f0;
    --revised: #b5790a; --revised-ink: #fff6e6;
  }}
}}
:root[data-theme="dark"] {{
  --bg: #12151a; --card: #1b1f26; --border: #272d36; --fg: #e6e9ed; --fg-muted: #8b93a1;
  --accent: #4fa3ff; --pass: #3ddc84; --pass-ink: #042415; --fail: #ff6b6b; --fail-ink: #300707;
  --revised: #f5a623; --revised-ink: #2e1c00;
}}
:root[data-theme="light"] {{
  --bg: #f5f6f8; --card: #ffffff; --border: #dde1e6; --fg: #1a1d23; --fg-muted: #5b6470;
  --accent: #2f6fe0; --pass: #1e9e5a; --pass-ink: #eafff3; --fail: #d64545; --fail-ink: #fff0f0;
  --revised: #b5790a; --revised-ink: #fff6e6;
}}
* {{ box-sizing: border-box; }}
body {{
  background: var(--bg); color: var(--fg); font-family: var(--sans);
  margin: 0; padding: 20px 24px 60px; -webkit-font-smoothing: antialiased;
}}
h1 {{ font-size: 15px; font-weight: 600; margin: 0 0 2px; letter-spacing: .01em; }}
.subtitle {{ color: var(--fg-muted); font-size: 13px; margin: 0 0 16px; }}
.subtitle b {{ color: var(--fg); font-variant-numeric: tabular-nums; }}

.toolbar {{
  display: inline-flex; gap: 2px; padding: 3px; margin-bottom: 20px;
  background: var(--card); border: 1px solid var(--border); border-radius: 999px;
}}
.toolbar button {{
  padding: 7px 16px; cursor: pointer; background: transparent; border: none;
  color: var(--fg-muted); border-radius: 999px; font: inherit; font-size: 13px;
  font-weight: 500; transition: background .15s, color .15s;
}}
.toolbar button:hover {{ color: var(--fg); }}
.toolbar button.active {{ background: var(--accent); color: #fff; }}
.toolbar button:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 2px; }}

#cards {{ display: flex; flex-direction: column; gap: 14px; }}
.card {{
  background: var(--card); border: 1px solid var(--border); border-radius: 10px;
  padding: 14px; border-left: 3px solid var(--border);
}}
.card.fail {{ border-left-color: var(--fail); }}
.card.pass {{ border-left-color: var(--pass); }}

.card-head {{ display: flex; align-items: center; gap: 10px; margin-bottom: 10px; font-size: 12.5px; }}
.chip {{
  padding: 3px 10px; border-radius: 999px; font-weight: 700; font-size: 11px;
  letter-spacing: .04em; text-transform: uppercase;
}}
.chip.pass {{ background: var(--pass); color: var(--pass-ink); }}
.chip.fail {{ background: var(--fail); color: var(--fail-ink); }}
.cid {{ color: var(--fg-muted); font-family: var(--mono); font-size: 11.5px; overflow-wrap: anywhere; }}
.score {{ color: var(--fg-muted); font-family: var(--mono); margin-left: auto; }}
.score b {{ color: var(--fg); }}
.revised {{
  background: var(--revised); color: var(--revised-ink); padding: 2px 9px;
  border-radius: 999px; font-size: 10.5px; font-weight: 700; letter-spacing: .03em;
}}

.images {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
.images figure {{ margin: 0; min-width: 0; }}
.images figcaption {{
  font-size: 11.5px; color: var(--fg-muted); margin-bottom: 6px;
  letter-spacing: .02em;
}}
.images img {{ width: 100%; height: auto; border-radius: 6px; display: block; border: 1px solid var(--border); }}

.issues {{
  margin: 10px 0 0; padding: 10px 12px; font-size: 12px; color: var(--fg);
  background: color-mix(in srgb, var(--fail) 8%, transparent);
  border: 1px solid color-mix(in srgb, var(--fail) 25%, transparent);
  border-radius: 6px; list-style: none;
}}
.issues li {{ margin-bottom: 4px; font-family: var(--mono); font-size: 11.5px; }}
.issues li:last-child {{ margin-bottom: 0; }}

@media (max-width: 720px) {{ .images {{ grid-template-columns: 1fr; }} }}
</style></head>
<body>
<h1>hardcase judge 리뷰</h1>
<p class="subtitle"><b>{n_total}</b>건 &nbsp;·&nbsp; PASS <b>{n_pass}</b> &nbsp;·&nbsp; FAIL <b>{n_fail}</b> &nbsp;·&nbsp; revised <b>{n_revised}</b></p>
<div class="toolbar" role="tablist">
  <button class="active" onclick="filterCards('all', this)">All &middot; {n_total}</button>
  <button onclick="filterCards('fail', this)">Fail &middot; {n_fail}</button>
  <button onclick="filterCards('pass', this)">Pass &middot; {n_pass}</button>
</div>
<div id="cards">
{cards_html}
</div>
<script>
function filterCards(which, btn) {{
  document.querySelectorAll('.toolbar button').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  document.querySelectorAll('.card').forEach(c => {{
    c.style.display = (which === 'all' || c.classList.contains(which)) ? '' : 'none';
  }});
}}
</script>
</body></html>
"""


def _load_rows(judge_out_paths: list[str]) -> list[dict]:
    rows = []
    for p in judge_out_paths:
        path = Path(p)
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def _issues_html(row: dict) -> str:
    issues = (row.get("verdict") or {}).get("element_issues") or []
    if not issues:
        return ""
    items = "".join(
        f"<li>[{i.get('element_index')}] {i.get('category')} {i.get('issue')} "
        f"({i.get('severity')}): {i.get('description')}</li>"
        for i in issues[:10]
    )
    more = f"<li>...외 {len(issues) - 10}건</li>" if len(issues) > 10 else ""
    return f'<ul class="issues">{items}{more}</ul>'


def build_gallery(judge_out_paths: list[str], out_path: str, limit: int | None = None) -> dict:
    rows = _load_rows(judge_out_paths)
    if limit is not None:
        rows = rows[:limit]

    n_pass = sum(1 for r in rows if r.get("resolved"))
    n_revised = sum(1 for r in rows if r.get("revised"))

    cards = []
    for row in rows:
        image_path = row.get("image_path")
        elements = row.get("final_elements") or []
        if not image_path or not Path(image_path).exists():
            continue
        try:
            from PIL import Image
            with Image.open(image_path) as im:
                im = im.convert("RGB")
                if max(im.size) > 950:
                    scale = 950 / max(im.size)
                    im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))))
                orig_uri = to_data_uri(im, fmt="JPEG", quality=80)
            overlay = draw_boxes(image_path, elements, coord_system="pixel", max_side=950)
            label_uri = to_data_uri(overlay, fmt="JPEG", quality=80)
        except Exception as e:  # noqa: BLE001 — 이미지 하나 실패해도 갤러리 전체는 계속
            continue

        resolved = bool(row.get("resolved"))
        status_class = "pass" if resolved else "fail"
        verdict = row.get("verdict") or {}
        cards.append(_CARD_TEMPLATE.format(
            status_class=status_class,
            status_text="PASS" if resolved else "FAIL",
            id_short=row.get("id", "")[-50:],
            score=verdict.get("overall_score", "?"),
            revised_badge='<span class="revised">revised</span>' if row.get("revised") else "",
            orig_uri=orig_uri,
            label_uri=label_uri,
            n_elements=len(elements),
            issues_html=_issues_html(row),
        ))

    html = _PAGE_TEMPLATE.format(
        n_total=len(cards), n_pass=n_pass, n_fail=len(rows) - n_pass, n_revised=n_revised,
        cards_html="\n".join(cards),
    )
    Path(out_path).write_text(html, encoding="utf-8")
    return {"n_total": len(cards), "n_pass": n_pass, "n_fail": len(rows) - n_pass}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-out", action="append", required=True, dest="judge_out_paths")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    stats = build_gallery(args.judge_out_paths, args.out, args.limit)
    print(f"완료: {stats} -> {args.out}")


if __name__ == "__main__":
    main()
