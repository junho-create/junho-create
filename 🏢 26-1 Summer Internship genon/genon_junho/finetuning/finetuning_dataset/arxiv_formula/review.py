#!/usr/bin/env python3
"""시각 QA — 크롭 원본 옆에 채택된 LaTeX 를 KaTeX 로 렌더해 나란히 보여주는 HTML 갤러리.

계획의 검증 5번(사람 눈 확인)용이다. 특히 `judge_authored` 는 두 모델이 다 틀렸다고 보고
judge 가 직접 쓴 것이라 **전수 확인** 대상이다.

크롭 이미지는 base64 로 박아 넣어서 파일 하나만 열면 되게 한다. KaTeX 는 벤더링된 것을
상대경로로 물리므로 출력 HTML 을 _vendor_katex 옆에 두거나 --katex 로 경로를 준다.

사용:
    python3 review.py --source judge_authored          # 전수
    python3 review.py --source all --limit 200
"""

from __future__ import annotations

import argparse
import base64
import html as _html_lib
import json
import random
from pathlib import Path


def _esc_attr(s: str) -> str:
    """HTML 속성값 이스케이프. 백슬래시는 건드리지 않는다(LaTeX 가 죽는다)."""
    return _html_lib.escape(s or "", quote=True)

HERE = Path(__file__).parent
KATEX = Path("/home/jhyeo/ocr_file_filter/ocr_filter/report/_vendor_katex")

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="%(katex)s/katex.min.css">
<script src="%(katex)s/katex.min.js"></script>
<style>
 body{font:14px sans-serif;margin:24px;background:#fafafa}
 .r{background:#fff;border:1px solid #ddd;border-radius:6px;margin-bottom:18px;padding:12px}
 .hd{font:12px monospace;color:#666;margin-bottom:8px}
 .tag{display:inline-block;padding:1px 7px;border-radius:3px;color:#fff;font-weight:bold}
 .dots{background:#2a7}.paddle{background:#27a}.judge_authored{background:#c62}
 .cols{display:flex;gap:16px;align-items:flex-start;flex-wrap:wrap}
 .col{flex:1 1 380px;min-width:320px}
 .lb{font:11px monospace;color:#999;margin-bottom:4px}
 img{max-width:100%%;border:1px solid #eee}
 .eq{padding:10px;background:#fcfcfc;border:1px solid #eee;overflow-x:auto}
 .tex{font:11px monospace;color:#555;word-break:break-all;margin-top:6px}
 .cand{font:11px monospace;color:#888;word-break:break-all;margin-top:3px}
</style></head><body>
<h2>수식 GT 검수 — %(n)d건 (%(src)s)</h2>
%(rows)s
<script>
document.querySelectorAll('.eq').forEach(function(e){
  try{ katex.render(e.dataset.tex, e, {displayMode:true, throwOnError:true}); }
  catch(err){ e.innerHTML='<span style="color:#b00;font:12px monospace">KaTeX ERROR: '+err.message+'</span>'; }
});
</script></body></html>"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=str(HERE / "formula_pairs.jsonl"))
    ap.add_argument("--verdicts", default=str(HERE / "formula_verdicts.jsonl"))
    ap.add_argument("--source", default="judge_authored",
                    choices=["judge_authored", "dots", "paddle", "unresolved", "all"])
    ap.add_argument("--limit", type=int, default=0, help="0=전수")
    ap.add_argument("--out", default=str(HERE / "review.html"))
    ap.add_argument("--katex", default=str(KATEX))
    args = ap.parse_args()

    pairs = {}
    with open(args.pairs, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                pairs[r["pair_id"]] = r

    rows = []
    with open(args.verdicts, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            v = json.loads(line)
            if args.source != "all" and v["source"] != args.source:
                continue
            p = pairs.get(v["pair_id"])
            if p:
                rows.append((v, p))

    if args.limit and len(rows) > args.limit:
        random.seed(0)
        rows = random.sample(rows, args.limit)

    html = []
    for v, p in rows:
        try:
            b64 = base64.b64encode(Path(p["crop_path"]).read_bytes()).decode()
            img = f'<img src="data:image/png;base64,{b64}">'
        except Exception:  # noqa: BLE001
            img = "<i>크롭 없음</i>"
        html.append(
            f'<div class="r"><div class="hd">'
            f'<span class="tag {v["source"]}">{v["source"]}</span> &nbsp;{v["pair_id"]}'
            f' &nbsp; iou={p["iou"]} scale={p["crop_scale"]} katex_ok={v["katex_ok"]}'
            f' &nbsp; note={v["judge_note"][:70]}</div>'
            f'<div class="cols">'
            f'<div class="col"><div class="lb">원본 크롭</div>{img}</div>'
            f'<div class="col"><div class="lb">채택 LaTeX 렌더</div>'
            # HTML 속성은 백슬래시를 그대로 실어야 한다. json.dumps 로 만들면 `\` 가 `\\` 로
            # 이스케이프되고, HTML 속성 파서는 그걸 되돌리지 않아서 KaTeX 가 `\\`(줄바꿈) +
            # `left(`(그냥 글자)로 읽는다 — 수식이 전부 이탤릭 평문으로 렌더된다.
            f'<div class="eq" data-tex="{_esc_attr(v["latex"])}"></div>'
            f'<div class="tex">{v["latex"][:400]}</div>'
            f'<div class="cand">dots  : {p["cand_dots"][:200]}</div>'
            f'<div class="cand">paddle: {p["cand_paddle"][:200]}</div>'
            f'</div></div></div>')

    Path(args.out).write_text(
        PAGE % {"katex": args.katex, "n": len(rows), "src": args.source,
                "rows": "\n".join(html)}, encoding="utf-8")
    print(f"{len(rows)}건 -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
