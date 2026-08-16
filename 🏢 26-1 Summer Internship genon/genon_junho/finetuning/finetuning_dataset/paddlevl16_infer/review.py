#!/usr/bin/env python3
"""표 추론 결과 육안 검수 갤러리. PaddleOCR-VL 1.6 과 dots.ocr 둘 다 본다.

표 하나가 한 행이고, 3그리드로 보여준다:

    [원본 이미지 crop | 모델이 뽑은 표 HTML 렌더 | 렌더 전 HTML 원문]

`--model paddle|dots` 로 결과 디렉터리만 바꿔 끼운다. 두 추론 스크립트가 같은 스키마로
저장하기 때문에 뷰어 코드는 하나면 된다. 다만 dots 의 bbox 는 smart_resize 좌표계라
crop 할 때 원본 크기로 되돌려야 한다(`crop_table`의 `src_wh`).

가운데 칸은 브라우저가 직접 렌더한다 — Playwright 스크린샷을 뜨지 않는다. 표 6,738개를
찍으면 시간도 디스크도 크게 드는데, 정적 HTML 에 그대로 심으면 공짜인 데다 표를 마우스로
긁어 복사할 수도 있다. 그래서 만들어야 하는 이미지는 왼쪽 crop 뿐이다.

모델이 표를 0개로 본 페이지도 빼지 않는다. GT 표 검증에서는 오히려 그쪽이 제일 볼
만한데, 예측 표 기준으로만 행을 만들면 통째로 사라진다. 이런 행은 GT bbox 로 crop 하고
가운데/오른쪽에 "못 찾음"을 표시한다.

정렬 기본값은 `mix` — 정상/개수불일치/미검출을 번갈아 내보내 첫 화면에서 세 부류를 다
보게 한다. 한 부류만 파고들려면 `--sort mismatch|missed`, 원래 순서는 `--sort order`.

사용:
    python3 review.py --model paddle --port 8910
    python3 review.py --model dots   --port 8911
    python3 review.py --model dots --split test --all --no-serve
"""

from __future__ import annotations

import argparse
import html as htmllib
import http.server
import json
import socketserver
from functools import partial
from pathlib import Path

from PIL import Image

HERE = Path(__file__).parent
DATASET = HERE.parent
MANIFEST = Path("/home/jhyeo/finetuning/finetuning_dataset/gt_table_audit/manifest.jsonl")
SPLITS = ("train", "valid", "test")

# 두 모델의 출력 스키마가 같아서 뷰어는 디렉터리만 바꿔 끼우면 된다.
MODELS = {
    "paddle": (DATASET / "paddlevl16_infer", "PaddleOCR-VL 1.6"),
    "dots": (DATASET / "dotsocr_infer", "dots.ocr"),
}

CROP_PAD = 0.02      # bbox 바깥 여백 비율 — 표 경계선이 잘리면 셀 병합을 못 본다
CROP_MAX_W = 1000
CROP_MAX_H = 2400


def load_gt() -> dict[str, dict]:
    gt = {}
    with MANIFEST.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            gt[r["key"]] = r
    return gt


def crop_table(image_path: str, bbox, out_path: Path, src_wh=None) -> tuple[int, int] | None:
    """페이지에서 표 영역을 잘라 저장. 실제 저장된 (w, h) 를 돌려준다.

    `src_wh` 는 bbox 가 찍힌 좌표계 크기다. dots 는 smart_resize 된 크기(1652x2352)를
    기준으로 bbox 를 주는데 파일은 원본(1654x2339)이라, 그대로 자르면 세로로 밀린다.
    """
    if not bbox:
        return None
    try:
        im = Image.open(image_path).convert("RGB")
    except OSError:
        return None
    W, H = im.size
    if src_wh and src_wh[0] and src_wh[1] and tuple(src_wh) != (W, H):
        sx, sy = W / src_wh[0], H / src_wh[1]
        bbox = [bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy]
    x0, y0, x1, y1 = bbox
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    px, py = (x1 - x0) * CROP_PAD, (y1 - y0) * CROP_PAD
    box = (max(0, int(x0 - px)), max(0, int(y0 - py)),
           min(W, int(x1 + px)), min(H, int(y1 + py)))
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    im = im.crop(box)
    w, h = im.size
    scale = min(1.0, CROP_MAX_W / w, CROP_MAX_H / h)
    if scale < 1.0:
        im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                       Image.Resampling.LANCZOS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path, format="JPEG", quality=82)
    return im.size


def build_rows(splits, gt, res_dir: Path) -> list[dict]:
    """검수 행 목록. 예측된 표 1개 = 1행, 모델이 표를 못 본 페이지도 1행."""
    rows = []
    for split in splits:
        path = res_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                g = gt.get(r["key"], {})
                diff = r["pred_n_tables"] - r["gt_n_tables"]
                # 예측 bbox 가 찍힌 좌표계. dots 는 smart_resize 크기라 원본과 다르다.
                pred_wh = (r.get("width"), r.get("height"))
                orig_wh = (r.get("orig_width") or r.get("width"),
                           r.get("orig_height") or r.get("height"))
                common = {
                    "key": r["key"],
                    "split": split,
                    "image_path": r["image_path"],
                    "gt_n": r["gt_n_tables"],
                    "pred_n": r["pred_n_tables"],
                    "diff": diff,
                    "note": r.get("parse_note"),
                }
                if r["pred_n_tables"] == 0:
                    # 볼 게 없어 보이지만 GT 는 표가 있다고 한 자리다. GT bbox 는
                    # 0-1000 정규화이고 원본 픽셀 기준이라 원본 크기로 되돌린다.
                    W, H = orig_wh[0] or 0, orig_wh[1] or 0
                    for t in g.get("tables", []):
                        b = t.get("bbox")
                        px = ([int(b[0] * W / 1000), int(b[1] * H / 1000),
                               int(b[2] * W / 1000), int(b[3] * H / 1000)]
                              if b and W and H else None)
                        rows.append({**common, "index": t["index"], "bbox": px,
                                     "src_wh": orig_wh, "html": None,
                                     "gt_html": t.get("html", ""), "missed": True})
                else:
                    for t in r["pred_tables"]:
                        rows.append({**common, "index": t["index"], "bbox": t["bbox"],
                                     "src_wh": pred_wh, "html": t["html"],
                                     "gt_html": None, "missed": False})
    return rows


def sort_rows(rows, how):
    """검수 순서.

    `mismatch` 로 몰아 정렬하면 상위 수백 개가 전부 "못 찾은 표"라 정작 모델이 뽑은
    표 렌더링을 한 건도 못 본다. 그래서 기본값은 세 부류를 번갈아 내보내는
    `mix` 로, 첫 화면에서 정상·개수불일치·미검출을 모두 보게 한다.
    """
    if how == "order":
        return rows
    normal = [r for r in rows if not r["missed"] and r["diff"] == 0]
    mismatch = sorted((r for r in rows if not r["missed"] and r["diff"] != 0),
                      key=lambda r: -abs(r["diff"]))
    missed = [r for r in rows if r["missed"]]
    if how == "mismatch":
        return mismatch + missed + normal
    if how == "missed":
        return missed + mismatch + normal

    out, buckets = [], [normal, mismatch, missed]
    idx = [0, 0, 0]
    while any(idx[b] < len(buckets[b]) for b in range(3)):
        for b in range(3):
            if idx[b] < len(buckets[b]):
                out.append(buckets[b][idx[b]])
                idx[b] += 1
    return out


PAGE_CSS = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body { margin:0; font-family: system-ui,-apple-system,"Noto Sans KR",sans-serif;
       font-size:14px; background:#f6f7f9; color:#111; }
@media (prefers-color-scheme: dark){ body{background:#16181c;color:#e6e6e6;} }
header { position:sticky; top:0; z-index:10; background:#222; color:#fff;
         padding:10px 16px; display:flex; gap:16px; align-items:center; flex-wrap:wrap; }
header b { font-size:15px; }
header .sp { margin-left:auto; }
input[type=search]{ padding:5px 9px; border-radius:6px; border:1px solid #555;
                    background:#fff; color:#111; min-width:220px; }
.row { background:#fff; margin:14px; border-radius:10px; overflow:hidden;
       box-shadow:0 1px 4px rgba(0,0,0,.12); }
@media (prefers-color-scheme: dark){ .row{background:#212429;} }
.meta { padding:8px 14px; border-bottom:1px solid #e2e4e8; display:flex; gap:12px;
        align-items:center; flex-wrap:wrap; font-size:13px; }
@media (prefers-color-scheme: dark){ .meta{border-color:#33373d;} }
.meta .k { font-weight:700; font-family:ui-monospace,monospace; }
.pill { padding:2px 8px; border-radius:999px; font-size:12px; font-weight:600; }
.ok   { background:#e3f4e5; color:#1c6b2a; }
.warn { background:#fdecc8; color:#8a5a00; }
.bad  { background:#fbdcdc; color:#a11; }
@media (prefers-color-scheme: dark){
  .ok{background:#1d3a22;color:#8fdc9c} .warn{background:#4a3a12;color:#f0c96a}
  .bad{background:#4a1f1f;color:#f2a0a0}
}
.grid { display:grid; grid-template-columns: 1fr 1fr 1fr; gap:0; }
@media (max-width: 1100px){ .grid{ grid-template-columns:1fr; } }
.cell { padding:12px; border-right:1px solid #e2e4e8; min-width:0; overflow:auto;
        max-height:78vh; }
.cell:last-child { border-right:none; }
@media (prefers-color-scheme: dark){ .cell{border-color:#33373d;} }
.cell h4 { margin:0 0 8px; font-size:12px; text-transform:uppercase;
           letter-spacing:.04em; color:#777; font-weight:700; }
.cell img { max-width:100%; height:auto; display:block; border:1px solid #ddd; }
/* 렌더된 표는 원본과 눈으로 맞춰야 하므로 격자를 또렷하게 준다 */
.render table { border-collapse:collapse; width:auto; background:#fff; color:#111; }
.render th,.render td { border:1px solid #444; padding:5px 9px; text-align:center;
                        vertical-align:middle; overflow-wrap:anywhere; }
.render thead th { background:#eee; font-weight:700; }
pre.src { margin:0; white-space:pre-wrap; word-break:break-all; font-size:12px;
          line-height:1.5; font-family:ui-monospace,SFMono-Regular,monospace;
          background:#f4f4f6; padding:10px; border-radius:6px; }
@media (prefers-color-scheme: dark){ pre.src{background:#15171b;} }
.none { color:#a11; font-style:italic; }
.hidden { display:none; }
"""

PAGE_JS = """
const rows = [...document.querySelectorAll('.row')];
const q = document.getElementById('q');
const only = document.getElementById('only');
function apply(){
  const t = q.value.trim().toLowerCase();
  const o = only.value;
  let n = 0;
  for (const r of rows){
    let show = true;
    if (t && !r.dataset.key.toLowerCase().includes(t)) show = false;
    if (show && o === 'missed' && r.dataset.missed !== '1') show = false;
    if (show && o === 'mismatch' && r.dataset.diff === '0') show = false;
    r.classList.toggle('hidden', !show);
    if (show) n++;
  }
  document.getElementById('cnt').textContent = n;
}
q.addEventListener('input', apply);
only.addEventListener('change', apply);
"""


def render_gallery(rows, out_dir: Path, title: str, model_label: str) -> Path:
    parts = []
    for r in rows:
        diff = r["diff"]
        cls = "ok" if diff == 0 else ("warn" if abs(diff) == 1 else "bad")
        sign = f"+{diff}" if diff > 0 else str(diff)
        badge = f'<span class="pill {cls}">GT {r["gt_n"]} / paddle {r["pred_n"]} ({sign})</span>'

        if r["img"]:
            left = f'<img src="images/{r["img"]}" loading="lazy" alt="{r["key"]} #{r["index"]}">'
        else:
            left = '<div class="none">crop 실패 (bbox 없음)</div>'

        if r["missed"]:
            mid = f'<div class="none">{htmllib.escape(model_label)} 가 이 자리에서 표를 못 찾음</div>'
            src = ('<h4>참고 — GT HTML</h4><pre class="src">'
                   + htmllib.escape(r["gt_html"] or "") + "</pre>")
        else:
            mid = r["html"] or '<div class="none">(내용 없음)</div>'
            src = '<pre class="src">' + htmllib.escape(r["html"] or "") + "</pre>"

        note = (f'<span class="pill bad">파싱 {htmllib.escape(r["note"])}</span>'
                if r.get("note") else "")

        parts.append(f"""
<div class="row" data-key="{htmllib.escape(r['key'])}" data-diff="{diff}" data-missed="{int(r['missed'])}">
  <div class="meta">
    <span class="k">{htmllib.escape(r['key'])} #{r['index']}</span>
    <span>{r['split']}</span>
    {badge}
    {note}
    <span style="color:#888">{htmllib.escape(Path(r['image_path']).name)}</span>
  </div>
  <div class="grid">
    <div class="cell"><h4>1. 원본 이미지</h4>{left}</div>
    <div class="cell render"><h4>2. {model_label} 표 렌더링</h4>{mid}</div>
    <div class="cell"><h4>3. 렌더링 전 HTML</h4>{src}</div>
  </div>
</div>""")

    html = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{htmllib.escape(title)}</title>
<style>{PAGE_CSS}</style></head>
<body>
<header>
  <b>{htmllib.escape(model_label)} 표 추론 검수</b>
  <span><span id="cnt">{len(rows)}</span> / {len(rows)} 행</span>
  <select id="only">
    <option value="all">전체</option>
    <option value="mismatch">GT 와 표 개수 다른 것만</option>
    <option value="missed">모델이 못 찾은 표만</option>
  </select>
  <input type="search" id="q" placeholder="key 로 검색 (예: train_000012)">
  <span class="sp"></span>
</header>
{''.join(parts)}
<script>{PAGE_JS}</script>
</body></html>"""

    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "index.html"
    p.write_text(html, encoding="utf-8")
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=tuple(MODELS), default="paddle")
    ap.add_argument("--split", choices=SPLITS, action="append")
    ap.add_argument("--out-dir", help="기본값은 모델별 review_out")
    ap.add_argument("--limit", type=int, default=400, help="렌더할 표 개수")
    ap.add_argument("--all", action="store_true", help="--limit 무시하고 전부")
    ap.add_argument("--sort", choices=("mix", "mismatch", "missed", "order"), default="mix")
    ap.add_argument("--order-like", choices=tuple(MODELS),
                    help="다른 모델 갤러리와 같은 페이지 순서로 낸다. 두 포트를 나란히 "
                         "띄우고 같은 자리에서 비교할 때 쓴다.")
    ap.add_argument("--port", type=int, default=8905)
    ap.add_argument("--no-serve", action="store_true")
    args = ap.parse_args()

    splits = args.split or list(SPLITS)
    res_dir, model_label = MODELS[args.model]
    out_dir = Path(args.out_dir) if args.out_dir else res_dir / "review_out"
    if not any((res_dir / f"{s}.jsonl").exists() for s in splits):
        ap.error(f"{res_dir} 에 결과 jsonl 이 없다. 먼저 추론을 돌릴 것.")

    gt = load_gt()
    rows = build_rows(splits, gt, res_dir)
    total = len(rows)

    if args.order_like:
        # 기준 모델이 뽑았을 순서를 그대로 재현한다(sort_rows 가 결정적이라 가능).
        # 행 단위로는 못 맞춘다 — 페이지당 표 개수가 모델마다 다르기 때문이다.
        # 그래서 페이지(key) 순서를 맞추고, 그 안에서는 표 번호 순으로 낸다.
        ref_dir, ref_label = MODELS[args.order_like]
        ref = sort_rows(build_rows(splits, gt, ref_dir), args.sort)
        if not args.all:
            ref = ref[: args.limit]
        seq, seen = [], set()
        for r in ref:
            if r["key"] not in seen:
                seen.add(r["key"])
                seq.append(r["key"])
        by_key = {}
        for r in rows:
            by_key.setdefault(r["key"], []).append(r)
        ordered = []
        for k in seq:
            ordered.extend(sorted(by_key.get(k, []), key=lambda r: r["index"]))
        rows = ordered if args.all else ordered[: args.limit]
        print(f"[{model_label}] 표 {total}개 중 {len(rows)}개 렌더 "
              f"({ref_label} 갤러리와 같은 페이지 순서, {len(seq)}개 페이지)")
    else:
        rows = sort_rows(rows, args.sort)
        if not args.all:
            rows = rows[: args.limit]
        print(f"[{model_label}] 표 {total}개 중 {len(rows)}개 렌더 (정렬: {args.sort})")

    img_dir = out_dir / "images"
    for i, r in enumerate(rows, 1):
        name = f"{r['key']}_t{r['index']}.jpg"
        ok = crop_table(r["image_path"], r["bbox"], img_dir / name, r.get("src_wh"))
        r["img"] = name if ok else None
        if i % 100 == 0:
            print(f"  crop {i}/{len(rows)}", flush=True)

    p = render_gallery(rows, out_dir, f"{model_label} 표 추론 검수", model_label)
    n_missed = sum(1 for r in rows if r["missed"])
    print(f"→ {p}  (표 {len(rows)}개, 그중 {model_label} 가 못 찾은 자리 {n_missed}개)")

    if args.no_serve:
        print(f"브라우저로 열 것: {p}")
        return 0

    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(out_dir))
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("0.0.0.0", args.port), handler) as httpd:
        print(f"http://localhost:{args.port}/  (Ctrl-C 로 종료)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n종료")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
