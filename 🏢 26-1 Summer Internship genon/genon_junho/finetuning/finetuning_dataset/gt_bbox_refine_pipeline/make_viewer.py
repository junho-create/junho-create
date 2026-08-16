"""
정적 비교 뷰어 HTML 생성 (self-contained, base64 임베드 → 서버 없이 열림).
변경된 박스 하나하나를 확대 crop 해서 빨강(기존)/파랑(보정)을 겹쳐 보여준다
(전체 페이지 축소본만으로는 몇 픽셀 차이가 안 보여서 crop 방식으로 교체).
입력은 원본 train.jsonl (읽기 전용). 보정은 refine_gt_bbox.process_record 로 재계산.
"""
from __future__ import annotations
import argparse, base64, html, io, json, re
from pathlib import Path
from PIL import Image, ImageDraw
from refine_gt_bbox import process_record

THUMB_W = 340     # 페이지 전체 썸네일 폭
CROP_MIN = 320    # 확대 crop 최소 출력 폭

def denorm(b, w, h, s=1000):
    return [b[0]/s*w, b[1]/s*h, b[2]/s*w, b[3]/s*h]

def to_data_uri(img, quality=82):
    buf = io.BytesIO(); img.convert("RGB").save(buf, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

def make_thumb(im):
    w, h = im.size
    r = THUMB_W / w
    return im.resize((THUMB_W, int(h * r)))

def crop_diff(im, old, new, pad_frac=0.7, min_out=CROP_MIN):
    w, h = im.size
    ox0, oy0, ox1, oy1 = denorm(old, w, h)
    nx0, ny0, nx1, ny1 = denorm(new, w, h)
    x0, y0 = min(ox0, nx0), min(oy0, ny0)
    x1, y1 = max(ox1, nx1), max(oy1, ny1)
    bw, bh = max(x1 - x0, 1), max(y1 - y0, 1)
    padx, pady = max(bw * pad_frac, 15), max(bh * pad_frac, 15)
    cx0, cy0 = max(0, x0 - padx), max(0, y0 - pady)
    cx1, cy1 = min(w, x1 + padx), min(h, y1 + pady)
    crop = im.crop((cx0, cy0, cx1, cy1)).convert("RGB")
    cw, ch = crop.size
    scale = 1.0
    if cw < min_out:
        scale = min_out / cw
        crop = crop.resize((int(cw * scale), int(ch * scale)))
    d = ImageDraw.Draw(crop)
    d.rectangle([(ox0-cx0)*scale, (oy0-cy0)*scale, (ox1-cx0)*scale, (oy1-cy0)*scale],
                outline=(230, 40, 40), width=3)
    d.rectangle([(nx0-cx0)*scale, (ny0-cy0)*scale, (nx1-cx0)*scale, (ny1-cy0)*scale],
                outline=(50, 130, 240), width=3)
    return crop

def edge_shift(o, n):
    gh = (o[3] - o[1]) or 1
    return max(abs(n[k] - o[k]) for k in range(4)) / gh

def fmt_bbox(b):
    return "[" + ", ".join(f"{v:.0f}" for v in b) + "]"

CROP_CARD = """<div class="crop">
  <img src="{img}">
  <div class="meta">
    <div class="lab">{label}</div>
    <div class="coord"><b class="r">old</b> {old}</div>
    <div class="coord"><b class="b">new</b> {new}</div>
    <div class="txt">"{text}"</div>
  </div>
</div>"""

PAGE_BLOCK = """<div class="page">
  <div class="pagehead">
    <img class="thumb" src="{thumb}">
    <div class="pageinfo">
      <div class="fn">{name}</div>
      <div class="stat">변경 {nch} / 대상 {ntg}</div>
    </div>
  </div>
  <div class="crops">
{crops}
  </div>
</div>"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=25, help="저장할 페이지 수")
    ap.add_argument("--stride", type=int, default=61, help="샘플 간격(다양성)")
    ap.add_argument("--max-crops", type=int, default=10, help="페이지당 최대 crop 수(큰 변경 우선)")
    ap.add_argument("--mode", default="loose")
    args = ap.parse_args()

    pages = []
    idx = -1
    picked = 0
    for line in open(args.inp, encoding="utf-8"):
        idx += 1
        if not line.strip():
            continue
        if idx % args.stride != 0:
            continue
        rec = json.loads(line)
        imgp = rec.get("image_path", "")
        if not Path(imgp).exists():
            continue
        divs, results = process_record(rec, mode=args.mode)
        changed = [(o, n, divs[i][4]) for i, o, n, r in results if n is not None and n != o]
        if not changed:
            continue
        try:
            im = Image.open(imgp).convert("RGB")
        except Exception:
            continue
        changed.sort(key=lambda c: -edge_shift(c[0], c[1]))
        top = changed[:args.max_crops]
        crop_html = []
        for o, n, txt in top:
            crop_im = crop_diff(im, o, n)
            label = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", txt)).strip()
            label = html.escape(label[:70]) or "(빈 텍스트)"
            crop_html.append(CROP_CARD.format(
                img=to_data_uri(crop_im), label=label,
                old=fmt_bbox(o), new=fmt_bbox(n),
                text=f"shift={edge_shift(o,n):.2f}",
            ))
        pages.append(PAGE_BLOCK.format(
            thumb=to_data_uri(make_thumb(im)),
            name=html.escape(Path(imgp).name),
            nch=len(changed), ntg=len(results),
            crops="\n".join(crop_html),
        ))
        picked += 1
        if picked >= args.n:
            break

    page = PAGE.replace("{{PAGES}}", "\n".join(pages)).replace("{{N}}", str(picked))
    Path(args.out).write_text(page, encoding="utf-8")
    print(f"wrote {args.out} ({picked} pages, {Path(args.out).stat().st_size/1e6:.1f} MB)")

PAGE = """<!doctype html>
<html><head><meta charset="utf-8">
<title>GT bbox Refine Viewer (확대 diff)</title>
<style>
  body { font-family: -apple-system, sans-serif; margin:0; padding:16px; background:#111; color:#eee; }
  h1 { font-size:18px; }
  .legend { font-size:13px; color:#bbb; margin-bottom:16px; }
  .legend b.r{color:#f66;} .legend b.b{color:#6af;}
  .page { border:1px solid #333; border-radius:8px; padding:12px; margin-bottom:20px; background:#161616; }
  .pagehead { display:flex; align-items:center; gap:12px; margin-bottom:10px; }
  .pagehead .thumb { max-width:180px; border:1px solid #333; border-radius:4px; }
  .pageinfo .fn { font-size:12px; color:#9cf; word-break:break-all; }
  .pageinfo .stat { font-size:12px; color:#999; margin-top:4px; }
  .crops { display:flex; flex-wrap:wrap; gap:10px; }
  .crop { background:#1e1e1e; border:1px solid #333; border-radius:6px; padding:6px; width:340px; }
  .crop img { max-width:100%; display:block; border-radius:4px; }
  .crop .meta { font-size:11px; margin-top:6px; line-height:1.5; }
  .crop .lab { color:#ddd; margin-bottom:4px; }
  .crop .coord { font-family:Consolas,Menlo,monospace; color:#aaa; }
  .crop .coord b.r{color:#f66;} .crop .coord b.b{color:#6af;}
  .crop .txt { color:#777; margin-top:2px; }
</style></head><body>
<h1>GT bbox Refine Viewer — 확대 diff ({{N}}페이지)</h1>
<div class="legend">
  각 카드 = 변경된 박스 하나를 확대한 crop. <b class="r">빨강</b>=기존 GT, <b class="b">파랑</b>=보정 후.
  페이지당 변경폭이 큰 순서로 최대 N개만 표시. gt_html 원문은 포함하지 않음(용량 절약) — 필요하면 train_refined.jsonl 직접 조회.
</div>
{{PAGES}}
</body></html>"""

if __name__ == "__main__":
    main()
