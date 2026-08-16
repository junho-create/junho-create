#!/usr/bin/env python3
"""페이지 1장에 대한 judge 입력 이미지 2장을 만든다.

    A(overlay) : 원본 페이지 이미지 + Table bbox 를 #1, #2 … 번호로 오버레이
    B(render)  : 그 페이지의 Table div GT HTML 만 같은 번호를 붙여 브라우저 렌더

`ocr_filter.report.render.draw_boxes` 와 labeler 의 Playwright `Renderer` 를 그대로
재사용한다. 단독 실행하면 manifest 의 앞 N건에 대해 두 이미지를 만들어 본다(렌더 확인용).

사용:
    PYTHONPATH=/home/jhyeo/ocr_file_filter/labeler:/home/jhyeo/ocr_file_filter \
        python3 render.py --limit 20
"""

from __future__ import annotations

import argparse
import html as html_mod
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BASE_DIR = Path(__file__).parent
WORK_DIR = BASE_DIR / "work"

# 오버레이 이미지 긴 변. draw_boxes 기본값 1000 은 1654x2339 페이지에서 본문 글자가
# 뭉개져 judge 가 표 내용을 못 읽는다.
OVERLAY_MAX_SIDE = 1600
# 렌더 스크린샷 상한. 긴 변 하나로 묶으면 표가 여러 개인 페이지에서 세로로 길쭉한
# 이미지가 통째로 줄어들어 글자가 뭉개진다(표 27개 페이지 = 세로 수만 px). 가로/세로를
# 따로 잡아 가독성을 지킨다.
RENDER_MAX_W = 1600
RENDER_MAX_H = 3000

RENDERER_CONFIG = {
    "renderer": {
        "browser": "chromium",
        # channel 미지정 = 번들 chromium. Google Chrome 설치 여부에 의존하지 않는다.
        "viewport_width": 1600,
        "viewport_height": 900,
        # 로컬 정적 HTML 이고 외부 리소스(KaTeX CDN 등)를 안 쓰므로 대기 불필요.
        # 기본값 2000ms 를 그대로 두면 5,030장 × 2초 = 2.8시간이 순수 대기로 날아간다.
        "wait_after_load_ms": 0,
        "full_page_screenshot": True,
        "device_scale_factor": 2,
    }
}

# labeler/tsr/templates/compare_template.html 의 CSS 그대로 + 두 가지 추가:
#   body{width:max-content} — full_page 스크린샷이 가로로 넘치는 표를 자르지 않게
#   .tag                    — 오버레이의 #N 과 짝을 맞추는 표 번호
PAGE_TEMPLATE = """<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8" />
<style>
  body {
    margin: 16px;
    width: max-content;
    min-width: calc(100vw - 32px);
    font-family: system-ui, -apple-system, sans-serif;
    font-size: 14px;
    background: white;
    color: #111;
  }
  table { border-collapse: collapse; width: auto; }
  th, td {
    border: 1px solid #333;
    padding: 6px 10px;
    vertical-align: middle;
    text-align: center;
    white-space: normal;
    overflow-wrap: anywhere;
  }
  thead th { background: #f0f0f0; font-weight: bold; }
  caption { caption-side: top; text-align: left; font-weight: bold; margin-bottom: 8px; }
  p, small { margin: 6px 0; }
  .tag {
    font-weight: bold;
    color: #b00;
    font-size: 18px;
    margin: 18px 0 6px;
    border-top: 2px dashed #b00;
    padding-top: 10px;
  }
  .tag:first-child { border-top: none; padding-top: 0; margin-top: 0; }
  .empty { color: #b00; font-style: italic; }
</style>
</head>
<body>
{{ CONTENT }}
</body>
</html>
"""


def build_page_html(tables: list[dict]) -> str:
    """표들을 #N 태그와 함께 한 페이지 HTML 로 조립."""
    parts = []
    for t in tables:
        parts.append(f'<div class="tag">#{t["index"]}</div>')
        inner = t["html"].strip()
        if not inner:
            parts.append('<div class="empty">(GT 내용 없음)</div>')
        else:
            parts.append(inner)
    return PAGE_TEMPLATE.replace("{{ CONTENT }}", "\n".join(parts))


def _downscale(path: Path, max_w: int, max_h: int) -> None:
    with Image.open(path) as im:
        w, h = im.size
        scale = min(1.0, max_w / w, max_h / h)
        if scale >= 1.0:
            return
        im.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                  Image.Resampling.LANCZOS).save(path)


# 이 머신에 DejaVu 는 없다(확인함). 존재하는 것 중 굵은 산세리프를 순서대로 시도한다.
FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/noto/NotoSansMono-Bold.ttf",
    "/usr/share/fonts/truetype/nanum/NanumSquareB.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
)
BOX_COLOR = (220, 0, 0)


def _label_font(image_width: int) -> ImageFont.ImageFont:
    size = max(24, image_width // 26)
    for path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    # 폴백은 약 11px 비트맵 폰트라 번호가 거의 안 보인다 — 여기까지 오면 안 된다.
    return ImageFont.load_default()


def make_overlay(row: dict, out_path: Path) -> Path:
    """페이지 이미지에 Table bbox 를 #N 번호와 함께 그려 저장.

    `ocr_filter.report.render.draw_boxes` 를 쓰려다 직접 그린다 — draw_boxes 는
    PIL 기본 비트맵 폰트(약 11px 고정)로 라벨을 찍어서 1600px 페이지에서는 번호가
    거의 안 보이고, 미등록 카테고리 색(#34495e)이 표 테두리와 섞인다. judge 가
    #N 을 못 읽으면 표 정렬 자체가 무너지므로 번호 가독성이 최우선이다.
    """
    im = Image.open(row["image_path"]).convert("RGB")
    w, h = im.size
    if max(w, h) > OVERLAY_MAX_SIDE:
        scale = OVERLAY_MAX_SIDE / max(w, h)
        im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                       Image.Resampling.LANCZOS)
        w, h = im.size

    draw = ImageDraw.Draw(im)
    font = _label_font(w)
    line_width = max(3, w // 300)

    for t in row["tables"]:
        if not t["bbox"]:
            continue
        x0, y0, x1, y1 = (t["bbox"][0] / 1000 * w, t["bbox"][1] / 1000 * h,
                          t["bbox"][2] / 1000 * w, t["bbox"][3] / 1000 * h)
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        draw.rectangle([x0, y0, x1, y1], outline=BOX_COLOR, width=line_width)

        tag = f"#{t['index']}"
        tw, th = draw.textbbox((0, 0), tag, font=font)[2:]
        # 라벨은 박스 바깥 위쪽에 붙이되, 페이지 최상단에 붙은 표는 안쪽으로 넣는다.
        ly = y0 - th - 8 if y0 - th - 8 >= 0 else y0 + 2
        lx = max(0, min(x0, w - tw - 12))
        draw.rectangle([lx, ly, lx + tw + 10, ly + th + 6], fill=BOX_COLOR)
        draw.text((lx + 5, ly + 2), tag, fill="white", font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path, format="JPEG", quality=88)
    return out_path


def render_tables(row: dict, renderer, work_dir: Path) -> Path:
    """GT 표 HTML 을 렌더해 PNG 로 저장. 실패하면 예외를 그대로 올린다."""
    key = row["key"]
    work_dir.mkdir(parents=True, exist_ok=True)
    html_path = work_dir / f"{key}.html"
    png_path = work_dir / f"{key}.png"
    html_path.write_text(build_page_html(row["tables"]), encoding="utf-8")
    renderer.render_and_screenshot(str(html_path), str(png_path))
    _downscale(png_path, RENDER_MAX_W, RENDER_MAX_H)
    return png_path


def new_renderer():
    """설정이 박힌 Renderer 를 만들어 start() 까지 마친 뒤 돌려준다."""
    from genos.doc_parser.labeler.core.renderer import Renderer

    r = Renderer(RENDERER_CONFIG)
    r.start()
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(BASE_DIR / "manifest.jsonl"))
    ap.add_argument("--out-dir", default=str(WORK_DIR))
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--keys", nargs="*", help="특정 key 만")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.manifest, encoding="utf-8")]
    if args.keys:
        want = set(args.keys)
        rows = [r for r in rows if r["key"] in want]
    else:
        rows = rows[: args.limit]

    out_dir = Path(args.out_dir)
    renderer = new_renderer()
    try:
        for r in rows:
            ov = make_overlay(r, out_dir / "overlays" / f"{r['key']}.jpg")
            pg = render_tables(r, renderer, out_dir / "renders")
            print(f"{r['key']}  tables={r['n_tables']}  {ov.name}  {pg.name}")
    finally:
        renderer.stop()
    print(f"\n→ {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
