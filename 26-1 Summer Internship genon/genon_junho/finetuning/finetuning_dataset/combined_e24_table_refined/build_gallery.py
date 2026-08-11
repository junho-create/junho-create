#!/usr/bin/env python3
"""final_gt_1471.jsonl을 static HTML 갤러리로 빌드한다.

이미지는 bbox 오버레이로 미리 생성해서 static/images/에 저장.
HTML 또한 static/로 생성.

사용:
    python3 build_gallery.py           # 1471개 모두 (느림)
    python3 build_gallery.py --limit 100  # 처음 100개만
"""

import argparse
import json
import os
import re
from pathlib import Path
from io import BytesIO

from PIL import Image, ImageDraw, ImageFont

DATA_FILE = Path("/home/jhyeo/finetuning/finetuning_dataset/combined_e24_table_refined/final_gt_1471.jsonl")
OUT_DIR = Path("/home/jhyeo/finetuning/finetuning_dataset/combined_e24_table_refined/static")

COLORS = {
    "Table": (255, 0, 0),
    "Text": (0, 0, 255),
    "Picture": (0, 255, 0),
    "Section-header": (255, 165, 0),
    "Title": (128, 0, 128),
    "Page-footer": (192, 192, 192),
    "Page-header": (192, 192, 192),
    "List-item": (165, 42, 42),
    "Caption": (255, 192, 203),
    "Footnote": (210, 180, 140),
    "Formula": (255, 255, 0),
}

# 폰트 준비
FONTS = [
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttf",
]
try:
    font = ImageFont.truetype(FONTS[0], 16)
except:
    try:
        font = ImageFont.truetype(FONTS[1], 16)
    except:
        font = ImageFont.load_default()

def extract_boxes(gt_html):
    """gt_html에서 bbox와 label을 추출한다."""
    boxes = []
    for match in re.finditer(r'<div data-bbox="([^"]+)" data-label="([^"]+)"', gt_html):
        bbox_str, label = match.groups()
        try:
            bbox = [int(float(x)) for x in bbox_str.split()]
            if len(bbox) == 4:
                boxes.append({"bbox": bbox, "label": label})
        except ValueError:
            pass
    return boxes

def denormalize_bbox(bbox, w, h):
    """0-1000 정규화 좌표를 픽셀로."""
    x0, y0, x1, y1 = bbox
    return [int(x0 * w / 1000), int(y0 * h / 1000), int(x1 * w / 1000), int(y1 * h / 1000)]

def build_overlaid_image(item, idx, out_path):
    """bbox 오버레이된 이미지를 저장한다."""
    img_path = item["image_path"]

    if not os.path.exists(img_path):
        img = Image.new("RGB", (800, 600), color=(240, 240, 240))
        draw = ImageDraw.Draw(img)
        draw.text((50, 50), f"Not found: {img_path.split('/')[-1]}", fill=(255, 0, 0), font=font)
    else:
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        draw = ImageDraw.Draw(img)

        boxes = extract_boxes(item["gt_html"])
        for b in boxes:
            bbox = denormalize_bbox(b["bbox"], w, h)
            label = b["label"]
            color = COLORS.get(label, (200, 200, 200))

            x0, y0, x1, y1 = bbox
            draw.rectangle([x0, y0, x1, y1], outline=color, width=2)

            try:
                text_bbox = draw.textbbox((x0, y0 - 20), label, font=font)
                draw.rectangle(text_bbox, fill=color)
            except AttributeError:
                pass
            draw.text((x0, y0 - 20), label, fill="white", font=font)

    img.save(out_path, "JPEG", quality=85)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    # 디렉터리 준비
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "images").mkdir(exist_ok=True)

    # 데이터 로드
    data = []
    with DATA_FILE.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    limit = args.limit if args.limit else len(data)
    print(f"처음 {limit}개 아이템을 처리합니다.")

    # 이미지 생성
    for idx, item in enumerate(data[:limit]):
        if (idx + 1) % 50 == 0:
            print(f"  {idx + 1}/{limit} 이미지 생성 중...")
        out_path = OUT_DIR / "images" / f"{idx:05d}.jpg"
        build_overlaid_image(item, idx, out_path)

    # HTML 생성
    print("HTML 갤러리 생성 중...")
    html_head = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GT 레이아웃 2그리드</title>
<style>
:root {{ color-scheme: light dark; }}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; font-family: system-ui, -apple-system, "Noto Sans CJK KR", sans-serif;
  font-size: 14px; background: #f6f7f9; color: #111;
}}
@media (prefers-color-scheme: dark) {{ body {{ background: #16181c; color: #e6e6e6; }} }}

header {{
  position: sticky; top: 0; z-index: 10; background: #222; color: #fff;
  padding: 12px 16px; display: flex; gap: 16px; align-items: center;
}}
header b {{ font-size: 15px; }}
.info {{ margin-left: auto; color: #aaa; font-size: 13px; }}

.container {{ max-width: 100%; margin: 0; padding: 16px; }}
.grid {{ display: grid; gap: 16px; }}
.row {{ background: #fff; border-radius: 8px; overflow: hidden;
       box-shadow: 0 1px 4px rgba(0,0,0,.12); display: grid; grid-template-columns: 1fr 1fr; }}
@media (prefers-color-scheme: dark) {{ .row {{ background: #212429; }} }}
@media (max-width: 1200px) {{ .row {{ grid-template-columns: 1fr; }} }}

.cell {{ padding: 12px; overflow: auto; max-height: 600px; }}
.cell h4 {{ margin: 0 0 8px; font-size: 12px; text-transform: uppercase;
           letter-spacing: .04em; color: #777; font-weight: 700; }}
.cell img {{ max-width: 100%; height: auto; display: block; }}
.cell-render {{ background: #fafafa; }}
@media (prefers-color-scheme: dark) {{ .cell-render {{ background: #1a1d21; }} }}

table {{ border-collapse: collapse; width: auto; background: #fff; color: #111; }}
@media (prefers-color-scheme: dark) {{ table {{ background: #212429; color: #e6e6e6; }} }}
table th, table td {{ border: 1px solid #444; padding: 5px 9px; text-align: center;
                     vertical-align: middle; font-size: 13px; overflow-wrap: break-word; }}
table th {{ background: #eee; font-weight: 700; }}
@media (prefers-color-scheme: dark) {{ table th {{ background: #2a2d32; }} }}

.meta {{ padding: 8px 12px; border-bottom: 1px solid #e2e4e8; font-size: 12px; }}
@media (prefers-color-scheme: dark) {{ .meta {{ border-color: #33373d; }} }}
.meta .path {{ color: #888; font-family: ui-monospace, monospace; word-break: break-all; }}

.legend {{ display: flex; gap: 12px; flex-wrap: wrap; padding: 8px 12px; font-size: 12px;
          border-top: 1px solid #e2e4e8; }}
@media (prefers-color-scheme: dark) {{ .legend {{ border-color: #33373d; }} }}
.legend-item {{ display: flex; align-items: center; gap: 4px; }}
.legend-dot {{ width: 12px; height: 12px; border: 2px solid; }}
</style>
</head>
<body>
<header>
  <b>GT 레이아웃 2그리드</b>
  <span class="info">총 {limit} 페이지</span>
</header>

<div class="container">
<div class="grid">
"""
    html_parts = [html_head]

    for idx, item in enumerate(data[:limit]):
        img_name = f"{idx:05d}.jpg"
        img_filename = item["image_path"].split("/")[-1] if "image_path" in item else f"idx-{idx}"

        html_parts.append(f"""<div class="row">
  <div class="cell">
    <h4>원본 + bbox 오버레이</h4>
    <img src="images/{img_name}" alt="idx-{idx}">
  </div>

  <div class="cell cell-render">
    <h4>GT HTML 렌더링</h4>
    <div class="meta">
      <div class="path" title="{item.get('image_path', '')}">{img_filename}</div>
    </div>
    <div style="padding: 8px; font-size: 13px;">
      {item.get('gt_html', '(No HTML)')}
    </div>
  </div>
</div>

<div style="background: #fff; border-radius: 8px; padding: 8px 12px; font-size: 12px;">
  <div class="legend">
    <div class="legend-item"><div class="legend-dot" style="border-color: #ff0000;"></div> Table</div>
    <div class="legend-item"><div class="legend-dot" style="border-color: #0000ff;"></div> Text</div>
    <div class="legend-item"><div class="legend-dot" style="border-color: #00ff00;"></div> Picture</div>
    <div class="legend-item"><div class="legend-dot" style="border-color: #ffa500;"></div> Section-header</div>
    <div class="legend-item"><div class="legend-dot" style="border-color: #800080;"></div> Title</div>
    <div class="legend-item"><div class="legend-dot" style="border-color: #c0c0c0;"></div> Header/Footer</div>
    <div class="legend-item"><div class="legend-dot" style="border-color: #a52a2a;"></div> List-item</div>
  </div>
</div>
""")

    html_parts.append("""</div>
</div>
</body>
</html>
""")

    html_content = "".join(html_parts)
    out_html = OUT_DIR / "index.html"
    with out_html.open("w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"완성! {out_html}를 브라우저에서 열어주세요.")

if __name__ == "__main__":
    main()
