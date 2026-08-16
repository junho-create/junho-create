#!/usr/bin/env python3
"""final_gt_1471.jsonl 의 레이아웃 GT를 2그리드로 본다.

왼쪽: 원본 이미지 + bbox 오버레이
오른쪽: GT HTML 렌더링

사용:
    python3 viewer.py            # 8901 포트
    python3 viewer.py --port 8902
"""

import argparse
import json
import os
import re
from io import BytesIO
from pathlib import Path

from flask import Flask, render_template, request, send_file
from PIL import Image, ImageDraw, ImageFont

app = Flask(__name__)

DATA_FILE = Path("/home/jhyeo/finetuning/finetuning_dataset/combined_e24_table_refined/final_gt_1471.jsonl")
data = []

print("데이터를 로드하는 중...")
with DATA_FILE.open(encoding="utf-8") as f:
    for line in f:
        if line.strip():
            data.append(json.loads(line))
print(f"총 {len(data)}개의 레이아웃을 로드했습니다.")

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

COLORS = {
    "Table": (255, 0, 0),        # 빨강
    "Text": (0, 0, 255),          # 파랑
    "Picture": (0, 255, 0),       # 초록
    "Section-header": (255, 165, 0),  # 주황
    "Title": (128, 0, 128),       # 보라
    "Page-footer": (192, 192, 192),  # 회색
    "Page-header": (192, 192, 192),
    "List-item": (165, 42, 42),   # 갈색
    "Caption": (255, 192, 203),   # 분홍
    "Footnote": (210, 180, 140),  # 흙색
    "Formula": (255, 255, 0),     # 노랑
}

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

@app.route("/")
def index():
    page = request.args.get("page", 1, type=int)
    per_page = 5
    total_pages = (len(data) + per_page - 1) // per_page
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    items = [(i, data[i]) for i in range(start_idx, min(end_idx, len(data)))]
    return render_template("viewer.html", items=items, page=page, total_pages=total_pages)

@app.route("/image/<int:idx>")
def serve_image(idx):
    if idx < 0 or idx >= len(data):
        return "Not found", 404

    item = data[idx]
    img_path = item["image_path"]

    if not os.path.exists(img_path):
        img = Image.new("RGB", (800, 600), color=(240, 240, 240))
        draw = ImageDraw.Draw(img)
        draw.text((50, 50), f"Not found:\n{img_path}", fill=(255, 0, 0), font=font)
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

    io = BytesIO()
    img.save(io, "JPEG", quality=85)
    io.seek(0)
    return send_file(io, mimetype="image/jpeg")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8901)
    args = parser.parse_args()
    app.run(host="0.0.0.0", port=args.port, debug=False)
