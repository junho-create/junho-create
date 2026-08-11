"""이미지 위에 레이아웃 bbox+category 오버레이 그리기 (gt/target/dots.ocr 비교용).

좌표계가 모델마다 다르다 (`ocr_filter.cmcv.normalize` 참고):
    gt / dots.ocr : 원본 이미지 픽셀 좌표
    target        : Chandra 스펙, 0-1000 정규화 좌표
    paddle        : bbox 없음 (순수 인식 텍스트) — 오버레이 대상 아님
"""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None

CATEGORY_COLORS = {
    "Title": "#e74c3c",
    "Text": "#2980b9",
    "Table": "#27ae60",
    "Page-header": "#8e44ad",
    "Page-footer": "#9b59b6",
    "Picture": "#f39c12",
    "Caption": "#16a085",
    "Footnote": "#7f8c8d",
    "Formula": "#c0392b",
    "List-item": "#2c3e50",
    "Section-header": "#d35400",
}
DEFAULT_COLOR = "#34495e"


def draw_boxes(
    image_path: str | Path,
    elements: list[dict],
    coord_system: str = "pixel",
    max_side: int = 1000,
) -> Image.Image:
    """coord_system: "pixel"(gt/dots.ocr 그대로) | "norm1000"(target, 0-1000 → 이미지 크기로 환산)."""
    im = Image.open(image_path).convert("RGB")
    w, h = im.size
    draw = ImageDraw.Draw(im)
    line_width = max(2, w // 400)

    for e in elements:
        bbox = e.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        if coord_system == "norm1000":
            x0, y0, x1, y1 = (bbox[0] / 1000 * w, bbox[1] / 1000 * h,
                               bbox[2] / 1000 * w, bbox[3] / 1000 * h)
        else:
            x0, y0, x1, y1 = bbox
        # 모델이 가끔 좌표 순서를 뒤집어 뱉는 경우(x1<x0 또는 y1<y0)가 있어 정렬해서
        # 방어한다 — 안 그러면 PIL이 ValueError 를 던져서 레코드 하나 때문에
        # ThreadPoolExecutor 전체 갤러리 생성이 통째로 죽는다.
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        category = e.get("category", "")
        color = CATEGORY_COLORS.get(category, DEFAULT_COLOR)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=line_width)
        label_y = max(0, y0 - 16)
        draw.rectangle([x0, label_y, x0 + 8 * len(category) + 6, label_y + 14], fill=color)
        draw.text((x0 + 3, label_y), category, fill="white")

    if max(im.size) > max_side:
        scale = max_side / max(im.size)
        im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    return im


def to_data_uri(im: Image.Image, fmt: str = "JPEG", quality: int = 82) -> str:
    buf = BytesIO()
    if fmt.upper() == "JPEG":
        im = im.convert("RGB")
        im.save(buf, format="JPEG", quality=quality)
    else:
        im.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = "image/png" if fmt.upper() == "PNG" else f"image/{fmt.lower()}"
    return f"data:{mime};base64,{b64}"
