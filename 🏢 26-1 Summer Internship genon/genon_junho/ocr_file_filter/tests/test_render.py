"""report/render.py 단위 테스트: 모델이 뒤집힌 bbox(x1<x0 또는 y1<y0)를 뱉어도
draw_boxes 가 죽지 않고 정렬해서 그려야 함 (2026-07-14 실측: target 모델이 가끔
이런 bbox 를 내서 PIL ValueError 로 갤러리 생성 전체가 죽는 문제 재현·확인함).

    pytest tests/test_render.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

from ocr_filter.report.render import draw_boxes  # noqa: E402


def _make_image(path: Path, size: tuple[int, int] = (200, 100)) -> None:
    Image.new("RGB", size, "white").save(path)


def test_draw_boxes_handles_flipped_pixel_bbox():
    with tempfile.TemporaryDirectory() as tmp:
        img_path = Path(tmp) / "x.png"
        _make_image(img_path)
        elements = [{"category": "Text", "text": "a", "bbox": [50, 80, 10, 20]}]  # x1<x0, y1<y0
        im = draw_boxes(img_path, elements, "pixel")
        assert im.size[0] > 0 and im.size[1] > 0


def test_draw_boxes_handles_flipped_norm1000_bbox():
    with tempfile.TemporaryDirectory() as tmp:
        img_path = Path(tmp) / "x.png"
        _make_image(img_path)
        elements = [{"category": "Title", "text": "a", "bbox": [500, 800, 100, 200]}]
        im = draw_boxes(img_path, elements, "norm1000")
        assert im.size[0] > 0 and im.size[1] > 0
