"""layout_src_9984 (labeler_converter_layout_source_9984.jsonl) 파서.

원본 필드: id, image_path, layout_elements([{bbox,category,text}]), layout_json(문자열
버전, 중복이라 버림), status, pdf_path, page_index, convert_path.

원본 image_path(예: "output/layout_images_full_20260604/.../foo_page_001.png")는
이 서버에 더 이상 존재하지 않는다 — build_unified_dataset 단계에서 이미
images/layout/<basename> 으로 평탄화·복사돼 있으므로 그 경로로 다시 매핑한다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from ocr_filter.io.schema import Record


def load_layout_source(jsonl_path: str | Path, images_root: str | Path) -> Iterator[Record]:
    images_root = Path(images_root)
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            basename = Path(item["image_path"]).name
            resolved = images_root / "layout" / basename
            yield Record(
                id=item["id"],
                image_path=str(resolved),
                gt=item.get("layout_elements", []),
                source_type="layout",
                meta={
                    "status": item.get("status"),
                    "convert_path": item.get("convert_path"),
                    "page_index": item.get("page_index"),
                    "raw_image_path": item["image_path"],
                    "image_exists": resolved.exists(),
                },
            )
