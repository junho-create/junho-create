"""table_src_6902 (data/{train,valid,test}.jsonl) 파서.

원본 필드: image_path, original_gt_html, gt_html(정제), thinking(CoT), complexity,
prompt_style, ocr_info. id 필드가 없어 "table:<split>:<basename(stem)>" 으로 합성한다.

image_path(예: "images/T02_C02_50008_1066_081.jpg")도 layout 과 마찬가지로
build_unified_dataset 단계에서 images/table/<basename> 으로 이미 평탄화돼 있다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from ocr_filter.io.schema import Record

SPLITS = ("train", "valid", "test")


def load_table_source(table_dir: str | Path, images_root: str | Path) -> Iterator[Record]:
    table_dir = Path(table_dir)
    images_root = Path(images_root)

    for split in SPLITS:
        split_path = table_dir / "data" / f"{split}.jsonl"
        if not split_path.exists():
            continue
        with open(split_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                basename = Path(item["image_path"]).name
                resolved = images_root / "table" / basename
                yield Record(
                    id=f"table:{split}:{Path(basename).stem}",
                    image_path=str(resolved),
                    gt=item.get("gt_html"),
                    source_type="table",
                    meta={
                        "split": split,
                        "original_gt_html": item.get("original_gt_html"),
                        "thinking": item.get("thinking"),
                        "complexity": item.get("complexity"),
                        "prompt_style": item.get("prompt_style"),
                        "ocr_info": item.get("ocr_info", []),
                        "raw_image_path": item["image_path"],
                        "image_exists": resolved.exists(),
                    },
                )
