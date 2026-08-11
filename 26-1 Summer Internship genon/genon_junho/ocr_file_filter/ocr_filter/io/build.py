"""layout_src + table_src → 통일 스키마 JSONL 로 합치기.

    python -m ocr_filter.cli io build      # ./_work/unified.jsonl 생성

이미지 존재 여부(meta.image_exists)까지 같이 적어두므로, 원본 경로가 깨진
레코드가 있으면 build() 반환값의 missing_images 로 바로 알 수 있다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ocr_filter.io.layout import load_layout_source
from ocr_filter.io.schema import Record, write_jsonl
from ocr_filter.io.table import load_table_source


@dataclass
class BuildStats:
    n_layout: int
    n_table: int
    n_missing_images: int

    @property
    def n_total(self) -> int:
        return self.n_layout + self.n_table


def build_unified(paths: dict, out_path: str | Path) -> BuildStats:
    """paths 는 default.yaml 의 `paths:` 섹션 (layout_src/table_src/images 키 사용)."""
    records: list[Record] = []
    records.extend(load_layout_source(paths["layout_src"], paths["images"]))
    n_layout = len(records)

    records.extend(load_table_source(paths["table_src"], paths["images"]))
    n_table = len(records) - n_layout

    n_missing = sum(1 for r in records if not r.meta.get("image_exists", True))

    write_jsonl(records, out_path)
    return BuildStats(n_layout=n_layout, n_table=n_table, n_missing_images=n_missing)
