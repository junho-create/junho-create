"""통일 스키마: {id, image_path, gt, source_type, meta}.

layout_src_9984 / table_src_6902 원본 두 파서(layout.py/table.py)가 공통으로
이 Record 로 변환해서 내놓는다. gt 의 모양은 source_type 에 따라 다르다:
  - layout: list[{bbox, category, text}]  (layout_elements 원본 그대로)
  - table:  str (gt_html)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator


@dataclass
class Record:
    id: str
    image_path: str          # 해결된 절대경로 (images/<layout|table>/<basename>)
    gt: Any
    source_type: str         # "layout" | "table"
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Record":
        return Record(
            id=d["id"], image_path=d["image_path"], gt=d.get("gt"),
            source_type=d["source_type"], meta=d.get("meta", {}),
        )


def write_jsonl(records: Iterable[Record], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[Record]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield Record.from_dict(json.loads(line))
