#!/usr/bin/env python3
"""unified_html_smoke의 layout 레코드를 OCR-enriched layout jsonl로 덮어쓴다."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def index_by_image_path(rows: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    dup = 0
    for row in rows:
        key = row.get("image_path", "")
        if not key:
            raise ValueError("layout OCR jsonl에 image_path가 없는 레코드가 있습니다")
        if key in out:
            dup += 1
        out[key] = row
    if dup:
        print(f"경고: OCR jsonl 중복 image_path {dup}건 (마지막 값 사용)")
    return out


def merge_split(
    unified_path: Path,
    ocr_index: dict[str, dict],
    output_path: Path,
) -> dict[str, int]:
    stats = Counter()
    merged: list[dict] = []

    for row in load_jsonl(unified_path):
        task_type = row.get("task_type", "")
        if task_type == "layout":
            key = row.get("image_path", "")
            if key not in ocr_index:
                stats["layout_missing_in_ocr"] += 1
                merged.append(row)
                continue
            new_row = ocr_index[key]
            if new_row.get("task_type") != "layout":
                raise ValueError(f"OCR 레코드 task_type 불일치: {key}")
            merged.append(new_row)
            if new_row.get("ocr_info"):
                stats["layout_with_ocr"] += 1
            else:
                stats["layout_empty_ocr"] += 1
        else:
            merged.append(row)
            stats[f"kept_{task_type or 'unknown'}"] += 1

    with output_path.open("w", encoding="utf-8") as f:
        for row in merged:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats["total_out"] = len(merged)
    return dict(stats)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--unified_dir",
        required=True,
        help="unified_html_smoke/data 디렉토리",
    )
    parser.add_argument(
        "--ocr_dir",
        required=True,
        help="layout OCR jsonl 디렉토리 (train/valid/test.jsonl)",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="기존 jsonl을 .bak_ocr_merge 로 백업",
    )
    args = parser.parse_args()

    unified_dir = Path(args.unified_dir)
    ocr_dir = Path(args.ocr_dir)

    for split in ("train", "valid", "test"):
        unified_path = unified_dir / f"{split}.jsonl"
        ocr_path = ocr_dir / f"{split}.jsonl"
        if not unified_path.exists():
            raise FileNotFoundError(unified_path)
        if not ocr_path.exists():
            raise FileNotFoundError(ocr_path)

        ocr_index = index_by_image_path(load_jsonl(ocr_path))
        tmp_path = unified_dir / f"{split}.jsonl.tmp"
        stats = merge_split(unified_path, ocr_index, tmp_path)

        if args.backup:
            backup_path = unified_dir / f"{split}.jsonl.bak_ocr_merge"
            shutil.copy2(unified_path, backup_path)

        tmp_path.replace(unified_path)
        print(f"[{split}] {stats}")


if __name__ == "__main__":
    main()
