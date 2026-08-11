#!/usr/bin/env python3
"""Layout convert.json → JSONL 수집기.

labeler_feat 파이프라인(convert_only/full)이 저장한 per-page convert.json을
모아 학습용 JSONL로 내보냅니다. review.html Save 버튼 없이 자동 산출물만 사용합니다.

제외 규칙 (기본):
  - final_result.json status == ERROR
  - convert.json 파싱 실패 / 빈 배열
  - (옵션) min_elements 미만
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)


@dataclass
class ExportRecord:
    key: str
    image_path: Path
    convert_path: Path
    elements: list[dict[str, Any]]
    status: str
    pdf_path: str | None
    page_index: int | None


def _iter_page_dirs(output_dir: Path) -> Iterator[Path]:
    for final_json in sorted(output_dir.rglob("final_result.json")):
        yield final_json.parent


def _load_json(path: Path) -> dict[str, Any] | list[Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("JSON read failed %s: %s", path, exc)
        return None


def _find_convert_json(page_dir: Path) -> Path | None:
    final_data = _load_json(page_dir / "final_result.json")
    attempts = 1
    if isinstance(final_data, dict):
        attempts = int(final_data.get("attempts", 1) or 1)

    for attempt in range(attempts, 0, -1):
        attempt_dir = page_dir / f"attempt_{attempt}"
        if not attempt_dir.is_dir():
            continue
        matches = sorted(attempt_dir.glob("*.convert.json"))
        if matches:
            return matches[0]

    fallback = sorted(page_dir.glob("attempt_*/**/*.convert.json"))
    return fallback[-1] if fallback else None


def _find_page_png(page_dir: Path) -> Path | None:
    pngs = sorted(page_dir.glob("*.png"))
    # annotated/screenshot png 제외
    for p in pngs:
        if p.name.endswith(".annotated.png"):
            continue
        return p
    return pngs[0] if pngs else None


def _parse_elements(convert_path: Path) -> list[dict[str, Any]] | None:
    payload = _load_json(convert_path)
    if payload is None:
        return None
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict) and "elements" in payload:
        elems = payload["elements"]
        if isinstance(elems, list):
            return [x for x in elems if isinstance(x, dict)]
    return None


def collect_records(
    output_dir: Path,
    *,
    exclude_error: bool = True,
    min_elements: int = 1,
) -> tuple[list[ExportRecord], dict[str, int]]:
    stats = {
        "pages_seen": 0,
        "exported": 0,
        "skipped_error": 0,
        "skipped_no_convert": 0,
        "skipped_empty": 0,
        "skipped_min_elements": 0,
    }
    records: list[ExportRecord] = []

    for page_dir in _iter_page_dirs(output_dir):
        stats["pages_seen"] += 1
        final_data = _load_json(page_dir / "final_result.json")
        status = "UNKNOWN"
        if isinstance(final_data, dict):
            status = str(final_data.get("status", "UNKNOWN"))

        if exclude_error and status == "ERROR":
            stats["skipped_error"] += 1
            continue

        convert_path = _find_convert_json(page_dir)
        if convert_path is None:
            stats["skipped_no_convert"] += 1
            continue

        elements = _parse_elements(convert_path)
        if elements is None:
            stats["skipped_empty"] += 1
            continue
        if len(elements) < min_elements:
            stats["skipped_min_elements"] += 1
            continue

        image_path = _find_page_png(page_dir)
        if image_path is None:
            stats["skipped_no_convert"] += 1
            continue

        pdf_path = None
        page_index = None
        if isinstance(final_data, dict):
            item_meta = final_data.get("metadata")
            if isinstance(item_meta, dict):
                pdf_path = item_meta.get("pdf_path")
                page_index = item_meta.get("page_index")

        key = page_dir.relative_to(output_dir).as_posix()
        records.append(
            ExportRecord(
                key=key,
                image_path=image_path,
                convert_path=convert_path,
                elements=elements,
                status=status,
                pdf_path=pdf_path,
                page_index=page_index,
            )
        )
        stats["exported"] += 1

    records.sort(key=lambda r: r.key)
    return records, stats


def write_jsonl(
    records: list[ExportRecord],
    jsonl_path: Path,
    *,
    output_root: Path,
    copy_images_dir: Path | None,
    target: int | None,
) -> int:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with jsonl_path.open("w", encoding="utf-8") as fout:
        for rec in records:
            if target is not None and written >= target:
                break

            rel_image = rec.image_path.relative_to(output_root)
            image_out = rel_image.as_posix()
            if copy_images_dir is not None:
                dst = copy_images_dir / rel_image
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    shutil.copy2(rec.image_path, dst)
                image_out = dst.as_posix()

            row = {
                "id": rec.key,
                "image_path": image_out,
                "layout_elements": rec.elements,
                "layout_json": json.dumps(rec.elements, ensure_ascii=False),
                "status": rec.status,
                "pdf_path": rec.pdf_path,
                "page_index": rec.page_index,
                "convert_path": rec.convert_path.as_posix(),
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Export layout convert.json dataset")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="labeler output_dir (e.g. output/layout_batch)",
    )
    parser.add_argument(
        "--out-jsonl",
        type=Path,
        required=True,
        help="Destination JSONL path",
    )
    parser.add_argument(
        "--copy-images-dir",
        type=Path,
        default=None,
        help="Optional flat/tree mirror for page PNGs referenced in JSONL",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=None,
        help="Stop after N exported rows (e.g. 10000)",
    )
    parser.add_argument(
        "--min-elements",
        type=int,
        default=1,
        help="Skip pages with fewer layout elements",
    )
    parser.add_argument(
        "--include-error",
        action="store_true",
        help="Include pages whose final_result status is ERROR",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    records, stats = collect_records(
        args.output_dir,
        exclude_error=not args.include_error,
        min_elements=args.min_elements,
    )
    written = write_jsonl(
        records,
        args.out_jsonl,
        output_root=args.output_dir,
        copy_images_dir=args.copy_images_dir,
        target=args.target,
    )

    summary = {
        **stats,
        "jsonl_written": written,
        "out_jsonl": str(args.out_jsonl),
    }
    summary_path = args.out_jsonl.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
