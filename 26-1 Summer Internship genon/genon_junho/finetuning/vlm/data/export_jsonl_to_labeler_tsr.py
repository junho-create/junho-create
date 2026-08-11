#!/usr/bin/env python3
"""JSONL(표) → labeler TSR task용 디렉터리 트리 export.

기존 ``_train_data`` 는 건드리지 않고 labeler TSR 입력 트리를 만든다.

출력 구조::

    labeler_ready_table/
    ├── input/{stem}.jpg              ← collect_inputs (표 crop 이미지)
    ├── output/{stem}/attempt_1/generated.html
    ├── gt/tsr/{stem}.html            ← gt_replay 용 (선택)
    └── meta/manifest.jsonl

evaluate_only 에는 ``output/{stem}/attempt_1/screenshot.png`` 가 추가로 필요하다.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_TABLE_RE = re.compile(r"<table[\s\S]*?</table>", re.IGNORECASE)
_DIV_TABLE_RE = re.compile(
    r'<div\b[^>]*\bdata-label="Table"[^>]*>([\s\S]*?)</div>',
    re.IGNORECASE,
)


@dataclass
class ExportStats:
    rows_read: int = 0
    exported: int = 0
    skipped_no_image: int = 0
    skipped_no_html: int = 0
    skipped_duplicate: int = 0
    errors: list[str] = field(default_factory=list)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} JSON parse error: {exc}") from exc
    return rows


def _read_jsonl_dir(jsonl_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(jsonl_dir.glob("*.jsonl")):
        rows.extend(_read_jsonl(path))
    return rows


def _resolve_image(image_path: str, roots: list[Path]) -> Optional[Path]:
    raw = Path(image_path)
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    for root in roots:
        candidates.extend(
            [
                root / raw,
                root / raw.name,
                root / "images" / raw.name,
                root / "images" / "table" / raw.name,
            ]
        )
        if raw.parts and raw.parts[0] == "images":
            candidates.append(root / Path(*raw.parts))
            if len(raw.parts) > 1:
                candidates.append(root / Path(*raw.parts[1:]))
    candidates.append(raw)
    seen: set[Path] = set()
    for c in candidates:
        c = c.expanduser()
        if c in seen:
            continue
        seen.add(c)
        if c.exists():
            return c.resolve()
    return None


def _extract_table_html(gt_html: str) -> Optional[str]:
    text = str(gt_html or "").strip()
    if not text:
        return None
    m = _TABLE_RE.search(text)
    if m:
        return m.group(0).strip()
    m = _DIV_TABLE_RE.search(text)
    if m:
        inner = m.group(1).strip()
        m2 = _TABLE_RE.search(inner)
        return (m2.group(0).strip() if m2 else inner) or None
    return None


def _link_or_copy(src: Path, dst: Path, *, use_symlink: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if use_symlink:
        dst.symlink_to(src)
    else:
        shutil.copy2(src, dst)


def _write_readme(out_dir: Path, *, source: str, exported: int) -> None:
    text = f"""# labeler_ready_table

``_train_data`` 원본은 그대로 두고, **labeler TSR(표) 입력용** 트리입니다.

- Source: ``{source}``
- Exported tables: **{exported}**

## labeler config 경로

```yaml
task: tsr
input_dir: "{out_dir.as_posix()}/input"
output_dir: "{out_dir.as_posix()}/output"
gt_replay:
  gt_dir: "{out_dir.as_posix()}/gt/tsr"
```

## 페이지(표) 1장 구조

```text
input/{{stem}}.jpg
output/{{stem}}/attempt_1/generated.html
gt/tsr/{{stem}}.html          # gt_replay 용 (선택)
```

evaluate_only 는 ``output/{{stem}}/attempt_1/screenshot.png`` 가 추가로 필요합니다.

## 재생성

```bash
cd /NHNHOME/WORKSPACE/0426030039_A/jhshin/tsr_test/train/vlm
OVERWRITE=1 bash scripts/export_labeler_table.sh
```
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def export_jsonl_to_labeler_tsr(
    *,
    rows: list[dict[str, Any]],
    image_roots: list[Path],
    out_dir: Path,
    max_samples: Optional[int] = None,
    use_symlink: bool = True,
    overwrite: bool = False,
    write_gt: bool = True,
) -> ExportStats:
    stats = ExportStats()
    stats.rows_read = len(rows)

    if overwrite and out_dir.exists():
        shutil.rmtree(out_dir)

    input_dir = out_dir / "input"
    output_dir = out_dir / "output"
    gt_dir = out_dir / "gt" / "tsr"
    meta_dir = out_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []
    seen_stems: set[str] = set()

    for rec in rows:
        if max_samples is not None and stats.exported >= max_samples:
            break

        image_field = rec.get("image_path")
        if not isinstance(image_field, str) or not image_field.strip():
            stats.skipped_no_image += 1
            continue

        image_path = _resolve_image(image_field, image_roots)
        if image_path is None:
            stats.skipped_no_image += 1
            continue

        stem = image_path.stem
        if stem in seen_stems:
            stats.skipped_duplicate += 1
            continue
        seen_stems.add(stem)

        table_html = _extract_table_html(str(rec.get("gt_html") or rec.get("original_gt_html") or ""))
        if not table_html:
            stats.skipped_no_html += 1
            continue

        input_image = input_dir / image_path.name
        _link_or_copy(image_path, input_image, use_symlink=use_symlink)

        attempt_dir = output_dir / stem / "attempt_1"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        generated_path = attempt_dir / "generated.html"
        generated_path.write_text(table_html, encoding="utf-8")

        gt_path: Optional[Path] = None
        if write_gt:
            gt_path = gt_dir / f"{stem}.html"
            gt_path.parent.mkdir(parents=True, exist_ok=True)
            gt_path.write_text(table_html, encoding="utf-8")

        manifest_rows.append(
            {
                "stem": stem,
                "output_key": stem,
                "source_image": str(image_path),
                "input_image": str(input_image),
                "generated_html": str(generated_path),
                "gt_html": str(gt_path) if gt_path else None,
                "split_hint": rec.get("split"),
            }
        )
        stats.exported += 1

        if stats.exported % 500 == 0:
            logger.info("exported %d ...", stats.exported)

    manifest_path = meta_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "out_dir": str(out_dir.resolve()),
        "rows_read": stats.rows_read,
        "exported": stats.exported,
        "skipped_no_image": stats.skipped_no_image,
        "skipped_no_html": stats.skipped_no_html,
        "skipped_duplicate": stats.skipped_duplicate,
        "errors": stats.errors[:50],
        "error_count": len(stats.errors),
    }
    (meta_dir / "export_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_readme(out_dir, source=str(image_roots[0]), exported=stats.exported)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Export table JSONL to labeler TSR tree")
    parser.add_argument("--train-data-root", type=Path, default=Path("_train_data"))
    parser.add_argument("--jsonl", type=Path, default=None)
    parser.add_argument(
        "--jsonl-dir",
        type=Path,
        default=None,
        help="여러 split jsonl 디렉터리 (예: table_src_6902/data)",
    )
    parser.add_argument("--image-root", type=Path, action="append", default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--copy-images", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-gt", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    train_root = args.train_data_root.expanduser().resolve()
    if args.jsonl_dir:
        rows = _read_jsonl_dir(args.jsonl_dir.expanduser().resolve())
        source = str(args.jsonl_dir)
    elif args.jsonl:
        rows = _read_jsonl(args.jsonl.expanduser().resolve())
        source = str(args.jsonl)
    else:
        jsonl_dir = train_root / "table_src_6902" / "data"
        rows = _read_jsonl_dir(jsonl_dir)
        source = str(jsonl_dir)

    image_roots = (
        [p.expanduser().resolve() for p in args.image_root]
        if args.image_root
        else [
            train_root / "table_src_6902",
            train_root / "chandra_table_layout_divhtml_16886",
            train_root,
        ]
    )
    out_dir = (
        args.out_dir.expanduser().resolve()
        if args.out_dir
        else train_root / "labeler_ready_table"
    )

    logger.info("source     : %s", source)
    logger.info("rows       : %d", len(rows))
    logger.info("image_roots: %s", ", ".join(str(p) for p in image_roots))
    logger.info("out_dir    : %s", out_dir)

    stats = export_jsonl_to_labeler_tsr(
        rows=rows,
        image_roots=image_roots,
        out_dir=out_dir,
        max_samples=args.max_samples,
        use_symlink=not args.copy_images,
        overwrite=args.overwrite,
        write_gt=not args.no_gt,
    )

    print(
        json.dumps(
            {
                "rows_read": stats.rows_read,
                "exported": stats.exported,
                "skipped_no_image": stats.skipped_no_image,
                "skipped_no_html": stats.skipped_no_html,
                "skipped_duplicate": stats.skipped_duplicate,
                "out_dir": str(out_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
