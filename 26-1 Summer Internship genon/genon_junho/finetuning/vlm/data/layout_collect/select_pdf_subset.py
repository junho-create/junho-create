#!/usr/bin/env python3
"""layout_data에서 목표 페이지 수만큼 PDF만 선별합니다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _iter_pdfs(root: Path) -> list[Path]:
    seen: set[str] = set()
    pdfs: list[Path] = []
    for pattern in ("*.pdf", "*.PDF"):
        for path in sorted(root.rglob(pattern)):
            key = str(path.resolve()).lower()
            if key in seen:
                continue
            seen.add(key)
            pdfs.append(path)
    return pdfs


def _page_count(pdf_path: Path) -> int:
    import fitz

    doc = fitz.open(str(pdf_path))
    try:
        return doc.page_count
    finally:
        doc.close()


def select_subset(
    root: Path,
    *,
    target_pages: int,
    buffer_ratio: float,
    max_rel_path_len: int = 240,
) -> tuple[list[dict], int]:
    goal = max(1, int(target_pages * buffer_ratio))
    selected: list[dict] = []
    total_pages = 0
    skipped_long_path = 0

    for pdf_path in _iter_pdfs(root):
        rel = pdf_path.relative_to(root).as_posix()
        if len(rel) > max_rel_path_len:
            skipped_long_path += 1
            continue

        try:
            pages = _page_count(pdf_path)
        except Exception as exc:
            print(f"SKIP (unreadable): {pdf_path} ({exc})")
            continue
        if pages <= 0:
            continue

        selected.append(
            {
                "relative_path": rel,
                "absolute_path": str(pdf_path.resolve()),
                "pages": pages,
            }
        )
        total_pages += pages
        if total_pages >= goal:
            break

    if skipped_long_path:
        print(f"SKIP (path too long): {skipped_long_path} PDFs")

    return selected, total_pages


def main() -> None:
    parser = argparse.ArgumentParser(description="Select PDFs up to target page count")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/Users/jae_hyeok_shin/tsr_test/layout_data"),
    )
    parser.add_argument("--target-pages", type=int, default=10000)
    parser.add_argument(
        "--buffer-ratio",
        type=float,
        default=1.05,
        help="ERROR/빈 페이지 대비 여유 (기본 5%%)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).with_name("pdf_subset.manifest.txt"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(__file__).with_name("pdf_subset.summary.json"),
    )
    args = parser.parse_args()

    root = args.root.resolve()
    selected, total_pages = select_subset(
        root,
        target_pages=args.target_pages,
        buffer_ratio=args.buffer_ratio,
    )

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        "\n".join(item["relative_path"] for item in selected) + "\n",
        encoding="utf-8",
    )

    summary = {
        "root": str(root),
        "target_pages": args.target_pages,
        "buffer_ratio": args.buffer_ratio,
        "goal_pages": int(args.target_pages * args.buffer_ratio),
        "selected_pdf_count": len(selected),
        "selected_page_count": total_pages,
        "manifest": str(args.manifest.resolve()),
        "pdfs": selected,
    }
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps({k: v for k, v in summary.items() if k != "pdfs"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
