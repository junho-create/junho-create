"""
Split one JSONL file into fixed-size chunks.

Usage:
    python -m data.split_jsonl \
        --input /path/to/merged.jsonl \
        --output_dir /path/to/output \
        --chunk_size 500
"""

import argparse
import sys
from pathlib import Path


def build_chunk_path(output_dir: Path, prefix: str, part_index: int, pad_width: int) -> Path:
    return output_dir / f"{prefix}_part{part_index:0{pad_width}d}.jsonl"


def split_jsonl(
    *,
    input_path: Path,
    output_dir: Path,
    chunk_size: int,
    prefix: str,
    pad_width: int,
    keep_empty_lines: bool,
) -> dict:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be >= 1")
    if pad_width <= 0:
        raise ValueError("pad_width must be >= 1")

    output_dir.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    parts: list[tuple[Path, int]] = []
    part_index = 0
    current_rows = 0
    current_path: Path | None = None
    current_file = None

    def close_current_chunk() -> None:
        nonlocal current_file, current_rows, current_path
        if current_file is None or current_path is None:
            return
        current_file.close()
        parts.append((current_path, current_rows))
        current_file = None
        current_path = None
        current_rows = 0

    def open_next_chunk() -> None:
        nonlocal current_file, part_index, current_path, current_rows
        close_current_chunk()
        part_index += 1
        current_path = build_chunk_path(output_dir, prefix, part_index, pad_width)
        current_file = current_path.open("w", encoding="utf-8")
        current_rows = 0

    try:
        with input_path.open("r", encoding="utf-8") as fin:
            for line in fin:
                if not keep_empty_lines and not line.strip():
                    continue

                if current_file is None or current_rows >= chunk_size:
                    open_next_chunk()

                if line.endswith("\n"):
                    current_file.write(line)
                else:
                    current_file.write(line + "\n")

                current_rows += 1
                total_rows += 1
    finally:
        close_current_chunk()

    return {
        "total_rows": total_rows,
        "chunks": parts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Split JSONL by fixed row count.")
    parser.add_argument("--input", required=True, help="Input JSONL file path")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory for chunk files (default: input file directory)",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=500,
        help="Rows per output chunk (default: 500)",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Output filename prefix (default: input stem)",
    )
    parser.add_argument(
        "--pad_width",
        type=int,
        default=3,
        help="Zero padding width for part index (default: 3)",
    )
    parser.add_argument(
        "--keep_empty_lines",
        action="store_true",
        help="Treat empty lines as rows instead of skipping them",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    if input_path.is_dir():
        print(f"Error: input path is a directory: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent
    prefix = args.prefix if args.prefix else input_path.stem

    result = split_jsonl(
        input_path=input_path,
        output_dir=output_dir,
        chunk_size=args.chunk_size,
        prefix=prefix,
        pad_width=args.pad_width,
        keep_empty_lines=args.keep_empty_lines,
    )

    print("=" * 50)
    print("JSONL split complete")
    print("=" * 50)
    print(f"input: {input_path}")
    print(f"output_dir: {output_dir}")
    print(f"chunk_size: {args.chunk_size}")
    print(f"total_rows: {result['total_rows']}")
    print(f"chunk_count: {len(result['chunks'])}")
    if result["chunks"]:
        for idx, (chunk_path, count) in enumerate(result["chunks"], start=1):
            print(f"  part{idx:0{args.pad_width}d}: {count} rows -> {chunk_path}")
    else:
        print("  no rows to write (input had no non-empty lines)")
    print("=" * 50)


if __name__ == "__main__":
    main()
