"""
AIHub zip 압축을 공통 경로로 해제한다.

목적:
- train/distill 모두 같은 extraction 루트 사용
- 매번 다른 임시 경로에 풀지 않도록 경로 통일

Usage:
    python -m data.extract_aihub \
        --input ../aihub_table_data/Training \
        --output_dir ./data/extracted/aihub

    python -m data.extract_aihub \
        --input ../aihub_table_data/Training ../aihub_table_data/Validation \
        --output_dir ./data/extracted/aihub \
        --max_zips 1
"""

import argparse
import os
import zipfile
from pathlib import Path


def _iter_zips(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(path.glob("*.zip"))


def _map_label_zips(label_zip_dir: Path) -> dict[str, Path]:
    mapping = {}
    for z in _iter_zips(label_zip_dir):
        key = z.stem.replace("TL_", "").replace("VL_", "")
        mapping[key] = z
    return mapping


def _extract_zip(zip_path: Path, target_dir: Path):
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)


def _already_extracted(target_dir: Path) -> bool:
    return target_dir.exists() and any(target_dir.rglob("*"))


def extract_split(
    input_dir: str,
    output_dir: str,
    max_zips: int | None = None,
    skip_existing: bool = True,
) -> dict:
    src_root = Path(input_dir)
    split_name = src_root.name
    source_zip_dir = src_root / "01.원천데이터"
    label_zip_dir = src_root / "02.라벨링데이터"

    if not source_zip_dir.exists():
        raise FileNotFoundError(f"source zip dir not found: {source_zip_dir}")

    source_zips = _iter_zips(source_zip_dir)
    if not source_zips:
        raise FileNotFoundError(f"no source zip files found in: {source_zip_dir}")

    if max_zips is not None:
        source_zips = source_zips[:max_zips]

    label_zip_map = _map_label_zips(label_zip_dir)

    split_out = Path(output_dir) / split_name
    source_out = split_out / "01.원천데이터"
    label_out = split_out / "02.라벨링데이터"
    source_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)

    stats = {
        "split": split_name,
        "source_zips_total": len(source_zips),
        "source_zips_extracted": 0,
        "source_zips_skipped": 0,
        "label_zips_extracted": 0,
        "label_zips_skipped": 0,
        "source_out": str(source_out),
        "label_out": str(label_out),
    }

    for src_zip in source_zips:
        key = src_zip.stem.replace("TS_", "").replace("VS_", "")
        src_target = source_out / key

        if skip_existing and _already_extracted(src_target):
            stats["source_zips_skipped"] += 1
        else:
            _extract_zip(src_zip, src_target)
            stats["source_zips_extracted"] += 1

        label_zip = label_zip_map.get(key)
        if label_zip is None:
            continue

        label_target = label_out / key
        if skip_existing and _already_extracted(label_target):
            stats["label_zips_skipped"] += 1
        else:
            _extract_zip(label_zip, label_target)
            stats["label_zips_extracted"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(description="Extract AIHub zip files to a unified path")
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="AIHub split path(s), e.g. ../aihub_table_data/Training",
    )
    parser.add_argument(
        "--output_dir",
        default="./data/extracted/aihub",
        help="Unified extraction root",
    )
    parser.add_argument(
        "--max_zips",
        type=int,
        default=None,
        help="Limit number of source zip files per split (for smoke test)",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-extract even if target already has files",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_stats = []
    for split in args.input:
        stats = extract_split(
            input_dir=split,
            output_dir=args.output_dir,
            max_zips=args.max_zips,
            skip_existing=not args.no_skip_existing,
        )
        all_stats.append(stats)

    print("AIHub extraction complete:")
    print(f"  output_dir: {args.output_dir}")
    for st in all_stats:
        print(f"  split={st['split']}")
        print(f"    source: total={st['source_zips_total']} extracted={st['source_zips_extracted']} skipped={st['source_zips_skipped']}")
        print(f"    label:  extracted={st['label_zips_extracted']} skipped={st['label_zips_skipped']}")
        print(f"    source_out: {st['source_out']}")
        print(f"    label_out:  {st['label_out']}")


if __name__ == "__main__":
    main()
