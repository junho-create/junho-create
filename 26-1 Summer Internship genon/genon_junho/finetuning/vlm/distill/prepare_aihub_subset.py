"""
Extract a small image subset from AIHub Training zip files for distillation smoke tests.

Usage:
    python -m distill.prepare_aihub_subset \
        --training_dir ../aihub_table_data/Training \
        --output_dir ./data/distill/images_subset \
        --max_images 64
"""

import argparse
import json
import os
import shutil
import zipfile
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def _is_image_member(name: str) -> bool:
    p = Path(name)
    if p.name.startswith("."):
        return False
    if "__MACOSX" in p.parts:
        return False
    return p.suffix.lower() in IMAGE_EXTENSIONS


def extract_subset(
    training_dir: str,
    output_dir: str,
    max_images: int = 64,
) -> dict:
    """
    Extract up to max_images image files from Training/01.원천데이터/*.zip.
    """
    source_zip_dir = Path(training_dir) / "01.원천데이터"
    if not source_zip_dir.exists():
        raise FileNotFoundError(f"Source zip directory not found: {source_zip_dir}")

    zip_files = sorted(source_zip_dir.glob("*.zip"))
    if not zip_files:
        raise FileNotFoundError(f"No zip files found in: {source_zip_dir}")

    image_dir = Path(output_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = image_dir / "manifest.jsonl"

    existing_count = len([
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ])
    remaining = max(max_images - existing_count, 0)
    if remaining == 0:
        return {
            "output_dir": str(image_dir),
            "manifest": str(manifest_path),
            "scanned_zips": 0,
            "requested": max_images,
            "existing": existing_count,
            "extracted": 0,
            "skipped": 0,
            "final_total": existing_count,
        }

    extracted = 0
    skipped = 0
    scanned_zips = 0

    manifest_file = open(manifest_path, "a", encoding="utf-8")
    try:
        for zip_path in zip_files:
            if extracted >= remaining:
                break

            scanned_zips += 1
            with zipfile.ZipFile(zip_path, "r") as zf:
                members = [name for name in zf.namelist() if _is_image_member(name)]
                for member in members:
                    if extracted >= remaining:
                        break

                    src_name = Path(member).name
                    ext = Path(src_name).suffix.lower()
                    stem = Path(src_name).stem
                    output_name = f"{zip_path.stem}_{stem}{ext}"
                    output_path = image_dir / output_name

                    if output_path.exists():
                        skipped += 1
                        continue

                    with zf.open(member) as src, open(output_path, "wb") as dst:
                        shutil.copyfileobj(src, dst)

                    extracted += 1
                    manifest = {
                        "index": extracted,
                        "image_path": str(output_path),
                        "source_zip": str(zip_path),
                        "source_member": member,
                    }
                    manifest_file.write(json.dumps(manifest, ensure_ascii=False) + "\n")
    finally:
        manifest_file.close()

    return {
        "output_dir": str(image_dir),
        "manifest": str(manifest_path),
        "scanned_zips": scanned_zips,
        "requested": max_images,
        "existing": existing_count,
        "extracted": extracted,
        "skipped": skipped,
        "final_total": existing_count + extracted,
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare AIHub image subset for distillation")
    parser.add_argument(
        "--training_dir",
        default="../aihub_table_data/Training",
        help="AIHub Training directory (contains 01.원천데이터/*.zip)",
    )
    parser.add_argument(
        "--output_dir",
        default="./data/distill/images_subset",
        help="Output directory for extracted subset images",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=64,
        help="Maximum number of images to extract",
    )
    args = parser.parse_args()

    if args.max_images <= 0:
        raise ValueError("--max_images must be > 0")

    stats = extract_subset(
        training_dir=args.training_dir,
        output_dir=args.output_dir,
        max_images=args.max_images,
    )

    print("Subset preparation complete:")
    print(f"  Output dir:   {stats['output_dir']}")
    print(f"  Manifest:     {stats['manifest']}")
    print(f"  Requested:    {stats['requested']}")
    print(f"  Existing:     {stats['existing']}")
    print(f"  Scanned zips: {stats['scanned_zips']}")
    print(f"  Extracted:    {stats['extracted']}")
    print(f"  Skipped:      {stats['skipped']}")
    print(f"  Final total:  {stats['final_total']}")


if __name__ == "__main__":
    main()
