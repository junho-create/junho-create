#!/usr/bin/env python3
"""이미 빌드된 JSON 통합 데이터셋을 HTML 통합 데이터셋으로 변환한다.

`build_unified_dataset.py --output-format json` 으로 만든 데이터셋(이미지 복사 완료,
train/valid/test split 고정)을 그대로 재사용하여, 정답(gt_html)과 prompt_style 만
HTML 파이프라인용으로 바꾼 새 데이터셋을 생성한다.

장점
----
- 원본 레이아웃 이미지/소스가 없어도 변환 가능(이미지는 심볼릭 링크로 재사용).
- JSON/HTML 두 파이프라인의 train/valid/test 분할이 동일해져 공정 비교가 가능하다.

변환 규칙
---------
- gt_html(JSON element 배열) → HTML 문서 조각 (utils.html_unified)
  - 단일 Table element  → 표 HTML 그대로
  - layout element 배열 → reading order HTML(h1/h2/p/ul/li/table)
- prompt_style: unified_* → unified_html_*

사용 예
-------
    python -m data.convert_unified_json_to_html \
        --src-dir ./_train_data/unified_smoke \
        --out-dir ./_train_data/unified_html_smoke
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.html_unified import unified_json_target_to_html  # noqa: E402

_STYLE_MAP = {
    "unified_layout": "unified_html_layout",
    "unified_table_with_ocr": "unified_html_table_with_ocr",
    "unified_table_without_ocr": "unified_html_table_without_ocr",
}


def _convert_record(rec: dict) -> dict:
    out = dict(rec)
    out["gt_html"] = unified_json_target_to_html(rec.get("gt_html", ""))
    style = str(rec.get("prompt_style", "")).strip().lower()
    out["prompt_style"] = _STYLE_MAP.get(style, style or "unified_html_layout")
    out["output_format"] = "html"
    return out


def _convert_split(src: Path, dst: Path) -> int:
    n = 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            fout.write(json.dumps(_convert_record(rec), ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description="JSON 통합 데이터셋 → HTML 통합 데이터셋 변환")
    parser.add_argument("--src-dir", type=Path, required=True, help="JSON 통합 데이터셋 루트")
    parser.add_argument("--out-dir", type=Path, required=True, help="HTML 통합 데이터셋 출력 루트")
    parser.add_argument(
        "--link-images",
        action="store_true",
        default=True,
        help="원본 images/ 디렉토리를 심볼릭 링크로 재사용(기본 활성)",
    )
    args = parser.parse_args()

    src_dir = args.src_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {"src_dir": str(src_dir), "out_dir": str(out_dir), "output_format": "html", "splits": {}}
    for split in ("train", "valid", "test"):
        src = src_dir / "data" / f"{split}.jsonl"
        if not src.exists():
            print(f"  [warn] split not found: {src}")
            continue
        dst = out_dir / "data" / f"{split}.jsonl"
        n = _convert_split(src, dst)
        summary["splits"][split] = n
        print(f"  {split}: {n} records -> {dst}")

    # 이미지 재사용 (심볼릭 링크)
    src_images = src_dir / "images"
    out_images = out_dir / "images"
    if args.link_images and src_images.exists() and not out_images.exists():
        try:
            os.symlink(src_images, out_images)
            print(f"  images symlink: {out_images} -> {src_images}")
        except OSError as e:
            print(f"  [warn] images symlink 실패({e}); image_dir 를 원본으로 지정해 사용하세요.")

    with (out_dir / "build_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  summary: {out_dir / 'build_summary.json'}")


if __name__ == "__main__":
    main()
