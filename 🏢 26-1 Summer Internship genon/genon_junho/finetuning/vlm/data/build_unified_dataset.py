#!/usr/bin/env python3
"""테이블 + 레이아웃 데이터를 동일한 JSON 출력 스키마로 통합하는 빌드 스크립트.

목적
----
TSR VLM 을 테이블 데이터와 레이아웃 데이터로 함께 학습할 때, 두 경우 모두
모델의 추출 결과가 동일한 JSON 형태로 나오도록 학습 데이터를 통일한다.

통합 정답(assistant target) 스키마
---------------------------------
각 샘플의 정답은 layout element 객체의 JSON 배열이다::

    [
      {"bbox": [x0, y0, x1, y1], "category": "<Label>", "text": "<content>"},
      ...
    ]

- bbox 는 0-bbox_scale 로 정규화된 정수 좌표.
- Table element 의 text 는 표 구조 HTML(<table>...</table>).
- 테이블 데이터(단일 표 이미지)는 category="Table" 단일 원소 배열로 표현된다.
- 레이아웃 데이터는 dots.mocr 가 추출한 모든 element 배열을 사용한다.

기존 학습 코드(`train.collator.MultimodalCollator`, `train.train_qlora.TSRDataset`)는
`gt_html` 필드를 "assistant 정답 텍스트"로 사용한다. 따라서 통합 정답 JSON 문자열을
`gt_html` 에 그대로 저장하여 기존 파이프라인을 최대한 재활용한다.

출력 레코드 스키마
-----------------
    {
      "image_path": "<abs or images/...>",
      "task_type": "table" | "layout",
      "gt_html": "<통합 정답 JSON 문자열>",
      "prompt_style": "unified_table_with_ocr" | "unified_table_without_ocr" | "unified_layout",
      "ocr_info": [...],     # table 이고 OCR 이 있을 때만
      "bbox_scale": 1024
    }

사용 예
-------
    python -m data.build_unified_dataset \
        --table-dir ./_train_data/table_src_6902 \
        --layout-jsonl /path/to/layout_train.jsonl \
        --layout-image-root /path/to/layout_images_root \
        --out-dir ./_train_data/unified_smoke \
        --bbox-scale 1024
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Optional

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.html_unified import layout_elements_to_html, table_element_to_html  # noqa: E402

# prompt style (통합 2종: 테이블/레이아웃 공통)
PROMPT_STYLE_WITH_OCR = "chandra_with_ocr"
PROMPT_STYLE_NO_OCR = "chandra_no_ocr"

TABLE_CATEGORY = "Table"

FORMAT_JSON = "json"
FORMAT_HTML = "html"


def _read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _resolve_image(image_path: str, root: Optional[Path]) -> Optional[Path]:
    raw = Path(image_path)
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    if root is not None:
        candidates.append(root / raw)
        candidates.append(root / raw.name)
        # table jsonl 은 image_path 가 'images/xxx.jpg' 형태이므로 그대로도 시도
        if raw.parts and raw.parts[0] == "images":
            candidates.append(root / Path(*raw.parts[1:]))
    candidates.append(raw)
    for c in candidates:
        if c.exists():
            return c.resolve()
    return None


def _image_size(path: Path) -> Optional[tuple[int, int]]:
    if Image is None:
        return None
    try:
        with Image.open(path) as im:
            return im.size  # (W, H)
    except Exception:
        return None


def _normalize_bbox(
    bbox: list[Any], width: int, height: int, bbox_scale: int
) -> Optional[list[int]]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    try:
        x0, y0, x1, y1 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    except (TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None

    def nx(v: float) -> int:
        return int(max(0, min(bbox_scale, round(v / width * bbox_scale))))

    def ny(v: float) -> int:
        return int(max(0, min(bbox_scale, round(v / height * bbox_scale))))

    rx0, ry0, rx1, ry1 = nx(x0), ny(y0), nx(x1), ny(y1)
    if rx1 < rx0:
        rx0, rx1 = rx1, rx0
    if ry1 < ry0:
        ry0, ry1 = ry1, ry0
    return [rx0, ry0, rx1, ry1]


def build_table_record(
    rec: dict,
    table_root: Path,
    bbox_scale: int,
    use_abs_path: bool,
    copy_dir: Optional[Path],
    output_format: str = FORMAT_JSON,
) -> Optional[dict]:
    gt_html = str(rec.get("gt_html") or "").strip()
    image_path = rec.get("image_path")
    if not gt_html or not isinstance(image_path, str) or not image_path.strip():
        return None

    resolved = _resolve_image(image_path, table_root)
    if resolved is None:
        return None

    ocr_info = rec.get("ocr_info")
    has_ocr = isinstance(ocr_info, list) and len(ocr_info) > 0

    out_image = _emit_image_path(resolved, copy_dir, use_abs_path, "table")

    if output_format == FORMAT_HTML:
        # HTML 파이프라인: 표 데이터는 이미 HTML 이므로 그대로 둔다(변형 불필요).
        target = table_element_to_html(gt_html)
        prompt_style = PROMPT_STYLE_WITH_OCR if has_ocr else PROMPT_STYLE_NO_OCR
    else:
        # JSON 파이프라인: 표 crop 은 이미지 전체가 표이므로 bbox 는 전체 영역
        elements = [
            {
                "bbox": [0, 0, bbox_scale, bbox_scale],
                "category": TABLE_CATEGORY,
                "text": gt_html,
            }
        ]
        target = json.dumps(elements, ensure_ascii=False)
        prompt_style = PROMPT_STYLE_WITH_OCR if has_ocr else PROMPT_STYLE_NO_OCR

    record = {
        "image_path": out_image,
        "task_type": "table",
        "gt_html": target,
        "prompt_style": prompt_style,
        "bbox_scale": int(rec.get("bbox_scale", bbox_scale) or bbox_scale),
    }
    if has_ocr:
        record["ocr_info"] = ocr_info
    return record


def build_layout_record(
    rec: dict,
    layout_root: Optional[Path],
    bbox_scale: int,
    use_abs_path: bool,
    copy_dir: Optional[Path],
    stats: dict,
    output_format: str = FORMAT_JSON,
) -> Optional[dict]:
    image_path = rec.get("image_path")
    if not isinstance(image_path, str) or not image_path.strip():
        return None
    elements_in = rec.get("layout_elements")
    if not isinstance(elements_in, list) or not elements_in:
        return None

    resolved = _resolve_image(image_path, layout_root)
    if resolved is None:
        stats["layout_image_missing"] += 1
        return None

    size = _image_size(resolved)
    if size is None:
        stats["layout_size_unknown"] += 1
        return None
    width, height = size

    elements_out = []
    for el in elements_in:
        if not isinstance(el, dict):
            continue
        category = str(el.get("category") or el.get("label") or "Text").strip() or "Text"
        text = str(el.get("text") or "")
        norm = _normalize_bbox(el.get("bbox", []), width, height, bbox_scale)
        if norm is None:
            norm = [0, 0, bbox_scale, bbox_scale]
        elements_out.append({"bbox": norm, "category": category, "text": text})

    if not elements_out:
        return None

    out_image = _emit_image_path(resolved, copy_dir, use_abs_path, "layout")

    # 레이아웃은 OCR 정보 유무로 prompt_style 을 정한다(없으면 이미지 전용).
    layout_ocr = rec.get("ocr_info")
    layout_has_ocr = isinstance(layout_ocr, list) and len(layout_ocr) > 0
    if output_format == FORMAT_HTML:
        # HTML 파이프라인: element 배열을 reading order HTML 문서 조각으로 변환
        target = layout_elements_to_html(elements_out)
    else:
        target = json.dumps(elements_out, ensure_ascii=False)
    prompt_style = PROMPT_STYLE_WITH_OCR if layout_has_ocr else PROMPT_STYLE_NO_OCR

    return {
        "image_path": out_image,
        "task_type": "layout",
        "gt_html": target,
        "prompt_style": prompt_style,
        "bbox_scale": int(bbox_scale),
    }


def _emit_image_path(
    resolved: Path,
    copy_dir: Optional[Path],
    use_abs_path: bool,
    subdir: str,
) -> str:
    if copy_dir is not None:
        dst_dir = copy_dir / subdir
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / resolved.name
        if not dst.exists():
            shutil.copy2(resolved, dst)
        # data/ 기준 상대경로 (images/<subdir>/<name>)
        return f"images/{subdir}/{resolved.name}"
    if use_abs_path:
        return str(resolved)
    return str(resolved)


def _write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build unified table+layout dataset")
    parser.add_argument(
        "--table-dir",
        type=Path,
        required=True,
        help="테이블 데이터 루트 (data/train.jsonl 등과 images/ 포함)",
    )
    parser.add_argument(
        "--layout-jsonl",
        type=Path,
        required=True,
        help="레이아웃 수집 JSONL (layout_elements 포함)",
    )
    parser.add_argument(
        "--layout-image-root",
        type=Path,
        default=None,
        help="레이아웃 JSONL 의 상대 image_path 해석용 루트",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bbox-scale", type=int, default=1024)
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="이미지를 out-dir/images/ 로 복사 (기본: 원본 절대경로 참조)",
    )
    parser.add_argument(
        "--layout-valid-ratio", type=float, default=0.1,
        help="레이아웃 데이터 valid 분할 비율",
    )
    parser.add_argument(
        "--layout-test-ratio", type=float, default=0.1,
        help="레이아웃 데이터 test 분할 비율",
    )
    parser.add_argument(
        "--max-table-per-split",
        type=int,
        default=None,
        help="(옵션) split 별 테이블 샘플 상한 (빠른 스모크용)",
    )
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument(
        "--output-format",
        choices=[FORMAT_JSON, FORMAT_HTML],
        default=FORMAT_JSON,
        help="정답/프롬프트 형식. json(기존 element 배열) 또는 html(통합 HTML 문서).",
    )
    args = parser.parse_args()

    output_format = args.output_format
    table_dir = args.table_dir.expanduser().resolve()
    table_data_dir = table_dir / "data"
    bbox_scale = int(args.bbox_scale)
    copy_dir = (args.out_dir / "images") if args.copy_images else None
    print(f"  output_format: {output_format}")

    stats = {
        "layout_image_missing": 0,
        "layout_size_unknown": 0,
    }

    # --- 레이아웃: split 이 없으므로 직접 분할 ---
    layout_records_raw = _read_jsonl(args.layout_jsonl.expanduser().resolve())
    layout_built = []
    for rec in layout_records_raw:
        built = build_layout_record(
            rec,
            args.layout_image_root.expanduser().resolve() if args.layout_image_root else None,
            bbox_scale,
            use_abs_path=not args.copy_images,
            copy_dir=copy_dir,
            stats=stats,
            output_format=output_format,
        )
        if built is not None:
            layout_built.append(built)

    rng = random.Random(args.seed)
    rng.shuffle(layout_built)
    n_layout = len(layout_built)
    n_val = int(round(n_layout * args.layout_valid_ratio))
    n_test = int(round(n_layout * args.layout_test_ratio))
    n_val = min(n_val, max(0, n_layout - 1))
    n_test = min(n_test, max(0, n_layout - n_val - 1))
    layout_split = {
        "valid": layout_built[:n_val],
        "test": layout_built[n_val:n_val + n_test],
        "train": layout_built[n_val + n_test:],
    }

    # --- 테이블: 기존 train/valid/test split 사용 ---
    table_split = {"train": [], "valid": [], "test": []}
    for split in ("train", "valid", "test"):
        src = table_data_dir / f"{split}.jsonl"
        if not src.exists():
            print(f"  [warn] table split not found: {src}")
            continue
        recs = _read_jsonl(src)
        if args.max_table_per_split is not None:
            recs = recs[: args.max_table_per_split]
        for rec in recs:
            built = build_table_record(
                rec,
                table_dir,
                bbox_scale,
                use_abs_path=not args.copy_images,
                copy_dir=copy_dir,
                output_format=output_format,
            )
            if built is not None:
                table_split[split].append(built)

    # --- 병합 후 기록 ---
    out_data_dir = args.out_dir / "data"
    summary = {
        "out_dir": str(args.out_dir.resolve()),
        "output_format": output_format,
        "bbox_scale": bbox_scale,
        "copy_images": bool(args.copy_images),
        "layout_total": n_layout,
        "splits": {},
        "build_stats": stats,
    }
    for split in ("train", "valid", "test"):
        merged = list(table_split[split]) + list(layout_split[split])
        rng.shuffle(merged)
        _write_jsonl(merged, out_data_dir / f"{split}.jsonl")
        summary["splits"][split] = {
            "table": len(table_split[split]),
            "layout": len(layout_split[split]),
            "total": len(merged),
        }

    summary_path = args.out_dir / "build_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
