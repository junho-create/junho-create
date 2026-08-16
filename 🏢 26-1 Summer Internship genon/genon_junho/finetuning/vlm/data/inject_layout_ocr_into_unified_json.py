#!/usr/bin/env python3
"""통합 JSON 데이터셋(unified_smoke)의 레이아웃 레코드에 ocr_info 를 주입한다.

배경
----
- JSON 파이프라인(unified_base)의 레이아웃 레코드에는 ocr_info 가 없어
  학습 시 OCR 프롬프트(chandra_with_ocr) 후보가 되지 못한다.
- HTML 파이프라인(unified_html_smoke)은 동일 레이아웃 이미지에 대해 이미
  ocr_info 를 보유한다(좌표는 0-1024 정규화, JSON GT bbox 와 동일 스케일).
- 두 데이터셋의 레이아웃 이미지 basename 은 1:1 로 일치하므로, html 데이터셋의
  ocr_info 를 basename 매칭으로 그대로 복사("복붙")한다.

동작
----
- src(unified_smoke)의 각 split 레코드를 읽어:
  * task_type 키 제거(통합 처리: 공통 프롬프트로 레이아웃 디텍션).
  * ocr_info 가 없으면 html 데이터셋 map(basename->ocr_info)에서 채운다.
    (table 레코드는 이미 ocr_info 보유 → 유지)
  * output_format="json" 명시.
  * gt_html / image_path / prompt_style / bbox_scale 유지.
- out(chandra_table_layout_divhtml_16886)에 동일 split 으로 기록한다.
  이미지는 복사하지 않고 src 의 images/ 를 재사용한다(config 의 image_dir 가 src 를 가리킴).

사용
----
    python data/inject_layout_ocr_into_unified_json.py \
        --src _train_data/unified_smoke \
        --html _train_data/unified_html_smoke \
        --out _train_data/chandra_table_layout_divhtml_16886
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _has_ocr(rec: dict) -> bool:
    oi = rec.get("ocr_info")
    return isinstance(oi, list) and len(oi) > 0


def build_ocr_map(html_root: Path) -> dict[str, list]:
    """html 데이터셋 전체 split 에서 {basename -> ocr_info} map 을 만든다."""
    mapping: dict[str, list] = {}
    for split in ("train", "valid", "test"):
        p = html_root / "data" / f"{split}.jsonl"
        if not p.exists():
            continue
        for rec in _iter_jsonl(p):
            if not _has_ocr(rec):
                continue
            base = os.path.basename(str(rec.get("image_path", "")))
            if base and base not in mapping:
                mapping[base] = rec["ocr_info"]
    return mapping


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="unified_smoke 루트")
    ap.add_argument("--html", type=Path, required=True, help="unified_html_smoke 루트")
    ap.add_argument("--out", type=Path, required=True, help="출력 데이터셋 루트")
    args = ap.parse_args()

    src = args.src.expanduser().resolve()
    html = args.html.expanduser().resolve()
    out = args.out.expanduser().resolve()
    out_data = out / "data"
    out_data.mkdir(parents=True, exist_ok=True)

    ocr_map = build_ocr_map(html)
    print(f"[map] html ocr entries: {len(ocr_map)}")

    summary = {}
    for split in ("train", "valid", "test"):
        src_path = src / "data" / f"{split}.jsonl"
        if not src_path.exists():
            print(f"[warn] missing split: {src_path}")
            continue

        total = 0
        already = 0
        filled = 0
        missing = 0
        out_path = out_data / f"{split}.jsonl"
        with out_path.open("w", encoding="utf-8") as fo:
            for rec in _iter_jsonl(src_path):
                total += 1
                rec.pop("task_type", None)
                rec["output_format"] = "json"
                if _has_ocr(rec):
                    already += 1
                else:
                    base = os.path.basename(str(rec.get("image_path", "")))
                    oi = ocr_map.get(base)
                    if oi:
                        rec["ocr_info"] = oi
                        filled += 1
                    else:
                        missing += 1
                if "bbox_scale" not in rec:
                    rec["bbox_scale"] = 1024
                fo.write(json.dumps(rec, ensure_ascii=False) + "\n")

        summary[split] = {
            "total": total,
            "ocr_already": already,
            "ocr_filled_from_html": filled,
            "ocr_still_missing": missing,
        }
        print(f"[{split}] {summary[split]}")

    (out / "inject_ocr_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
