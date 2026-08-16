#!/usr/bin/env python3
"""OCR vs NO-OCR 비교용 평가 manifest 한 쌍을 생성한다(동일 GT).

입력(통합 JSON test split, ocr_info 보유)을 받아:
- with_ocr : ocr_info 가 있으면 prompt_style=chandra_with_ocr, 없으면 chandra_no_ocr.
             (원본 prompt_style 이 'unified_layout' 등 레거시 alias 라 no_ocr 로
              정규화돼 OCR 이 무시되는 문제를 방지)
- no_ocr   : ocr_info 제거 + prompt_style=chandra_no_ocr 강제.

두 manifest 는 image_path/gt_html/bbox_scale/output_format 이 동일하므로
동일 GT 기준으로 OCR 유무 효과를 직접 비교할 수 있다.

사용:
    python data/make_ocr_compare_manifests.py \
        --in  _train_data/chandra_table_layout_divhtml_16886/data/test.jsonl \
        --out-dir eval_data/ocr_compare
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _has_ocr(rec: dict) -> bool:
    oi = rec.get("ocr_info")
    return isinstance(oi, list) and len(oi) > 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out-dir", dest="out_dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with_path = args.out_dir / "test_with_ocr.jsonl"
    no_path = args.out_dir / "test_no_ocr.jsonl"

    n = 0
    n_with = 0
    with args.inp.open("r", encoding="utf-8") as fi, \
         with_path.open("w", encoding="utf-8") as fw, \
         no_path.open("w", encoding="utf-8") as fn:
        for line in fi:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            n += 1

            # with_ocr
            w = dict(rec)
            if _has_ocr(w):
                w["prompt_style"] = "chandra_with_ocr"
                n_with += 1
            else:
                w["prompt_style"] = "chandra_no_ocr"
            fw.write(json.dumps(w, ensure_ascii=False) + "\n")

            # no_ocr
            no = dict(rec)
            no.pop("ocr_info", None)
            no["prompt_style"] = "chandra_no_ocr"
            fn.write(json.dumps(no, ensure_ascii=False) + "\n")

    print(f"[with_ocr] {with_path} (n={n}, with_ocr_prompt={n_with})")
    print(f"[no_ocr]   {no_path} (n={n})")


if __name__ == "__main__":
    main()
