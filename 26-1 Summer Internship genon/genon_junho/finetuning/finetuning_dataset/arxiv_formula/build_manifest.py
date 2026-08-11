#!/usr/bin/env python3
"""ArxivFormula 9,100장을 dots/paddle 러너가 먹는 manifest.jsonl 로 만든다.

두 러너(`../dotsocr_infer/run_infer.py`, `../paddlevl16_infer/run_infer.py`)는
`{key, split, line_no, image_path, n_tables}` 를 요구한다. ArxivFormula 에는 라벨이
전혀 없으므로 표 관련 필드는 `build_formula_manifest.py` 와 같이 스텁으로 둔다.

split 은 디렉터리 이름에서 그대로 나온다:
    Training_set_images_*        -> train
    Validation_set_images/.../val -> valid
    Testing_set_images           -> test

사용:
    python3 build_manifest.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/home/jhyeo/ocr_filter_result/additional_dataset/55600dataextract/ArxivFormula")
OUT = Path(__file__).parent / "manifest.jsonl"


def split_of(rel: Path) -> str:
    top = rel.parts[0]
    if top.startswith("Training_set_images"):
        return "train"
    if top.startswith("Validation_set_images"):
        return "valid"
    if top.startswith("Testing_set_images"):
        return "test"
    raise ValueError(f"알 수 없는 디렉터리: {top}")


def main() -> int:
    imgs = sorted(ROOT.rglob("*.jpg"))
    if not imgs:
        raise SystemExit(f"이미지가 없다: {ROOT}")

    counts: dict[str, int] = {}
    with OUT.open("w", encoding="utf-8") as f:
        for i, p in enumerate(imgs):
            rel = p.relative_to(ROOT)
            sp = split_of(rel)
            counts[sp] = counts.get(sp, 0) + 1
            rec = {
                "key": str(rel.with_suffix("")),
                "split": sp,
                "line_no": i,
                "image_path": str(p),
                # 표 라벨이 없는 데이터셋이라 스텁. 러너는 gt_n_tables 로만 흘려보낸다.
                "n_tables": 0,
                "tables": [],
                "static_flags": [],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"{OUT}: {len(imgs)}장")
    for k in ("train", "valid", "test"):
        print(f"  {k}: {counts.get(k, 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
