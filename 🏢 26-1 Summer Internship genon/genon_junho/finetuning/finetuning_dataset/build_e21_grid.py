#!/usr/bin/env python3
"""e21 학습용 데이터: combined_e20(픽셀 좌표, dots.ocr 라벨 교정 + 저품질 필터 적용됨)를
e19 방식(0~1000 grid 좌표)으로 되돌린다.

목적: e20에서 확인된 두 회귀 요인(픽셀 좌표계, bbox_weight loss)을 데이터 개선분과
분리하기 위함. combined_e20 레코드에 image_width/image_height가 이미 저장돼 있어
이미지를 다시 열 필요 없이 순수 산술 변환만 하면 된다(build_e20_pixelscale.py의 역연산).

출력 스키마는 e19(combined/*.jsonl)와 동일한 6-key: image_path, gt_html, prompt_style,
bbox_scale(=1000), output_format, ocr_info.

사용:
    cd /home/jhyeo/finetuning/finetuning_dataset
    python build_e21_grid.py
    python build_e21_grid.py --verify-only
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC_DIR = HERE / "combined_e20"
OUT_DIR = HERE / "combined_e21_grid"
SPLITS = ["train", "valid", "test"]
GRID = 1000

_BBOX_RE = re.compile(r'data-bbox="(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)"')


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def px_to_norm(x: int, y: int, w: int, h: int, scale: int = GRID) -> tuple[int, int]:
    return (
        _clamp(round(x * scale / w), 0, scale),
        _clamp(round(y * scale / h), 0, scale),
    )


def convert_gt_html_px_to_norm(gt_html: str, w: int, h: int) -> str:
    def repl(m: re.Match) -> str:
        x0, y0, x1, y1 = (int(m.group(i)) for i in range(1, 5))
        nx0, ny0 = px_to_norm(x0, y0, w, h)
        nx1, ny1 = px_to_norm(x1, y1, w, h)
        return f'data-bbox="{nx0} {ny0} {nx1} {ny1}"'

    return _BBOX_RE.sub(repl, gt_html)


def convert_ocr_info_px_to_norm(ocr_info, w: int, h: int):
    if not isinstance(ocr_info, list) or not ocr_info:
        return ocr_info
    out = []
    for item in ocr_info:
        if not isinstance(item, dict):
            out.append(item)
            continue
        new_item = dict(item)
        bbox = item.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            x0, y0, x1, y1 = (int(round(float(v))) for v in bbox[:4])
            nx0, ny0 = px_to_norm(x0, y0, w, h)
            nx1, ny1 = px_to_norm(x1, y1, w, h)
            new_item["bbox"] = [nx0, ny0, nx1, ny1]
        out.append(new_item)
    return out


def build():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stats = {"converted": 0, "missing_dims": 0}
    for split in SPLITS:
        src = SRC_DIR / f"{split}.jsonl"
        dst = OUT_DIR / f"{split}.jsonl"
        n_in = n_out = 0
        with open(src, encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
            for line in fin:
                if not line.strip():
                    continue
                n_in += 1
                r = json.loads(line)
                w = r.get("image_width")
                h = r.get("image_height")
                if not w or not h:
                    stats["missing_dims"] += 1
                    continue
                out = {
                    "image_path": r["image_path"],
                    "gt_html": convert_gt_html_px_to_norm(r["gt_html"], w, h),
                    "prompt_style": r.get("prompt_style", "unified_layout"),
                    "bbox_scale": GRID,
                    "output_format": r.get("output_format", "html"),
                    "ocr_info": convert_ocr_info_px_to_norm(r.get("ocr_info", []), w, h),
                }
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                n_out += 1
                stats["converted"] += 1
        print(f"[{split}] in={n_in} out={n_out} -> {dst}")
    print("stats:", stats)


def verify():
    total = bad = 0
    examples = []
    for split in SPLITS:
        path = OUT_DIR / f"{split}.jsonl"
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                total += 1
                r = json.loads(line)
                assert r["bbox_scale"] == GRID
                for m in _BBOX_RE.finditer(r["gt_html"]):
                    vals = [int(m.group(i)) for i in range(1, 5)]
                    if any(v < 0 or v > GRID for v in vals):
                        bad += 1
                        if len(examples) < 5:
                            examples.append((r["image_path"], vals))
                        break
    print(f"[verify] records={total} out_of_range_records={bad}")
    if examples:
        print("[verify] examples:", examples)
    return bad == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()
    if not args.verify_only:
        build()
    ok = verify()
    print("VERIFY", "OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
