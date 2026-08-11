#!/usr/bin/env python3
"""e20 학습용 데이터셋 빌드: bbox 좌표를 0~1000 정규화 → 원본 이미지 픽셀로 변환.

e19 의 combined/{train,valid,test}.jsonl 의 split 멤버십을 그대로 유지하되,
각 레코드의 bbox 를 이미지 실제 픽셀 좌표로 바꾼다.

- reference_16886 출신 레코드: gt_html/ocr_info 의 bbox 가 0~1000 정규화 상태.
  이미지를 열어 (W,H) 를 얻고 픽셀로 역변환한다. (x_px = round(x_norm/1000*W))
- new_63541 출신 레코드: 이미 픽셀로 역변환된 gt_html 을 담은
  combined/new_63541_gt_html_pixelscale.jsonl 로 gt_html + (W,H) 를 교체한다.
  (pixelscale 파일은 ocr_info 가 비어 있으므로 ocr_info 는 원본 combined 에서
   가져와 동일 (W,H) 로 픽셀 변환한다.)

출력 스키마(9-key, pixelscale 파일과 동일):
    image_path, gt_html, prompt_style, bbox_scale(=None), bbox_space("image_px"),
    image_width, image_height, output_format, ocr_info

사용:
    cd /home/jhyeo/finetuning/finetuning_dataset
    python build_e20_pixelscale.py
    python build_e20_pixelscale.py --verify-only   # 이미 만든 산출물 검증만
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
COMBINED = HERE / "combined"
PIXELSCALE_FILE = COMBINED / "new_63541_gt_html_pixelscale.jsonl"
OUT_DIR = HERE / "combined_e20"

SPLITS = ["train", "valid", "test"]

# new_63541(자동 레이블) 품질 필터: 순수 텍스트가 이 길이 이하면 드롭.
# reference_16886(사람 GT)에는 적용하지 않는다.
NEW63541_MIN_TEXT_LEN = 50  # len(text) <= 50 이면 드롭

# data-bbox="x0 y0 x1 y1" (공백 구분 정수) 캡처.
_BBOX_RE = re.compile(r'data-bbox="(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)"')
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def gt_text_len(gt_html: str) -> int:
    """gt_html 에서 태그를 모두 제거한 순수 텍스트 길이."""
    t = _TAG_RE.sub(" ", gt_html or "")
    return len(_WS_RE.sub(" ", t).strip())


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _norm_to_px(x: int, y: int, w: int, h: int, scale: int = 1000) -> tuple[int, int]:
    return (
        _clamp(round(x * w / scale), 0, w),
        _clamp(round(y * h / scale), 0, h),
    )


def convert_gt_html_norm_to_px(gt_html: str, w: int, h: int, scale: int = 1000) -> str:
    """gt_html 내 모든 data-bbox 를 0~scale 정규화 → 픽셀로 변환."""
    def repl(m: re.Match) -> str:
        x0, y0, x1, y1 = (int(m.group(i)) for i in range(1, 5))
        px0, py0 = _norm_to_px(x0, y0, w, h, scale)
        px1, py1 = _norm_to_px(x1, y1, w, h, scale)
        return f'data-bbox="{px0} {py0} {px1} {py1}"'

    return _BBOX_RE.sub(repl, gt_html)


def convert_ocr_info_norm_to_px(ocr_info, w: int, h: int, scale: int = 1000):
    """ocr_info(list[dict{text,bbox[4]}]) 의 bbox 를 픽셀로 변환. 원본 스키마 보존."""
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
            px0, py0 = _norm_to_px(x0, y0, w, h, scale)
            px1, py1 = _norm_to_px(x1, y1, w, h, scale)
            new_item["bbox"] = [px0, py0, px1, py1]
        out.append(new_item)
    return out


def pixelscale_key(p: str) -> str:
    p = p.replace("\\", "/")
    i = p.find("ocr_filter_result")
    return p[i:] if i >= 0 else Path(p).name


def load_pixelscale_index() -> dict:
    idx = {}
    with open(PIXELSCALE_FILE, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            idx[pixelscale_key(r["image_path"])] = r
    return idx


def _get_image_size(path: str, cache: dict) -> tuple[int, int] | None:
    if path in cache:
        return cache[path]
    try:
        with Image.open(path) as im:
            wh = im.size  # (W, H), header-only lazy read
        cache[path] = wh
        return wh
    except Exception as e:  # noqa: BLE001
        print(f"[warn] image open failed: {path} ({e})", file=sys.stderr)
        cache[path] = None
        return None


def build():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ps_index = load_pixelscale_index()
    print(f"pixelscale indexed: {len(ps_index)}")

    size_cache: dict = {}
    stats = {
        "reference": 0, "new_63541": 0,
        "ref_img_fail": 0, "new_missing_pixelscale": 0, "other_skip": 0,
        "new_dropped_short_text": 0,
    }

    for split in SPLITS:
        src = COMBINED / f"{split}.jsonl"
        dst = OUT_DIR / f"{split}.jsonl"
        n_in = n_out = 0
        with open(src, encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
            for line in fin:
                if not line.strip():
                    continue
                n_in += 1
                r = json.loads(line)
                p = r["image_path"]

                if "reference_16886" in p:
                    wh = _get_image_size(p, size_cache)
                    if wh is None:
                        stats["ref_img_fail"] += 1
                        continue
                    w, h = wh
                    out = {
                        "image_path": p,
                        "gt_html": convert_gt_html_norm_to_px(r["gt_html"], w, h),
                        "prompt_style": r.get("prompt_style", "unified_layout"),
                        "bbox_scale": None,
                        "bbox_space": "image_px",
                        "image_width": int(w),
                        "image_height": int(h),
                        "output_format": r.get("output_format", "html"),
                        "ocr_info": convert_ocr_info_norm_to_px(r.get("ocr_info", []), w, h),
                    }
                    stats["reference"] += 1

                elif "ocr_filter_result" in p:
                    ps = ps_index.get(pixelscale_key(p))
                    if ps is None:
                        stats["new_missing_pixelscale"] += 1
                        continue
                    # 품질 필터: 순수 텍스트가 너무 짧은 자동 레이블은 드롭.
                    if gt_text_len(ps.get("gt_html", "")) <= NEW63541_MIN_TEXT_LEN:
                        stats["new_dropped_short_text"] += 1
                        continue
                    w = int(ps["image_width"])
                    h = int(ps["image_height"])
                    out = {
                        "image_path": p,  # combined 의 절대경로 유지(학습 시 해석 가능)
                        "gt_html": ps["gt_html"],  # 이미 픽셀
                        "prompt_style": r.get("prompt_style", "unified_layout"),
                        "bbox_scale": None,
                        "bbox_space": "image_px",
                        "image_width": w,
                        "image_height": h,
                        "output_format": r.get("output_format", "html"),
                        # pixelscale ocr_info 는 비어 있으므로 combined(0~1000) 에서 변환
                        "ocr_info": convert_ocr_info_norm_to_px(r.get("ocr_info", []), w, h),
                    }
                    stats["new_63541"] += 1
                else:
                    stats["other_skip"] += 1
                    continue

                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                n_out += 1
        print(f"[{split}] in={n_in} out={n_out} -> {dst}")

    print("stats:", stats)
    return OUT_DIR


def verify():
    """전 레코드 bbox 가 이미지 범위 내인지 assert + 요약."""
    import random
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
                w, h = int(r["image_width"]), int(r["image_height"])
                for m in _BBOX_RE.finditer(r["gt_html"]):
                    x0, y0, x1, y1 = (int(m.group(i)) for i in range(1, 5))
                    if not (0 <= x0 <= w and 0 <= x1 <= w and 0 <= y0 <= h and 0 <= y1 <= h):
                        bad += 1
                        if len(examples) < 5:
                            examples.append((r["image_path"], (x0, y0, x1, y1), (w, h)))
                        break
                assert r["bbox_space"] == "image_px"
    print(f"[verify] records={total} out_of_range_records={bad}")
    if examples:
        print("[verify] out-of-range examples:")
        for e in examples:
            print("   ", e)
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
