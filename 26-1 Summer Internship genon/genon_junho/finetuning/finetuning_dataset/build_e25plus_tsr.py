#!/usr/bin/env python3
"""e25_plus 이어학습용 데이터: combined_e25_28791 에서 TSR(표 크롭) 출신 레코드만
추출한다.

목적: e25 이후 checkpoint-3900 을 merge 한 모델 위에 저rank·저LR 로 이어학습하되,
TSR(prompt_style == "unified_table_with_ocr") 데이터만 사용한다. 결정적 이유는
combined_e24_table_refined 단계에서 unified_layout 행의 Table div 내용이 PaddleOCR-VL
1.6 추론 결과(pseudo-label)로 교체됐다는 점이다(finetuning_dataset/combined_e24_table_refined
/CHANGES.md 참고) — 레이아웃 행으로 표를 학습시키면 paddle 오류를 정답으로 배우게 된다.
TSR 크롭(reference_16886/images/table/*)만 사람이 라벨링한 정확한 GT 다.

두 가지 산출물을 만든다:
  1) combined_e25plus_tsr6598/  : TSR 원본 그대로(train 5,783 / valid 601 / test 214).
     A/B 두 실험 셀(LR 대조)이 이 데이터를 그대로 쓴다.
  2) combined_e25plus_tsrsynth5783/ : train 의 절반을 "정확한 TSR 크롭을 페이지 크기
     캔버스에 합성 배치"한 버전으로 교체한 것. GT 는 100% TSR 원본이고(pseudo-label
     아님), 이미지 조건만 "레이아웃 페이지 속 작은 표"에 가깝게 만든다. valid 는
     6598 산출물의 TSR 601행을 그대로 재사용(세 셀 간 eval_loss 비교 가능하게).
     캔버스 총 픽셀은 학습 max_pixels(1,568,000) 와 정확히 일치시켜, processor 의
     smart-resize 가 추가로 축소하지 않게 한다(표의 실효 픽셀이 면적비로 결정됨).

사용:
    cd /home/jhyeo/finetuning/finetuning_dataset
    python build_e25plus_tsr.py
    python build_e25plus_tsr.py --verify-only
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
SRC_DIR = HERE / "combined_e25_28791"
OUT_TSR_DIR = HERE / "combined_e25plus_tsr6598"
OUT_SYNTH_DIR = HERE / "combined_e25plus_tsrsynth5783"
SPLITS = ["train", "valid", "test"]

SYNTH_IMAGE_DIR = HERE.parent / "imagedata" / "e25plus_synth"

TSR_PROMPT_STYLE = "unified_table_with_ocr"
LAYOUT_PROMPT_STYLE = "unified_layout"

SEED = 20260803
CANVAS_TOTAL_PX = 1_568_000       # 학습 max_pixels(cell C) 와 반드시 일치
CANVAS_ASPECT_WH = 0.735          # 레이아웃 페이지 실측 w/h 중앙값
AREA_FRAC_MIN = 0.10
AREA_FRAC_MAX = 0.60
MARGIN_FRAC = 0.03                # 각 변 최소 여백(캔버스 대비)

_BBOX_ATTR_RE = re.compile(r'data-bbox="(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)"')

# 페이지 속 Table div 면적비 실측 분포(전체 6,248개, e25_28791 train 기준).
# 합성 시 이 분포에서 샘플링하되 [AREA_FRAC_MIN, AREA_FRAC_MAX] 로 clip 한다.
_MEASURED_AREA_FRAC_QUANTILES = [
    0.010, 0.036, 0.073, 0.148, 0.351, 0.585, 0.900,
]


def _is_tsr(rec: dict) -> bool:
    return rec.get("prompt_style") == TSR_PROMPT_STYLE


def _load_split(split: str) -> list[dict]:
    path = SRC_DIR / f"{split}.jsonl"
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


# =============================================================================
# 1) TSR 원본 그대로 추출 (cell A/B 용)
# =============================================================================

def build_tsr_only() -> dict[str, list[dict]]:
    OUT_TSR_DIR.mkdir(parents=True, exist_ok=True)
    result = {}
    for split in SPLITS:
        rows = _load_split(split)
        tsr_rows = [r for r in rows if _is_tsr(r)]
        rng = random.Random(SEED + hash(split) % 1000)
        rng.shuffle(tsr_rows)
        dst = OUT_TSR_DIR / f"{split}.jsonl"
        with open(dst, "w", encoding="utf-8") as fout:
            for r in tsr_rows:
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[tsr_only][{split}] in={len(rows)} tsr={len(tsr_rows)} -> {dst}")
        result[split] = tsr_rows
    return result


# =============================================================================
# 2) 합성 페이지 (cell C 용)
# =============================================================================

def _sample_area_frac(rng: random.Random) -> float:
    """실측 분위수 사이를 선형보간으로 샘플링한 뒤 clip."""
    qs = _MEASURED_AREA_FRAC_QUANTILES
    n = len(qs) - 1
    u = rng.random()
    idx = min(int(u * n), n - 1)
    lo, hi = qs[idx], qs[idx + 1]
    frac = lo + (hi - lo) * ((u * n) - idx)
    return max(AREA_FRAC_MIN, min(AREA_FRAC_MAX, frac))


def _canvas_size() -> tuple[int, int]:
    # w/h = CANVAS_ASPECT_WH, w*h = CANVAS_TOTAL_PX
    h = int(round((CANVAS_TOTAL_PX / CANVAS_ASPECT_WH) ** 0.5))
    w = int(round(h * CANVAS_ASPECT_WH))
    return w, h


def _transform_bbox_str(gt_html: str, x0: int, y0: int, x1: int, y1: int) -> str:
    """gt_html 안의 모든 data-bbox(0-1000 정규화) 를 캔버스 내 배치 사각형
    (x0,y0)-(x1,y1) (역시 0-1000 정규화) 기준 affine 으로 재사상한다."""
    sx = (x1 - x0) / 1000.0
    sy = (y1 - y0) / 1000.0

    def repl(m: re.Match) -> str:
        ox0, oy0, ox1, oy1 = (int(m.group(i)) for i in range(1, 5))
        nx0 = round(x0 + ox0 * sx)
        ny0 = round(y0 + oy0 * sy)
        nx1 = round(x0 + ox1 * sx)
        ny1 = round(y0 + oy1 * sy)
        nx0 = max(0, min(1000, nx0))
        ny0 = max(0, min(1000, ny0))
        nx1 = max(0, min(1000, nx1))
        ny1 = max(0, min(1000, ny1))
        return f'data-bbox="{nx0} {ny0} {nx1} {ny1}"'

    return _BBOX_ATTR_RE.sub(repl, gt_html)


def _transform_ocr_info(ocr_info, x0: int, y0: int, x1: int, y1: int):
    if not isinstance(ocr_info, list) or not ocr_info:
        return ocr_info
    sx = (x1 - x0) / 1000.0
    sy = (y1 - y0) / 1000.0
    out = []
    for item in ocr_info:
        if not isinstance(item, dict):
            out.append(item)
            continue
        bbox = item.get("bbox")
        new_item = dict(item)
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            ox0, oy0, ox1, oy1 = bbox
            nx0 = max(0, min(1000, round(x0 + ox0 * sx)))
            ny0 = max(0, min(1000, round(y0 + oy0 * sy)))
            nx1 = max(0, min(1000, round(x0 + ox1 * sx)))
            ny1 = max(0, min(1000, round(y0 + oy1 * sy)))
            new_item["bbox"] = [nx0, ny0, nx1, ny1]
        out.append(new_item)
    return out


def _make_synth_record(rec: dict, idx: int, rng: random.Random) -> dict | None:
    src_path = rec["image_path"]
    try:
        img = Image.open(src_path).convert("RGB")
    except Exception as e:
        print(f"[synth][warn] cannot open {src_path}: {e}", file=sys.stderr)
        return None

    canvas_w, canvas_h = _canvas_size()
    area_frac = _sample_area_frac(rng)
    target_area = area_frac * canvas_w * canvas_h

    src_w, src_h = img.size
    src_aspect = src_w / src_h
    # 종횡비 유지하며 target_area 에 맞는 크기 산출
    paste_h = int(round((target_area / src_aspect) ** 0.5))
    paste_w = int(round(paste_h * src_aspect))
    paste_w = max(8, min(paste_w, canvas_w - 2))
    paste_h = max(8, min(paste_h, canvas_h - 2))

    margin_x = int(canvas_w * MARGIN_FRAC)
    margin_y = int(canvas_h * MARGIN_FRAC)
    max_x = max(margin_x, canvas_w - margin_x - paste_w)
    max_y = max(margin_y, canvas_h - margin_y - paste_h)
    px = rng.randint(margin_x, max_x) if max_x > margin_x else margin_x
    py = rng.randint(margin_y, max_y) if max_y > margin_y else margin_y

    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    resized = img.resize((paste_w, paste_h), Image.LANCZOS)
    canvas.paste(resized, (px, py))

    # 배치 사각형을 0-1000 정규화 좌표로
    bx0 = round(px * 1000 / canvas_w)
    by0 = round(py * 1000 / canvas_h)
    bx1 = round((px + paste_w) * 1000 / canvas_w)
    by1 = round((py + paste_h) * 1000 / canvas_h)

    SYNTH_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    out_img_path = SYNTH_IMAGE_DIR / f"synth_{idx:06d}.jpg"
    canvas.save(out_img_path, quality=92)

    new_rec = dict(rec)
    new_rec["image_path"] = str(out_img_path.resolve())
    new_rec["gt_html"] = _transform_bbox_str(rec["gt_html"], bx0, by0, bx1, by1)
    if "ocr_info" in rec:
        new_rec["ocr_info"] = _transform_ocr_info(rec["ocr_info"], bx0, by0, bx1, by1)
    new_rec["_synth_area_frac"] = round(area_frac, 4)
    return new_rec


def build_synth(tsr_rows_by_split: dict[str, list[dict]]) -> None:
    OUT_SYNTH_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    # --- train: 절반 원본 + 절반 합성 ---
    train_tsr = list(tsr_rows_by_split["train"])
    rng.shuffle(train_tsr)
    half = len(train_tsr) // 2  # 2891
    keep_orig = train_tsr[:half]
    to_synth = train_tsr[half:]

    synth_records = []
    for i, rec in enumerate(to_synth):
        synth_rec = _make_synth_record(rec, i, rng)
        if synth_rec is not None:
            synth_rec.pop("_synth_area_frac", None)
            synth_records.append(synth_rec)

    train_out = keep_orig + synth_records
    rng.shuffle(train_out)
    dst = OUT_SYNTH_DIR / "train.jsonl"
    with open(dst, "w", encoding="utf-8") as fout:
        for r in train_out:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[synth][train] orig_kept={len(keep_orig)} synthesized={len(synth_records)} "
          f"total={len(train_out)} -> {dst}")

    # --- valid: A/B 와 동일한 TSR 601행 그대로 재사용 ---
    valid_rows = tsr_rows_by_split["valid"]
    dst = OUT_SYNTH_DIR / "valid.jsonl"
    with open(dst, "w", encoding="utf-8") as fout:
        for r in valid_rows:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[synth][valid] tsr={len(valid_rows)} (원본 그대로) -> {dst}")

    # --- test: TSR 원본 + 합성(오프라인 분석용, 학습 미사용) ---
    test_tsr = tsr_rows_by_split["test"]
    test_rng = random.Random(SEED + 1)
    test_synth = []
    for i, rec in enumerate(test_tsr):
        synth_rec = _make_synth_record(rec, 100_000 + i, test_rng)
        if synth_rec is not None:
            synth_rec.pop("_synth_area_frac", None)
            test_synth.append(synth_rec)
    test_out = test_tsr + test_synth
    dst = OUT_SYNTH_DIR / "test.jsonl"
    with open(dst, "w", encoding="utf-8") as fout:
        for r in test_out:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[synth][test] orig={len(test_tsr)} synth={len(test_synth)} "
          f"total={len(test_out)} -> {dst}")


def build() -> None:
    tsr_rows_by_split = build_tsr_only()
    build_synth(tsr_rows_by_split)


# =============================================================================
# Verify
# =============================================================================

def verify() -> bool:
    ok = True

    # --- tsr6598 ---
    total_by_split = {}
    for split in SPLITS:
        path = OUT_TSR_DIR / f"{split}.jsonl"
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        total_by_split[split] = len(rows)
        bad_style = sum(1 for r in rows if r.get("prompt_style") != TSR_PROMPT_STYLE)
        bad_scale = sum(1 for r in rows if r.get("bbox_scale") != 1000)
        paths = [r["image_path"] for r in rows]
        dup = len(paths) - len(set(paths))
        missing = sum(1 for p in paths[:200] if not Path(p).exists())
        print(f"[verify][tsr6598][{split}] rows={len(rows)} bad_style={bad_style} "
              f"bad_scale={bad_scale} dup_path={dup} missing_sample(first200)={missing}")
        if bad_style or bad_scale or dup or missing:
            ok = False

    expected = {"train": 5783, "valid": 601, "test": 214}
    for split, exp in expected.items():
        if total_by_split.get(split) != exp:
            print(f"[verify][tsr6598][{split}] COUNT MISMATCH: got={total_by_split.get(split)} expected={exp}")
            ok = False

    # --- tsrsynth5783 ---
    train_rows = [json.loads(l) for l in open(OUT_SYNTH_DIR / "train.jsonl", encoding="utf-8") if l.strip()]
    valid_rows = [json.loads(l) for l in open(OUT_SYNTH_DIR / "valid.jsonl", encoding="utf-8") if l.strip()]
    n_synth_img = sum(1 for r in train_rows if "e25plus_synth" in r["image_path"])
    n_orig_img = len(train_rows) - n_synth_img
    print(f"[verify][tsrsynth][train] total={len(train_rows)} orig={n_orig_img} synth={n_synth_img}")
    print(f"[verify][tsrsynth][valid] total={len(valid_rows)} (tsr6598 valid 와 동일해야 함: {total_by_split.get('valid')})")
    if len(train_rows) != 5783:
        print(f"[verify][tsrsynth][train] COUNT MISMATCH: got={len(train_rows)} expected=5783")
        ok = False
    if len(valid_rows) != total_by_split.get("valid"):
        ok = False

    missing_synth = sum(1 for r in train_rows if "e25plus_synth" in r["image_path"] and not Path(r["image_path"]).exists())
    if missing_synth:
        print(f"[verify][tsrsynth][train] missing synth images: {missing_synth}")
        ok = False

    # bbox range check on synth records
    bad_bbox = 0
    examples = []
    for r in train_rows:
        if "e25plus_synth" not in r["image_path"]:
            continue
        for m in _BBOX_ATTR_RE.finditer(r["gt_html"]):
            vals = [int(m.group(i)) for i in range(1, 5)]
            if any(v < 0 or v > 1000 for v in vals):
                bad_bbox += 1
                if len(examples) < 5:
                    examples.append((r["image_path"], vals))
                break
    print(f"[verify][tsrsynth][train] out_of_range_bbox_records={bad_bbox}")
    if examples:
        print("[verify] examples:", examples)
    if bad_bbox:
        ok = False

    return ok


def main() -> None:
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
