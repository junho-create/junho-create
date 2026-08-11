#!/usr/bin/env python3
"""dots.ocr 와 PaddleOCR-VL 1.6 이 **공통으로 수식이라고 본 영역**만 뽑고, 그 영역을
크롭 이미지로 만든다.

판정은 페이지 단위가 아니라 **수식 단위**다. 두 모델의 Formula 블록을 bbox IoU 로
그리디 매칭해서 짝이 지어진 것만 남긴다 — 한쪽만 잡은 건 오탐일 확률이 높다.

좌표계 함정
-----------
dots 의 `bbox` 는 원본 픽셀이 아니라 Qwen2VL smart_resize 좌표계다(레코드의 width/height
가 그 크기). ArxivFormula 는 640x828 / 640x905 라 28 의 배수가 아니어서 644x840 / 644x896
으로 리사이즈된다. paddle 의 `bbox` 는 원본 픽셀이다. 그래서 매칭도 크롭도 전부
**두 러너가 각자 올바른 기준으로 이미 정규화해 둔 `bbox_1000`** 으로만 한다.

크롭 해상도
-----------
원본이 640px 폭이라 수식 하나는 대략 200x25px 밖에 안 된다. 그대로 두면 judge 도 모델도
못 읽으므로 Lanczos 로 업스케일한다(짧은 변 >= MIN_SHORT, 배율 <= MAX_SCALE).

사용:
    python3 build_formula_consensus.py                # 전체
    python3 build_formula_consensus.py --limit 200    # 페이지 수 제한(디버그)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

HERE = Path(__file__).parent
SPLITS = ("train", "valid", "test")

IOU_MIN = 0.5
PAD_PX = 6
MIN_SHORT = 96      # 크롭 짧은 변 최소 픽셀
MIN_LONG = 700      # 크롭 긴 변 최소 픽셀
MAX_LONG = 1400     # 크롭 긴 변 최대 픽셀
MAX_SCALE = 6.0     # 업스케일 배율 상한 (더 키워봐야 없는 정보가 생기진 않는다)


# ── bbox 유틸 (ocr_filter/cmcv/run.py 의 _bbox_iou / _match_by_bbox 와 같은 로직) ──
def bbox_iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = min(a[0], a[2]), min(a[1], a[3]), max(a[0], a[2]), max(a[1], a[3])
    bx0, by0, bx1, by1 = min(b[0], b[2]), min(b[1], b[3]), max(b[0], b[2]), max(b[1], b[3])
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union > 0 else 0.0


def match_by_bbox(a_items, b_items, iou_min=IOU_MIN):
    """(el, bbox_1000) 목록 두 개를 IoU 그리디 매칭. 반환 [(a_el, b_el, iou), ...].

    등장 순서로 짝지으면 한쪽이 수식을 하나만 더/덜 잡아도 뒤가 전부 밀려 죄다 오답이
    된다(표에서 TEDS 중앙값이 정확히 0 으로 나왔던 원인). bbox 로 짝지어야 개수가 달라도
    같은 자리끼리 붙는다."""
    cand = []
    for ia, (_, ba) in enumerate(a_items):
        for ib, (_, bb) in enumerate(b_items):
            iou = bbox_iou(ba, bb)
            if iou >= iou_min:
                cand.append((iou, ia, ib))
    cand.sort(key=lambda t: -t[0])
    used_a, used_b, pairs = set(), set(), []
    for iou, ia, ib in cand:
        if ia in used_a or ib in used_b:
            continue
        used_a.add(ia)
        used_b.add(ib)
        pairs.append((a_items[ia][0], b_items[ib][0], iou))
    return pairs


def union_box(a, b):
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


# 두 모델의 라벨 체계가 다르다. dots 는 `formula`, paddle(PP-DocLayoutV3)은
# `display_formula` 를 쓴다. 이걸 안 맞추면 paddle 쪽 수식이 0개로 잡혀 합의가 통째로
# 비어버린다(실측: dots 1,873 vs paddle 0). `inline_formula` 는 본문 줄 안에 박힌
# 인라인 수식이라 dots 쪽에 대응하는 독립 블록이 없으므로 제외한다.
FORMULA_LABELS = {"formula", "display_formula", "isolate_formula", "interline_equation",
                  "equation", "formula_caption"}


def formulas_of(rec: dict) -> list[tuple[dict, list[float]]]:
    """레코드에서 (블록, bbox_1000) 목록. bbox_1000 이 없는 블록은 버린다."""
    out = []
    for b in rec.get("pred_blocks") or []:
        if (b.get("label") or "").lower() not in FORMULA_LABELS:
            continue
        bb = b.get("bbox_1000")
        if not bb or len(bb) != 4:
            continue
        if not (b.get("content") or "").strip():
            continue
        out.append((b, [float(v) for v in bb]))
    return out


def crop_formula(img: Image.Image, bbox_1000, out_path: Path) -> dict | None:
    """bbox_1000 -> 원본 픽셀 크롭 + 업스케일 저장. 반환 메타(없으면 None)."""
    W, H = img.size
    x0 = max(0, int(bbox_1000[0] * W / 1000) - PAD_PX)
    y0 = max(0, int(bbox_1000[1] * H / 1000) - PAD_PX)
    x1 = min(W, int(bbox_1000[2] * W / 1000) + PAD_PX)
    y1 = min(H, int(bbox_1000[3] * H / 1000) + PAD_PX)
    if x1 - x0 < 8 or y1 - y0 < 6:
        return None

    crop = img.crop((x0, y0, x1, y1))
    cw, ch = crop.size
    # 짧은 변만 기준으로 잡으면 여러 줄 수식(행렬/cases)이 배율 1.1 근처에 머물러
    # 원본 640px 페이지의 작은 글씨 그대로 남는다. 긴 변 하한을 같이 걸어야
    # 그런 것들이 실제로 커진다.
    scale = min(MAX_SCALE, max(1.0, MIN_SHORT / min(cw, ch), MIN_LONG / max(cw, ch)))
    if max(cw, ch) * scale > MAX_LONG:
        scale = max(1.0, MAX_LONG / max(cw, ch))
    if scale > 1.0:
        crop = crop.resize((max(1, round(cw * scale)), max(1, round(ch * scale))), Image.LANCZOS)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(out_path)

    # 크롭 이미지 안에서 수식이 실제로 차지하는 영역(패딩 제외)을 0-1000 으로.
    # 이미지 가장자리라 패딩이 잘린 경우까지 반영하려고 원래 좌표에서 되짚는다.
    fx0 = int(bbox_1000[0] * W / 1000)
    fy0 = int(bbox_1000[1] * H / 1000)
    fx1 = int(bbox_1000[2] * W / 1000)
    fy1 = int(bbox_1000[3] * H / 1000)
    cw2, ch2 = crop.size
    inner = [
        max(0, min(1000, round((fx0 - x0) / (x1 - x0) * 1000))),
        max(0, min(1000, round((fy0 - y0) / (y1 - y0) * 1000))),
        max(0, min(1000, round((fx1 - x0) / (x1 - x0) * 1000))),
        max(0, min(1000, round((fy1 - y0) / (y1 - y0) * 1000))),
    ]
    return {
        "crop_path": str(out_path),
        "bbox_orig": [x0, y0, x1, y1],
        "crop_size": [cw2, ch2],
        "crop_scale": round(scale, 2),
        "inner_bbox_1000": inner,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dots-dir", default=str(HERE / "dots_out"))
    ap.add_argument("--paddle-dir", default=str(HERE / "paddle_out"))
    ap.add_argument("--crops-dir", default=str(HERE / "crops"))
    ap.add_argument("--out", default=str(HERE / "formula_pairs.jsonl"))
    ap.add_argument("--iou-min", type=float, default=IOU_MIN)
    ap.add_argument("--limit", type=int, help="split 당 페이지 수 제한(디버그)")
    args = ap.parse_args()

    stats = {
        "pages_both": 0, "pages_with_pair": 0,
        "dots_formulas": 0, "paddle_formulas": 0, "matched": 0,
        "crop_failed": 0,
    }
    sizes: list[int] = []

    with open(args.out, "w", encoding="utf-8") as fout:
        for split in SPLITS:
            dots_p = Path(args.dots_dir) / f"{split}.jsonl"
            pad_p = Path(args.paddle_dir) / f"{split}.jsonl"
            if not (dots_p.is_file() and pad_p.is_file()):
                print(f"[{split}] 건너뜀 (파일 없음)")
                continue

            paddle = {}
            with pad_p.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        paddle[r["key"]] = r

            n_page = 0
            with dots_p.open(encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    d = json.loads(line)
                    p = paddle.get(d["key"])
                    if p is None:
                        continue
                    if args.limit and n_page >= args.limit:
                        break
                    n_page += 1
                    stats["pages_both"] += 1

                    a_items, b_items = formulas_of(d), formulas_of(p)
                    stats["dots_formulas"] += len(a_items)
                    stats["paddle_formulas"] += len(b_items)
                    if not (a_items and b_items):
                        continue
                    pairs = match_by_bbox(a_items, b_items, args.iou_min)
                    if not pairs:
                        continue

                    try:
                        img = Image.open(d["image_path"]).convert("RGB")
                    except Exception as e:  # noqa: BLE001
                        print(f"  이미지 열기 실패 {d['image_path']}: {e}")
                        continue

                    n_kept = 0
                    for idx, (a_el, b_el, iou) in enumerate(pairs):
                        ub = union_box(a_el["bbox_1000"], b_el["bbox_1000"])
                        safe = d["key"].replace("/", "__")
                        meta = crop_formula(
                            img, ub, Path(args.crops_dir) / split / f"{safe}__f{idx}.png")
                        if meta is None:
                            stats["crop_failed"] += 1
                            continue
                        sizes.append(meta["crop_size"][0])
                        fout.write(json.dumps({
                            "pair_id": f"{d['key']}#f{idx}",
                            "key": d["key"],
                            "split": split,
                            "formula_idx": idx,
                            "image_path": d["image_path"],
                            "bbox_1000": [round(v) for v in ub],
                            # layout 버전에서 dots 블록을 되찾아 갈아끼우기 위한 키
                            "dots_bbox_1000": [int(v) for v in a_el["bbox_1000"]],
                            "iou": round(iou, 3),
                            "cand_dots": (a_el.get("content") or "").strip(),
                            "cand_paddle": (b_el.get("content") or "").strip(),
                            **meta,
                        }, ensure_ascii=False) + "\n")
                        n_kept += 1
                        stats["matched"] += 1
                    if n_kept:
                        stats["pages_with_pair"] += 1
                    img.close()
            print(f"[{split}] 페이지 {n_page}")

    print("\n=== 합의 통계 ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    if sizes:
        sizes.sort()
        print(f"  크롭 폭 중앙값 {sizes[len(sizes) // 2]}px "
              f"(min {sizes[0]}, max {sizes[-1]})")
    print(f"  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
