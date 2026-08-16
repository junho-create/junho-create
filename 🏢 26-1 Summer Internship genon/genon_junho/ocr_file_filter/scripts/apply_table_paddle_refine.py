#!/usr/bin/env python3
"""export 된 gt_html 중 data-label="Table" div 내부를 paddle(external_b) 표 HTML로 교체.

combined_e24_table_refined/CHANGES.md 의 규칙을 그대로 이식(2026-08-01, batch5400 에 처음 적용).
cmcv_results.jsonl 에 이미 저장된 external_b(paddle) 요소를 재사용하므로 추가 모델 콜 없음.

교체 조건 (모두 만족해야 교체, 하나라도 실패하면 원본 유지):
  1. bbox IoU >= 0.3 로 paddle Table 요소와 1:1 매칭 (0-1000 정규화 좌표계)
  2. GT/paddle 양쪽 다 <table>...</table> 단일 블록 (다른 콘텐츠가 섞인 비정형 표는 제외)
  3. 글자수 비율 min(len(gt),len(pred))/max(...) >= 0.85
  4. 앞부분 15자 유사도 >= 0.6, 뒷부분 15자 유사도 >= 0.6 (공백 제거 후, difflib.SequenceMatcher)
  5. 전체 텍스트 유사도(difflib.SequenceMatcher) >= 0.5
paddle 쪽 셀 안 리터럴 개행은 비교 전 공백으로 치환하고, 실제로 삽입하는 html에도 동일하게 치환.

사용:
    python3 scripts/apply_table_paddle_refine.py \
        --gt-html      /home/jhyeo/ocr_filter_result/batch5400/final_gt_1571.jsonl \
        --unified      /home/jhyeo/ocr_filter_result/batch5400/unified.jsonl \
        --cmcv-results /home/jhyeo/ocr_filter_result/batch5400/cmcv_results.jsonl \
        --out          /home/jhyeo/ocr_filter_result/batch5400/final_gt_1571_tablefix.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

from ocr_filter.cmcv.normalize import normalize_category  # noqa: E402

_TAG_RE = re.compile(r"<[^>]+>")
_SINGLE_TABLE_RE = re.compile(r"^\s*<table\b.*</table>\s*$", re.DOTALL | re.IGNORECASE)
_DIV_TABLE_RE = re.compile(
    r'(<div data-bbox="([\d\s]+)" data-label="Table">\n)(.*?)(\n</div>)', re.DOTALL,
)


def strip_tags(html: str) -> str:
    return re.sub(r"\s+", " ", _TAG_RE.sub(" ", html or "")).strip()


def clean_newlines_to_space(html: str) -> str:
    return (html or "").replace("\r\n", " ").replace("\r", " ").replace("\n", " ")


def iou(a: list[float], b: list[float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def gates_pass(gt_text: str, pred_text: str) -> bool:
    if not gt_text or not pred_text:
        return False
    lo, hi = sorted((len(gt_text), len(pred_text)))
    if lo / hi < 0.85:
        return False
    gt_ns = gt_text.replace(" ", "")
    pd_ns = pred_text.replace(" ", "")
    front = SequenceMatcher(None, gt_ns[:15], pd_ns[:15]).ratio()
    back = SequenceMatcher(None, gt_ns[-15:], pd_ns[-15:]).ratio()
    if front < 0.6 or back < 0.6:
        return False
    overall = SequenceMatcher(None, gt_text, pred_text).ratio()
    return overall >= 0.5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-html", required=True, help="export gt-html 출력 (image_path/gt_html 필드 필요)")
    ap.add_argument("--unified", required=True, help="id<->image_path 조인용 unified.jsonl")
    ap.add_argument("--cmcv-results", required=True, help="external_b(paddle) 요소 출처")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-iou", type=float, default=0.3)
    args = ap.parse_args()

    image_to_id: dict[str, str] = {}
    with open(args.unified, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            image_to_id[d["image_path"]] = d["id"]

    id_to_paddle_tables: dict[str, list[dict]] = {}
    with open(args.cmcv_results, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            els = (d.get("elements") or {}).get("external_b") or []
            tables = [e for e in els
                      if normalize_category(e.get("category") or "") == "Table"
                      and e.get("bbox") and len(e["bbox"]) == 4 and (e.get("text") or "").strip()]
            if tables:
                id_to_paddle_tables[d["id"]] = tables

    print(f"unified id: {len(image_to_id)}건, cmcv paddle-table 보유 id: {len(id_to_paddle_tables)}건")

    counts = Counter()
    img_size_cache: dict[str, tuple[int, int]] = {}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.gt_html, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            counts["total_records"] += 1
            gt_html = rec.get("gt_html", "")

            div_matches = list(_DIV_TABLE_RE.finditer(gt_html))
            if not div_matches:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue

            rec_id = image_to_id.get(rec["image_path"])
            paddle_tables = id_to_paddle_tables.get(rec_id, []) if rec_id else []

            img_w = img_h = None
            if paddle_tables:
                key = rec["image_path"]
                if key not in img_size_cache:
                    try:
                        with Image.open(key) as im:
                            img_size_cache[key] = im.size
                    except Exception:
                        img_size_cache[key] = (0, 0)
                img_w, img_h = img_size_cache[key]

            used_paddle_idx: set[int] = set()
            pieces = []
            last_end = 0

            for m in div_matches:
                counts["table_div_total"] += 1
                prefix, bbox_str, content, suffix = m.group(1), m.group(2), m.group(3), m.group(4)
                pieces.append(gt_html[last_end:m.start()])
                last_end = m.end()

                if not _SINGLE_TABLE_RE.match(content.strip()):
                    counts["skip_not_single_table"] += 1
                    pieces.append(prefix + content + suffix)
                    continue

                if not paddle_tables or not img_w or not img_h:
                    counts["skip_no_paddle_candidate"] += 1
                    pieces.append(prefix + content + suffix)
                    continue

                gt_bbox = [float(x) for x in bbox_str.split()]  # 0-1000
                gt_bbox_n = [v / 1000 for v in gt_bbox]

                best_iou, best_idx = 0.0, -1
                for idx, pe in enumerate(paddle_tables):
                    if idx in used_paddle_idx:
                        continue
                    px0, py0, px1, py1 = pe["bbox"]
                    pbbox_n = [px0 / img_w, py0 / img_h, px1 / img_w, py1 / img_h]
                    v = iou(gt_bbox_n, pbbox_n)
                    if v > best_iou:
                        best_iou, best_idx = v, idx

                if best_idx < 0 or best_iou < args.min_iou:
                    counts["skip_no_iou_match"] += 1
                    pieces.append(prefix + content + suffix)
                    continue

                pred_html_raw = paddle_tables[best_idx]["text"]
                pred_html_clean = clean_newlines_to_space(pred_html_raw)

                if not _SINGLE_TABLE_RE.match(pred_html_clean.strip()):
                    counts["skip_pred_not_single_table"] += 1
                    pieces.append(prefix + content + suffix)
                    continue

                gt_text = strip_tags(content)
                pred_text = strip_tags(pred_html_clean)

                if not gates_pass(gt_text, pred_text):
                    counts["skip_gate_fail"] += 1
                    pieces.append(prefix + content + suffix)
                    continue

                used_paddle_idx.add(best_idx)
                counts["replaced"] += 1
                pieces.append(prefix + pred_html_clean.strip() + suffix)

            pieces.append(gt_html[last_end:])
            rec["gt_html"] = "".join(pieces)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print()
    for k, v in counts.most_common():
        print(f"  {k}: {v}")
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
