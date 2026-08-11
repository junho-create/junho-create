#!/usr/bin/env python3
"""combined_e21_grid 학습 데이터에서 라벨 품질이 의심되는 레코드를 걸러낸다.

원본은 건드리지 않고 --out_dir 에 정제본을 새로 쓴다. 제거 사유는 removed.jsonl 에
레코드별로 남기므로 나중에 되돌리거나 개별 검토할 수 있다.

필터 (2026-07-28 사용자 지정):
  1 tiny_table_bbox   reference_16886 의 T*/nested* (표 crop) 인데 Table bbox 가 지나치게 작음.
                      판정: 최대 bbox 면적 < --tiny_area (페이지 대비) 또는
                            bbox 면적 / OCR 분포영역 면적 < --tiny_cover
                      -> 라벨러가 표가 아닌 조각(페이지번호 등)을 표로 잡은 케이스.
  2 short_ocr         new_63541 인데 추출된 OCR 총 글자수 <= --min_chars.
                      -> OCR 이 거의 안 나온 페이지(빈 페이지/이미지 위주)라 학습가치 낮음.
  3 uncovered_ocr     OCR 이 잡은 영역 중 bbox 합집합이 못 덮는 면적 비율 > --max_uncovered.
                      -> "OCR 은 글자를 잡았는데 라벨 박스가 그걸 안 덮는다" = 라벨 누락.
                      면적 비율로 재므로 누락 규모에 비례한다.
                      주의: 정상 문서도 중앙값 2~5% 는 삐져나간다(OCR 은 글자에 딱 붙고
                      레이아웃 박스는 블록 단위라 경계에서 어긋남). 그래서 0 이 아니라
                      new_63541 은 0.05, reference_16886 은 0.10 을 기본값으로 둔다.
  4 invented_ellipsis gt_html 에는 말줄임(... 또는 …)이 있는데 OCR 텍스트엔 없음.
                      -> 원본에 없는 축약을 라벨이 임의로 만든 케이스. 표 안이면 특히 유해.
  5 irregular_table    new_63541 레이아웃 페이지에 있는 Table 블록이 사각형이 아님.
                      판정: rowspan 을 반영한 행별 실효 폭(자기 colspan 합 + 위에서 내려오는
                      rowspan 몫)이 표 안에서 전부 같아야 정상인데, 어긋나는 행이 하나라도
                      있으면 걸린다(튀어나옴 + 좁아짐 모두, 2026-07-28 실측 492/2663 문서
                      해당). reference_16886 표(unified_table_with_ocr)는 대상 아님 —
                      그쪽은 crop 자체가 표 하나라 다른 방식으로 이미 검증됨.

사용:
  python filter_e21grid.py                      # 기본 임계값으로 train/valid/test 정제
  python filter_e21grid.py --splits train       # train 만
  python filter_e21grid.py --dry_run            # 파일 안 쓰고 통계만
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re

import numpy as np
from bs4 import BeautifulSoup

BB = re.compile(r'data-bbox="([^"]+)"[^>]*data-label="([^"]+)"')
ELL = re.compile(r"\.{3,}|…")
TABLE_NAME = re.compile(r"^T\d")
GRID = 250          # 합집합 면적 계산용 래스터 해상도(1000 좌표계를 250x250 으로)
SCALE = 1000.0


def raster(boxes):
    """박스들을 GRID x GRID 불리언 마스크로 그린다(합집합 계산용)."""
    m = np.zeros((GRID, GRID), bool)
    for b in boxes:
        x0, y0, x1, y1 = [max(0, min(GRID, int(round(v / SCALE * GRID)))) for v in b]
        if x1 > x0 and y1 > y0:
            m[y0:y1, x0:x1] = True
    return m


def union_area(boxes) -> float:
    """박스들의 합집합 면적(페이지 대비 0~1). 겹침을 중복 계산하지 않는다."""
    if not boxes:
        return 0.0
    m = np.zeros((GRID, GRID), bool)
    for b in boxes:
        x0, y0, x1, y1 = [max(0, min(GRID, int(round(v / SCALE * GRID)))) for v in b]
        if x1 > x0 and y1 > y0:
            m[y0:y1, x0:x1] = True
    return float(m.sum()) / (GRID * GRID)


DIV_BLOCK = re.compile(r'<div\s+data-bbox="([^"]+)"\s+data-label="([^"]+)"\s*>(.*?)</div>', re.S)


def strip_giant(gt_html: str, max_area: float):
    """페이지 전면을 덮는 div 블록을 통째로 제거한다.

    실측상 이런 박스는 144/150 이 정확히 "0 0 1000 1000" 이고 라벨은 Picture/Page-footer
    다(라벨러 실패 시 넣는 전면 박스). 내부에 중첩 div 가 없으므로 블록 단위 제거가 안전하다.

    중요: 이 제거는 uncovered_ocr 계산 **전에** 해야 한다. 전면 박스가 남아 있으면 모든
    OCR 을 덮어버려 uncovered 가 0 으로 나오고, 라벨이 실제로 빠진 문서가 필터를 통과한다.
    """
    removed = 0

    def sub(m):
        nonlocal removed
        try:
            b = [float(x) for x in m.group(1).split()]
        except ValueError:
            return m.group(0)
        if len(b) == 4 and (b[2] - b[0]) * (b[3] - b[1]) / (SCALE * SCALE) > max_area:
            removed += 1
            return ""
        return m.group(0)

    out = DIV_BLOCK.sub(sub, gt_html or "")
    return (out.strip(), removed)


def bboxes_of(gt_html: str):
    out = []
    for b, _ in BB.findall(gt_html or ""):
        try:
            v = [float(x) for x in b.split()]
        except ValueError:
            continue
        if len(v) == 4:
            out.append(v)
    return out


def table_shape_irregular(gt_html: str) -> bool:
    """gt_html 안의 <table> 이 사각형이 아니면 True.

    행별 실효 폭 = 그 행 자신의 colspan 합 + 위 행에서 rowspan 으로 내려오는 폭.
    정상 표는 이 값이 모든 행에서 같다(표 전체 폭). 어긋나면(튀어나오든 좁아지든)
    셀 라벨링이 잘못됐다는 뜻이다.
    """
    soup = BeautifulSoup(gt_html or "", "html.parser")
    for table in soup.find_all("table"):
        active = []  # [남은 rowspan-1, colspan] 아직 소모되지 않은 위쪽 rowspan
        widths = []
        for tr in table.find_all("tr"):
            carry_in = sum(c for _, c in active)
            own = 0
            new_spans = []
            for cell in tr.find_all(["td", "th"]):
                try:
                    cs = int(cell.get("colspan", 1))
                except ValueError:
                    cs = 1
                own += cs
                try:
                    rs = int(cell.get("rowspan", 1))
                except ValueError:
                    rs = 1
                if rs > 1:
                    new_spans.append([rs - 1, cs])
            widths.append(carry_in + own)
            active = [[r - 1, c] for r, c in active if r - 1 > 0] + new_spans
        if len(widths) < 2:
            continue
        mode = collections.Counter(widths).most_common(1)[0][0]
        if any(w != mode for w in widths):
            return True
    return False


def source_of(path: str) -> str:
    return "reference_16886" if "reference_16886" in path else "new_63541"


def judge(rec, args):
    """제거 사유 리스트를 반환한다(빈 리스트면 통과)."""
    reasons = []
    path = rec["image_path"]
    name = os.path.basename(path)
    src = source_of(path)
    boxes = bboxes_of(rec.get("gt_html", ""))
    ocr_items = rec.get("ocr_info") or []
    ocr_boxes = [o["bbox"] for o in ocr_items if o.get("bbox") and len(o["bbox"]) == 4]

    # 1) 표 crop 인데 bbox 가 조각
    if src == "reference_16886" and (TABLE_NAME.match(name) or "nested" in name.lower()):
        if not boxes:
            reasons.append("tiny_table_bbox:no_box")
        else:
            amax = max((b[2] - b[0]) * (b[3] - b[1]) for b in boxes) / (SCALE * SCALE)
            if amax < args.tiny_area:
                reasons.append(f"tiny_table_bbox:area={amax:.4f}")
            elif ocr_boxes:
                ox0 = min(o[0] for o in ocr_boxes); oy0 = min(o[1] for o in ocr_boxes)
                ox1 = max(o[2] for o in ocr_boxes); oy1 = max(o[3] for o in ocr_boxes)
                oext = max(1.0, (ox1 - ox0) * (oy1 - oy0)) / (SCALE * SCALE)
                if amax / oext < args.tiny_cover:
                    reasons.append(f"tiny_table_bbox:cover={amax/oext:.3f}")

    # 2) new_63541 인데 OCR 이 너무 적음
    if src == "new_63541":
        chars = sum(len(o.get("text") or "") for o in ocr_items)
        if chars <= args.min_chars:
            reasons.append(f"short_ocr:{chars}")

    # 3) OCR 이 잡은 영역 중 라벨 박스가 못 덮는 비율
    #    면적 비율이 아니라 "덮지 못한 몫"을 직접 재므로, 밀집 표에서 비율이 1.0 근처가 되는
    #    구조적 오탐(예: nested_000030)이 생기지 않는다. 표/레이아웃 모두에 적용한다.
    #    임계값은 소스별로 다르다(2026-07-28 사용자 지정): new_63541 이 라벨 누락이 더 잦아
    #    0.05 로 더 엄격하게, reference_16886 은 0.10.
    if ocr_boxes:
        om = raster(ocr_boxes)
        oa = int(om.sum())
        if oa > 0:
            unc = float((om & ~raster(boxes)).sum()) / oa
            th = args.max_uncovered_new if src == "new_63541" else args.max_uncovered
            if unc > th:
                reasons.append(f"uncovered_ocr:{unc:.3f}>{th}")

    # 5) new_63541 레이아웃의 비사각형 Table
    if (src == "new_63541" and rec.get("prompt_style") == "unified_layout"
            and 'data-label="Table"' in (rec.get("gt_html") or "")):
        if table_shape_irregular(rec.get("gt_html", "")):
            reasons.append("irregular_table_shape")

    # 4) 원본에 없는 말줄임(임의 축약)
    gt = rec.get("gt_html", "")
    if ELL.search(gt):
        otxt = " ".join((o.get("text") or "") for o in ocr_items)
        if not ELL.search(otxt):
            in_table = False
            for m in ELL.finditer(gt):
                seg = gt[max(0, m.start() - 200):m.start()]
                if "<td" in seg or "<th" in seg:
                    in_table = True
                    break
            reasons.append("invented_ellipsis:table" if in_table else "invented_ellipsis:text")

    return reasons, src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="combined_e21_grid")
    ap.add_argument("--out_dir", default="combined_e23_refined")
    ap.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    ap.add_argument("--tiny_area", type=float, default=0.10,
                    help="표 crop 최대 bbox 면적(페이지 대비) 하한. 실측상 8건이 0.005 미만이고 다음이 0.178")
    ap.add_argument("--tiny_cover", type=float, default=0.50,
                    help="표 bbox 면적 / OCR 분포영역 면적 하한")
    ap.add_argument("--min_chars", type=int, default=100, help="new_63541 최소 OCR 글자수(이하 제거)")
    ap.add_argument("--max_uncovered", type=float, default=0.10,
                    help="reference_16886 의 uncovered 상한(초과 시 제거). 실측 중앙값 0.053")
    ap.add_argument("--max_uncovered_new", type=float, default=0.05,
                    help="new_63541 의 uncovered 상한. 실측 중앙값 0.023 이라 더 조인다")
    ap.add_argument("--no_ellipsis_filter", action="store_true", help="4번 필터 끄기")
    ap.add_argument("--strip_giant_area", type=float, default=0.85,
                    help="이 면적(페이지 대비)을 넘는 div 블록은 제거한다. uncovered 계산 전에 수행")
    ap.add_argument("--no_strip_giant", action="store_true", help="전면 박스 제거 끄기")
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()

    if not a.dry_run:
        os.makedirs(a.out_dir, exist_ok=True)
    grand = collections.Counter()
    removed_all = []

    for split in a.splits:
        src_path = os.path.join(a.data_dir, f"{split}.jsonl")
        if not os.path.exists(src_path):
            print(f"[skip] {src_path} 없음")
            continue
        recs = [json.loads(l) for l in open(src_path, encoding="utf-8") if l.strip()]
        kept, removed = [], []
        cnt = collections.Counter()
        bysrc = collections.Counter()
        n_strip_box = n_strip_rec = 0
        for i, r in enumerate(recs):
            # 표 crop 은 Table bbox 가 원래 페이지의 90% 이상이라 대상에서 제외
            if not a.no_strip_giant and r.get("prompt_style") == "unified_layout":
                new_html, ns = strip_giant(r.get("gt_html", ""), a.strip_giant_area)
                if ns:
                    r = {**r, "gt_html": new_html}
                    n_strip_box += ns
                    n_strip_rec += 1
            reasons, rsrc = judge(r, a)
            if not bboxes_of(r.get("gt_html", "")):
                reasons = reasons + ["empty_after_strip"] if "empty_after_strip" not in reasons else reasons
            if a.no_ellipsis_filter:
                reasons = [x for x in reasons if not x.startswith("invented_ellipsis")]
            if reasons:
                removed.append({"split": split, "index": i, "image_path": r["image_path"],
                                "source": rsrc, "reasons": reasons})
                for x in reasons:
                    cnt[x.split(":")[0]] += 1
                bysrc[rsrc] += 1
            else:
                kept.append(r)
        print(f"[{split}] {len(recs)} -> {len(kept)} (제거 {len(removed)}, {100*len(removed)/max(len(recs),1):.1f}%)")
        for k, v in cnt.most_common():
            print(f"    {k}: {v}")
        print(f"    소스별 제거: {dict(bysrc)}")
        print(f"    전면 박스 제거: {n_strip_box}개 (레코드 {n_strip_rec}건에서)")
        grand.update(cnt)
        removed_all.extend(removed)

        if not a.dry_run:
            with open(os.path.join(a.out_dir, f"{split}.jsonl"), "w", encoding="utf-8") as f:
                for r in kept:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n총 제거 사유 집계: {dict(grand)}")
    if not a.dry_run:
        with open(os.path.join(a.out_dir, "removed.jsonl"), "w", encoding="utf-8") as f:
            for r in removed_all:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        json.dump({"thresholds": vars(a), "counts": dict(grand), "removed": len(removed_all)},
                  open(os.path.join(a.out_dir, "filter_report.json"), "w"), ensure_ascii=False, indent=1)
        print(f"저장: {a.out_dir}/ (removed.jsonl, filter_report.json 포함)")


if __name__ == "__main__":
    main()
