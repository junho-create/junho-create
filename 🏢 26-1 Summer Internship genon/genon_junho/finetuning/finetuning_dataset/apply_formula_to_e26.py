#!/usr/bin/env python3
"""combined_e26_28520 안에 이미 존재하는 "수식 있는 layout 페이지" 351장(train286/
valid36/test29, dotsocr_infer/e25_out_formula_only)의 Formula div 내용만 dots.ocr
예측으로 바꾼다(제자리 교체, 행 추가 아님 — 이 페이지들은 이미 e26 안에 원본
unified_layout 행으로 들어있다). Table/Text/... 등 다른 div 는 손대지 않는다.

매칭 로직은 build_c_formula_augmented.py 와 동일한 containment 방식(dots box 가
GT formula 여러 개를 하나로 합쳐 검출하는 경우 대응).
"""
import json
from pathlib import Path

from bs4 import BeautifulSoup

E26_DIR = Path("/home/jhyeo/finetuning/finetuning_dataset/combined_e26_28520")
FORMULA_DIR = Path("/home/jhyeo/finetuning/finetuning_dataset/dotsocr_infer/e25_out_formula_only")
SPLITS = ("train", "valid", "test")


def area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def inter_area(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    return max(0, ix1 - ix0) * max(0, iy1 - iy0)


def containment(gt_box, dots_box):
    a = area(gt_box)
    if a <= 0:
        return 0.0
    return inter_area(gt_box, dots_box) / a


def load_dots_formulas(split: str) -> dict[str, list[dict]]:
    out = {}
    path = FORMULA_DIR / f"{split}.jsonl"
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            soup = BeautifulSoup(r["gt_html"], "html.parser")
            items = []
            for div in soup.find_all("div", attrs={"data-label": "Formula"}):
                parts = div.get("data-bbox", "").split()
                if len(parts) != 4:
                    continue
                bbox = [int(p) for p in parts]
                p_tag = div.find("p")
                items.append({"bbox": bbox, "html": str(p_tag) if p_tag else ""})
            out[r["image_path"]] = items
    return out


def replace_formulas(gt_html: str, dots_items: list[dict], min_containment: float = 0.5) -> tuple[str, int]:
    soup = BeautifulSoup(gt_html, "html.parser")
    formula_divs = soup.find_all("div", attrs={"data-label": "Formula"})

    gt_boxes = []
    for div in formula_divs:
        parts = div.get("data-bbox", "").split()
        gt_boxes.append([int(p) for p in parts] if len(parts) == 4 else None)

    assign = [-1] * len(formula_divs)
    for di, gt_box in enumerate(gt_boxes):
        if gt_box is None:
            continue
        best_i, best_c = -1, min_containment
        for i, item in enumerate(dots_items):
            c = containment(gt_box, item["bbox"])
            if c > best_c:
                best_c, best_i = c, i
        assign[di] = best_i

    groups: dict[int, list[int]] = {}
    for di, dots_i in enumerate(assign):
        if dots_i >= 0:
            groups.setdefault(dots_i, []).append(di)

    n_replaced = 0
    for dots_i, div_idxs in groups.items():
        first_div = formula_divs[div_idxs[0]]
        dots_item = dots_items[dots_i]
        bbox_str = " ".join(str(v) for v in dots_item["bbox"])
        new_div = BeautifulSoup(
            f'<div data-bbox="{bbox_str}" data-label="Formula">{dots_item["html"]}</div>',
            "html.parser",
        )
        first_div.replace_with(new_div)
        for di in div_idxs[1:]:
            formula_divs[di].decompose()
        n_replaced += len(div_idxs)

    return str(soup), n_replaced


def main():
    for split in SPLITS:
        dots_by_image = load_dots_formulas(split)
        path = E26_DIR / f"{split}.jsonl"
        rows = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

        n_pages = n_replaced_total = 0
        for r in rows:
            dots_items = dots_by_image.get(r["image_path"])
            if not dots_items:
                continue
            new_html, n_rep = replace_formulas(r["gt_html"], dots_items)
            r["gt_html"] = new_html
            n_pages += 1
            n_replaced_total += n_rep

        tmp_path = path.with_suffix(".jsonl.tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        tmp_path.replace(path)

        print(f"[{split}] 총 {len(rows)}행 중 수식페이지 {n_pages}장 in-place 교체 "
              f"(formula div 교체 {n_replaced_total}건) -> {path}")


if __name__ == "__main__":
    main()
