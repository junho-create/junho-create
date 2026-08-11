#!/usr/bin/env python3
"""cell C 전용 데이터: combined_e25plus_tsr6598_noragged 에 "수식이 있는 layout
페이지"(dotsocr_infer/e25_out_formula_only 의 351장, train286/valid36/test29) 를
얹는다. 단, 그 페이지들의 GT 를 그대로 쓰지 않고 Formula div 내용만 dots.ocr
예측(latex)으로 바꾼다 — Table/Text/... 등 다른 div 는 GT 그대로 둔다.

매칭은 IoU 가 아니라 containment(포함비율) 로 한다: dots.ocr 은 수식 여러 개를
하나의 넓은 영역으로 합쳐서 검출하는 경우가 흔해서(실측: 표 하나가 GT 2개를
포함하는 사례 확인), "GT 박스 넓이 중 dots 박스와 겹치는 비율"이 임계값
이상이면 그 GT 는 해당 dots 예측에 속하는 것으로 본다. 같은 dots 박스에 속한
GT 가 여러 개면 전부 지우고 그 자리(첫 GT 위치)에 dots 예측 하나로 합쳐 넣는다.
매칭 안 된 GT Formula 는 원본 그대로 둔다. 매칭 안 된 dots.ocr 예측은 버린다
(새 div 를 추가하지 않음 — 위치가 GT 에 없는 걸 새로 넣지는 않는다).
"""
import json
import re
from pathlib import Path

from bs4 import BeautifulSoup

BASE_DIR = Path("/home/jhyeo/finetuning/finetuning_dataset/combined_e25plus_tsr6598_noragged")
FORMULA_DIR = Path("/home/jhyeo/finetuning/finetuning_dataset/dotsocr_infer/e25_out_formula_only")
OUT_DIR = Path("/home/jhyeo/finetuning/finetuning_dataset/combined_e25plus_tsr6598_noragged_plusformula")
SPLITS = ("train", "valid", "test")

BBOX_RE = re.compile(r'data-bbox="(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)"')


def area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def inter_area(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    return max(0, ix1 - ix0) * max(0, iy1 - iy0)


def containment(gt_box, dots_box):
    """GT 박스 넓이 중 dots 박스와 겹치는 비율. dots 가 GT 보다 넓어도(여러 GT 를
    합쳐 하나로 검출) 1.0 에 가깝게 나온다 — IoU 와 달리 dots 박스 크기에 안 깎인다."""
    a = area(gt_box)
    if a <= 0:
        return 0.0
    return inter_area(gt_box, dots_box) / a


def load_dots_formulas(split: str) -> dict[str, list[dict]]:
    """image_path -> [{bbox:[x0,y0,x1,y1], html: '<p>...</p>'}]"""
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
                bbox_attr = div.get("data-bbox", "")
                parts = bbox_attr.split()
                if len(parts) != 4:
                    continue
                bbox = [int(p) for p in parts]
                p_tag = div.find("p")
                items.append({"bbox": bbox, "html": str(p_tag) if p_tag else ""})
            out[r["image_path"]] = items
    return out


def replace_formulas(gt_html: str, dots_items: list[dict], min_containment: float = 0.5) -> tuple[str, int]:
    """dots box 별로, 그 안에 대부분(containment>=min_containment) 들어가는 GT div
    를 전부 묶는다. 묶인 GT div 들은 다 지우고, 첫 번째 GT div 자리에 dots 예측
    하나를 새 div 로 끼워 넣는다(원래 GT bbox 대신 dots 의 bbox 를 쓴다 — 여러 GT
    를 합친 넓은 영역이 실제 수식 범위이므로)."""
    soup = BeautifulSoup(gt_html, "html.parser")
    formula_divs = soup.find_all("div", attrs={"data-label": "Formula"})

    gt_boxes = []
    for div in formula_divs:
        bbox_attr = div.get("data-bbox", "")
        parts = bbox_attr.split()
        gt_boxes.append([int(p) for p in parts] if len(parts) == 4 else None)

    # 각 GT div 를 containment 가 가장 높은 dots item 에 배정(임계값 이상인 것만)
    assign = [-1] * len(formula_divs)  # div idx -> dots idx
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
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        dots_by_image = load_dots_formulas(split)

        base_rows = []
        with (BASE_DIR / f"{split}.jsonl").open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    base_rows.append(line)

        added_rows = []
        n_pages = n_replaced_total = 0
        gt_path = FORMULA_DIR / f"gt_{split}.jsonl"
        with gt_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                dots_items = dots_by_image.get(r["image_path"], [])
                new_html, n_rep = replace_formulas(r["gt_html"], dots_items)
                r["gt_html"] = new_html
                added_rows.append(json.dumps(r, ensure_ascii=False))
                n_pages += 1
                n_replaced_total += n_rep

        out_path = OUT_DIR / f"{split}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for line in base_rows + added_rows:
                f.write(line + "\n")

        print(f"[{split}] base={len(base_rows)} + formula_pages={n_pages} "
              f"(formula div 교체 {n_replaced_total}건) = {len(base_rows) + len(added_rows)} -> {out_path}")


if __name__ == "__main__":
    main()
