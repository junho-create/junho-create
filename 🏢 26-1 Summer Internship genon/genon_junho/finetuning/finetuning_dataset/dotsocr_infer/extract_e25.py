#!/usr/bin/env python3
"""combined_e25_28791 (unified_layout) 에서 Table 또는 Formula div 가 있는 페이지를 뽑아
`manifest_e25.jsonl` 을 만든다. `gt_table_audit/manifest.jsonl` 과 같은 스키마에
formula 관련 필드(n_formulas/formulas)를 추가했다.

bbox 는 원본 gt_html 의 data-bbox 그대로(1000 스케일, dataset 의 bbox_scale=1000 과 동일)
써서 run_infer_e25.py 가 norm_bbox 로 예측 bbox 와 나란히 비교할 수 있게 한다.
"""
from __future__ import annotations

import json
from pathlib import Path

from bs4 import BeautifulSoup

DATASET_DIR = Path("/home/jhyeo/finetuning/finetuning_dataset/combined_e25_28791")
OUT_PATH = Path(__file__).parent / "manifest_e25.jsonl"
SPLITS = ("train", "valid", "test")


def parse_bbox(div) -> list[int] | None:
    raw = div.get("data-bbox")
    if not raw:
        return None
    parts = raw.split()
    if len(parts) != 4:
        return None
    try:
        return [int(p) for p in parts]
    except ValueError:
        return None


def extract_page(gt_html: str) -> tuple[list[dict], list[dict]]:
    soup = BeautifulSoup(gt_html, "html.parser")
    tables, formulas = [], []
    for div in soup.find_all("div", attrs={"data-label": "Table"}):
        table_tag = div.find("table")
        tables.append({
            "bbox": parse_bbox(div),
            "html": str(table_tag) if table_tag else div.decode_contents().strip(),
        })
    for div in soup.find_all("div", attrs={"data-label": "Formula"}):
        formulas.append({
            "bbox": parse_bbox(div),
            "latex": div.get_text(" ", strip=True),
        })
    return tables, formulas


def main() -> int:
    n_pages = 0
    n_tables = 0
    n_formulas = 0
    with OUT_PATH.open("w", encoding="utf-8") as fout:
        for split in SPLITS:
            path = DATASET_DIR / f"{split}.jsonl"
            with path.open(encoding="utf-8") as f:
                for line_no, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    if r["prompt_style"] != "unified_layout":
                        continue
                    if "data-label=\"Table\"" not in r["gt_html"] and "data-label=\"Formula\"" not in r["gt_html"]:
                        continue
                    tables, formulas = extract_page(r["gt_html"])
                    if not tables and not formulas:
                        continue
                    for i, t in enumerate(tables, start=1):
                        t["index"] = i
                    for i, fm in enumerate(formulas, start=1):
                        fm["index"] = i
                    row = {
                        "key": f"{split}_{line_no:06d}",
                        "split": split,
                        "line_no": line_no,
                        "image_path": r["image_path"],
                        "n_tables": len(tables),
                        "tables": tables,
                        "n_formulas": len(formulas),
                        "formulas": formulas,
                    }
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                    n_pages += 1
                    n_tables += len(tables)
                    n_formulas += len(formulas)

    print(f"pages={n_pages}  tables={n_tables}  formulas={n_formulas}  -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
