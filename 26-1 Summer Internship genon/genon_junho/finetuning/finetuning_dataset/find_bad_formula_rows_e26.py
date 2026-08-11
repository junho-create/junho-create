#!/usr/bin/env python3
"""combined_e26 에 아직 남아있는 "수식이 이상한" 행을 뽑아 fix_tmp.json 으로 낸다.

app.py 뷰어에 그대로 넣을 수 있게 원본 행(image_path + gt_html) 그대로 담는다.
판정: Formula div 의 텍스트가 비었거나 `$` 로 시작하지 않으면 이상한 것으로 본다.
"""
import json
import sys
from pathlib import Path

from bs4 import BeautifulSoup

E26 = Path(sys.argv[1] if len(sys.argv) > 1
           else "/home/jhyeo/finetuning/finetuning_dataset/combined_e26_28484")
DOTS_DIR = Path("/home/jhyeo/finetuning/finetuning_dataset/dotsocr_infer/e25_out")
FORMULA_DIR = Path("/home/jhyeo/finetuning/finetuning_dataset/dotsocr_infer/e25_out_formula_only")
SPLITS = ("train", "valid", "test")


def dots_replaced_pages() -> set[str]:
    """build_e26.py 가 실제로 dots.ocr 예측으로 덮어쓴 페이지 집합(같은 조건 재현)."""
    parse_ok = {}
    for split in SPLITS:
        with (DOTS_DIR / f"{split}.jsonl").open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    parse_ok[r["image_path"]] = not r.get("parse_note")
    formula_pages = set()
    for split in SPLITS:
        with (FORMULA_DIR / f"gt_{split}.jsonl").open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    formula_pages.add(json.loads(line)["image_path"])
    return {p for p in formula_pages
            if "reference_16886" not in p and parse_ok.get(p, False)}


def bad_reasons(gt_html: str) -> list[str]:
    soup = BeautifulSoup(gt_html, "html.parser")
    out = []
    for div in soup.find_all("div", attrs={"data-label": "Formula"}):
        text = div.get_text().strip()
        if not text:
            out.append("empty")
        elif not text.startswith("$"):
            out.append(f"no_leading_dollar: {text[:80]}")
    return out


def main():
    replaced = dots_replaced_pages()
    rows = []
    n_formula_rows = 0
    for split in SPLITS:
        with (E26 / f"{split}.jsonl").open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if 'data-label="Formula"' not in r["gt_html"]:
                    continue
                n_formula_rows += 1
                reasons = bad_reasons(r["gt_html"])
                if reasons:
                    rows.append({
                        **r, "_split": split, "_bad": reasons,
                        "_source": "dots" if r["image_path"] in replaced else "original_gt",
                    })

    from_dots = [r for r in rows if r["_source"] == "dots"]
    from_gt = [r for r in rows if r["_source"] == "original_gt"]

    # 우리가 바꾼 것(dots)을 앞에 오게 정렬 — 검수 우선순위
    rows.sort(key=lambda r: 0 if r["_source"] == "dots" else 1)
    out_path = E26 / "fix_tmp.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print(f"수식 있는 행 {n_formula_rows} 중 이상한 행 {len(rows)}")
    print(f"  - dots.ocr 로 교체한 행:  {len(from_dots)}  <- 우리가 만든 것")
    print(f"  - 원본 GT 그대로인 행:   {len(from_gt)}  <- 원래부터 그랬음")
    print(f"-> {out_path}")
    print()
    for r in from_dots:
        print(f"  [dots/{r['_split']}] {Path(r['image_path']).name}")
        for b in r["_bad"]:
            print(f"      {b}")


if __name__ == "__main__":
    main()
