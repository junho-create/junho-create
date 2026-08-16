#!/usr/bin/env python3
"""combined_e26_28520 에서, Formula 태그가 하나라도 있는 페이지의 gt_html **전체**를
dots.ocr 전체 페이지 예측(pred_blocks)으로 통째로 교체한다. 단:
  - image_path 가 reference_16886 출신이면 (사람이 라벨링한 신뢰 GT) 손대지 않는다.
  - dots.ocr 추론에 parse_note 가 있는 페이지(truncated/unparsable 등, 추론 자체가
    불완전했던 페이지)는 신뢰할 수 없는 예측이므로 제외하고 GT 를 그대로 둔다.

컨텐츠 wrapping 은 손으로 짜지 않고 `data.build_chandra_dataset` 의 공식 변환
함수(_content_for/_div)를 그대로 재사용한다 — 그 함수가 하는 일:
  - Table: <table>...</table> 원본 그대로(escape 안 함)
  - Picture: 빈 div
  - 나머지: HTML escape 후 <p>...</p> 로 감싸고 개행은 <br/>, 맨 앞 markdown
    heading(#, ##, ...)은 제거
직접 짠 버전은 escape 를 안 해서 GT 관례(예: 수식 안 `&` -> `&amp;`)와 어긋나고
markdown `#` 잔재도 안 지워졌다 — 이 스크립트는 그 두 버그를 고친 재실행이다.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/jhyeo/finetuning/vlm")
from data.build_chandra_dataset import _content_for, _div  # noqa: E402

E26_DIR = Path("/home/jhyeo/finetuning/finetuning_dataset/combined_e26_28520")
DOTS_DIR = Path("/home/jhyeo/finetuning/finetuning_dataset/dotsocr_infer/e25_out")
FORMULA_DIR = Path("/home/jhyeo/finetuning/finetuning_dataset/dotsocr_infer/e25_out_formula_only")
SPLITS = ("train", "valid", "test")


def load_dots_pages() -> dict[str, dict]:
    """image_path -> {"pred_blocks": [...], "parse_note": str|None}"""
    out = {}
    for split in SPLITS:
        path = DOTS_DIR / f"{split}.jsonl"
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                out[r["image_path"]] = {
                    "pred_blocks": r.get("pred_blocks", []),
                    "parse_note": r.get("parse_note"),
                }
    return out


def load_formula_page_set() -> set[str]:
    pages = set()
    for split in SPLITS:
        path = FORMULA_DIR / f"gt_{split}.jsonl"
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    pages.add(json.loads(line)["image_path"])
    return pages


def blocks_to_gt_html(blocks: list[dict]) -> str:
    divs = []
    for b in blocks:
        bbox = b.get("bbox_1000")
        if not bbox or len(bbox) != 4:
            continue
        label_raw = (b.get("label") or "").strip()
        if not label_raw:
            continue
        label = label_raw.capitalize()  # "page-header" -> "Page-header"
        content = _content_for(label, b.get("content") or "")
        divs.append(_div([int(v) for v in bbox], label, content))
    return "\n".join(divs)


def main():
    dots_pages = load_dots_pages()
    formula_pages = load_formula_page_set()
    print(f"수식 있는 페이지: {len(formula_pages)}")

    for split in SPLITS:
        path = E26_DIR / f"{split}.jsonl"
        rows = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

        n_replaced = n_skipped_ref = n_skipped_parse = n_no_dots = 0
        for r in rows:
            img = r["image_path"]
            if img not in formula_pages:
                continue
            if "reference_16886" in img:
                n_skipped_ref += 1
                continue
            page = dots_pages.get(img)
            if not page:
                n_no_dots += 1
                continue
            if page["parse_note"]:
                n_skipped_parse += 1
                continue
            r["gt_html"] = blocks_to_gt_html(page["pred_blocks"])
            n_replaced += 1

        tmp_path = path.with_suffix(".jsonl.tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        tmp_path.replace(path)

        print(f"[{split}] 총 {len(rows)}행 / dots로 전체교체 {n_replaced} / "
              f"reference_16886라 보존 {n_skipped_ref} / "
              f"parse_note 있어서 보존 {n_skipped_parse} / dots예측없음 {n_no_dots}")


if __name__ == "__main__":
    main()
