#!/usr/bin/env python3
"""e25_out (dots.ocr 예측)에서 Formula 만 뽑아 gt_html 스키마로 재포장한다.

다른 라벨(Table/Text/Picture/...)은 다 버리고 Formula div만 남긴다. bbox는
bbox_1000(정규화된 1000 스케일)을 쓰고, latex 안의 개행(\\n)은 <br/>로 바꿔서
combined_e25_28791_before_tablefix 의 GT Formula div와 같은 모양
(`<p>$$<br/>...<br/>$$</p>`)으로 맞춘다 - 같은 렌더러로 그대로 열어볼 수 있게.

파일명/스키마는 combined_e25_28791_before_tablefix/{split}.jsonl 을 따른다
(image_path + gt_html).
"""
import json
from pathlib import Path

SRC_DIR = Path("/home/jhyeo/finetuning/finetuning_dataset/dotsocr_infer/e25_out")
OUT_DIR = Path("/home/jhyeo/finetuning/finetuning_dataset/dotsocr_infer/e25_out_formula_only")
SPLITS = ("train", "valid", "test")


def formula_div(f: dict) -> str | None:
    latex = (f.get("latex") or "").strip()
    if not latex:
        return None
    bbox = f.get("bbox_1000")
    bbox_str = " ".join(str(v) for v in bbox) if bbox else "0 0 0 0"
    body = latex.replace("\r\n", "\n").replace("\n", "<br/>")
    return f'<div data-bbox="{bbox_str}" data-label="Formula">\n<p>{body}</p>\n</div>'


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        src = SRC_DIR / f"{split}.jsonl"
        out = OUT_DIR / f"{split}.jsonl"
        n_in = n_out = n_formulas = 0
        with src.open(encoding="utf-8") as f, out.open("w", encoding="utf-8") as g:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                n_in += 1
                r = json.loads(line)
                divs = [formula_div(fm) for fm in r.get("pred_formulas", [])]
                divs = [d for d in divs if d]
                if not divs:
                    continue
                g.write(json.dumps({
                    "image_path": r["image_path"],
                    "gt_html": "\n".join(divs),
                    "bbox_scale": 1000,
                }, ensure_ascii=False) + "\n")
                n_out += 1
                n_formulas += len(divs)
        print(f"[{split}] 입력 {n_in}행 -> 출력 {n_out}행 (수식 {n_formulas}개) -> {out}")


if __name__ == "__main__":
    main()
