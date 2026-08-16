#!/usr/bin/env python3
"""e25_out_formula_only/{split}.jsonl 에 들어있는 image_path 들만, combined_e25_28791
원본(GT, 전체 필드)에서 그대로 뽑아 같은 디렉토리에 gt_{split}.jsonl 로 저장한다.
"""
import json
from pathlib import Path

GT_DIR = Path("/home/jhyeo/finetuning/finetuning_dataset/combined_e25_28791")
FORMULA_DIR = Path("/home/jhyeo/finetuning/finetuning_dataset/dotsocr_infer/e25_out_formula_only")
SPLITS = ("train", "valid", "test")


def main():
    for split in SPLITS:
        wanted = set()
        with (FORMULA_DIR / f"{split}.jsonl").open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    wanted.add(json.loads(line)["image_path"])

        out_path = FORMULA_DIR / f"gt_{split}.jsonl"
        n_found = 0
        with (GT_DIR / f"{split}.jsonl").open(encoding="utf-8") as f, out_path.open("w", encoding="utf-8") as g:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r["image_path"] in wanted:
                    g.write(json.dumps(r, ensure_ascii=False) + "\n")
                    n_found += 1

        print(f"[{split}] 찾을 이미지 {len(wanted)} / 매칭된 GT 행 {n_found} -> {out_path}")
        if n_found != len(wanted):
            print(f"    경고: 개수 불일치 (매칭 안 된 이미지 {len(wanted) - n_found}개 있을 수 있음)")


if __name__ == "__main__":
    main()
