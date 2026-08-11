#!/usr/bin/env python3
"""dotsocr_infer/e25_out_formula_only/{split}.jsonl 에 있는 (수식 예측이 있었던) 페이지들의
image_path 를 combined_e25_28791 원본과 대조해서, run_infer.py 가 쓰는 manifest.jsonl 과
같은 스키마로 formula_manifest.jsonl 을 만든다.
"""
import json
from pathlib import Path

GT_DIR = Path("/home/jhyeo/finetuning/finetuning_dataset/combined_e25_28791")
FORMULA_DIR = Path("/home/jhyeo/finetuning/finetuning_dataset/dotsocr_infer/e25_out_formula_only")
OUT_PATH = Path(__file__).parent / "formula_manifest.jsonl"
SPLITS = ("train", "valid", "test")


def main():
    n_total = 0
    with OUT_PATH.open("w", encoding="utf-8") as fout:
        for split in SPLITS:
            wanted = set()
            with (FORMULA_DIR / f"{split}.jsonl").open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        wanted.add(json.loads(line)["image_path"])

            n_found = 0
            with (GT_DIR / f"{split}.jsonl").open(encoding="utf-8") as f:
                for line_no, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    if r["image_path"] not in wanted:
                        continue
                    fout.write(json.dumps({
                        "key": f"{split}_{line_no:06d}",
                        "split": split,
                        "line_no": line_no,
                        "image_path": r["image_path"],
                        "n_tables": 0,
                        "tables": [],
                        "static_flags": [],
                    }, ensure_ascii=False) + "\n")
                    n_found += 1
            print(f"[{split}] {n_found}/{len(wanted)}")
            n_total += n_found

    print(f"총 {n_total}건 -> {OUT_PATH}")


if __name__ == "__main__":
    main()
