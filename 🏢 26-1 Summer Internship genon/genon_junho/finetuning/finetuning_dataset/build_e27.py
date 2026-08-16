#!/usr/bin/env python3
"""combined_e27 데이터셋 = combined_e26_28484 + arxiv_formula/dataset_formula_crop,
합친 뒤 TARGET_TOTAL 근처로 무작위 다운샘플.

스키마가 이미 동일하다(image_path/gt_html/prompt_style/bbox_scale/output_format/ocr_info,
crop 쪽만 meta 추가 필드가 더 있음 — 학습 로더는 안 쓰는 필드라 문제 없음). 변환 없이
합친 뒤 split별 비율(원래 37375:4465:3722)을 유지한 채 다운샘플한다 — e26:crop 비율
(28484:17078 ≈ 62.5:37.5)도 균일 샘플링이라 기대값으로 그대로 보존된다.

prompt_style 호환 확인(2026-08-10): e26은 unified_layout/unified_table_with_ocr/
chandra_with_ocr, crop은 chandra_no_ocr 뿐인데 utils/prompt_templates.py의
normalize_prompt_style()이 unified_layout -> chandra_no_ocr로 정규화하므로 실제
canonical 값 기준으로는 이미 같은 체계다.
"""
import json
import random
import shutil
from pathlib import Path

SPLITS = ("train", "valid", "test")
E26_DIR = Path("/home/jhyeo/finetuning/finetuning_dataset/combined_e26_28484")
CROP_DIR = Path("/home/jhyeo/finetuning/finetuning_dataset/arxiv_formula/dataset_formula_crop")
OUT_ROOT = Path("/home/jhyeo/finetuning/finetuning_dataset")

TARGET_TOTAL = 30000
SEED = 20260810


def load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    rng = random.Random(SEED)
    merged = {}
    n_e26_by_split = {}
    for split in SPLITS:
        a = load(E26_DIR / f"{split}.jsonl")
        b = load(CROP_DIR / f"{split}.jsonl")
        merged[split] = a + b
        n_e26_by_split[split] = len(a)

    full_total = sum(len(v) for v in merged.values())
    scale = TARGET_TOTAL / full_total

    result = {}
    per_split = {}
    for split in SPLITS:
        pool = merged[split]
        n_e26 = n_e26_by_split[split]
        target_n = round(len(pool) * scale)
        idx = rng.sample(range(len(pool)), target_n)
        idx.sort()
        sampled = [pool[i] for i in idx]
        n_e26_kept = sum(1 for i in idx if i < n_e26)
        n_crop_kept = target_n - n_e26_kept
        result[split] = sampled
        per_split[split] = (len(pool), target_n, n_e26_kept, n_crop_kept)

    total = sum(len(v) for v in result.values())
    out_dir = OUT_ROOT / f"combined_e27_{total}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    for split in SPLITS:
        with (out_dir / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for r in result[split]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"{'split':<7} {'원본':>7} {'샘플':>7} {'e26':>7} {'crop':>7}")
    for split in SPLITS:
        full, tgt, ke26, kcrop = per_split[split]
        print(f"{split:<7} {full:>7} {tgt:>7} {ke26:>7} {kcrop:>7}")
    print(f"{'합계':<7} {full_total:>7} {total:>7} "
          f"{sum(v[2] for v in per_split.values()):>7} {sum(v[3] for v in per_split.values()):>7}")
    print(f"-> {out_dir}")
    return out_dir


if __name__ == "__main__":
    main()
