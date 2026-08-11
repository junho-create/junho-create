# generate.py
import os
import json
from typing import List, Dict, Any

from synth.config import SynthConfig
from synth.curriculum import default_curriculum
from synth.table_graph import build_table_graph
from synth.render import render_table
from synth.export_tflop import export_tflop_record_like_dataset


def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


def save_jsonl(path: str, records: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main(
    out_dir: str = "output",
    n_per_stage: int = 80,
):
    cfg = SynthConfig(seed=42)
    rng = cfg.rng()

    # ✅ 한글 폰트 (macOS 예시)
    cfg.font_paths = [
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    ]

    img_dir = os.path.join(out_dir, "train")
    ensure_dirs(img_dir)

    tfl_records: List[Dict[str, Any]] = []

    stages = default_curriculum()
    idx = 0

    for st in stages:
        cfg.rows_range = st.rows_range
        cfg.cols_range = st.cols_range
        cfg.merge_ratio = st.merge_ratio
        cfg.max_rowspan = st.max_rowspan
        cfg.max_colspan = st.max_colspan
        cfg.empty_cell_ratio = st.empty_cell_ratio

        for _ in range(n_per_stage):
            rows = rng.randint(*cfg.rows_range)
            cols = rng.randint(*cfg.cols_range)

            tg = build_table_graph(
                rows, cols, rng,
                merge_ratio=cfg.merge_ratio,
                max_rowspan=cfg.max_rowspan,
                max_colspan=cfg.max_colspan,
                enable_random_merge=True,
            )

            sample = render_table(tg, cfg, rng)

            fname = f"{idx:06d}.png"
            sample.image.save(os.path.join(img_dir, fname))

            tfl_records.append(
                export_tflop_record_like_dataset(
                    image_filename=f"{fname}",
                    tg=tg,
                    cell_polys=sample.cell_polys,
                    ocr_boxes=sample.ocr_boxes,
                    split="train",
                )
            )

            idx += 1

    jsonl_path = os.path.join(out_dir, "dataset_synth_tflop.jsonl")
    save_jsonl(jsonl_path, tfl_records)

    print(f"[OK] images: {idx} -> {img_dir}")
    print(f"[OK] labels: {jsonl_path}")


if __name__ == "__main__":
    main()
