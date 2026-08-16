# split_dataset.py
import os
import json
import shutil
import random
from typing import List, Dict


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_jsonl(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save_jsonl(path: str, records: List[Dict]):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def split_dataset(
    input_jsonl: str,
    image_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    seed: int = 42,
    move_files: bool = False,  # False면 copy, True면 move
):
    assert 0 < train_ratio < 1.0

    rng = random.Random(seed)

    records = load_jsonl(input_jsonl)
    print(f"[INFO] total records: {len(records)}")

    rng.shuffle(records)

    n_train = int(len(records) * train_ratio)
    train_records = records[:n_train]
    val_records = records[n_train:]

    print(f"[INFO] train: {len(train_records)}, val: {len(val_records)}")

    # output dirs
    train_img_dir = os.path.join(output_dir, "train", "images")
    val_img_dir = os.path.join(output_dir, "val", "images")
    ensure_dir(train_img_dir)
    ensure_dir(val_img_dir)

    # copy/move images
    def handle_images(records, dst_dir):
        for r in records:
            fname = r["file_name"]
            src = os.path.join(image_dir, fname)
            dst = os.path.join(dst_dir, fname)
            if not os.path.exists(src):
                raise FileNotFoundError(f"image not found: {src}")
            if move_files:
                shutil.move(src, dst)
            else:
                shutil.copy2(src, dst)

    handle_images(train_records, train_img_dir)
    handle_images(val_records, val_img_dir)

    # save jsonl
    save_jsonl(os.path.join(output_dir, "train", "dataset_train.jsonl"), train_records)
    save_jsonl(os.path.join(output_dir, "val", "dataset_val.jsonl"), val_records)

    print("[OK] dataset split complete")


if __name__ == "__main__":
    split_dataset(
        input_jsonl="output/dataset_synth_tflop.jsonl",
        image_dir="output/train",
        output_dir="output_split",
        train_ratio=0.8,
        seed=42,
        move_files=False,  # 원본 유지
    )
