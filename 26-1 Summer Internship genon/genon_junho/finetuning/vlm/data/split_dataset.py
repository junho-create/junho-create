"""
Train/Valid/Test 데이터셋 분리
- 복잡도 기준 층화 추출 (stratified split)
- span 비율 유지

Usage:
    # 기존 모드(train/eval)
    python -m data.split_dataset \
        --input data/processed/dataset.jsonl \
        --output_dir data/processed/ \
        --eval_ratio 0.1

    # count 지정 모드(train/valid/test)
    python -m data.split_dataset \
        --input data/processed/dataset.jsonl \
        --output_dir data/processed/ \
        --train_count 800 \
        --valid_count 100 \
        --test_count 100

생성 리포트:
    <output_dir>/split_complexity_report.json
"""

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_records(input_path: str):
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def get_complexity_label(record):
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        value = metadata.get("complexity")
        if value not in (None, ""):
            return str(value)

    for key in ("complexity", "table_complexity"):
        value = record.get(key)
        if value not in (None, ""):
            return str(value)

    return "unknown"


def group_indices_by_complexity(records):
    groups = defaultdict(list)
    for i, record in enumerate(records):
        complexity = get_complexity_label(record)
        groups[complexity].append(i)
    return groups


def write_jsonl(path, records, indices):
    with open(path, "w", encoding="utf-8") as f:
        for idx in indices:
            f.write(json.dumps(records[idx], ensure_ascii=False) + "\n")


def print_split_stats(split_name, records, indices):
    dist = Counter(get_complexity_label(records[i]) for i in indices)
    print(f"  {split_name}: {len(indices)}")
    print(f"    complexity: {dict(dist)}")


def build_complexity_summary(records, indices):
    total = len(indices)
    dist = Counter(get_complexity_label(records[i]) for i in indices)
    sorted_keys = sorted(dist.keys())
    counts = {k: dist[k] for k in sorted_keys}
    ratios = {k: (dist[k] / total if total > 0 else 0.0) for k in sorted_keys}
    return {
        "total": total,
        "complexity_counts": counts,
        "complexity_ratios": ratios,
    }


def write_complexity_report(
    *,
    input_path,
    output_dir,
    mode,
    records,
    split_indices_map,
    split_file_map,
    report_path=None,
):
    if report_path is None:
        report_path = os.path.join(output_dir, "split_complexity_report.json")

    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)

    all_indices = list(range(len(records)))
    report = {
        "input_path": input_path,
        "mode": mode,
        "total_records": len(records),
        "source_summary": build_complexity_summary(records, all_indices),
        "splits": {},
    }

    for split_name, indices in split_indices_map.items():
        report["splits"][split_name] = {
            "path": split_file_map[split_name],
            **build_complexity_summary(records, indices),
        }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"  Complexity report: {report_path}")
    return report_path


def stratified_take(groups, take_count, rng):
    total = sum(len(indices) for indices in groups.values())
    if take_count < 0 or take_count > total:
        raise ValueError(f"take_count({take_count}) must be in [0, {total}]")

    if take_count == 0:
        return [], {k: list(v) for k, v in groups.items()}

    if take_count == total:
        selected = []
        for indices in groups.values():
            selected.extend(indices)
        rng.shuffle(selected)
        return selected, {k: [] for k in groups}

    sizes = {k: len(v) for k, v in groups.items()}
    raw_targets = {k: (take_count * sizes[k] / total) for k in groups}
    alloc = {k: int(raw_targets[k]) for k in groups}

    remaining = take_count - sum(alloc.values())
    order = list(groups.keys())
    rng.shuffle(order)  # 동률일 때 랜덤 tie-break
    order.sort(key=lambda k: raw_targets[k] - alloc[k], reverse=True)

    for key in order:
        if remaining == 0:
            break
        if alloc[key] < sizes[key]:
            alloc[key] += 1
            remaining -= 1

    if remaining != 0:
        raise RuntimeError(f"Failed to allocate stratified sample exactly (remaining={remaining})")

    selected = []
    remaining_groups = {}
    for key, indices in groups.items():
        take_n = alloc[key]
        selected.extend(indices[:take_n])
        remaining_groups[key] = indices[take_n:]

    rng.shuffle(selected)
    return selected, remaining_groups


def stratified_split_with_counts(
    input_path: str,
    output_dir: str,
    train_count: int,
    valid_count: int,
    test_count: int,
    seed: int = 42,
    report_path: str = None,
):
    """
    복잡도 기반 층화 추출로 train/valid/test 분리(각 split 개수 지정).
    """
    rng = random.Random(seed)

    records = load_records(input_path)
    total_count = len(records)
    requested_total = train_count + valid_count + test_count
    if requested_total != total_count:
        raise ValueError(
            f"train+valid+test must equal total records ({requested_total} != {total_count})"
        )
    if min(train_count, valid_count, test_count) < 0:
        raise ValueError("train_count, valid_count, test_count must be >= 0")

    groups = group_indices_by_complexity(records)
    for indices in groups.values():
        rng.shuffle(indices)

    test_indices, remaining_groups = stratified_take(groups, test_count, rng)
    valid_indices, remaining_groups = stratified_take(remaining_groups, valid_count, rng)

    train_indices = []
    for indices in remaining_groups.values():
        train_indices.extend(indices)
    rng.shuffle(train_indices)

    if len(train_indices) != train_count:
        raise RuntimeError(f"train_count mismatch ({len(train_indices)} != {train_count})")

    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.jsonl")
    valid_path = os.path.join(output_dir, "valid.jsonl")
    test_path = os.path.join(output_dir, "test.jsonl")

    write_jsonl(train_path, records, train_indices)
    write_jsonl(valid_path, records, valid_indices)
    write_jsonl(test_path, records, test_indices)

    print("Split complete (count mode):")
    print(f"  train.jsonl -> {train_path}")
    print(f"  valid.jsonl -> {valid_path}")
    print(f"  test.jsonl  -> {test_path}")
    print_split_stats("Train", records, train_indices)
    print_split_stats("Valid", records, valid_indices)
    print_split_stats("Test", records, test_indices)
    write_complexity_report(
        input_path=input_path,
        output_dir=output_dir,
        mode="count",
        records=records,
        split_indices_map={
            "train": train_indices,
            "valid": valid_indices,
            "test": test_indices,
        },
        split_file_map={
            "train": train_path,
            "valid": valid_path,
            "test": test_path,
        },
        report_path=report_path,
    )


def stratified_split_with_ratio(
    input_path: str,
    output_dir: str,
    eval_ratio: float = 0.1,
    seed: int = 42,
    report_path: str = None,
):
    """
    복잡도 기반 층화 추출로 train/eval 분리(기존 호환 모드).
    """
    if not (0.0 < eval_ratio < 1.0):
        raise ValueError(f"eval_ratio must be in (0, 1), got {eval_ratio}")

    random.seed(seed)
    records = load_records(input_path)
    groups = group_indices_by_complexity(records)
    train_indices = []
    eval_indices = []

    for complexity, indices in groups.items():
        random.shuffle(indices)
        n_eval = max(1, int(len(indices) * eval_ratio))
        eval_indices.extend(indices[:n_eval])
        train_indices.extend(indices[n_eval:])

    # 셔플
    random.shuffle(train_indices)
    random.shuffle(eval_indices)

    # 저장
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train.jsonl")
    eval_path = os.path.join(output_dir, "eval.jsonl")

    write_jsonl(train_path, records, train_indices)
    write_jsonl(eval_path, records, eval_indices)

    # 통계 출력
    print("Split complete (ratio mode):")
    print(f"  Train: {len(train_indices)} → {train_path}")
    print(f"  Eval:  {len(eval_indices)} → {eval_path}")
    print_split_stats("Train", records, train_indices)
    print_split_stats("Eval", records, eval_indices)
    write_complexity_report(
        input_path=input_path,
        output_dir=output_dir,
        mode="ratio",
        records=records,
        split_indices_map={
            "train": train_indices,
            "eval": eval_indices,
        },
        split_file_map={
            "train": train_path,
            "eval": eval_path,
        },
        report_path=report_path,
    )


def main():
    parser = argparse.ArgumentParser(description="데이터셋 분리")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--eval_ratio",
        type=float,
        default=None,
        help="기존 train/eval 비율 분할 모드. count 모드 미사용 시 기본값 0.1",
    )
    parser.add_argument("--train_count", type=int, default=None)
    parser.add_argument("--valid_count", type=int, default=None)
    parser.add_argument("--test_count", type=int, default=None)
    parser.add_argument(
        "--report_path",
        default=None,
        help="복잡도 리포트 저장 경로. 기본: <output_dir>/split_complexity_report.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    count_args = [args.train_count, args.valid_count, args.test_count]
    use_count_mode = any(v is not None for v in count_args)

    if use_count_mode:
        if not all(v is not None for v in count_args):
            parser.error("--train_count, --valid_count, --test_count must all be provided together")
        stratified_split_with_counts(
            input_path=args.input,
            output_dir=args.output_dir,
            train_count=args.train_count,
            valid_count=args.valid_count,
            test_count=args.test_count,
            seed=args.seed,
            report_path=args.report_path,
        )
    else:
        eval_ratio = 0.1 if args.eval_ratio is None else args.eval_ratio
        stratified_split_with_ratio(
            input_path=args.input,
            output_dir=args.output_dir,
            eval_ratio=eval_ratio,
            seed=args.seed,
            report_path=args.report_path,
        )


if __name__ == "__main__":
    main()
