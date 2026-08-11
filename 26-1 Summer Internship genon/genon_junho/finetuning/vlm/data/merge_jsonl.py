"""
JSONL 파일 병합 유틸리티

여러 JSONL 파일을 하나로 합치고, 중복 제거 및 셔플을 지원한다.

Usage:
    # 개별 파일 지정
    python -m data.merge_jsonl \
        --inputs a.jsonl b.jsonl c.jsonl \
        --output merged.jsonl \
        --dedup --shuffle --seed 42

    # 디렉토리 지정 (디렉토리 내 모든 *.jsonl 수집)
    python -m data.merge_jsonl \
        --input_dirs /path/to/dir1 /path/to/dir2 \
        --output merged.jsonl \
        --dedup --shuffle

    # 혼합 사용 (개별 파일 + 디렉토리)
    python -m data.merge_jsonl \
        --inputs a.jsonl \
        --input_dirs /path/to/dir \
        --output merged.jsonl \
        --recursive
"""

import argparse
import json
import random
import sys
from pathlib import Path


def collect_jsonl_from_dirs(dirs: list[str], recursive: bool = False) -> list[str]:
    """디렉토리에서 *.jsonl 파일 경로를 수집한다.

    Args:
        dirs: 탐색할 디렉토리 경로 목록
        recursive: True이면 서브디렉토리까지 재귀 탐색

    Returns:
        발견된 jsonl 파일 경로 목록 (정렬됨)
    """
    paths = []
    for d in dirs:
        dir_path = Path(d)
        if not dir_path.is_dir():
            print(f"Error: 디렉토리를 찾을 수 없습니다: {d}", file=sys.stderr)
            sys.exit(1)
        pattern = "**/*.jsonl" if recursive else "*.jsonl"
        found = sorted(dir_path.glob(pattern))
        if not found:
            print(f"Warning: {d} 에서 *.jsonl 파일을 찾지 못했습니다.", file=sys.stderr)
        paths.extend(str(p) for p in found)
    return paths


def merge_jsonl(
    input_paths: list[str],
    output_path: str,
    dedup: bool = False,
    shuffle: bool = False,
    seed: int = 42,
) -> dict:
    """JSONL 파일들을 병합한다.

    Returns:
        파일별 건수 및 최종 건수를 담은 통계 dict
    """
    records: list[dict] = []
    stats: dict[str, int] = {}

    for path in input_paths:
        count = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
                count += 1
        stats[path] = count

    total_before = len(records)

    # 중복 제거 (image_path 기준, 첫 번째 유지)
    if dedup:
        seen = set()
        unique = []
        for rec in records:
            key = rec.get("image_path", "")
            if key and key in seen:
                continue
            seen.add(key)
            unique.append(rec)
        records = unique

    dedup_removed = total_before - len(records)

    # 셔플
    if shuffle:
        random.seed(seed)
        random.shuffle(records)

    # 출력
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return {
        "per_file": stats,
        "total_before_dedup": total_before,
        "dedup_removed": dedup_removed,
        "final_count": len(records),
        "shuffled": shuffle,
    }


def main():
    parser = argparse.ArgumentParser(description="JSONL 파일 병합")
    parser.add_argument("--inputs", nargs="+", default=[], help="입력 JSONL 경로들")
    parser.add_argument("--input_dirs", nargs="+", default=[], help="*.jsonl을 수집할 디렉토리 경로들")
    parser.add_argument("--recursive", action="store_true", help="input_dirs 탐색 시 서브디렉토리 포함")
    parser.add_argument("--output", required=True, help="출력 JSONL 경로")
    parser.add_argument("--dedup", action="store_true", help="image_path 기준 중복 제거")
    parser.add_argument("--shuffle", action="store_true", help="결과 셔플")
    parser.add_argument("--seed", type=int, default=42, help="셔플 시드")
    args = parser.parse_args()

    if not args.inputs and not args.input_dirs:
        parser.error("--inputs 또는 --input_dirs 중 하나 이상을 지정해야 합니다.")

    # 디렉토리에서 jsonl 수집
    dir_files = collect_jsonl_from_dirs(args.input_dirs, recursive=args.recursive)

    all_inputs = list(args.inputs) + dir_files

    # 개별 파일 존재 확인
    for path in args.inputs:
        if not Path(path).exists():
            print(f"Error: 입력 파일을 찾을 수 없습니다: {path}", file=sys.stderr)
            sys.exit(1)

    if not all_inputs:
        print("Error: 병합할 JSONL 파일이 없습니다.", file=sys.stderr)
        sys.exit(1)

    if args.input_dirs:
        print(f"디렉토리에서 수집된 파일: {len(dir_files)}개")

    stats = merge_jsonl(
        input_paths=all_inputs,
        output_path=args.output,
        dedup=args.dedup,
        shuffle=args.shuffle,
        seed=args.seed,
    )

    # 통계 출력
    print("=" * 50)
    print("JSONL 병합 완료")
    print("=" * 50)
    for path, count in stats["per_file"].items():
        print(f"  {Path(path).name}: {count}건")
    print(f"  ---")
    print(f"  전체 합계: {stats['total_before_dedup']}건")
    if stats["dedup_removed"] > 0:
        print(f"  중복 제거: {stats['dedup_removed']}건")
    print(f"  최종 출력: {stats['final_count']}건")
    if stats["shuffled"]:
        print(f"  셔플: 적용 (seed={args.seed})")
    print(f"  출력 파일: {args.output}")
    print("=" * 50)


if __name__ == "__main__":
    main()
