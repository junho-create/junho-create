"""
데이터셋 검증 및 통계 분석

검증 항목:
- HTML 파싱 가능 여부
- 이미지 파일 존재 여부
- 대화 포맷 유효성
- Thinking chain 존재 여부
- Span 분포 통계

Usage:
    python -m data.validate_dataset --input data/processed/dataset.jsonl --image_dir data/images/
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.html_utils import parse_html_table, normalize_html
from utils.span_analyzer import analyze_span, DatasetSpanProfile


def validate_record(record: dict, check_images: bool = True) -> dict:
    """
    단일 레코드를 검증한다.

    Returns:
        {"valid": bool, "errors": [str], "warnings": [str]}
    """
    result = {"valid": True, "errors": [], "warnings": []}
    messages = record.get("messages", [])

    # 1. 메시지 구조 확인
    roles = [m.get("role") for m in messages]
    if "system" not in roles:
        result["warnings"].append("No system message")
    if "user" not in roles:
        result["errors"].append("No user message")
        result["valid"] = False
    if "assistant" not in roles:
        result["errors"].append("No assistant message")
        result["valid"] = False

    # 2. User 메시지에 이미지 확인
    user_msg = next((m for m in messages if m["role"] == "user"), None)
    if user_msg:
        content = user_msg.get("content", "")
        if isinstance(content, list):
            has_image = any(
                item.get("type") == "image" for item in content
            )
            has_text = any(
                item.get("type") == "text" for item in content
            )
            if not has_image:
                result["errors"].append("No image in user message")
                result["valid"] = False
            if not has_text:
                result["warnings"].append("No text instruction in user message")

            # 이미지 파일 존재 확인
            if check_images and has_image:
                for item in content:
                    if item.get("type") == "image":
                        img_path = item.get("image", "")
                        if img_path and not os.path.exists(img_path):
                            result["errors"].append(f"Image not found: {img_path}")
                            result["valid"] = False
                        elif img_path and os.path.exists(img_path):
                            # 이미지 읽기 가능 확인
                            try:
                                img = Image.open(img_path)
                                w, h = img.size
                                if w < 50 or h < 50:
                                    result["warnings"].append(
                                        f"Very small image: {w}×{h}"
                                    )
                            except Exception as e:
                                result["errors"].append(
                                    f"Cannot read image: {e}"
                                )
                                result["valid"] = False

    # 3. Assistant 메시지 검증
    assistant_msg = next(
        (m for m in messages if m["role"] == "assistant"), None
    )
    if assistant_msg:
        content = assistant_msg.get("content", "")

        # Thinking chain 확인
        has_thinking = "<think>" in content and "</think>" in content
        if not has_thinking:
            result["warnings"].append("No thinking chain in assistant response")

        # HTML 추출 및 검증
        if has_thinking:
            html_part = content.split("</think>")[-1].strip()
        else:
            html_part = content.strip()

        if not html_part:
            result["errors"].append("Empty HTML in assistant response")
            result["valid"] = False
        else:
            try:
                structure = parse_html_table(html_part)
                if structure.num_rows == 0:
                    result["errors"].append("Parsed table has 0 rows")
                    result["valid"] = False
                if structure.num_cols == 0:
                    result["errors"].append("Parsed table has 0 columns")
                    result["valid"] = False
            except Exception as e:
                result["errors"].append(f"HTML parse error: {e}")
                result["valid"] = False

    return result


def validate_dataset(
    input_path: str,
    check_images: bool = True,
    verbose: bool = False,
) -> dict:
    """
    전체 데이터셋을 검증하고 통계를 반환한다.

    Returns:
        {
            "total": int,
            "valid": int,
            "invalid": int,
            "error_types": Counter,
            "warning_types": Counter,
            "span_profile": DatasetSpanProfile,
            "seq_length_stats": dict,
        }
    """
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    total = len(records)
    valid_count = 0
    invalid_count = 0
    error_counter = Counter()
    warning_counter = Counter()
    html_list = []
    seq_lengths = []
    complexity_counter = Counter()

    for i, record in enumerate(records):
        result = validate_record(record, check_images=check_images)

        if result["valid"]:
            valid_count += 1
        else:
            invalid_count += 1
            if verbose:
                print(f"  Invalid record {i}: {result['errors']}")

        for err in result["errors"]:
            error_counter[err] += 1
        for warn in result["warnings"]:
            warning_counter[warn] += 1

        # HTML 추출 (span 분석용)
        for msg in record.get("messages", []):
            if msg["role"] == "assistant":
                content = msg.get("content", "")
                if "</think>" in content:
                    html = content.split("</think>")[-1].strip()
                else:
                    html = content.strip()
                if html:
                    html_list.append(html)

                # 시퀀스 길이 추정
                seq_lengths.append(len(content))

        # 복잡도
        metadata = record.get("metadata", {})
        if "complexity" in metadata:
            complexity_counter[metadata["complexity"]] += 1

    # Span 프로파일
    from utils.span_analyzer import analyze_dataset
    span_profile = analyze_dataset(html_list)

    # 시퀀스 길이 통계
    seq_stats = {}
    if seq_lengths:
        seq_lengths.sort()
        seq_stats = {
            "min": seq_lengths[0],
            "max": seq_lengths[-1],
            "median": seq_lengths[len(seq_lengths) // 2],
            "mean": sum(seq_lengths) / len(seq_lengths),
            "p95": seq_lengths[int(len(seq_lengths) * 0.95)],
        }

    return {
        "total": total,
        "valid": valid_count,
        "invalid": invalid_count,
        "error_types": dict(error_counter.most_common()),
        "warning_types": dict(warning_counter.most_common()),
        "complexity_distribution": dict(complexity_counter),
        "span_profile": span_profile,
        "seq_length_stats": seq_stats,
    }


def print_report(stats: dict):
    """검증 결과 리포트 출력."""
    print("=" * 60)
    print("DATASET VALIDATION REPORT")
    print("=" * 60)

    print(f"\nTotal records:   {stats['total']}")
    print(f"Valid:           {stats['valid']} ({stats['valid']/max(stats['total'],1):.1%})")
    print(f"Invalid:         {stats['invalid']}")

    if stats["error_types"]:
        print("\nErrors:")
        for err, count in stats["error_types"].items():
            print(f"  [{count:>4}] {err}")

    if stats["warning_types"]:
        print("\nWarnings:")
        for warn, count in stats["warning_types"].items():
            print(f"  [{count:>4}] {warn}")

    if stats["complexity_distribution"]:
        print("\nComplexity Distribution (from metadata):")
        for comp, count in sorted(stats["complexity_distribution"].items()):
            print(f"  {comp:<10}: {count}")

    print(f"\n{stats['span_profile'].summary()}")

    if stats["seq_length_stats"]:
        s = stats["seq_length_stats"]
        print(f"\nSequence Length (chars):")
        print(f"  Min:    {s['min']:>8,}")
        print(f"  Median: {s['median']:>8,}")
        print(f"  Mean:   {s['mean']:>8,.0f}")
        print(f"  P95:    {s['p95']:>8,}")
        print(f"  Max:    {s['max']:>8,}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="데이터셋 검증 및 통계")
    parser.add_argument("--input", required=True, help="입력 JSONL")
    parser.add_argument("--no-check-images", action="store_true", help="이미지 파일 검사 스킵")
    parser.add_argument("--verbose", action="store_true", help="상세 출력")
    args = parser.parse_args()

    stats = validate_dataset(
        args.input,
        check_images=not args.no_check_images,
        verbose=args.verbose,
    )
    print_report(stats)


if __name__ == "__main__":
    main()
