#!/usr/bin/env python3
"""
V9.x pilot: eval jsonl 의 prompt_style 을 complex variant 로 override.

목적:
    SFT/V8.8 에서 학습된 모델이 multi-header / row-group / long-table 케이스에서
    body cell drift (한 칸 밀림) 패턴을 보이는 문제 → prompt 에 explicit
    body alignment guidance 를 추가해서 inference-time 으로 개선 시도.

검출 휴리스틱 (top header 또는 left row group 또는 long table):
    1. Top multi-header   : OCR text 가 상단 영역 (image 위 25%) 에서 2개 이상 y-band
    2. Left row group     : 좌측 첫 컬럼 (image 좌 15%) 에 큰 vertical cell
                            (텍스트 사이 큰 vertical gap)
    3. Long table         : OCR text 의 추정 행 수 ≥ 10
    4. Many cells         : OCR text 개수 ≥ 30 (proxy for table size)

이 중 하나라도 해당하면 complex prompt 로 override.
False positive 는 안전 (extra instruction 이 simple 표에 해롭지 않음).
False negative 가 위험 → conservative 로 검출.

Usage:
    # 분석만 (분포 출력)
    python -m utils.inject_complex_prompt \\
        --input data/eval.jsonl \\
        --stats-only

    # prompt_style override + 새 jsonl 저장
    python -m utils.inject_complex_prompt \\
        --input data/eval.jsonl \\
        --output data/eval_complex_prompt.jsonl

    # 강제로 모든 sample 에 complex prompt 적용 (검출 무시)
    python -m utils.inject_complex_prompt \\
        --input data/eval.jsonl \\
        --output data/eval_all_complex.jsonl \\
        --force-all
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


# =============================================================================
# 검출 휴리스틱
# =============================================================================


def _bbox_y_center(bbox) -> float:
    return (float(bbox[1]) + float(bbox[3])) / 2.0


def _bbox_x_center(bbox) -> float:
    return (float(bbox[0]) + float(bbox[2])) / 2.0


def _count_distinct_y_bands(y_centers: list[float], gap_threshold: float = 15.0) -> int:
    """y-center 들을 정렬 후 gap 으로 분리해서 distinct band 수 계산."""
    if not y_centers:
        return 0
    sorted_y = sorted(y_centers)
    n_bands = 1
    for i in range(1, len(sorted_y)):
        if sorted_y[i] - sorted_y[i - 1] > gap_threshold:
            n_bands += 1
    return n_bands


def detect_complex_structure(
    ocr_info: list[dict] | None,
    bbox_scale: int = 1024,
    top_band_ratio: float = 0.25,
    left_band_ratio: float = 0.15,
    top_y_gap: float = 20.0,
    left_x_gap: float = 30.0,
    long_table_rows: int = 10,
    many_cells: int = 30,
) -> tuple[bool, list[str]]:
    """OCR 정보 기반 complex 구조 검출.

    Returns:
        (is_complex, reasons) — reasons 는 검출 사유 라벨 리스트
    """
    reasons: list[str] = []

    if not ocr_info or not isinstance(ocr_info, list):
        return False, reasons

    bboxes = []
    for item in ocr_info:
        if not isinstance(item, dict):
            continue
        bbox = item.get("bbox")
        if isinstance(bbox, list) and len(bbox) >= 4:
            bboxes.append(bbox)

    if not bboxes:
        return False, reasons

    n_total = len(bboxes)

    # 1. Many cells (proxy for table size)
    if n_total >= many_cells:
        reasons.append("many_cells")

    # 2. Top multi-header — 상단 영역의 y-band 수
    top_threshold = bbox_scale * top_band_ratio
    top_bboxes = [b for b in bboxes if _bbox_y_center(b) < top_threshold]
    if len(top_bboxes) >= 4:
        n_bands = _count_distinct_y_bands(
            [_bbox_y_center(b) for b in top_bboxes], gap_threshold=top_y_gap
        )
        if n_bands >= 2:
            reasons.append("top_multi_header")

    # 3. Left row group — 좌측 첫 컬럼에 큰 vertical cell
    left_threshold = bbox_scale * left_band_ratio
    left_bboxes = [b for b in bboxes if _bbox_x_center(b) < left_threshold]
    if len(left_bboxes) >= 2:
        y_centers = sorted(_bbox_y_center(b) for b in left_bboxes)
        gaps = [y_centers[i] - y_centers[i - 1] for i in range(1, len(y_centers))]
        if gaps and max(gaps) > left_x_gap * 2:
            reasons.append("left_row_group")

    # 4. Long table — y-band 수로 추정
    n_y_bands = _count_distinct_y_bands(
        [_bbox_y_center(b) for b in bboxes], gap_threshold=top_y_gap
    )
    if n_y_bands >= long_table_rows:
        reasons.append("long_table")

    is_complex = len(reasons) > 0
    return is_complex, reasons


# =============================================================================
# Prompt style override
# =============================================================================


def map_to_complex_prompt(prompt_style: str) -> str:
    """기존 prompt_style 을 대응되는 complex variant 로 매핑."""
    s = (prompt_style or "").strip().lower()
    if s == "chandra_table_with_ocr":
        return "chandra_table_complex_with_ocr"
    if s == "chandra_table_without_ocr":
        return "chandra_table_complex_without_ocr"
    return "chandra_table_complex_with_ocr"


# =============================================================================
# Main
# =============================================================================


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True, help="입력 eval jsonl")
    ap.add_argument("--output", type=Path, default=None,
                    help="출력 jsonl (생략 시 stats-only 와 동일)")
    ap.add_argument("--stats-only", action="store_true",
                    help="통계만 출력, 파일 저장 안 함")
    ap.add_argument("--force-all", action="store_true",
                    help="검출 무시하고 모든 sample 에 complex prompt 적용")
    ap.add_argument("--bbox-scale", type=int, default=1024,
                    help="bbox 좌표 정규화 기준 (default 1024)")
    args = ap.parse_args()

    if not args.input.exists():
        print(f"[ERR] {args.input} 없음", file=sys.stderr)
        sys.exit(1)

    save = args.output is not None and not args.stats_only

    n_total = 0
    n_complex = 0
    reason_counts: Counter = Counter()
    style_before: Counter = Counter()
    style_after: Counter = Counter()
    bucket_complex_distribution: dict[str, int] = {}

    out_records = []

    with args.input.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("_meta"):
                if save:
                    out_records.append(r)
                continue

            n_total += 1
            old_style = r.get("prompt_style", "default")
            style_before[old_style] += 1

            if args.force_all:
                is_complex, reasons = True, ["force_all"]
            else:
                ocr = r.get("ocr_info")
                bbox_scale = r.get("bbox_scale", args.bbox_scale)
                is_complex, reasons = detect_complex_structure(
                    ocr, bbox_scale=bbox_scale,
                )

            if is_complex:
                n_complex += 1
                for rs in reasons:
                    reason_counts[rs] += 1

                bucket = r.get("complexity") or r.get("bucket") or "unknown"
                bucket_complex_distribution[bucket] = (
                    bucket_complex_distribution.get(bucket, 0) + 1
                )

                new_style = map_to_complex_prompt(old_style)
                r["prompt_style"] = new_style
                r["_complex_detection"] = {
                    "is_complex": True,
                    "reasons": reasons,
                    "original_prompt_style": old_style,
                }
                style_after[new_style] += 1
            else:
                style_after[old_style] += 1

            if save:
                out_records.append(r)

    print(f"\n[Total]   {n_total} samples")
    print(f"[Complex] {n_complex} ({100*n_complex/max(1,n_total):.1f}%)")
    print()
    print(f"[Reason 별 검출 수 (중복 가능)]")
    for k, v in sorted(reason_counts.items(), key=lambda x: -x[1]):
        print(f"  {k:25s}: {v}")
    print()
    print(f"[검출된 complex 의 bucket 분포]")
    for k, v in sorted(bucket_complex_distribution.items(), key=lambda x: -x[1]):
        print(f"  {k:20s}: {v}")
    print()
    print(f"[Prompt style 변환]")
    print(f"  Before: {dict(style_before)}")
    print(f"  After:  {dict(style_after)}")

    if save:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as f:
            for r in out_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n[Saved] {args.output}")


if __name__ == "__main__":
    main()
