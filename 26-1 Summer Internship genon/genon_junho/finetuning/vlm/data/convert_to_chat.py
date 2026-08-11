"""
TSR 라벨 데이터 → 대화형 포맷 변환

지원 입력 포맷:
1. JSONL: {"image": "path", "html": "<table>...</table>"}
2. PubTabNet 형식: {"filename": "...", "html": {"structure": {"tokens": [...]}}}
3. 디렉토리: image + .html 파일 쌍

Usage:
    python -m data.convert_to_chat \
        --input ./raw_data/labels.jsonl \
        --image_dir ./raw_data/images/ \
        --output ./data/processed/dataset.jsonl \
        --format jsonl
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.html_utils import (
    parse_html_table,
    normalize_html,
    extract_spans_from_html,
    get_table_dimensions,
)
from utils.prompt_templates import build_thinking_chain, build_chat_messages
from utils.span_analyzer import analyze_span


def convert_pubtabnet_tokens(tokens: list[str]) -> str:
    """PubTabNet의 토큰 리스트를 HTML 문자열로 변환."""
    return "".join(tokens)


def load_raw_data(
    input_path: str,
    image_dir: Optional[str] = None,
    format: str = "jsonl",
) -> list[dict]:
    """
    원시 데이터를 로드한다.

    Returns:
        [{"image_path": str, "html": str}, ...]
    """
    records = []

    if format == "jsonl":
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)

                # 다양한 필드명 대응
                image = (
                    item.get("image")
                    or item.get("filename")
                    or item.get("image_path")
                    or item.get("img")
                )

                # HTML 추출
                html = item.get("html", "")
                if isinstance(html, dict):
                    # PubTabNet 형식
                    tokens = html.get("structure", {}).get("tokens", [])
                    html = convert_pubtabnet_tokens(tokens)
                    # PubTabNet은 셀 내용이 토큰에 없을 수 있음
                    cells = html.get("cells", [])  # noqa

                if not image or not html:
                    continue

                # 이미지 경로 조합
                if image_dir and not os.path.isabs(image):
                    image = os.path.join(image_dir, image)

                records.append({"image_path": image, "html": html})

    elif format == "directory":
        # 디렉토리에서 image + .html 쌍 찾기
        input_dir = Path(input_path)
        image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

        for img_file in sorted(input_dir.iterdir()):
            if img_file.suffix.lower() not in image_extensions:
                continue
            html_file = img_file.with_suffix(".html")
            if html_file.exists():
                html = html_file.read_text(encoding="utf-8").strip()
                records.append({
                    "image_path": str(img_file),
                    "html": html,
                })

    elif format == "csv":
        import csv
        with open(input_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image = row.get("image", row.get("image_path", ""))
                html = row.get("html", "")
                if image_dir and not os.path.isabs(image):
                    image = os.path.join(image_dir, image)
                if image and html:
                    records.append({"image_path": image, "html": html})

    print(f"Loaded {len(records)} records from {input_path} (format={format})")
    return records


def convert_single_record(
    record: dict,
    include_thinking: bool = True,
    prompt_idx: Optional[int] = None,
) -> Optional[dict]:
    """
    단일 레코드를 대화형 포맷으로 변환.

    Args:
        record: {"image_path": str, "html": str}
        include_thinking: thinking chain 포함 여부
        prompt_idx: 프롬프트 변형 인덱스 (None이면 랜덤)

    Returns:
        {"messages": [...], "metadata": {...}} or None (실패 시)
    """
    try:
        html = record["html"]
        image_path = record["image_path"]

        # HTML 유효성 검증
        structure = parse_html_table(html)
        if structure.num_rows == 0 or structure.num_cols == 0:
            return None

        # Thinking chain 생성
        thinking = ""
        if include_thinking:
            spans = extract_spans_from_html(html)
            thinking = build_thinking_chain(
                num_rows=structure.num_rows,
                num_cols=structure.num_cols,
                spans=spans,
                header_rows=structure.header_rows,
            )

        # 대화 메시지 구성
        normalized_html = normalize_html(html)
        messages = build_chat_messages(
            image_path=image_path,
            html=normalized_html,
            thinking=thinking,
            prompt_idx=prompt_idx,
        )

        # Span 통계 (메타데이터)
        span_stats = analyze_span(html)

        return {
            "messages": messages,
            "metadata": {
                "image_path": image_path,
                "num_rows": structure.num_rows,
                "num_cols": structure.num_cols,
                "num_span_cells": span_stats.span_cells,
                "complexity": span_stats.complexity,
                "complexity_score": span_stats.complexity_score,
            },
        }

    except Exception as e:
        print(f"Warning: Failed to convert {record.get('image_path', '?')}: {e}")
        return None


def convert_dataset(
    records: list[dict],
    include_thinking: bool = True,
    prompt_variation: bool = True,
) -> list[dict]:
    """
    전체 데이터셋을 변환한다.

    Args:
        records: 원시 레코드 리스트
        include_thinking: thinking chain 포함 여부
        prompt_variation: 프롬프트 다양화 여부

    Returns:
        변환된 레코드 리스트
    """
    converted = []
    failed = 0

    for i, record in enumerate(records):
        prompt_idx = None if prompt_variation else 0
        result = convert_single_record(record, include_thinking, prompt_idx)

        if result is not None:
            converted.append(result)
        else:
            failed += 1

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(records)} (failed: {failed})")

    print(f"Conversion complete: {len(converted)} success, {failed} failed")
    return converted


def save_dataset(data: list[dict], output_path: str):
    """JSONL 형식으로 저장."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved {len(data)} records to {output_path}")

    # 통계 출력
    complexities = [item["metadata"]["complexity"] for item in data]
    from collections import Counter
    dist = Counter(complexities)
    print(f"  Complexity distribution: {dict(dist)}")


def main():
    parser = argparse.ArgumentParser(description="TSR 라벨 데이터를 대화형 포맷으로 변환")
    parser.add_argument("--input", required=True, help="입력 파일/디렉토리 경로")
    parser.add_argument("--image_dir", default=None, help="이미지 디렉토리 (상대경로 해석용)")
    parser.add_argument("--output", required=True, help="출력 JSONL 경로")
    parser.add_argument(
        "--format",
        choices=["jsonl", "directory", "csv"],
        default="jsonl",
        help="입력 포맷",
    )
    parser.add_argument(
        "--no-thinking", action="store_true", help="Thinking chain 제외"
    )
    parser.add_argument(
        "--no-prompt-variation", action="store_true", help="프롬프트 다양화 비활성화"
    )
    args = parser.parse_args()

    # 로드
    records = load_raw_data(args.input, args.image_dir, args.format)

    # 변환
    converted = convert_dataset(
        records,
        include_thinking=not args.no_thinking,
        prompt_variation=not args.no_prompt_variation,
    )

    # 저장
    save_dataset(converted, args.output)


if __name__ == "__main__":
    main()
