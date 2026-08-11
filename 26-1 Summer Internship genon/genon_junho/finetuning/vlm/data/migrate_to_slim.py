"""
Legacy → Slim 포맷 마이그레이션 스크립트

legacy JSONL(messages/metadata 포맷)을 slim 포맷으로 일회성 변환한다.

slim 포맷:
    {
      "image_path": "...",
      "gt_html": "<table>...</table>",
      "thinking": "...",
      "complexity": "medium",
      "prompt_style": "chandra_table_with_ocr",
      "ocr_info": [{"text": "1톤", "bbox": [523, 30, 570, 53]}],
      "bbox_scale": 1024
    }

legacy 포맷:
    messages[0] = system
    messages[1] = user (image + text)
    messages[2] = assistant (<think>...</think> HTML)
    metadata.image_path, metadata.complexity, metadata.prompt_style
    metadata.ocr_info, metadata.bbox_scale (OCR 포함 시)

Usage:
    python -m data.migrate_to_slim \\
        --input  data/experiments/e7_gtfilter/train.jsonl \\
        --output data/experiments/e7_gtfilter/train_slim.jsonl

    # 이미 slim이면 그대로 통과
    python -m data.migrate_to_slim \\
        --input old.jsonl --output new.jsonl --skip_slim
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def _is_slim(record: dict) -> bool:
    """slim 포맷 여부를 판별한다."""
    return "image_path" in record and "gt_html" in record


def _extract_thinking_and_html(assistant_content: str) -> tuple[str, str]:
    """assistant 텍스트에서 thinking과 HTML을 분리한다.

    Returns:
        (thinking, gt_html)
    """
    if "</think>" in assistant_content:
        parts = assistant_content.split("</think>", 1)
        thinking_raw = parts[0]
        # <think> 태그 내부 텍스트 추출
        think_match = re.search(r"<think>([\s\S]*)", thinking_raw)
        thinking = think_match.group(1).strip() if think_match else thinking_raw.strip()
        gt_html = parts[1].strip()
    else:
        thinking = ""
        gt_html = assistant_content.strip()

    # HTML에서 <table>...</table>만 추출 (있으면)
    table_match = re.search(r"(<table[\s\S]*?</table>)", gt_html, re.IGNORECASE)
    if table_match:
        gt_html = table_match.group(1).strip()

    return thinking, gt_html


def _extract_image_path_from_messages(messages: list[dict]) -> str:
    """legacy messages에서 이미지 경로를 추출한다."""
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image":
                image = item.get("image", "")
                if image:
                    return image
    return ""


def _extract_assistant_content(messages: list[dict]) -> str:
    """legacy messages에서 assistant 텍스트를 추출한다."""
    for msg in messages:
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                chunks = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        chunks.append(str(item.get("text", "")))
                return "\n".join(chunks)
    return ""


def _convert_record(record: dict) -> dict:
    """단일 레코드를 slim 포맷으로 변환한다.

    이미 slim이면 그대로 반환한다.
    """
    if _is_slim(record):
        return record

    messages = record.get("messages", [])
    metadata = record.get("metadata", {}) or {}

    # image_path 추출
    image_path = metadata.get("image_path", "") or _extract_image_path_from_messages(messages)

    # assistant 텍스트에서 thinking/HTML 분리
    assistant_content = _extract_assistant_content(messages)
    thinking, gt_html = _extract_thinking_and_html(assistant_content)

    slim: dict = {
        "image_path": image_path,
        "gt_html": gt_html,
        "thinking": thinking,
        "complexity": metadata.get("complexity", ""),
        "prompt_style": metadata.get("prompt_style", ""),
    }

    # OCR 필드 (있으면 포함)
    if "ocr_info" in metadata:
        slim["ocr_info"] = metadata["ocr_info"]
    if "bbox_scale" in metadata:
        slim["bbox_scale"] = metadata["bbox_scale"]

    return slim


def run(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    converted = 0
    passed_through = 0
    errors = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for lineno, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  경고: line {lineno} JSON 파싱 실패 ({e}), 건너뜀")
                errors += 1
                continue

            if args.skip_slim and _is_slim(record):
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                passed_through += 1
                continue

            try:
                slim_record = _convert_record(record)
            except Exception as e:
                print(f"  경고: line {lineno} 변환 실패 ({e}), 건너뜀")
                errors += 1
                continue

            fout.write(json.dumps(slim_record, ensure_ascii=False) + "\n")
            if _is_slim(record):
                passed_through += 1
            else:
                converted += 1

    total = converted + passed_through
    print(f"\n=== 마이그레이션 완료 ===")
    print(f"  입력:        {input_path}")
    print(f"  출력:        {output_path}")
    print(f"  변환됨:      {converted}건")
    print(f"  통과(slim):  {passed_through}건")
    print(f"  오류:        {errors}건")
    print(f"  합계:        {total}건")


def main() -> None:
    parser = argparse.ArgumentParser(description="legacy JSONL → slim 포맷 변환")
    parser.add_argument("--input", required=True, help="입력 JSONL 경로 (legacy 포맷)")
    parser.add_argument("--output", required=True, help="출력 JSONL 경로 (slim 포맷)")
    parser.add_argument(
        "--skip_slim",
        action="store_true",
        help="이미 slim 포맷인 레코드는 변환 없이 그대로 통과",
    )
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"오류: 입력 파일 없음: {args.input}", file=sys.stderr)
        sys.exit(1)

    run(args)


if __name__ == "__main__":
    main()
