"""
프롬프트 스타일 변환기

동일 이미지/GT를 유지하면서 프롬프트 스타일만 변경한다.
주 용도: OCR on 데이터를 OCR off 데이터로 변환 (또는 그 반대).

Usage:
    python -m data.convert_prompt_style \
        --input ./data/experiments/shared/train.jsonl \
        --output ./data/experiments/shared/train_ocr_off.jsonl \
        --prompt_style chandra_table_without_ocr
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.prompt_templates import (
    get_system_prompt,
    get_user_prompt_with_style,
    normalize_prompt_style,
    prompt_requires_ocr,
    PROMPT_STYLE_CHANDRA_WITH_OCR,
    PROMPT_STYLE_CHANDRA_NO_OCR,
)


def convert_record(record: dict, target_style: str, bbox_scale: int) -> dict:
    """단일 레코드의 프롬프트 스타일을 변환한다."""
    messages = record.get("messages", [])
    metadata = record.get("metadata", {})

    if len(messages) < 3:
        return record

    # 기존 system prompt의 인덱스를 보존하기 위해 prompt_idx=0 고정
    system_msg = messages[0]
    user_msg = messages[1]
    assistant_msg = messages[2]

    # OCR 정보: 타겟 스타일이 OCR을 요구하면 metadata에서 가져옴
    ocr_info = None
    if prompt_requires_ocr(target_style):
        ocr_info = metadata.get("ocr_info", [])

    # user content에서 이미지 부분 유지, 텍스트만 교체
    new_user_content = []
    for item in (user_msg.get("content") or []):
        if isinstance(item, dict) and item.get("type") == "image":
            new_user_content.append(item)
        elif isinstance(item, dict) and item.get("type") == "text":
            new_text = get_user_prompt_with_style(
                idx=0,
                prompt_style=target_style,
                ocr_info=ocr_info,
                bbox_scale=bbox_scale,
            )
            new_user_content.append({"type": "text", "text": new_text})
        else:
            new_user_content.append(item)

    new_messages = [
        system_msg,
        {"role": "user", "content": new_user_content},
        assistant_msg,
    ]

    new_metadata = {**metadata}
    new_metadata["prompt_style"] = target_style
    new_metadata["ocr_prompt_enabled"] = prompt_requires_ocr(target_style)

    return {"messages": new_messages, "metadata": new_metadata}


def main():
    parser = argparse.ArgumentParser(description="프롬프트 스타일 변환")
    parser.add_argument("--input", required=True, help="입력 JSONL 경로")
    parser.add_argument("--output", required=True, help="출력 JSONL 경로")
    parser.add_argument(
        "--prompt_style",
        required=True,
        choices=[
            PROMPT_STYLE_CHANDRA_WITH_OCR,
            PROMPT_STYLE_CHANDRA_NO_OCR,
        ],
        help="변환 타겟 스타일: chandra_with_ocr | chandra_no_ocr",
    )
    parser.add_argument("--bbox_scale", type=int, default=1024)
    args = parser.parse_args()

    target_style = normalize_prompt_style(args.prompt_style)
    if args.prompt_style != target_style:
        print(f"Warning: '{args.prompt_style}' -> '{target_style}'")

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    converted = 0
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            record = json.loads(line)
            new_record = convert_record(record, target_style, args.bbox_scale)
            fout.write(json.dumps(new_record, ensure_ascii=False) + "\n")
            converted += 1

    print(f"Converted {converted} records")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")
    print(f"  Target style: {target_style}")
    print(f"  OCR enabled:  {prompt_requires_ocr(target_style)}")


if __name__ == "__main__":
    main()
