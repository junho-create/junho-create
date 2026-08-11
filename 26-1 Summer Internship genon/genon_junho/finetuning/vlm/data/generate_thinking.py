"""
Thinking Chain 자동 생성기

기본 템플릿 기반 생성 외에, LLM을 사용하여 고품질 thinking chain을 생성한다.
- 템플릿 기반: 빠르지만 정형화됨 (convert_to_chat.py에서 기본 사용)
- LLM 기반: 느리지만 자연스럽고 다양한 추론 과정 (이 스크립트)

Usage:
    # 템플릿 기반 (기본, convert_to_chat.py에서 이미 수행)
    python -m data.generate_thinking --input dataset.jsonl --mode template

    # LLM 보조 생성 (OpenAI/Anthropic API 사용)
    python -m data.generate_thinking --input dataset.jsonl --mode llm \
        --api_provider anthropic --api_key $ANTHROPIC_API_KEY
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.html_utils import parse_html_table, extract_spans_from_html
from utils.prompt_templates import build_thinking_chain, format_assistant_response


# =============================================================================
# LLM-based Thinking Generation
# =============================================================================

LLM_PROMPT_FOR_THINKING = """당신은 테이블 구조 분석 전문가입니다.
아래 HTML 테이블의 구조를 분석하는 사고 과정을 작성해주세요.

HTML:
{html}

다음 형식으로 사고 과정을 작성해주세요:
1. 테이블 크기 (행 × 열) 파악
2. 헤더/바디 영역 구분
3. Span 셀(병합 셀) 탐지 - 각 span 셀의 위치와 크기 설명
4. HTML 구조 생성 계획

주의사항:
- 한국어로 작성
- 간결하게 (5~10줄)
- 각 span 셀의 위치를 (행,열) 좌표로 표기
- 실제 관찰에 기반하여 작성 (추측하지 않음)
"""


def generate_thinking_with_llm(
    html: str,
    api_provider: str = "anthropic",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    """
    LLM API를 사용하여 thinking chain을 생성한다.

    Args:
        html: 테이블 HTML
        api_provider: "anthropic" 또는 "openai"
        api_key: API 키
        model: 모델명

    Returns:
        thinking chain 텍스트
    """
    prompt = LLM_PROMPT_FOR_THINKING.format(html=html)

    if api_provider == "anthropic":
        return _call_anthropic(prompt, api_key, model or "claude-sonnet-4-5-20250929")
    elif api_provider == "openai":
        return _call_openai(prompt, api_key, model or "gpt-4o")
    else:
        raise ValueError(f"Unknown API provider: {api_provider}")


def _call_anthropic(prompt: str, api_key: str, model: str) -> str:
    """Anthropic API 호출."""
    try:
        import anthropic
    except ImportError:
        raise ImportError("pip install anthropic")

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


def _call_openai(prompt: str, api_key: str, model: str) -> str:
    """OpenAI API 호출."""
    try:
        import openai
    except ImportError:
        raise ImportError("pip install openai")

    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()


def generate_thinking_template(html: str) -> str:
    """템플릿 기반 thinking chain 생성."""
    structure = parse_html_table(html)
    spans = extract_spans_from_html(html)
    return build_thinking_chain(
        num_rows=structure.num_rows,
        num_cols=structure.num_cols,
        spans=spans,
        header_rows=structure.header_rows,
    )


# =============================================================================
# Batch Processing
# =============================================================================


def process_dataset(
    input_path: str,
    output_path: str,
    mode: str = "template",
    api_provider: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    overwrite: bool = False,
):
    """
    데이터셋의 thinking chain을 생성/갱신한다.

    이미 thinking이 포함된 레코드는 overwrite=True가 아니면 건너뜀.
    """
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    updated = 0
    skipped = 0
    failed = 0

    for i, record in enumerate(records):
        messages = record.get("messages", [])

        # assistant 메시지에서 기존 thinking 확인
        assistant_msg = None
        for msg in messages:
            if msg["role"] == "assistant":
                assistant_msg = msg
                break

        if not assistant_msg:
            failed += 1
            continue

        content = assistant_msg["content"]
        has_thinking = "<think>" in content and "</think>" in content

        if has_thinking and not overwrite:
            skipped += 1
            continue

        # HTML 추출 (thinking 이후 또는 전체가 HTML)
        if has_thinking:
            html = content.split("</think>")[-1].strip()
        else:
            html = content.strip()

        # Thinking 생성
        try:
            if mode == "template":
                thinking = generate_thinking_template(html)
            elif mode == "llm":
                thinking = generate_thinking_with_llm(
                    html, api_provider, api_key, model
                )
            else:
                raise ValueError(f"Unknown mode: {mode}")

            assistant_msg["content"] = format_assistant_response(thinking, html)
            updated += 1

        except Exception as e:
            print(f"  Warning: Failed record {i}: {e}")
            failed += 1

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(records)}")

    # 저장
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Done: {updated} updated, {skipped} skipped, {failed} failed → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Thinking chain 생성기")
    parser.add_argument("--input", required=True, help="입력 JSONL")
    parser.add_argument("--output", default=None, help="출력 JSONL (기본: 입력 덮어쓰기)")
    parser.add_argument(
        "--mode", choices=["template", "llm"], default="template", help="생성 모드"
    )
    parser.add_argument("--api_provider", default="anthropic", help="LLM API 제공자")
    parser.add_argument("--api_key", default=None, help="API 키 (환경변수 우선)")
    parser.add_argument("--model", default=None, help="LLM 모델명")
    parser.add_argument("--overwrite", action="store_true", help="기존 thinking 덮어쓰기")
    args = parser.parse_args()

    output = args.output or args.input
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")

    process_dataset(
        args.input, output, args.mode,
        args.api_provider, api_key, args.model,
        args.overwrite,
    )


if __name__ == "__main__":
    main()
