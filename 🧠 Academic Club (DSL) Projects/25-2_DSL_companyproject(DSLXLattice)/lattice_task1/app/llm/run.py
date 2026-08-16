# -*- coding: utf-8 -*-
"""
app/llm/run.py

LLM 전용 전체 필드 추출 파이프라인.

- 기존 app/pipeline/run.py는 규칙 기반 + 날짜 LLM을 사용하는 "메인" 파이프라인.
- 이 파일은 LLM을 통해 전체 필드를 한 번에 뽑는 "보조/실험" 파이프라인이다.
"""

from typing import Dict
from app.ingest.loader_pdf import load_pdf_with_fallback_ocr
from app.ingest.loader_pdf import TextBlock
from app.llm.fields_llm_extractor import extract_fields_with_llm_from_blocks


def run_llm_on_file(pdf_path: str) -> Dict[str, Dict]:
    """
    단일 PDF 파일을 대상으로:
    1) PDF -> TextBlock[] (OCR 포함)
    2) LLM 전체 필드 추출

    Args:
        pdf_path: 처리할 PDF 파일 경로

    Returns:
        Dict[str, Dict]: 필드별 LLM 결과
    """
    # 1) 기존 ingest 경로 그대로 재사용
    blocks: list[TextBlock] = load_pdf_with_fallback_ocr(pdf_path)

    # 2) LLM 전체 필드 추출
    result = extract_fields_with_llm_from_blocks(blocks)

    return result
