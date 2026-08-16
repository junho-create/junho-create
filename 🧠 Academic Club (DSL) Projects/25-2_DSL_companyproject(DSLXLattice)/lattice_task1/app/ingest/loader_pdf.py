# -*- coding: utf-8 -*-
"""pdfminer.six로 텍스트 PDF를 줄 단위 TextBlock으로 변환.
스캔 PDF는 OCR 모듈을 나중에 앞단에 삽입(동일 포맷 반환)하면 됨.
"""
from dataclasses import dataclass
from typing import List, Tuple, Optional
from pdfminer.high_level import extract_text

@dataclass
class TextBlock:
    page: int
    line_no: int
    text: str
    bbox: Optional[Tuple[float, float, float, float]] = None  # 단순화(옵션)


def load_pdf_as_blocks(pdf_path: str) -> List[TextBlock]:
    """
    pdfminer.six를 사용하여 텍스트 PDF를 TextBlock 리스트로 변환합니다.
    
    Args:
        pdf_path: PDF 파일 경로
    
    Returns:
        List[TextBlock]: 추출된 텍스트 블록 리스트
    """
    blocks: List[TextBlock] = []
    # 간단하게: 페이지별 텍스트를 얻기 위해 laparams로 페이지 구분을 유지
    # extract_text는 page_numbers를 받지 않으므로, 전체 텍스트 후에 페이지 분리 대신
    # 두 번 호출 비용을 줄이기 위해 한 번에 뽑고 페이지 마커로 나눌 수도 있지만,
    # 여기서는 간단히 페이지별 호출(성능 문제시 최적화 가능)
    from pdfminer.pdfpage import PDFPage
    from pdfminer.pdfdocument import PDFDocument
    from pdfminer.pdfparser import PDFParser

    with open(pdf_path, "rb") as f:
        parser = PDFParser(f)
        doc = PDFDocument(parser)
        pages = list(PDFPage.create_pages(doc))

    for pageno in range(len(pages)):
        # 개별 페이지 텍스트 추출
        page_text = extract_text(pdf_path, page_numbers=[pageno]) or ""
        for i, line in enumerate(page_text.splitlines()):
            # 공백만 있는 라인은 스킵
            if line.strip() == "":
                continue
            blocks.append(TextBlock(page=pageno+1, line_no=i+1, text=line.rstrip()))
    return blocks


def load_pdf_via_ocr(
    pdf_path: str,
    ocr_url: Optional[str] = None,
    timeout: int = 60
) -> List[TextBlock]:
    """
    OCR API를 사용하여 PDF/이미지에서 텍스트 블록을 추출합니다.
    
    Args:
        pdf_path: PDF/이미지 파일 경로
        ocr_url: OCR API 엔드포인트 URL (None이면 환경변수 또는 기본값 사용)
        timeout: 요청 타임아웃 (초)
    
    Returns:
        List[TextBlock]: 추출된 텍스트 블록 리스트
    
    Raises:
        FileNotFoundError: 파일이 존재하지 않는 경우
        requests.RequestException: API 호출 실패 시
        ValueError: 응답 파싱 실패 시
    """
    from app.ingest.loader_ocr import load_pdf_via_ocr as ocr_loader
    return ocr_loader(pdf_path, ocr_url, timeout)


def load_pdf_with_fallback_ocr(
    pdf_path: str,
    ocr_url: Optional[str] = None,
    min_blocks_threshold: int = 10,
    timeout: int = 60
) -> List[TextBlock]:
    """
    PDF를 항상 OCR로 먼저 시도하고, 실패하면 텍스트 추출로 폴백합니다.
    
    처리 순서:
    1. 먼저 OCR API 호출 시도
    2. OCR 성공 시 OCR 결과 반환
    3. OCR 실패 시 pdfminer.six로 텍스트 추출 시도 (폴백)
    
    Args:
        pdf_path: PDF 파일 경로
        ocr_url: OCR API 엔드포인트 URL (None이면 환경변수 또는 기본값 사용)
        min_blocks_threshold: 사용되지 않음 (하위 호환성을 위해 유지)
        timeout: OCR API 요청 타임아웃 (초)
    
    Returns:
        List[TextBlock]: 추출된 텍스트 블록 리스트
    """
    # 1단계: 먼저 OCR API 호출 시도
    print(f"[INFO] OCR API 호출 시도...")
    try:
        ocr_blocks = load_pdf_via_ocr(pdf_path, ocr_url, timeout)
        if ocr_blocks:
            print(f"[INFO] OCR 성공: {len(ocr_blocks)}개 블록 추출")
            return ocr_blocks
        else:
            print(f"[WARN] OCR 응답이 비어있습니다. 텍스트 추출로 폴백...")
    except Exception as e:
        print(f"[WARN] OCR 실패, 텍스트 추출로 폴백: {e}")
    
    # 2단계: OCR 실패 시 텍스트 추출로 폴백
    print(f"[INFO] pdfminer.six로 텍스트 추출 시도...")
    blocks = load_pdf_as_blocks(pdf_path)
    print(f"[INFO] 텍스트 추출 완료: {len(blocks)}개 블록 추출")
    return blocks