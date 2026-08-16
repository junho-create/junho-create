# -*- coding: utf-8 -*-
from typing import List
from app.ingest.loader_pdf import TextBlock

TITLE_KEYWORDS_PRIMARY = ["계약서", "agreement", "contract"]
TITLE_KEYWORDS_SECONDARY = [
    "서약서", "각서", "동의서", "확인서", "약정서", "합의서", "위임장", "신청서"
]

def is_title_zone(block: TextBlock) -> bool:
    """간단한 타이틀 영역 휴리스틱: 1페이지 상단의 비교적 짧은 라인.
    두 계층으로 나누어 검색: PRIMARY 키워드를 먼저 검색하고, 없으면 SECONDARY를 검색.
    (필요 시 글자수/대문자 비율 등 추가 가능)
    """
    if block.page != 1:
        return False
    if block.line_no > 10:  # 첫 10줄 정도만
        return False
    t = block.text.strip()
    
    # 첫 번째 계층: PRIMARY 키워드 먼저 검색
    # 영문 키워드는 대소문자 무시, 한글 키워드는 그대로 검색
    t_lower = t.lower()
    for keyword in TITLE_KEYWORDS_PRIMARY:
        # 한글 키워드는 원본 텍스트에서, 영문 키워드는 소문자 변환된 텍스트에서 검색
        if keyword.isascii():
            if keyword.lower() in t_lower:
                # 키워드가 매칭되면 길이 제한 완화 (최소 2자, 최대 60자)
                if len(t) >= 2 and len(t) <= 60:
                    return True
        else:
            if keyword in t:
                # 키워드가 매칭되면 길이 제한 완화 (최소 2자, 최대 60자)
                if len(t) >= 2 and len(t) <= 60:
                    return True
    
    # 두 번째 계층: PRIMARY에서 없으면 SECONDARY 검색
    for keyword in TITLE_KEYWORDS_SECONDARY:
        if keyword in t:
            # 키워드가 매칭되면 길이 제한 완화 (최소 2자, 최대 60자)
            if len(t) >= 2 and len(t) <= 60:
                return True
    
    # 키워드가 매칭되지 않은 경우 기본 길이 제한 적용
    if len(t) < 6 or len(t) > 60:
        return False
    
    return False