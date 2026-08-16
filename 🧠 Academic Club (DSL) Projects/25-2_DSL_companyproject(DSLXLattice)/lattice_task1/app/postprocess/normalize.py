# -*- coding: utf-8 -*-
import re
from typing import List
from app.extractors.base import Candidate

MONTHS = {
    "jan":"01","feb":"02","mar":"03","apr":"04","may":"05","jun":"06",
    "jul":"07","aug":"08","sep":"09","oct":"10","nov":"11","dec":"12",
}

CCY_MAP = {"원":"KRW","₩":"KRW","$":"USD","€":"EUR","달러":"USD","eur":"EUR","usd":"USD","krw":"KRW"}


def _to_yyyymmdd(s: str) -> str:
    """날짜 문자열을 YYYYMMDD 형식으로 정규화합니다.
    
    지원 형식:
    - 2024.7.1 / 2024-07-01 / 2024/07/01
    - 2024년 7월 1일 / 2024 년 7 월 1 일 (공백 많음)
    - 1 July 2024
    - 20240701 (이미 정규화된 경우)
    """
    if not s:
        return s
    
    # 1단계: 공백 정리 (연속된 공백을 하나로, 앞뒤 공백 제거)
    s = re.sub(r'\s+', ' ', s.strip())
    
    # 2단계: 숫자 8자리로 이미 들어온 경우 (YYYYMMDD)
    m = re.search(r'\b(\d{8})\b', s)
    if m:
        return m.group(1)
    
    # 3단계: 한국어 날짜 형식 (2024년 7월 1일) - 공백 허용
    # 연속된 공백을 고려하여 \s+ 사용
    patterns = [
        r'(\d{4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일?',  # 2024년 7월 1일
        r'(\d{4})\s*[.\-/]\s*(\d{1,2})\s*[.\-/]\s*(\d{1,2})',  # 2024.7.1 / 2024-07-01
    ]
    
    for pattern in patterns:
        m = re.search(pattern, s)
        if m:
            try:
                y, mo, d = m.group(1), int(m.group(2)), int(m.group(3))
                # 유효성 검사
                if 1900 <= int(y) <= 2100 and 1 <= mo <= 12 and 1 <= d <= 31:
                    return f"{y}{mo:02d}{d:02d}"
            except (ValueError, IndexError):
                continue
    
    # 4단계: 영문 날짜 형식 (1 July 2024)
    m = re.search(r'(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})', s, re.IGNORECASE)
    if m:
        try:
            d, mon, y = int(m.group(1)), m.group(2)[:3].lower(), m.group(3)
            month_num = MONTHS.get(mon, None)
            if month_num and 1 <= d <= 31:
                return f"{y}{month_num}{d:02d}"
        except (ValueError, IndexError):
            pass
    
    # 5단계: 실패 시 원문 반환 (디버깅을 위해)
    return s


def _norm_currency(s: str) -> str:
    v = CCY_MAP.get(s.strip(), s.strip()).upper()
    if v in {"KRW","USD","JPY","EUR"}:
        return v
    return v


def apply(cands: List[Candidate]) -> List[Candidate]:
    out: List[Candidate] = []
    for c in cands:
        if c.field in {"CNT_CON_DATE","CNT_ST_DATE","CNT_END_DATE"}:
            c.raw_value = _to_yyyymmdd(c.raw_value)
        elif c.field == "CNT_AMT":
            # 콤마와 공백 제거
            c.raw_value = re.sub(r"[,\s]", "", c.raw_value)
        elif c.field == "CNT_AMT_CRY":
            c.raw_value = _norm_currency(c.raw_value)
        elif c.field == "CNT_AUTO_RNW_TERM_UNIT":
            # 이미 영문 대문자 기대, 한국어 유입은 추출단계에서 변환됨
            c.raw_value = c.raw_value.upper()
        out.append(c)
    return out