# -*- coding: utf-8 -*-
"""
공통 날짜 LLM 추출 모듈
한 번의 LLM 호출로 체결일, 시작일, 종료일을 모두 추출하여 비용 절감
"""
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import re
import hashlib
import requests
from dotenv import load_dotenv
from app.ingest.loader_pdf import TextBlock
from app.utils.regexes import RE_DATE_GENERIC, RE_RANGE1, RE_RANGE2, RE_AUTO_PERIOD

# .env 파일 로드 (프로젝트 루트에서)
env_path = Path(__file__).parents[2] / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    # .env 파일이 없어도 환경변수는 사용 가능
    load_dotenv()

# Bedrock API 설정 (환경변수에서만 읽기, 하드코딩 없음)
BEDROCK_URL = os.getenv("BEDROCK_API_URL")
BEDROCK_TIMEOUT = int(os.getenv("BEDROCK_TIMEOUT", "60"))

# 캐시: blocks 해시 -> LLM 결과
_DATE_LLM_CACHE: Dict[str, Dict[str, Optional[str]]] = {}


def _hash_blocks(blocks: List[TextBlock]) -> str:
    """blocks의 해시값 생성 (캐시 키로 사용)"""
    # 간단하게 텍스트 내용만 해시
    text_content = "\n".join([f"{b.page}:{b.line_no}:{b.text}" for b in blocks])
    return hashlib.md5(text_content.encode('utf-8')).hexdigest()


def collect_date_related_texts(blocks: List[TextBlock]) -> List[Tuple[int, int, str, TextBlock]]:
    """
    날짜 관련 텍스트를 모두 수집 (기간 범위 포함)
    - 날짜 패턴 (YYYY-MM-DD, 2024년 7월 1일 등)
    - 기간 표현 (1년, 6개월, 30일 등)
    - 계약 기간 관련 키워드와 함께 기간 표현이 있는 경우
    - 효력일, 발효일 등 시작일 관련 키워드가 있는 텍스트
    """
    date_texts: List[Tuple[int, int, str, TextBlock]] = []
    
    # 효력일/발효일 관련 키워드
    effective_keywords = re.compile(
        r"효력일|발효일|효력\s*기간|발효\s*기간|효력\s*시작|발효\s*시작|효력\s*일자|발효\s*일자|"
        r"Effective\s*Date|Effective\s*Period|Effective\s*from|"
        r"다음날|다음\s*날|이튿날|익일",
        re.IGNORECASE
    )
    
    for b in blocks:
        text = b.text
        
        # 1. 날짜 패턴이 있는 경우 (기존 로직)
        if RE_DATE_GENERIC.search(text):
            date_texts.append((b.page, b.line_no, text, b))
            continue
        
        # 2. 기간 표현(년/월/일)이 있는 경우
        # 계약 기간 관련 키워드와 함께 있거나, 기간 표현만 있어도 포함 (LLM이 판단하도록)
        if RE_AUTO_PERIOD.search(text):
            date_texts.append((b.page, b.line_no, text, b))
            continue
        
        # 3. 효력일/발효일 관련 키워드가 있는 경우
        # 예: "계약 효력일은 다음날로부터 한다", "발효일은 체결일로부터" 등
        if effective_keywords.search(text):
            date_texts.append((b.page, b.line_no, text, b))
    
    return date_texts


def build_date_prompt(date_texts: List[Tuple[int, int, str, TextBlock]]) -> str:
    """날짜 관련 텍스트들을 프롬프트로 구성 (체결일, 시작일, 종료일 모두 추출)"""
    prompt_parts = [
        "다음은 계약서에서 추출한 날짜 관련 텍스트들입니다. 이 중에서 다음 3가지 날짜를 찾아주세요:",
        "",
        "1. 계약 체결일 (서명일, 날인일): 계약서에 서명하거나 날인한 날짜",
        "2. 계약 시작일: 계약이 시작되는 날짜",
        "3. 계약 종료일: 계약이 종료되는 날짜",
        "",
        "참고사항:",
        "- 계약 체결일은 '체결일', '서명일', '날인일' 등의 키워드가 있는 날짜입니다.",
        "- 계약 기간은 '~', '부터~까지', 'from~to' 등의 패턴으로 표현됩니다.",
        "- 기간 표현(예: '1년', '6개월', '30일')이 있는 경우, 체결일이나 시작일을 기준으로 종료일을 계산할 수 있습니다.",
        "- 효력일, 발효일 관련 텍스트(예: '계약 효력일은 체결일 다음날로부터 한다', '계약 효력일은 체결일 당일로 한다.')가 있으면 시작일을 추론할 수 있습니다.",
        "- 시작일이 명시되지 않은 경우, 계약 체결일을 시작일로 사용합니다.",
        "- 여러 날짜가 있다면 가장 최근 날짜이거나 문서 하단에 있는 날짜일 가능성이 높습니다.",
        "- 명확하게 찾을 수 없는 날짜는 'NOT_FOUND'로 표시해주세요.",
        "",
        "=== 날짜 관련 텍스트 ===",
    ]
    
    for page, line_no, text, _ in date_texts:
        prompt_parts.append(f"[페이지 {page}, 라인 {line_no}] {text}")
    
    prompt_parts.extend([
        "",
        "위 텍스트들에서 다음 형식으로 답변해주세요:",
        "체결일: [날짜 또는 NOT_FOUND]",
        "시작일: [날짜 또는 NOT_FOUND]",
        "종료일: [날짜 또는 NOT_FOUND]",
        "",
        "날짜는 원본 형식 그대로 반환해주세요 (예: '2024년 7월 1일', '2024-07-01', '2024.07.01' 등).",
        "",
        "답변:"
    ])
    
    return "\n".join(prompt_parts)


def parse_llm_date_response(response: str) -> Dict[str, Optional[str]]:
    """LLM 응답에서 체결일, 시작일, 종료일을 파싱"""
    result = {
        "con_date": None,
        "st_date": None,
        "end_date": None
    }
    
    if not response:
        print(f"[DEBUG] parse_llm_date_response: 빈 응답")
        return result
    
    response = response.strip()
    print(f"[DEBUG] parse_llm_date_response: 파싱 시작, 응답 길이={len(response)}")
    
    # 각 필드 추출
    patterns = {
        "con_date": [
            r"체결일\s*[:：]\s*([^\n]+)",
            r"서명일\s*[:：]\s*([^\n]+)",
            r"날인일\s*[:：]\s*([^\n]+)",
        ],
        "st_date": [
            r"시작일\s*[:：]\s*([^\n]+)",
            r"계약\s*시작일\s*[:：]\s*([^\n]+)",
        ],
        "end_date": [
            r"종료일\s*[:：]\s*([^\n]+)",
            r"계약\s*종료일\s*[:：]\s*([^\n]+)",
        ]
    }
    
    for field, field_patterns in patterns.items():
        for pattern in field_patterns:
            m = re.search(pattern, response, re.IGNORECASE)
            if m:
                value = m.group(1).strip()
                print(f"[DEBUG] {field} 패턴 매칭 성공: '{value}'")
                
                # NOT_FOUND 체크
                if "NOT_FOUND" in value.upper() or "찾을 수 없" in value:
                    print(f"[DEBUG] {field}: NOT_FOUND로 판단, 스킵")
                    continue
                
                # 날짜 패턴 추출 (공백 제거 후)
                value_clean = value.strip()
                
                # 직접 날짜 패턴 확인 (더 정확한 매칭을 위해 우선 사용)
                date_patterns = [
                    r'\d{4}\s*년\s*\d{1,2}\s*월\s*\d{1,2}\s*일',  # 2025년 4월 1일
                    r'\d{4}[.\-/]\s*\d{1,2}[.\-/]\s*\d{1,2}',      # 2025-04-01, 2025.4.1
                    r'\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}',  # 1 Jan 2025
                ]
                
                extracted = None
                for date_pattern in date_patterns:
                    date_m = re.search(date_pattern, value_clean, re.IGNORECASE)
                    if date_m:
                        extracted = date_m.group(0)
                        print(f"[DEBUG] {field}: 직접 패턴으로 추출 성공: '{extracted}' (패턴: {date_pattern})")
                        break
                
                # 직접 패턴으로 못 찾았으면 RE_DATE_GENERIC 시도
                if not extracted:
                    date_match = RE_DATE_GENERIC.search(value_clean)
                    if date_match:
                        # group(0)은 전체 매칭
                        extracted = date_match.group(0).strip()
                        if extracted:
                            print(f"[DEBUG] {field}: RE_DATE_GENERIC으로 추출 성공: '{extracted}'")
                
                if extracted:
                    result[field] = extracted
                    break
                else:
                    print(f"[DEBUG] {field}: 모든 패턴 매칭 실패, value_clean='{value_clean}'")
        if result[field]:
            print(f"[DEBUG] {field}: 최종 결과 = '{result[field]}'")
        else:
            print(f"[DEBUG] {field}: 추출 실패")
    
    return result


def call_bedrock_api(
    prompt: str, 
    bedrock_url: Optional[str] = None, 
    timeout: int = BEDROCK_TIMEOUT,
    temperature: float = 0.0
) -> dict:
    """
    Bedrock API를 호출하여 LLM 응답을 받습니다.
    
    Args:
        prompt: LLM에 전달할 프롬프트
        bedrock_url: Bedrock API 엔드포인트 URL (None이면 환경변수에서 읽기)
        timeout: 요청 타임아웃 (초)
        temperature: LLM의 temperature 파라미터 (0.0~1.0, 기본값 0.0으로 일관성 확보)
                     추출 작업(Extraction)에서는 창의성이 필요 없으므로 0.0을 권장합니다.
    
    Returns:
        dict: Bedrock API 응답 JSON
    
    Raises:
        ValueError: API URL이 설정되지 않았거나 응답이 유효한 JSON이 아닌 경우
        requests.RequestException: API 호출 실패 시
    """
    if bedrock_url is None:
        bedrock_url = BEDROCK_URL
    
    if not bedrock_url:
        raise ValueError(
            "Bedrock API URL이 설정되지 않았습니다.\n"
            "설정 방법:\n"
            "1. .env 파일 생성: cp .env.example .env\n"
            "2. .env 파일에 BEDROCK_API_URL=... 추가\n"
            "3. 또는 환경변수로 설정: export BEDROCK_API_URL=...\n"
            "4. 또는 함수 호출 시 bedrock_url 인자 제공"
        )
    
    # temperature를 payload에 포함 (API가 지원하는 경우)
    payload = {
        "prompt": prompt,
        "temperature": temperature
    }
    
    try:
        response = requests.post(bedrock_url, json=payload, timeout=timeout)
        response.raise_for_status()  # HTTP 오류 시 예외 발생
        
        # JSON 파싱
        try:
            return response.json()
        except ValueError as e:
            raise ValueError(f"Bedrock API 응답이 유효한 JSON이 아닙니다: {e}")
    
    except requests.RequestException as e:
        raise requests.RequestException(f"Bedrock API 호출 실패: {e}")


def extract_dates_with_llm(blocks: List[TextBlock], use_cache: bool = True) -> Dict[str, Optional[str]]:
    """
    LLM을 사용하여 체결일, 시작일, 종료일을 한 번에 추출
    
    Returns:
        {
            "con_date": "2024년 7월 1일" 또는 None,
            "st_date": "2024-01-01" 또는 None,
            "end_date": "2024-12-31" 또는 None
        }
    """
    # 캐시 확인
    cache_key = _hash_blocks(blocks) if use_cache else None
    if cache_key and cache_key in _DATE_LLM_CACHE:
        print(f"[DEBUG] 날짜 LLM 캐시 사용")
        return _DATE_LLM_CACHE[cache_key]
    
    # 날짜 관련 텍스트 수집
    date_texts = collect_date_related_texts(blocks)
    
    if not date_texts:
        result = {"con_date": None, "st_date": None, "end_date": None}
        if cache_key:
            _DATE_LLM_CACHE[cache_key] = result
        return result
    
    # 프롬프트 구성
    prompt = build_date_prompt(date_texts)
    
    try:
        # Bedrock API 호출
        llm_response = call_bedrock_api(prompt)
        
        # 응답이 dict인 경우 처리
        if isinstance(llm_response, dict):
            # 다양한 응답 구조 지원
            if "data" in llm_response and isinstance(llm_response["data"], dict):
                # {'ok': True, 'data': {'result': '...'}} 형태
                response_text = llm_response["data"].get("result", "")
            elif "result" in llm_response:
                # {'result': '...'} 형태
                response_text = llm_response["result"]
            else:
                # 기타 형태
                response_text = llm_response.get("response", llm_response.get("text", llm_response.get("content", "")))
            
            if not response_text:
                response_text = str(llm_response)
        else:
            response_text = str(llm_response)
        
        print(f"[DEBUG] 날짜 LLM 응답 텍스트: {response_text[:500]}...")
        print(f"[DEBUG] 응답 텍스트 전체 길이: {len(response_text)}")
        
        # 파싱
        result = parse_llm_date_response(response_text)
        
        # 시작일이 없고 체결일이 있으면 체결일을 시작일로 사용
        if not result.get("st_date") and result.get("con_date"):
            result["st_date"] = result["con_date"]
            print(f"[DEBUG] 시작일이 없어서 체결일을 시작일로 사용: {result['st_date']}")
        
        print(f"[DEBUG] 파싱된 날짜 - 체결일: {result['con_date']}, 시작일: {result['st_date']}, 종료일: {result['end_date']}")
        
        # 캐시 저장
        if cache_key:
            _DATE_LLM_CACHE[cache_key] = result
        
        return result
    
    except Exception as e:
        print(f"[WARN] 날짜 LLM 추출 실패: {e}")
        import traceback
        traceback.print_exc()
        
        result = {"con_date": None, "st_date": None, "end_date": None}
        if cache_key:
            _DATE_LLM_CACHE[cache_key] = result
        return result


def clear_cache():
    """캐시 초기화 (테스트용)"""
    global _DATE_LLM_CACHE
    _DATE_LLM_CACHE.clear()

