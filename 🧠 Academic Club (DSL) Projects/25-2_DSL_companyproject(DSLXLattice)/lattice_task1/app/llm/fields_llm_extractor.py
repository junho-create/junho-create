# -*- coding: utf-8 -*-
"""
app/llm/fields_llm_extractor.py

역할:
- TextBlock 리스트(이미 OCR/파싱까지 끝난 상태)를 입력으로 받아,
- Bedrock LLM을 한 번 호출해서
- 계약서 메타데이터 11개 필드를 JSON 형태로 반환하는 모듈.

기본 아이디어:
1) 기존 ingest 단계 (loader_pdf.load_pdf_with_fallback_ocr)는 그대로 사용.
   - 이 모듈은 "TextBlock -> LLM -> dict" 부분만 담당.
2) date_llm_extractor에서 이미 구현된 call_bedrock_api를 재사용하여
   - .env, BEDROCK_API_URL, timeout 설정 등을 그대로 공유.
3) postprocess.normalize의 _to_yyyymmdd, _norm_currency를 그대로 재사용하여
   - 날짜/통화 정규화 로직을 중복 정의하지 않는다.
4) 결과는 아래와 같은 dict로 반환 (llm 전용 결과):
   {
       "CNT_NAME": { "value": "...", "confidence": 0.85, "note": "llm_only" },
       ...
   }
"""

import json
import hashlib
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from dotenv import load_dotenv  # type: ignore

from app.ingest.loader_pdf import TextBlock  # TextBlock 타입 재사용
from app.ingest.date_llm_extractor import call_bedrock_api  # LLM 호출 로직 재사용
from app.llm.prompt_templates import build_full_fields_prompt
from app.postprocess.normalize import _to_yyyymmdd, _norm_currency

# .env 로드 (date_llm_extractor와 동일한 방식으로 동작하게 유지)
PROJECT_ROOT = Path(__file__).parents[2]
ENV_PATH = PROJECT_ROOT / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
else:
    load_dotenv()

# LLM input/output 로깅 설정
LOG_LLM_IO = os.getenv("LOG_LLM_IO", "true").lower() == "true"
LLM_DEBUG_DIR = PROJECT_ROOT / "outputs" / "llm" / "debug"

# 이 모듈에서 다루는 필드 목록 (대장 기준)
FIELDS = [
    "CNT_NAME",
    "CNT_CON_DATE",
    "CNT_ST_DATE",
    "CNT_END_DATE",
    "CNT_EXEC_FLAG",
    "CNT_RENEWAL",
    "CNT_AMT",
    "CNT_AMT_CRY",
    "CNT_AUTO_RNW_TERM_NUM",
    "CNT_AUTO_RNW_TERM_UNIT",
]

# 간단한 메모리 캐시 (같은 계약서를 여러 번 LLM에 던지지 않도록 방지)
_FIELDS_LLM_CACHE: Dict[str, Dict[str, Any]] = {}


# ---------------------------------------------------------------------
# 헬퍼 함수들
# ---------------------------------------------------------------------

def _hash_blocks(blocks: List[TextBlock]) -> str:
    """
    TextBlock 리스트를 간단히 해시해서 캐시 키로 사용한다.
    - 페이지/라인 번호 + 텍스트를 모두 이어 붙여서 SHA256으로 해시.
    """
    h = hashlib.sha256()
    for b in blocks:
        line = f"{b.page}:{b.line_no}:{(b.text or '').strip()}\n"
        h.update(line.encode("utf-8", errors="ignore"))
    return h.hexdigest()


def blocks_to_compact_text(blocks: List[TextBlock], max_chars: Optional[int] = None, recursion_depth: int = 0) -> str:
    """
    TextBlock 리스트를 LLM에 보낼 문자열로 변환한다.

    - 모든 OCR 블록을 포함하여 프롬프트에 전달한다.
    - 각 줄 앞에 [p01 l003] 같은 prefix를 붙여서, 나중에 사람이 눈으로
      추적하기 쉽게 한다 (LLM에게는 힌트 정도).
    - 환경변수 LLM_MAX_PROMPT_CHARS로 최대 길이를 설정할 수 있다 (기본값: 50000자).
    - 프롬프트가 너무 길면 중요한 키워드가 포함된 블록을 우선 선택한다.
    """
    # 재귀 깊이 제한 (무한 재귀 방지)
    if recursion_depth >= 3:
        print(f"[WARN] 재귀 깊이 제한({recursion_depth})에 도달했습니다. 현재 블록 수로 반환합니다.")
        lines: List[str] = []
        for b in blocks:
            if not b.text:
                continue
            prefix = f"[p{b.page:02d} l{b.line_no:03d}] "
            lines.append(prefix + b.text.strip())
        return "\n".join(lines)
    
    # 환경변수에서 최대 길이 확인 (기본값: 25000자, 전체 프롬프트 최대 길이)
    # 프롬프트 오버헤드(instruction + example 등)를 고려하여 contract_text 최대 길이 조정
    from app.llm.prompt_templates import estimate_prompt_overhead
    
    total_max_chars = None
    if max_chars is None:
        env_max = os.getenv("LLM_MAX_PROMPT_CHARS")
        if env_max:
            try:
                total_max_chars = int(env_max)
            except ValueError:
                total_max_chars = 25000  # 기본값: 전체 프롬프트 최대 길이
        else:
            total_max_chars = 25000  # 기본값
    
    # 프롬프트 오버헤드 계산 (instruction + example + 마지막 지시문 등)
    overhead = estimate_prompt_overhead()
    # contract_text 최대 길이 = 전체 최대 길이 - 오버헤드 (여유를 위해 500자 더 빼기)
    calculated_max_chars = total_max_chars - overhead - 500 if total_max_chars else (25000 - overhead - 500)
    # max_chars가 명시적으로 전달된 경우 그것을 우선 사용
    effective_max_chars = max_chars if max_chars is not None else calculated_max_chars
    
    if recursion_depth == 0:
        print(f"[INFO] 프롬프트 오버헤드: {overhead:,}자, contract_text 최대 길이: {effective_max_chars:,}자")
    
    lines: List[str] = []
    for b in blocks:
        if not b.text:
            continue
        prefix = f"[p{b.page:02d} l{b.line_no:03d}] "
        lines.append(prefix + b.text.strip())

    joined = "\n".join(lines)
    
    # 프롬프트 길이 정보 출력 (재귀 깊이 0일 때만 상세 출력)
    if recursion_depth == 0:
        print(f"[INFO] 프롬프트 길이: {len(joined):,}자 (제한: {effective_max_chars:,}자)")
    
    # 길이가 제한 이하면 그대로 반환
    if len(joined) <= effective_max_chars:
        return joined
    
    # 프롬프트가 너무 길면 스마트 필터링 적용
    if recursion_depth == 0:
        print(f"[WARN] 프롬프트가 {len(joined):,}자로 너무 깁니다 (제한: {effective_max_chars:,}자). 중요한 블록만 선택합니다.")
    filtered_blocks = _filter_important_blocks(blocks, effective_max_chars)
    
    # 필터링 후에도 길이가 줄어들지 않으면 (모든 블록이 우선순위인 경우) 재귀 중단
    filtered_lines: List[str] = []
    for b in filtered_blocks:
        if not b.text:
            continue
        prefix = f"[p{b.page:02d} l{b.line_no:03d}] "
        filtered_lines.append(prefix + b.text.strip())
    filtered_joined = "\n".join(filtered_lines)
    
    if len(filtered_joined) >= len(joined) * 0.95:  # 길이가 5% 이상 줄어들지 않았으면
        print(f"[WARN] 필터링 후에도 길이가 거의 줄어들지 않았습니다 ({len(joined):,}자 -> {len(filtered_joined):,}자). 현재 상태로 반환합니다.")
        return filtered_joined
    
    return blocks_to_compact_text(filtered_blocks, max_chars=effective_max_chars, recursion_depth=recursion_depth + 1)  # 재귀 호출


def _filter_important_blocks(blocks: List[TextBlock], target_chars: int) -> List[TextBlock]:
    """
    프롬프트가 너무 길 때 중요한 블록만 선택한다.
    
    전략:
    1. 우선순위 블록(중요 키워드 포함)은 절대 제거하지 않음
    2. 우선순위 블록의 근처 3블록도 함께 포함
    3. 나머지 블록만 줄이는 방식
    
    우선순위:
    1. 첫 3페이지 (계약서 시작 부분)
    2. 날짜 관련 핵심 키워드가 포함된 블록
    3. 금액 관련 핵심 키워드가 포함된 블록
    4. 자동갱신 관련 핵심 키워드가 포함된 블록
    5. 나머지 블록
    """
    if not blocks:
        return []
    
    # 중요 키워드 정의 (더 많은 핵심 키워드 추가)
    important_keywords = {
        "날짜": ["체결일", "시작일", "종료일", "유효기간", "계약기간", "작성일자", "계약일", "서명일", "날인일"],
        "금액": ["계약금액", "총액", "총 공사금액", "총 공급가액", "보증금", "선입금", "담보금", "전체 이용대가", "원", "달러"],
        "자동갱신": ["자동갱신", "자동연장", "갱신", "연장"],
        "계약": ["계약서", "계약명", "제목", "서명", "체결", "당사자"],
    }
    
    # 블록 인덱스 맵 생성 (주변 블록 찾기용)
    block_index_map = {id(block): idx for idx, block in enumerate(blocks)}
    
    # 1단계: 우선순위 블록 식별 (절대 제거하지 않을 블록)
    priority_block_indices: set[int] = set()
    
    for idx, block in enumerate(blocks):
        if not block.text:
            continue
        
        text_lower = block.text.lower()
        text = block.text
        is_priority = False
        
        # 첫 3페이지는 우선 포함
        if block.page <= 3:
            priority_block_indices.add(idx)
            is_priority = True
        else:
            # 숫자가 포함된 블록도 우선 포함 (금액, 날짜, 기간 등 중요한 정보)
            # 아라비아 숫자 또는 한글 숫자 패턴
            has_number = (
                re.search(r'\d', text) or  # 아라비아 숫자
                re.search(r'[일이삼사오육칠팔구십백천만억]', text) or  # 한글 숫자
                re.search(r'일금|오천|일만|이만|삼만|사만|오만|일억|이억', text)  # 한글 금액 표현
            )
            if has_number:
                priority_block_indices.add(idx)
                is_priority = True
            else:
                # 중요 키워드 확인
                for category, keywords in important_keywords.items():
                    if any(kw in text_lower or kw in text for kw in keywords):
                        priority_block_indices.add(idx)
                        is_priority = True
                        break
        
        # 우선순위 블록의 근처 3블록도 포함
        if is_priority:
            for offset in [-3, -2, -1, 1, 2, 3]:
                neighbor_idx = idx + offset
                if 0 <= neighbor_idx < len(blocks):
                    priority_block_indices.add(neighbor_idx)
    
    # 2단계: 우선순위 블록과 나머지 블록 분리
    priority_blocks: List[TextBlock] = []
    other_blocks: List[TextBlock] = []
    
    for idx, block in enumerate(blocks):
        if not block.text:
            continue
        
        if idx in priority_block_indices:
            priority_blocks.append(block)
        else:
            other_blocks.append(block)
    
    # 3단계: 우선순위 블록 처리
    # 우선순위 블록이 target_chars를 초과하는 경우, 우선순위 내에서도 선택적으로 포함
    selected: List[TextBlock] = []
    current_length = 0
    
    # 우선순위 블록도 길이에 맞춰서 선택적으로 포함 (페이지 순서대로)
    priority_blocks_sorted = sorted(priority_blocks, key=lambda b: (b.page, b.line_no))
    
    for block in priority_blocks_sorted:
        prefix = f"[p{block.page:02d} l{block.line_no:03d}] "
        block_text = prefix + (block.text or "").strip() + "\n"
        block_length = len(block_text)
        
        # target_chars의 90%까지는 우선순위 블록 모두 포함
        if current_length + block_length <= target_chars * 0.9:
            selected.append(block)
            current_length += block_length
        elif current_length + block_length <= target_chars:
            # 90%~100% 사이는 선택적으로 포함
            selected.append(block)
            current_length += block_length
        else:
            # 초과하는 경우 중단
            break
    
    # 4단계: 나머지 블록 추가
    remaining = target_chars - current_length
    
    for block in other_blocks:
        prefix = f"[p{block.page:02d} l{block.line_no:03d}] "
        block_text = prefix + (block.text or "").strip() + "\n"
        if len(block_text) <= remaining:
            selected.append(block)
            remaining -= len(block_text)
        else:
            break
    
    # 블록을 원래 순서대로 정렬
    selected = sorted(selected, key=lambda b: (b.page, b.line_no))
    
    final_length = sum(
        len(f"[p{b.page:02d} l{b.line_no:03d}] {(b.text or '').strip()}\n")
        for b in selected
    )
    
    print(f"[INFO] {len(blocks)}개 블록 중 {len(selected)}개 블록 선택됨 (우선순위: {len(priority_blocks)}개, 기타: {len(selected) - len(priority_blocks)}개, 약 {final_length}자)")
    
    return selected


def _extract_response_text(llm_response: Any) -> str:
    """
    date_llm_extractor.parse_llm_date_response와 비슷한 역할:
    - Bedrock 프록시에서 내려오는 다양한 형태의 응답(dict/str)을 받아
      "실제 LLM 출력 텍스트"만 뽑아낸다.
    """
    if llm_response is None:
        return ""

    # 문자열이면 그대로
    if isinstance(llm_response, str):
        return llm_response

    if isinstance(llm_response, dict):
        # date_llm_extractor에서 지원하는 구조를 그대로 따라감
        if "data" in llm_response and isinstance(llm_response["data"], dict):
            # {'ok': True, 'data': {'result': '...'}} 형태
            text = (
                llm_response["data"].get("output")
                or llm_response["data"].get("result")
                or llm_response["data"].get("text")
                or llm_response["data"].get("content", "")
            )
            return text or ""
        # 여러 가지 키 후보를 순서대로 체크
        for key in ["output", "result", "response", "text", "content"]:
            if key in llm_response:
                return str(llm_response[key])
        # fallback
        return str(llm_response)

    # 그 외 타입은 그냥 문자열로 변환
    return str(llm_response)


def _extract_json_from_text(response_text: str) -> Optional[Dict[str, Any]]:
    """
    LLM 응답 텍스트에서 JSON 객체 부분만 뽑아서 dict로 파싱한다.

    - <result> 태그가 있으면 그 안의 내용만 추출한다.
    - 태그가 없으면 응답 전체에서 JSON을 찾는다.
    - "설명 + \\n { ...json... }"일 수도 있어서,
      제일 앞의 '{'부터 마지막 '}'까지를 잡아서 json.loads를 시도한다.
    """
    if not response_text:
        return None

    # <result> 태그가 있는 경우, 그 안의 내용만 추출
    result_match = re.search(r'<result>(.*?)</result>', response_text, re.DOTALL | re.IGNORECASE)
    if result_match:
        response_text = result_match.group(1).strip()

    start = response_text.find("{")
    end = response_text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    json_str = response_text[start : end + 1]
    try:
        return json.loads(json_str)
    except Exception as e:
        print(f"[WARN] LLM JSON 파싱 실패: {e}")
        print(f"[DEBUG] raw snippet: {json_str[:200]}...")
        return None


def _calculate_amount_expression(expr: str) -> Optional[str]:
    """
    계산식 문자열을 파싱하고 계산합니다.
    
    다양한 계산 패턴 지원:
    - 반복 지급: "10000000 + 1000000 * 12" → "22000000"
    - 비율 역산: "5000000 / 0.05" 또는 "5000000 * 100 / 5" → "100000000"
    - 단위 포함: "500만원 + 1000000 * 12" → 계산 후 숫자
    - 혼합 계산식
    
    Args:
        expr: 계산식 문자열
    
    Returns:
        Optional[str]: 계산된 금액(문자열) 또는 None (계산 실패 시)
    """
    if not expr or not expr.strip():
        return None
    
    try:
        normalized = expr.strip()
        
        # 1) 단위 정규화 (만원=10000, 천원=1000, 억=100000000)
        normalized = re.sub(r'(\d+(?:,\d+)*)\s*억', lambda m: str(int(m.group(1).replace(',', '')) * 100000000), normalized)
        normalized = re.sub(r'(\d+(?:,\d+)*)\s*만원', lambda m: str(int(m.group(1).replace(',', '')) * 10000), normalized)
        normalized = re.sub(r'(\d+(?:,\d+)*)\s*만', lambda m: str(int(m.group(1).replace(',', '')) * 10000), normalized)
        normalized = re.sub(r'(\d+(?:,\d+)*)\s*천원', lambda m: str(int(m.group(1).replace(',', '')) * 1000), normalized)
        normalized = re.sub(r'(\d+(?:,\d+)*)\s*천', lambda m: str(int(m.group(1).replace(',', '')) * 1000), normalized)
        
        # 2) 비율 표현 정규화 (% 처리)
        # "5000000 / 5%" → "5000000 / 0.05"
        normalized = re.sub(r'(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*%', 
                          lambda m: f"{m.group(1)} / ({m.group(2)} / 100)", normalized)
        # "5000000 * 100 / 5" 형태는 그대로 유지
        
        # 3) 변수명이나 한글이 포함되어 있으면 계산 불가 (미리 체크)
        if re.search(r'[가-힣a-zA-Z_]+', normalized):
            # 한글, 영문자, 언더스코어가 있으면 변수명이나 설명이 포함된 것으로 판단
            print(f"[WARN] 계산식에 변수명이나 한글이 포함되어 계산 불가: {expr}")
            return None
        
        # 4) 한글 단어 및 설명 제거 (보증금, 월료, 선입금 등) - 추가 안전장치
        normalized = re.sub(r'[가-힣]+', '', normalized)
        
        # 5) 영문자/변수명 제거 (추가 안전장치)
        normalized = re.sub(r'[a-zA-Z_]+', '', normalized)
        
        # 6) 콤마 제거
        normalized = normalized.replace(',', '')
        
        # 7) 연산자 정규화 (×, ÷ → *, /)
        normalized = normalized.replace('×', '*').replace('÷', '/')
        
        # 8) 허용된 문자만 남기기 (숫자, 연산자, 공백, 소수점, 괄호)
        normalized = re.sub(r'[^\d\+\-\*/\.\(\)\s]', '', normalized)
        
        # 9) 공백 제거
        normalized = normalized.replace(' ', '')
        
        if not normalized:
            return None
        
        # 10) 안전성 검증: 숫자와 연산자만 포함된 표현식인지 확인
        if not re.match(r'^[\d\+\-\*/\.\(\)]+$', normalized):
            return None
        
        # 9) 계산 수행 (eval은 제한적이지만 계산식이므로 사용)
        # 연산 우선순위를 고려하기 위해 eval 사용
        result = eval(normalized)
        
        # 10) 결과 검증 및 반환
        if isinstance(result, (int, float)):
            if result <= 0:
                # 음수나 0이면 잘못된 계산일 가능성
                return None
            return str(int(result))
        else:
            return None
            
    except (ValueError, ZeroDivisionError, SyntaxError, TypeError) as e:
        # 계산 실패 시 None 반환
        print(f"[WARN] 금액 계산식 파싱 실패: {expr} → {e}")
        return None
    except Exception as e:
        # 예상치 못한 에러도 처리
        print(f"[WARN] 금액 계산식 처리 중 예상치 못한 에러: {expr} → {e}")
        return None


def _normalize_fields(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    LLM이 준 원시 JSON(raw)을 받아서
    - 날짜: YYYYMMDD로 정규화
    - 금액: 콤마/공백 제거
    - 통화: KRW/USD/EUR... 으로 정규화
    - 자동갱신 단위: YEAR/MONTH 대문자로 통일
    등 후처리를 수행한다.
    """
    out: Dict[str, Any] = {}

    def _get(key: str, default=None):
        return raw.get(key, default) if isinstance(raw, dict) else default

    # 1) 이름
    out["CNT_NAME"] = (_get("CNT_NAME") or "").strip() or None

    # 2) 날짜들
    for k in ["CNT_CON_DATE", "CNT_ST_DATE", "CNT_END_DATE"]:
        v = _get(k)
        if v is None:
            out[k] = None
        else:
            out[k] = _to_yyyymmdd(str(v))

    # 3) 체결 여부 / 재계약 여부 (O/X)
    # 체결 여부: 항상 None (LLM이 뭘 줬든 무시)
    out["CNT_EXEC_FLAG"] = None

    # 재계약 여부: 기존 로직 유지
    v = (_get("CNT_RENEWAL") or "").strip().upper()
    if v not in {"O", "X"}:
        # 애매하면 X로
        v = "X"
    out["CNT_RENEWAL"] = v
    


    # 4) 금액
    amt = _get("CNT_AMT")
    if amt is None:
        out["CNT_AMT"] = None
    else:
        # LLM이 객체 형태로 반환한 경우 value 필드 추출
        if isinstance(amt, dict):
            amt_value = amt.get("value") if isinstance(amt, dict) else amt
            # llm_reason은 나중에 _attach_llm_reasons에서 처리되므로 여기서는 value만 사용
            amt = amt_value
        
        if amt is None:
            out["CNT_AMT"] = None
        else:
            s = str(amt).strip()
            
            # 순수 숫자 문자열인지 확인 (콤마와 공백 제거 후)
            s_no_spaces = s.replace(",", "").replace(" ", "")
            if s_no_spaces.isdigit():
                # 순수 숫자면 그대로 사용
                out["CNT_AMT"] = s_no_spaces
            else:
                # 계산식이면 파싱하고 계산 시도
                calculated = _calculate_amount_expression(s)
                out["CNT_AMT"] = calculated  # 계산 실패 시 None

    # 5) 통화
    ccy = _get("CNT_AMT_CRY")
    if ccy is None:
        out["CNT_AMT_CRY"] = None
    else:
        out["CNT_AMT_CRY"] = _norm_currency(str(ccy))

    # 6) 자동갱신 숫자/단위
    num = _get("CNT_AUTO_RNW_TERM_NUM")
    try:
        out["CNT_AUTO_RNW_TERM_NUM"] = int(num) if num is not None else None
    except Exception:
        out["CNT_AUTO_RNW_TERM_NUM"] = None

    unit = _get("CNT_AUTO_RNW_TERM_UNIT")
    if unit is None:
        out["CNT_AUTO_RNW_TERM_UNIT"] = None
    else:
        u = str(unit).strip().upper()
        # LLM이 "YEAR", "YEARS", "연", "년" 등으로 줄 수 있으니 정규화
        if any(x in u for x in ["YEAR", "연", "년"]):
            out["CNT_AUTO_RNW_TERM_UNIT"] = "YEAR"
        elif any(x in u for x in ["MONTH", "개월", "달"]):
            out["CNT_AUTO_RNW_TERM_UNIT"] = "MONTH"
        else:
            out["CNT_AUTO_RNW_TERM_UNIT"] = u  # 잘 모르겠으면 그대로

    return out


def _make_empty_llm_result() -> Dict[str, Dict[str, Any]]:
    """
    모든 필드를 기본값(None, 0.0)으로 채운 결과 dict를 만든다.
    """
    return {
        field: {
            "value": None,
            "confidence": 0.0,
            "note": "llm_no_evidence",
            "llm_reason": None,
        }
        for field in FIELDS
    }


def _apply_confidence(normalized: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    정규화된 값 dict(normalized)를 받아서
    필드별로 간단한 confidence 점수를 붙인다.

    - 이 부분은 완전 heuristic이므로, 나중에 rule-based와 merge할 때
      원하는 방식으로 조정하면 된다.
    """
    result = _make_empty_llm_result()

    def _conf_date(v):
        return 0.9 if v else 0.0

    def _conf_amt(v):
        return 0.85 if v else 0.0

    def _conf_flag(v):
        return 0.8 if v in ("O", "X") else 0.0

    for k, v in normalized.items():
        if k not in result:
            continue

        if k in {"CNT_CON_DATE", "CNT_ST_DATE", "CNT_END_DATE"}:
            conf = _conf_date(v)
        elif k in {"CNT_AMT", "CNT_AMT_CRY"}:
            conf = _conf_amt(v)
        elif k in {"CNT_EXEC_FLAG", "CNT_RENEWAL"}:
            conf = _conf_flag(v)
        elif k in {"CNT_AUTO_RNW_TERM_NUM", "CNT_AUTO_RNW_TERM_UNIT"}:
            conf = 0.75 if v is not None else 0.0
        else:  # CNT_NAME 등
            conf = 0.7 if v else 0.0

        result[k]["value"] = v
        result[k]["confidence"] = conf

    return result


def _save_llm_io(prompt: str, llm_response: Any, response_text: str, raw_json: Optional[Dict[str, Any]] = None):
    """
    LLM의 input(프롬프트)과 output(응답)을 파일로 저장한다.
    
    Args:
        prompt: LLM에 전송한 프롬프트
        llm_response: LLM API 원본 응답
        response_text: 추출된 응답 텍스트
        raw_json: 파싱된 JSON (있는 경우)
    """
    if not LOG_LLM_IO:
        return
    
    try:
        LLM_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        
        # 타임스탬프 기반 파일명 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        # Input 저장 (프롬프트)
        input_file = LLM_DEBUG_DIR / f"{timestamp}_input_prompt.txt"
        with input_file.open("w", encoding="utf-8") as f:
            f.write("=== LLM Input (Prompt) ===\n")
            f.write(f"생성 시간: {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")
            f.write(prompt)
        
        # Output 저장 (원본 응답)
        output_file = LLM_DEBUG_DIR / f"{timestamp}_output_raw.json"
        with output_file.open("w", encoding="utf-8") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "raw_response": llm_response,
                "extracted_text": response_text,
                "parsed_json": raw_json
            }, f, ensure_ascii=False, indent=2, default=str)
        
        # Output 텍스트만 저장 (읽기 쉽게)
        output_text_file = LLM_DEBUG_DIR / f"{timestamp}_output_text.txt"
        with output_text_file.open("w", encoding="utf-8") as f:
            f.write("=== LLM Output (Response Text) ===\n")
            f.write(f"생성 시간: {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")
            f.write(response_text)
            # response_text에 <result> 태그가 있으면 Parsed JSON은 중복이므로 생략
            # <result> 태그가 없을 때만 Parsed JSON 추가
            if raw_json and "<result>" not in response_text:
                f.write("\n\n=== Parsed JSON ===\n")
                f.write(json.dumps(raw_json, ensure_ascii=False, indent=2))
        
        print(f"[INFO] LLM I/O 저장됨: {timestamp}")
        print(f"  - Input: {input_file.name}")
        print(f"  - Output: {output_text_file.name}")
        
    except Exception as e:
        print(f"[WARN] LLM I/O 저장 실패: {e}")
        
        
def _parse_thinking_tag(response_text: str) -> Dict[str, str]:
    """
    LLM 응답의 <thinking> 태그 내용을 파싱하여 각 필드별 설명을 추출합니다.
    
    Args:
        response_text: LLM 응답 텍스트 (<thinking> 태그 포함)
    
    Returns:
        Dict[str, str]: 필드명 -> 설명 텍스트 매핑
    """
    field_reasons: Dict[str, str] = {}
    
    if not response_text:
        return field_reasons
    
    # <thinking> 태그 내용 추출
    thinking_match = re.search(r'<thinking>(.*?)</thinking>', response_text, re.DOTALL | re.IGNORECASE)
    if not thinking_match:
        return field_reasons
    
    thinking_content = thinking_match.group(1).strip()
    
    # 필드명 매핑 (한글/영문 키워드 -> 필드명)
    field_keywords = {
        "CNT_NAME": ["제목", "계약명", "계약서", "이름"],
        "CNT_CON_DATE": ["체결일", "계약일", "작성일자", "서명일", "날인일"],
        "CNT_ST_DATE": ["시작일", "유효기간 시작", "계약 시작"],
        "CNT_END_DATE": ["종료일", "유효기간 끝", "계약 종료", "까지"],
        "CNT_RENEWAL": ["재계약", "갱신", "연장"],
        "CNT_AMT": ["금액", "계약금액", "총액", "총 공사금액", "총 공급가액","보증금","선급금","로열티","대가","대금","전체 금액"],
        "CNT_AMT_CRY": ["통화", "원", "달러", "USD", "KRW"],
        "CNT_AUTO_RNW_TERM_NUM": ["자동","자동갱신", "자동연장", "갱신 기간"],
        "CNT_AUTO_RNW_TERM_UNIT": ["자동","자동갱신", "자동연장", "단위"],
    }
    
    # 각 줄을 분석하여 필드별 설명 추출
    lines = thinking_content.split('\n')
    for line in lines:
        line = line.strip()
        if not line or not line.startswith('-'):
            continue
        
        # "- 필드명: 설명" 형태 파싱
        for field, keywords in field_keywords.items():
            for keyword in keywords:
                # 키워드가 포함된 줄 찾기
                if keyword in line:
                    # 이미 해당 필드에 대한 설명이 있으면 더 긴 설명 사용
                    if field not in field_reasons or len(line) > len(field_reasons[field]):
                        # "- " 제거하고 설명만 추출
                        reason = line.lstrip('- ').strip()
                        field_reasons[field] = reason
                    break
    
    return field_reasons


def _attach_llm_reasons(
    result: Dict[str, Dict[str, Any]],
    response_text: str,
    raw_json: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    LLM 응답의 <thinking> 태그를 파싱하여 각 필드의 'llm_reason'에 붙입니다.
    만약 raw_json에서 객체 형태로 llm_reason이 제공되면 그것을 우선 사용합니다.
    
    Args:
        result: 필드별 결과 dict
        response_text: LLM 응답 텍스트 (<thinking> 태그 포함)
        raw_json: LLM 원본 JSON 응답 (객체 형태의 llm_reason 포함 가능)
    
    Returns:
        Dict[str, Dict[str, Any]]: llm_reason이 추가된 result
    """
    # 1) raw_json에서 객체 형태의 llm_reason 추출 (우선순위 높음)
    if raw_json and isinstance(raw_json, dict):
        for field in FIELDS:
            field_value = raw_json.get(field)
            if isinstance(field_value, dict) and "llm_reason" in field_value:
                result[field]["llm_reason"] = field_value["llm_reason"]
    
    # 2) <thinking> 태그에서 필드별 설명 추출 (보조)
    field_reasons = _parse_thinking_tag(response_text)
    
    # 각 필드에 reason 할당 (이미 객체에서 온 경우 덮어쓰지 않음)
    for field in FIELDS:
        if field in field_reasons and result[field].get("llm_reason") is None:
            result[field]["llm_reason"] = field_reasons[field]
    
    return result


# ---------------------------------------------------------------------
# 공개 함수
# ---------------------------------------------------------------------

def extract_fields_with_llm_from_blocks(
    blocks: List[TextBlock],
    use_cache: bool = True,
    max_retries: int = 2,
) -> Dict[str, Dict[str, Any]]:
    """
    TextBlock 리스트를 입력으로 받아
    LLM을 한 번 호출하여 전체 필드를 추출한다.
    
    프롬프트가 너무 길어서 실패하면 자동으로 블록을 줄여가며 재시도한다.

    Args:
        blocks: OCR/파싱을 통해 얻은 TextBlock 리스트
        use_cache: 같은 계약서를 여러 번 처리할 때 LLM 호출을 피하기 위한 캐시 사용 여부
        max_retries: 프롬프트 길이로 인한 실패 시 최대 재시도 횟수 (기본값: 3)

    Returns:
        Dict[str, Dict[str, Any]]: 필드별 {value, confidence, note, llm_reason} 구조의 dict
    """
    if not blocks:
        return _make_empty_llm_result()

    cache_key = _hash_blocks(blocks) if use_cache else None
    if use_cache and cache_key in _FIELDS_LLM_CACHE:
        return _FIELDS_LLM_CACHE[cache_key]

    # 재시도 로직: 프롬프트 길이를 점진적으로 줄여가며 시도
    current_blocks = blocks
    retry_count = 0
    
    from app.llm.prompt_templates import estimate_prompt_overhead
    overhead = estimate_prompt_overhead()
    total_max_chars = int(os.getenv("LLM_MAX_PROMPT_CHARS", "25000"))
    
    while retry_count <= max_retries:
        try:
            # 1) 블록을 LLM용 텍스트로 변환
            # 재시도 시 max_chars를 점진적으로 줄임 (각 재시도마다 20%씩 감소)
            
            if retry_count > 0:
                # 재시도 시: 전체 최대 길이를 점진적으로 줄임
                reduction_factor = 0.8 ** retry_count  # 0.8, 0.64, 0.512...
                adjusted_max = int(total_max_chars * reduction_factor)
                print(f"[INFO] 재시도 {retry_count}/{max_retries}: 최대 프롬프트 길이를 {adjusted_max:,}자로 조정 (원본: {total_max_chars:,}자)")
                contract_text_max_chars = adjusted_max - overhead - 500
                
                # 블록을 더 줄여서 재시도 (더 엄격하게 필터링)
                print(f"[INFO] 블록을 필터링하여 줄입니다...")
                current_blocks = _filter_important_blocks(current_blocks, contract_text_max_chars)
                
                if len(current_blocks) == 0:
                    print(f"[ERROR] 재시도 후에도 블록이 0개가 되었습니다. 처리를 중단합니다.")
                    return _make_empty_llm_result()
                
                print(f"[INFO] 필터링 완료: {len(current_blocks)}개 블록")
            else:
                # 첫 시도: 전체 프롬프트 길이를 미리 예상해서 블록을 줄일지 결정
                contract_text_max_chars = None  # 기본값 사용
            
            contract_text = blocks_to_compact_text(current_blocks, max_chars=contract_text_max_chars)

            # 2) 프롬프트 구성
            prompt = build_full_fields_prompt(contract_text)
            
            # 전체 프롬프트 길이 확인 및 출력
            print(f"[INFO] 전체 프롬프트 길이: {len(prompt):,}자 (contract_text: {len(contract_text):,}자)")

            # 3) Bedrock API 호출 (date_llm_extractor와 동일한 엔드포인트 사용)
            # 프롬프트가 너무 길면 경고
            if len(prompt) > 25000:
                print(f"[WARN] 프롬프트가 {len(prompt):,}자로 매우 깁니다. API 제한에 걸릴 수 있습니다.")
            # 추출 작업(Extraction)에서는 일관성을 위해 temperature=0 사용
            llm_response = call_bedrock_api(prompt, temperature=0.0)
            
            # 성공 시: 결과 처리
            break  # while 루프 탈출하여 아래 결과 처리 코드로 이동
            
        except Exception as e:
            error_msg = str(e)
            
            # prompt 변수가 정의되었는지 확인
            prompt_len = len(prompt) if 'prompt' in locals() else 0
            
            # 서버 에러나 프롬프트 길이 관련 에러인 경우 재시도
            is_retryable = (
                "500" in error_msg or
                "502" in error_msg or
                "503" in error_msg or
                "504" in error_msg or
                "timeout" in error_msg.lower() or
                "too long" in error_msg.lower() or
                "length" in error_msg.lower() or
                "service unavailable" in error_msg.lower()
            )
            
            if is_retryable and retry_count < max_retries:
                retry_count += 1
                print(f"[WARN] LLM 호출 실패 (재시도 가능): {error_msg}")
                if prompt_len > 0:
                    print(f"[WARN] 프롬프트 길이 {prompt_len:,}자 - 재시도 {retry_count}/{max_retries}...")
                else:
                    print(f"[WARN] 재시도 {retry_count}/{max_retries}...")
                
                # 503, 502 등 서버 에러의 경우 지수 백오프로 대기 (서버 복구 시간 확보)
                if any(err in error_msg for err in ["500", "502", "503", "504", "service unavailable"]):
                    # 지수 백오프: 1초, 2초, 4초...
                    wait_time = min(2 ** (retry_count - 1), 10)  # 최대 10초
                    print(f"[INFO] 서버 에러 감지 - {wait_time}초 대기 후 재시도...")
                    time.sleep(wait_time)
                elif "timeout" in error_msg.lower():
                    # 타임아웃의 경우 짧은 대기
                    wait_time = min(1 * retry_count, 5)  # 최대 5초
                    print(f"[INFO] 타임아웃 감지 - {wait_time}초 대기 후 재시도...")
                    time.sleep(wait_time)
                # 프롬프트 길이 관련 에러는 즉시 재시도 (블록 줄이기)
                
                # 다음 재시도에서는 블록을 줄이도록 설정 (while 루프의 다음 반복에서 처리됨)
                continue  # while 루프 계속 (재시도)
            else:
                # 재시도 불가능한 에러이거나 최대 재시도 횟수 초과
                if any(err in error_msg for err in ["500", "502", "503", "504"]):
                    print(f"[ERROR] LLM 전체 필드 호출 실패 (서버 에러): {error_msg}")
                    if prompt_len > 0:
                        print(f"[ERROR] 프롬프트 길이: {prompt_len:,}자")
                    if retry_count >= max_retries:
                        print(f"[ERROR] 재시도 횟수 초과 ({retry_count}/{max_retries})")
                    else:
                        print(f"[ERROR] 재시도 불가능한 에러입니다.")
                else:
                    print(f"[ERROR] LLM 전체 필드 호출 실패: {error_msg}")
                    import traceback
                    traceback.print_exc()
                return _make_empty_llm_result()

    # 4) 응답 텍스트 추출 후 JSON 파싱
    response_text = _extract_response_text(llm_response)
    raw_json = _extract_json_from_text(response_text)
    
    # LLM I/O 저장 (로깅)
    _save_llm_io(prompt, llm_response, response_text, raw_json)
    
    if raw_json is None:
        print("[WARN] LLM 응답에서 유효한 JSON을 찾지 못했습니다.")
        return _make_empty_llm_result()

    # 5) 정규화 & confidence 부여
    normalized = _normalize_fields(raw_json)
    result = _apply_confidence(normalized)
    
    # 6) LLM 근거(reason) 붙이기: <thinking> 태그에서 파싱 + 객체의 llm_reason 우선 사용
    result = _attach_llm_reasons(result, response_text, raw_json)

    if use_cache and cache_key:
        _FIELDS_LLM_CACHE[cache_key] = result

    return result
