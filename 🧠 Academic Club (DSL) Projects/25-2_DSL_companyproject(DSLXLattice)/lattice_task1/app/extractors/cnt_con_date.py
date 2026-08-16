# -*- coding: utf-8 -*-
from typing import List
import re
from app.extractors.base import FieldExtractor, Candidate, Evidence
from app.ingest.loader_pdf import TextBlock
from app.utils.regexes import (
    RE_DATE_GENERIC, RE_CON_LABEL, RE_SIGN_CONTEXT, RE_RANGE1, RE_RANGE2
)
from app.ingest.date_llm_extractor import extract_dates_with_llm, collect_date_related_texts

class ConDateExtractor(FieldExtractor):
    """
    계약 체결일 추출기 (LLM 기반, 공통 모듈 사용):
    공통 날짜 LLM 추출 모듈에서 체결일만 추출하여 Candidate로 변환
    """

    def extract(self, blocks: List[TextBlock]) -> List[Candidate]:
        cands: List[Candidate] = []
        if not blocks:
            return cands

        # 공통 LLM 모듈에서 날짜 추출 (체결일, 시작일, 종료일 모두 추출)
        date_results = extract_dates_with_llm(blocks)
        extracted_date = date_results.get("con_date")
        
        if not extracted_date:
            # LLM이 체결일을 찾지 못했으면 Fallback
            print(f"[DEBUG] ConDateExtractor LLM이 체결일을 찾지 못함, Fallback 로직 사용")
            return self._fallback_extract(blocks)
        
        # 추출된 날짜가 포함된 텍스트 블록 찾기
        date_texts = collect_date_related_texts(blocks)
        matching_blocks = []
        normalized_extracted = extracted_date.replace(" ", "").replace(".", "").replace("-", "").replace("년", "").replace("월", "").replace("일", "")
        
        for page, line_no, text, block in date_texts:
            # 기간 라인은 체결일 후보에서 제외
            if RE_RANGE1.search(text) or RE_RANGE2.search(text):
                continue
            # 추출된 날짜가 텍스트에 포함되어 있는지 확인
            text_normalized = text.replace(" ", "").replace(".", "").replace("-", "").replace("년", "").replace("월", "").replace("일", "")
            if extracted_date in text or normalized_extracted in text_normalized:
                if RE_DATE_GENERIC.search(text):
                    score = 0
                    if RE_CON_LABEL.search(text):
                        score += 10
                    if RE_SIGN_CONTEXT.search(text):
                        score += 5
                    matching_blocks.append((score, page, line_no, block, text))
        
        # 매칭된 블록이 있으면 사용
        if matching_blocks:
            matching_blocks.sort(key=lambda x: x[0], reverse=True)
            best_score, best_page, best_line, best_block, best_text = matching_blocks[0]
        else:
            # 매칭 블록이 없으면 가장 관련성 높은 블록 찾기
            best_block = None
            best_score = 0
            best_page = 1
            best_line = 1
            
            for page, line_no, text, block in date_texts:
                if RE_RANGE1.search(text) or RE_RANGE2.search(text):
                    continue
                score = 0
                if RE_CON_LABEL.search(text):
                    score += 10
                if RE_SIGN_CONTEXT.search(text):
                    score += 5
                if score > best_score:
                    best_score = score
                    best_block = block
                    best_page = page
                    best_line = line_no
            
            if best_block is None:
                # 최적 블록이 없으면 첫 번째 블록 사용
                if date_texts:
                    best_page, best_line, _, best_block = date_texts[0]
                else:
                    return self._fallback_extract(blocks)
        
        # Evidence 생성
        ev = Evidence(page=best_page, lines=[best_line], snippet=best_block.text)
        
        # Features 설정
        features = {"llm_extracted": 1.0, "format": 1.0}
        if RE_CON_LABEL.search(best_block.text):
            features["label"] = 1.0
        if RE_SIGN_CONTEXT.search(best_block.text):
            features["sign_ctx"] = 1.0
        
        last_page = max(b.page for b in blocks)
        if best_page == last_page:
            features["last_page"] = 1.0
        
        # Candidate 생성
        source = "header" if RE_CON_LABEL.search(best_block.text) else "sentence"
        cands.append(Candidate("CNT_CON_DATE", extracted_date, source, ev, features))
        
        return cands
    
    def _is_date_only_block(self, text: str) -> bool:
        """텍스트가 날짜만 있는지 확인 (별 내용 없이 날짜만)"""
        text_clean = text.strip()
        # 날짜 패턴으로 전체 텍스트가 매칭되는지 확인
        date_match = RE_DATE_GENERIC.search(text_clean)
        if not date_match:
            return False
        
        # 날짜 부분을 제거한 나머지 텍스트
        date_part = date_match.group(0)
        remaining = text_clean.replace(date_part, "").strip()
        
        # 나머지가 거의 없거나 (공백, 구두점만) 매우 짧으면 날짜만 있는 것으로 판단
        # 예: "2024년 7월 1일", "2024.07.01", "2024-07-01" 등
        if len(remaining) <= 3:  # 공백, 구두점 등만 남은 경우
            return True
        
        # 나머지가 한글/영문 단어가 아닌 경우 (구두점, 공백만)
        if re.match(r'^[\s\.,;:()\-_]+$', remaining):
            return True
        
        return False
    
    def _fallback_extract(self, blocks: List[TextBlock]) -> List[Candidate]:
        """Fallback: 마지막 부분에서 날짜만 있는 블록 찾기"""
        cands: List[Candidate] = []
        if not blocks:
            return cands
        
        last_page = max(b.page for b in blocks)
        max_line_per_page = {}
        for b in blocks:
            max_line_per_page[b.page] = max(max_line_per_page.get(b.page, 0), b.line_no)
        
        # 1순위: 라벨이 있는 줄 (체결일, 서명일 등)
        for b in blocks:
            text = b.text
            if not RE_CON_LABEL.search(text):
                continue
            if RE_RANGE1.search(text) or RE_RANGE2.search(text):
                continue
            m = RE_DATE_GENERIC.search(text)
            if not m:
                continue
            value = m.group(0)
            ev = Evidence(page=b.page, lines=[b.line_no], snippet=text)
            features = {"label": 1.0, "format": 1.0, "fallback": 1.0}
            if b.page == last_page:
                features["last_page"] = 1.0
            if b.line_no >= max_line_per_page.get(b.page, 0) - 10:
                features["bottom_zone"] = 1.0
            if RE_SIGN_CONTEXT.search(text):
                features["sign_ctx"] = 1.0
            cands.append(Candidate("CNT_CON_DATE", value, "header", ev, features))
            print(f"[DEBUG] Fallback: 라벨 있는 블록에서 체결일 발견 - {value}")
            return cands
        
        # 2순위: 마지막 페이지의 마지막 부분에서 날짜만 있는 블록
        # 마지막 페이지의 하단 20줄 정도를 확인
        bottom_blocks = []
        for b in blocks:
            if b.page == last_page:
                if b.line_no >= max_line_per_page.get(b.page, 0) - 20:
                    if RE_RANGE1.search(b.text) or RE_RANGE2.search(b.text):
                        continue
                    if RE_DATE_GENERIC.search(b.text):
                        bottom_blocks.append((b.line_no, b))
        
        # 라인 번호 역순으로 정렬 (마지막 라인부터)
        bottom_blocks.sort(key=lambda x: x[0], reverse=True)
        
        # 날짜만 있는 블록 찾기
        for line_no, b in bottom_blocks:
            if self._is_date_only_block(b.text):
                m = RE_DATE_GENERIC.search(b.text)
                if m:
                    value = m.group(0)
                    ev = Evidence(page=b.page, lines=[b.line_no], snippet=b.text)
                    features = {"format": 1.0, "fallback": 1.0, "last_page": 1.0, "bottom_zone": 1.0, "date_only": 1.0}
                    cands.append(Candidate("CNT_CON_DATE", value, "sentence", ev, features))
                    print(f"[DEBUG] Fallback: 마지막 페이지 하단에서 날짜만 있는 블록 발견 - {value} (라인 {line_no})")
                    return cands
        
        # 3순위: 마지막 페이지의 하단에서 일반 날짜 블록
        for line_no, b in bottom_blocks:
            m = RE_DATE_GENERIC.search(b.text)
            if m:
                value = m.group(0)
                ev = Evidence(page=b.page, lines=[b.line_no], snippet=b.text)
                features = {"format": 1.0, "fallback": 1.0, "last_page": 1.0, "bottom_zone": 1.0}
                cands.append(Candidate("CNT_CON_DATE", value, "sentence", ev, features))
                print(f"[DEBUG] Fallback: 마지막 페이지 하단에서 날짜 발견 - {value} (라인 {line_no})")
                return cands
        
        # 4순위: 전체 문서에서 마지막 날짜 (기간 범위 제외)
        all_date_blocks = []
        for b in blocks:
            if RE_RANGE1.search(b.text) or RE_RANGE2.search(b.text):
                continue
            m = RE_DATE_GENERIC.search(b.text)
            if m:
                all_date_blocks.append((b.page, b.line_no, b))
        
        # 페이지와 라인 번호 순으로 정렬, 마지막 것 선택
        if all_date_blocks:
            all_date_blocks.sort(key=lambda x: (x[0], x[1]))
            last_page, last_line, last_block = all_date_blocks[-1]
            m = RE_DATE_GENERIC.search(last_block.text)
            if m:
                value = m.group(0)
                ev = Evidence(page=last_page, lines=[last_line], snippet=last_block.text)
                features = {"format": 1.0, "fallback": 1.0}
                if last_page == max(b.page for b in blocks):
                    features["last_page"] = 1.0
                cands.append(Candidate("CNT_CON_DATE", value, "sentence", ev, features))
                print(f"[DEBUG] Fallback: 문서 전체에서 마지막 날짜 발견 - {value} (페이지 {last_page}, 라인 {last_line})")
        
        return cands
