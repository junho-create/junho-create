# -*- coding: utf-8 -*-
from typing import List
import re
from app.extractors.base import FieldExtractor, Candidate, Evidence
from app.ingest.loader_pdf import TextBlock
from app.utils.regexes import RE_DATE_GENERIC, RE_RANGE1, RE_RANGE2
from app.ingest.date_llm_extractor import extract_dates_with_llm, collect_date_related_texts

RE_TERM_LABEL = re.compile(r"유효기간|계약기간|Term", re.IGNORECASE)

class PeriodExtractor(FieldExtractor):
    """
    계약 기간 추출기 (LLM 기반, 공통 모듈 사용):
    공통 날짜 LLM 추출 모듈에서 시작일, 종료일만 추출하여 Candidate로 변환
    """
    
    def extract(self, blocks: List[TextBlock]) -> List[Candidate]:
        cands: List[Candidate] = []
        if not blocks:
            return cands
        
        # 우선 LLM 사용 (LLM 결과를 신뢰)
        print(f"[DEBUG] PeriodExtractor: LLM 사용 시도")
        date_results = extract_dates_with_llm(blocks)
        st_date = date_results.get("st_date")
        end_date = date_results.get("end_date")
        
        # LLM에서 시작일/종료일을 찾았으면 사용 (LLM 결과를 그대로 신뢰)
        if st_date and end_date:
            print(f"[DEBUG] PeriodExtractor: LLM에서 시작일({st_date}), 종료일({end_date}) 발견 - LLM 결과를 그대로 사용")
            # 추출된 날짜가 포함된 텍스트 블록 찾기 (Evidence용)
            date_texts = collect_date_related_texts(blocks)
            
            # 정규화된 버전 준비
            st_normalized = st_date.replace(" ", "").replace(".", "").replace("-", "").replace("년", "").replace("월", "").replace("일", "")
            end_normalized = end_date.replace(" ", "").replace(".", "").replace("-", "").replace("년", "").replace("월", "").replace("일", "")
            
            # 시작일 블록 찾기 (Evidence용)
            st_block = None
            st_page = 1
            st_line = 1
            st_score = 0
            
            for page, line_no, text, block in date_texts:
                text_normalized = text.replace(" ", "").replace(".", "").replace("-", "").replace("년", "").replace("월", "").replace("일", "")
                if st_normalized in text_normalized:
                    score = 1
                    if RE_RANGE1.search(text) or RE_RANGE2.search(text):
                        score += 2
                    if RE_TERM_LABEL.search(text):
                        score += 1
                    if score > st_score:
                        st_score = score
                        st_block = block
                        st_page = page
                        st_line = line_no
            
            # 종료일 블록 찾기 (Evidence용)
            end_block = None
            end_page = 1
            end_line = 1
            end_score = 0
            
            for page, line_no, text, block in date_texts:
                text_normalized = text.replace(" ", "").replace(".", "").replace("-", "").replace("년", "").replace("월", "").replace("일", "")
                if end_normalized in text_normalized:
                    score = 1
                    if RE_RANGE1.search(text) or RE_RANGE2.search(text):
                        score += 2
                    if RE_TERM_LABEL.search(text):
                        score += 1
                    if score > end_score:
                        end_score = score
                        end_block = block
                        end_page = page
                        end_line = line_no
            
            # 시작일과 종료일이 모두 포함된 블록 찾기 (우선순위)
            best_block = None
            best_score = 0
            best_page = 1
            best_line = 1
            
            for page, line_no, text, block in date_texts:
                if RE_RANGE1.search(text) or RE_RANGE2.search(text):
                    score = 1
                    if RE_TERM_LABEL.search(text):
                        score += 1
                    text_normalized = text.replace(" ", "").replace(".", "").replace("-", "").replace("년", "").replace("월", "").replace("일", "")
                    if st_normalized in text_normalized and end_normalized in text_normalized:
                        score += 3
                    elif st_normalized in text_normalized or end_normalized in text_normalized:
                        score += 1
                    if score > best_score:
                        best_score = score
                        best_block = block
                        best_page = page
                        best_line = line_no
            
            # LLM에서 추출한 날짜 값을 그대로 사용 (raw_value는 LLM 결과)
            # Evidence는 적절한 블록에서 찾기
            
            # 시작일과 종료일이 모두 포함된 블록이 있으면 그것 사용
            if best_block and st_normalized in best_block.text.replace(" ", "").replace(".", "").replace("-", "").replace("년", "").replace("월", "").replace("일", "") and end_normalized in best_block.text.replace(" ", "").replace(".", "").replace("-", "").replace("년", "").replace("월", "").replace("일", ""):
                # 같은 블록에 둘 다 있으면 같은 Evidence 사용
                ev = Evidence(page=best_page, lines=[best_line], snippet=best_block.text)
                header_hit = 1 if RE_TERM_LABEL.search(best_block.text) else 0
                # LLM에서 추출한 값을 그대로 사용
                cands.append(Candidate("CNT_ST_DATE", st_date, "sentence", ev, {"header": header_hit, "llm_extracted": 1.0}))
                cands.append(Candidate("CNT_END_DATE", end_date, "sentence", ev, {"header": header_hit, "llm_extracted": 1.0}))
                print(f"[DEBUG] PeriodExtractor: 같은 블록에서 Evidence 발견, LLM 값 사용 - 시작일: {st_date}, 종료일: {end_date}")
            else:
                # 다른 블록에 있으면 각각의 Evidence 사용
                if st_block:
                    st_ev = Evidence(page=st_page, lines=[st_line], snippet=st_block.text)
                    st_header = 1 if RE_TERM_LABEL.search(st_block.text) else 0
                else:
                    # 시작일 블록을 못 찾았으면 첫 번째 날짜 블록 또는 기간 블록 사용
                    if date_texts:
                        st_page, st_line, _, st_block = date_texts[0]
                        st_ev = Evidence(page=st_page, lines=[st_line], snippet=st_block.text)
                        st_header = 1 if RE_TERM_LABEL.search(st_block.text) else 0
                    else:
                        # date_texts도 없으면 첫 번째 블록 사용
                        if blocks:
                            st_block = blocks[0]
                            st_ev = Evidence(page=st_block.page, lines=[st_block.line_no], snippet=st_block.text)
                            st_header = 0
                        else:
                            # blocks도 없으면 기본값
                            st_ev = Evidence(page=1, lines=[1], snippet="")
                            st_header = 0
                
                if end_block:
                    end_ev = Evidence(page=end_page, lines=[end_line], snippet=end_block.text)
                    end_header = 1 if RE_TERM_LABEL.search(end_block.text) else 0
                else:
                    # 종료일 블록을 못 찾았으면 마지막 날짜 블록 또는 기간 블록 사용
                    if date_texts:
                        end_page, end_line, _, end_block = date_texts[-1]
                        end_ev = Evidence(page=end_page, lines=[end_line], snippet=end_block.text)
                        end_header = 1 if RE_TERM_LABEL.search(end_block.text) else 0
                    else:
                        # date_texts도 없으면 마지막 블록 사용
                        if blocks:
                            end_block = blocks[-1]
                            end_ev = Evidence(page=end_block.page, lines=[end_block.line_no], snippet=end_block.text)
                            end_header = 0
                        else:
                            # blocks도 없으면 기본값
                            end_ev = Evidence(page=1, lines=[1], snippet="")
                            end_header = 0
                
                # LLM에서 추출한 값을 그대로 사용
                cands.append(Candidate("CNT_ST_DATE", st_date, "sentence", st_ev, {"header": st_header, "llm_extracted": 1.0}))
                cands.append(Candidate("CNT_END_DATE", end_date, "sentence", end_ev, {"header": end_header, "llm_extracted": 1.0}))
                print(f"[DEBUG] PeriodExtractor: 각각의 Evidence 사용, LLM 값 그대로 - 시작일: {st_date}, 종료일: {end_date}")
            
            if cands:
                print(f"[DEBUG] PeriodExtractor: Candidate 생성 완료 - 시작일: {st_date}, 종료일: {end_date}")
                return cands
        
        # LLM이 찾지 못했을 때만 Fallback 사용
        print(f"[DEBUG] PeriodExtractor: LLM에서 찾지 못함, Fallback 로직 사용")
        fallback_cands = self._fallback_extract(blocks)
        if fallback_cands:
            print(f"[DEBUG] PeriodExtractor: Fallback으로 {len(fallback_cands)}개 후보 발견")
        return fallback_cands
    
    def _fallback_extract(self, blocks: List[TextBlock]) -> List[Candidate]:
        """Fallback: 기존 하드코딩 방식"""
        cands: List[Candidate] = []
        for b in blocks:
            text = b.text
            header_hit = 1 if RE_TERM_LABEL.search(text) else 0
            m1 = RE_RANGE1.search(text)
            m2 = RE_RANGE2.search(text)
            if m1:
                st_raw, ed_raw = m1.group(1), m1.group(2)
            elif m2:
                st_raw, ed_raw = m2.group(1), m2.group(3)
            else:
                continue
            ev = Evidence(page=b.page, lines=[b.line_no], snippet=text)
            cands.append(Candidate("CNT_ST_DATE", st_raw, "sentence", ev, {"header": header_hit, "fallback": 1.0}))
            cands.append(Candidate("CNT_END_DATE", ed_raw, "sentence", ev, {"header": header_hit, "fallback": 1.0}))
        return cands