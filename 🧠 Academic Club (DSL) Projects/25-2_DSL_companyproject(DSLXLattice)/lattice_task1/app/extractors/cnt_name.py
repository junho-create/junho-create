# -*- coding: utf-8 -*-
from typing import List
from app.extractors.base import FieldExtractor, Candidate, Evidence
from app.ingest.loader_pdf import TextBlock
from app.utils.text import is_title_zone
from app.utils.regexes import RE_NAME_LABEL

class NameExtractor(FieldExtractor):
    def extract(self, blocks: List[TextBlock]) -> List[Candidate]:
        cands: List[Candidate] = []
        
        # 1단계: "계약명" 라벨 기반 추출 (우선순위 높음)
        # 1-2페이지에서만 검색
        for b in blocks:
            if b.page > 2:
                break  # 2페이지까지만 검색
            
            text = b.text.strip()
            m = RE_NAME_LABEL.search(text)
            if m:
                # 매칭된 그룹 중 None이 아닌 첫 번째 값 사용
                name_value = None
                for group in m.groups():
                    if group:
                        name_value = group.strip()
                        break
                
                if name_value:
                    # 앞뒤 공백 제거 및 불필요한 문자 정리
                    name_value = name_value.strip()
                    # 길이 제한 (너무 긴 경우 제외)
                    if len(name_value) > 200:
                        continue
                    
                    ev = Evidence(page=b.page, lines=[b.line_no], snippet=text)
                    # 라벨 기반 추출은 더 높은 점수를 위해 "label" feature 추가
                    cands.append(Candidate(
                        "CNT_NAME", 
                        name_value, 
                        "header",  # 라벨이 있으면 header로 분류
                        ev, 
                        {"label": 1, "name_label": 1}  # label feature로 점수 가중치 증가
                    ))
                    # 첫 번째 매칭만 사용 (가장 위에 있는 것 우선)
                    break
        
        # 2단계: 기존 타이틀 영역 추출 (라벨 기반 추출이 없을 때만)
        # 라벨 기반 추출이 있으면 기존 로직은 스킵하지 않고 후보로 추가
        # (점수로 우선순위 결정)
        for b in blocks:
            if is_title_zone(b):
                ev = Evidence(page=b.page, lines=[b.line_no], snippet=b.text)
                cands.append(Candidate("CNT_NAME", b.text.strip(), "title", ev, {"title":1}))
        
        return cands