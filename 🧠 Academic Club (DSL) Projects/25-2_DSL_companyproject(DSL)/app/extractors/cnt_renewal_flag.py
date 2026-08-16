# -*- coding: utf-8 -*-
from typing import List
from app.extractors.base import FieldExtractor, Candidate, Evidence
from app.ingest.loader_pdf import TextBlock
from app.utils.regexes import RE_RENEWAL_WORD, RE_PRIOR_REF


class RenewalFlagExtractor(FieldExtractor):
    """재계약 여부(O/X): 재체결 문서일 때만 O.
    자동갱신 문구만 있는 경우는 X로 본다.

    - 재계약/갱신 관련 키워드가 전혀 없으면 기본값을 X로 둔다.
    """

    def extract(self, blocks: List[TextBlock]) -> List[Candidate]:
        cands: List[Candidate] = []

        # 1) 키워드가 포함된 줄에서만 먼저 탐색
        for b in blocks:
            t = b.text or ""
            if RE_RENEWAL_WORD.search(t):
                # 선행 계약 참조가 있으면 강한 O 후보
                value = "O" if RE_PRIOR_REF.search(t) else "X"  # 모호하면 기본 X
                ev = Evidence(page=b.page, lines=[b.line_no], snippet=t)
                cands.append(
                    Candidate(
                        "CNT_RENEWAL",
                        value,
                        "sentence",
                        ev,
                        {"renewal_kw": 1},
                    )
                )

        # 2) 어떤 후보도 없으면 "재계약 없음"을 기본값 X로 가정
        if not cands:
            if blocks:
                first_page = blocks[0].page
            else:
                first_page = 1  # 안전용 기본값

            ev = Evidence(
                page=first_page,
                lines=[],  # 특정 라인에 근거한 것이 아님
                snippet="재계약/갱신 관련 키워드 미검출: 기본값 X",
            )
            cands.append(
                Candidate(
                    "CNT_RENEWAL",
                    "X",
                    "sentence",      # scoring에서 처리 가능한 기존 source 사용
                    ev,
                    {"default": 1},  # 기본 추론임을 표시
                )
            )

        return cands