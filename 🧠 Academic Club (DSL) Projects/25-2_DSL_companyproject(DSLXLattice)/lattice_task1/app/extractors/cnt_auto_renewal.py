# -*- coding: utf-8 -*-
from typing import List
from app.extractors.base import FieldExtractor, Candidate, Evidence
from app.ingest.loader_pdf import TextBlock
from app.utils.regexes import RE_AUTO_TRIGGER, RE_AUTO_PERIOD

UNIT_MAP = {
    "년": "YEAR", "year": "YEAR",
    "개월": "MONTH", "월": "MONTH", "month": "MONTH",
    "일": "DAY", "day": "DAY",
}

def _has_auto_trigger_around(blocks: List[TextBlock], idx: int, window: int = 1) -> bool:
    """
    현재 줄(idx)을 기준으로 앞뒤 window 줄까지 합친 컨텍스트에서
    자동갱신 트리거(RE_AUTO_TRIGGER)가 있는지 확인.
    """
    start = max(0, idx - window)
    end = min(len(blocks), idx + window + 1)
    context = " ".join((blocks[i].text or "") for i in range(start, end))
    return bool(RE_AUTO_TRIGGER.search(context))

class AutoRenewalExtractor(FieldExtractor):
    def extract(self, blocks: List[TextBlock]) -> List[Candidate]:
        cands: List[Candidate] = []

        for idx, b in enumerate(blocks):
            t = b.text or ""
            # 기간 표현이 있는 줄에서, 주변 컨텍스트에 자동갱신 트리거가 있는지 확인
            for m in RE_AUTO_PERIOD.finditer(t):
                if not _has_auto_trigger_around(blocks, idx, window=1):
                    continue  # 주변에 자동갱신 문맥이 없으면 스킵

                num_raw, unit_raw = m.group(1), m.group(2)

                # '1년간', '1개월씩' 등에서 '간', '씩' 같은 꼬리는 제거
                unit_key = unit_raw.replace("간", "").replace("씩", "").lower()

                ev = Evidence(page=b.page, lines=[b.line_no], snippet=t)
                cands.append(
                    Candidate(
                        "CNT_AUTO_RNW_TERM_AMT",
                        num_raw,
                        "sentence",
                        ev,
                        {"trigger": 1},
                    )
                )
                cands.append(
                    Candidate(
                        "CNT_AUTO_RNW_TERM_UNIT",
                        UNIT_MAP.get(unit_key, unit_key.upper()),
                        "sentence",
                        ev,
                        {"trigger": 1},
                    )
                )
        return cands