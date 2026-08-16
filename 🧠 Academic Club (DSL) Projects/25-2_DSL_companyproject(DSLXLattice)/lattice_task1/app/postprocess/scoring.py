# -*- coding: utf-8 -*-
from typing import List
from app.extractors.base import Candidate
from app.pipeline.config import WEIGHTS

SOURCE_W = WEIGHTS["source"]

# 구체 힌트 가중치(우선순위 정렬용)
HINT_W = {
    "label": 0.30,       # 체결/서명/날인 라벨
    "format": 0.20,      # 날짜/형식 적합
    "last_page": 0.25,   # 마지막 페이지
    "bottom_zone": 0.20, # 페이지 하단(마지막 10줄)
    "sign_ctx": 0.20,    # 서명/날인/갑/을 등
    "kw": 0.25,          # 금액: 좋은 키워드
    "verb": 0.20,        # 금액: 포함 동사(…으로 한다/지급한다 등)
    "ccy": 0.20,         # 금액: 통화 단서
}

def apply(cands: List[Candidate]) -> List[Candidate]:
    out: List[Candidate] = []
    for c in cands:
        score = 0.0
        score += SOURCE_W.get(c.source, 0.0)
        # 명시 힌트 먼저
        for k, v in c.features.items():
            if k in HINT_W and v:
                score += HINT_W[k]
        # 나머지 힌트는 소량 가점
        for k, v in c.features.items():
            if k not in HINT_W and isinstance(v, (int, float)) and v:
                score += 0.1 * float(v)
        c.score = max(0.0, min(1.0, score))
        out.append(c)
    return out
