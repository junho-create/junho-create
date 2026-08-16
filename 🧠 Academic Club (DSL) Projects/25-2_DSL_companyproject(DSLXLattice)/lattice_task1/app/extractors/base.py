# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class Evidence:
    page: int
    lines: List[int]
    snippet: str

@dataclass
class Candidate:
    field: str            # 예: "CNT_ST_DATE"
    raw_value: str        # 정규화 전 값 문자열
    source: str           # "header" | "sentence" | "table" | "title"
    evidence: Evidence    # 근거(페이지/라인/스니펫)
    features: Dict[str, Any]
    score: float = 0.0    # postprocess.scoring에서 채워짐

class FieldExtractor:
    """필드 모듈 공통 인터페이스."""
    def extract(self, blocks: List) -> List[Candidate]:  # blocks: TextBlock 리스트
        raise NotImplementedError