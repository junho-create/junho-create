# app/postprocess/validate.py
# -*- coding: utf-8 -*-
from typing import List, Dict, Optional
from collections import defaultdict
from app.extractors.base import Candidate
from app.pipeline.config import THRESHOLDS

FIELDS = [
    "TEMP_KEY","CNT_NAME","CNT_CON_DATE","CNT_ST_DATE","CNT_END_DATE",
    "CNT_CONCLUDED","CNT_RENEWAL","CNT_AMT","CNT_AMT_CRY",
    "CNT_AUTO_RNW_TERM_AMT","CNT_AUTO_RNW_TERM_UNIT",
]

def _best_by_score(cands: List[Candidate]) -> Optional[Candidate]:
    if not cands:
        return None
    return sorted(cands, key=lambda x: x.score, reverse=True)[0]

def select_best(cands: List[Candidate]) -> Dict[str, dict]:
    grouped = defaultdict(list)
    for c in cands:
        grouped[c.field].append(c)

    result: Dict[str, dict] = {f: {"value": None, "confidence": None, "evidence": None} for f in FIELDS}

    # 1) 점수 Top-1 선정 (CNT_NAME은 특별 처리)
    for field, items in grouped.items():
        # CNT_NAME 필드 특별 처리: 타이틀 영역 후보와 라벨 후보가 모두 있으면 결합
        if field == "CNT_NAME":
            title_cand = None
            header_cand = None
            
            # 타이틀 영역 후보 찾기 (source="title")
            for item in items:
                if item.source == "title":
                    if title_cand is None or item.score > title_cand.score:
                        title_cand = item
            
            # 라벨 후보 찾기 (source="header")
            for item in items:
                if item.source == "header":
                    if header_cand is None or item.score > header_cand.score:
                        header_cand = item
            
            # 둘 다 있으면 결합
            if title_cand and header_cand:
                combined_value = f"{title_cand.raw_value}_{header_cand.raw_value}"
                # 결합된 값의 신뢰도는 두 후보의 평균 또는 최고값 사용 (평균 사용)
                combined_conf = (title_cand.score + header_cand.score) / 2.0
                combined_conf = min(1.0, max(0.0, combined_conf))
                
                # Evidence는 두 후보를 모두 포함 (페이지는 첫 번째, 라인은 합치기)
                combined_lines = sorted(set(title_cand.evidence.lines + header_cand.evidence.lines))
                combined_snippet = f"{title_cand.evidence.snippet} | {header_cand.evidence.snippet}"
                
                result[field] = {
                    "value": combined_value,
                    "confidence": combined_conf,
                    "evidence": {
                        "page": min(title_cand.evidence.page, header_cand.evidence.page),
                        "lines": combined_lines,
                        "snippet": combined_snippet[:300]
                    },
                }
            else:
                # 하나만 있거나 둘 다 없으면 기존 로직대로 최고 점수 후보 선택
                best = _best_by_score(items)
                if not best:
                    continue
                conf = min(1.0, max(0.0, best.score))
                result[field] = {
                    "value": best.raw_value,
                    "confidence": conf,
                    "evidence": {"page": best.evidence.page, "lines": best.evidence.lines, "snippet": best.evidence.snippet[:300]},
                }
        else:
            # 다른 필드는 기존 로직대로
            best = _best_by_score(items)
            if not best:
                continue
            conf = min(1.0, max(0.0, best.score))
            result[field] = {
                "value": best.raw_value,
                "confidence": conf,
                "evidence": {"page": best.evidence.page, "lines": best.evidence.lines, "snippet": best.evidence.snippet[:300]},
            }

    # 2) 간단 교차 검증
    st = result.get("CNT_ST_DATE", {}).get("value")
    ed = result.get("CNT_END_DATE", {}).get("value")
    if st and ed and st > ed:
        if result["CNT_END_DATE"]["confidence"] is not None:
            result["CNT_END_DATE"]["confidence"] = max(0.0, result["CNT_END_DATE"]["confidence"] - 0.3)
            result["CNT_END_DATE"]["note"] = "종료일이 시작일보다 빠릅니다. 원문 확인 필요"

    amt = result.get("CNT_AMT", {}).get("value")
    ccy = result.get("CNT_AMT_CRY", {}).get("value")
    if amt and not ccy:
        result["CNT_AMT_CRY"]["note"] = "통화가 확인되지 않았습니다. KRW/USD/JPY/EUR 중 선택해주세요."

    # 3) 임계치 미달 플래그
    thr = THRESHOLDS["accept"]
    for f, payload in result.items():
        conf = payload.get("confidence")
        if conf is not None and conf < thr:
            payload["note"] = (payload.get("note") or "") + " 검수 필요"

    return result
